"""
City-level LLM pandemic simulation — modified (no-CBG) version with LLM+IPF sampling.

For each city this script:
  1. Reads city demographics from data/city_data/{city}/cbg_detail_info.csv.
  2. Aggregates CBG rows whose GEOID starts with the 5-digit county FIPS prefix.
  3. Generates NUM_AGENTS agents with the two-phase LLM+IPF sampling algorithm.
  4. Saves the agent pool to results/{city}/llm_ipf_agents.jsonl.
  5. For each model × simulation date, calls the LLM once per agent with the
     all-POI prompt and records predicted visitation changes.

All tunable parameters are loaded from data/config.json.

Output per model:
  results/{city}/llm_ipf_agents_result_{model}_modified_n300.jsonl
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    OCCUPATION_TO_MAJOR_CATEGORY,
    safe_float,
    _normalize_cbg_geoid_str,
    _parse_llm_json,
    select_LLM_IPF_individuals_from_cbg,
)
from models import create_model

# =============================================================================
# Load config
# =============================================================================

REPO_ROOT   = Path(__file__).parent.parent
CONFIG_PATH = REPO_ROOT / "config.json"

with open(CONFIG_PATH, encoding="utf-8") as _f:
    _CFG = json.load(_f)

# Directories
ADVAN_DATA_DIR = REPO_ROOT / "data" / "city_data"
PANDEMIC_DIR   = REPO_ROOT / "data" / "pandemic_data"
POLICY_DIR     = REPO_ROOT / "data" / "policy_data"
RESULTS_DIR    = REPO_ROOT / "results"
PROMPTS_DIR    = REPO_ROOT / "prompts"

PROMPT_TEMPLATE_PATH = PROMPTS_DIR / "advanced_prompt_num_nocbg_allpoi.txt"
CONFIRMED_FILE       = PANDEMIC_DIR / "aggregate_confirmed.csv"
DEATHS_FILE          = PANDEMIC_DIR / "aggregate_deaths.csv"

# From config
CITY_LIST        = _CFG["cities"]
MODEL_LIST       = _CFG["models"]["simulation_models"]
MODEL_REGISTRY   = _CFG["models"]["registry"]
SIMULATION_DATES = _CFG["simulation"]["simulation_dates"]
NUM_AGENTS       = _CFG["simulation"]["num_agents"]
RANDOM_SEED      = _CFG["simulation"]["random_seed"]
DROPOUT_RATE     = _CFG["simulation"]["dropout_rate"]

CANONICAL_TO_DISPLAY    = _CFG["poi"]["canonical_to_display"]
STATE_ABBR_TO_FULL_NAME = _CFG["state_names"]

_NEWS_CUTOFF = datetime.strptime(_CFG["simulation"]["news_cutoff"], "%Y-%m-%d")
_NEWS_ITEMS  = _CFG["simulation"]["news_items"]

# Regex fallback to extract *_change values from raw LLM text
_CHANGE_KV_RE = re.compile(
    r'"(?P<key>[^"]+?)_change"\s*:\s*(?P<val>-?\d+(?:\.\d+)?)',
    re.DOTALL,
)


# =============================================================================
# CBG data loading and city-row aggregation
# =============================================================================

def load_city_row(city_name: str, cbg_prefix: str) -> pd.Series:
    """
    Read cbg_detail_info.csv, normalise 11-digit GEOIDs to 12-digit, filter by
    the 5-digit county FIPS prefix, then sum all numeric columns into a single
    city-aggregated row (pandas Series).
    """
    csv_path = ADVAN_DATA_DIR / city_name / "cbg_detail_info.csv"
    df = pd.read_csv(csv_path, dtype={"visitor_cbg": str})
    df["visitor_cbg"] = df["visitor_cbg"].map(_normalize_cbg_geoid_str)

    mask    = df["visitor_cbg"].str.startswith(cbg_prefix)
    city_df = df.loc[mask].drop(columns=["visitor_cbg"])
    print(f"  CBGs matched: {mask.sum()} / {len(df)}")
    if mask.sum() == 0:
        raise ValueError(
            f"No CBGs found for prefix '{cbg_prefix}' in {csv_path.name}. "
            "Check that cbg_prefix is the correct 5-digit county FIPS code."
        )

    city_row = city_df.sum(numeric_only=True)
    city_row["visitor_cbg"] = f"{cbg_prefix}_aggregated"
    return city_row


# =============================================================================
# Agent generation and persistence
# =============================================================================

def generate_and_save_agents(city_cfg: dict) -> list[dict]:
    """
    Run LLM+IPF to generate NUM_AGENTS agents for the city and save them
    to results/{city}/llm_ipf_agents.jsonl.  Returns the agent list.
    """
    city_name  = city_cfg["name"]
    cbg_prefix = city_cfg["cbg_prefix"]

    print(f"\n[Agent Generation] City: {city_cfg['display_name']}  "
          f"(prefix={cbg_prefix})")

    city_row = load_city_row(city_name, cbg_prefix)

    print(f"[Agent Generation] Running LLM+IPF for {NUM_AGENTS} agents …")
    agents = select_LLM_IPF_individuals_from_cbg(city_row, k=NUM_AGENTS)

    out_dir  = RESULTS_DIR / city_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "llm_ipf_agents.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for agent in agents:
            f.write(json.dumps(agent, ensure_ascii=False) + "\n")
    print(f"[Agent Generation] Saved {len(agents)} agents → "
          f"{out_path.relative_to(REPO_ROOT)}")

    return agents


# =============================================================================
# Pandemic / policy data helpers
# =============================================================================

def load_pandemic_data(state_abbr: str):
    """
    Load state and US pandemic rows from the aggregate CSVs.
    Returns (state_confirmed_row, state_deaths_row, us_confirmed_row, us_deaths_row).
    """
    confirmed_df = pd.read_csv(CONFIRMED_FILE, index_col=0)
    deaths_df    = pd.read_csv(DEATHS_FILE,    index_col=0)

    state_full = STATE_ABBR_TO_FULL_NAME.get(state_abbr, state_abbr)
    if state_full not in confirmed_df.index:
        raise KeyError(f"State '{state_full}' not in {CONFIRMED_FILE.name}")

    return (
        confirmed_df.loc[state_full],
        deaths_df.loc[state_full],
        confirmed_df.sum(axis=0),
        deaths_df.sum(axis=0),
    )


def get_counts_for_date(row: pd.Series, simulation_date: str) -> int:
    """Look up cumulative count for a given YYYY-MM-DD date."""
    dt  = datetime.strptime(simulation_date, "%Y-%m-%d")
    col = f"{dt.month}/{dt.day}/{str(dt.year)[2:]}"
    return int(row[col]) if col in row.index else 0


def load_policy_detail(state_abbr: str) -> pd.DataFrame:
    """Load policy_detail.csv for the state; adds a 'date_str' column (YYYY-MM-DD)."""
    state_full = STATE_ABBR_TO_FULL_NAME.get(state_abbr, state_abbr)
    path       = POLICY_DIR / state_full / "policy_detail.csv"
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    df = pd.read_csv(path)
    df["date_str"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def get_policy_text(policy_df: pd.DataFrame, simulation_date: str) -> str:
    """Return policy_detail text for a date, or empty string."""
    row = policy_df[policy_df["date_str"] == simulation_date]
    if row.empty:
        return ""
    val = row.iloc[0].get("policy_detail")
    return "" if pd.isna(val) else str(val).strip()


# =============================================================================
# Prompt construction
# =============================================================================

def build_prompt(
    template: str,
    city_cfg: dict,
    individual: dict,
    simulation_date: str,
    policy_detail_text: str,
    state_confirmed: int,
    state_deaths: int,
    us_confirmed: int,
    us_deaths: int,
    rng: random.Random,
) -> str:
    """
    Fill the prompt template.  A random fraction (DROPOUT_RATE) of context items
    (policy paragraphs, WHO/federal news, disease-stats sentences) are dropped to
    introduce per-agent diversity.
    """
    state_full       = STATE_ABBR_TO_FULL_NAME.get(city_cfg["state_abbr"], city_cfg["state_abbr"])
    occupation_subcat = individual.get("occupation", "")
    occupation_major  = OCCUPATION_TO_MAJOR_CATEGORY.get(occupation_subcat, occupation_subcat)

    policy_items  = [s.strip() for s in policy_detail_text.split("\n\n") if s.strip()]
    sim_dt        = datetime.strptime(simulation_date, "%Y-%m-%d")
    news_items    = list(_NEWS_ITEMS) if sim_dt >= _NEWS_CUTOFF else []
    disease_items = [
        f"As of the current date, there are **{state_confirmed}** confirmed cases "
        f"attributed to this disease in the state.",
        f"As of the current date, there are **{state_deaths}** deaths "
        f"attributed to this disease in the state.",
        f"Nationwide in the United States, there are **{us_confirmed}** confirmed cases "
        f"attributed to this disease.",
        f"Nationwide in the United States, there are **{us_deaths}** deaths "
        f"attributed to this disease.",
    ]

    all_items = policy_items + news_items + disease_items
    n_drop    = math.floor(len(all_items) * DROPOUT_RATE)
    drop_set  = set(rng.sample(range(len(all_items)), n_drop)) if 0 < n_drop < len(all_items) else set()

    n_p, n_n     = len(policy_items), len(news_items)
    kept_policy  = [x for i, x in enumerate(policy_items)              if i not in drop_set]
    kept_news    = [x for i, x in enumerate(news_items,    start=n_p)  if i not in drop_set]
    kept_disease = [x for i, x in enumerate(disease_items, start=n_p + n_n) if i not in drop_set]

    part2_text   = "\n\n".join(kept_policy + kept_news) or "No policy recorded for this date."
    disease_text = " ".join(kept_disease)

    prompt = template
    prompt = prompt.replace("**{{City_Name}}**",               city_cfg["display_name"])
    prompt = prompt.replace("**{{State}}**",                   state_full)
    prompt = prompt.replace("**{{City_State_Introduction}}**", city_cfg["introduction"])
    prompt = prompt.replace("**{{Total_Population}}**",        str(city_cfg["total_population"]))
    prompt = prompt.replace("**{{Age}}**",                     str(individual.get("age",             "Unknown")))
    prompt = prompt.replace("**{{Race}}**",                    individual.get("race",                "Unknown"))
    prompt = prompt.replace("**{{Gender}}**",                  individual.get("gender",              "Unknown"))
    prompt = prompt.replace("**{{Education}}**",               individual.get("education",           "Unknown"))
    prompt = prompt.replace("**{{Household_Income}}**",        individual.get("household_income",    "Unknown"))
    prompt = prompt.replace("**{{Occupation}}**",              occupation_major)
    prompt = prompt.replace("{{Policy_Timeline}}",             part2_text)
    prompt = prompt.replace("{{Disease_Situation_Stats}}",     disease_text)
    return prompt


# =============================================================================
# Response parsing
# =============================================================================

def _normalize_poi_key(raw_key: str) -> str:
    """Normalise a raw POI key from LLM output to one of the four canonical keys."""
    k = re.sub(r"[&]", "and", raw_key.strip())
    k = re.sub(r"\s+", "_", k)
    k = re.sub(r"[^A-Za-z0-9_]", "", k)
    k = re.sub(r"^Restaurant(?:s)?(?:_and)?(?:_Bar)?s?$",  "Restaurants_and_Bars",   k, flags=re.I)
    k = re.sub(r"^Retail$",                                 "Retail",                  k, flags=re.I)
    k = re.sub(r"^Arts(?:_and)?_Entertainment$",           "Arts_and_Entertainment",  k, flags=re.I)
    return k


def extract_poi_changes(parsed: dict | None, raw_text: str | None) -> dict:
    """
    Extract {canonical_key: float} for all four POI types.
    Uses parsed JSON first; falls back to regex scan of raw text.
    """
    changes: dict[str, float] = {}

    if isinstance(parsed, dict):
        for k, v in parsed.items():
            if k.endswith("_change"):
                base = _normalize_poi_key(k[:-len("_change")])
                try:
                    changes[base] = float(v)
                except (TypeError, ValueError):
                    pass

    if raw_text and len(changes) < len(CANONICAL_TO_DISPLAY):
        for m in _CHANGE_KV_RE.finditer(raw_text):
            base = _normalize_poi_key(m.group("key"))
            if base not in changes:
                try:
                    changes[base] = float(m.group("val"))
                except ValueError:
                    pass

    return {k: changes.get(k) for k in CANONICAL_TO_DISPLAY}


# =============================================================================
# Single-agent simulation
# =============================================================================

def simulate_agent(
    model,
    individual: dict,
    agent_index: int,
    simulation_date: str,
    city_cfg: dict,
    prompt_template: str,
    state_confirmed: int,
    state_deaths: int,
    us_confirmed: int,
    us_deaths: int,
    policy_detail_text: str,
    prompt_rng: random.Random,
) -> dict:
    """Run the all-POI prompt for one agent × one date and return the result dict."""
    occupation_subcat = individual.get("occupation", "")
    occupation_major  = OCCUPATION_TO_MAJOR_CATEGORY.get(occupation_subcat, occupation_subcat)

    prompt = build_prompt(
        template           = prompt_template,
        city_cfg           = city_cfg,
        individual         = individual,
        simulation_date    = simulation_date,
        policy_detail_text = policy_detail_text,
        state_confirmed    = state_confirmed,
        state_deaths       = state_deaths,
        us_confirmed       = us_confirmed,
        us_deaths          = us_deaths,
        rng                = prompt_rng,
    )

    response          = model.call(prompt)
    parsed            = _parse_llm_json(response) if response else None
    predicted_changes = extract_poi_changes(parsed, response)

    return {
        "city_name":       city_cfg["display_name"],
        "state_abbr":      city_cfg["state_abbr"],
        "simulation_date": simulation_date,
        "agent_index":     agent_index,
        "sampling_method": "llm_ipf",
        "individual_info": {
            "race":             individual.get("race",             "Unknown"),
            "gender":           individual.get("gender",           "Unknown"),
            "age":              individual.get("age",              "Unknown"),
            "education":        individual.get("education",        "Unknown"),
            "household_income": individual.get("household_income", "Unknown"),
            "occupation_raw":   occupation_subcat,
            "occupation_major": occupation_major,
        },
        "pandemic_info": {
            "state_confirmed_cases": state_confirmed,
            "state_deaths":          state_deaths,
            "us_confirmed_cases":    us_confirmed,
            "us_deaths":             us_deaths,
        },
        "response":          response,
        "parsed_prediction": parsed,
        "predicted_changes": predicted_changes,
    }


# =============================================================================
# City × model simulation pipeline
# =============================================================================

def run_city_model(
    city_cfg: dict,
    model_name: str,
    prompt_template: str,
    agents: list[dict],
) -> None:
    """Run simulation for one city × one model using a pre-generated agent pool."""
    state_abbr = city_cfg["state_abbr"]

    print(f"\n{'#' * 70}")
    print(f"  City: {city_cfg['display_name']}  |  Model: {model_name}")
    print(f"{'#' * 70}")

    # Instantiate LLM from registry
    cfg = MODEL_REGISTRY[model_name]
    model_kwargs: dict = {
        "model_name":  cfg["api_model_name"],
        "temperature": cfg.get("temperature", 1),
        "max_tokens":  cfg.get("max_tokens", 1024),
    }
    if "enable_thinking" in cfg:
        model_kwargs["enable_thinking"] = cfg["enable_thinking"]
    model = create_model(cfg["type"], **model_kwargs)

    # Load pandemic + policy data
    state_conf_row, state_deaths_row, us_conf_row, us_deaths_row = load_pandemic_data(state_abbr)
    policy_df = load_policy_detail(state_abbr)

    # Prepare output — each model gets its own subfolder under results/{city}/
    out_dir  = RESULTS_DIR / city_cfg["name"] / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / _CFG["simulation"]["agent_file_template"].format(model=model_name)
    out_file.write_text("")
    print(f"  Agents: {len(agents)}")
    print(f"  Output: {out_file.relative_to(REPO_ROOT)}")

    total_records = 0
    t0            = time.time()
    prompt_rng    = random.Random(RANDOM_SEED)

    with open(out_file, "a", encoding="utf-8") as f_out:
        for d_idx, sim_date in enumerate(SIMULATION_DATES, start=1):
            state_confirmed = get_counts_for_date(state_conf_row,  sim_date)
            state_deaths    = get_counts_for_date(state_deaths_row, sim_date)
            us_confirmed    = get_counts_for_date(us_conf_row,      sim_date)
            us_deaths       = get_counts_for_date(us_deaths_row,    sim_date)
            policy_text     = get_policy_text(policy_df,            sim_date)

            print(f"\n  Date {d_idx}/{len(SIMULATION_DATES)}: {sim_date}"
                  f"  (state_cases={state_confirmed}, us_cases={us_confirmed})")

            for a_idx, agent in enumerate(agents):
                result = simulate_agent(
                    model              = model,
                    individual         = agent,
                    agent_index        = a_idx,
                    simulation_date    = sim_date,
                    city_cfg           = city_cfg,
                    prompt_template    = prompt_template,
                    state_confirmed    = state_confirmed,
                    state_deaths       = state_deaths,
                    us_confirmed       = us_confirmed,
                    us_deaths          = us_deaths,
                    policy_detail_text = policy_text,
                    prompt_rng         = prompt_rng,
                )
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                total_records += 1

                if (a_idx + 1) % 50 == 0:
                    print(f"    agent {a_idx + 1}/{len(agents)} "
                          f"({time.time() - t0:.0f}s elapsed)")

    print(f"\n  [DONE] {total_records} records → {out_file.name}  "
          f"({time.time() - t0:.1f}s)")


# =============================================================================
# Entry point
# =============================================================================

def main():
    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {PROMPT_TEMPLATE_PATH}")
    prompt_template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    print("=" * 70)
    print("Policy-Agent: City-level Pandemic Simulation (LLM+IPF / modified)")
    print("=" * 70)
    print(f"Cities : {[c['display_name'] for c in CITY_LIST]}")
    print(f"Models : {MODEL_LIST}")
    print(f"Dates  : {SIMULATION_DATES}")
    print(f"Agents : {NUM_AGENTS} per city (shared across models)")
    print(f"Config : {CONFIG_PATH.relative_to(REPO_ROOT)}")
    print("=" * 70)

    # Step 1: Generate one LLM+IPF agent pool per city
    print("\n" + "=" * 70)
    print("Step 1: LLM+IPF Agent Generation")
    print("=" * 70)
    city_agents: dict[str, list[dict]] = {}
    for city_cfg in CITY_LIST:
        try:
            city_agents[city_cfg["name"]] = generate_and_save_agents(city_cfg)
        except Exception as exc:
            print(f"\n  [ERROR] Agent generation for {city_cfg['display_name']}: {exc}")
            import traceback; traceback.print_exc()

    # Step 2: Run simulation for each model × city
    print("\n" + "=" * 70)
    print("Step 2: LLM Simulation")
    print("=" * 70)
    for model_name in MODEL_LIST:
        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{model_name}'. Check models.registry in config.json.")
        for city_cfg in CITY_LIST:
            city_name = city_cfg["name"]
            if city_name not in city_agents:
                print(f"\n  [SKIP] No agents for {city_cfg['display_name']}.")
                continue
            try:
                run_city_model(city_cfg, model_name, prompt_template, city_agents[city_name])
            except Exception as exc:
                print(f"\n  [ERROR] {city_cfg['display_name']} / {model_name}: {exc}")
                import traceback; traceback.print_exc()

    print("\n[INFO] Simulation complete.")


if __name__ == "__main__":
    main()
