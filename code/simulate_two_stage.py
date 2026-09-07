"""
City-level LLM pandemic simulation — TWO-STAGE variant.

Difference from simulate.py: each agent × date is simulated in two sequential
LLM calls instead of one.

  Stage 1 (compliance scoring)
      Input : demographics + policy timeline + current disease level
      Output: an integer policy-compliance score from 1 (non-compliant) to
              5 (fully compliant), plus the reasoning behind it.

  Stage 2 (visitation prediction)
      Input : the same context, PLUS the Stage-1 compliance score and reasoning
      Output: the final per-POI fractional visitation change.

Both stages see the *identical* dropped-out context (the dropout draw is made
once per agent × date and reused), so the Stage-2 answer is conditioned on the
same information the Stage-1 score was formed from.

Gemini models here authenticate with Application Default Credentials (ADC) via
Vertex AI — no API key file is read.  Run `gcloud auth application-default
login` once, and make sure a project is resolvable (see VERTEX PROJECT below).
GPT / Grok / Qwen still use the credentials/ key files via models.create_model.

All tunable parameters are loaded from config.json.

Output per model:
  results/{city}/{model}/llm_ipf_agents_result_{model}_twostage_n{N}.jsonl
  (beside the one-stage llm_ipf_agents_result_{model}_onestage_n300.jsonl)
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    _normalize_cbg_geoid_str,
    _parse_llm_json,
    select_LLM_IPF_individuals_from_cbg,
)
from models import BaseModel, create_model

# =============================================================================
# Load config
# =============================================================================

REPO_ROOT   = Path(__file__).resolve().parent.parent   # .../Pandemics/Policy-Agent
CONFIG_PATH = REPO_ROOT / "config.json"

with open(CONFIG_PATH, encoding="utf-8") as _f:
    _CFG = json.load(_f)

# Directories
ADVAN_DATA_DIR = REPO_ROOT / "data" / "city_data"
PANDEMIC_DIR   = REPO_ROOT / "data" / "pandemic_data"
POLICY_DIR     = REPO_ROOT / "data" / "policy_data"
RESULTS_DIR    = REPO_ROOT / "results"
PROMPTS_DIR    = REPO_ROOT / "prompts"

def model_dir(city_name: str, model_name: str) -> Path:
    """
    Per-city x model results folder — the same one simulate.py writes its
    one-stage llm_ipf_agents_result_{model}_onestage_n300.jsonl into.  The
    two-stage output lands beside it under a distinct filename, so the two never
    collide and calculate_results.py never picks the wrong one up.
    """
    return RESULTS_DIR / city_name / model_name


def _rel(path: Path) -> str:
    """Path relative to the repo root, for readable logging."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)

PHASE1_PROMPT_PATH = PROMPTS_DIR / "agent_two_stage_phase1_compliance.txt"
PHASE2_PROMPT_PATH = PROMPTS_DIR / "agent_two_stage_phase2_visitation.txt"
CONFIRMED_FILE     = PANDEMIC_DIR / "aggregate_confirmed.csv"
DEATHS_FILE        = PANDEMIC_DIR / "aggregate_deaths.csv"

# From config
CITY_LIST        = _CFG["cities"]
MODEL_LIST       = _CFG["models"]["simulation_models"]
MODEL_REGISTRY   = _CFG["models"]["registry"]
SIMULATION_DATES = _CFG["simulation"]["simulation_dates"]
NUM_AGENTS       = _CFG["simulation"]["num_agents"]
RANDOM_SEED      = _CFG["simulation"]["random_seed"]
DROPOUT_RATE     = _CFG["simulation"]["dropout_rate"]

# Two-stage results are written alongside the single-stage ones under a distinct
# filename so calculate_results.py (which reads simulation.agent_file_template)
# never picks them up by accident.  Override in config with
# simulation.agent_file_template_two_stage if you want a different name.
AGENT_FILE_TEMPLATE = _CFG["simulation"].get(
    "agent_file_template_two_stage",
    "llm_ipf_agents_result_{model}_twostage_n{n}.jsonl",
)

# The existing single-stage results.  These are the authoritative record of which
# agents the latest pipeline actually ran on, so by default the two-stage run
# rebuilds its agent pool from them (see load_agents_from_results).
SINGLE_STAGE_FILE_TEMPLATE = _CFG["simulation"]["agent_file_template"]

CANONICAL_TO_DISPLAY    = _CFG["poi"]["canonical_to_display"]
STATE_ABBR_TO_FULL_NAME = _CFG["state_names"]

# Current run target.  Override on the command line with --cities / --models;
# --cities all restores every city in config.cities.
DEFAULT_CITIES = ["san_antonio"]

_NEWS_CUTOFF = datetime.strptime(_CFG["simulation"]["news_cutoff"], "%Y-%m-%d")
_NEWS_ITEMS  = _CFG["simulation"]["news_items"]

# Vertex AI settings for ADC-authenticated Gemini calls.  Config wins, then env,
# then whatever the ADC credentials themselves carry.
_VERTEX_CFG      = _CFG.get("vertex", {})
VERTEX_PROJECT   = _VERTEX_CFG.get("project")  or os.environ.get("GOOGLE_CLOUD_PROJECT")
VERTEX_LOCATION  = _VERTEX_CFG.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

COMPLIANCE_LABELS = {
    1: "Non-compliant",
    2: "Weakly compliant",
    3: "Moderately compliant",
    4: "Strongly compliant",
    5: "Fully compliant",
}

# Regex fallbacks for parsing raw LLM text
_CHANGE_KV_RE = re.compile(
    r'"(?P<key>[^"]+?)_change"\s*:\s*(?P<val>-?\d+(?:\.\d+)?)',
    re.DOTALL,
)
_SCORE_RE = re.compile(
    r'"?compliance[_ ]?score"?\s*[:=]\s*"?\s*(?P<val>[1-5])\b',
    re.IGNORECASE,
)


# =============================================================================
# Gemini via Application Default Credentials (no API key)
# =============================================================================

class GeminiADCModel(BaseModel):
    """
    Google Gemini through Vertex AI, authenticated with Application Default
    Credentials.  No key file is read.

    Setup:
        gcloud auth application-default login
        gcloud auth application-default set-quota-project <PROJECT_ID>

    The project is resolved from config.vertex.project, then
    $GOOGLE_CLOUD_PROJECT, then the project attached to the ADC credentials.
    """

    def __init__(self,
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 model_name: str = "gemini-2.5-pro",
                 enable_thinking: bool = False,
                 project: str | None = None,
                 location: str | None = None):
        self._model_name_config = model_name
        self.enable_thinking    = enable_thinking
        self.project            = project  or VERTEX_PROJECT
        self.location           = location or VERTEX_LOCATION
        super().__init__(temperature, max_tokens)

    def _initialize_client(self):
        from google import genai

        if not self.project:
            # Fall back to whatever project the ADC credentials carry.
            import google.auth
            _, self.project = google.auth.default()
        if not self.project:
            raise RuntimeError(
                "No Google Cloud project for ADC-authenticated Gemini. Set "
                "config.json -> vertex.project, or export GOOGLE_CLOUD_PROJECT, "
                "or run: gcloud auth application-default set-quota-project <PROJECT_ID>"
            )

        self.client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
        )
        self.model_name = self._model_name_config
        print(f"  [Gemini/ADC] project={self.project} location={self.location} "
              f"model={self.model_name}")

    def call(self, prompt: str) -> str | None:
        from google.genai import types
        try:
            thinking = types.ThinkingConfig(
                thinking_budget=None if self.enable_thinking else 0
            )
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    response_mime_type="application/json",
                    thinking_config=thinking,
                ),
            )
            return resp.text
        except Exception as e:
            print(f"[Gemini/ADC] Error: {e}")
            return None


def build_model(model_name: str, gemini_auth: str = "adc") -> BaseModel:
    """
    Instantiate a model from config.models.registry.

    gemini_auth="adc" (default) routes Gemini to the ADC-authenticated Vertex
    client above; "api-key" routes it through the standard key-file factory in
    models.py, which reads credentials/gemini_api_key.json like every other
    script here.  Everything that is not Gemini always uses the key files.
    """
    cfg = MODEL_REGISTRY[model_name]
    kwargs: dict = {
        "model_name":  cfg["api_model_name"],
        "temperature": cfg.get("temperature", 1),
        "max_tokens":  cfg.get("max_tokens", 1024),
    }
    if "enable_thinking" in cfg:
        kwargs["enable_thinking"] = cfg["enable_thinking"]

    if cfg["type"].lower() == "gemini" and gemini_auth == "adc":
        return GeminiADCModel(**kwargs)
    return create_model(cfg["type"], **kwargs)


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
# Agent generation / loading
# =============================================================================

def agent_pool_path(city_name: str) -> Path:
    return RESULTS_DIR / city_name / "llm_ipf_agents.jsonl"


def load_agents(city_name: str) -> list[dict]:
    """Load a previously generated LLM+IPF agent pool."""
    path = agent_pool_path(city_name)
    with open(path, encoding="utf-8") as f:
        agents = [json.loads(line) for line in f if line.strip()]
    print(f"[Agent Pool] Reused {len(agents)} agents ← {_rel(path)}")
    return agents


def generate_and_save_agents(city_cfg: dict) -> list[dict]:
    """
    Run LLM+IPF to generate NUM_AGENTS agents for the city and save them
    to results/{city}/llm_ipf_agents.jsonl.  Returns the agent list.
    """
    city_name  = city_cfg["name"]
    cbg_prefix = city_cfg["cbg_prefix"]

    print(f"\n[Agent Generation] City: {city_cfg['display_name']}  (prefix={cbg_prefix})")
    city_row = load_city_row(city_name, cbg_prefix)

    print(f"[Agent Generation] Running LLM+IPF for {NUM_AGENTS} agents …")
    agents = select_LLM_IPF_individuals_from_cbg(city_row, k=NUM_AGENTS)

    out_path = agent_pool_path(city_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for agent in agents:
            f.write(json.dumps(agent, ensure_ascii=False) + "\n")
    print(f"[Agent Generation] Saved {len(agents)} agents → "
          f"{_rel(out_path)}")
    return agents


def load_agents_from_results(city_name: str, model_name: str) -> list[dict]:
    """
    Rebuild the agent pool from an existing single-stage result file.

    This is the default source, and it matters: the latest pipeline's agent set
    was produced by merge_highschool_with_full_simulation.py, which stitched a
    dedicated "Highschool Degree or Lower" run together with a random top-up
    sample drawn from the full pool.  That merge is NOT reproducible — its RNG
    is seeded with `RANDOM_SEED + hash((city, model))`, and Python randomises
    str/tuple hashing per process — and its output does not match
    results/{city}/llm_ipf_agents.jsonl.  The result files are therefore the
    only surviving record of who was actually simulated.

    Agents keep their original agent_index so two-stage records line up 1:1 with
    the single-stage records for the same agent.
    """
    path = model_dir(city_name, model_name) / SINGLE_STAGE_FILE_TEMPLATE.format(model=model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No single-stage result file to take the agent pool from: {path}\n"
            f"Run with --agent-source pool-file to use "
            f"results/{city_name}/llm_ipf_agents.jsonl instead."
        )

    agents: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec  = json.loads(line)
            idx  = rec.get("agent_index")
            if idx is None or idx in agents:
                continue
            info = rec.get("individual_info") or {}
            agents[idx] = {
                "gender":           info.get("gender",           "Unknown"),
                "age":              info.get("age",              "Unknown"),
                "race":             info.get("race",             "Unknown"),
                "education":        info.get("education",        "Unknown"),
                "household_income": info.get("household_income", "Unknown"),
                # simulate.py writes occupation_raw / occupation_major; the pool
                # schema and OCCUPATION_TO_MAJOR_CATEGORY both key off the raw value.
                "occupation":       info.get("occupation_raw") or info.get("occupation_major", ""),
                "_agent_index":     idx,
                "_source_file":     rec.get("source_file", path.name),
                "_original_agent_index": rec.get("original_agent_index"),
            }

    ordered = [agents[i] for i in sorted(agents)]
    n_hs = sum(1 for a in ordered if a["education"] == "Highschool Degree or Lower")
    print(f"[Agent Pool] {len(ordered)} agents ← {_rel(path)}")
    print(f"             education 'Highschool Degree or Lower': {n_hs} "
          f"({100 * n_hs / max(len(ordered), 1):.1f}%)")
    return ordered


def get_agents(city_cfg: dict, model_name: str, args) -> list[dict]:
    """
    Resolve the agent pool for one city × model.

      --agent-source result-file  (default)
          Take the exact 300 agents the latest single-stage run used for this
          city × model.  Gives a paired two-stage vs single-stage comparison.

      --agent-source result-file --reference-model M
          Take model M's agent set and run every model on it.  Use this when
          the cross-model comparison matters more than pairing with the
          existing single-stage results (the merge sampled a different top-up
          stratum per model, so the per-model sets are NOT the same people —
          only the highschool stratum is shared).

      --agent-source pool-file
          Use results/{city}/llm_ipf_agents.jsonl.  A different agent set from
          the one the latest results were produced on.
    """
    city_name = city_cfg["name"]

    if args.agent_source == "result-file":
        return load_agents_from_results(city_name, args.reference_model or model_name)

    if not args.regenerate_agents and agent_pool_path(city_name).exists():
        return load_agents(city_name)
    return generate_and_save_agents(city_cfg)


# =============================================================================
# Pandemic / policy data helpers
# =============================================================================

def _lookup_state_row(df: pd.DataFrame, state_abbr: str, fname: str) -> pd.Series:
    """
    Pull one state's row from an aggregate CSV.  The shipped files are indexed by
    two-letter abbreviation ('MA'), but tolerate a full-name index too.
    """
    if state_abbr in df.index:
        return df.loc[state_abbr]
    state_full = STATE_ABBR_TO_FULL_NAME.get(state_abbr, state_abbr)
    if state_full in df.index:
        return df.loc[state_full]
    raise KeyError(
        f"State '{state_abbr}' / '{state_full}' not in {fname} "
        f"(index: {list(df.index)[:10]})"
    )


def _national_row(df: pd.DataFrame) -> pd.Series:
    """
    National totals.  The aggregate CSVs carry an explicit 'US' row — use it.
    Only fall back to summing the state rows when no such row exists, and never
    include the 'US' row in that sum.
    """
    for key in ("US", "United States"):
        if key in df.index:
            return df.loc[key]
    return df.drop(index=[i for i in df.index if str(i).upper() in {"US", "UNITED STATES"}],
                   errors="ignore").sum(axis=0)


def load_pandemic_data(state_abbr: str):
    """
    Load state and US pandemic rows from the aggregate CSVs.
    Returns (state_confirmed_row, state_deaths_row, us_confirmed_row, us_deaths_row).
    """
    confirmed_df = pd.read_csv(CONFIRMED_FILE, index_col=0)
    deaths_df    = pd.read_csv(DEATHS_FILE,    index_col=0)

    return (
        _lookup_state_row(confirmed_df, state_abbr, CONFIRMED_FILE.name),
        _lookup_state_row(deaths_df,    state_abbr, DEATHS_FILE.name),
        _national_row(confirmed_df),
        _national_row(deaths_df),
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
# Shared context construction (drawn ONCE, used by both stages)
# =============================================================================

def build_shared_context(
    simulation_date: str,
    policy_detail_text: str,
    state_confirmed: int,
    state_deaths: int,
    us_confirmed: int,
    us_deaths: int,
    rng: random.Random,
) -> dict:
    """
    Apply the DROPOUT_RATE draw over policy paragraphs, WHO/federal news, and
    disease-stat sentences, and return the resulting text blocks.

    Called once per agent × date; the same blocks feed Stage 1 and Stage 2 so
    the compliance score and the final prediction rest on identical evidence.
    """
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
    kept_policy  = [x for i, x in enumerate(policy_items)                   if i not in drop_set]
    kept_news    = [x for i, x in enumerate(news_items,    start=n_p)       if i not in drop_set]
    kept_disease = [x for i, x in enumerate(disease_items, start=n_p + n_n) if i not in drop_set]

    return {
        "policy_timeline": "\n\n".join(kept_policy + kept_news) or "No policy recorded for this date.",
        "disease_stats":   " ".join(kept_disease),
        "n_dropped":       len(drop_set),
        "n_context_items": len(all_items),
    }


def fill_common_placeholders(
    template: str,
    city_cfg: dict,
    individual: dict,
    context: dict,
) -> str:
    """Substitute the placeholders shared by both stage templates."""
    state_full        = STATE_ABBR_TO_FULL_NAME.get(city_cfg["state_abbr"], city_cfg["state_abbr"])
    occupation_subcat = individual.get("occupation", "")
    occupation_major  = OCCUPATION_TO_MAJOR_CATEGORY.get(occupation_subcat, occupation_subcat)

    out = template
    out = out.replace("**{{City_Name}}**",               city_cfg["display_name"])
    out = out.replace("**{{State}}**",                   state_full)
    out = out.replace("**{{City_State_Introduction}}**", city_cfg["introduction"])
    out = out.replace("**{{Total_Population}}**",        str(city_cfg["total_population"]))
    out = out.replace("**{{Age}}**",                     str(individual.get("age",          "Unknown")))
    out = out.replace("**{{Race}}**",                    individual.get("race",             "Unknown"))
    out = out.replace("**{{Gender}}**",                  individual.get("gender",           "Unknown"))
    out = out.replace("**{{Education}}**",               individual.get("education",        "Unknown"))
    out = out.replace("**{{Household_Income}}**",        individual.get("household_income", "Unknown"))
    out = out.replace("**{{Occupation}}**",              occupation_major)
    out = out.replace("{{Policy_Timeline}}",             context["policy_timeline"])
    out = out.replace("{{Disease_Situation_Stats}}",     context["disease_stats"])
    # City_Name also appears unbolded inside Part 1 of both templates.
    out = out.replace("{{City_Name}}",                   city_cfg["display_name"])
    return out


def build_phase1_prompt(template: str, city_cfg: dict, individual: dict, context: dict) -> str:
    """Stage 1: demographics + policy + disease level → compliance score."""
    return fill_common_placeholders(template, city_cfg, individual, context)


def build_phase2_prompt(
    template: str,
    city_cfg: dict,
    individual: dict,
    context: dict,
    compliance_score: int | None,
    compliance_reasoning: str,
) -> str:
    """Stage 2: the same context + the Stage-1 score → per-POI visitation change."""
    prompt = fill_common_placeholders(template, city_cfg, individual, context)

    if compliance_score is None:
        score_text  = "not determined"
        label_text  = "unresolved; judge your own compliance as you go"
        reason_text = (compliance_reasoning.strip()
                       or "No compliance assessment was recorded. Reason about your "
                          "own willingness to comply from Parts 1-3 before predicting.")
    else:
        score_text  = str(compliance_score)
        label_text  = COMPLIANCE_LABELS[compliance_score]
        reason_text = compliance_reasoning.strip() or "(reasoning not recorded)"

    prompt = prompt.replace("{{Compliance_Score}}",     score_text)
    prompt = prompt.replace("{{Compliance_Label}}",     label_text)
    prompt = prompt.replace("{{Compliance_Reasoning}}", reason_text)
    prompt = prompt.replace("{{POI_List}}",             POI_LIST_TEXT)
    prompt = prompt.replace("{{POI_Output_Example}}",   POI_OUTPUT_EXAMPLE)
    return prompt


def _build_poi_blocks() -> tuple[str, str]:
    """Render the POI bullet list and the JSON output example from config.poi."""
    keys = list(CANONICAL_TO_DISPLAY)
    bullets = "\n".join(f"- **{CANONICAL_TO_DISPLAY[k]}**" for k in keys)

    lines = ["{"]
    for i, k in enumerate(keys):
        tail = "," if i < len(keys) - 1 else ""
        lines.append(
            f'    "{k}_reasoning": "[Your step-by-step reasoning for '
            f'{CANONICAL_TO_DISPLAY[k]}, including how your compliance score '
            f'shaped it, followed by your final decision.]",'
        )
        lines.append(f'    "{k}_change": -0.15{tail}')
    lines.append("}")
    return bullets, "\n".join(lines)


POI_LIST_TEXT, POI_OUTPUT_EXAMPLE = _build_poi_blocks()


# =============================================================================
# Response parsing
# =============================================================================

def _normalize_poi_key(raw_key: str) -> str:
    """Normalise a raw POI key from LLM output to one of the canonical keys."""
    k = re.sub(r"[&]", "and", raw_key.strip())
    k = re.sub(r"\s+", "_", k)
    k = re.sub(r"[^A-Za-z0-9_]", "", k)
    # Underscores are optional throughout: models emit Restaurant_Bars,
    # RestaurantAndBars, Arts&Entertainment, EducationalSettings, etc.
    k = re.sub(r"^Restaurants?(?:_?and)?_?Bars?$",  "Restaurants_and_Bars",   k, flags=re.I)
    k = re.sub(r"^Retail$",                         "Retail",                 k, flags=re.I)
    k = re.sub(r"^Arts(?:_?and)?_?Entertainment$",  "Arts_and_Entertainment", k, flags=re.I)
    k = re.sub(r"^Educational(?:_?Settings?)?$",    "Educational_Settings",   k, flags=re.I)
    return k


def extract_poi_changes(parsed: dict | None, raw_text: str | None) -> dict:
    """
    Extract {canonical_key: float} for every configured POI type.
    Uses parsed JSON first; falls back to a regex scan of the raw text.
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


def extract_compliance(parsed: dict | None, raw_text: str | None) -> tuple[int | None, str]:
    """
    Extract (compliance_score, compliance_reasoning) from a Stage-1 response.
    Returns (None, "") when no score in 1-5 can be recovered.
    """
    score: int | None = None
    reasoning = ""

    if isinstance(parsed, dict):
        for key in ("compliance_score", "compliance", "score"):
            if key in parsed:
                try:
                    val = int(round(float(parsed[key])))
                except (TypeError, ValueError):
                    continue
                if 1 <= val <= 5:
                    score = val
                    break
        for key in ("compliance_reasoning", "reasoning"):
            if isinstance(parsed.get(key), str):
                reasoning = parsed[key].strip()
                break

    if score is None and raw_text:
        m = _SCORE_RE.search(raw_text)
        if m:
            score = int(m.group("val"))
    if not reasoning and raw_text and score is not None:
        reasoning = raw_text.strip()

    return score, reasoning


# =============================================================================
# Single-agent two-stage simulation
# =============================================================================

def simulate_agent_two_stage(
    model,
    individual: dict,
    agent_index: int,
    simulation_date: str,
    city_cfg: dict,
    phase1_template: str,
    phase2_template: str,
    state_confirmed: int,
    state_deaths: int,
    us_confirmed: int,
    us_deaths: int,
    policy_detail_text: str,
    prompt_rng: random.Random,
) -> dict:
    """Run Stage 1 then Stage 2 for one agent × one date; return the record."""
    occupation_subcat = individual.get("occupation", "")
    occupation_major  = OCCUPATION_TO_MAJOR_CATEGORY.get(occupation_subcat, occupation_subcat)

    # One dropout draw, shared by both stages.
    context = build_shared_context(
        simulation_date    = simulation_date,
        policy_detail_text = policy_detail_text,
        state_confirmed    = state_confirmed,
        state_deaths       = state_deaths,
        us_confirmed       = us_confirmed,
        us_deaths          = us_deaths,
        rng                = prompt_rng,
    )

    # ── Stage 1: policy compliance score (1-5) ────────────────────────────────
    p1_prompt   = build_phase1_prompt(phase1_template, city_cfg, individual, context)
    p1_response = model.call(p1_prompt)
    p1_parsed   = _parse_llm_json(p1_response) if p1_response else None
    score, score_reasoning = extract_compliance(p1_parsed, p1_response)

    # ── Stage 2: per-POI visitation change, conditioned on the score ──────────
    p2_prompt = build_phase2_prompt(
        template             = phase2_template,
        city_cfg             = city_cfg,
        individual           = individual,
        context              = context,
        compliance_score     = score,
        compliance_reasoning = score_reasoning,
    )
    p2_response       = model.call(p2_prompt)
    p2_parsed         = _parse_llm_json(p2_response) if p2_response else None
    predicted_changes = extract_poi_changes(p2_parsed, p2_response)

    record = {
        "city_name":       city_cfg["display_name"],
        "state_abbr":      city_cfg["state_abbr"],
        "simulation_date": simulation_date,
        "agent_index":     agent_index,
        "sampling_method": "llm_ipf",
        "pipeline":        "two_stage",
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
        "context_info": {
            "n_context_items": context["n_context_items"],
            "n_dropped":       context["n_dropped"],
        },
        "stage1_compliance": {
            "response":         p1_response,
            "parsed":           p1_parsed,
            "compliance_score": score,
            "reasoning":        score_reasoning,
        },
        "compliance_score":  score,          # duplicated at top level for easy analysis
        "response":          p2_response,
        "parsed_prediction": p2_parsed,
        "predicted_changes": predicted_changes,
        # Provenance back to the single-stage twin when the pool came from a
        # result file.  Written unconditionally, null when there is none, so
        # every agent-level record across the pipelines carries the same keys.
        "source_file":          individual.get("_source_file"),
        "original_agent_index": individual.get("_original_agent_index"),
    }

    return record


# =============================================================================
# City × model simulation pipeline
# =============================================================================

def run_city_model(
    city_cfg: dict,
    model_name: str,
    phase1_template: str,
    phase2_template: str,
    agents: list[dict],
    gemini_auth: str = "adc",
) -> None:
    """Run the two-stage simulation for one city × one model."""
    state_abbr = city_cfg["state_abbr"]

    print(f"\n{'#' * 70}")
    print(f"  City: {city_cfg['display_name']}  |  Model: {model_name}  |  two-stage")
    print(f"{'#' * 70}")

    model = build_model(model_name, gemini_auth)

    state_conf_row, state_deaths_row, us_conf_row, us_deaths_row = load_pandemic_data(state_abbr)
    policy_df = load_policy_detail(state_abbr)

    # Same folder as the one-stage llm_ipf_agents_result_{model}_onestage_n300.jsonl;
    # only the filename differs, so nothing existing is overwritten.
    out_dir = model_dir(city_cfg["name"], model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / AGENT_FILE_TEMPLATE.format(model=model_name, n=len(agents))
    if out_file.exists():
        print(f"  [WARN] Overwriting existing {out_file.name}")
    out_file.write_text("")
    print(f"  Agents: {len(agents)}")
    print(f"  Output: {_rel(out_file)}")

    total_records = 0
    n_scored      = 0
    t0            = time.time()
    prompt_rng    = random.Random(RANDOM_SEED)

    with open(out_file, "a", encoding="utf-8") as f_out:
        for d_idx, sim_date in enumerate(SIMULATION_DATES, start=1):
            state_confirmed = get_counts_for_date(state_conf_row,   sim_date)
            state_deaths    = get_counts_for_date(state_deaths_row, sim_date)
            us_confirmed    = get_counts_for_date(us_conf_row,      sim_date)
            us_deaths       = get_counts_for_date(us_deaths_row,    sim_date)
            policy_text     = get_policy_text(policy_df,            sim_date)

            print(f"\n  Date {d_idx}/{len(SIMULATION_DATES)}: {sim_date}"
                  f"  (state_cases={state_confirmed}, us_cases={us_confirmed})")

            for a_idx, agent in enumerate(agents):
                result = simulate_agent_two_stage(
                    model              = model,
                    individual         = agent,
                    agent_index        = agent.get("_agent_index", a_idx),
                    simulation_date    = sim_date,
                    city_cfg           = city_cfg,
                    phase1_template    = phase1_template,
                    phase2_template    = phase2_template,
                    state_confirmed    = state_confirmed,
                    state_deaths       = state_deaths,
                    us_confirmed       = us_confirmed,
                    us_deaths          = us_deaths,
                    policy_detail_text = policy_text,
                    prompt_rng         = prompt_rng,
                )
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                total_records += 1
                if result["compliance_score"] is not None:
                    n_scored += 1

                if (a_idx + 1) % 50 == 0:
                    print(f"    agent {a_idx + 1}/{len(agents)} "
                          f"({time.time() - t0:.0f}s elapsed, "
                          f"{n_scored}/{total_records} scored)")

    miss = total_records - n_scored
    print(f"\n  [DONE] {total_records} records → {out_file.name}  "
          f"({time.time() - t0:.1f}s)")
    if miss:
        print(f"  [WARN] {miss} record(s) had no parsable Stage-1 compliance score; "
              f"Stage 2 ran unconditioned for those.")


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--cities", nargs="+", default=DEFAULT_CITIES,
                   help=f"City names to run (default: {DEFAULT_CITIES}). "
                        f"Pass 'all' for every city in config.cities.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model names to run (default: config.models.simulation_models, "
                        "currently all three).")
    p.add_argument("--agent-source", choices=["result-file", "pool-file"], default="result-file",
                   help="Where the agent pool comes from. 'result-file' (default) reuses the exact "
                        "agents from the latest single-stage results for each city x model — the "
                        "batch that includes the 'Highschool Degree or Lower' stratum. 'pool-file' "
                        "uses results/{city}/llm_ipf_agents.jsonl, which is a DIFFERENT agent set.")
    p.add_argument("--reference-model", default=None,
                   help="With --agent-source result-file: take every model's agent pool from THIS "
                        "model's result file, so all models run on one common agent set. Omit to "
                        "give each model its own agents (pairs with that model's single-stage run).")
    p.add_argument("--print-prompt", action="store_true",
                   help="Print both stage prompts for the first agent x date and exit "
                        "without calling any model. Stage 2 is shown with a placeholder "
                        "compliance score, since the real one comes from the Stage-1 call.")
    p.add_argument("--gemini-auth", choices=["adc", "api-key"], default="adc",
                   help="How Gemini authenticates. 'adc' (default) uses Vertex AI with "
                        "Application Default Credentials (run `gcloud auth application-default "
                        "login` once); 'api-key' uses credentials/gemini_api_key.json, the same "
                        "way simulate.py does. GPT / Grok always use the key files.")
    p.add_argument("--regenerate-agents", action="store_true",
                   help="With --agent-source pool-file: re-run LLM+IPF instead of reusing the pool file.")
    return p.parse_args()


def main():
    args = parse_args()

    for path in (PHASE1_PROMPT_PATH, PHASE2_PROMPT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
    phase1_template = PHASE1_PROMPT_PATH.read_text(encoding="utf-8")
    phase2_template = PHASE2_PROMPT_PATH.read_text(encoding="utf-8")

    if args.cities is None or "all" in args.cities:
        cities = CITY_LIST
    else:
        cities = [c for c in CITY_LIST if c["name"] in args.cities]
        if len(cities) != len(set(args.cities)):
            missing = set(args.cities) - {c["name"] for c in cities}
            raise KeyError(f"Unknown city name(s): {sorted(missing)}")
    models = args.models or MODEL_LIST

    print("=" * 70)
    print("Policy-Agent: Two-Stage City-level Pandemic Simulation")
    print("=" * 70)
    if args.reference_model and args.reference_model not in MODEL_REGISTRY:
        raise KeyError(f"Unknown --reference-model '{args.reference_model}'.")

    if args.agent_source == "result-file":
        src = (f"single-stage results of '{args.reference_model}' (shared across all models)"
               if args.reference_model else
               "each model's own single-stage results (paired per model)")
    else:
        src = "results/{city}/llm_ipf_agents.jsonl"

    print(f"Cities : {[c['display_name'] for c in cities]}")
    print(f"Models : {models}")
    print(f"Dates  : {SIMULATION_DATES}")
    print(f"Agents : from {src}")
    print(f"Stages : 1) compliance score 1-5   2) per-POI visitation change")
    print(f"POIs   : {list(CANONICAL_TO_DISPLAY)}")
    print(f"Config : {_rel(CONFIG_PATH)}")
    print("=" * 70)

    for model_name in models:
        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{model_name}'. Check models.registry in config.json.")
        for city_cfg in cities:
            try:
                # Resolved per city x model: with --agent-source result-file the
                # pool is model-specific unless --reference-model pins it.
                agents = get_agents(city_cfg, model_name, args)
            except Exception as exc:
                print(f"\n  [ERROR] Agent pool for {city_cfg['display_name']} / {model_name}: {exc}")
                continue
            if args.print_prompt:
                agent   = agents[0]
                date    = SIMULATION_DATES[0]
                conf, deaths, us_conf, us_deaths = load_pandemic_data(city_cfg["state_abbr"])
                context = build_shared_context(
                    simulation_date    = date,
                    policy_detail_text = get_policy_text(
                        load_policy_detail(city_cfg["state_abbr"]), date),
                    state_confirmed    = get_counts_for_date(conf,      date),
                    state_deaths       = get_counts_for_date(deaths,    date),
                    us_confirmed       = get_counts_for_date(us_conf,   date),
                    us_deaths          = get_counts_for_date(us_deaths, date),
                    rng                = random.Random(RANDOM_SEED),
                )
                print("\n" + "=" * 70)
                print(f"[{city_cfg['display_name']} / {model_name}] agent "
                      f"{agent.get('_agent_index', 0)} x {date}  —  STAGE 1")
                print("=" * 70)
                print(build_phase1_prompt(phase1_template, city_cfg, agent, context))
                print("\n" + "=" * 70)
                print(f"[{city_cfg['display_name']} / {model_name}] agent "
                      f"{agent.get('_agent_index', 0)} x {date}  —  STAGE 2 "
                      f"(placeholder compliance score 4)")
                print("=" * 70)
                print(build_phase2_prompt(phase2_template, city_cfg, agent, context,
                                          compliance_score=4,
                                          compliance_reasoning="(example Stage-1 reasoning)"))
                print("=" * 70)
                continue

            try:
                run_city_model(city_cfg, model_name, phase1_template,
                               phase2_template, agents, args.gemini_auth)
            except Exception as exc:
                print(f"\n  [ERROR] {city_cfg['display_name']} / {model_name}: {exc}")
                import traceback; traceback.print_exc()

    if not args.print_prompt:
        print("\n[INFO] Two-stage simulation complete.")


if __name__ == "__main__":
    main()
