"""
City-level LLM pandemic simulation — NEUTRAL-PROMPT variant.

Difference from simulate.py: every prompt is built from the behaviourally
neutral template

    prompts/agent_one_stage_neutral.txt

instead of prompts/agent_one_stage.txt.  Nothing else changes:
same agents, same dates, same dropout rate, same disease/policy inputs, same
model parameters, same output schema.  A neutral vs original comparison
therefore isolates the effect of the prompt's framing.

Relative to the original template the neutral one removes every framing cue that
could narrow the simulated distribution on its own:

  1. No compliance framing.  Part 2 is a "Local Situation Timeline", not a
     "Policy Timeline", and nothing asks the agent whether it would comply.
  2. No requested reasoning trace.  The Chain-of-Thought part is gone and no
     `*_reasoning` field is asked for — the model returns four numbers.  (A
     thinking model's internal reasoning is untouched; only the prompt's demand
     for an explicit trace is removed.)
  3. Actual-behaviour framing.  Part 4 asks what the agent "would actually do
     next week", not for a normative probability of visitation change.
  4. No homogenising steers.  The "if the case count is relatively small, do not
     let the pandemic influence your decision too much" clause and the
     stay-at-home caveat are both dropped; each pushed every agent toward one
     common answer regardless of demographics.
  5. No value anchor.  The output example carries <number> placeholders instead
     of filled-in values.  Only the sign-convention examples survive, because
     they define what the sign means.

The agent cohort is read out of the existing single-stage result files rather
than re-sampled, so record N here corresponds to record N there: same
agent_index, same individual_info, same simulation_date.

Reproducibility: the dropout draw is seeded deterministically per
(city, model, agent_index, date), so a re-run drops the identical context items.
The original `_onestage_n300` run did not seed its RNG, so the dropout *rate* is
held constant against it but not the specific dropped items.

`prompt_sha256` is written on every record and checked on resume, so a file
produced with a different version of the template is never silently mixed with
one produced from the current version (override with --allow-prompt-mismatch).

Output per model:
  results/{city}/{model}/llm_ipf_agents_result_{model}_neutral_n{N}.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
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
    _parse_llm_json,
)
from models import create_model

# =============================================================================
# Load config
# =============================================================================

REPO_ROOT   = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.json"

with open(CONFIG_PATH, encoding="utf-8") as _f:
    _CFG = json.load(_f)

# Directories
PANDEMIC_DIR = REPO_ROOT / "data" / "pandemic_data"
POLICY_DIR   = REPO_ROOT / "data" / "policy_data"
RESULTS_DIR  = REPO_ROOT / "results"
PROMPTS_DIR  = REPO_ROOT / "prompts"

PROMPT_TEMPLATE_PATH = PROMPTS_DIR / "agent_one_stage_neutral.txt"
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

# The single-stage results this arm re-runs, and the file it writes.  Both are
# overridable in config.simulation so the two never collide in one folder.
SINGLE_STAGE_FILE_TEMPLATE = _CFG["simulation"]["agent_file_template"]
AGENT_FILE_TEMPLATE        = _CFG["simulation"].get(
    "agent_file_template_neutral",
    "llm_ipf_agents_result_{model}_neutral_n{n}.jsonl",
)

CANONICAL_TO_DISPLAY    = _CFG["poi"]["canonical_to_display"]
STATE_ABBR_TO_FULL_NAME = _CFG["state_names"]

_NEWS_CUTOFF = datetime.strptime(_CFG["simulation"]["news_cutoff"], "%Y-%m-%d")
_NEWS_ITEMS  = _CFG["simulation"]["news_items"]

# Sensitivity arm: San Antonio by default, matching the two-stage arm.
# --cities all runs every city in config.cities.
DEFAULT_CITIES = ["san_antonio"]

# Retry policy for transient API failures (model.call returns None on error).
MAX_RETRIES     = 3
RETRY_SLEEP_SEC = 5.0

# Placeholders the template must still contain — they are filled per agent × date.
REQUIRED_PLACEHOLDERS = [
    "**{{City_Name}}**", "**{{State}}**", "**{{City_State_Introduction}}**",
    "**{{Total_Population}}**", "**{{Age}}**", "**{{Race}}**", "**{{Gender}}**",
    "**{{Education}}**", "**{{Household_Income}}**", "**{{Occupation}}**",
    "{{Policy_Timeline}}", "{{Disease_Situation_Stats}}",
]

# Framing cues this arm exists to remove.  Checked case-insensitively with the
# {{...}} tokens stripped, so {{Policy_Timeline}} does not trip the check, and a
# later edit cannot quietly reintroduce what the analysis holds out.
FORBIDDEN_TERMS = [
    "comply", "compliance", "policy", "policies",
    "step by step", "step-by-step", "chain of thought",
    "_reasoning", "lockdown",
]

# Regex fallback to extract *_change values from raw LLM text
_CHANGE_KV_RE = re.compile(
    r'"(?P<key>[^"]+?)_change"\s*:\s*(?P<val>-?\d+(?:\.\d+)?)',
    re.DOTALL,
)


def _rel(path: Path) -> str:
    """Path relative to the repo root, for readable logging."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# =============================================================================
# Pandemic / policy data helpers
# =============================================================================

def _lookup_state_row(df: pd.DataFrame, state_abbr: str, fname: str) -> pd.Series:
    """
    Pull one state's row from an aggregate CSV.  The shipped files are indexed by
    two-letter abbreviation ('TX'), but tolerate a full-name index too.
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
    """Look up the cumulative count for a given YYYY-MM-DD date."""
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
    """Return the policy_detail text for a date, or an empty string."""
    row = policy_df[policy_df["date_str"] == simulation_date]
    if row.empty:
        return ""
    val = row.iloc[0].get("policy_detail")
    return "" if pd.isna(val) else str(val).strip()


# =============================================================================
# Prompt construction
# =============================================================================

def build_context(
    simulation_date: str,
    policy_detail_text: str,
    state_confirmed: int,
    state_deaths: int,
    us_confirmed: int,
    us_deaths: int,
    rng: random.Random,
    dropout_rate: float,
) -> dict:
    """
    Apply the dropout draw over policy paragraphs, WHO/federal news and
    disease-stat sentences, and return the resulting text blocks.
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
    n_drop    = math.floor(len(all_items) * dropout_rate)
    drop_set  = (set(rng.sample(range(len(all_items)), n_drop))
                 if 0 < n_drop < len(all_items) else set())

    n_p, n_n     = len(policy_items), len(news_items)
    kept_policy  = [x for i, x in enumerate(policy_items)                   if i not in drop_set]
    kept_news    = [x for i, x in enumerate(news_items,    start=n_p)       if i not in drop_set]
    kept_disease = [x for i, x in enumerate(disease_items, start=n_p + n_n) if i not in drop_set]

    return {
        "policy_timeline": "\n\n".join(kept_policy + kept_news)
                           or "No policy recorded for this date.",
        "disease_stats":   " ".join(kept_disease),
        "n_context_items": len(all_items),
        "n_dropped":       len(drop_set),
    }


def build_prompt(template: str, city_cfg: dict, individual: dict, context: dict) -> str:
    """Fill the neutral template for one agent × date."""
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
    # City_Name also appears unbolded inside the body of the template.
    out = out.replace("{{City_Name}}",                   city_cfg["display_name"])
    return out


def prompt_fingerprint(template: str) -> str:
    """Short sha256 of the template, recorded on every record and checked on resume."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def check_prompt_template(template: str) -> tuple[list[str], list[str]]:
    """
    Return (missing_placeholders, framing_cues_still_present).

    The second list is what makes this arm neutral: {{...}} tokens are stripped
    before the scan so a placeholder name never counts as a cue.
    """
    missing = [p for p in REQUIRED_PLACEHOLDERS if p not in template]
    body    = re.sub(r"\{\{[A-Za-z_]+\}\}", " ", template).lower()
    found   = [t for t in FORBIDDEN_TERMS if t in body]
    return missing, found


def dropout_seed_key(city_name: str, model_name: str, agent_index: int, date: str) -> str:
    """
    The per-(agent × date) dropout key, as recorded on every output record.

    The RNG is seeded with config.simulation.random_seed prepended to this key
    (see dropout_rng), which is the form the shipped San Antonio results were
    produced with — recording the key without the seed keeps a re-run's records
    byte-comparable with those.
    """
    return f"{city_name}|{model_name}|{agent_index}|{date}"


def dropout_rng(seed_key: str) -> random.Random:
    """
    Deterministic RNG for one agent × date.

    random.Random(str) seeds from the sha512 of the string, so this is stable
    across processes and unaffected by PYTHONHASHSEED — unlike hash()-based keys.
    """
    return random.Random(f"{RANDOM_SEED}|{seed_key}")


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
    Extract {canonical_key: float | None} for every configured POI type.
    Uses parsed JSON first; falls back to a regex scan of the raw text.
    """
    changes: dict[str, float] = {}

    if isinstance(parsed, dict):
        for k, v in parsed.items():
            if isinstance(k, str) and k.endswith("_change"):
                base = _normalize_poi_key(k[: -len("_change")])
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
# Cohort loading
# =============================================================================

def model_dir(city_name: str, model_name: str) -> Path:
    """Per-city × model results folder."""
    return RESULTS_DIR / city_name / model_name


def load_cohort_from_results(city_name: str, model_name: str,
                             dates: list[str]) -> list[dict]:
    """
    Read the (agent × date) tasks straight out of the single-stage result file,
    so this arm re-runs exactly the cohort the original run simulated.

    Only individual_info and the identifying fields are kept — the original
    predictions are deliberately not carried over.
    """
    path = model_dir(city_name, model_name) / SINGLE_STAGE_FILE_TEMPLATE.format(model=model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No single-stage result file to take the cohort from: {_rel(path)}\n"
            f"Run with --agent-source pool-file to use "
            f"results/{city_name}/llm_ipf_agents.jsonl instead."
        )

    tasks: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec  = json.loads(line)
            date = rec.get("simulation_date")
            if not date or date not in dates:
                continue
            info = rec.get("individual_info") or {}
            tasks.append({
                "agent_index":     rec.get("agent_index"),
                "simulation_date": date,
                "individual": {
                    "race":             info.get("race",             "Unknown"),
                    "gender":           info.get("gender",           "Unknown"),
                    "age":              info.get("age",              "Unknown"),
                    "education":        info.get("education",        "Unknown"),
                    "household_income": info.get("household_income", "Unknown"),
                    # The pool schema and OCCUPATION_TO_MAJOR_CATEGORY both key
                    # off the raw value; the stored record splits it raw/major.
                    "occupation":       info.get("occupation_raw")
                                        or info.get("occupation_major", ""),
                },
                "source_file":          rec.get("source_file"),
                "original_agent_index": rec.get("original_agent_index"),
            })

    n_agents = len({t["agent_index"] for t in tasks})
    print(f"[Cohort] {len(tasks)} tasks / {n_agents} agents ← {_rel(path)}")
    return tasks


def load_cohort_from_pool(city_name: str, dates: list[str]) -> list[dict]:
    """
    Build the (agent × date) tasks from the LLM+IPF agent pool file.  This is a
    DIFFERENT agent set from the one the shipped results were produced on.
    """
    path = RESULTS_DIR / city_name / "llm_ipf_agents.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Agent pool not found: {_rel(path)}")

    with open(path, encoding="utf-8") as f:
        agents = [json.loads(line) for line in f if line.strip()]

    tasks = [
        {
            "agent_index":          idx,
            "simulation_date":      date,
            "individual":           agent,
            "source_file":          path.name,
            "original_agent_index": None,
        }
        for date in dates
        for idx, agent in enumerate(agents)
    ]
    print(f"[Cohort] {len(tasks)} tasks / {len(agents)} agents ← {_rel(path)}")
    return tasks


def load_completed(path: Path) -> tuple[set, set]:
    """
    (agent_index, simulation_date) pairs already written, plus the set of
    prompt_sha256 values those records carry.  More than one fingerprint, or one
    that differs from the template about to be used, means two prompt versions
    would be mixed in a single arm.
    """
    done: set    = set()
    hashes: set  = set()
    if not path.exists():
        return done, hashes
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((rec.get("agent_index"), rec.get("simulation_date")))
            hashes.add(rec.get("prompt_sha256"))
    return done, hashes


# =============================================================================
# Model construction
# =============================================================================

def build_model(model_name: str):
    """Instantiate a model from config.models.registry via the credentials/ keys."""
    cfg = MODEL_REGISTRY[model_name]
    kwargs: dict = {
        "model_name":  cfg["api_model_name"],
        "temperature": cfg.get("temperature", 1),
        "max_tokens":  cfg.get("max_tokens", 1024),
    }
    if "enable_thinking" in cfg:
        kwargs["enable_thinking"] = cfg["enable_thinking"]
    # Recorded params exclude model_name: the API model name is fixed by the
    # registry entry the folder is named after, and leaving it out is the shape
    # the shipped results already use.
    params = {k: v for k, v in kwargs.items() if k != "model_name"}
    return create_model(cfg["type"], **kwargs), params


def call_with_retry(model, prompt: str) -> str | None:
    """model.call swallows exceptions and returns None; retry a few times."""
    for attempt in range(1, MAX_RETRIES + 1):
        response = model.call(prompt)
        if response:
            return response
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_SLEEP_SEC * attempt)
    return None


# =============================================================================
# City × model pipeline
# =============================================================================

def run_city_model(city_cfg: dict, model_name: str, prompt_template: str,
                   tasks: list[dict], args) -> None:
    """Run the neutral-prompt simulation for one city × one model."""
    city_name  = city_cfg["name"]
    state_abbr = city_cfg["state_abbr"]
    fingerprint = prompt_fingerprint(prompt_template)

    print(f"\n{'#' * 70}")
    print(f"  City: {city_cfg['display_name']}  |  Model: {model_name}  |  neutral prompt")
    print(f"{'#' * 70}")

    out_dir = model_dir(city_name, model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_agents = len({t["agent_index"] for t in tasks})
    out_file = out_dir / AGENT_FILE_TEMPLATE.format(model=model_name, n=n_agents)

    if args.overwrite and out_file.exists():
        print(f"  [WARN] Overwriting existing {out_file.name}")
        out_file.unlink()

    done, hashes = load_completed(out_file)
    stale = {h for h in hashes if h and h != fingerprint}
    if stale and not args.allow_prompt_mismatch:
        raise RuntimeError(
            f"{_rel(out_file)} holds records produced from a different prompt "
            f"version ({sorted(stale)} vs current {fingerprint}). Re-run with "
            f"--overwrite to start the file fresh, or --allow-prompt-mismatch "
            f"to append anyway."
        )

    pending = [t for t in tasks if (t["agent_index"], t["simulation_date"]) not in done]
    print(f"  Agents : {n_agents}   tasks: {len(tasks)}   "
          f"already done: {len(done)}   to run: {len(pending)}")
    print(f"  Output : {_rel(out_file)}")
    if not pending:
        print("  [SKIP] Nothing left to run.")
        return

    model, model_params = build_model(model_name)

    state_conf_row, state_deaths_row, us_conf_row, us_deaths_row = load_pandemic_data(state_abbr)
    policy_df = load_policy_detail(state_abbr)

    # Per-date context inputs are identical across agents; resolve them once.
    date_inputs = {}
    for date in sorted({t["simulation_date"] for t in pending}):
        date_inputs[date] = {
            "state_confirmed_cases": get_counts_for_date(state_conf_row,   date),
            "state_deaths":          get_counts_for_date(state_deaths_row, date),
            "us_confirmed_cases":    get_counts_for_date(us_conf_row,      date),
            "us_deaths":             get_counts_for_date(us_deaths_row,    date),
            "policy_text":           get_policy_text(policy_df,            date),
        }

    t0        = time.time()
    n_written = 0
    n_failed  = 0

    with open(out_file, "a", encoding="utf-8") as f_out:
        for i, task in enumerate(pending, start=1):
            date        = task["simulation_date"]
            di          = date_inputs[date]
            agent_index = task["agent_index"]
            individual  = task["individual"]

            seed_key = dropout_seed_key(city_name, model_name, agent_index, date)
            context  = build_context(
                simulation_date    = date,
                policy_detail_text = di["policy_text"],
                state_confirmed    = di["state_confirmed_cases"],
                state_deaths       = di["state_deaths"],
                us_confirmed       = di["us_confirmed_cases"],
                us_deaths          = di["us_deaths"],
                rng                = dropout_rng(seed_key),
                dropout_rate       = args.dropout_rate,
            )
            prompt            = build_prompt(prompt_template, city_cfg, individual, context)
            response          = call_with_retry(model, prompt)
            parsed            = _parse_llm_json(response) if response else None
            predicted_changes = extract_poi_changes(parsed, response)

            if response is None:
                n_failed += 1

            occupation_subcat = individual.get("occupation", "")
            occupation_major  = OCCUPATION_TO_MAJOR_CATEGORY.get(occupation_subcat,
                                                                 occupation_subcat)
            record = {
                "city_name":       city_cfg["display_name"],
                "state_abbr":      state_abbr,
                "simulation_date": date,
                "agent_index":     agent_index,
                "sampling_method": "llm_ipf",
                "pipeline":        "neutral_prompt",
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
                    "state_confirmed_cases": di["state_confirmed_cases"],
                    "state_deaths":          di["state_deaths"],
                    "us_confirmed_cases":    di["us_confirmed_cases"],
                    "us_deaths":             di["us_deaths"],
                },
                "context_info": {
                    "n_context_items": context["n_context_items"],
                    "n_dropped":       context["n_dropped"],
                },
                "response":          response,
                "parsed_prediction": parsed,
                "predicted_changes": predicted_changes,
                # Provenance the single-stage files do not record.
                "prompt_template":     PROMPT_TEMPLATE_PATH.name,
                "prompt_person":       "first",
                "prompt_variant":      "neutral",
                # The neutral template requests no reasoning trace, so
                # parsed_prediction holds only the four *_change numbers.
                "reasoning_requested": False,
                "prompt_sha256":       fingerprint,
                "model_params":        model_params,
                "dropout_rate":        args.dropout_rate,
                "dropout_seed_key":    seed_key,
                # Kept even when absent, so every record carries the same keys.
                "source_file":          task.get("source_file"),
                "original_agent_index": task.get("original_agent_index"),
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            n_written += 1

            if i % 50 == 0 or i == len(pending):
                print(f"    {i}/{len(pending)} tasks  "
                      f"({time.time() - t0:.0f}s elapsed, {n_failed} API failure(s))")

    print(f"\n  [DONE] {n_written} records → {out_file.name}  ({time.time() - t0:.1f}s)")
    if n_failed:
        print(f"  [WARN] {n_failed} record(s) had no response after {MAX_RETRIES} attempts; "
              f"re-run to fill them in (delete those lines first).")


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--cities", nargs="+", default=DEFAULT_CITIES,
                   help=f"City names to run (default: {DEFAULT_CITIES}). "
                        f"Pass 'all' for every city in config.cities.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model names to run (default: config.models.simulation_models).")
    p.add_argument("--dates", nargs="+", default=None,
                   help="Simulation dates (default: config.simulation.simulation_dates).")
    p.add_argument("--agent-source", choices=["result-file", "pool-file"],
                   default="result-file",
                   help="Where the cohort comes from. 'result-file' (default) re-runs the exact "
                        "agents and dates of the single-stage results for each city x model. "
                        "'pool-file' uses results/{city}/llm_ipf_agents.jsonl, a DIFFERENT set.")
    p.add_argument("--dropout-rate", type=float, default=DROPOUT_RATE,
                   help=f"Fraction of context items to drop (default: {DROPOUT_RATE}, "
                        f"the value the original run used).")
    p.add_argument("--overwrite", action="store_true",
                   help="Truncate the output file instead of resuming into it.")
    p.add_argument("--allow-prompt-mismatch", action="store_true",
                   help="Append even when the existing file was written from another "
                        "version of the prompt template.")
    p.add_argument("--skip-prompt-checks", action="store_true",
                   help="Do not abort when the template still carries a framing cue.")
    p.add_argument("--print-prompt", action="store_true",
                   help="Print the first assembled prompt and exit without calling any model.")
    return p.parse_args()


def main():
    args = parse_args()

    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {PROMPT_TEMPLATE_PATH}")
    prompt_template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    missing, cues = check_prompt_template(prompt_template)
    if missing:
        raise ValueError(f"Prompt template is missing placeholders: {missing}")
    if cues and not args.skip_prompt_checks:
        raise ValueError(
            f"Prompt template still contains framing cue(s) this arm removes: {cues}. "
            f"Re-run with --skip-prompt-checks to proceed anyway."
        )

    if "all" in args.cities:
        cities = CITY_LIST
    else:
        cities  = [c for c in CITY_LIST if c["name"] in args.cities]
        missing_cities = set(args.cities) - {c["name"] for c in cities}
        if missing_cities:
            raise KeyError(f"Unknown city name(s): {sorted(missing_cities)}")

    models = args.models or MODEL_LIST
    for model_name in models:
        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{model_name}'. Check models.registry in config.json.")

    dates = args.dates or SIMULATION_DATES

    print("=" * 70)
    print("Policy-Agent: Neutral-Prompt City-level Simulation")
    print("=" * 70)
    print(f"Cities : {[c['display_name'] for c in cities]}")
    print(f"Models : {models}")
    print(f"Dates  : {dates}")
    print(f"Cohort : {'single-stage result files' if args.agent_source == 'result-file' else 'llm_ipf_agents.jsonl'}")
    print(f"Prompt : {_rel(PROMPT_TEMPLATE_PATH)}  (sha256[:16]={prompt_fingerprint(prompt_template)})")
    print(f"Dropout: {args.dropout_rate}")
    print(f"POIs   : {list(CANONICAL_TO_DISPLAY)}")
    print(f"Config : {_rel(CONFIG_PATH)}")
    print("=" * 70)

    for model_name in models:
        for city_cfg in cities:
            try:
                if args.agent_source == "result-file":
                    tasks = load_cohort_from_results(city_cfg["name"], model_name, dates)
                else:
                    tasks = load_cohort_from_pool(city_cfg["name"], dates)
                if not tasks:
                    print(f"  [SKIP] No tasks for {city_cfg['display_name']} / {model_name}.")
                    continue
            except Exception as exc:
                print(f"\n  [ERROR] Cohort for {city_cfg['display_name']} / {model_name}: {exc}")
                continue

            if args.print_prompt:
                task    = tasks[0]
                policy  = get_policy_text(load_policy_detail(city_cfg["state_abbr"]),
                                          task["simulation_date"])
                conf, deaths, us_conf, us_deaths = load_pandemic_data(city_cfg["state_abbr"])
                context = build_context(
                    simulation_date    = task["simulation_date"],
                    policy_detail_text = policy,
                    state_confirmed    = get_counts_for_date(conf,      task["simulation_date"]),
                    state_deaths       = get_counts_for_date(deaths,    task["simulation_date"]),
                    us_confirmed       = get_counts_for_date(us_conf,   task["simulation_date"]),
                    us_deaths          = get_counts_for_date(us_deaths, task["simulation_date"]),
                    rng                = dropout_rng(dropout_seed_key(
                                             city_cfg["name"], model_name,
                                             task["agent_index"], task["simulation_date"])),
                    dropout_rate       = args.dropout_rate,
                )
                print("\n" + "=" * 70)
                print(f"[{city_cfg['display_name']} / {model_name}] agent "
                      f"{task['agent_index']} × {task['simulation_date']}")
                print("=" * 70)
                print(build_prompt(prompt_template, city_cfg, task["individual"], context))
                print("=" * 70)
                continue

            try:
                run_city_model(city_cfg, model_name, prompt_template, tasks, args)
            except Exception as exc:
                print(f"\n  [ERROR] {city_cfg['display_name']} / {model_name}: {exc}")
                import traceback; traceback.print_exc()

    if not args.print_prompt:
        print("\n[INFO] Neutral-prompt simulation complete.")


if __name__ == "__main__":
    main()
