"""
City-level LLM pandemic simulation — DIRECT DISTRIBUTION (median) variant.

Difference from simulate.py / simulate_two_stage.py: there is no agent pool at
all.  Instead of asking 300 residents for one number each and assembling the
city distribution from their answers, this script asks the model *once per date*
to simulate the city directly and report the distribution itself.

  Input  : city background (population + aggregate demographic composition),
           the policy level in force on that date, and the current disease level
           (state and national case / death counts).
  Output : for every POI type, the five box-plot statistics of the city-level
           visitation-change distribution —
               whisker_low <= q25 <= median <= q75 <= whisker_high
           so each (model, date) result draws a box plot directly, with the
           median read straight off the model rather than aggregated from agents.

Cost is 1 LLM call per model × date (15 calls for 3 models × 5 dates), against
4,500 calls for the single-stage agent pipeline and 9,000 for the two-stage one.

Because a date is now a single call rather than 300, the per-agent context
dropout that gives the agent pipelines their diversity has no role here: there is
no population of prompts to diversify, and dropping context would only blind the
one call that has to answer for the whole city.  Dropout therefore defaults to 0
(see config.city_level.dropout_rate and --dropout-rate).

Data note: the city composition block is built from
data/city_data/{city}/cbg_detail_info.csv, which is derived from the POI data
that cannot be redistributed (see README → Data Availability).  Without that file
run with --demographics none (the city introduction and population alone), or
point --city-data-dir at your own copy.

All tunable parameters are loaded from config.json (section `city_level`).

Output per model:
  results/city_level_simulation/{city}/{model}/city_level_boxstats_{model}.jsonl
  results/city_level_simulation/{city}/{model}/city_level_boxstats_{model}.csv
"""

from __future__ import annotations

import argparse
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
    _extract_all_marginals,
    _normalize_cbg_geoid_str,
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
CITY_DATA_DIR = REPO_ROOT / "data" / "city_data"
PANDEMIC_DIR  = REPO_ROOT / "data" / "pandemic_data"
POLICY_DIR    = REPO_ROOT / "data" / "policy_data"
RESULTS_DIR   = REPO_ROOT / "results" / "city_level_simulation"
PROMPTS_DIR   = REPO_ROOT / "prompts"

PROMPT_TEMPLATE_PATH = PROMPTS_DIR / "city_level_distribution.txt"
CONFIRMED_FILE       = PANDEMIC_DIR / "aggregate_confirmed.csv"
DEATHS_FILE          = PANDEMIC_DIR / "aggregate_deaths.csv"

# From config
CITY_LIST        = _CFG["cities"]
MODEL_REGISTRY   = _CFG["models"]["registry"]
SIMULATION_DATES = _CFG["simulation"]["simulation_dates"]
RANDOM_SEED      = _CFG["simulation"]["random_seed"]

CANONICAL_TO_DISPLAY    = _CFG["poi"]["canonical_to_display"]
STATE_ABBR_TO_FULL_NAME = _CFG["state_names"]

_NEWS_CUTOFF = datetime.strptime(_CFG["simulation"]["news_cutoff"], "%Y-%m-%d")
_NEWS_ITEMS  = _CFG["simulation"]["news_items"]

_CL = _CFG.get("city_level", {})

DEFAULT_CITIES  = _CL.get("cities", ["san_antonio"])
DEFAULT_MODELS  = _CL.get("models", _CFG["models"]["simulation_models"])
DEFAULT_REPEATS = _CL.get("repeats", 1)
DROPOUT_RATE    = _CL.get("dropout_rate", 0.0)
MAX_TOKENS      = _CL.get("max_tokens_override", 8192)
MAX_RETRIES     = _CL.get("max_call_retries", 3)
RETRY_SLEEP     = _CL.get("retry_sleep", 2.0)
DEMOGRAPHICS    = _CL.get("demographics", "full")

RESULT_FILE_TEMPLATE  = _CL.get("result_file_template",  "city_level_boxstats_{model}.jsonl")
SUMMARY_FILE_TEMPLATE = _CL.get("summary_file_template", "city_level_boxstats_{model}.csv")

# The five box-plot statistics, in ascending order.  This ordering is load
# bearing: sanitisation sorts a model's five values into it.
STAT_ORDER = ["whisker_low", "q25", "median", "q75", "whisker_high"]

# Accepted spellings for each statistic in model output.  Matched as suffixes of
# "{poi}_{stat}", longest alias first so "25th_percentile" is never truncated to
# "percentile" and "upper_whisker" never loses to a shorter alias.
_STAT_ALIASES = {
    "whisker_low":  ["whisker_low", "whiskerlow", "whisker_lower", "lower_whisker",
                     "whislo", "whisker_min", "lower_fence", "p5", "min"],
    "q25":          ["q25", "q1", "p25", "quartile_25", "25th_percentile",
                     "percentile_25", "lower_quartile", "box_low", "box_bottom"],
    "median":       ["median", "q2", "p50", "med", "50th_percentile", "percentile_50"],
    "q75":          ["q75", "q3", "p75", "quartile_75", "75th_percentile",
                     "percentile_75", "upper_quartile", "box_high", "box_top"],
    "whisker_high": ["whisker_high", "whiskerhigh", "whisker_upper", "upper_whisker",
                     "whishi", "whisker_max", "upper_fence", "p95", "max"],
}

_ALIAS_LOOKUP = sorted(
    ((alias, stat) for stat, aliases in _STAT_ALIASES.items() for alias in aliases),
    key=lambda pair: -len(pair[0]),
)

# Fallback scan of raw text for "some_key": <number>
_NUMERIC_KV_RE = re.compile(r'"(?P<key>[^"]+?)"\s*:\s*(?P<val>-?\d+(?:\.\d+)?)')

# A city cannot lose more visits than it has.  Values below this are clipped.
CHANGE_FLOOR = -1.0


def _rel(path: Path) -> str:
    """Path relative to the repo root, for readable logging."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# =============================================================================
# City background: population composition
# =============================================================================

def resolve_city_file(city_name: str, filename: str, extra_root: Path | None) -> Path:
    """Find a per-city data file, preferring an explicit --city-data-dir root."""
    roots = ([extra_root] if extra_root else []) + [CITY_DATA_DIR]
    tried = []
    for root in roots:
        candidate = Path(root) / city_name / filename
        tried.append(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename} for '{city_name}'. Tried:\n  "
        + "\n  ".join(str(p) for p in tried)
        + "\nRun with --demographics none, or point --city-data-dir at your copy."
    )


def load_city_row(city_name: str, cbg_prefix: str, extra_root: Path | None) -> pd.Series:
    """
    Read cbg_detail_info.csv, normalise 11-digit GEOIDs to 12-digit, filter by
    the 5-digit county FIPS prefix, then sum all numeric columns into a single
    city-aggregated row.
    """
    csv_path = resolve_city_file(city_name, "cbg_detail_info.csv", extra_root)
    df = pd.read_csv(csv_path, dtype={"visitor_cbg": str})
    df["visitor_cbg"] = df["visitor_cbg"].map(_normalize_cbg_geoid_str)

    mask = df["visitor_cbg"].str.startswith(cbg_prefix)
    if mask.sum() == 0:
        raise ValueError(
            f"No CBGs found for prefix '{cbg_prefix}' in {csv_path.name}. "
            "Check that cbg_prefix is the correct 5-digit county FIPS code."
        )
    print(f"  CBGs matched: {mask.sum()} / {len(df)}  ({csv_path})")

    city_row = df.loc[mask].drop(columns=["visitor_cbg"]).sum(numeric_only=True)
    city_row["visitor_cbg"] = f"{cbg_prefix}_aggregated"
    return city_row


def _fmt_share_block(title: str, dist: dict, universe: str) -> str:
    """Render one demographic variable as 'label: NN.N%' lines."""
    total = sum(dist.values())
    if total <= 0:
        return ""
    lines = [f"- **{title}** (share of {universe}):"]
    for label, count in sorted(dist.items(), key=lambda kv: -kv[1]):
        lines.append(f"    - {label}: {100 * count / total:.1f}%")
    return "\n".join(lines)


def build_demographic_profile(city_row: pd.Series | None, detail: str = "full") -> str:
    """
    Render the city's aggregate population composition as prompt text.

    The 23 census occupation sub-categories are collapsed to the 7 major
    categories the agent prompts already use, so the model sees the same
    occupational vocabulary in both experiments.

    detail:
      "full"  — all six variables.
      "brief" — drops the age breakdown, keeping the five behaviourally
                strongest variables.
      "none"  — no composition block; the city introduction alone.
    """
    if detail == "none" or city_row is None:
        return "(No detailed composition provided.)"

    marginals = _extract_all_marginals(city_row)

    occupation_major: dict[str, float] = {}
    for sub, count in marginals["occupation"].items():
        major = OCCUPATION_TO_MAJOR_CATEGORY.get(sub, sub)
        occupation_major[major] = occupation_major.get(major, 0.0) + count

    blocks = [
        _fmt_share_block("Gender",           marginals["gender"],           "residents"),
        _fmt_share_block("Race",             marginals["race"],             "residents"),
        _fmt_share_block("Age",              marginals["age"],              "adults aged 21+"),
        _fmt_share_block("Education",        marginals["education"],        "adults"),
        _fmt_share_block("Household income", marginals["household_income"], "households"),
        _fmt_share_block("Occupation",       occupation_major,              "employed residents"),
    ]
    if detail == "brief":
        blocks = [b for b in blocks if not b.startswith("- **Age**")]

    return "\n".join(b for b in blocks if b)


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
# Context construction (policy level + disease level)
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
    Assemble the policy-timeline and disease-level text blocks for one date.

    With dropout_rate = 0 (the default for this experiment) the single city-level
    call sees the complete context.  A non-zero rate drops that fraction of the
    context items, matching the agent pipelines' behaviour, which is only useful
    when --repeats is high enough for the variation to mean something.
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


# =============================================================================
# Prompt construction
# =============================================================================

def _build_poi_blocks() -> tuple[str, str]:
    """Render the POI bullet list and the JSON output example from config.poi."""
    keys    = list(CANONICAL_TO_DISPLAY)
    bullets = "\n".join(f"- **{CANONICAL_TO_DISPLAY[k]}**" for k in keys)

    # Placeholder numbers deliberately differ per POI and per statistic so the
    # example cannot be mistaken for a suggested answer.
    example_values = [
        (-0.85, -0.60, -0.45, -0.30, -0.05),
        (-0.70, -0.40, -0.25, -0.10,  0.10),
        (-0.95, -0.80, -0.65, -0.40, -0.15),
        (-0.90, -0.75, -0.55, -0.35, -0.10),
    ]

    lines = ["{"]
    for i, k in enumerate(keys):
        vals = example_values[i % len(example_values)]
        tail = "," if i < len(keys) - 1 else ""
        lines.append(
            f'    "{k}_reasoning": "[Your step-by-step reasoning for '
            f'{CANONICAL_TO_DISPLAY[k]} across the five points in Part 5, ending '
            f'with the distribution you settled on.]",'
        )
        for stat, val in zip(STAT_ORDER, vals):
            last = stat == STAT_ORDER[-1]
            lines.append(f'    "{k}_{stat}": {val}{tail if last else ","}')
    lines.append("}")
    return bullets, "\n".join(lines)


POI_LIST_TEXT, POI_OUTPUT_EXAMPLE = _build_poi_blocks()


def build_prompt(template: str, city_cfg: dict, demographic_profile: str,
                 context: dict) -> str:
    """Fill the city-level prompt template."""
    state_full = STATE_ABBR_TO_FULL_NAME.get(city_cfg["state_abbr"], city_cfg["state_abbr"])

    out = template
    out = out.replace("**{{City_Name}}**",               city_cfg["display_name"])
    out = out.replace("**{{State}}**",                   state_full)
    out = out.replace("**{{City_State_Introduction}}**", city_cfg["introduction"])
    out = out.replace("**{{Total_Population}}**",        f"{city_cfg['total_population']:,}")
    out = out.replace("{{Demographic_Profile}}",         demographic_profile)
    out = out.replace("{{Policy_Timeline}}",             context["policy_timeline"])
    out = out.replace("{{Disease_Situation_Stats}}",     context["disease_stats"])
    out = out.replace("{{POI_List}}",                    POI_LIST_TEXT)
    out = out.replace("{{POI_Output_Example}}",          POI_OUTPUT_EXAMPLE)
    # City_Name also appears unbolded in the body of the template.
    out = out.replace("{{City_Name}}",                   city_cfg["display_name"])
    return out


# =============================================================================
# Response parsing
# =============================================================================

def _normalize_poi_key(raw_key: str) -> str:
    """Normalise a raw POI key from LLM output to one of the canonical keys."""
    k = re.sub(r"[&]", "and", raw_key.strip())
    k = re.sub(r"\s+", "_", k)
    k = re.sub(r"[^A-Za-z0-9_]", "", k)
    k = re.sub(r"^Restaurants?(?:_?and)?_?Bars?$",  "Restaurants_and_Bars",   k, flags=re.I)
    k = re.sub(r"^Retail$",                         "Retail",                 k, flags=re.I)
    k = re.sub(r"^Arts(?:_?and)?_?Entertainment$",  "Arts_and_Entertainment", k, flags=re.I)
    k = re.sub(r"^Educational(?:_?Settings?)?$",    "Educational_Settings",   k, flags=re.I)
    return k


def _split_poi_stat(raw_key: str) -> tuple[str, str] | None:
    """
    Split a flat output key such as 'Retail_q25' into ('Retail', 'q25').

    Returns None when the key carries no recognised statistic suffix.
    """
    key = raw_key.strip().rstrip(":").lower()
    for alias, stat in _ALIAS_LOOKUP:
        if key.endswith("_" + alias) or key == alias:
            prefix = raw_key.strip()[: len(raw_key.strip()) - len(alias)].rstrip("_ :")
            if not prefix:
                return None
            return _normalize_poi_key(prefix), stat
    return None


def _as_float(value) -> float | None:
    """Coerce a model-supplied value to float, tolerating '-45%' and '-45'."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        text = value.strip().rstrip("%").replace(",", "")
        try:
            num = float(text)
        except ValueError:
            return None
        # A trailing % means the model reported percentage points, not a fraction.
        return num / 100.0 if value.strip().endswith("%") else num
    return None


def extract_box_stats(parsed: dict | None, raw_text: str | None) -> tuple[dict, dict]:
    """
    Extract the five box-plot statistics for every configured POI type.

    Handles both the flat shape the prompt asks for ("Retail_q25": -0.3) and the
    nested shape models sometimes prefer ("Retail": {"q25": -0.3}).  Falls back
    to a regex scan of the raw text when JSON parsing came up short.

    Returns (stats, reasoning):
      stats     — {poi: {stat: float | None}} for every canonical POI.
      reasoning — {poi: str} for whatever reasoning fields were present.
    """
    stats: dict[str, dict[str, float]] = {poi: {} for poi in CANONICAL_TO_DISPLAY}
    reasoning: dict[str, str] = {}

    def _record(poi: str, stat: str, value) -> None:
        num = _as_float(value)
        if num is not None and poi in stats and stat not in stats[poi]:
            stats[poi][stat] = num

    if isinstance(parsed, dict):
        for key, value in parsed.items():
            if not isinstance(key, str):
                continue

            # Nested: {"Retail": {"q25": ..., "median": ...}}
            if isinstance(value, dict):
                poi = _normalize_poi_key(key)
                if poi in stats:
                    for sub_key, sub_val in value.items():
                        if not isinstance(sub_key, str):
                            continue
                        if sub_key.lower().endswith("reasoning"):
                            if isinstance(sub_val, str):
                                reasoning.setdefault(poi, sub_val.strip())
                            continue
                        split = _split_poi_stat(f"{poi}_{sub_key}")
                        if split:
                            _record(poi, split[1], sub_val)
                continue

            if key.lower().endswith("reasoning"):
                if isinstance(value, str):
                    poi = _normalize_poi_key(key[: -len("_reasoning")].rstrip("_"))
                    if poi in stats:
                        reasoning.setdefault(poi, value.strip())
                continue

            split = _split_poi_stat(key)
            if split:
                _record(split[0], split[1], value)

    # Regex fallback for any POI still missing statistics.
    if raw_text and any(len(s) < len(STAT_ORDER) for s in stats.values()):
        for match in _NUMERIC_KV_RE.finditer(raw_text):
            split = _split_poi_stat(match.group("key"))
            if split:
                _record(split[0], split[1], match.group("val"))

    return ({poi: {stat: vals.get(stat) for stat in STAT_ORDER}
             for poi, vals in stats.items()},
            reasoning)


def sanitize_stats(raw: dict) -> tuple[dict | None, dict]:
    """
    Validate one POI's five statistics and repair what is safely repairable.

    Returns (clean_stats, flags).  clean_stats is None when any statistic is
    missing — the caller retries the whole call in that case.

    Repairs applied:
      - values below CHANGE_FLOOR are clipped up to it ("clipped");
      - the five values are sorted ascending ("reordered"), which is the correct
        repair because they are order statistics of one distribution and so can
        only ever have been emitted out of order.

    Flagged but not modified:
      - whiskers lying outside the Tukey 1.5 × IQR fences ("whisker_outside_fence"),
        which means the model did not use the whisker convention the prompt asked
        for.  The box is still drawable, so the value is kept and reported.
    """
    values = [raw.get(stat) for stat in STAT_ORDER]
    flags  = {"complete": all(v is not None for v in values),
              "clipped": False, "reordered": False, "whisker_outside_fence": False}

    if not flags["complete"]:
        flags["missing"] = [s for s, v in zip(STAT_ORDER, values) if v is None]
        return None, flags

    clipped = [max(float(v), CHANGE_FLOOR) for v in values]
    flags["clipped"] = clipped != [float(v) for v in values]

    ordered = sorted(clipped)
    flags["reordered"] = ordered != clipped

    clean = dict(zip(STAT_ORDER, ordered))

    iqr = clean["q75"] - clean["q25"]
    tol = 1e-9
    if (clean["whisker_low"]  < clean["q25"] - 1.5 * iqr - tol or
            clean["whisker_high"] > clean["q75"] + 1.5 * iqr + tol):
        flags["whisker_outside_fence"] = True

    return clean, flags


# =============================================================================
# Single city-level simulation call
# =============================================================================

def simulate_city_date(
    model,
    model_name: str,
    city_cfg: dict,
    demographic_profile: str,
    prompt_template: str,
    simulation_date: str,
    repeat_index: int,
    context: dict,
    pandemic_info: dict,
) -> dict:
    """
    Run one city-level call and return the record.

    Retries up to MAX_RETRIES times while any POI is missing statistics — a
    single truncated or malformed response would otherwise cost the whole date.
    """
    prompt = build_prompt(prompt_template, city_cfg, demographic_profile, context)

    response = parsed = None
    box_stats: dict[str, dict | None] = {}
    quality:   dict[str, dict] = {}
    reasoning: dict[str, str]  = {}
    attempts = 0

    for attempt in range(1, MAX_RETRIES + 1):
        attempts  = attempt
        response  = model.call(prompt)
        parsed    = _parse_llm_json(response) if response else None
        raw_stats, reasoning = extract_box_stats(parsed, response)

        box_stats, quality = {}, {}
        for poi, values in raw_stats.items():
            clean, flags   = sanitize_stats(values)
            box_stats[poi] = clean
            quality[poi]   = flags

        missing = [poi for poi, clean in box_stats.items() if clean is None]
        if not missing:
            break
        if attempt < MAX_RETRIES:
            print(f"      [retry {attempt}/{MAX_RETRIES - 1}] incomplete statistics "
                  f"for {missing}; re-calling")
            time.sleep(RETRY_SLEEP * attempt)

    return {
        "city_name":       city_cfg["display_name"],
        "city":            city_cfg["name"],
        "state_abbr":      city_cfg["state_abbr"],
        "simulation_date": simulation_date,
        "model":           model_name,
        "repeat_index":    repeat_index,
        "level":           "city",
        "pipeline":        "city_level_direct",
        "pandemic_info":   pandemic_info,
        "context_info": {
            "n_context_items": context["n_context_items"],
            "n_dropped":       context["n_dropped"],
            "policy_timeline": context["policy_timeline"],
            "disease_stats":   context["disease_stats"],
        },
        "attempts":          attempts,
        "response":          response,
        "parsed_prediction": parsed,
        "reasoning":         reasoning,
        "box_stats":         box_stats,
        "quality":           quality,
    }


# =============================================================================
# Model construction
# =============================================================================

def build_model(model_name: str, max_tokens: int):
    """
    Instantiate a model from config.models.registry, overriding max_tokens.

    The registry budgets are sized for one agent answering about one week; a
    city-level call has to carry reasoning plus five statistics for every POI in
    a single response, so it needs a larger ceiling
    (config.city_level.max_tokens_override).
    """
    cfg = MODEL_REGISTRY[model_name]
    kwargs: dict = {
        "model_name":  cfg["api_model_name"],
        "temperature": cfg.get("temperature", 1),
        "max_tokens":  max_tokens,
    }
    if "enable_thinking" in cfg:
        kwargs["enable_thinking"] = cfg["enable_thinking"]
    return create_model(cfg["type"], **kwargs)


# =============================================================================
# City × model pipeline
# =============================================================================

def run_city_model(city_cfg: dict, model_name: str, prompt_template: str,
                   demographic_profile: str, dates: list[str], args) -> None:
    """Run every simulation date for one city × one model."""
    state_abbr = city_cfg["state_abbr"]

    print(f"\n{'#' * 70}")
    print(f"  City: {city_cfg['display_name']}  |  Model: {model_name}  |  city-level")
    print(f"{'#' * 70}")

    model = build_model(model_name, args.max_tokens)

    state_conf_row, state_deaths_row, us_conf_row, us_deaths_row = load_pandemic_data(state_abbr)
    policy_df = load_policy_detail(state_abbr)

    out_dir = RESULTS_DIR / city_cfg["name"] / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / RESULT_FILE_TEMPLATE.format(model=model_name)
    csv_file = out_dir / SUMMARY_FILE_TEMPLATE.format(model=model_name)
    if out_file.exists():
        print(f"  [WARN] Overwriting existing {out_file.name}")
    out_file.write_text("")
    print(f"  Output: {_rel(out_file)}")
    print(f"          {_rel(csv_file)}")

    prompt_rng   = random.Random(RANDOM_SEED)
    tidy_rows    = []
    n_incomplete = 0
    t0           = time.time()

    with open(out_file, "a", encoding="utf-8") as f_out:
        for d_idx, sim_date in enumerate(dates, start=1):
            state_confirmed = get_counts_for_date(state_conf_row,   sim_date)
            state_deaths    = get_counts_for_date(state_deaths_row, sim_date)
            us_confirmed    = get_counts_for_date(us_conf_row,      sim_date)
            us_deaths       = get_counts_for_date(us_deaths_row,    sim_date)
            policy_text     = get_policy_text(policy_df,            sim_date)

            pandemic_info = {
                "state_confirmed_cases": state_confirmed,
                "state_deaths":          state_deaths,
                "us_confirmed_cases":    us_confirmed,
                "us_deaths":             us_deaths,
            }

            print(f"\n  Date {d_idx}/{len(dates)}: {sim_date}"
                  f"  (state_cases={state_confirmed}, us_cases={us_confirmed})")

            for repeat in range(args.repeats):
                context = build_context(
                    simulation_date    = sim_date,
                    policy_detail_text = policy_text,
                    state_confirmed    = state_confirmed,
                    state_deaths       = state_deaths,
                    us_confirmed       = us_confirmed,
                    us_deaths          = us_deaths,
                    rng                = prompt_rng,
                    dropout_rate       = args.dropout_rate,
                )
                record = simulate_city_date(
                    model               = model,
                    model_name          = model_name,
                    city_cfg            = city_cfg,
                    demographic_profile = demographic_profile,
                    prompt_template     = prompt_template,
                    simulation_date     = sim_date,
                    repeat_index        = repeat,
                    context             = context,
                    pandemic_info       = pandemic_info,
                )
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                for poi, clean in record["box_stats"].items():
                    flags = record["quality"][poi]
                    if clean is None:
                        n_incomplete += 1
                        print(f"      [WARN] {poi}: incomplete after {record['attempts']} "
                              f"attempt(s); missing {flags.get('missing')}")
                        continue
                    tidy_rows.append({
                        "city":         city_cfg["name"],
                        "model":        model_name,
                        "date":         sim_date,
                        "repeat_index": repeat,
                        "poi":          poi,
                        **clean,
                        "iqr":          clean["q75"] - clean["q25"],
                        "reordered":    flags["reordered"],
                        "clipped":      flags["clipped"],
                        "whisker_outside_fence": flags["whisker_outside_fence"],
                    })

                shown = ", ".join(
                    f"{poi.split('_')[0]}={record['box_stats'][poi]['median']:+.2f}"
                    for poi in record["box_stats"] if record["box_stats"][poi]
                )
                print(f"      repeat {repeat + 1}/{args.repeats}: median {shown or '(no output)'}"
                      f"  ({time.time() - t0:.0f}s)")

    if tidy_rows:
        pd.DataFrame(tidy_rows).to_csv(csv_file, index=False, encoding="utf-8")
        print("\n  Medians by date × POI:")
        pivot = (pd.DataFrame(tidy_rows)
                 .pivot_table(index="date", columns="poi", values="median", aggfunc="mean"))
        print(pivot.to_string(float_format=lambda v: f"{v:+.3f}"))

    print(f"\n  [DONE] {len(tidy_rows)} POI-distributions → {out_file.name}  "
          f"({time.time() - t0:.1f}s)")
    if n_incomplete:
        print(f"  [WARN] {n_incomplete} POI-distribution(s) never came back complete.")


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--cities", nargs="+", default=DEFAULT_CITIES,
                   help=f"City names to run (default: {DEFAULT_CITIES}). "
                        f"Pass 'all' for every city in config.cities.")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help=f"Model names to run (default: {DEFAULT_MODELS}).")
    p.add_argument("--dates", nargs="+", default=None,
                   help="Simulation dates to run (default: config.simulation.simulation_dates).")
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                   help=f"LLM calls per model x date (default: {DEFAULT_REPEATS}). "
                        "The experiment is specified as one call per date; raise this "
                        "to average the five statistics over independent runs.")
    p.add_argument("--dropout-rate", type=float, default=DROPOUT_RATE,
                   help=f"Fraction of context items to drop (default: {DROPOUT_RATE}). "
                        "0 gives the single city-level call the complete context.")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                   help=f"Response token ceiling, overriding the registry value "
                        f"(default: {MAX_TOKENS}).")
    p.add_argument("--demographics", choices=["full", "brief", "none"], default=DEMOGRAPHICS,
                   help=f"How much of the city's population composition to include in the "
                        f"prompt (default: {DEMOGRAPHICS}). 'none' needs no CBG data file.")
    p.add_argument("--city-data-dir", default=None,
                   help="Directory holding {city}/cbg_detail_info.csv, searched before "
                        "data/city_data/. Use it when the CBG data lives outside the repo.")
    p.add_argument("--print-prompt", action="store_true",
                   help="Print the first assembled prompt and exit without calling any model.")
    return p.parse_args()


def main():
    args = parse_args()

    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {PROMPT_TEMPLATE_PATH}")
    prompt_template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    if "all" in args.cities:
        cities = CITY_LIST
    else:
        cities  = [c for c in CITY_LIST if c["name"] in args.cities]
        missing = set(args.cities) - {c["name"] for c in cities}
        if missing:
            raise KeyError(f"Unknown city name(s): {sorted(missing)}")

    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{model_name}'. "
                           f"Check models.registry in config.json.")

    dates = args.dates or SIMULATION_DATES
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    extra_root = Path(args.city_data_dir).expanduser() if args.city_data_dir else None

    print("=" * 70)
    print("Policy-Agent: Direct City-level Distribution Simulation")
    print("=" * 70)
    print(f"Cities  : {[c['display_name'] for c in cities]}")
    print(f"Models  : {args.models}")
    print(f"Dates   : {dates}")
    print(f"POIs    : {list(CANONICAL_TO_DISPLAY)}")
    print(f"Output  : {len(STAT_ORDER)} box-plot statistics per POI x date "
          f"({', '.join(STAT_ORDER)})")
    print(f"Calls   : {len(args.models)} models x {len(dates)} dates x "
          f"{args.repeats} repeat(s) = "
          f"{len(args.models) * len(dates) * args.repeats} per city")
    print(f"Dropout : {args.dropout_rate}")
    print(f"Prompt  : {_rel(PROMPT_TEMPLATE_PATH)}")
    print(f"Results : {_rel(RESULTS_DIR)}")
    print(f"Config  : {_rel(CONFIG_PATH)}")
    print("=" * 70)

    for city_cfg in cities:
        print(f"\n[City Background] {city_cfg['display_name']}")
        try:
            city_row = (None if args.demographics == "none"
                        else load_city_row(city_cfg["name"], city_cfg["cbg_prefix"], extra_root))
            demographic_profile = build_demographic_profile(city_row, args.demographics)
        except Exception as exc:
            print(f"  [ERROR] Cannot build city background: {exc}")
            continue

        if args.print_prompt:
            context = build_context(
                simulation_date    = dates[0],
                policy_detail_text = get_policy_text(
                    load_policy_detail(city_cfg["state_abbr"]), dates[0]),
                state_confirmed = 0, state_deaths = 0,
                us_confirmed    = 0, us_deaths    = 0,
                rng             = random.Random(RANDOM_SEED),
                dropout_rate    = args.dropout_rate,
            )
            print("\n" + "=" * 70)
            print(build_prompt(prompt_template, city_cfg, demographic_profile, context))
            print("=" * 70)
            continue

        for model_name in args.models:
            try:
                run_city_model(city_cfg, model_name, prompt_template,
                               demographic_profile, dates, args)
            except Exception as exc:
                print(f"\n  [ERROR] {city_cfg['display_name']} / {model_name}: {exc}")
                import traceback; traceback.print_exc()

    if not args.print_prompt:
        print("\n[INFO] City-level simulation complete.")


if __name__ == "__main__":
    main()
