"""
Shared utilities for CBG demographic marginals, agent sampling, and LLM parsing.
Used by simulate.py and calculate_results.py.

Includes:
  - Column mappings for 6 demographic variables
  - Marginal / joint distribution extraction from CBG rows
  - Simple weighted-random and full LLM+IPF agent sampling
  - Prompt-formatting helpers for Phase 1 & Phase 2 prompts
  - JSON response parsing
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Paths
# =============================================================================

# Repo root is two levels above this file: code/ → Policy-Agent/
REPO_ROOT    = Path(__file__).resolve().parent.parent
PROMPTS_DIR  = REPO_ROOT / "prompts"

# =============================================================================
# Demographic column mappings
# =============================================================================

RACE_COLUMNS = {
    "White":      "white population",
    "Black":      "black population",
    "Other Race": "other race population",
}

EDUCATION_COLUMNS = {
    "Highschool Degree or Lower": "highschool degree or lower degree population",
    "Associate's Degree":         "associate's degree population",
    "Bachelor's Degree":          "bachelor's degree population",
    "Master's Degree":            "master's degree population",
    "Professional School Degree": "professional school degree population",
    "Doctorate Degree":           "doctorate degree population",
}

INCOME_COLUMNS = {
    "Low Income":    "low income households",
    "Medium Income": "medium income households",
    "High Income":   "high income households",
}

GENDER_COLUMNS = {
    "Male":   "male population",
    "Female": "female population",
}

MALE_AGE_GROUPS = {
    "21-29 years": ["male & 21 years", "male & 22 to 24 years", "male & 25 to 29 years"],
    "30-39 years": ["male & 30 to 34 years", "male & 35 to 39 years"],
    "40-49 years": ["male & 40 to 44 years", "male & 45 to 49 years"],
    "50-59 years": ["male & 50 to 54 years", "male & 55 to 59 years"],
    "60-69 years": ["male & 60 and 61 years", "male & 62 to 64 years",
                    "male & 65 and 66 years", "male & 67 to 69 years"],
    "70-79 years": ["male & 70 to 74 years", "male & 75 to 79 years"],
    "80+ years":   ["male & 80 to 84 years", "male & 85 years and over"],
}

FEMALE_AGE_GROUPS = {
    "21-29 years": ["female & 21 years", "female & 22 to 24 years", "female & 25 to 29 years"],
    "30-39 years": ["female & 30 to 34 years", "female & 35 to 39 years"],
    "40-49 years": ["female & 40 to 44 years", "female & 45 to 49 years"],
    "50-59 years": ["female & 50 to 54 years", "female & 55 to 59 years"],
    "60-69 years": ["female & 60 and 61 years", "female & 62 to 64 years",
                    "female & 65 and 66 years", "female & 67 to 69 years"],
    "70-79 years": ["female & 70 to 74 years", "female & 75 to 79 years"],
    "80+ years":   ["female & 80 to 84 years", "female & 85 years and over"],
}

MALE_OCCUPATION_COLUMNS = {
    "Management":                                    "male & management occupations",
    "Business and Financial Operations":             "male & business and financial operations occupations",
    "Computer and Mathematical":                     "male & computer and mathematical occupations",
    "Architecture and Engineering":                  "male & architecture and engineering occupations",
    "Life, Physical and Social Science":             "male & life physical and social science occupations",
    "Community and Social Service":                  "male & community and social service occupations",
    "Legal":                                         "male & legal occupations",
    "Educational Instruction and Library":           "male & educational instruction and library occupations",
    "Arts, Design, Entertainment, Sports and Media": "male & arts design entertainment sports and media occupations",
    "Healthcare Practitioners and Technical":        "male & healthcare practitioners and technical occupations",
    "Healthcare Support":                            "male & healthcare support occupations",
    "Protective Service":                            "male & protective service occupations",
    "Food Preparation and Serving":                  "male & food preparation and serving related occupations",
    "Building and Grounds Cleaning and Maintenance": "male & building and grounds cleaning and maintenance occupations",
    "Personal Care and Service":                     "male & personal care and service occupations",
    "Sales and Related":                             "male & sales and related occupations",
    "Office and Administrative Support":             "male & office and administrative support occupations",
    "Farming, Fishing and Forestry":                 "male & farming fishing and forestry occupations",
    "Construction and Extraction":                   "male & construction and extraction occupations",
    "Installation, Maintenance and Repair":          "male & installation maintenance and repair occupations",
    "Production":                                    "male & production occupations",
    "Transportation":                                "male & transportation occupations",
    "Material Moving":                               "male & material moving occupations",
}

FEMALE_OCCUPATION_COLUMNS = {
    k: v.replace("male & ", "female & ")
    for k, v in MALE_OCCUPATION_COLUMNS.items()
}

OCCUPATION_TO_MAJOR_CATEGORY = {
    "Management":                                    "High-Skill Professional & Technical",
    "Business and Financial Operations":             "High-Skill Professional & Technical",
    "Computer and Mathematical":                     "High-Skill Professional & Technical",
    "Architecture and Engineering":                  "High-Skill Professional & Technical",
    "Life, Physical and Social Science":             "High-Skill Professional & Technical",
    "Legal":                                         "High-Skill Professional & Technical",
    "Community and Social Service":                  "Education & Public / Social Services",
    "Educational Instruction and Library":           "Education & Public / Social Services",
    "Healthcare Practitioners and Technical":        "Healthcare",
    "Healthcare Support":                            "Healthcare",
    "Arts, Design, Entertainment, Sports and Media": "Creative & Cultural",
    "Protective Service":                            "Service Occupations",
    "Food Preparation and Serving":                  "Service Occupations",
    "Building and Grounds Cleaning and Maintenance": "Service Occupations",
    "Personal Care and Service":                     "Service Occupations",
    "Sales and Related":                             "Clerical & Sales",
    "Office and Administrative Support":             "Clerical & Sales",
    "Farming, Fishing and Forestry":                 "Blue-Collar / Manual Labor",
    "Construction and Extraction":                   "Blue-Collar / Manual Labor",
    "Installation, Maintenance and Repair":          "Blue-Collar / Manual Labor",
    "Production":                                    "Blue-Collar / Manual Labor",
    "Transportation":                                "Blue-Collar / Manual Labor",
    "Material Moving":                               "Blue-Collar / Manual Labor",
}

# The 6 demographic variables used throughout LLM+IPF
_LLM_IPF_VARIABLES = ["race", "education", "household_income", "gender", "age", "occupation"]

# Real joint distributions available from census (variable-pair → canonical name)
_REAL_JOINT_VAR_ORDER = {
    "gender_x_age":        ["gender", "age"],
    "gender_x_occupation": ["gender", "occupation"],
}


# =============================================================================
# Basic helpers
# =============================================================================

def safe_float(value, default=0.0):
    """Convert value to float, returning default on NaN / None / error."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_cbg_geoid_str(val) -> str:
    """US CBG GEOID is 12 digits; prepend '0' if stored as 11-digit string."""
    s = str(val).strip()
    return ("0" + s) if (len(s) == 11 and s.isdigit()) else s


def _scale_dist(dist: dict, k: float) -> dict:
    """Scale a count distribution so its total equals k."""
    total = sum(dist.values())
    if total <= 0:
        n = len(dist) or 1
        return {c: k / n for c in dist}
    return {c: v / total * k for c, v in dist.items()}


def _max_relative_error(delta: dict, target: dict) -> float:
    """Maximum |delta| / target across all categories with target >= 1."""
    mx = 0.0
    for var in delta:
        for cat, dv in delta[var].items():
            tv = target[var].get(cat, 0)
            if tv >= 1.0:
                mx = max(mx, abs(dv) / tv)
    return mx


# =============================================================================
# Distribution extraction from CBG rows
# =============================================================================

def _extract_all_marginals(row) -> dict:
    """Extract 6-variable marginal counts from a CBG row (or aggregated city row)."""
    return {
        "race": {
            n: safe_float(row.get(c, 0)) for n, c in RACE_COLUMNS.items()
        },
        "education": {
            n: safe_float(row.get(c, 0)) for n, c in EDUCATION_COLUMNS.items()
        },
        "household_income": {
            n: safe_float(row.get(c, 0)) for n, c in INCOME_COLUMNS.items()
        },
        "gender": {
            n: safe_float(row.get(c, 0)) for n, c in GENDER_COLUMNS.items()
        },
        "age": {
            grp: (
                sum(safe_float(row.get(c, 0)) for c in MALE_AGE_GROUPS[grp])
                + sum(safe_float(row.get(c, 0)) for c in FEMALE_AGE_GROUPS[grp])
            )
            for grp in MALE_AGE_GROUPS
        },
        "occupation": {
            n: (
                safe_float(row.get(MALE_OCCUPATION_COLUMNS[n], 0))
                + safe_float(row.get(FEMALE_OCCUPATION_COLUMNS[n], 0))
            )
            for n in MALE_OCCUPATION_COLUMNS
        },
    }


def _extract_all_joints(row) -> dict:
    """Extract available joint distributions: gender×age and gender×occupation."""
    ga = {}
    for gender in ("Male", "Female"):
        adict = MALE_AGE_GROUPS if gender == "Male" else FEMALE_AGE_GROUPS
        for grp, cols in adict.items():
            ga[f"{gender} & {grp}"] = sum(safe_float(row.get(c, 0)) for c in cols)
    go = {}
    for gender in ("Male", "Female"):
        odict = MALE_OCCUPATION_COLUMNS if gender == "Male" else FEMALE_OCCUPATION_COLUMNS
        for occ, col in odict.items():
            go[f"{gender} & {occ}"] = safe_float(row.get(col, 0))
    return {"gender_x_age": ga, "gender_x_occupation": go}


# =============================================================================
# LLM response parsing
# =============================================================================

def _parse_llm_json(response_text: str):
    """Parse a JSON object from an LLM response (tolerates code fences / trailing commas)."""
    if not response_text or not response_text.strip():
        return None
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    for candidate in (text, cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        raw = match.group(0)
        for candidate in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    return None


# =============================================================================
# Weighted-random sampling (used as fallback inside LLM+IPF)
# =============================================================================

def _weighted_random_choice(options_dict: dict, row) -> str:
    """Randomly select a category label proportionally to its population weight."""
    labels  = list(options_dict.keys())
    weights = [safe_float(row.get(col, 0)) for col in options_dict.values()]
    total   = sum(weights)
    if total <= 0:
        return np.random.choice(labels)
    return np.random.choice(labels, p=[w / total for w in weights])


def _weighted_random_age_group(age_groups: dict, row) -> str:
    """Randomly select an age group proportionally to summed sub-column population."""
    groups  = list(age_groups.keys())
    weights = [sum(safe_float(row.get(c, 0)) for c in cols)
               for cols in age_groups.values()]
    total   = sum(weights)
    if total <= 0:
        return np.random.choice(groups)
    return np.random.choice(groups, p=[w / total for w in weights])


def select_random_individuals_from_cbg(row, k: int = 1) -> list[dict]:
    """
    Generate k individuals by independently sampling each attribute from
    population-weighted marginals.  Used as a fallback inside LLM+IPF.
    """
    individuals = []
    for _ in range(k):
        gender = _weighted_random_choice(GENDER_COLUMNS, row)
        age_groups  = MALE_AGE_GROUPS        if gender == "Male" else FEMALE_AGE_GROUPS
        occ_columns = MALE_OCCUPATION_COLUMNS if gender == "Male" else FEMALE_OCCUPATION_COLUMNS
        individuals.append({
            "race":             _weighted_random_choice(RACE_COLUMNS,      row),
            "education":        _weighted_random_choice(EDUCATION_COLUMNS, row),
            "household_income": _weighted_random_choice(INCOME_COLUMNS,    row),
            "gender":           gender,
            "age":              _weighted_random_age_group(age_groups, row),
            "occupation":       _weighted_random_choice(occ_columns, row),
        })
    return _map_occupations_to_major(individuals)


def _map_occupations_to_major(individuals: list[dict]) -> list[dict]:
    """Replace each individual's occupation sub-category with its 7-category major label."""
    for ind in individuals:
        ind["occupation"] = OCCUPATION_TO_MAJOR_CATEGORY.get(
            ind.get("occupation", ""), ind.get("occupation", "")
        )
    return individuals


# =============================================================================
# LLM+IPF internal helpers
# =============================================================================

def _count_individual_marginals(individuals: list[dict]) -> dict:
    """Count marginal frequencies of each variable across a list of agent dicts."""
    counts = {v: {} for v in _LLM_IPF_VARIABLES}
    for ind in individuals:
        for v in _LLM_IPF_VARIABLES:
            val = ind.get(v)
            if val is not None:
                counts[v][val] = counts[v].get(val, 0) + 1
    return counts


def _compute_llm_ipf_delta(target: dict, current: dict) -> dict:
    """Return target - current for every variable / category."""
    return {
        var: {cat: t - current.get(var, {}).get(cat, 0)
              for cat, t in tgt.items()}
        for var, tgt in target.items()
    }


def _count_joint_from_individuals(individuals: list[dict], variables: list[str]) -> dict:
    """Count joint frequencies for a combination of variables."""
    counts = {}
    for ind in individuals:
        vals = [ind.get(v) for v in variables]
        if all(v is not None for v in vals):
            key = " & ".join(vals)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _compute_joint_delta(target_joint: dict, current_joint: dict) -> dict:
    """Return target - current for each category in a joint distribution."""
    return {key: tgt - current_joint.get(key, 0) for key, tgt in target_joint.items()}


def _max_joint_relative_error(joint_deltas: dict, joint_targets: dict) -> float:
    """Maximum |delta| / target across all real joint distributions with target >= 1."""
    mx = 0.0
    for jname, delta in joint_deltas.items():
        tgt = joint_targets.get(jname, {})
        for key, dv in delta.items():
            tv = tgt.get(key, 0)
            if tv >= 1.0:
                mx = max(mx, abs(dv) / tv)
    return mx


def _residual_sample(var_name: str, target: dict, current_counts: dict) -> str:
    """
    Sample one category for var_name using residual-weighted sampling
    (prefers under-generated categories; falls back to proportional from target).
    """
    tgt  = target.get(var_name, {})
    cur  = current_counts.get(var_name, {})
    cats = list(tgt.keys())
    if not cats:
        return None
    residuals = [max(tgt.get(c, 0) - cur.get(c, 0), 0) for c in cats]
    total_r   = sum(residuals)
    if total_r > 0:
        return np.random.choice(cats, p=[r / total_r for r in residuals])
    weights = [max(tgt.get(c, 0), 0) for c in cats]
    tw = sum(weights)
    if tw > 0:
        return np.random.choice(cats, p=[w / tw for w in weights])
    return np.random.choice(cats)


def _fallback_proposals(delta: dict, remaining: int) -> list[dict]:
    """Generate proposals from the largest positive delta when the LLM returns nothing usable."""
    best_var, best_total = None, 0.0
    for var, d in delta.items():
        pt = sum(max(dv, 0) for dv in d.values())
        if pt > best_total:
            best_var, best_total = var, pt
    if best_var is None or best_total == 0:
        return [{"count": remaining, "constraints": {}}]
    pos = {c: dv for c, dv in delta[best_var].items() if dv > 0.5}
    if not pos:
        return [{"count": remaining, "constraints": {}}]
    return [
        {"count": max(1, int(remaining * dv / best_total)),
         "constraints": {best_var: cat}}
        for cat, dv in pos.items()
    ]


# =============================================================================
# Prompt-formatting helpers (Phase 1 & Phase 2)
# =============================================================================

def _fmt_marginals_text(marginals: dict) -> str:
    parts = []
    for var, dist in marginals.items():
        total = sum(dist.values())
        parts.append(f"\n{var} (total: {total:.0f}):")
        for cat, cnt in dist.items():
            pct = cnt / total * 100 if total > 0 else 0
            parts.append(f"  {cat}: {cnt:.0f} ({pct:.1f}%)")
    return "\n".join(parts)


def _fmt_joints_text(joints: dict) -> str:
    parts = []
    for jname, dist in joints.items():
        total = sum(dist.values())
        parts.append(f"\n{jname} (total: {total:.0f}):")
        for key, cnt in dist.items():
            if cnt > 0:
                parts.append(f"  {key}: {cnt:.0f}")
    return "\n".join(parts)


def _fmt_counts_text(counts: dict) -> str:
    parts = []
    for var in _LLM_IPF_VARIABLES:
        dist = counts.get(var, {})
        if dist:
            parts.append(f"\n{var}:")
            for cat, cnt in dist.items():
                parts.append(f"  {cat}: {cnt:.1f}")
        else:
            parts.append(f"\n{var}: (none yet)")
    return "\n".join(parts)


def _fmt_delta_text(delta: dict, target: dict) -> str:
    parts = []
    for var in _LLM_IPF_VARIABLES:
        d   = delta.get(var, {})
        pos = {c: dv for c, dv in d.items() if dv > 0.5}
        if pos:
            parts.append(f"\n{var} — under-generated:")
            for cat, dv in sorted(pos.items(), key=lambda x: -x[1]):
                tv = target[var].get(cat, 0)
                parts.append(f"  {cat}: need +{dv:.1f} more (target: {tv:.1f})")
    for var in _LLM_IPF_VARIABLES:
        d   = delta.get(var, {})
        neg = {c: dv for c, dv in d.items() if dv < -0.5}
        if neg:
            parts.append(f"\n{var} — over-generated (avoid these):")
            for cat, dv in sorted(neg.items(), key=lambda x: x[1]):
                parts.append(f"  {cat}: excess {abs(dv):.1f}")
    return "\n".join(parts)


def _fmt_joint_delta_text(
    real_joint_deltas: dict,
    real_joint_targets: dict,
    approx_joint_deltas: dict,
    approx_joint_targets: dict,
    lambda_weight: float,
) -> str:
    parts = []
    for jname, delta in real_joint_deltas.items():
        tgt = real_joint_targets.get(jname, {})
        pos = {k: v for k, v in delta.items() if v > 0.5}
        if pos:
            parts.append(f"\n{jname} (census data, high confidence) — under-generated:")
            for key, dv in sorted(pos.items(), key=lambda x: -x[1]):
                tv = tgt.get(key, 0)
                parts.append(f"  {key}: need +{dv:.1f} more (target: {tv:.1f})")
        neg = {k: v for k, v in delta.items() if v < -0.5}
        if neg:
            parts.append(f"\n{jname} (census data) — over-generated (avoid):")
            for key, dv in sorted(neg.items(), key=lambda x: x[1]):
                parts.append(f"  {key}: excess {abs(dv):.1f}")
    for jname, delta in approx_joint_deltas.items():
        weighted = {k: v * lambda_weight for k, v in delta.items() if v > 0.5}
        if weighted:
            parts.append(
                f"\n{jname} (LLM-estimated, low confidence — "
                f"directional hints only, weight={lambda_weight}):"
            )
            for key, dv in sorted(weighted.items(), key=lambda x: -x[1]):
                parts.append(f"  {key}: suggest ~+{dv:.1f} more")
    if not parts:
        parts.append("\n(No significant joint distribution gaps)")
    return "\n".join(parts)


# =============================================================================
# LLM+IPF agent sampling
# =============================================================================

def select_LLM_IPF_individuals_from_cbg(
    row,
    k: int = 300,
    max_iterations: int = 5,
    tolerance: float = 0.1,
    model=None,
    lambda_approx: float = 0.1,
) -> list[dict]:
    """
    Generate k demographically representative individuals using a two-phase LLM+IPF algorithm.

    Phase 1 — Dependency inference (ensemble):
        Calls an LLM 10 times with marginal + joint data.  Each call returns
        ranked dependency subsets and approximate joint distributions.  Results
        are aggregated by frequency and rank to select the top-5 subsets.

    Phase 2 — Iterative constrained generation:
        Iteratively computes marginal and joint distribution gaps (delta), asks
        the LLM for targeted proposals, executes the proposals (filling
        unconstrained attributes with residual-weighted sampling), then repeats
        until convergence (max relative error < tolerance) or max_iterations.

    A correction phase follows, replacing the most over-represented agents.
    Any remaining shortfall is filled with weighted-random sampling.

    Args:
        row:            pandas Series / dict with CBG (or city-aggregated) population data.
        k:              Number of agents to generate (default 300).
        max_iterations: Maximum iterations for Phase 2 and the correction phase (default 5).
        tolerance:      Convergence threshold for max relative error (default 0.10 = 10%).
        model:          Pre-initialised LLM model (BaseModel).  If None, creates
                        Gemini-2.5-Pro automatically using credentials/gemini_api_key.json.
        lambda_approx:  Weight applied to LLM-estimated approximate joint deltas (default 0.1).

    Returns:
        List of k dicts with keys: race, education, household_income, gender, age, occupation.
        Occupations are mapped to 7 major categories via OCCUPATION_TO_MAJOR_CATEGORY.
    """
    # ── model initialisation ──────────────────────────────────────────────────
    if model is None:
        try:
            from models import create_model
            model = create_model(
                "gemini",
                model_name="gemini-2.5-pro",
                temperature=1.0,
                max_tokens=16384,
                enable_thinking=True,
            )
        except Exception as exc:
            print(f"[LLM+IPF] Cannot create LLM model ({exc}); falling back to random sampling.")
            return select_random_individuals_from_cbg(row, k=k)

    # ── extract distributions ──────────────────────────────────────────────────
    marginals = _extract_all_marginals(row)
    joints    = _extract_all_joints(row)

    # ── load prompt templates ──────────────────────────────────────────────────
    try:
        phase1_tpl = (PROMPTS_DIR / "LLM+IPF_phase1.txt").read_text(encoding="utf-8")
        phase2_tpl = (PROMPTS_DIR / "LLM+IPF_phase2.txt").read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        print(f"[LLM+IPF] Prompt file missing ({exc}); falling back to random sampling.")
        return select_random_individuals_from_cbg(row, k=k)

    # ==========================================================================
    # PHASE 1 — Variable Dependency Structure Inference (10-call ensemble)
    # ==========================================================================
    phase1_prompt = phase1_tpl.format(
        marginals=_fmt_marginals_text(marginals),
        joints=_fmt_joints_text(joints),
    )

    n_phase1_calls = 10
    print(f"[LLM+IPF] Phase 1: inferring dependency structure "
          f"({n_phase1_calls} ensemble calls) …")

    all_parsed = []
    for call_idx in range(n_phase1_calls):
        print(f"[LLM+IPF]   call {call_idx + 1}/{n_phase1_calls} …")
        all_parsed.append(_parse_llm_json(model.call(phase1_prompt)))

    # ── aggregate subset rankings ─────────────────────────────────────────────
    subset_scores = {}
    for parsed in all_parsed:
        if not parsed or not isinstance(parsed.get("dependency_subsets"), list):
            continue
        for rank_idx, subset in enumerate(parsed["dependency_subsets"]):
            valid = [v for v in subset if v in _LLM_IPF_VARIABLES]
            if len(valid) < 2:
                continue
            key = frozenset(valid)
            if key not in subset_scores:
                subset_scores[key] = {"count": 0, "rank_sum": 0.0, "example": valid}
            subset_scores[key]["count"]    += 1
            subset_scores[key]["rank_sum"] += rank_idx + 1

    ranked = []
    for key, info in subset_scores.items():
        avg_rank = info["rank_sum"] / info["count"]
        score    = info["count"] * (6 - avg_rank)
        ranked.append((score, info["count"], avg_rank, info["example"], key))
    ranked.sort(reverse=True)

    dependency_subsets = []
    print("[LLM+IPF]   aggregated ranking:")
    for score, count, avg_rank, example, _key in ranked[:5]:
        dependency_subsets.append(example)
        print(f"[LLM+IPF]     {example}: "
              f"appeared {count}/{n_phase1_calls}, "
              f"avg rank {avg_rank:.1f}, score {score:.1f}")

    if not dependency_subsets:
        dependency_subsets = [
            ["gender", "age"],
            ["gender", "occupation"],
            ["education", "household_income"],
            ["race", "household_income"],
            ["education", "occupation"],
        ]
        print(f"[LLM+IPF]   -> using default top-5 subsets: {dependency_subsets}")
    else:
        print(f"[LLM+IPF]   -> final top-5 subsets: {dependency_subsets}")

    # ── aggregate approximate joints ──────────────────────────────────────────
    selected_keys    = {frozenset(s) for s in dependency_subsets}
    approx_collections = {}

    for parsed in all_parsed:
        if not parsed or not isinstance(parsed.get("approximate_joints"), dict):
            continue
        for jname, jdata in parsed["approximate_joints"].items():
            if not isinstance(jdata, dict):
                continue
            dist = jdata.get("distribution")
            if not isinstance(dist, dict) or not dist:
                continue
            var_names  = jname.split("_x_")
            valid_vars = [v for v in var_names if v in _LLM_IPF_VARIABLES]
            if len(valid_vars) < 2:
                continue
            key = frozenset(valid_vars)
            if key not in selected_keys:
                continue
            if key not in approx_collections:
                approx_collections[key] = {"variables": valid_vars, "distributions": [],
                                            "confidences": []}
            approx_collections[key]["distributions"].append(dist)
            approx_collections[key]["confidences"].append(jdata.get("confidence", "low"))

    approx_joints_raw = {}
    for key, info in approx_collections.items():
        all_cats = set()
        for d in info["distributions"]:
            all_cats.update(d.keys())
        avg_dist = {cat: sum(d.get(cat, 0) for d in info["distributions"]) / len(info["distributions"])
                    for cat in all_cats}
        conf_counts = {}
        for c in info["confidences"]:
            conf_counts[c] = conf_counts.get(c, 0) + 1
        best_conf = max(conf_counts, key=conf_counts.get)
        approx_joints_raw[key] = {
            "variables": info["variables"],
            "distribution": avg_dist,
            "confidence": best_conf,
        }
        print(f"[LLM+IPF]   -> approx joint {info['variables']}: "
              f"averaged from {len(info['distributions'])} calls, confidence={best_conf}")

    # ── build target statistics ───────────────────────────────────────────────
    target = {var: _scale_dist(marginals[var], k) for var in _LLM_IPF_VARIABLES}

    available_joint_keys = {
        frozenset(["gender", "age"]):        "gender_x_age",
        frozenset(["gender", "occupation"]): "gender_x_occupation",
    }
    real_joint_targets  = {}
    real_joint_vars     = {}
    approx_joint_targets = {}
    approx_joint_vars    = {}

    for subset in dependency_subsets:
        key = frozenset(subset)
        if key in available_joint_keys:
            jname = available_joint_keys[key]
            if jname in joints:
                real_joint_targets[jname] = _scale_dist(joints[jname], k)
                real_joint_vars[jname]    = _REAL_JOINT_VAR_ORDER[jname]
        elif key in approx_joints_raw:
            info  = approx_joints_raw[key]
            jname = "_x_".join(info["variables"])
            approx_joint_targets[jname] = _scale_dist(info["distribution"], k)
            approx_joint_vars[jname]    = info["variables"]

    if real_joint_targets:
        print(f"[LLM+IPF]   -> real joint targets: {list(real_joint_targets.keys())}")
    if approx_joint_targets:
        print(f"[LLM+IPF]   -> approx joint targets (λ={lambda_approx}): "
              f"{list(approx_joint_targets.keys())}")

    # ==========================================================================
    # PHASE 2 — Iterative Constrained Generation
    # ==========================================================================
    individuals: list[dict] = []

    def _phase2_prompt(cur_inds):
        cur   = _count_individual_marginals(cur_inds)
        delta = _compute_llm_ipf_delta(target, cur)
        rjd   = {jn: _compute_joint_delta(jt, _count_joint_from_individuals(cur_inds, real_joint_vars[jn]))
                 for jn, jt in real_joint_targets.items()}
        ajd   = {jn: _compute_joint_delta(jt, _count_joint_from_individuals(cur_inds, approx_joint_vars[jn]))
                 for jn, jt in approx_joint_targets.items()}
        return (
            phase2_tpl.format(
                k              = k,
                target_stats   = _fmt_counts_text(target),
                current_count  = len(cur_inds),
                current_stats  = _fmt_counts_text(cur),
                delta          = _fmt_delta_text(delta, target),
                joint_delta_info = _fmt_joint_delta_text(rjd, real_joint_targets,
                                                         ajd, approx_joint_targets,
                                                         lambda_approx),
                remaining      = k - len(cur_inds),
            ),
            cur, delta, rjd, ajd,
        )

    def _execute_proposals(proposals, cur_inds, remaining):
        total_proposed = sum(p["count"] for p in proposals)
        if total_proposed > remaining:
            factor = remaining / total_proposed
            for p in proposals:
                p["count"] = max(1, round(p["count"] * factor))
        for proposal in proposals:
            for _ in range(proposal["count"]):
                if len(cur_inds) >= k:
                    break
                ind        = dict(proposal["constraints"])
                live_counts = _count_individual_marginals(cur_inds)
                for var in _LLM_IPF_VARIABLES:
                    if var not in ind:
                        ind[var] = _residual_sample(var, target, live_counts)
                cur_inds.append(ind)
            if len(cur_inds) >= k:
                break

    def _parse_proposals(parsed2, cur_delta, remaining):
        proposals = []
        if parsed2 and isinstance(parsed2.get("proposals"), list):
            for p in parsed2["proposals"]:
                try:
                    cnt = int(p.get("count", 0))
                except (TypeError, ValueError):
                    continue
                raw_c   = p.get("constraints") or {}
                valid_c = {v: val for v, val in raw_c.items()
                           if v in target and val in target[v]}
                if cnt > 0:
                    proposals.append({"count": cnt, "constraints": valid_c})
        if not proposals:
            print("[LLM+IPF]   -> no usable proposals; generating from delta directly")
            proposals = _fallback_proposals(cur_delta, remaining)
        return proposals

    for it in range(max_iterations):
        prompt2, cur, delta, rjd, ajd = _phase2_prompt(individuals)
        marg_err  = _max_relative_error(delta, target)
        joint_err = _max_joint_relative_error(rjd, real_joint_targets)
        max_err   = max(marg_err, joint_err)

        print(f"[LLM+IPF] Iteration {it + 1}/{max_iterations}: "
              f"{len(individuals)}/{k} agents, "
              f"max relative error = {max_err:.3f} "
              f"(marginal={marg_err:.3f}, joint={joint_err:.3f})")

        if len(individuals) >= k:
            break
        if max_err <= tolerance and len(individuals) > 0:
            print("[LLM+IPF]   -> converged")
            break

        print("[LLM+IPF]   -> Phase 2: requesting proposals …")
        parsed2   = _parse_llm_json(model.call(prompt2))
        proposals = _parse_proposals(parsed2, delta, k - len(individuals))
        _execute_proposals(proposals, individuals, k - len(individuals))

    # ==========================================================================
    # Correction phase — replace most over-represented agents
    # ==========================================================================
    for corr_it in range(max_iterations):
        if len(individuals) < k:
            break

        prompt2, cur, delta, rjd, ajd = _phase2_prompt(individuals)
        marg_err  = _max_relative_error(delta, target)
        joint_err = _max_joint_relative_error(rjd, real_joint_targets)
        max_err   = max(marg_err, joint_err)

        print(f"[LLM+IPF] Correction {corr_it + 1}/{max_iterations}: "
              f"max relative error = {max_err:.3f} "
              f"(marginal={marg_err:.3f}, joint={joint_err:.3f})")

        if max_err <= tolerance:
            print("[LLM+IPF]   -> converged")
            break

        # Identify and remove the most over-represented agents (up to 25% of k)
        max_pos_delta = max(
            (dv for d in delta.values() for dv in d.values() if dv > 0),
            default=0,
        )
        n_replace = min(k // 4, max(1, round(max_pos_delta)))

        agent_scores = [
            (sum(delta.get(var, {}).get(ind.get(var), 0)
                 for var in _LLM_IPF_VARIABLES
                 if delta.get(var, {}).get(ind.get(var), 0) < -0.5),
             idx)
            for idx, ind in enumerate(individuals)
        ]
        agent_scores.sort()
        remove_indices = {idx for _, idx in agent_scores[:n_replace]}
        individuals    = [ind for i, ind in enumerate(individuals) if i not in remove_indices]

        print(f"[LLM+IPF]   -> replacing {n_replace} over-represented agents")

        remaining = k - len(individuals)
        prompt2, cur, delta, rjd, ajd = _phase2_prompt(individuals)
        print("[LLM+IPF]   -> requesting correction proposals …")
        parsed2   = _parse_llm_json(model.call(prompt2))
        proposals = _parse_proposals(parsed2, delta, remaining)
        _execute_proposals(proposals, individuals, remaining)

    # ── pad shortfall with random sampling ────────────────────────────────────
    if len(individuals) < k:
        shortfall = k - len(individuals)
        print(f"[LLM+IPF] Padding {shortfall} agent(s) with weighted random sampling")
        individuals.extend(select_random_individuals_from_cbg(row, k=shortfall))

    return _map_occupations_to_major(individuals[:k])
