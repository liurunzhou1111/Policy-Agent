"""
City-level accuracy pipeline — modified version.

Reads agent predictions from:
  results/{city}/llm_ipf_agents_result_{model}_modified_n300.jsonl

Computes city-level metrics (KL divergence, Wasserstein, median diff) and
saves comparison plots. No per-CBG processing.

Output:
  results/{city}/{folder}/num_result/city_level_agents/
    {poi}_city_metrics.csv
    {poi}_city_comparison.png
"""

from __future__ import annotations

import errno
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import entropy as _scipy_entropy

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
from utils import _normalize_cbg_geoid_str

# =============================================================================
# Load config
# =============================================================================

# Repo root is two levels above this file: code/ → Policy-Agent/
REPO_ROOT      = Path(__file__).parent.parent
ADVAN_DATA_DIR = REPO_ROOT / "data" / "city_data"
RESULTS_DIR    = REPO_ROOT / "results"
CONFIG_PATH    = REPO_ROOT / "config.json"

with open(CONFIG_PATH, encoding="utf-8") as _f:
    _CFG = json.load(_f)

CITY_LIST           = _CFG["cities"]
MODEL_LIST          = _CFG["models"]["evaluation_models"]
AGENT_FILE_TEMPLATE = _CFG["simulation"]["agent_file_template"]
BASELINE_DATES      = _CFG["simulation"]["baseline_dates"]
SIMULATION_DATES    = _CFG["simulation"]["simulation_dates"]
POI_PRED_TO_CSV     = _CFG["poi"]["pred_to_csv"]
ALL_POI_TYPES       = list(POI_PRED_TO_CSV.keys())

CITY_YLIM             = tuple(_CFG["evaluation"]["city_ylim"])
CITY_YLIM_OVERRIDES   = {k: tuple(v) for k, v in _CFG["evaluation"]["city_ylim_overrides"].items()}
READ_MAX_RETRIES      = _CFG["evaluation"]["read_max_retries"]
READ_RETRY_SLEEP      = _CFG["evaluation"]["read_retry_sleep"]

CITY_DISPLAY_NAMES = {c["name"]: c["display_name"] for c in CITY_LIST}

# ── Box-plot style constants (matching reference draw_city_simulation_box_plot.py) ──
POI_FULL_LABELS       = ["Restaurants & Bars", "Retail", "Arts & Entertainment"]
DATE_GROUP_GAP        = 4.2
POI_POSITION_OFFSETS  = [0.0, 0.95, 1.9]
BOX_HALF_SPACING      = 0.22
BOX_WIDTH             = 0.38
COMBINED_POI_GT_FACE  = ["#7976A2", "#4292C6", "#86B5A1"]   # muted purple / blue / sage
BOX_EDGE              = "#333333"
BOX_PATCH_ALPHA       = 0.82
MEDIAN_LINE_COLOR     = "#E65100"   # orange
WHISKER_CAP_COLOR     = "#000000"   # black
BOX_LINEWIDTH         = 0.9
WHISKER_LINEWIDTH     = 0.9
MEDIAN_LINEWIDTH      = 1.15
DATE_TICK_FONT_SIZE   = 14
LEGEND_FONT_SIZE      = 14


# =============================================================================
# JSONL loading
# =============================================================================

def _load_jsonl(path: Path) -> list[dict]:
    """Load JSONL with retry on transient OS errors."""
    for attempt in range(1, READ_MAX_RETRIES + 1):
        try:
            records = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return records
        except OSError as e:
            if e.errno == errno.ECANCELED and attempt < READ_MAX_RETRIES:
                time.sleep(READ_RETRY_SLEEP * attempt)
                continue
            print(f"[WARN] Cannot read {path}: {e}")
            return []
    return []


# =============================================================================
# Ground truth
# =============================================================================

def load_cbg_true_changes(
    patterns_df: pd.DataFrame, cbg_id: str, poi_csv: str
) -> dict | None:
    """Return {date: change_ratio} = visits/baseline_mean - 1 for one CBG/POI.

    Only dates in SIMULATION_DATES are included.
    """
    sub = patterns_df[
        (patterns_df["CBG"] == _normalize_cbg_geoid_str(cbg_id))
        & (patterns_df["POI_CATEGORY"] == poi_csv)
    ]
    if sub.empty:
        return None
    baseline = sub[sub["DATE_STR"].isin(BASELINE_DATES)]["VISITS"].mean()
    if baseline <= 0:
        return None
    return {
        row["DATE_STR"]: float(row["VISITS"] / baseline - 1.0)
        for _, row in sub.iterrows()
        if row["DATE_STR"] in SIMULATION_DATES
    } or None


def build_city_true_distribution(
    patterns_path: Path, cbg_prefix: str
) -> dict[str, dict[str, list[float]]]:
    """Aggregate per-CBG change ratios into city-wide per-POI per-date lists."""
    df = pd.read_csv(patterns_path)
    df["CBG"] = df["CBG"].astype(str).map(_normalize_cbg_geoid_str)
    df["DATE_RANGE_START"] = pd.to_datetime(df["DATE_RANGE_START"])
    df["DATE_STR"] = df["DATE_RANGE_START"].dt.strftime("%Y-%m-%d")
    df = df[df["CBG"].str.startswith(cbg_prefix)]

    city_true: dict[str, dict[str, list[float]]] = {
        poi: defaultdict(list) for poi in ALL_POI_TYPES
    }
    for cbg_id in sorted(df["CBG"].unique()):
        for poi_key in ALL_POI_TYPES:
            changes = load_cbg_true_changes(df, cbg_id, POI_PRED_TO_CSV[poi_key])
            if changes:
                for date, val in changes.items():
                    city_true[poi_key][date].append(val)

    print(f"  [INFO] {df['CBG'].nunique()} CBGs loaded from {patterns_path.name}")
    return {k: dict(v) for k, v in city_true.items()}


# =============================================================================
# Agent predictions
# =============================================================================

def load_city_agent_predictions(
    agents_path: Path, poi_key: str
) -> dict[str, list[float]]:
    """Return {date: [predicted_changes]} for one POI.

    Only dates in SIMULATION_DATES are included.
    """
    by_date: dict[str, list[float]] = defaultdict(list)
    for rec in _load_jsonl(agents_path):
        date = rec.get("simulation_date", "")
        if date not in SIMULATION_DATES:
            continue
        changes = rec.get("predicted_changes") or {}
        val = changes.get(poi_key)
        if val is None:
            continue
        try:
            by_date[date].append(float(val))
        except (TypeError, ValueError):
            pass
    return dict(by_date)


# =============================================================================
# Metrics
# =============================================================================

def _js_divergence(true_arr: np.ndarray, pred_arr: np.ndarray, n_bins: int = 50) -> float:
    """Jensen-Shannon divergence via histogram with Laplace smoothing.

    Matches the reference implementation in KL_divergence_compare_modified.py:
      - Shared bin edges across both distributions
      - Laplace smoothing eps=1e-10
      - JSD = 0.5*KL(P||M) + 0.5*KL(Q||M) using scipy.stats.entropy (natural log, nats)
      - Bounded in [0, ln(2)] nats
    """
    if len(true_arr) < 2 or len(pred_arr) < 2:
        return np.nan
    all_v = np.concatenate([true_arr, pred_arr])
    bins  = np.linspace(all_v.min(), all_v.max(), n_bins + 1)
    eps   = 1e-10
    p_h, _ = np.histogram(true_arr, bins=bins)
    q_h, _ = np.histogram(pred_arr, bins=bins)
    p = (p_h + eps) / (p_h + eps).sum()
    q = (q_h + eps) / (q_h + eps).sum()
    m = 0.5 * (p + q)
    return float(0.5 * _scipy_entropy(p, m) + 0.5 * _scipy_entropy(q, m))


def compute_city_metrics(
    true_by_date: dict[str, list[float]],
    pred_by_date: dict[str, list[float]],
) -> list[dict]:
    """Per-date JS divergence and median difference."""
    rows = []
    for date in sorted(set(true_by_date) & set(pred_by_date)):
        t = np.array(true_by_date[date], dtype=float)
        p = np.array(pred_by_date[date], dtype=float)
        rows.append({
            "date":           date,
            "js_divergence":  _js_divergence(t, p),
            "median_diff":    float(np.median(p) - np.median(t)),
        })
    return rows


# =============================================================================
# Plotting — combined 3-POI boxplot
# =============================================================================

def _box_style_kwargs() -> dict:
    """Return fresh boxplot style kwargs (new dicts each call to avoid mutation)."""
    return {
        "medianprops":  {"color": MEDIAN_LINE_COLOR,  "linewidth": MEDIAN_LINEWIDTH},
        "boxprops":     {"linewidth": BOX_LINEWIDTH,   "edgecolor": BOX_EDGE},
        "whiskerprops": {"linewidth": WHISKER_LINEWIDTH, "color": WHISKER_CAP_COLOR},
        "capprops":     {"linewidth": WHISKER_LINEWIDTH, "color": WHISKER_CAP_COLOR},
    }


def plot_three_poi_combined_box(
    out_path: Path,
    city_true: dict[str, dict[str, list[float]]],
    city_pred: dict[str, dict[str, list[float]]],
    city_name: str,
    ylim: tuple | None = None,
) -> None:
    """
    One figure showing all 3 POIs for every simulation date.

    Layout: dates on the x-axis separated by DATE_GROUP_GAP; within each date
    group, three POI slots spaced by POI_POSITION_OFFSETS; within each slot
    ground-truth (solid fill) and predicted (same colour + hatch) boxes sit
    ±BOX_HALF_SPACING apart.
    """
    # Collect dates that have both GT and prediction for at least one POI
    dates: set[str] = set()
    for poi in ALL_POI_TYPES:
        for d in set(city_true.get(poi, {})) & set(city_pred.get(poi, {})):
            if city_true[poi][d] and city_pred[poi][d]:
                dates.add(d)
    common_dates = sorted(dates)
    if not common_dates:
        print(f"  [WARN] No overlapping dates for combined box plot ({city_name}).")
        return

    pos_gt, pos_pred   = [], []
    data_gt, data_pred = [], []
    gt_face            = []

    for di, d in enumerate(common_dates):
        base = di * DATE_GROUP_GAP
        for pi, poi in enumerate(ALL_POI_TYPES):
            t = city_true.get(poi, {}).get(d, [])
            p = city_pred.get(poi, {}).get(d, [])
            if not t or not p:
                continue
            c = base + POI_POSITION_OFFSETS[pi]
            pos_gt.append(c - BOX_HALF_SPACING)
            pos_pred.append(c + BOX_HALF_SPACING)
            data_gt.append(t)
            data_pred.append(p)
            gt_face.append(COMBINED_POI_GT_FACE[pi])

    if not pos_gt:
        print(f"  [WARN] No box plot data for combined plot ({city_name}).")
        return

    fig, ax = plt.subplots(figsize=(max(14, len(common_dates) * DATE_GROUP_GAP * 0.55), 7.5))

    # Ground-truth boxes (solid fill)
    bp_gt = ax.boxplot(data_gt, positions=pos_gt, widths=BOX_WIDTH,
                       patch_artist=True, showfliers=False, **_box_style_kwargs())
    for patch, fc in zip(bp_gt["boxes"], gt_face):
        patch.set_facecolor(fc)
        patch.set_edgecolor(BOX_EDGE)
        patch.set_linewidth(BOX_LINEWIDTH)
        patch.set_alpha(BOX_PATCH_ALPHA)

    # Predicted boxes (same colour + hatch)
    bp_pred = ax.boxplot(data_pred, positions=pos_pred, widths=BOX_WIDTH,
                         patch_artist=True, showfliers=False, **_box_style_kwargs())
    for patch, fc in zip(bp_pred["boxes"], gt_face):
        patch.set_facecolor(fc)
        patch.set_edgecolor(BOX_EDGE)
        patch.set_linewidth(BOX_LINEWIDTH)
        patch.set_alpha(BOX_PATCH_ALPHA)
        patch.set_hatch("///")

    ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    if ylim is not None:
        ax.set_ylim(ylim)

    mid_offset = (POI_POSITION_OFFSETS[0] + POI_POSITION_OFFSETS[-1]) / 2
    ax.set_xticks([di * DATE_GROUP_GAP + mid_offset for di in range(len(common_dates))])
    ax.set_xticklabels(
        [d[5:] for d in common_dates],   # MM-DD only
        rotation=45, ha="right", fontsize=DATE_TICK_FONT_SIZE,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Relative Mobility Changes", fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title(
        f"{CITY_DISPLAY_NAMES.get(city_name, city_name)} city-level distribution",
        fontsize=12, fontweight="bold",
    )

    poi_handles = [
        Patch(facecolor=COMBINED_POI_GT_FACE[i], edgecolor=BOX_EDGE,
              linewidth=0.6, alpha=BOX_PATCH_ALPHA, label=POI_FULL_LABELS[i])
        for i in range(len(ALL_POI_TYPES))
    ]
    gt_patch  = Patch(facecolor="0.85", edgecolor=BOX_EDGE, linewidth=0.6,
                      alpha=BOX_PATCH_ALPHA, label="Ground truth")
    sim_patch = Patch(facecolor="0.85", edgecolor=BOX_EDGE, linewidth=0.6,
                      alpha=BOX_PATCH_ALPHA, hatch="///", label="Simulated")
    ax.legend(handles=poi_handles + [gt_patch, sim_patch],
              loc="best", ncol=2, fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [INFO] Saved {out_path.name}")


def emit_poi_csv(
    out_dir: Path,
    true_by_date: dict,
    pred_by_date: dict,
    poi_key: str,
) -> None:
    """Write JS-divergence + median-diff CSV for one POI."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = compute_city_metrics(true_by_date, pred_by_date)
    if rows:
        pd.DataFrame(rows).to_csv(
            out_dir / f"{poi_key}_city_metrics.csv", index=False, encoding="utf-8"
        )


# =============================================================================
# Pipeline
# =============================================================================

def process_city_model(city_cfg: dict, model_spec: str) -> None:
    city_name   = city_cfg["name"]
    cbg_prefix  = city_cfg["cbg_prefix"]
    model_names = model_spec.split("+") if "+" in model_spec else [model_spec]

    print(f"\n[INFO] {city_name} | {model_spec}")

    # Locate agent files under results/{city}/{model}/
    agent_paths: list[tuple[str, Path]] = []
    for m in model_names:
        p = RESULTS_DIR / city_name / m / AGENT_FILE_TEMPLATE.format(model=m)
        if not p.exists():
            print(f"  [WARN] Missing: {p}")
            return
        agent_paths.append((m, p))

    # Locate ground-truth patterns file
    patterns_path = ADVAN_DATA_DIR / city_name / f"{city_name}_patterns_updated.csv"
    if not patterns_path.exists():
        print(f"  [WARN] Missing: {patterns_path}")
        return

    city_true = build_city_true_distribution(patterns_path, cbg_prefix)

    # Union predictions from all models per date (filtered to SIMULATION_DATES)
    city_pred: dict[str, dict[str, list[float]]] = {
        poi: defaultdict(list) for poi in ALL_POI_TYPES
    }
    for m, ap in agent_paths:
        print(f"  [INFO] Loading {ap.name}")
        for poi_key in ALL_POI_TYPES:
            for date, vals in load_city_agent_predictions(ap, poi_key).items():
                city_pred[poi_key][date].extend(vals)
    city_pred = {k: dict(v) for k, v in city_pred.items()}

    # Outputs go into results/{city}/{model_spec}/ (flat, no deep nesting)
    out_base = RESULTS_DIR / city_name / model_spec

    # Write per-POI metrics CSVs (JS divergence + median diff)
    for poi_key in ALL_POI_TYPES:
        true_d = city_true.get(poi_key, {})
        pred_d = city_pred.get(poi_key, {})
        if not true_d or not pred_d:
            print(f"  [WARN] Skipping {poi_key} (no data)")
            continue
        emit_poi_csv(out_base, true_d, pred_d, poi_key)

    # Write one combined 3-POI boxplot
    ylim = CITY_YLIM_OVERRIDES.get(city_name, CITY_YLIM)
    plot_three_poi_combined_box(
        out_base / "all_pois_city_comparison_box.png",
        city_true, city_pred, city_name, ylim=ylim,
    )

    print(f"  [INFO] Done -> {out_base}")


def main():
    for city in CITY_LIST:
        for model_spec in MODEL_LIST:
            process_city_model(city, model_spec)
    print("\n[INFO] All city-level (modified) outputs written.")


if __name__ == "__main__":
    main()
