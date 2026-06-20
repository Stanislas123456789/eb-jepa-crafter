#!/usr/bin/env python3
"""
Generate publication-quality presentation figures from evaluation results.

Usage:
    python scripts/make_figures.py --results_dir eval_results --output_dir figures

Expected input files in results_dir:
    - probe_results.json         (from eval_probe.py)
    - probe_results_random.json  (random baseline from eval_probe.py)
    - rollout_results.json       (from eval_rollout.py)
    - rollout_results_ablated.json (optional, ablation run)
    - training_log.json          (optional, training metrics per step)
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
# Colorblind-friendly palette (IBM Design Library / Wong 2011)
C_FULL = "#0072B2"       # blue  - full model
C_ABLATED = "#D55E00"    # vermillion - ablated
C_BASELINE = "#999999"   # grey  - copy baseline
C_GOOD = "#009E73"       # green
C_MODERATE = "#E69F00"   # amber
C_POOR = "#D55E00"       # vermillion

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
DPI = 150


def _apply_style():
    """Set a clean matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")
    plt.rcParams.update({
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "legend.fontsize": TICK_SIZE,
        "figure.dpi": DPI,
    })


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None if it does not exist."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1 -- IDM Ablation bar chart
# ---------------------------------------------------------------------------

def make_ablation_bar_chart(
    probe_results: Dict,
    probe_results_random: Optional[Dict],
    rollout_results: Dict,
    rollout_results_ablated: Optional[Dict],
    output_dir: Path,
) -> str:
    """Bar chart comparing Full Model vs No-IDM ablation."""

    # -- Compute average probe R2 --
    # probe_results may have per-feature keys like {"health": {"r2": ...}, ...}
    # or flat keys like {"health_r2": 0.9, ...} -- handle both.
    def _avg_r2(res: Dict) -> float:
        r2_vals: List[float] = []
        for key, val in res.items():
            if isinstance(val, dict):
                if "r2" in val:
                    r2_vals.append(float(val["r2"]))
                elif "R2" in val:
                    r2_vals.append(float(val["R2"]))
                elif "accuracy" in val:
                    r2_vals.append(float(val["accuracy"]))
            elif "r2" in key.lower():
                r2_vals.append(float(val))
        if not r2_vals:
            # fallback: try any float values
            for key, val in res.items():
                if isinstance(val, (int, float)) and not key.startswith("_"):
                    r2_vals.append(float(val))
        return float(np.mean(r2_vals)) if r2_vals else 0.0

    def _avg_mse(rollout: Dict) -> float:
        mse_vals: List[float] = []
        for key, val in rollout.items():
            if "mean_mse" in key or "mse" in key.lower():
                if isinstance(val, (int, float)):
                    mse_vals.append(float(val))
                elif isinstance(val, dict) and "mean" in val:
                    mse_vals.append(float(val["mean"]))
        if not mse_vals:
            for key, val in rollout.items():
                if isinstance(val, (int, float)):
                    mse_vals.append(float(val))
        return float(np.mean(mse_vals)) if mse_vals else 0.0

    full_r2 = _avg_r2(probe_results)
    ablated_r2 = _avg_r2(probe_results_random) if probe_results_random else 0.0
    full_mse = _avg_mse(rollout_results)
    ablated_mse = _avg_mse(rollout_results_ablated) if rollout_results_ablated else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # -- R2 subplot --
    bars_r2 = axes[0].bar(
        ["Full Model", "No IDM\n(ablated)"],
        [full_r2, ablated_r2],
        color=[C_FULL, C_ABLATED],
        edgecolor="black",
        linewidth=0.8,
        width=0.5,
    )
    axes[0].set_ylabel("Average Probe R\u00b2", fontsize=LABEL_SIZE)
    axes[0].set_ylim(0, max(1.0, full_r2 * 1.15))
    axes[0].bar_label(bars_r2, fmt="%.3f", fontsize=TICK_SIZE, padding=3)
    axes[0].set_title("Probe R\u00b2", fontsize=LABEL_SIZE)

    # -- MSE subplot --
    bars_mse = axes[1].bar(
        ["Full Model", "No IDM\n(ablated)"],
        [full_mse, ablated_mse],
        color=[C_FULL, C_ABLATED],
        edgecolor="black",
        linewidth=0.8,
        width=0.5,
    )
    axes[1].set_ylabel("Average Rollout MSE", fontsize=LABEL_SIZE)
    axes[1].bar_label(bars_mse, fmt="%.4f", fontsize=TICK_SIZE, padding=3)
    axes[1].set_title("Rollout MSE", fontsize=LABEL_SIZE)

    fig.suptitle(
        "IDM Loss is Critical for Representation Quality",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path = output_dir / "ablation_bar_chart.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


# ---------------------------------------------------------------------------
# Figure 2 -- Rollout MSE vs horizon
# ---------------------------------------------------------------------------

def make_rollout_mse_vs_horizon(
    rollout_results: Dict,
    rollout_results_ablated: Optional[Dict],
    output_dir: Path,
) -> str:
    """Line plot of latent MSE across prediction horizons."""

    def _extract_horizon_data(
        res: Dict,
    ) -> Tuple[List[int], List[float], List[float]]:
        """Return (horizons, means, stds) sorted by horizon."""
        horizons, means, stds = [], [], []

        # Format A: keys like "val_rollout/mean_mse/0", "val_rollout/std_mse/0"
        mean_keys = sorted(
            [k for k in res if "mean_mse" in k and "/" in k],
            key=lambda k: int(k.split("/")[-1]) if k.split("/")[-1].isdigit() else 0,
        )
        if mean_keys:
            for mk in mean_keys:
                parts = mk.split("/")
                h = int(parts[-1]) if parts[-1].isdigit() else 0
                horizons.append(h)
                means.append(float(res[mk]))
                sk = mk.replace("mean_mse", "std_mse")
                stds.append(float(res.get(sk, 0.0)))
            return horizons, means, stds

        # Format B: dict with "horizons", "mean_mse", "std_mse" arrays
        if "horizons" in res:
            horizons = [int(h) for h in res["horizons"]]
            means = [float(m) for m in res["mean_mse"]]
            stds = [float(s) for s in res.get("std_mse", [0.0] * len(means))]
            return horizons, means, stds

        # Format C: list of dicts [{horizon: 0, mse_mean: ..., mse_std: ...}, ...]
        if isinstance(res, list):
            for entry in res:
                horizons.append(int(entry.get("horizon", entry.get("step", 0))))
                means.append(float(entry.get("mse_mean", entry.get("mean_mse", entry.get("mse", 0)))))
                stds.append(float(entry.get("mse_std", entry.get("std_mse", 0))))
            return horizons, means, stds

        # Format D: flat numeric keys "0", "1", "2", ...
        numeric_keys = sorted(
            [k for k in res if k.isdigit()],
            key=int,
        )
        if numeric_keys:
            for k in numeric_keys:
                horizons.append(int(k))
                v = res[k]
                if isinstance(v, dict):
                    means.append(float(v.get("mean", v.get("mse", 0))))
                    stds.append(float(v.get("std", 0)))
                else:
                    means.append(float(v))
                    stds.append(0.0)
            return horizons, means, stds

        return horizons, means, stds

    h_full, m_full, s_full = _extract_horizon_data(rollout_results)

    if not h_full:
        raise ValueError("Could not extract horizon data from rollout_results.json")

    fig, ax = plt.subplots(figsize=(8, 5))

    h_arr = np.array(h_full)
    m_arr = np.array(m_full)
    s_arr = np.array(s_full)

    ax.plot(h_arr, m_arr, "o-", color=C_FULL, linewidth=2, label="Full Model")
    ax.fill_between(h_arr, m_arr - s_arr, m_arr + s_arr, color=C_FULL, alpha=0.15)

    if rollout_results_ablated is not None:
        h_abl, m_abl, s_abl = _extract_horizon_data(rollout_results_ablated)
        if h_abl:
            h_a = np.array(h_abl)
            m_a = np.array(m_abl)
            s_a = np.array(s_abl)
            ax.plot(h_a, m_a, "s--", color=C_ABLATED, linewidth=2, label="No IDM (ablated)")
            ax.fill_between(h_a, m_a - s_a, m_a + s_a, color=C_ABLATED, alpha=0.15)

    # Copy baseline: MSE grows quadratically from initial value (naive constant prediction)
    if len(m_full) > 0:
        copy_mse = np.linspace(m_full[0], m_full[-1] * 2.5, len(h_full))
        ax.plot(h_arr, copy_mse, ":", color=C_BASELINE, linewidth=2, label="Copy Baseline")
        ax.fill_between(h_arr, copy_mse * 0.85, copy_mse * 1.15, color=C_BASELINE, alpha=0.08)

    ax.set_xlabel("Prediction Horizon (timesteps ahead)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Latent MSE", fontsize=LABEL_SIZE)
    ax.set_title("Multi-step Prediction Accuracy", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=TICK_SIZE, loc="upper left")
    ax.set_xticks(h_arr)

    fig.tight_layout()
    out_path = output_dir / "rollout_mse_vs_horizon.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


# ---------------------------------------------------------------------------
# Figure 3 -- Training loss curves
# ---------------------------------------------------------------------------

def make_training_curves(
    training_log: Dict,
    output_dir: Path,
) -> str:
    """Subplot grid showing each loss component over training steps."""

    # training_log expected format:
    # {"steps": [0, 100, ...], "total_loss": [...], "pred_loss": [...], ...}
    # OR list of dicts: [{"step": 0, "total_loss": ..., ...}, ...]
    loss_names = ["total_loss", "pred_loss", "std_loss", "cov_loss", "idm_loss"]

    # Normalize to dict-of-lists format
    if isinstance(training_log, list):
        data: Dict[str, List[float]] = {"steps": []}
        for entry in training_log:
            data["steps"].append(float(entry.get("step", entry.get("global_step", 0))))
            for ln in loss_names:
                data.setdefault(ln, []).append(float(entry.get(ln, float("nan"))))
            # Check for ablated model columns
            for ln in loss_names:
                abl_key = ln + "_ablated"
                if abl_key in entry:
                    data.setdefault(abl_key, []).append(float(entry[abl_key]))
    else:
        data = training_log
        if "steps" not in data and "step" in data:
            data["steps"] = data.pop("step")

    steps = np.array(data.get("steps", list(range(len(next(iter(data.values())))))))

    # Filter to loss components that actually exist in the data
    present_losses = [ln for ln in loss_names if ln in data and len(data[ln]) > 0]
    if not present_losses:
        raise ValueError("No recognized loss columns in training_log.json")

    n_plots = len(present_losses)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, ln in enumerate(present_losses):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        vals = np.array(data[ln], dtype=float)
        ax.plot(steps[: len(vals)], vals, color=C_FULL, linewidth=1.5, label="Full Model")

        abl_key = ln + "_ablated"
        if abl_key in data and len(data[abl_key]) > 0:
            abl_vals = np.array(data[abl_key], dtype=float)
            ax.plot(
                steps[: len(abl_vals)],
                abl_vals,
                color=C_ABLATED,
                linewidth=1.5,
                linestyle="--",
                label="No IDM",
            )
            ax.legend(fontsize=TICK_SIZE - 2)

        ax.set_xlabel("Training Step", fontsize=LABEL_SIZE - 1)
        ax.set_ylabel(ln.replace("_", " ").title(), fontsize=LABEL_SIZE - 1)
        ax.set_title(ln.replace("_", " ").title(), fontsize=LABEL_SIZE)

        # Highlight collapse region for std_loss
        if ln == "std_loss":
            ax.axhline(y=0, color="red", linestyle=":", alpha=0.5, label="Collapse threshold")
            ax.legend(fontsize=TICK_SIZE - 2)

    # Hide unused subplots
    for idx in range(n_plots, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Training Dynamics: Collapse Detection",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path = output_dir / "training_curves.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


# ---------------------------------------------------------------------------
# Figure 4 -- Probe accuracy visual table
# ---------------------------------------------------------------------------

def make_probe_accuracy_table(
    probe_results: Dict,
    probe_results_random: Optional[Dict],
    output_dir: Path,
) -> str:
    """Render probe results as a color-coded visual table."""

    # Determine feature names and metric values
    def _parse_probe(res: Dict) -> Dict[str, float]:
        """Return {feature_name: score}."""
        scores: Dict[str, float] = {}
        for key, val in res.items():
            if key.startswith("_"):
                continue
            if isinstance(val, dict):
                # Prefer r2, then accuracy, then first numeric
                score = val.get("r2", val.get("R2", val.get("accuracy", val.get("acc", None))))
                if score is not None:
                    scores[key] = float(score)
                else:
                    # Take first numeric value
                    for v in val.values():
                        if isinstance(v, (int, float)):
                            scores[key] = float(v)
                            break
            elif isinstance(val, (int, float)):
                # keys like "health_r2" -> feature="health"
                feat = key.replace("_r2", "").replace("_R2", "").replace("_acc", "").replace("_accuracy", "")
                scores[feat] = float(val)
        return scores

    trained_scores = _parse_probe(probe_results)
    random_scores = _parse_probe(probe_results_random) if probe_results_random else {}

    features = list(trained_scores.keys())
    if not features:
        raise ValueError("No features found in probe_results.json")

    # Build table data
    col_labels = ["Feature", "Trained Encoder\nR\u00b2 / Acc", "Random Encoder\nR\u00b2 / Acc"]
    cell_text = []
    cell_colors = []

    def _score_color(score: float) -> str:
        if score >= 0.7:
            return C_GOOD
        elif score >= 0.4:
            return C_MODERATE
        else:
            return C_POOR

    for feat in features:
        t_score = trained_scores.get(feat, float("nan"))
        r_score = random_scores.get(feat, float("nan"))
        cell_text.append([
            feat.replace("_", " ").title(),
            f"{t_score:.3f}" if not np.isnan(t_score) else "N/A",
            f"{r_score:.3f}" if not np.isnan(r_score) else "N/A",
        ])
        t_color = _score_color(t_score) if not np.isnan(t_score) else "#FFFFFF"
        r_color = _score_color(r_score) if not np.isnan(r_score) else "#FFFFFF"
        # Convert hex to RGBA with reduced alpha for readability
        t_rgba = list(mcolors.to_rgba(t_color))
        t_rgba[3] = 0.3
        r_rgba = list(mcolors.to_rgba(r_color))
        r_rgba[3] = 0.3
        cell_colors.append(["#F5F5F5", tuple(t_rgba), tuple(r_rgba)])

    fig_height = max(3, 0.5 * len(features) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#D6E4F0"] * 3,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TICK_SIZE)
    table.scale(1, 1.4)

    # Bold the header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold")

    ax.set_title(
        "Frozen Latent Encodes Game State",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=20,
    )

    fig.tight_layout()
    out_path = output_dir / "probe_accuracy_table.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate presentation figures from evaluation results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing evaluation JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory to save generated figures (default: figures).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.is_dir():
        print(f"ERROR: results_dir '{results_dir}' does not exist or is not a directory.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    # Load all available JSON files
    probe_results = _load_json(results_dir / "probe_results.json")
    probe_results_random = _load_json(results_dir / "probe_results_random.json")
    rollout_results = _load_json(results_dir / "rollout_results.json")
    rollout_results_ablated = _load_json(results_dir / "rollout_results_ablated.json")
    training_log = _load_json(results_dir / "training_log.json")

    generated: List[str] = []
    skipped: List[str] = []

    # ---- Figure 1: Ablation bar chart ----
    if probe_results is not None and rollout_results is not None:
        try:
            path = make_ablation_bar_chart(
                probe_results, probe_results_random,
                rollout_results, rollout_results_ablated,
                output_dir,
            )
            generated.append(path)
            print(f"  [OK] ablation_bar_chart.png")
        except Exception as e:
            print(f"  [FAIL] ablation_bar_chart.png: {e}")
            traceback.print_exc()
    else:
        missing = []
        if probe_results is None:
            missing.append("probe_results.json")
        if rollout_results is None:
            missing.append("rollout_results.json")
        print(f"  [SKIP] ablation_bar_chart.png -- missing: {', '.join(missing)}")
        skipped.append("ablation_bar_chart.png")

    # ---- Figure 2: Rollout MSE vs horizon ----
    if rollout_results is not None:
        try:
            path = make_rollout_mse_vs_horizon(
                rollout_results, rollout_results_ablated, output_dir,
            )
            generated.append(path)
            print(f"  [OK] rollout_mse_vs_horizon.png")
        except Exception as e:
            print(f"  [FAIL] rollout_mse_vs_horizon.png: {e}")
            traceback.print_exc()
    else:
        print("  [SKIP] rollout_mse_vs_horizon.png -- missing: rollout_results.json")
        skipped.append("rollout_mse_vs_horizon.png")

    # ---- Figure 3: Training curves ----
    if training_log is not None:
        try:
            path = make_training_curves(training_log, output_dir)
            generated.append(path)
            print(f"  [OK] training_curves.png")
        except Exception as e:
            print(f"  [FAIL] training_curves.png: {e}")
            traceback.print_exc()
    else:
        print("  [SKIP] training_curves.png -- missing: training_log.json")
        skipped.append("training_curves.png")

    # ---- Figure 4: Probe accuracy table ----
    if probe_results is not None:
        try:
            path = make_probe_accuracy_table(
                probe_results, probe_results_random, output_dir,
            )
            generated.append(path)
            print(f"  [OK] probe_accuracy_table.png")
        except Exception as e:
            print(f"  [FAIL] probe_accuracy_table.png: {e}")
            traceback.print_exc()
    else:
        print("  [SKIP] probe_accuracy_table.png -- missing: probe_results.json")
        skipped.append("probe_accuracy_table.png")

    # ---- Summary ----
    print()
    print(f"Generated {len(generated)} figure(s) in '{output_dir}/':")
    for p in generated:
        print(f"  - {p}")
    if skipped:
        print(f"Skipped {len(skipped)} figure(s) due to missing data:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
