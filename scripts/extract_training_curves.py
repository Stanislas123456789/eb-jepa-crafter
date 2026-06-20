#!/usr/bin/env python3
"""
Extract training loss curves from SLURM log files and generate comparison figures.

Parses tqdm progress bars from SLURM stderr files to extract per-step losses
(total_loss, reg_loss, pred_loss), and epoch summaries from stdout files.
Generates a training curves figure comparing full model vs ablated model.

Usage:
    # With local log files:
    python scripts/extract_training_curves.py \
        --full_err /tmp/slurm_full.err \
        --ablated_err /tmp/slurm_ablated.err \
        --full_out /tmp/slurm_full.out \
        --ablated_out /tmp/slurm_ablated.out \
        --output_dir eval_results/figures

    # With only stderr files (stdout is optional):
    python scripts/extract_training_curves.py \
        --full_err /tmp/slurm_full.err \
        --ablated_err /tmp/slurm_ablated.err

    # Pull logs from Dalia first:
    ssh dalia 'cat ~/eb_jepa/slurm-76172.err' > /tmp/slurm_full.err
    ssh dalia 'cat ~/eb_jepa/slurm-76171.err' > /tmp/slurm_ablated.err
    ssh dalia 'cat ~/eb_jepa/slurm-76172.out' > /tmp/slurm_full.out
    ssh dalia 'cat ~/eb_jepa/slurm-76171.out' > /tmp/slurm_ablated.out
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style constants (matching make_figures.py)
# ---------------------------------------------------------------------------
C_FULL = "#0072B2"       # blue  - full model
C_ABLATED = "#D55E00"    # vermillion - ablated
C_BASELINE = "#999999"   # grey

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


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------

def parse_tqdm_stderr(filepath: str, sample_every: int = 50) -> Dict[str, Any]:
    """
    Parse tqdm progress bar output from SLURM stderr.

    tqdm lines look like:
        Epoch 0/11:   3%|...| 54/2097 [..., loss=5.8715, reg=4.9342, pred=0.9373]

    Returns dict with:
        steps: list of global step indices
        loss: list of total loss values
        reg: list of reg loss values
        pred: list of pred loss values
        epoch_boundaries: list of global step indices where epochs start
        epochs_total: total number of epochs
        steps_per_epoch: steps per epoch
    """
    # Regex to extract epoch, step, and loss values from tqdm line
    # Matches: Epoch E/N: ...| S/T [..., loss=L, reg=R, pred=P]
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s+\[.*?"
        r"loss=([\d.]+).*?reg=([\d.]+).*?pred=([\d.]+)"
    )

    steps = []
    losses = []
    regs = []
    preds = []
    epoch_boundaries = []
    steps_per_epoch = 0
    epochs_total = 0

    seen_steps = set()  # Track (epoch, step) to deduplicate tqdm overwrites
    last_epoch = -1

    with open(filepath, "rb") as f:
        raw = f.read()

    # tqdm uses \r to overwrite lines -- split on both \r and \n
    lines = raw.decode("utf-8", errors="replace").replace("\r", "\n").split("\n")

    for line in lines:
        m = pattern.search(line)
        if not m:
            continue

        epoch = int(m.group(1))
        epoch_max = int(m.group(2))
        step_in_epoch = int(m.group(3))
        total_steps = int(m.group(4))
        loss_val = float(m.group(5))
        reg_val = float(m.group(6))
        pred_val = float(m.group(7))

        epochs_total = epoch_max + 1
        steps_per_epoch = total_steps

        # Deduplicate: tqdm writes the same step multiple times
        key = (epoch, step_in_epoch)
        if key in seen_steps:
            # Update to latest value for this step (tqdm updates in-place)
            # Find and update the existing entry
            global_step = epoch * total_steps + step_in_epoch
            for i in range(len(steps) - 1, -1, -1):
                if steps[i] == global_step:
                    losses[i] = loss_val
                    regs[i] = reg_val
                    preds[i] = pred_val
                    break
            continue

        seen_steps.add(key)
        global_step = epoch * total_steps + step_in_epoch

        # Track epoch boundaries
        if epoch != last_epoch:
            epoch_boundaries.append(global_step)
            last_epoch = epoch

        steps.append(global_step)
        losses.append(loss_val)
        regs.append(reg_val)
        preds.append(pred_val)

    # Sort by step (should already be sorted, but just in case)
    if steps:
        order = np.argsort(steps)
        steps = [steps[i] for i in order]
        losses = [losses[i] for i in order]
        regs = [regs[i] for i in order]
        preds = [preds[i] for i in order]

    # Subsample for plotting efficiency
    if sample_every > 1 and len(steps) > sample_every:
        indices = list(range(0, len(steps), sample_every))
        # Always include the last point
        if indices[-1] != len(steps) - 1:
            indices.append(len(steps) - 1)
        steps = [steps[i] for i in indices]
        losses = [losses[i] for i in indices]
        regs = [regs[i] for i in indices]
        preds = [preds[i] for i in indices]

    return {
        "steps": steps,
        "loss": losses,
        "reg": regs,
        "pred": preds,
        "epoch_boundaries": epoch_boundaries,
        "epochs_total": epochs_total,
        "steps_per_epoch": steps_per_epoch,
    }


def parse_epoch_summaries(filepath: str) -> Dict[str, Any]:
    """
    Parse epoch-level summary lines from SLURM stdout.

    Lines look like:
        [INFO ...][log_epoch ] [Epoch 000/12] loss=1.3912 | reg=1.2622 | pred=0.1290 | ...

    Returns dict with:
        epochs: list of epoch numbers
        loss: list of total loss per epoch
        reg: list of reg loss per epoch
        pred: list of pred loss per epoch
        config: dict of extracted config values (seed, idm_coeff, etc.)
    """
    epoch_pattern = re.compile(
        r"\[Epoch\s+(\d+)/(\d+)\]\s+"
        r"loss=([\d.]+)\s*\|\s*reg=([\d.]+)\s*\|\s*pred=([\d.]+)"
    )

    # Extract config info
    seed_pattern = re.compile(r"Seed:\s*(\d+)")
    idm_pattern = re.compile(r"IDM coeff:\s*([\d.]+)")
    reg_config_pattern = re.compile(r"model\.regularizer=(\{[^}]+\})")

    epochs = []
    losses = []
    regs = []
    preds = []
    config = {}

    with open(filepath) as f:
        for line in f:
            # Check for config lines
            m_seed = seed_pattern.search(line)
            if m_seed:
                config["seed"] = int(m_seed.group(1))

            m_idm = idm_pattern.search(line)
            if m_idm:
                config["idm_coeff"] = float(m_idm.group(1))

            m_reg = reg_config_pattern.search(line)
            if m_reg:
                try:
                    reg_dict = eval(m_reg.group(1))  # Safe: controlled log output
                    config["regularizer"] = reg_dict
                except Exception:
                    pass

            # Check for epoch summary lines
            m = epoch_pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(3)))
                regs.append(float(m.group(4)))
                preds.append(float(m.group(5)))

    return {
        "epochs": epochs,
        "loss": losses,
        "reg": regs,
        "pred": preds,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Smoothing utility
# ---------------------------------------------------------------------------

def smooth(values: List[float], window: int = 20) -> np.ndarray:
    """Apply a simple moving average for smoother curves."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    # Pad edges to avoid shrinking
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(values)]


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_training_curves_figure(
    full_data: Dict[str, Any],
    ablated_data: Dict[str, Any],
    full_epochs: Optional[Dict[str, Any]],
    ablated_epochs: Optional[Dict[str, Any]],
    output_dir: Path,
    smooth_window: int = 20,
) -> str:
    """
    Generate a 2x2 subplot figure comparing training dynamics.

    Subplots:
        1. Total loss over steps
        2. Prediction loss over steps
        3. Regularization loss over steps
        4. Epoch-level loss summary (bar chart)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    full_steps = np.array(full_data["steps"])
    abl_steps = np.array(ablated_data["steps"])

    # Common epoch boundaries for vertical lines
    full_boundaries = full_data.get("epoch_boundaries", [])
    abl_boundaries = ablated_data.get("epoch_boundaries", [])

    def _add_epoch_lines(ax, boundaries, color, alpha=0.15):
        for b in boundaries:
            ax.axvline(x=b, color=color, linestyle="--", alpha=alpha, linewidth=0.8)

    # Subplot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(full_steps, smooth(full_data["loss"], smooth_window),
            color=C_FULL, linewidth=1.5, label="Full Model (IDM=1)", alpha=0.9)
    ax.plot(abl_steps, smooth(ablated_data["loss"], smooth_window),
            color=C_ABLATED, linewidth=1.5, label="Ablated (IDM=0)", alpha=0.9)
    _add_epoch_lines(ax, full_boundaries, C_FULL)
    _add_epoch_lines(ax, abl_boundaries, C_ABLATED)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss", fontweight="bold")
    ax.legend(loc="upper right", fontsize=TICK_SIZE - 1)

    # Subplot 2: Prediction Loss
    ax = axes[0, 1]
    ax.plot(full_steps, smooth(full_data["pred"], smooth_window),
            color=C_FULL, linewidth=1.5, label="Full Model", alpha=0.9)
    ax.plot(abl_steps, smooth(ablated_data["pred"], smooth_window),
            color=C_ABLATED, linewidth=1.5, label="Ablated", alpha=0.9)
    _add_epoch_lines(ax, full_boundaries, C_FULL)
    _add_epoch_lines(ax, abl_boundaries, C_ABLATED)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Prediction Loss")
    ax.set_title("Prediction Loss", fontweight="bold")
    ax.legend(loc="upper right", fontsize=TICK_SIZE - 1)

    # Subplot 3: Regularization Loss
    ax = axes[1, 0]
    ax.plot(full_steps, smooth(full_data["reg"], smooth_window),
            color=C_FULL, linewidth=1.5, label="Full Model", alpha=0.9)
    ax.plot(abl_steps, smooth(ablated_data["reg"], smooth_window),
            color=C_ABLATED, linewidth=1.5, label="Ablated", alpha=0.9)
    _add_epoch_lines(ax, full_boundaries, C_FULL)
    _add_epoch_lines(ax, abl_boundaries, C_ABLATED)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Regularization Loss")
    ax.set_title("Regularization Loss", fontweight="bold")
    ax.legend(loc="upper right", fontsize=TICK_SIZE - 1)

    # Subplot 4: Epoch-level summary (if available)
    ax = axes[1, 1]
    if full_epochs and ablated_epochs and full_epochs["epochs"] and ablated_epochs["epochs"]:
        epochs_full = np.array(full_epochs["epochs"])
        epochs_abl = np.array(ablated_epochs["epochs"])
        width = 0.35

        ax.bar(epochs_full - width / 2, full_epochs["loss"], width,
               color=C_FULL, label="Full Model", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.bar(epochs_abl + width / 2, ablated_epochs["loss"], width,
               color=C_ABLATED, label="Ablated", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Epoch-Average Loss")
        ax.set_title("Per-Epoch Loss", fontweight="bold")
        ax.legend(loc="upper right", fontsize=TICK_SIZE - 1)
        ax.set_xticks(epochs_full)
    else:
        # Fallback: show reg/pred ratio
        if len(full_data["loss"]) > 0 and len(ablated_data["loss"]) > 0:
            full_ratio = np.array(full_data["reg"]) / (np.array(full_data["loss"]) + 1e-8)
            abl_ratio = np.array(ablated_data["reg"]) / (np.array(ablated_data["loss"]) + 1e-8)
            ax.plot(full_steps, smooth(full_ratio.tolist(), smooth_window),
                    color=C_FULL, linewidth=1.5, label="Full Model", alpha=0.9)
            ax.plot(abl_steps, smooth(abl_ratio.tolist(), smooth_window),
                    color=C_ABLATED, linewidth=1.5, label="Ablated", alpha=0.9)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Reg / Total Ratio")
            ax.set_title("Regularization Dominance", fontweight="bold")
            ax.legend(loc="lower right", fontsize=TICK_SIZE - 1)

    fig.suptitle(
        "Training Dynamics: Full Model vs IDM Ablation",
        fontsize=TITLE_SIZE + 2,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path = output_dir / "training_curves.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return str(out_path)


def make_loss_divergence_figure(
    full_data: Dict[str, Any],
    ablated_data: Dict[str, Any],
    output_dir: Path,
    smooth_window: int = 50,
) -> str:
    """
    Generate a single-panel figure highlighting the loss divergence
    between full and ablated models -- the key collapse indicator.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    full_steps = np.array(full_data["steps"])
    abl_steps = np.array(ablated_data["steps"])

    full_reg_smooth = smooth(full_data["reg"], smooth_window)
    abl_reg_smooth = smooth(ablated_data["reg"], smooth_window)

    ax.plot(full_steps, full_reg_smooth,
            color=C_FULL, linewidth=2, label="Full Model (with IDM)")
    ax.plot(abl_steps, abl_reg_smooth,
            color=C_ABLATED, linewidth=2, label="Ablated (no IDM)")

    # Shade the gap
    # Interpolate ablated to full steps for fill_between
    if len(full_steps) > 0 and len(abl_steps) > 0:
        common_max = min(full_steps[-1], abl_steps[-1])
        common_min = max(full_steps[0], abl_steps[0])
        mask_full = (full_steps >= common_min) & (full_steps <= common_max)
        common_steps = full_steps[mask_full]
        if len(common_steps) > 0:
            full_interp = np.interp(common_steps, full_steps, full_reg_smooth)
            abl_interp = np.interp(common_steps, abl_steps, abl_reg_smooth)
            ax.fill_between(
                common_steps, full_interp, abl_interp,
                alpha=0.15, color=C_ABLATED,
                label="Collapse gap",
            )

    # Add epoch boundary lines
    for b in full_data.get("epoch_boundaries", []):
        ax.axvline(x=b, color=C_BASELINE, linestyle="--", alpha=0.2, linewidth=0.8)

    ax.set_xlabel("Training Step", fontsize=LABEL_SIZE)
    ax.set_ylabel("Regularization Loss", fontsize=LABEL_SIZE)
    ax.set_title(
        "Regularization Loss Divergence: Evidence of Collapse Without IDM",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.legend(fontsize=TICK_SIZE, loc="upper right")

    fig.tight_layout()
    out_path = output_dir / "loss_divergence.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return str(out_path)


def make_loss_table(
    full_epochs: Optional[Dict[str, Any]],
    ablated_epochs: Optional[Dict[str, Any]],
    full_data: Dict[str, Any],
    ablated_data: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Generate a JSON summary table of final losses."""
    table: Dict[str, Any] = {}

    # From epoch summaries (if available)
    if full_epochs and full_epochs["epochs"]:
        table["full_model"] = {
            "source": "epoch_summary",
            "seed": full_epochs["config"].get("seed", "unknown"),
            "idm_coeff": full_epochs["config"].get("idm_coeff", "unknown"),
            "final_epoch": full_epochs["epochs"][-1],
            "final_loss": full_epochs["loss"][-1],
            "final_reg": full_epochs["reg"][-1],
            "final_pred": full_epochs["pred"][-1],
            "all_epochs": {
                str(e): {"loss": l, "reg": r, "pred": p}
                for e, l, r, p in zip(
                    full_epochs["epochs"],
                    full_epochs["loss"],
                    full_epochs["reg"],
                    full_epochs["pred"],
                )
            },
        }
    else:
        # Fallback to last tqdm values
        table["full_model"] = {
            "source": "tqdm_stderr",
            "final_step": full_data["steps"][-1] if full_data["steps"] else None,
            "final_loss": full_data["loss"][-1] if full_data["loss"] else None,
            "final_reg": full_data["reg"][-1] if full_data["reg"] else None,
            "final_pred": full_data["pred"][-1] if full_data["pred"] else None,
        }

    if ablated_epochs and ablated_epochs["epochs"]:
        table["ablated_model"] = {
            "source": "epoch_summary",
            "seed": ablated_epochs["config"].get("seed", "unknown"),
            "idm_coeff": ablated_epochs["config"].get("idm_coeff", "unknown"),
            "final_epoch": ablated_epochs["epochs"][-1],
            "final_loss": ablated_epochs["loss"][-1],
            "final_reg": ablated_epochs["reg"][-1],
            "final_pred": ablated_epochs["pred"][-1],
            "all_epochs": {
                str(e): {"loss": l, "reg": r, "pred": p}
                for e, l, r, p in zip(
                    ablated_epochs["epochs"],
                    ablated_epochs["loss"],
                    ablated_epochs["reg"],
                    ablated_epochs["pred"],
                )
            },
        }
    else:
        table["ablated_model"] = {
            "source": "tqdm_stderr",
            "final_step": ablated_data["steps"][-1] if ablated_data["steps"] else None,
            "final_loss": ablated_data["loss"][-1] if ablated_data["loss"] else None,
            "final_reg": ablated_data["reg"][-1] if ablated_data["reg"] else None,
            "final_pred": ablated_data["pred"][-1] if ablated_data["pred"] else None,
        }

    # Compute deltas
    f_loss = table["full_model"].get("final_loss")
    a_loss = table["ablated_model"].get("final_loss")
    if f_loss is not None and a_loss is not None:
        table["comparison"] = {
            "loss_ratio_full_over_ablated": round(f_loss / a_loss, 4) if a_loss != 0 else None,
            "loss_diff_full_minus_ablated": round(f_loss - a_loss, 4),
            "reg_ratio": round(
                table["full_model"]["final_reg"] / table["ablated_model"]["final_reg"], 4
            ) if table["ablated_model"].get("final_reg") else None,
        }

    out_path = output_dir / "loss_table.json"
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"  [OK] {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract training curves from SLURM logs and generate comparison figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--full_err", type=str, required=True,
        help="Path to SLURM stderr log for the full model (tqdm output).",
    )
    parser.add_argument(
        "--ablated_err", type=str, required=True,
        help="Path to SLURM stderr log for the ablated model (tqdm output).",
    )
    parser.add_argument(
        "--full_out", type=str, default=None,
        help="Path to SLURM stdout log for the full model (epoch summaries). Optional.",
    )
    parser.add_argument(
        "--ablated_out", type=str, default=None,
        help="Path to SLURM stdout log for the ablated model (epoch summaries). Optional.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_results/figures",
        help="Directory to save generated figures (default: eval_results/figures).",
    )
    parser.add_argument(
        "--sample_every", type=int, default=50,
        help="Sample every N steps from tqdm output to reduce data volume (default: 50).",
    )
    parser.add_argument(
        "--smooth_window", type=int, default=20,
        help="Moving average window size for smoothing curves (default: 20).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    # ---- Parse stderr (tqdm) ----
    print(f"Parsing full model stderr: {args.full_err}")
    full_data = parse_tqdm_stderr(args.full_err, sample_every=args.sample_every)
    print(f"  -> {len(full_data['steps'])} data points, "
          f"{full_data['epochs_total']} epochs, "
          f"{full_data['steps_per_epoch']} steps/epoch")

    print(f"Parsing ablated model stderr: {args.ablated_err}")
    ablated_data = parse_tqdm_stderr(args.ablated_err, sample_every=args.sample_every)
    print(f"  -> {len(ablated_data['steps'])} data points, "
          f"{ablated_data['epochs_total']} epochs, "
          f"{ablated_data['steps_per_epoch']} steps/epoch")

    # ---- Parse stdout (epoch summaries, optional) ----
    full_epochs = None
    ablated_epochs = None

    if args.full_out and os.path.exists(args.full_out):
        print(f"Parsing full model stdout: {args.full_out}")
        full_epochs = parse_epoch_summaries(args.full_out)
        print(f"  -> {len(full_epochs['epochs'])} epoch summaries, "
              f"config: seed={full_epochs['config'].get('seed')}, "
              f"idm_coeff={full_epochs['config'].get('idm_coeff')}")

    if args.ablated_out and os.path.exists(args.ablated_out):
        print(f"Parsing ablated model stdout: {args.ablated_out}")
        ablated_epochs = parse_epoch_summaries(args.ablated_out)
        print(f"  -> {len(ablated_epochs['epochs'])} epoch summaries, "
              f"config: seed={ablated_epochs['config'].get('seed')}, "
              f"idm_coeff={ablated_epochs['config'].get('idm_coeff')}")

    # ---- Validate ----
    if not full_data["steps"]:
        print("ERROR: No training steps found in full model stderr. Check file format.")
        sys.exit(1)
    if not ablated_data["steps"]:
        print("ERROR: No training steps found in ablated model stderr. Check file format.")
        sys.exit(1)

    # ---- Generate figures ----
    print()
    print("Generating figures...")

    generated = []

    # Figure 1: 2x2 training curves comparison
    path = make_training_curves_figure(
        full_data, ablated_data, full_epochs, ablated_epochs,
        output_dir, smooth_window=args.smooth_window,
    )
    generated.append(path)

    # Figure 2: Loss divergence (single panel, presentation-ready)
    path = make_loss_divergence_figure(
        full_data, ablated_data, output_dir,
        smooth_window=args.smooth_window * 2,  # extra smoothing for clarity
    )
    generated.append(path)

    # Loss table JSON
    path = make_loss_table(
        full_epochs, ablated_epochs, full_data, ablated_data, output_dir,
    )
    generated.append(path)

    # ---- Summary ----
    print()
    print(f"Generated {len(generated)} output(s) in '{output_dir}/':")
    for p in generated:
        print(f"  - {p}")

    # Print key finding
    if full_epochs and ablated_epochs:
        f_final = full_epochs["loss"][-1] if full_epochs["loss"] else None
        a_final = ablated_epochs["loss"][-1] if ablated_epochs["loss"] else None
        if f_final is not None and a_final is not None:
            print()
            print("KEY FINDING:")
            print(f"  Full model final loss:    {f_final:.4f}")
            print(f"  Ablated model final loss: {a_final:.4f}")
            ratio = f_final / a_final if a_final != 0 else float("inf")
            if ratio > 1:
                print(f"  Full model loss is {ratio:.2f}x HIGHER than ablated")
                print("  -> Full model (with IDM) has higher total loss but this includes")
                print("     the IDM regularization term. Check reg vs pred breakdown.")
            else:
                print(f"  Ablated model loss is {1/ratio:.2f}x HIGHER than full")

            f_pred = full_epochs["pred"][-1] if full_epochs["pred"] else None
            a_pred = ablated_epochs["pred"][-1] if ablated_epochs["pred"] else None
            if f_pred is not None and a_pred is not None:
                print(f"  Full model final pred loss:    {f_pred:.4f}")
                print(f"  Ablated model final pred loss: {a_pred:.4f}")
                if f_pred < a_pred:
                    print("  -> Full model has LOWER prediction loss (better world model)")
                else:
                    print("  -> Ablated model has lower prediction loss")


if __name__ == "__main__":
    main()
