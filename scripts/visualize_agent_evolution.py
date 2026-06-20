#!/usr/bin/env python3
"""
Visualize a JEPA planning agent's BEHAVIOR evolving across training epochs.

At epoch 0 the agent has a random/untrained world model and dies quickly.
By epoch 11 the model has learned to predict, and the planner keeps the agent
alive longer, gathering resources along the way.

Outputs:
  1. agent_evolution.gif  -- side-by-side animated GIF showing actual Crafter
     game frames from episodes played by the planner at each epoch.
  2. agent_stats.png      -- bar-chart summary of mean episode length,
     mean reward, and achievement count per epoch.

Usage:
    python scripts/visualize_agent_evolution.py \
        --checkpoint_dir checkpoints/.../idm1_seed1000/ \
        --output_dir eval_results/agent_evolution \
        --epochs 0,5,11 \
        --max_steps 100 \
        --num_samples 100
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse model-building utilities from the planning agent script
from scripts.planning_agent import (
    build_jepa,
    load_checkpoint,
    RandomShootingPlanner,
    DEFAULT_MODEL_CFG,
    DEFAULT_NORM_MEAN,
    DEFAULT_NORM_STD,
    CRAFTER_ACHIEVEMENTS,
)
from eb_jepa.datasets.crafter.normalizer import CrafterNormalizer


# ---------------------------------------------------------------------------
# Run one episode with the planner, recording every frame + stats
# ---------------------------------------------------------------------------

def run_agent_episode(
    jepa,
    normalizer,
    env,
    max_steps: int = 100,
    num_samples: int = 100,
    horizon: int = 8,
    device: torch.device = torch.device("cpu"),
):
    """Run one episode and return frames, rewards, achievements, and inventory."""
    obs = env.reset()
    frames = [obs.copy()]
    rewards = []
    achievements = set()
    inventories = [{}]  # placeholder for t=0

    planner = RandomShootingPlanner(
        jepa,
        normalizer,
        num_samples=num_samples,
        horizon=horizon,
        objective="exploration",
        num_actions=17,
        device=device,
    )

    info = {}  # in case episode ends on first step
    for step in range(max_steps):
        action = planner.plan(obs)
        obs, reward, done, info = env.step(action)
        frames.append(obs.copy())
        rewards.append(reward)
        inv = info.get("inventory", {})
        inventories.append(dict(inv))
        if "achievements" in info:
            for k, v in info["achievements"].items():
                if v > 0:
                    achievements.add(k)
        if done:
            break

    return {
        "frames": np.stack(frames),          # [T+1, 64, 64, 3]  uint8
        "total_reward": float(sum(rewards)),
        "length": len(rewards),
        "achievements": sorted(achievements),
        "inventories": inventories,          # list of dicts, length T+1
    }


# ---------------------------------------------------------------------------
# Frame annotation helpers (PIL-based for quality text rendering)
# ---------------------------------------------------------------------------

EPOCH_COLORS = {
    0: (220, 50, 50),     # red
    5: (220, 180, 30),    # yellow-gold
    11: (40, 180, 60),    # green
}

# Fallback for arbitrary epochs
def _epoch_color(epoch):
    if epoch in EPOCH_COLORS:
        return EPOCH_COLORS[epoch]
    frac = min(epoch / 11.0, 1.0)
    r = int(220 * (1 - frac) + 40 * frac)
    g = int(50 * (1 - frac) + 180 * frac)
    b = int(50 * (1 - frac) + 60 * frac)
    return (r, g, b)


def upscale_frame(frame_uint8, scale: int = 4):
    """Nearest-neighbor upscale a [H, W, 3] uint8 array."""
    img = Image.fromarray(frame_uint8)
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.NEAREST)


def annotate_frame(pil_img, epoch, step, inventory, border_color, font=None):
    """Add a coloured border and text overlay to a single upscaled frame.

    Returns a new PIL Image.
    """
    border = 4
    w, h = pil_img.size
    new_w = w + 2 * border
    new_h = h + 2 * border + 28  # extra space at bottom for text
    canvas = Image.new("RGB", (new_w, new_h), border_color)
    canvas.paste(pil_img, (border, border))

    draw = ImageDraw.Draw(canvas)

    health = inventory.get("health", "?")
    wood = inventory.get("wood", 0)
    stone = inventory.get("stone", 0)

    label = f"Epoch {epoch} | Step {step} | HP:{health} | W:{wood} S:{stone}"

    # Use a small default font
    if font is None:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
            except Exception:
                font = ImageFont.load_default()

    text_y = h + 2 * border + 2
    # Semi-transparent backdrop via rectangle
    draw.rectangle(
        [(0, text_y - 2), (new_w, new_h)],
        fill=(0, 0, 0),
    )
    draw.text((4, text_y), label, fill=(255, 255, 255), font=font)

    # Health bar at top
    bar_w_max = w
    bar_h = 3
    if isinstance(health, (int, float)):
        frac = max(0, min(health / 9.0, 1.0))
        bar_color = (0, 200, 0) if frac > 0.5 else (200, 200, 0) if frac > 0.25 else (200, 0, 0)
        draw.rectangle(
            [(border, border), (border + int(bar_w_max * frac), border + bar_h)],
            fill=bar_color,
        )

    return canvas


# ---------------------------------------------------------------------------
# Build the side-by-side GIF
# ---------------------------------------------------------------------------

def build_sidebyside_gif(
    epoch_data: dict,
    output_path: str,
    fps: int = 5,
    scale: int = 4,
    max_gif_frames: int = 60,
):
    """Create an animated GIF with all epochs shown side-by-side per frame.

    epoch_data: {epoch_int: episode_result_dict}
    """
    epochs_sorted = sorted(epoch_data.keys())
    num_epochs = len(epochs_sorted)

    # Determine per-epoch frame counts and the frame indices to use
    per_epoch_indices = {}
    for ep in epochs_sorted:
        n_frames = len(epoch_data[ep]["frames"])
        max_per = max_gif_frames // num_epochs
        if n_frames <= max_per:
            indices = list(range(n_frames))
        else:
            # Sub-sample evenly
            indices = np.linspace(0, n_frames - 1, max_per, dtype=int).tolist()
        per_epoch_indices[ep] = indices

    # Maximum number of time-steps across epochs
    max_t = max(len(v) for v in per_epoch_indices.values())

    # Pre-render annotated frames per epoch
    rendered = {}  # {epoch: [pil_img, ...]}
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
        except Exception:
            font = ImageFont.load_default()

    for ep in epochs_sorted:
        ep_result = epoch_data[ep]
        color = _epoch_color(ep)
        rendered[ep] = []
        indices = per_epoch_indices[ep]
        for idx in indices:
            raw = ep_result["frames"][idx]
            inv = ep_result["inventories"][idx] if idx < len(ep_result["inventories"]) else {}
            pil = upscale_frame(raw, scale)
            annotated = annotate_frame(pil, ep, idx, inv, color, font=font)
            rendered[ep].append(annotated)

    # Build composite frames: side-by-side
    # Get single-cell size from first rendered frame
    cell_w, cell_h = rendered[epochs_sorted[0]][0].size
    gap = 6
    title_h = 36
    composite_w = num_epochs * cell_w + (num_epochs - 1) * gap
    composite_h = cell_h + title_h

    gif_frames = []
    for t_idx in range(max_t):
        composite = Image.new("RGB", (composite_w, composite_h), (20, 20, 30))
        draw = ImageDraw.Draw(composite)

        for col, ep in enumerate(epochs_sorted):
            ep_frames = rendered[ep]
            # Clamp: if this epoch has fewer frames, hold on last
            frame_idx = min(t_idx, len(ep_frames) - 1)
            cell = ep_frames[frame_idx]
            x_off = col * (cell_w + gap)
            composite.paste(cell, (x_off, title_h))

            # Epoch header
            color = _epoch_color(ep)
            header = f"Epoch {ep}"
            try:
                header_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
            except Exception:
                try:
                    header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16)
                except Exception:
                    header_font = ImageFont.load_default()

            # Center header text
            bbox = draw.textbbox((0, 0), header, font=header_font)
            tw = bbox[2] - bbox[0]
            tx = x_off + (cell_w - tw) // 2
            draw.text((tx, 8), header, fill=color, font=header_font)

        # If an epoch has finished (agent died), stamp "DONE" on held frame
        for col, ep in enumerate(epochs_sorted):
            ep_len = len(rendered[ep])
            if t_idx >= ep_len:
                x_off = col * (cell_w + gap)
                draw.text(
                    (x_off + cell_w // 2 - 30, title_h + cell_h // 2),
                    "DEAD",
                    fill=(255, 60, 60),
                    font=header_font,
                )

        gif_frames.append(composite)

    # Save GIF using Pillow
    duration_ms = int(1000 / fps)
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"Saved GIF ({len(gif_frames)} frames, {fps} fps) to {output_path}")


# ---------------------------------------------------------------------------
# Build sequential GIF (epoch 0 full, then epoch 5, then epoch 11)
# ---------------------------------------------------------------------------

def build_sequential_gif(
    epoch_data: dict,
    output_path: str,
    fps: int = 5,
    scale: int = 4,
    frames_per_epoch: int = 25,
):
    """Create an animated GIF showing epochs sequentially with a title card."""
    epochs_sorted = sorted(epoch_data.keys())

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
        except Exception:
            font = ImageFont.load_default()

    try:
        big_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 28)
    except Exception:
        try:
            big_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 28)
        except Exception:
            big_font = ImageFont.load_default()

    # Determine canvas size from one upscaled frame
    sample_raw = epoch_data[epochs_sorted[0]]["frames"][0]
    sample_pil = upscale_frame(sample_raw, scale)
    sample_annotated = annotate_frame(sample_pil, 0, 0, {}, (128, 128, 128), font=font)
    cell_w, cell_h = sample_annotated.size
    canvas_w = cell_w + 20
    canvas_h = cell_h + 20

    gif_frames = []

    for ep in epochs_sorted:
        color = _epoch_color(ep)
        ep_result = epoch_data[ep]
        n_frames = len(ep_result["frames"])

        # Title card (shown for ~1 second = fps frames)
        title_card = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 30))
        draw = ImageDraw.Draw(title_card)
        title_text = f"Epoch {ep}"
        bbox = draw.textbbox((0, 0), title_text, font=big_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            ((canvas_w - tw) // 2, (canvas_h - th) // 2 - 20),
            title_text,
            fill=color,
            font=big_font,
        )
        sub_text = f"Length: {ep_result['length']} | Reward: {ep_result['total_reward']:.1f} | Achievements: {len(ep_result['achievements'])}"
        sub_bbox = draw.textbbox((0, 0), sub_text, font=font)
        stw = sub_bbox[2] - sub_bbox[0]
        draw.text(
            ((canvas_w - stw) // 2, (canvas_h - th) // 2 + 20),
            sub_text,
            fill=(200, 200, 200),
            font=font,
        )
        for _ in range(fps):  # show title for ~1 second
            gif_frames.append(title_card.copy())

        # Sub-sample frames
        if n_frames <= frames_per_epoch:
            indices = list(range(n_frames))
        else:
            indices = np.linspace(0, n_frames - 1, frames_per_epoch, dtype=int).tolist()

        for idx in indices:
            raw = ep_result["frames"][idx]
            inv = ep_result["inventories"][idx] if idx < len(ep_result["inventories"]) else {}
            pil = upscale_frame(raw, scale)
            annotated = annotate_frame(pil, ep, idx, inv, color, font=font)
            # Center on canvas
            canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 30))
            x_off = (canvas_w - annotated.size[0]) // 2
            y_off = (canvas_h - annotated.size[1]) // 2
            canvas.paste(annotated, (x_off, y_off))
            gif_frames.append(canvas)

    duration_ms = int(1000 / fps)
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"Saved sequential GIF ({len(gif_frames)} frames) to {output_path}")


# ---------------------------------------------------------------------------
# Stats bar chart
# ---------------------------------------------------------------------------

def plot_agent_stats(epoch_data: dict, output_path: str):
    """Bar chart of mean episode length, reward, and achievements per epoch."""
    epochs_sorted = sorted(epoch_data.keys())
    n = len(epochs_sorted)

    lengths = [epoch_data[e]["length"] for e in epochs_sorted]
    rewards = [epoch_data[e]["total_reward"] for e in epochs_sorted]
    n_ach = [len(epoch_data[e]["achievements"]) for e in epochs_sorted]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="white")

    bar_colors = [
        tuple(c / 255.0 for c in _epoch_color(e)) for e in epochs_sorted
    ]
    labels = [f"Epoch {e}" for e in epochs_sorted]
    x = np.arange(n)

    # --- Episode Length ---
    ax = axes[0]
    bars = ax.bar(x, lengths, color=bar_colors, edgecolor="gray", linewidth=0.8, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Steps survived", fontsize=12)
    ax.set_title("Episode Length", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, lengths):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, val + 1,
            str(val), ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    ax.set_ylim(0, max(lengths) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    # --- Total Reward ---
    ax = axes[1]
    bars = ax.bar(x, rewards, color=bar_colors, edgecolor="gray", linewidth=0.8, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Total reward", fontsize=12)
    ax.set_title("Episode Reward", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, rewards):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + (0.1 if val >= 0 else -0.3),
            f"{val:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    y_min = min(0, min(rewards) * 1.3)
    y_max = max(rewards) * 1.3 if max(rewards) > 0 else 1
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.3)

    # --- Achievements ---
    ax = axes[2]
    bars = ax.bar(x, n_ach, color=bar_colors, edgecolor="gray", linewidth=0.8, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Achievements Unlocked", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, n_ach):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, val + 0.15,
            str(val), ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    ax.set_ylim(0, max(n_ach) * 1.4 + 1)
    ax.grid(axis="y", alpha=0.3)

    # Add achievement names for last epoch
    best_ep = epochs_sorted[-1]
    achs = epoch_data[best_ep]["achievements"]
    if achs:
        ach_str = ", ".join(achs)
        fig.text(
            0.5, 0.01,
            f"Epoch {best_ep} achievements: {ach_str}",
            ha="center", fontsize=10, style="italic", color="#333",
        )

    fig.suptitle(
        "Agent Behavior Across Training Epochs",
        fontsize=16, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved agent stats chart to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a JEPA planning agent's behavior evolving across training epochs."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Directory containing e-{epoch}.pth.tar checkpoints.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_results/agent_evolution",
        help="Where to save the GIF and stats chart.",
    )
    parser.add_argument(
        "--epochs", type=str, default="0,5,11",
        help="Comma-separated list of epoch numbers to compare (default: 0,5,11).",
    )
    parser.add_argument(
        "--max_steps", type=int, default=100,
        help="Maximum steps per episode (default: 100).",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="Random-shooting samples per planning step (default: 100).",
    )
    parser.add_argument(
        "--horizon", type=int, default=8,
        help="Planning horizon (default: 8).",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1,
        help="Episodes to run per epoch (uses the best one for GIF, averages stats) (default: 1).",
    )
    parser.add_argument(
        "--fps", type=int, default=5,
        help="GIF frames per second (default: 5).",
    )
    parser.add_argument(
        "--scale", type=int, default=4,
        help="Upscale factor for frames, 64->256 at 4x (default: 4).",
    )
    parser.add_argument(
        "--gif_mode", type=str, default="both",
        choices=["sidebyside", "sequential", "both"],
        help="GIF layout: side-by-side, sequential, or both (default: both).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: auto-detect).",
    )
    args = parser.parse_args()

    # --- Parse epochs ---
    epochs = [int(e.strip()) for e in args.epochs.split(",")]
    print(f"Epochs to evaluate: {epochs}")

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Seed ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Output dir ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # --- Normalizer ---
    normalizer = CrafterNormalizer(
        mean=torch.tensor(DEFAULT_NORM_MEAN, dtype=torch.float32),
        std=torch.tensor(DEFAULT_NORM_STD, dtype=torch.float32),
    )

    # --- Crafter environment ---
    import crafter
    env = crafter.Env()
    print("Crafter environment created.")

    # --- Check available checkpoints ---
    print("\n--- Checking checkpoints ---")
    available = []
    for ep in epochs:
        ckpt_path = os.path.join(args.checkpoint_dir, f"e-{ep}.pth.tar")
        if os.path.exists(ckpt_path):
            available.append(ep)
            print(f"  Found: e-{ep}.pth.tar")
        else:
            print(f"  MISSING: e-{ep}.pth.tar -- skipping")
    if not available:
        print("ERROR: No checkpoints found. Check --checkpoint_dir.")
        sys.exit(1)
    epochs = available

    # --- Run episodes for each epoch ---
    epoch_data = {}  # {epoch: best_episode_result}
    model_cfg = dict(DEFAULT_MODEL_CFG)

    for ep in epochs:
        ckpt_path = os.path.join(args.checkpoint_dir, f"e-{ep}.pth.tar")
        print(f"\n{'='*60}")
        print(f"  EPOCH {ep}")
        print(f"{'='*60}")

        # Build fresh model each time to avoid weight leakage
        jepa = build_jepa(model_cfg, device)
        load_checkpoint(jepa, ckpt_path, device)
        jepa.eval()

        best_result = None
        all_results = []

        for ep_i in range(args.num_episodes):
            # Reset seed per episode for variety but reproducibility
            env_seed = args.seed + ep * 1000 + ep_i
            np.random.seed(env_seed)

            t0 = time.time()
            result = run_agent_episode(
                jepa, normalizer, env,
                max_steps=args.max_steps,
                num_samples=args.num_samples,
                horizon=args.horizon,
                device=device,
            )
            dt = time.time() - t0

            all_results.append(result)
            print(
                f"  Episode {ep_i+1}/{args.num_episodes}: "
                f"length={result['length']}, "
                f"reward={result['total_reward']:.1f}, "
                f"achievements={result['achievements']}, "
                f"time={dt:.1f}s"
            )

            # Keep the longest episode for the GIF
            if best_result is None or result["length"] > best_result["length"]:
                best_result = result

        # If we ran multiple episodes, store averaged stats but best frames
        if args.num_episodes > 1:
            best_result["avg_length"] = float(np.mean([r["length"] for r in all_results]))
            best_result["avg_reward"] = float(np.mean([r["total_reward"] for r in all_results]))
            all_achs = set()
            for r in all_results:
                all_achs.update(r["achievements"])
            best_result["all_achievements"] = sorted(all_achs)

        epoch_data[ep] = best_result

        # Free model memory
        del jepa
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Generate outputs
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    # 1. Side-by-side GIF
    if args.gif_mode in ("sidebyside", "both"):
        gif_path = os.path.join(args.output_dir, "agent_evolution.gif")
        build_sidebyside_gif(
            epoch_data, gif_path,
            fps=args.fps, scale=args.scale, max_gif_frames=80,
        )

    # 2. Sequential GIF
    if args.gif_mode in ("sequential", "both"):
        seq_path = os.path.join(args.output_dir, "agent_evolution_sequential.gif")
        build_sequential_gif(
            epoch_data, seq_path,
            fps=args.fps, scale=args.scale, frames_per_epoch=25,
        )

    # 3. Stats bar chart
    stats_path = os.path.join(args.output_dir, "agent_stats.png")
    plot_agent_stats(epoch_data, stats_path)

    # 4. Save numerical results
    json_results = {
        "epochs": epochs,
        "config": {
            "max_steps": args.max_steps,
            "num_samples": args.num_samples,
            "horizon": args.horizon,
            "num_episodes": args.num_episodes,
            "seed": args.seed,
        },
        "per_epoch": {
            str(ep): {
                "length": epoch_data[ep]["length"],
                "total_reward": epoch_data[ep]["total_reward"],
                "achievements": epoch_data[ep]["achievements"],
                "num_frames": len(epoch_data[ep]["frames"]),
            }
            for ep in epochs
        },
    }
    json_path = os.path.join(args.output_dir, "agent_evolution_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved results JSON to {json_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  AGENT EVOLUTION SUMMARY")
    print(f"{'='*60}")
    for ep in epochs:
        r = epoch_data[ep]
        print(
            f"  Epoch {ep:>2d}: "
            f"survived {r['length']:>3d} steps, "
            f"reward={r['total_reward']:>6.1f}, "
            f"achievements={r['achievements']}"
        )
    print(f"\n  Outputs:")
    if args.gif_mode in ("sidebyside", "both"):
        print(f"    Side-by-side GIF: {args.output_dir}/agent_evolution.gif")
    if args.gif_mode in ("sequential", "both"):
        print(f"    Sequential GIF:   {args.output_dir}/agent_evolution_sequential.gif")
    print(f"    Stats chart:      {args.output_dir}/agent_stats.png")
    print(f"    Results JSON:     {args.output_dir}/agent_evolution_results.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
