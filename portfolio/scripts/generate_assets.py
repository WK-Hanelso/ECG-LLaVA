#!/usr/bin/env python3
"""Generate public-safe portfolio figures from committed Phase 1 artifacts only.

This script does not load a model, run evaluation, or create new performance data.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "portfolio" / "assets"
METRICS_PATH = REPO_ROOT / "results" / "resnet1d_superdiag_base" / "metrics.json"
MANIFEST_PATH = REPO_ROOT / "results" / "resnet1d_superdiag_base" / "prepare_manifest.json"

NAVY = "#15324A"
BLUE = "#1F6FEB"
LIGHT_BLUE = "#EAF2FF"
TEAL = "#14866D"
LIGHT_TEAL = "#E8F6F2"
ORANGE = "#D97706"
LIGHT_ORANGE = "#FFF4DF"
GRAY = "#667085"
LIGHT_GRAY = "#F2F4F7"
DARK = "#101828"


def load_sources() -> tuple[dict, dict]:
    with METRICS_PATH.open(encoding="utf-8") as file:
        metrics = json.load(file)
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)
    return metrics, manifest


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 24,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 16,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_both(fig: plt.Figure, basename: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_DIR / f"{basename}.png", dpi=150, facecolor="white")
    fig.savefig(ASSET_DIR / f"{basename}.svg", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, title, subtitle, face=LIGHT_BLUE, edge=BLUE, dashed=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=2.0,
        edgecolor=edge,
        facecolor=face,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=15, weight="bold", color=DARK, linespacing=1.05)
    ax.text(x + w / 2, y + h * 0.25, subtitle, ha="center", va="center", fontsize=14, color=GRAY, linespacing=1.18)
    return patch


def arrow(ax, start, end, color=GRAY, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2,
            color=color,
            linestyle="--" if dashed else "-",
            shrinkA=4,
            shrinkB=4,
        )
    )


def generate_pipeline(metrics: dict, manifest: dict) -> None:
    used = manifest["n_used"]
    split = manifest["split"]
    shape = manifest["signal_shape"]
    integrity = (
        manifest["n_lead_order_mismatch"],
        manifest["n_records_with_nan"],
        manifest["n_length_mismatch"],
    )
    feature = metrics["feature_shape"]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.94, "Phase 1: Evidence-backed ECG pipeline", fontsize=26, weight="bold", color=NAVY)
    ax.text(0.04, 0.895, "Data integrity and leakage controls are explicit before model training", fontsize=16, color=GRAY)

    top_y, w, h = 0.61, 0.137, 0.23
    xs = [0.035, 0.196, 0.357, 0.518, 0.679, 0.840]
    items = [
        ("PTB-XL\nWFDB", "v1.0.3 · 100 Hz\n12 leads\n10 seconds"),
        ("Integrity\ncheck", f"lead mismatch {integrity[0]}\nNaN records {integrity[1]}\nlength mismatch {integrity[2]}"),
        ("Label\nmapping", f"5 classes\nmultilabel\n{used:,} records"),
        ("Fixed-shape\ncache", f"{shape[0]:,} samples\n{shape[1]} × {shape[2]}\nfloat32"),
        ("Official\nstrat_fold", f"train {split['n_train']:,}\nval {split['n_val']:,}\ntest {split['n_test']:,}"),
        ("Train-only\nnormalization", "lead-wise mean / std\nfolds 1–8 only"),
    ]
    for index, (title, subtitle) in enumerate(items):
        box(ax, xs[index], top_y, w, h, title, subtitle)
        if index < len(items) - 1:
            arrow(ax, (xs[index] + w, top_y + h / 2), (xs[index + 1], top_y + h / 2))

    model_x, model_y = 0.35, 0.19
    class_x, token_x = 0.635, 0.825
    box(ax, model_x, model_y, 0.20, 0.18, "ResNet1D encoder", "8,739,973 parameters\nPhase 1 baseline", LIGHT_TEAL, TEAL)
    box(ax, class_x, model_y + 0.10, 0.155, 0.16, "Classification", "5 sigmoid logits\nmacro AUROC 0.9143", LIGHT_ORANGE, ORANGE)
    box(ax, token_x, model_y - 0.04, 0.145, 0.16, "Token interface", f"{feature[1]} × {feature[2]}\n{feature[2]} temporal tokens", LIGHT_TEAL, TEAL)
    route_y = 0.545
    ax.plot([xs[-1] + w / 2, xs[-1] + w / 2, model_x + 0.10], [top_y, route_y, route_y], color=TEAL, linewidth=2)
    arrow(ax, (model_x + 0.10, route_y), (model_x + 0.10, model_y + 0.18), TEAL)
    arrow(ax, (model_x + 0.20, model_y + 0.12), (class_x, model_y + 0.18), ORANGE)
    arrow(ax, (model_x + 0.20, model_y + 0.06), (token_x, model_y + 0.04), TEAL)

    ax.text(0.04, 0.08, "Source: committed prepare_manifest.json and metrics.json · No new training or evaluation", fontsize=13, color=GRAY)
    save_both(fig, "01_ecg_pipeline")


def generate_auroc(metrics: dict) -> None:
    test = metrics["test"]
    labels = ["NORM", "MI", "STTC", "CD", "HYP"]
    values = [test["per_class_auroc"][label] for label in labels]
    macro = test["macro_auroc"]
    macro_f1 = test["macro_f1"]

    fig, ax = plt.subplots(figsize=(16, 9))
    positions = list(range(len(labels)))
    bars = ax.barh(positions, values, height=0.56, color=[BLUE, TEAL, BLUE, TEAL, ORANGE], alpha=0.92)
    ax.invert_yaxis()
    ax.set_yticks(positions, labels)
    ax.set_xlim(0.86, 0.95)
    ax.set_xlabel("AUROC on official test fold")
    ax.set_title("Class-wise AUROC — PTB-XL v1.0.3 test fold", loc="left", weight="bold", color=NAVY, pad=22)
    ax.text(0.0, 1.015, f"n = {test['n_samples']:,} · five-label multilabel classification", transform=ax.transAxes, fontsize=15, color=GRAY)
    ax.axvline(macro, color=NAVY, linewidth=2.5, linestyle="--", label=f"Macro AUROC {macro:.4f}")
    for bar, value in zip(bars, values):
        ax.text(value - 0.001, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", ha="right", va="center", fontsize=16, weight="bold", color="white")
    ax.grid(axis="x", color="#D0D5DD", linewidth=1, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=15)
    ax.text(
        0.99,
        0.98,
        f"Macro F1  {macro_f1:.4f}\nthreshold  {test['threshold']:.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=16,
        color=DARK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT_GRAY, edgecolor="none"),
    )
    fig.text(0.125, 0.02, "Source: results/resnet1d_superdiag_base/metrics.json · Stored test artifact", fontsize=13, color=GRAY)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.84, bottom=0.13)
    save_both(fig, "02_class_auroc")


def generate_token_interface(metrics: dict) -> None:
    feature = metrics["feature_shape"]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.05, 0.91, "Encoder output is the Phase 1 contract", fontsize=27, weight="bold", color=NAVY)
    ax.text(0.05, 0.855, "The projector and language model remain outside the implemented scope", fontsize=16, color=GRAY)

    y, h = 0.43, 0.25
    box(ax, 0.04, y, 0.18, h, "12-lead ECG", "(B, 12, 1000)\n10 s at 100 Hz", LIGHT_BLUE, BLUE)
    box(ax, 0.27, y, 0.18, h, "ResNet1D", "stem + 4 stages\nbefore global pooling", LIGHT_TEAL, TEAL)
    box(ax, 0.50, y, 0.18, h, "Feature map", f"(B, {feature[1]}, {feature[2]})\nchannels × time", LIGHT_TEAL, TEAL)
    box(ax, 0.73, y, 0.15, h, "32 temporal tokens", "one token per\ntime step", LIGHT_ORANGE, ORANGE)
    arrow(ax, (0.22, y + h / 2), (0.27, y + h / 2), BLUE)
    arrow(ax, (0.45, y + h / 2), (0.50, y + h / 2), TEAL)
    arrow(ax, (0.68, y + h / 2), (0.73, y + h / 2), ORANGE)

    boundary_x = 0.90
    ax.plot([boundary_x, boundary_x], [0.33, 0.78], linestyle="--", linewidth=2.5, color=ORANGE)
    ax.text(boundary_x - 0.005, 0.29, "Phase 1 boundary", ha="right", va="top", fontsize=14, weight="bold", color=ORANGE)
    ax.text(0.92, 0.555, "Phase 2\nnot built\n\nProjector\n→ LLM", ha="left", va="center", fontsize=15, color=GRAY)

    token_left, token_bottom = 0.57, 0.19
    for index in range(32):
        column = index % 16
        row = index // 16
        ax.add_patch(
            Rectangle(
                (token_left + column * 0.011, token_bottom - row * 0.035),
                0.008,
                0.022,
                facecolor=TEAL,
                alpha=0.85,
                edgecolor="none",
            )
        )
    ax.text(0.57, 0.095, "32 positions preserved before pooling", fontsize=14, color=GRAY)
    ax.text(0.05, 0.06, "Verified by forward_features contract and stored feature_shape [1, 512, 32]", fontsize=13, color=GRAY)
    save_both(fig, "03_token_interface")


def main() -> None:
    configure_style()
    metrics, manifest = load_sources()
    generate_pipeline(metrics, manifest)
    generate_auroc(metrics)
    generate_token_interface(metrics)


if __name__ == "__main__":
    main()
