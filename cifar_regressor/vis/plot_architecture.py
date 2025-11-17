from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
from matplotlib.patches import Rectangle, FancyArrow  # type: ignore[import-not-found]


def add_box(ax, xy, width, height, text, fc="#ffffff", ec="#333333", lw=1.2, fontsize=10, color_bar=None):
    rect = Rectangle(xy, width, height, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    x, y = xy
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    if color_bar is not None:
        # small color bar on the left to indicate module type
        cb = Rectangle((x - 0.05, y), 0.05, height, facecolor=color_bar, edgecolor=ec, linewidth=lw)
        ax.add_patch(cb)
    return rect


def add_arrow(ax, x0, y0, x1, y1, text: str = "", fontsize=9):
    ax.add_patch(FancyArrow(x0, y0, x1 - x0, y1 - y0, width=0.02, head_width=0.12, head_length=0.18, length_includes_head=True, color="#555555"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", va="bottom", fontsize=fontsize, color="#444444")


def draw_resnet18_film_arch(out_dir: str):
    # Bigger canvas and proper axis limits to avoid squeezing
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axis("off")
    ax.set_xlim(0, 20.5)
    ax.set_ylim(0, 7.5)

    # Color legend for module types
    COLORS = {
        "input": "#f0f0f0",
        "backbone": "#d0e4f7",
        "attn": "#ffe0b2",
        "pool": "#e1f5fe",
        "proj": "#f0f4c3",
        "head": "#c8e6c9",
        "film": "#f8bbd0",
        "softmax": "#fff9c4",
    }

    # Row y positions
    y_main = 4.5
    y_head = 2.2
    y_text = 7.0

    # Input
    add_box(ax, (0.4, y_main), 1.2, 0.8, "Input\n(3×224×224)", fc=COLORS["input"], color_bar="#999999")
    add_arrow(ax, 1.6, y_main + 0.4, 2.0, y_main + 0.4, "")

    # Unified Encoder (ResNet18/34 or ViT-S/B)
    add_box(ax, (2.0, y_main), 3.2, 0.8, "Encoder\n(ResNet18/34 or ViT‑S/B)", fc=COLORS["backbone"], color_bar="#4e79a7")

    # GAP/Flatten or Pooled features
    add_arrow(ax, 5.2, y_main + 0.4, 5.8, y_main + 0.4, "")
    add_box(ax, (5.8, y_main), 1.3, 0.8, "Pooling/Flatten\n(Feature dim ~512/768)", fc=COLORS["pool"], color_bar="#59a14f")

    # Coarse head (MLP)
    add_arrow(ax, 7.1, y_main + 0.4, 7.7, y_main + 0.4, "")
    add_box(ax, (7.7, y_main), 1.7, 0.8, "Coarse MLP\nd → 256 → 20", fc=COLORS["head"], color_bar="#8cd17d")
    add_arrow(ax, 9.4, y_main + 0.4, 10.0, y_main + 0.4, "")
    add_box(ax, (10.0, y_main), 1.2, 0.8, "Softmax\n(coarse)", fc=COLORS["softmax"], color_bar="#ffdd57")

    # FiLM adapter (20→hidden→γ,β(512))
    # Route from coarse_probs (14.0, y_main) down to features path near fine head
    # Feature path for fine head (duplicate of flattened features)
    # Draw duplication note
    ax.text(6.5, y_main + 1.0, "Features (d-dim)", ha="center", va="bottom", fontsize=9, color="#444444")

    # FiLM box below
    # Place Adapter to the right of coarse softmax on the MAIN row (not the same row as Modulation)
    add_box(ax, (12.2, y_main), 2.0, 0.8, "FiLM Adapter\n(coarse→γ,β in d)", fc=COLORS["film"], color_bar="#f8bbd0")
    # Arrow: coarse softmax → adapter (horizontal, same row)
    add_arrow(ax, 11.2, y_main + 0.4, 12.2, y_main + 0.4, "conditioning", fontsize=8)

    # Fine head (after FiLM modulation)
    # Place Modulation to the right of Adapter
    add_box(ax, (14.2, y_head), 2.2, 0.8, "Modulation\n(1+γ)·x + β", fc=COLORS["proj"], color_bar="#b07aa1")
    # Connect pooled features to modulation (diagonal)
    add_arrow(ax, 6.4, y_main, 14.2, y_head + 0.8, "", fontsize=8)

    # Adapter → Modulation (down-right diagonal: FiLM outputs γ,β to modulation)
    add_arrow(ax, 13.2, y_main, 14.2, y_head + 0.8, "")
    # Modulation → Fine MLP
    add_arrow(ax, 16.4, y_head + 0.4, 16.8, y_head + 0.4, "")

    add_box(ax, (16.8, y_head), 2.0, 0.8, "Fine MLP\nd → 256 → 100", fc=COLORS["head"], color_bar="#8cd17d")
    add_arrow(ax, 18.8, y_head + 0.4, 19.2, y_head + 0.4, "")
    add_box(ax, (19.2, y_head), 1.0, 0.8, "Softmax\n(fine)", fc=COLORS["softmax"], color_bar="#ffdd57")

    # Title
    ax.text(0.4, y_text, "Encoder + FiLM + MLP Heads (coarse+fine)", fontsize=14, ha="left", va="center")
    ax.text(0.4, y_text - 0.5,
            "Flow: Input → Encoder(ResNet18/34 or ViT‑S/B) → Pooling/Flatten → Coarse MLP → Softmax → "
            "FiLM(γ,β) → Modulation → Fine MLP → Softmax",
            fontsize=10, ha="left", va="center", color="#555")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    # 命名：带 FiLM 用后缀 _film
    png_path = os.path.join(out_dir, "encoder_arch_film.png")
    svg_path = os.path.join(out_dir, "encoder_arch_film.svg")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(svg_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {png_path}\n[OK] saved: {svg_path}")


def draw_encoder_nofilm_arch(out_dir: str):
    # Canvas
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axis("off")
    ax.set_xlim(0, 20.5)
    ax.set_ylim(0, 7.5)

    COLORS = {
        "input": "#f0f0f0",
        "backbone": "#d0e4f7",
        "pool": "#e1f5fe",
        "head": "#c8e6c9",
        "softmax": "#fff9c4",
    }
    y_main = 4.5
    y_fine = 2.2
    y_text = 7.0

    # Input → Encoder → Pooling
    add_box(ax, (0.4, y_main), 1.2, 0.8, "Input\n(3×224×224)", fc=COLORS["input"], color_bar="#999999")
    add_arrow(ax, 1.6, y_main + 0.4, 2.0, y_main + 0.4, "")
    add_box(ax, (2.0, y_main), 3.2, 0.8, "Encoder\n(ResNet18/34 or ViT‑S/B)", fc=COLORS["backbone"], color_bar="#4e79a7")
    add_arrow(ax, 5.2, y_main + 0.4, 5.8, y_main + 0.4, "")
    add_box(ax, (5.8, y_main), 1.3, 0.8, "Pooling/Flatten\n(Feature dim ~512/768)", fc=COLORS["pool"], color_bar="#59a14f")

    # Coarse branch (main row)
    add_arrow(ax, 7.1, y_main + 0.4, 7.7, y_main + 0.4, "")
    add_box(ax, (7.7, y_main), 1.7, 0.8, "Coarse MLP\nd → 256 → 20", fc=COLORS["head"], color_bar="#8cd17d")
    add_arrow(ax, 9.4, y_main + 0.4, 10.0, y_main + 0.4, "")
    add_box(ax, (10.0, y_main), 1.2, 0.8, "Softmax\n(coarse)", fc=COLORS["softmax"], color_bar="#ffdd57")

    # Fine branch (lower row)
    add_arrow(ax, 6.4, y_main, 7.7, y_fine + 0.8, "")
    add_box(ax, (7.7, y_fine), 1.9, 0.8, "Fine MLP\nd → 256 → 100", fc=COLORS["head"], color_bar="#8cd17d")
    add_arrow(ax, 9.6, y_fine + 0.4, 10.8, y_fine + 0.4, "")
    add_box(ax, (10.8, y_fine), 1.2, 0.8, "Softmax\n(fine)", fc=COLORS["softmax"], color_bar="#ffdd57")

    # Title
    ax.text(0.4, y_text, "Encoder + MLP Heads (coarse+fine) — No FiLM", fontsize=14, ha="left", va="center")
    ax.text(0.4, y_text - 0.5,
            "Flow: Input → Encoder(ResNet18/34 or ViT‑S/B) → Pooling/Flatten → "
            "Coarse MLP → Softmax & Fine MLP → Softmax",
            fontsize=10, ha="left", va="center", color="#555")

    os.makedirs(out_dir, exist_ok=True)
    # 命名：默认无后缀（不带 FiLM/CBAM）
    png_path = os.path.join(out_dir, "encoder_arch.png")
    svg_path = os.path.join(out_dir, "encoder_arch.svg")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(svg_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {png_path}\n[OK] saved: {svg_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot network architecture")
    parser.add_argument("--output_dir", type=str, default="./cifar_regressor/vis/out")
    parser.add_argument("--variant", type=str, default="both", choices=["film", "nofilm", "both"])
    args = parser.parse_args()
    if args.variant in ("film", "both"):
        draw_resnet18_film_arch(args.output_dir)
    if args.variant in ("nofilm", "both"):
        draw_encoder_nofilm_arch(args.output_dir)


if __name__ == "__main__":
    main()


