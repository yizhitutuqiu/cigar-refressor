from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt  # type: ignore[import-not-found]


def load_results(input_root: str) -> List[Dict]:
    entries = []
    for tag in sorted(os.listdir(input_root)):
        tag_dir = os.path.join(input_root, tag)
        if not os.path.isdir(tag_dir):
            continue
        report_path = os.path.join(tag_dir, "eval_report_hier.json")
        if not os.path.isfile(report_path):
            continue
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        entries.append(
            {
                "tag": tag,
                "coarse_top1": float(rep["coarse"]["top1"]),
                "fine_top1": float(rep["fine"]["top1"]),
                "param_count": int(rep.get("model_meta", {}).get("param_count", 0)),
                "forward_ms_bs1_224": float(rep.get("model_meta", {}).get("forward_ms_bs1_224", 0.0)),
                "peak_memory_allocated_bytes": int(rep.get("model_meta", {}).get("peak_memory_allocated_bytes", 0)),
                "checkpoint_path": rep.get("checkpoint_path", ""),
                "config": rep.get("config", {}),
            }
        )
    return entries


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv_md(entries: List[Dict], out_dir: str, prefix: str = "ablation_table") -> None:
    import csv

    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    md_path = os.path.join(out_dir, f"{prefix}.md")

    headers = [
        "tag",
        "coarse_top1",
        "fine_top1",
        "param_count(M)",
        "forward_ms_bs1_224",
        "peak_mem(MB)",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for e in entries:
            writer.writerow(
                [
                    e["tag"],
                    f'{e["coarse_top1"]:.4f}',
                    f'{e["fine_top1"]:.4f}',
                    f'{e["param_count"] / 1e6:.2f}',
                    f'{e["forward_ms_bs1_224"]:.2f}',
                    f'{e["peak_memory_allocated_bytes"] / (1024**2):.1f}',
                ]
            )

    # Markdown table
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for e in entries:
        lines.append(
            "| "
            + " | ".join(
                [
                    e["tag"],
                    f'{e["coarse_top1"]:.4f}',
                    f'{e["fine_top1"]:.4f}',
                    f'{e["param_count"] / 1e6:.2f}',
                    f'{e["forward_ms_bs1_224"]:.2f}',
                    f'{e["peak_memory_allocated_bytes"] / (1024**2):.1f}',
                ]
            )
            + " |"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_table(entries: List[Dict], out_dir: str, filename: str = "ablation_table.png") -> None:
    # Render a table figure using matplotlib
    fig, ax = plt.subplots(figsize=(max(10, len(entries) * 0.9), 0.6 * len(entries) + 2))
    ax.axis("off")
    col_labels = ["tag", "c_top1", "f_top1", "params(M)", "fwd(ms)", "peak_mem(MB)"]
    table_data = []
    for e in entries:
        table_data.append(
            [
                e["tag"],
                f'{e["coarse_top1"]:.4f}',
                f'{e["fine_top1"]:.4f}',
                f'{e["param_count"] / 1e6:.2f}',
                f'{e["forward_ms_bs1_224"]:.2f}',
                f'{e["peak_memory_allocated_bytes"] / (1024**2):.1f}',
            ]
        )
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar_f_acc(entries: List[Dict], out_dir: str) -> None:
    tags = [e["tag"] for e in entries]
    f_vals = [e["fine_top1"] for e in entries]
    fig, ax = plt.subplots(figsize=(max(10, len(tags) * 0.6), 6))
    bars = ax.bar(range(len(tags)), f_vals, color="#4e79a7")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_ylabel("fine top-1 accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # add value labels (3 decimals)
    for rect, val in zip(bars, f_vals):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_f_acc.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize ablation results")
    parser.add_argument("--input_root", type=str, default="./cifar_regressor/test/ablation")
    parser.add_argument("--output_dir", type=str, default="./cifar_regressor/vis/out")
    parser.add_argument("--sort_by", type=str, default="fine_top1", choices=["fine_top1", "coarse_top1", "param_count", "forward_ms_bs1_224"])
    parser.add_argument("--descending", action="store_true", default=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    entries = load_results(args.input_root)
    if not entries:
        print(f"[WARN] No eval_report_hier.json found under {args.input_root}")
        return
    entries.sort(key=lambda e: e[args.sort_by], reverse=args.descending)

    # Save raw JSON
    with open(os.path.join(args.output_dir, "ablation_sorted.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    save_csv_md(entries, args.output_dir)
    plot_table(entries, args.output_dir, filename="ablation_table.png")
    plot_bar_f_acc(entries, args.output_dir)
    print(f"[OK] outputs saved to: {args.output_dir}")

    # Extra tables:
    # 1) Remove all 'nofilm'
    no_nofilm = [e for e in entries if not e["tag"].endswith("_nofilm")]
    if no_nofilm:
        save_csv_md(no_nofilm, args.output_dir, prefix="ablation_table_no_nofilm")
        plot_table(no_nofilm, args.output_dir, filename="ablation_table_no_nofilm.png")
    # 2) Remove both 'nofilm' and 'cbam'
    no_nofilm_nocbam = [e for e in entries if (not e["tag"].endswith("_nofilm")) and ("_cbam" not in e["tag"])]
    if no_nofilm_nocbam:
        save_csv_md(no_nofilm_nocbam, args.output_dir, prefix="ablation_table_no_nofilm_nocbam")
        plot_table(no_nofilm_nocbam, args.output_dir, filename="ablation_table_no_nofilm_nocbam.png")


if __name__ == "__main__":
    main()


