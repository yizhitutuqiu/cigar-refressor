from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
from matplotlib.patches import Patch  # type: ignore[import-not-found]
from typing import Optional, Tuple
import re

try:
    # TensorBoard event reader
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    EventAccumulator = None  # type: ignore[assignment]


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


def infer_category(tag: str) -> str:
    t = tag.lower()
    if "vit_base" in t or "vit_b" in t:
        return "vit_b"
    if "vit_small" in t or "vit_s" in t:
        return "vit_s"
    if "resnet50" in t:
        return "resnet50"
    if "resnet34" in t:
        return "resnet34"
    if "resnet18" in t:
        return "resnet18"
    return "other"


CATEGORY_COLORS = {
    "vit_b": "#4e79a7",
    "vit_s": "#59a14f",
    "resnet50": "#e15759",
    "resnet34": "#f28e2b",
    "resnet18": "#b07aa1",
    "other": "#9c755f",
}


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
    # Render a prettier table figure using matplotlib
    n = len(entries)
    fig_h = max(4, 1.0 * n + 2)
    fig_w = max(12, 0.9 * n)
    fig, ax = plt.subplots(figsize=(min(fig_w, 16), fig_h))
    ax.axis("off")
    col_labels = ["tag", "category", "c_top1", "f_top1", "params(M)", "fwd(ms)", "peak_mem(MB)"]
    table_data = []
    row_colors = []
    for e in entries:
        cat = infer_category(e["tag"])
        table_data.append(
            [
                e["tag"],
                cat,
                f'{e["coarse_top1"]:.4f}',
                f'{e["fine_top1"]:.4f}',
                f'{e["param_count"] / 1e6:.2f}',
                f'{e["forward_ms_bs1_224"]:.2f}',
                f'{e["peak_memory_allocated_bytes"] / (1024**2):.1f}',
            ]
        )
        # alternating background
        row_colors.append(["#f9f9f9"] * len(col_labels) if (len(row_colors) % 2 == 0) else ["#ffffff"] * len(col_labels))
        # color the category cell
        row_colors[-1][1] = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])
    # header style
    header_colors = ["#eaeaea"] * len(col_labels)
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        colColours=header_colors,
        cellColours=row_colors,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.2)
    # adjust column widths a bit
    for i, key in enumerate(col_labels):
        for j in range(len(table_data) + 1):  # +1 for header
            cell = table[j, i]
            if cell is None:
                continue
            if i == 0:
                cell.set_width(0.36)
            elif i == 1:
                cell.set_width(0.14)
            else:
                cell.set_width(0.14)
    # add legend for categories
    handles = [Patch(color=CATEGORY_COLORS[k], label=k) for k in ["vit_b", "vit_s", "resnet50", "resnet34", "resnet18"]]
    ax.legend(handles=handles, bbox_to_anchor=(1.0, 1.02), loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar_f_acc(entries: List[Dict], out_dir: str, filename: str = "ablation_f_acc.png") -> None:
    tags = [e["tag"] for e in entries]
    f_vals = [e["fine_top1"] for e in entries]
    categories = [infer_category(t) for t in tags]
    colors = [CATEGORY_COLORS.get(c, CATEGORY_COLORS["other"]) for c in categories]
    fig, ax = plt.subplots(figsize=(max(10, len(tags) * 0.6), 6))
    bars = ax.bar(range(len(tags)), f_vals, color=colors, edgecolor="#333333", linewidth=0.5)
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
    # legend
    handles = [Patch(color=CATEGORY_COLORS[k], label=k) for k in ["vit_b", "vit_s", "resnet50", "resnet34", "resnet18"]]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _list_model_dirs_without_nofilm(ablation_root: str) -> List[str]:
    if not os.path.isdir(ablation_root):
        return []
    dirs = [d for d in sorted(os.listdir(ablation_root)) if os.path.isdir(os.path.join(ablation_root, d))]
    return [os.path.join(ablation_root, d) for d in dirs if not d.endswith("_nofilm")]


def _find_latest_tb_event_file(model_dir: str) -> Optional[str]:
    """
    Find the latest timestamp sub-directory under a model_dir and return the newest TB event file path inside its 'tb' folder.
    """
    if not os.path.isdir(model_dir):
        return None
    # timestamp subfolders, e.g. 20251110_115503
    subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not subdirs:
        return None
    # Sort by directory name first (lexicographic works for YYYYMMDD_HHMMSS), fallback to mtime
    subdirs_sorted = sorted(subdirs, reverse=True)
    candidate_dirs = [os.path.join(model_dir, d) for d in subdirs_sorted]
    # Fallback: ensure the tb exists
    for cand in candidate_dirs:
        tb_dir = os.path.join(cand, "tb")
        if not os.path.isdir(tb_dir):
            continue
        # event files often start with 'events.'; but be robust
        files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if os.path.isfile(os.path.join(tb_dir, f))]
        if not files:
            continue
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files[0]
    return None


def _select_train_loss_tag(tags: List[str]) -> Optional[str]:
    """
    Choose a 'train loss' scalar tag from a list of scalar tags robustly.
    Preference order:
      - exact matches: 'train/loss', 'train_loss'
      - contains both 'train' and 'loss' (case-insensitive)
      - endswith 'loss' while containing 'train'
    """
    lower = {t.lower(): t for t in tags}
    for pref in ["train/loss", "train_loss", "loss/train"]:
        if pref in lower:
            return lower[pref]
    # contains both
    for t in tags:
        tl = t.lower()
        if "train" in tl and "loss" in tl:
            return t
    # endswith 'loss' and has 'train'
    for t in tags:
        tl = t.lower()
        if tl.endswith("loss") and "train" in tl:
            return t
    return None


def _select_val_loss_tag(tags: List[str]) -> Optional[str]:
    """
    Choose a 'val loss' scalar tag from a list of scalar tags robustly.
    Preference order:
      - exact matches: 'val/loss', 'val_loss', 'validation/loss', 'valid/loss', 'eval/loss'
      - contains both 'val' (or synonyms) and 'loss' (case-insensitive)
      - endswith 'loss' while containing 'val' (or synonyms)
    """
    synonyms = ["val", "validation", "valid", "eval"]
    lower = {t.lower(): t for t in tags}
    for pref in ["val/loss", "val_loss", "validation/loss", "valid/loss", "eval/loss", "loss/val"]:
        if pref in lower:
            return lower[pref]
    # contains both
    for t in tags:
        tl = t.lower()
        if "loss" in tl and any(s in tl for s in synonyms):
            return t
    # endswith loss with val synonym
    for t in tags:
        tl = t.lower()
        if tl.endswith("loss") and any(s in tl for s in synonyms):
            return t
    return None


def _select_train_acc_tag(tags: List[str]) -> Optional[str]:
    """
    Choose a 'train accuracy' scalar tag robustly.
    Preference order:
      - exact: 'train/acc', 'train_accuracy', 'train/accuracy', 'acc/train', 'top1/train'
      - contains both 'train' and any of ['acc', 'accuracy', 'top1', 'acc1']
    """
    prefs = ["train/acc", "train_accuracy", "train/accuracy", "acc/train", "top1/train", "train/top1", "train/acc1"]
    lower = {t.lower(): t for t in tags}
    for p in prefs:
        if p in lower:
            return lower[p]
    for t in tags:
        tl = t.lower()
        if "train" in tl and any(k in tl for k in ["acc1", "top1", "acc", "accuracy"]):
            return t
    return None


def _select_val_acc_tag(tags: List[str]) -> Optional[str]:
    """
    Choose a 'val accuracy' scalar tag robustly.
    Preference order:
      - exact: 'val/acc', 'val_accuracy', 'val/accuracy', 'validation/accuracy', 'valid/acc', 'eval/acc', 'top1/val'
      - contains any of ['val','validation','valid','eval'] and any of ['acc','accuracy','top1','acc1']
    """
    prefs = [
        "val/acc",
        "val_accuracy",
        "val/accuracy",
        "validation/accuracy",
        "valid/acc",
        "eval/acc",
        "top1/val",
        "val/top1",
        "val/acc1",
    ]
    lower = {t.lower(): t for t in tags}
    for p in prefs:
        if p in lower:
            return lower[p]
    synonyms = ["val", "validation", "valid", "eval"]
    for t in tags:
        tl = t.lower()
        if any(s in tl for s in synonyms) and any(k in tl for k in ["acc1", "top1", "acc", "accuracy"]):
            return t
    return None


def plot_train_loss_grid(ablation_root: str, out_dir: str, filename: str = "ablation_train_loss_grid.png") -> None:
    """
    Visualize training loss curves from TensorBoard event files for each model directory (excluding *_nofilm).
    Select the latest timestamp subdir and its 'tb' events file per model, then plot as subplots into a single figure.
    """
    if EventAccumulator is None:
        print("[WARN] tensorboard not installed; skip loss grid plot")
        return
    model_dirs = _list_model_dirs_without_nofilm(ablation_root)
    if not model_dirs:
        print(f"[WARN] No model directories found at {ablation_root}")
        return
    tags_to_plot = []
    data_series = []
    for mdir in model_dirs:
        tag = os.path.basename(mdir)
        event_path = _find_latest_tb_event_file(mdir)
        tr_steps: List[int] = []
        tr_values: List[float] = []
        va_steps: List[int] = []
        va_values: List[float] = []
        tr_a_steps: List[int] = []
        tr_a_values: List[float] = []
        va_a_steps: List[int] = []
        va_a_values: List[float] = []
        if event_path is not None:
            try:
                acc = EventAccumulator(event_path, size_guidance={"scalars": 0})
                acc.Reload()
                scalar_tags = acc.Tags().get("scalars", [])
                chosen_tr = _select_train_loss_tag(scalar_tags or [])
                if chosen_tr:
                    scalar_events = acc.Scalars(chosen_tr)
                    tr_steps = [se.step for se in scalar_events]
                    tr_values = [float(se.value) for se in scalar_events]
                # val loss removed per requirement: keep only train loss, train acc, val acc
                chosen_tr_a = _select_train_acc_tag(scalar_tags or [])
                if chosen_tr_a:
                    scalar_events_ta = acc.Scalars(chosen_tr_a)
                    tr_a_steps = [se.step for se in scalar_events_ta]
                    tr_a_values = [float(se.value) for se in scalar_events_ta]
                chosen_va_a = _select_val_acc_tag(scalar_tags or [])
                if chosen_va_a:
                    scalar_events_va = acc.Scalars(chosen_va_a)
                    va_a_steps = [se.step for se in scalar_events_va]
                    va_a_values = [float(se.value) for se in scalar_events_va]
            except Exception as e:  # pragma: no cover
                print(f"[WARN] Failed to read TB file for {tag}: {e}")
        tags_to_plot.append(tag)
        data_series.append(((tr_steps, tr_values), (va_steps, va_values), (tr_a_steps, tr_a_values), (va_a_steps, va_a_values)))

    n = len(tags_to_plot)
    cols = 4
    rows = (n + cols - 1) // cols
    fig_h = max(6, 3 * rows)
    fig_w = max(12, 4 * cols)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    for idx, (tag, ((tr_steps, tr_values), (va_steps, va_values), (tr_a_steps, tr_a_values), (va_a_steps, va_a_values))) in enumerate(zip(tags_to_plot, data_series)):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        has_loss = False
        has_acc = False
        loss_lines = []
        acc_lines = []
        if tr_steps and tr_values:
            line, = ax.plot(tr_steps, tr_values, color="#4e79a7", linewidth=1.5, label="train loss")
            loss_lines.append(line)
            has_loss = True
        # val loss removed
        # accuracy on right Y-axis
        ax2 = ax.twinx()
        if tr_a_steps and tr_a_values:
            line2, = ax2.plot(tr_a_steps, tr_a_values, color="#59a14f", linewidth=1.0, linestyle="-", label="train acc")
            acc_lines.append(line2)
            has_acc = True
        if va_a_steps and va_a_values:
            line2, = ax2.plot(va_a_steps, va_a_values, color="#f28e2b", linewidth=1.0, linestyle="-", label="val acc")
            acc_lines.append(line2)
            has_acc = True
        if not has_loss and not has_acc:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#666666")
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("loss", fontsize=9)
        ax2.set_ylabel("acc", fontsize=9)
        ax2.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3, linestyle="--")
        if loss_lines or acc_lines:
            lines = loss_lines + acc_lines
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="upper right", fontsize=8, frameon=False)
    # Hide unused subplots if any
    for j in range(n, rows * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize ablation results")
    parser.add_argument("--input_root", type=str, default="./cifar_regressor/test/ablation")
    parser.add_argument("--output_dir", type=str, default="./cifar_regressor/vis/out")
    parser.add_argument("--sort_by", type=str, default="fine_top1", choices=["fine_top1", "coarse_top1", "param_count", "forward_ms_bs1_224"])
    parser.add_argument("--descending", action="store_true", default=True)
    parser.add_argument("--ablation_ckpt_root", type=str, default="./cifar_regressor/checkpoints/ablation")
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
        # Extra bar chart without entries ending with '_nofilm'
        plot_bar_f_acc(no_nofilm, args.output_dir, filename="ablation_f_acc_no_nofilm.png")
    # 2) Remove both 'nofilm' and 'cbam'
    no_nofilm_nocbam = [e for e in entries if (not e["tag"].endswith("_nofilm")) and ("_cbam" not in e["tag"])]
    if no_nofilm_nocbam:
        save_csv_md(no_nofilm_nocbam, args.output_dir, prefix="ablation_table_no_nofilm_nocbam")
        plot_table(no_nofilm_nocbam, args.output_dir, filename="ablation_table_no_nofilm_nocbam.png")

    # Train loss grid (from TensorBoard)
    plot_train_loss_grid(args.ablation_ckpt_root, args.output_dir, filename="ablation_train_loss_grid.png")


if __name__ == "__main__":
    main()


