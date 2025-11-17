from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict

import numpy as np  # type: ignore[import-not-found]
from PIL import Image  # type: ignore[import-not-found]

import torch  # type: ignore[import-not-found]
from torch.utils.data import Dataset, DataLoader  # type: ignore[import-not-found]
from torchvision import transforms  # type: ignore[import-not-found]
import matplotlib.pyplot as plt  # type: ignore[import-not-found]
from typing import List, Tuple
from matplotlib.patches import Rectangle  # type: ignore[import-not-found]

# Ensure project root is importable when running by absolute path
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from cifar_regressor import CifarHierarchicalRegressor  # type: ignore


class Cifar100DualTest(Dataset):
	def __init__(self, root_dir: str, transform: transforms.Compose | None = None) -> None:
		super().__init__()
		self.root_dir = root_dir
		self.transform = transform
		import pickle
		with open(os.path.join(root_dir, "test"), "rb") as f:
			data = pickle.load(f, encoding="latin1")
		self.data = data["data"]
		self.coarse = data["coarse_labels"]
		self.fine = data["fine_labels"]

	def __len__(self) -> int:
		return len(self.fine)

	def __getitem__(self, index: int):
		row = self.data[index]
		img = np.reshape(row, (3, 32, 32))
		img = np.transpose(img, (1, 2, 0))
		img = Image.fromarray(img.astype(np.uint8))
		if self.transform is not None:
			img = self.transform(img)
		return img, int(self.coarse[index]), int(self.fine[index])


def build_val_transform() -> transforms.Compose:
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	return transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	])

def _denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
	# img_tensor: (C,H,W), normalized by ImageNet mean/std
	mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
	std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
	img = img_tensor.detach().cpu().float().numpy()
	img = (img * std + mean)
	img = np.clip(img, 0.0, 1.0)
	img = (np.transpose(img, (1, 2, 0)) * 255.0).astype(np.uint8)
	return img

@torch.no_grad()
def collect_hard_cases(model: CifarHierarchicalRegressor, loader: DataLoader, device: torch.device, max_cases: int = 100):
	model.eval()
	cases = []
	for images, coarse_t, fine_t in loader:
		images = images.to(device, non_blocking=True)
		coarse_t = coarse_t.to(device, non_blocking=True)
		fine_t = fine_t.to(device, non_blocking=True)
		out = model(images)
		c_probs = out["coarse_probs"]
		f_probs = out["fine_probs"]
		c_pred1 = torch.argmax(c_probs, dim=1)
		f_pred1 = torch.argmax(f_probs, dim=1)
		for i in range(images.size(0)):
			wrong_c = (c_pred1[i] != coarse_t[i]).item()
			wrong_f = (f_pred1[i] != fine_t[i]).item()
			if wrong_c or wrong_f:
				img_np = _denormalize_image(images[i])
				cases.append({
					"image": img_np,
					"coarse_true": int(coarse_t[i].item()),
					"fine_true": int(fine_t[i].item()),
					"coarse_pred": int(c_pred1[i].item()),
					"fine_pred": int(f_pred1[i].item()),
					"wrong_c": bool(wrong_c),
					"wrong_f": bool(wrong_f),
				})
				if len(cases) >= max_cases:
					return cases
	return cases

def _safe_name(names: List[str], idx: int) -> str:
	if names and 0 <= idx < len(names):
		return str(names[idx])
	return str(idx)

def plot_hard_cases_grid(cases, save_path: str, coarse_names: List[str], fine_names: List[str], model_tag: str, rows: int = 5, cols: int = 8) -> None:
	# rows x cols grid, each cell shows image + two text rows (GT fine, PR fine)
	n = min(rows * cols, len(cases))
	fig_w, fig_h = max(10, cols * 1.6), max(7, rows * 1.6)
	fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
	fig.suptitle(f"Hard cases — {model_tag}", fontsize=14)
	for idx in range(rows * cols):
		ax = axes[idx // cols][idx % cols]
		ax.axis("off")
		if idx >= n:
			continue
		case = cases[idx]
		img = case["image"]
		ax.imshow(img)
		# reserve lower area for a table-like annotation block
		true_f = case["fine_true"]
		pred_f = case["fine_pred"]
		wf = case["wrong_f"]
		# map to names (only fine labels requested)
		true_f_name = _safe_name(fine_names, true_f)
		pred_f_name = _safe_name(fine_names, pred_f)
		# Draw table background (bottom 22% height)
		table_bottom = 0.0
		table_height = 0.22
		bg = Rectangle((0.0, table_bottom), 1.0, table_height, transform=ax.transAxes, facecolor="white", alpha=0.75, edgecolor="#cccccc", linewidth=0.5)
		ax.add_patch(bg)
		# divider line between GT row and PR row
		ax.plot([0.05, 0.95], [table_bottom + table_height * 0.5, table_bottom + table_height * 0.5], transform=ax.transAxes, color="#dddddd", linewidth=0.8)
		# Text rows (non-overlapping fixed positions)
		ax.text(0.05, table_bottom + table_height * 0.75, f"GT: {true_f_name}", ha="left", va="center", fontsize=8, color="#000000", transform=ax.transAxes)
		ax.text(0.05, table_bottom + table_height * 0.25, f"PR: {pred_f_name}", ha="left", va="center", fontsize=8, color=("#d62728" if wf else "#000000"), transform=ax.transAxes)
	# reduce whitespace
	fig.tight_layout(rect=[0, 0, 1, 0.93])
	fig.subplots_adjust(hspace=0.25, wspace=0.1)
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	fig.savefig(save_path, dpi=200, bbox_inches="tight")
	plt.close(fig)

def _load_cifar100_label_names(dataset_root: str) -> Tuple[List[str], List[str]]:
	"""
	Load CIFAR-100 coarse and fine label names from the 'meta' file in dataset root.
	Fallback to empty lists if not available.
	"""
	import pickle
	meta_path = os.path.join(dataset_root, "meta")
	try:
		with open(meta_path, "rb") as f:
			meta = pickle.load(f, encoding="latin1")
		coarse = list(meta.get("coarse_label_names", []))
		fine = list(meta.get("fine_label_names", []))
		return coarse, fine
	except Exception:
		return [], []

@torch.no_grad()
def evaluate(model: CifarHierarchicalRegressor, loader: DataLoader, device: torch.device) -> Dict:
	model.eval()
	total = 0
	# coarse
	coarse_top1 = 0
	coarse_top5 = 0
	coarse_num = model.num_coarse if hasattr(model, "num_coarse") else 20
	# fine
	fine_top1 = 0
	fine_top5 = 0
	fine_num = model.num_fine if hasattr(model, "num_fine") else 100

	for images, coarse_t, fine_t in loader:
		images = images.to(device, non_blocking=True)
		coarse_t = coarse_t.to(device, non_blocking=True)
		fine_t = fine_t.to(device, non_blocking=True)
		out = model(images)
		c_probs = out["coarse_probs"]
		f_probs = out["fine_probs"]

		# coarse
		c_pred1 = torch.argmax(c_probs, dim=1)
		coarse_top1 += (c_pred1 == coarse_t).sum().item()
		_, c_pred5 = torch.topk(c_probs, k=min(5, c_probs.size(1)), dim=1)
		coarse_top5 += (c_pred5 == coarse_t.view(-1, 1)).any(dim=1).sum().item()
		# fine
		f_pred1 = torch.argmax(f_probs, dim=1)
		fine_top1 += (f_pred1 == fine_t).sum().item()
		_, f_pred5 = torch.topk(f_probs, k=min(5, f_probs.size(1)), dim=1)
		fine_top5 += (f_pred5 == fine_t.view(-1, 1)).any(dim=1).sum().item()

		total += images.size(0)

	return {
		"total": int(total),
		"coarse": {
			"top1": float(coarse_top1 / max(1, total)),
			"top5": float(coarse_top5 / max(1, total)),
		},
		"fine": {
			"top1": float(fine_top1 / max(1, total)),
			"top5": float(fine_top5 / max(1, total)),
		},
	}


def build_model_from_checkpoint(ckpt_path: str, device: torch.device) -> tuple[CifarHierarchicalRegressor, Dict]:
	ckpt = torch.load(ckpt_path, map_location="cpu")
	cfg = ckpt.get("config", {}) or {}
	model = CifarHierarchicalRegressor(
		pretrained_backbone=False,
		encoder_name=str(cfg.get("encoder_name", "resnet18")),
		use_cbam=bool(cfg.get("use_cbam", False)),
		num_coarse=int(cfg.get("num_coarse", 20)),
		num_fine=int(cfg.get("num_fine", 100)),
		hidden_features=int(cfg.get("hidden_features", 256)),
		dropout_p=float(cfg.get("dropout_p", 0.1)),
		use_film=bool(cfg.get("use_film", True)),
		film_hidden=int(cfg.get("film_hidden", 256)),
		film_use_probs=bool(cfg.get("film_use_probs", True)),
	)
	state_dict = ckpt.get("model", ckpt)
	model.load_state_dict(state_dict, strict=False)
	model.to(device).eval()
	return model, cfg


def measure_meta(model: CifarHierarchicalRegressor, device: torch.device) -> Dict:
	param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
	# 前向时间与峰值显存（bs=1, 224）
	dummy = torch.randn(1, 3, 224, 224, device=device)
	with torch.no_grad():
		for _ in range(5):
			_ = model(dummy)
		if device.type == "cuda":
			torch.cuda.synchronize()
		torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
		import time
		t0 = time.perf_counter()
		for _ in range(10):
			_ = model(dummy)
		if device.type == "cuda":
			torch.cuda.synchronize()
		t1 = time.perf_counter()
	forward_ms = (t1 - t0) / 10.0 * 1000.0
	peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
	return {
		"param_count": int(param_count),
		"forward_ms_bs1_224": forward_ms,
		"peak_memory_allocated_bytes": peak_mem,
	}


def main():
	parser = argparse.ArgumentParser(description="Evaluate all ablation checkpoints on CIFAR-100 test set")
	parser.add_argument("--checkpoint_root", type=str, default="./cifar_regressor/checkpoints/ablation")
	parser.add_argument("--dataset_root", type=str, default="./cifar-100-python")
	parser.add_argument("--output_root", type=str, default="./cifar_regressor/test/ablation")
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--gpu", type=int, default=6)
	args = parser.parse_args()

	device_str = args.device
	if args.gpu is not None:
		device_str = f"cuda:{args.gpu}"
	device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")

	os.makedirs(args.output_root, exist_ok=True)
	val_tf = build_val_transform()
	test_ds = Cifar100DualTest(args.dataset_root, transform=val_tf)
	loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

	entries = sorted([d for d in os.listdir(args.checkpoint_root) if os.path.isdir(os.path.join(args.checkpoint_root, d))])
	summary = {"results": []}
	coarse_names, fine_names = _load_cifar100_label_names(args.dataset_root)
	for tag in entries:
		ckpt_dir = os.path.join(args.checkpoint_root, tag)
		ckpt_path = os.path.join(ckpt_dir, "best.pth")
		if not os.path.isfile(ckpt_path):
			print(f"[SKIP] no best.pth in {ckpt_dir}")
			continue
		out_dir = os.path.join(args.output_root, tag)
		os.makedirs(out_dir, exist_ok=True)

		# Build and evaluate
		model, cfg = build_model_from_checkpoint(ckpt_path, device)
		metrics = evaluate(model, loader, device)
		meta = measure_meta(model, device)

		# Compose and save
		report = {
			"tag": tag,
			"checkpoint_path": ckpt_path,
			"dataset_root": args.dataset_root,
			"num_samples": metrics["total"],
			"coarse": metrics["coarse"],
			"fine": metrics["fine"],
			"config": cfg,
			"model_meta": meta,
		}
		with open(os.path.join(out_dir, "eval_report_hier.json"), "w", encoding="utf-8") as f:
			json.dump(report, f, ensure_ascii=False, indent=2)

		# Visualize hard cases (first 40 wrong predictions)
		try:
			hard_cases = collect_hard_cases(model, loader, device, max_cases=40)
			if hard_cases:
				hard_path = os.path.join(out_dir, f"hard_cases_{tag}.png")
				plot_hard_cases_grid(hard_cases, hard_path, coarse_names, fine_names, model_tag=tag, rows=5, cols=8)
				print(f"[OK] hard cases saved -> {hard_path}")
			else:
				print("[INFO] no hard cases found")
		except Exception as e:
			print(f"[WARN] failed to generate hard cases for {tag}: {e}")

		# Keep a flat summary
		summary["results"].append({
			"tag": tag,
			"coarse_top1": report["coarse"]["top1"],
			"fine_top1": report["fine"]["top1"],
			"param_count": meta["param_count"],
			"forward_ms_bs1_224": meta["forward_ms_bs1_224"],
			"peak_memory_allocated_bytes": meta["peak_memory_allocated_bytes"],
		})
		print(f"[OK] {tag} -> fine_top1={report['fine']['top1']:.4f}")

	summary_path = os.path.join(args.output_root, f"ablation_eval_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)
	print(f"[DONE] summary => {summary_path}")


if __name__ == "__main__":
	main()


