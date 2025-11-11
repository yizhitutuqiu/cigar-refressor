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


