from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Tuple, Optional

import numpy as np  # type: ignore[import-not-found]
from PIL import Image  # type: ignore[import-not-found]

import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]
from torch.utils.data import Dataset, DataLoader  # type: ignore[import-not-found]
from torchvision import transforms  # type: ignore[import-not-found]

# Ensure project root
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


def build_val_transform(image_size: int = 224) -> transforms.Compose:
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	return transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(image_size),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	])


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
	model.eval()
	total = 0
	c_top1 = 0
	f_top1 = 0
	for images, coarse_t, fine_t in loader:
		images = images.to(device, non_blocking=True)
		coarse_t = coarse_t.to(device, non_blocking=True)
		fine_t = fine_t.to(device, non_blocking=True)
		out = model(images)
		c_pred = out["coarse_logits"].argmax(1)
		f_pred = out["fine_logits"].argmax(1)
		c_top1 += (c_pred == coarse_t).sum().item()
		f_top1 += (f_pred == fine_t).sum().item()
		total += images.size(0)
	return {"c_top1": c_top1 / max(1, total), "f_top1": f_top1 / max(1, total), "total": total}


def count_params(model: nn.Module) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_forward_ms(model: nn.Module, device: torch.device, image_size: int = 224) -> float:
	dummy = torch.randn(1, 3, image_size, image_size, device=device)
	with torch.no_grad():
		for _ in range(5):
			_ = model(dummy)
		if device.type == "cuda":
			torch.cuda.synchronize()
		t0 = time.perf_counter()
		for _ in range(10):
			_ = model(dummy)
		if device.type == "cuda":
			torch.cuda.synchronize()
		t1 = time.perf_counter()
	return (t1 - t0) / 10.0 * 1000.0


def find_latest_pruned_dir(base_dir: str) -> Optional[str]:
	candidates = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("pruned_channel_")]
	if not candidates:
		return None
	candidates.sort(key=lambda d: os.path.getmtime(os.path.join(base_dir, d)), reverse=True)
	return os.path.join(base_dir, candidates[0])


def main():
	parser = argparse.ArgumentParser(description="Compare pre/post pruning performance under a checkpoint dir")
	parser.add_argument("--checkpoint_dir", type=str, required=True, help="原模型目录，包含 best.pth 及 pruned_channel_* 子目录")
	parser.add_argument("--dataset_root", type=str, default="./cifar-100-python")
	parser.add_argument("--output_path", type=str, default="./cifar_regressor/test/ablation/prune_compare.json")
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--gpu", type=int, default=6)
	parser.add_argument("--pruned_subdir", type=str, default="", help="指定剪枝子目录名（默认取最新的 pruned_channel_*）")
	args = parser.parse_args()

	device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
	orig_ckpt = os.path.join(args.checkpoint_dir, "best.pth")
	if not os.path.isfile(orig_ckpt):
		raise FileNotFoundError(f"未找到原模型 best.pth: {orig_ckpt}")

	if args.pruned_subdir:
		pruned_dir = os.path.join(args.checkpoint_dir, args.pruned_subdir)
	else:
		pruned_dir = find_latest_pruned_dir(args.checkpoint_dir)
	if pruned_dir is None:
		raise FileNotFoundError(f"未发现剪枝子目录 pruned_channel_* 于: {args.checkpoint_dir}")
	pruned_ckpt = os.path.join(pruned_dir, "student_best.pth")
	if not os.path.isfile(pruned_ckpt):
		raise FileNotFoundError(f"未找到剪枝后的 student_best.pth: {pruned_ckpt}")
	# 若存在 scripted/pickled 导出，优先直接加载，避免因结构变化造成的 shape mismatch
	script_path = os.path.join(pruned_dir, "student_scripted.pt")
	pickle_path = os.path.join(pruned_dir, "student_model.pth")

	# data
	val_tf = build_val_transform(224)
	ds = Cifar100DualTest(args.dataset_root, transform=val_tf)
	loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

	# load and eval
	orig_model, orig_cfg = load_model_from_ckpt(orig_ckpt, device)
	orig_metrics = evaluate(orig_model, loader, device)
	orig_meta = {"params": count_params(orig_model), "fwd_ms": measure_forward_ms(orig_model, device, 224)}

	# 加载剪枝后模型
	if os.path.isfile(script_path):
		pruned_model = torch.jit.load(script_path, map_location="cpu").to(device).eval()
		pruned_cfg = {}
	elif os.path.isfile(pickle_path):
		pruned_model = torch.load(pickle_path, map_location="cpu").to(device).eval()
		pruned_cfg = {}
	else:
		pruned_model, pruned_cfg = load_model_from_ckpt(pruned_ckpt, device)
	pruned_metrics = evaluate(pruned_model, loader, device)
	prune_meta_path = os.path.join(pruned_dir, "prune_meta.json")
	pruned_meta = {
		"params": count_params(pruned_model),
		"fwd_ms": measure_forward_ms(pruned_model, device, 224),
	}
	if os.path.isfile(prune_meta_path):
		with open(prune_meta_path, "r", encoding="utf-8") as f:
			pruned_meta.update(json.load(f))

	report = {
		"base_dir": args.checkpoint_dir,
		"pruned_dir": pruned_dir,
		"dataset_root": args.dataset_root,
		"original": {
			"ckpt": orig_ckpt,
			"config": orig_cfg,
			"metrics": orig_metrics,
			"meta": orig_meta,
		},
		"pruned": {
			"ckpt": pruned_ckpt,
			"config": pruned_cfg,
			"metrics": pruned_metrics,
			"meta": pruned_meta,
		},
		"delta": {
			"f_top1": pruned_metrics["f_top1"] - orig_metrics["f_top1"],
			"c_top1": pruned_metrics["c_top1"] - orig_metrics["c_top1"],
			"params": pruned_meta["params"] - orig_meta["params"],
			"fwd_ms": pruned_meta["fwd_ms"] - orig_meta["fwd_ms"],
		},
	}

	os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
	with open(args.output_path, "w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)
	print(f"[OK] saved compare to: {args.output_path}")


if __name__ == "__main__":
	main()


