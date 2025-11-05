from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Tuple

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

from cifar_regressor import CifarHierarchicalRegressor


class Cifar100DualTest(Dataset):
	def __init__(self, root_dir: str, transform: transforms.Compose | None = None) -> None:
		super().__init__()
		self.root_dir = root_dir
		self.transform = transform
		import pickle
		with open(os.path.join(root_dir, "test"), "rb") as f:
			data = pickle.load(f, encoding="latin1")
		with open(os.path.join(root_dir, "meta"), "rb") as f:
			meta = pickle.load(f, encoding="latin1")
		self.data = data["data"]
		self.coarse = data["coarse_labels"]
		self.fine = data["fine_labels"]
		self.coarse_label_names = meta.get("coarse_label_names", [str(i) for i in range(20)])
		self.fine_label_names = meta.get("fine_label_names", [str(i) for i in range(100)])

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
	coarse_conf = np.zeros((coarse_num, coarse_num), dtype=np.int64)
	coarse_total = np.zeros(coarse_num, dtype=np.int64)
	coarse_correct = np.zeros(coarse_num, dtype=np.int64)
	# fine
	fine_top1 = 0
	fine_top5 = 0
	fine_num = model.num_fine if hasattr(model, "num_fine") else 100
	fine_conf = np.zeros((fine_num, fine_num), dtype=np.int64)
	fine_total = np.zeros(fine_num, dtype=np.int64)
	fine_correct = np.zeros(fine_num, dtype=np.int64)

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
		for t, p in zip(coarse_t.tolist(), c_pred1.tolist()):
			coarse_conf[t, p] += 1
			coarse_total[t] += 1
			if t == p:
				coarse_correct[t] += 1

		# fine
		f_pred1 = torch.argmax(f_probs, dim=1)
		fine_top1 += (f_pred1 == fine_t).sum().item()
		_, f_pred5 = torch.topk(f_probs, k=min(5, f_probs.size(1)), dim=1)
		fine_top5 += (f_pred5 == fine_t.view(-1, 1)).any(dim=1).sum().item()
		for t, p in zip(fine_t.tolist(), f_pred1.tolist()):
			fine_conf[t, p] += 1
			fine_total[t] += 1
			if t == p:
				fine_correct[t] += 1

		total += images.size(0)

	coarse_per_acc = (coarse_correct / np.maximum(1, coarse_total)).tolist()
	fine_per_acc = (fine_correct / np.maximum(1, fine_total)).tolist()
	return {
		"total": int(total),
		"coarse": {
			"top1": float(coarse_top1 / max(1, total)),
			"top5": float(coarse_top5 / max(1, total)),
			"per_class_acc": coarse_per_acc,
			"per_class_total": coarse_total.tolist(),
			"per_class_correct": coarse_correct.tolist(),
			"confusion_matrix": coarse_conf.tolist(),
		},
		"fine": {
			"top1": float(fine_top1 / max(1, total)),
			"top5": float(fine_top5 / max(1, total)),
			"per_class_acc": fine_per_acc,
			"per_class_total": fine_total.tolist(),
			"per_class_correct": fine_correct.tolist(),
			"confusion_matrix": fine_conf.tolist(),
		},
	}


def build_model_from_checkpoint(ckpt_path: str, device: torch.device) -> CifarHierarchicalRegressor:
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
	model.to(device)
	return model


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate hierarchical (coarse+fine) model on CIFAR-100 test set")
	parser.add_argument("--dataset_root", type=str, default="/data/litengmo/ml-test/cifar-100-python")
	parser.add_argument("--checkpoint_dir", type=str, default="", help="checkpoint 子目录路径，包含 best.pth")
	parser.add_argument("--checkpoint_path", type=str, default="", help="直接指定 best.pth 的绝对路径（优先）")
	parser.add_argument("--output_root", type=str, default="/data/litengmo/ml-test/cifar_regressor/test")
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--gpu", type=int, default=None, help="选择使用的 GPU 编号，如 0、1、7；不填则按 --device")
	args = parser.parse_args()

	# Resolve checkpoint path
	ckpt_path = args.checkpoint_path
	if not ckpt_path:
		if not args.checkpoint_dir:
			raise ValueError("必须提供 --checkpoint_path 或 --checkpoint_dir 其中之一")
		ckpt_path = os.path.join(args.checkpoint_dir, "best.pth")
	if not os.path.isfile(ckpt_path):
		raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")

	# Prepare IO
	os.makedirs(args.output_root, exist_ok=True)
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_dir = os.path.join(args.output_root, stamp)
	os.makedirs(out_dir, exist_ok=True)

	# Build dataset/loader
	val_tf = build_val_transform()
	dualds = Cifar100DualTest(args.dataset_root, transform=val_tf)
	loader = DataLoader(
		dualds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
	)

	# Device
	device_str = args.device
	if args.gpu is not None:
		device_str = f"cuda:{args.gpu}"
	device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")

	# Build model and evaluate
	model = build_model_from_checkpoint(ckpt_path, device)
	metrics = evaluate(model, loader, device=device)

	# Report
	report = {
		"checkpoint_path": ckpt_path,
		"checkpoint_dir": os.path.dirname(ckpt_path),
		"dataset_root": args.dataset_root,
		"num_samples": metrics["total"],
		"coarse": {
			"top1": metrics["coarse"]["top1"],
			"top5": metrics["coarse"]["top5"],
			"per_class_acc": metrics["coarse"]["per_class_acc"],
		},
		"fine": {
			"top1": metrics["fine"]["top1"],
			"top5": metrics["fine"]["top5"],
			"per_class_acc": metrics["fine"]["per_class_acc"],
		},
	}

	print("=== CIFAR-100 Hierarchical Test Evaluation ===")
	print(f"checkpoint: {ckpt_path}")
	print(f"coarse - top1: {report['coarse']['top1']:.4f} | top5: {report['coarse']['top5']:.4f}")
	print(f"fine   - top1: {report['fine']['top1']:.4f} | top5: {report['fine']['top5']:.4f}")

	# Save outputs
	with open(os.path.join(out_dir, "eval_report_hier.json"), "w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)
	with open(os.path.join(out_dir, "coarse_confusion.json"), "w", encoding="utf-8") as f:
		json.dump({"confusion_matrix": metrics["coarse"]["confusion_matrix"]}, f, ensure_ascii=False)
	with open(os.path.join(out_dir, "fine_confusion.json"), "w", encoding="utf-8") as f:
		json.dump({"confusion_matrix": metrics["fine"]["confusion_matrix"]}, f, ensure_ascii=False)

	print(f"saved evaluation to: {out_dir}")


if __name__ == "__main__":
	main()


