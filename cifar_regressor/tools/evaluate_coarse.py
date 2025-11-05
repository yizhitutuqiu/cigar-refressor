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

from cifar_regressor import CifarCoarseRegressor


class Cifar100CoarseTest(Dataset):
	def __init__(self, root_dir: str, transform: transforms.Compose | None = None) -> None:
		super().__init__()
		self.root_dir = root_dir
		self.transform = transform
		import pickle
		with open(os.path.join(root_dir, "test"), "rb") as f:
			data_dict = pickle.load(f, encoding="latin1")
		with open(os.path.join(root_dir, "meta"), "rb") as f:
			meta = pickle.load(f, encoding="latin1")
		self.data = data_dict["data"]
		self.labels = data_dict["coarse_labels"]
		self.coarse_label_names = meta.get("coarse_label_names", [str(i) for i in range(20)])

	def __len__(self) -> int:
		return len(self.labels)

	def __getitem__(self, index: int):
		row = self.data[index]
		img = np.reshape(row, (3, 32, 32))
		img = np.transpose(img, (1, 2, 0))
		img = Image.fromarray(img.astype(np.uint8))
		if self.transform is not None:
			img = self.transform(img)
		label = int(self.labels[index])
		return img, label


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
def evaluate(model: CifarCoarseRegressor, loader: DataLoader, device: torch.device, num_classes: int = 20) -> Dict:
	model.eval()
	correct_top1 = 0
	correct_top5 = 0
	total = 0
	conf = np.zeros((num_classes, num_classes), dtype=np.int64)
	per_class_total = np.zeros(num_classes, dtype=np.int64)
	per_class_correct = np.zeros(num_classes, dtype=np.int64)

	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		logits, probs = model(images)
		# top-1
		pred1 = torch.argmax(probs, dim=1)
		correct_top1 += (pred1 == targets).sum().item()
		# top-5
		_, pred5 = torch.topk(probs, k=min(5, probs.size(1)), dim=1)
		correct_top5 += (pred5 == targets.view(-1, 1)).any(dim=1).sum().item()
		# confusion
		for t, p in zip(targets.tolist(), pred1.tolist()):
			conf[t, p] += 1
			per_class_total[t] += 1
			if t == p:
				per_class_correct[t] += 1
		total += targets.size(0)

	per_class_acc = (per_class_correct / np.maximum(1, per_class_total)).tolist()
	metrics = {
		"total": int(total),
		"top1": float(correct_top1 / max(1, total)),
		"top5": float(correct_top5 / max(1, total)),
		"per_class_total": per_class_total.tolist(),
		"per_class_correct": per_class_correct.tolist(),
		"per_class_acc": per_class_acc,
		"confusion_matrix": conf.tolist(),
	}
	return metrics


def build_model_from_checkpoint(ckpt_path: str, device: torch.device) -> CifarCoarseRegressor:
	ckpt = torch.load(ckpt_path, map_location="cpu")
	cfg = ckpt.get("config", {}) or {}
	model = CifarCoarseRegressor(
		pretrained_backbone=False,
		num_classes=int(cfg.get("num_classes", 20)),
		hidden_features=int(cfg.get("hidden_features", 256)),
		dropout_p=float(cfg.get("dropout_p", 0.1)),
		use_cbam=bool(cfg.get("use_cbam", False)),
		encoder_name=str(cfg.get("encoder_name", "resnet18")),
	)
	state_dict = ckpt.get("model", ckpt)
	model.load_state_dict(state_dict, strict=False)
	model.to(device)
	return model


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate CIFAR-100 coarse on test set using a checkpoint")
	parser.add_argument("--dataset_root", type=str, default="./cifar-100-python")
	parser.add_argument("--checkpoint_dir", type=str, default="", help="checkpoint 子目录路径，包含 best.pth")
	parser.add_argument("--checkpoint_path", type=str, default="", help="直接指定 best.pth 的绝对路径（优先）")
	parser.add_argument("--output_root", type=str, default="./cifar_regressor/test")
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
	test_ds = Cifar100CoarseTest(args.dataset_root, transform=val_tf)
	test_loader = DataLoader(
		test_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
	)

	# Build model and evaluate
	# 设备选择：--gpu 优先，其次 --device
	device_str = args.device
	if args.gpu is not None:
		device_str = f"cuda:{args.gpu}"
	device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")
	model = build_model_from_checkpoint(ckpt_path, device)
	metrics = evaluate(model, test_loader, device=device, num_classes=20)

	# Compose report
	report = {
		"checkpoint_path": ckpt_path,
		"checkpoint_dir": os.path.dirname(ckpt_path),
		"dataset_root": args.dataset_root,
		"num_samples": metrics["total"],
		"top1": metrics["top1"],
		"top5": metrics["top5"],
		"per_class_acc": metrics["per_class_acc"],
		"per_class_total": metrics["per_class_total"],
		"per_class_correct": metrics["per_class_correct"],
	}

	# Print concise summary
	print("=== CIFAR-100 Coarse Test Evaluation ===")
	print(f"checkpoint: {ckpt_path}")
	print(f"top1: {report['top1']:.4f} | top5: {report['top5']:.4f} | samples: {report['num_samples']}")

	# Save outputs
	with open(os.path.join(out_dir, "eval_report.json"), "w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)
	with open(os.path.join(out_dir, "confusion_matrix.json"), "w", encoding="utf-8") as f:
		json.dump({"confusion_matrix": metrics["confusion_matrix"]}, f, ensure_ascii=False)

	print(f"saved evaluation to: {out_dir}")


if __name__ == "__main__":
	main()


