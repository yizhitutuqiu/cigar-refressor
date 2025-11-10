from __future__ import annotations

from typing import Dict, Any

import os
import json
import random
from datetime import datetime

import numpy as np  # type: ignore[import-not-found]
from PIL import Image  # type: ignore[import-not-found]
from torchvision import transforms


class WithProbability:
	"""Wrap a torchvision transform to apply with probability p."""

	def __init__(self, t, p: float) -> None:
		self.t = t
		self.p = max(0.0, min(1.0, float(p)))

	def __call__(self, x):
		if self.p <= 0.0:
			return x
		if random.random() < self.p:
			return self.t(x)
		return x


def build_train_transform(aug: Dict[str, Any] | None, image_size: int = 224) -> transforms.Compose:
	"""Build train-time transform pipeline with configurable augmentations.

	Supported options in `aug` (all optional, p=0 disables):
	- use_val_preprocess: bool, true 时训练直接使用验证集的确定性预处理（无随机增强）
	- hflip_p: float in [0,1]
	- rotate_p: float; rotate_deg: float (max degrees, symmetric)
	- affine_p: float; scale_min: float; scale_max: float; translate_frac: float (0..0.45)
	- erase_p: float; erase_scale: [min,max] area ratio; erase_ratio: [min,max] aspect

	Defaults are conservative and safe.
	"""
	aug = aug or {}
	# deterministic path
	if bool(aug.get("use_val_preprocess", False)):
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		return transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(image_size),
			transforms.ToTensor(),
			transforms.Normalize(mean, std),
		])
	# defaults
	crop_scale_min = float(aug.get("crop_scale_min", 0.90))
	crop_scale_max = float(aug.get("crop_scale_max", 1.00))
	hflip_p = float(aug.get("hflip_p", 0.5))
	rotate_p = float(aug.get("rotate_p", 0.1))
	rotate_deg = float(aug.get("rotate_deg", 8.0))
	affine_p = float(aug.get("affine_p", 0.1))
	scale_min = float(aug.get("scale_min", 0.97))
	scale_max = float(aug.get("scale_max", 1.03))
	translate_frac = float(aug.get("translate_frac", 0.03))  # per-dim fraction of image size
	erase_p = float(aug.get("erase_p", 0.01))
	erase_scale = aug.get("erase_scale", [0.01, 0.03])
	erase_ratio = aug.get("erase_ratio", [0.3, 3.3])

	# clamp sensible bounds
	crop_scale_min = max(0.5, min(crop_scale_min, 1.0))
	crop_scale_max = max(crop_scale_min, min(crop_scale_max, 1.0))
	scale_min = max(0.5, min(scale_min, 1.5))
	scale_max = max(scale_min, min(scale_max, 1.5))
	translate_frac = max(0.0, min(translate_frac, 0.45))
	rotate_deg = max(0.0, min(rotate_deg, 45.0))

	ops = []
	# base resize/crop to match encoder input
	ops.append(transforms.RandomResizedCrop(image_size, scale=(crop_scale_min, crop_scale_max)))

	# flips
	ops.append(WithProbability(transforms.RandomHorizontalFlip(p=1.0), p=hflip_p))

	# affine (rotation + scale + translate)
	if rotate_deg > 0.0 or translate_frac > 0.0 or (scale_min != 1.0 or scale_max != 1.0):
		ops.append(WithProbability(
			transforms.RandomAffine(
				degrees=rotate_deg,
				translate=(translate_frac, translate_frac) if translate_frac > 0 else None,
				scale=(scale_min, scale_max) if (scale_min != 1.0 or scale_max != 1.0) else None,
			),
			p=affine_p,
		))

	# to tensor first for erasing
	ops.append(transforms.ToTensor())

	# random erasing as small mask occlusion
	if erase_p > 0.0:
		ops.append(transforms.RandomErasing(p=erase_p, scale=tuple(erase_scale), ratio=tuple(erase_ratio)))

	# normalize (ImageNet stats)
	ops.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

	return transforms.Compose(ops)


def _load_random_image(dataset_root: str, split: str = "test") -> Image.Image:
	import pickle
	with open(os.path.join(dataset_root, split), "rb") as f:
		data = pickle.load(f, encoding="latin1")
	arr = data["data"]
	idx = random.randrange(len(arr))
	row = arr[idx]
	img = np.reshape(row, (3, 32, 32))
	img = np.transpose(img, (1, 2, 0))
	return Image.fromarray(img.astype(np.uint8))


def _tensor_to_pil(t) -> Image.Image:
	# inverse of Normalize ImageNet stats
	mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
	std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
	arr = t.detach().cpu().numpy()
	arr = (arr * std + mean).clip(0.0, 1.0)
	arr = (arr * 255.0).round().astype(np.uint8)
	arr = np.transpose(arr, (1, 2, 0))
	return Image.fromarray(arr)


def _make_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
	left = left.convert("RGB")
	right = right.convert("RGB")
	h = max(left.height, right.height)
	canvas = Image.new("RGB", (left.width + right.width, h), (0, 0, 0))
	canvas.paste(left, (0, 0))
	canvas.paste(right, (left.width, 0))
	return canvas


def main():
	import argparse

	# resolve project root for defaults
	PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
	default_config = "./cifar_regressor/config/hierarchical_default.json"
	default_dataset = "./cifar-100-python"
	default_out = "./cifar_regressor/demo/aug_demo"

	parser = argparse.ArgumentParser(description="Augmentation demo: apply train-time aug to a random CIFAR-100 image")
	parser.add_argument("--config", type=str, default=default_config, help="包含 aug 段的配置文件路径")
	parser.add_argument("--dataset_root", type=str, default=default_dataset)
	parser.add_argument("--split", type=str, default="test", choices=["train", "test"]) 
	parser.add_argument("--output_dir", type=str, default=default_out)
	args = parser.parse_args()

	with open(args.config, "r", encoding="utf-8") as f:
		cfg = json.load(f)
	aug_cfg = cfg.get("aug", {})

	# build transforms
	train_tf = build_train_transform(aug_cfg, image_size=224)
	# load a random image
	img = _load_random_image(args.dataset_root, split=args.split)
	img_resized = img.resize((224, 224), Image.BILINEAR)
	aug_tensor = train_tf(img)
	aug_img = _tensor_to_pil(aug_tensor)

	os.makedirs(args.output_dir, exist_ok=True)
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	canvas = _make_side_by_side(img_resized, aug_img)
	out_path = os.path.join(args.output_dir, f"aug_demo_{stamp}.png")
	canvas.save(out_path)
	with open(os.path.join(args.output_dir, f"aug_demo_{stamp}.json"), "w", encoding="utf-8") as f:
		json.dump({"config": args.config, "dataset_root": args.dataset_root, "split": args.split, "aug": aug_cfg, "output": out_path}, f, ensure_ascii=False, indent=2)
	print(f"saved: {out_path}")


if __name__ == "__main__":
	main()


