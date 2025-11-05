from __future__ import annotations

from typing import Dict, Any

import random
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
	- hflip_p: float in [0,1]
	- rotate_p: float; rotate_deg: float (max degrees, symmetric)
	- affine_p: float; scale_min: float; scale_max: float; translate_frac: float (0..0.45)
	- erase_p: float; erase_scale: [min,max] area ratio; erase_ratio: [min,max] aspect

	Defaults are conservative and safe.
	"""
	aug = aug or {}
	# defaults
	hflip_p = float(aug.get("hflip_p", 0.5))
	rotate_p = float(aug.get("rotate_p", 0.2))
	rotate_deg = float(aug.get("rotate_deg", 15.0))
	affine_p = float(aug.get("affine_p", 0.2))
	scale_min = float(aug.get("scale_min", 0.9))
	scale_max = float(aug.get("scale_max", 1.1))
	translate_frac = float(aug.get("translate_frac", 0.1))  # per-dim fraction of image size
	erase_p = float(aug.get("erase_p", 0.0))
	erase_scale = aug.get("erase_scale", [0.02, 0.1])
	erase_ratio = aug.get("erase_ratio", [0.3, 3.3])

	# clamp sensible bounds
	scale_min = max(0.5, min(scale_min, 1.5))
	scale_max = max(scale_min, min(scale_max, 1.5))
	translate_frac = max(0.0, min(translate_frac, 0.45))
	rotate_deg = max(0.0, min(rotate_deg, 45.0))

	ops = []
	# base resize/crop to match encoder input
	ops.append(transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)))

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


