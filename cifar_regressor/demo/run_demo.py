from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Tuple

import numpy as np  # type: ignore[import-not-found]
from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]

import torch  # type: ignore[import-not-found]
from torchvision import transforms  # type: ignore[import-not-found]

# Make project importable when running by absolute path
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from cifar_regressor import CifarCoarseRegressor, CifarHierarchicalRegressor


def build_val_transform() -> transforms.Compose:
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	return transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	])


def load_cifar100_test_sample(dataset_root: str, index: int | None = None) -> Tuple[Image.Image, int, int, list[str], list[str]]:
	import pickle
	with open(os.path.join(dataset_root, "test"), "rb") as f:
		data_dict = pickle.load(f, encoding="latin1")
	with open(os.path.join(dataset_root, "meta"), "rb") as f:
		meta = pickle.load(f, encoding="latin1")

	data = data_dict["data"]
	coarse_labels = data_dict["coarse_labels"]
	fine_labels = data_dict.get("fine_labels", None)
	coarse_label_names = meta.get("coarse_label_names", [str(i) for i in range(20)])
	fine_label_names = meta.get("fine_label_names", [str(i) for i in range(100)])

	N = len(coarse_labels)
	if index is None:
		index = random.randrange(N)

	row = data[index]
	img = np.reshape(row, (3, 32, 32))
	img = np.transpose(img, (1, 2, 0))
	pil_img = Image.fromarray(img.astype(np.uint8))
	coarse_label = int(coarse_labels[index])
	fine_label = int(fine_labels[index]) if fine_labels is not None else -1
	return pil_img, coarse_label, fine_label, coarse_label_names, fine_label_names


def annotate_image(img: Image.Image, text: str) -> Image.Image:
	img = img.copy().convert("RGB")
	draw = ImageDraw.Draw(img)
	# Try to load a default font; fall back to bitmap
	try:
		font = ImageFont.truetype("DejaVuSans.ttf", 14)
	except Exception:
		font = ImageFont.load_default()
	# Draw background rectangle for readability
	margin = 4
	w, h = draw.textbbox((0, 0), text, font=font)[2:]
	box = (0, img.height - h - 2 * margin, w + 2 * margin, img.height)
	draw.rectangle(box, fill=(0, 0, 0))
	draw.text((margin, img.height - h - margin), text, fill=(255, 255, 255), font=font)
	return img


def main() -> None:
	parser = argparse.ArgumentParser(description="CIFAR-100 coarse demo: random test sample inference")
	parser.add_argument("--dataset_root", type=str, default="/data/litengmo/ml-test/cifar-100-python")
	parser.add_argument("--checkpoint", type=str, default="")
	parser.add_argument("--output_dir", type=str, default="/data/litengmo/ml-test/cifar_regressor/demo/outputs")
	parser.add_argument("--device", type=str, default="cuda")
	args = parser.parse_args()

	# Prepare I/O
	os.makedirs(args.output_dir, exist_ok=True)
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	case_dir = os.path.join(args.output_dir, stamp)
	os.makedirs(case_dir, exist_ok=True)

	# Load sample
	pil_img, gt_coarse, gt_fine, coarse_label_names, fine_label_names = load_cifar100_test_sample(args.dataset_root)
	val_tf = build_val_transform()
	img_tensor = val_tf(pil_img).unsqueeze(0)

	# Build model (prefer reconstruct from checkpoint config if available)
	device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
	# Default assume coarse-only
	model_type = "coarse"
	model_cfg = {
		"num_classes": 20,
		"hidden_features": 256,
		"dropout_p": 0.1,
		"use_cbam": False,
		"encoder_name": "resnet18",
	}
	ckpt = None
	if args.checkpoint and os.path.isfile(args.checkpoint):
		ckpt = torch.load(args.checkpoint, map_location="cpu")
		if isinstance(ckpt, dict) and "config" in ckpt:
			cfg = ckpt["config"] or {}
			# decide hierarchical vs coarse by presence of num_fine
			if "num_fine" in cfg or "use_film" in cfg:
				model_type = "hier"
			model_cfg.update({
				"num_classes": int(cfg.get("num_classes", model_cfg["num_classes"])),
				"hidden_features": int(cfg.get("hidden_features", model_cfg["hidden_features"])),
				"dropout_p": float(cfg.get("dropout_p", model_cfg["dropout_p"])),
				"use_cbam": bool(cfg.get("use_cbam", model_cfg["use_cbam"])),
				"encoder_name": str(cfg.get("encoder_name", model_cfg["encoder_name"])),
			})

	if model_type == "hier":
		# Build hierarchical model
		cfg = ckpt.get("config", {}) if ckpt is not None else {}
		model = CifarHierarchicalRegressor(
			pretrained_backbone=False,
			encoder_name=str(cfg.get("encoder_name", model_cfg["encoder_name"])),
			use_cbam=bool(cfg.get("use_cbam", False)),
			num_coarse=int(cfg.get("num_coarse", 20)),
			num_fine=int(cfg.get("num_fine", 100)),
			hidden_features=int(cfg.get("hidden_features", model_cfg["hidden_features"])),
			dropout_p=float(cfg.get("dropout_p", model_cfg["dropout_p"])),
			use_film=bool(cfg.get("use_film", True)),
			film_hidden=int(cfg.get("film_hidden", 256)),
			film_use_probs=bool(cfg.get("film_use_probs", True)),
		)
	else:
		# Build coarse-only model
		model = CifarCoarseRegressor(
			pretrained_backbone=False,  # weights will come from checkpoint if provided
			num_classes=model_cfg["num_classes"],
			hidden_features=model_cfg["hidden_features"],
			dropout_p=model_cfg["dropout_p"],
			use_cbam=model_cfg["use_cbam"],
			encoder_name=model_cfg["encoder_name"],
		)

	model.eval().to(device)
	if ckpt is not None:
		state_dict = ckpt.get("model", ckpt)
		model.load_state_dict(state_dict, strict=False)

	# Inference
	with torch.no_grad():
		img_tensor = img_tensor.to(device)
		if model_type == "hier":
			out = model(img_tensor)
			c_probs = out["coarse_probs"]
			f_probs = out["fine_probs"]
			c_prob, c_pred = torch.max(c_probs, dim=1)
			f_prob, f_pred = torch.max(f_probs, dim=1)
			c_idx = int(c_pred.item())
			f_idx = int(f_pred.item())
			c_p = float(c_prob.item())
			f_p = float(f_prob.item())
			gt_c_name = coarse_label_names[gt_coarse] if 0 <= gt_coarse < len(coarse_label_names) else str(gt_coarse)
			gt_f_name = fine_label_names[gt_fine] if 0 <= gt_fine < len(fine_label_names) else str(gt_fine)
			pred_c_name = coarse_label_names[c_idx] if 0 <= c_idx < len(coarse_label_names) else str(c_idx)
			pred_f_name = fine_label_names[f_idx] if 0 <= f_idx < len(fine_label_names) else str(f_idx)
			print(f"Coarse: GT {gt_coarse} ({gt_c_name}) | Pred {c_idx} ({pred_c_name}) p={c_p:.4f}")
			print(f"Fine  : GT {gt_fine} ({gt_f_name}) | Pred {f_idx} ({pred_f_name}) p={f_p:.4f}")
			original_path = os.path.join(case_dir, "original.png")
			pil_img.save(original_path)
			text = f"c:{pred_c_name}({c_p:.2f})|gt:{gt_c_name}; f:{pred_f_name}({f_p:.2f})|gt:{gt_f_name}"
			annotated = annotate_image(pil_img.resize((256, 256)), text)
			annotated_path = os.path.join(case_dir, "annotated.png")
			annotated.save(annotated_path)
			with open(os.path.join(case_dir, "result.json"), "w", encoding="utf-8") as f:
				json.dump({
					"model_type": "hierarchical",
					"coarse": {"pred_index": c_idx, "pred_name": pred_c_name, "pred_prob": c_p, "gt_index": gt_coarse, "gt_name": gt_c_name},
					"fine": {"pred_index": f_idx, "pred_name": pred_f_name, "pred_prob": f_p, "gt_index": gt_fine, "gt_name": gt_f_name},
					"original": original_path,
					"annotated": annotated_path,
				}, f, ensure_ascii=False, indent=2)
		else:
			logits, probs = model(img_tensor)
			prob, pred = torch.max(probs, dim=1)
			pred_idx = int(pred.item())
			pred_prob = float(prob.item())
			gt_name = coarse_label_names[gt_coarse] if 0 <= gt_coarse < len(coarse_label_names) else str(gt_coarse)
			pred_name = coarse_label_names[pred_idx] if 0 <= pred_idx < len(coarse_label_names) else str(pred_idx)
			print(f"GT: {gt_coarse} ({gt_name}) | Pred: {pred_idx} ({pred_name}) prob={pred_prob:.4f}")
			original_path = os.path.join(case_dir, "original.png")
			pil_img.save(original_path)
			annotated = annotate_image(pil_img.resize((256, 256)), f"pred={pred_name} ({pred_prob:.2f}) | gt={gt_name}")
			annotated_path = os.path.join(case_dir, "annotated.png")
			annotated.save(annotated_path)
			with open(os.path.join(case_dir, "result.json"), "w", encoding="utf-8") as f:
				json.dump({
					"model_type": "coarse",
					"pred_index": pred_idx,
					"pred_name": pred_name,
					"pred_prob": pred_prob,
					"gt_index": gt_coarse,
					"gt_name": gt_name,
					"original": original_path,
					"annotated": annotated_path,
				}, f, ensure_ascii=False, indent=2)

	print(f"saved to: {case_dir}")


if __name__ == "__main__":
	main()


