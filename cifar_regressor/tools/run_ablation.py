from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch  # type: ignore[import-not-found]

# Ensure project root on sys.path for imports when run by absolute path
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from cifar_regressor import CifarHierarchicalRegressor  # type: ignore


def measure_model_meta(encoder_name: str, use_cbam: bool, use_film: bool, device: str) -> Dict:
	device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
	model = CifarHierarchicalRegressor(
		pretrained_backbone=False,  # meta 测试不需要下载预训练
		encoder_name=encoder_name,
		use_cbam=use_cbam,
		use_film=use_film,
	)
	model.eval().to(device_obj)
	param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
	# 前向时间（单次，含 5 次预热与 10 次平均）
	dummy = torch.randn(1, 3, 224, 224, device=device_obj)
	with torch.no_grad():
		for _ in range(5):
			_ = model(dummy)
		if device_obj.type == "cuda":
			torch.cuda.synchronize()
		t0 = time.perf_counter()
		for _ in range(10):
			_ = model(dummy)
		if device_obj.type == "cuda":
			torch.cuda.synchronize()
		t1 = time.perf_counter()
	forward_ms = (t1 - t0) / 10.0 * 1000.0
	return {
		"encoder_name": encoder_name,
		"use_cbam": use_cbam,
		"use_film": use_film,
		"param_count": int(param_count),
		"forward_ms_bs1_224": forward_ms,
	}


def build_config(
	dataset_root: str,
	checkpoint_dir: str,
	encoder_name: str,
	use_cbam: bool,
	use_film: bool,
) -> Dict:
	# 为大模型设置更保守的 batch_size（避免显存溢出）
	default_bs = 128
	enc_l = encoder_name.lower()
	if "vit_huge" in enc_l or "vit_h" in enc_l:
		default_bs = 16
	return {
		"dataset_root": dataset_root,
		"checkpoint_dir": checkpoint_dir,
		"encoder_name": encoder_name,
		"pretrained_backbone": True,
		"use_cbam": use_cbam,
		"num_coarse": 20,
		"num_fine": 100,
		"hidden_features": 256,
		"dropout_p": 0.1,
		"use_film": use_film,
		"film_hidden": 256,
		"film_use_probs": True,
		"batch_size": default_bs,
		"epochs": 20,
		"learning_rate": 3e-4,
		"backbone_lr_mult": 0.1,
		"weight_decay": 0.05,
		"num_workers": 4,
		"log_interval": 50,
		"val_split": 0.1,
		"seed": 42,
		"device": "cuda",
		"mixed_precision": True,
		"lambda_coarse": 0.2,
		"scheduler": {"type": "cosine", "eta_min": 1e-6},
		"aug": {
			"use_val_preprocess": True  # 关闭所有随机增强（按需求）
		},
	}


def run_experiment(
	encoder_name: str,
	use_cbam: bool,
	use_film: bool,
	gpu: int,
	dataset_root: str,
	ckpt_root: str,
	config_root: str,
) -> Tuple[int, str]:
	tag = f"{encoder_name.replace('/', '_')}" + (("_cbam" if use_cbam else "")) + (("_nofilm" if not use_film else ""))
	checkpoint_dir = os.path.join(ckpt_root, tag)
	os.makedirs(checkpoint_dir, exist_ok=True)
	# 记录 meta
	meta = measure_model_meta(encoder_name, use_cbam, use_film, device=f"cuda:{gpu}")
	meta_path = os.path.join(checkpoint_dir, "model_meta.json")
	with open(meta_path, "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)
	# 写入 config
	cfg = build_config(dataset_root, checkpoint_dir, encoder_name, use_cbam, use_film)
	os.makedirs(config_root, exist_ok=True)
	cfg_path = os.path.join(config_root, f"{tag}.json")
	with open(cfg_path, "w", encoding="utf-8") as f:
		json.dump(cfg, f, ensure_ascii=False, indent=2)
	# 启动训练（串行）
	cmd = [
		sys.executable,
		"./cifar_regressor/train/train_hierarchical.py",
		"--config",
		cfg_path,
		"--gpu",
		str(gpu),
	]
	print(f"[RUN] {' '.join(cmd)}")
	proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
	return proc.returncode, tag


def main():
	parser = argparse.ArgumentParser(description="Run ablation experiments sequentially on a single GPU")
	parser.add_argument("--gpu", type=int, default=6, help="GPU 编号，默认 6")
	parser.add_argument("--dataset_root", type=str, default="./cifar-100-python")
	parser.add_argument("--output_root", type=str, default="./cifar_regressor/checkpoints/ablation")
	parser.add_argument("--config_root", type=str, default="./cifar_regressor/config/ablation")
	args = parser.parse_args()

	os.makedirs(args.output_root, exist_ok=True)
	os.makedirs(args.config_root, exist_ok=True)

	# 组合：8 组（默认 use_film=True），再加 8 组（use_film=False）= 16 组
	base_set: List[Tuple[str, bool]] = [
		("resnet18", False),
		("resnet18", True),
		("resnet34", False),
		("resnet34", True),
		("resnet50", False),
		("resnet50", True),
		("vit_small_patch16_224", False),  # CBAM 对 ViT 无效，固定 False
		("vit_base_patch16_224", False),
		("vit_huge_patch14_224", False),
	]
	film_flags = [True, False]

	# 训练前检查：已存在的方案不再重复运行（以 ablation/<tag> 子目录是否存在为准）
	existing = {
		d for d in os.listdir(args.output_root) if os.path.isdir(os.path.join(args.output_root, d))
	}
	def make_tag(enc: str, cbam: bool, film: bool) -> str:
		return f"{enc.replace('/', '_')}{'_cbam' if cbam else ''}{'_nofilm' if not film else ''}"
	targets: List[Tuple[str, bool, bool]] = []
	for enc, cbam in base_set:
		for film in film_flags:
			tag = make_tag(enc, cbam, film)
			if tag in existing:
				print(f"[SKIP] already exists: {tag}")
				continue
			targets.append((enc, cbam, film))
	print(f"[INFO] total={len(base_set)*len(film_flags)}; existing={len(existing)}; to_run={len(targets)}")

	results: List[Dict] = []
	for encoder_name, use_cbam, use_film in targets:
		rc, tag = run_experiment(
			encoder_name=encoder_name,
			use_cbam=use_cbam,
			use_film=use_film,
			gpu=args.gpu,
			dataset_root=args.dataset_root,
			ckpt_root=args.output_root,
			config_root=args.config_root,
		)
		results.append({"tag": tag, "returncode": rc})
		# 简单的间隔，避免日志/缓存抖动
		time.sleep(2.0)

	# 保存 ablation 汇总
	summary_path = os.path.join(args.output_root, f"ablation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump({"results": results}, f, ensure_ascii=False, indent=2)
	print(f"[DONE] summary => {summary_path}")


if __name__ == "__main__":
	main()


