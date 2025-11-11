from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, Tuple

import numpy as np  # type: ignore[import-not-found]
from PIL import Image  # type: ignore[import-not-found]

import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]
from torch.utils.data import Dataset, DataLoader  # type: ignore[import-not-found]
from torchvision import transforms  # type: ignore[import-not-found]

# Ensure project root on sys.path for absolute-path execution
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from cifar_regressor import CifarHierarchicalRegressor  # type: ignore
from cifar_regressor.utils import build_train_transform  # type: ignore
import torch.backends.cudnn as cudnn


class Cifar100DualLabels(Dataset):
	def __init__(self, root_dir: str, split: str, transform: transforms.Compose | None = None) -> None:
		if split not in {"train", "test"}:
			raise ValueError("split 必须是 'train' 或 'test'")
		self.root_dir = root_dir
		self.split = split
		self.transform = transform
		import pickle
		with open(os.path.join(root_dir, split), "rb") as f:
			data = pickle.load(f, encoding="latin1")
		self.data = data["data"]
		self.fine_labels = data["fine_labels"]
		self.coarse_labels = data["coarse_labels"]

	def __len__(self) -> int:
		return len(self.fine_labels)

	def __getitem__(self, index: int):
		row = self.data[index]
		img = np.reshape(row, (3, 32, 32))
		img = np.transpose(img, (1, 2, 0))
		img = Image.fromarray(img.astype(np.uint8))
		if self.transform is not None:
			img = self.transform(img)
		fine = int(self.fine_labels[index])
		coarse = int(self.coarse_labels[index])
		return img, coarse, fine


def build_val_transform(image_size: int = 224) -> transforms.Compose:
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	return transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(image_size),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	])


def load_teacher(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
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


def count_params(model: nn.Module) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_forward_ms(model: nn.Module, device: torch.device, image_size: int = 224) -> float:
	dummy = torch.randn(1, 3, image_size, image_size, device=device)
	with torch.no_grad():
		for _ in range(5):
			_ = model(dummy)
		if device.type == "cuda":
			torch.cuda.synchronize()
		# prefer CUDA events for timing
		if device.type == "cuda":
			start = torch.cuda.Event(enable_timing=True)
			end = torch.cuda.Event(enable_timing=True)
			start.record()
			for _ in range(10):
				_ = model(dummy)
			end.record()
			torch.cuda.synchronize()
			ms = start.elapsed_time(end) / 10.0  # milliseconds
			return float(ms)
		else:
			t0 = time.perf_counter()
			for _ in range(10):
				_ = model(dummy)
			t1 = time.perf_counter()
			return (t1 - t0) / 10.0 * 1000.0


def prune_backbone_resnet(student: nn.Module, example_inputs: torch.Tensor, ch_sparsity: float = 0.3) -> nn.Module:
	try:
		import torch_pruning as tp  # type: ignore
	except Exception as exc:
		raise RuntimeError("需要安装 torch-pruning 才能进行结构化通道剪枝：pip install torch-pruning") from exc

	# 仅支持 ResNet 分支
	if not hasattr(student, "encoder_type") or student.encoder_type != "resnet":
		raise RuntimeError("当前脚本仅支持对 ResNet 编码器进行通道剪枝（encoder_name=resnet18/34/50）")

	# 选择重要性策略
	strategy = tp.importance.MagnitudeImportance(p=1)  # L1
	ignored_layers = set()
	# 保护头部与调制模块（不剪）
	if hasattr(student, "coarse_head") and isinstance(getattr(student, "coarse_head"), nn.Module):
		ignored_layers.add(student.coarse_head)
	if hasattr(student, "fine_head") and isinstance(getattr(student, "fine_head"), nn.Module):
		ignored_layers.add(student.fine_head)
	if hasattr(student, "cbam") and isinstance(getattr(student, "cbam"), nn.Module):
		ignored_layers.add(student.cbam)
	if hasattr(student, "film") and isinstance(getattr(student, "film"), nn.Module):
		ignored_layers.add(student.film)
	# 构建 pruner
	pruner = tp.pruner.MagnitudePruner(
		model=student,
		example_inputs=example_inputs,
		importance=strategy,
		ch_sparsity=ch_sparsity,
		ignored_layers=ignored_layers,
		global_pruning=True,
	)
	# 执行剪枝
	pruner.step()
	return student


def kd_train(
	teacher: nn.Module,
	student: nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader,
	device: torch.device,
	epochs: int = 5,
	lr: float = 3e-4,
	weight_decay: float = 5e-2,
	alpha: float = 0.9,
	temperature: float = 4.0,
) -> Dict:
	optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
	ce = nn.CrossEntropyLoss()

	def kd_loss_fn(t_logits: torch.Tensor, s_logits: torch.Tensor) -> torch.Tensor:
		T = temperature
		t_prob = (t_logits / T).softmax(dim=1)
		log_s = (s_logits / T).log_softmax(dim=1)
		return nn.KLDivLoss(reduction="batchmean")(log_s, t_prob) * (T * T)

	best = {"val_fine_top1": -1.0, "epoch": 0}
	history = []
	for epoch in range(1, epochs + 1):
		student.train()
		running = {"loss": 0.0, "acc_fine": 0.0, "acc_coarse": 0.0, "count": 0}
		for images, coarse, fine in train_loader:
			images = images.to(device, non_blocking=True)
			coarse = coarse.to(device, non_blocking=True)
			fine = fine.to(device, non_blocking=True)
			with torch.no_grad():
				t_out = teacher(images)
			s_out = student(images)
			loss_f = ce(s_out["fine_logits"], fine)
			loss_c = ce(s_out["coarse_logits"], coarse)
			loss_kd_f = kd_loss_fn(t_out["fine_logits"], s_out["fine_logits"])
			loss_kd_c = kd_loss_fn(t_out["coarse_logits"], s_out["coarse_logits"])
			loss = alpha * (loss_kd_f + loss_kd_c) + (1 - alpha) * (loss_f + 0.2 * loss_c)

			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

			# metrics
			with torch.no_grad():
				acc_f = (s_out["fine_logits"].argmax(1) == fine).float().mean().item()
				acc_c = (s_out["coarse_logits"].argmax(1) == coarse).float().mean().item()
			bs = images.size(0)
			running["loss"] += float(loss.item()) * bs
			running["acc_fine"] += acc_f * bs
			running["acc_coarse"] += acc_c * bs
			running["count"] += bs

		# eval
		student.eval()
		val = {"acc_fine": 0.0, "acc_coarse": 0.0, "count": 0}
		with torch.no_grad():
			for images, coarse, fine in val_loader:
				images = images.to(device, non_blocking=True)
				coarse = coarse.to(device, non_blocking=True)
				fine = fine.to(device, non_blocking=True)
				out = student(images)
				acc_f = (out["fine_logits"].argmax(1) == fine).float().mean().item()
				acc_c = (out["coarse_logits"].argmax(1) == coarse).float().mean().item()
				bs = images.size(0)
				val["acc_fine"] += acc_f * bs
				val["acc_coarse"] += acc_c * bs
				val["count"] += bs
		val_fine_top1 = val["acc_fine"] / max(1, val["count"])
		history.append({"epoch": epoch, "train_loss": running["loss"] / running["count"], "val_fine_top1": val_fine_top1})
		if val_fine_top1 > best["val_fine_top1"]:
			best["val_fine_top1"] = val_fine_top1
			best["epoch"] = epoch
	return {"best": best, "history": history}


def main():
	parser = argparse.ArgumentParser(description="Structured channel pruning + KD (ResNet only)")
	parser.add_argument("--checkpoint_dir", type=str, required=True, help="原模型目录，包含 best.pth")
	parser.add_argument("--dataset_root", type=str, default="./cifar-100-python")
	parser.add_argument("--output_subdir", type=str, default="pruned_channel")
	parser.add_argument("--prune_ratio", type=float, default=0.3, help="通道剪枝比例（全局）")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--gpu", type=int, default=6)
	parser.add_argument("--alpha", type=float, default=0.9, help="KD loss 权重")
	parser.add_argument("--temperature", type=float, default=4.0)
	args = parser.parse_args()

	device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		cudnn.benchmark = True

	ckpt_path = os.path.join(args.checkpoint_dir, "best.pth")
	if not os.path.isfile(ckpt_path):
		raise FileNotFoundError(f"未找到 best.pth: {ckpt_path}")

	teacher, cfg = load_teacher(ckpt_path, device)
	enc = str(cfg.get("encoder_name", "resnet18")).lower()
	if "resnet" not in enc:
		raise RuntimeError(f"当前脚本仅适配 ResNet（检测到 encoder_name={enc}）")

	# 构建学生并剪枝
	student = deepcopy(teacher).train()
	example = torch.randn(1, 3, 224, 224, device=device)
	student = prune_backbone_resnet(student, example, ch_sparsity=float(args.prune_ratio))

	# 数据加载：关闭增强，使用确定性预处理
	train_tf = build_train_transform({"use_val_preprocess": True}, image_size=224)
	val_tf = build_val_transform(224)
	train_ds = Cifar100DualLabels(args.dataset_root, split="train", transform=train_tf)
	val_ds = Cifar100DualLabels(args.dataset_root, split="test", transform=val_tf)
	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

	# KD 微调
	stats = kd_train(
		teacher=teacher,
		student=student,
		train_loader=train_loader,
		val_loader=val_loader,
		device=device,
		epochs=int(args.epochs),
		alpha=float(args.alpha),
		temperature=float(args.temperature),
	)

	# 输出目录
	tag = f"{args.output_subdir}_r{int(args.prune_ratio*100)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	out_dir = os.path.join(args.checkpoint_dir, tag)
	os.makedirs(out_dir, exist_ok=True)

	# 统计并保存
	meta = {
		"encoder_name": cfg.get("encoder_name", "resnet18"),
		"use_cbam": bool(cfg.get("use_cbam", False)),
		"use_film": bool(cfg.get("use_film", True)),
		"prune_ratio": float(args.prune_ratio),
		"param_count": int(count_params(student)),
		"forward_ms_bs1_224": float(measure_forward_ms(student.eval().to(device), device, 224)),
	}
	with open(os.path.join(out_dir, "prune_meta.json"), "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)
	with open(os.path.join(out_dir, "prune_log.json"), "w", encoding="utf-8") as f:
		json.dump(stats, f, ensure_ascii=False, indent=2)

	# 保存 student 权重（last/best 都一致保存一份）
	state = {
		"epoch": stats["best"]["epoch"],
		"model": student.state_dict(),
		"best_val_acc_fine": stats["best"]["val_fine_top1"],
		"config": cfg,
		"prune": {"ratio": float(args.prune_ratio), "method": "channel+kd"},
	}
	torch.save(state, os.path.join(out_dir, "student_last.pth"))
	torch.save(state, os.path.join(out_dir, "student_best.pth"))
	# 额外导出：可直接加载的模型
	try:
		student.eval().to(device)
		example = torch.randn(1, 3, 224, 224, device=device)
		scripted = torch.jit.trace(student, example, strict=False)
		script_path = os.path.join(out_dir, "student_scripted.pt")
		scripted.save(script_path)
		# 也保存可pickle的整模型（需同代码库环境下加载）
		pickle_path = os.path.join(out_dir, "student_model.pth")
		torch.save(student.cpu(), pickle_path)
		# 更新 meta 导出信息
		meta["export"] = {"scripted": script_path, "pickled": pickle_path}
		with open(os.path.join(out_dir, "prune_meta.json"), "w", encoding="utf-8") as f:
			json.dump(meta, f, ensure_ascii=False, indent=2)
	except Exception as _:
		# 忽略导出失败，不影响主流程
		pass
	print(f"[DONE] pruned model saved to: {out_dir}")


if __name__ == "__main__":
	main()


