from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np  # type: ignore[import-not-found]
from PIL import Image  # type: ignore[import-not-found]

import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]
from torch.utils.data import Dataset, DataLoader, random_split  # type: ignore[import-not-found]

from torchvision import transforms  # type: ignore[import-not-found]

# Ensure project root is on sys.path for absolute-path execution
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from cifar_regressor import CifarCoarseRegressor


@dataclass
class TrainConfig:
	# paths
	dataset_root: str
	checkpoint_dir: str

	# model
	num_classes: int = 20
	pretrained_backbone: bool = True
	use_cbam: bool = False
	encoder_name: str = "resnet18"
	hidden_features: int = 256
	dropout_p: float = 0.1

	# train
	batch_size: int = 128
	epochs: int = 30
	learning_rate: float = 3e-4
	weight_decay: float = 5e-2
	backbone_lr_mult: float = 0.1
	num_workers: int = 4
	log_interval: int = 50
	val_split: float = 0.1
	seed: int = 42
	device: str = "cuda"
	mixed_precision: bool = True

	# scheduler
	scheduler: Dict[str, Any] | None = None


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


class Cifar100CoarseDataset(Dataset):
	"""CIFAR-100 python 版数据读取（使用 coarse_labels）。"""

	def __init__(self, root_dir: str, split: str, transform: transforms.Compose | None = None) -> None:
		super().__init__()
		if split not in {"train", "test"}:
			raise ValueError("split 必须是 'train' 或 'test'")
		self.root_dir = root_dir
		self.split = split
		self.transform = transform

		# 载入 pickle
		import pickle
		with open(os.path.join(root_dir, split), "rb") as f:
			data_dict = pickle.load(f, encoding="latin1")

		self.data = data_dict["data"]  # (N, 3072)
		self.coarse_labels = data_dict["coarse_labels"]  # (N,)

		# 载入 coarse label 名称（可选，用于可视化/日志）
		try:
			with open(os.path.join(root_dir, "meta"), "rb") as f:
				meta = pickle.load(f, encoding="latin1")
			self.coarse_label_names = meta.get("coarse_label_names", None)
		except Exception:
			self.coarse_label_names = None

	def __len__(self) -> int:
		return len(self.coarse_labels)

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
		row = self.data[index]
		# CIFAR 存储为 (3072,) -> (3, 32, 32)
		img = np.reshape(row, (3, 32, 32))
		img = np.transpose(img, (1, 2, 0))  # -> (32, 32, 3)
		img = Image.fromarray(img.astype(np.uint8))
		if self.transform is not None:
			img = self.transform(img)
		label = int(self.coarse_labels[index])
		return img, label


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
	# 采用 ImageNet 预训练常用预处理
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	train_tf = transforms.Compose([
		transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	])
	val_tf = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	])
	return train_tf, val_tf


def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
	train_tf, val_tf = build_transforms()
	full = Cifar100CoarseDataset(cfg.dataset_root, split="train", transform=train_tf)
	val_size = int(len(full) * cfg.val_split)
	train_size = len(full) - val_size
	train_ds, val_ds = random_split(full, [train_size, val_size])
	# 验证集使用 val_tf
	val_ds.dataset.transform = val_tf  # type: ignore[attr-defined]

	train_loader = DataLoader(
		train_ds,
		batch_size=cfg.batch_size,
		shuffle=True,
		num_workers=cfg.num_workers,
		pin_memory=True,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=cfg.batch_size,
		shuffle=False,
		num_workers=cfg.num_workers,
		pin_memory=True,
	)
	return train_loader, val_loader


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
	with torch.no_grad():
		pred = torch.argmax(logits, dim=1)
		correct = (pred == targets).sum().item()
		return correct / max(1, targets.size(0))


def build_model(cfg: TrainConfig) -> nn.Module:
	model = CifarCoarseRegressor(
		pretrained_backbone=cfg.pretrained_backbone,
		num_classes=cfg.num_classes,
		hidden_features=cfg.hidden_features,
		dropout_p=cfg.dropout_p,
		use_cbam=cfg.use_cbam,
		encoder_name=cfg.encoder_name,
	)
	return model


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
	# 参数组：预训练骨干较低学习率，CBAM/decoder 使用基础学习率
	if hasattr(model, "vit"):
		# ViT 分支
		backbone_params = list(model.vit.parameters())
		head_params = list(model.decoder.parameters())
	else:
		# ResNet 分支
		backbone_params = list(model.stem.parameters()) + \
			list(model.layer1.parameters()) + list(model.layer2.parameters()) + \
			list(model.layer3.parameters()) + list(model.layer4.parameters())
		head_params = list(model.decoder.parameters())
		if getattr(model, "use_cbam", False):
			head_params += list(model.cbam.parameters())

	param_groups = [
		{"params": backbone_params, "lr": cfg.learning_rate * cfg.backbone_lr_mult},
		{"params": head_params, "lr": cfg.learning_rate},
	]
	return torch.optim.AdamW(param_groups, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def build_scheduler(cfg: TrainConfig, optimizer: torch.optim.Optimizer):
	if not cfg.scheduler:
		return None
	stype = cfg.scheduler.get("type", "").lower()
	if stype == "cosine":
		eta_min = float(cfg.scheduler.get("eta_min", 1e-6))
		return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=eta_min)
	elif stype == "steplr":
		step_size = int(cfg.scheduler.get("step_size", 10))
		gamma = float(cfg.scheduler.get("gamma", 0.1))
		return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
	return None


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer, device: torch.device, scaler: torch.cuda.amp.GradScaler | None, log_interval: int, epoch: int) -> Tuple[float, float]:
	model.train()
	running_loss = 0.0
	running_acc = 0.0
	count = 0
	for step, (images, targets) in enumerate(loader):
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		optimizer.zero_grad(set_to_none=True)
		if scaler is not None:
			with torch.cuda.amp.autocast():
				logits, probs = model(images)
				loss = criterion(logits, targets)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			logits, probs = model(images)
			loss = criterion(logits, targets)
			loss.backward()
			optimizer.step()

		acc = accuracy_top1(logits, targets)
		running_loss += float(loss.item()) * images.size(0)
		running_acc += acc * images.size(0)
		count += images.size(0)

		if (step + 1) % log_interval == 0:
			curr_loss = running_loss / count
			curr_acc = running_acc / count
			print(f"epoch {epoch} step {step+1}/{len(loader)} - loss: {curr_loss:.4f} acc: {curr_acc:.4f}")

	return running_loss / count, running_acc / count


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
	model.eval()
	total_loss = 0.0
	total_acc = 0.0
	count = 0
	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		logits, probs = model(images)
		loss = criterion(logits, targets)
		acc = accuracy_top1(logits, targets)
		total_loss += float(loss.item()) * images.size(0)
		total_acc += acc * images.size(0)
		count += images.size(0)
	return total_loss / count, total_acc / count


def save_checkpoint(state: Dict[str, Any], is_best: bool, out_dir: str, filename: str = "last.pth") -> None:
	os.makedirs(out_dir, exist_ok=True)
	path = os.path.join(out_dir, filename)
	torch.save(state, path)
	if is_best:
		best_path = os.path.join(out_dir, "best.pth")
		torch.save(state, best_path)


def load_config(path: str) -> TrainConfig:
	with open(path, "r", encoding="utf-8") as f:
		cfg_dict = json.load(f)
	return TrainConfig(**cfg_dict)


def main() -> None:
	parser = argparse.ArgumentParser(description="Train ResNet18+Head on CIFAR-100 coarse labels")
	parser.add_argument("--config", type=str, default="/data/litengmo/ml-test/cifar_regressor/config/coarse_default.json")
	parser.add_argument("--gpu", type=int, default=None, help="选择使用的 GPU 编号，如 0、1、7；不填则按 config.device")
	parser.add_argument("--print-config", action="store_true", help="打印训练配置与 CBAM 启用状态")
	args = parser.parse_args()

	cfg = load_config(args.config)
	# 覆盖设备为指定 GPU（优先级高于配置文件）
	if args.gpu is not None:
		cfg.device = f"cuda:{args.gpu}"

	if args.print_config:
		print("===== Train Config =====")
		print(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2, sort_keys=True))
		print(f"CBAM requested (config.use_cbam): {cfg.use_cbam}")
		print("========================")
	set_seed(cfg.seed)

	# 运行目录：每次训练一个子目录，用于日志与TensorBoard
	run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = os.path.join(cfg.checkpoint_dir, run_id)
	os.makedirs(run_dir, exist_ok=True)
	# 保存配置副本
	with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
		json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2, sort_keys=True)
	print(f"Run directory: {run_dir}")

	# 尝试创建 TensorBoard Writer（若不可用则跳过）
	writer = None
	try:
		from torch.utils.tensorboard import SummaryWriter  # type: ignore
		writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
	except Exception:
		writer = None

	device = torch.device(cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu")
	model = build_model(cfg).to(device)
	if args.print_config:
		# 打印模型上是否实际启用了 CBAM（模型内的 use_cbam 与模块类型）
		use_cbam_runtime = getattr(model, "use_cbam", False)
		cbam_module = getattr(model, "cbam", None)
		cbam_type = cbam_module.__class__.__name__ if cbam_module is not None else "None"
		print(f"CBAM enabled (runtime): {use_cbam_runtime} | module: {cbam_type}")
	criterion = nn.CrossEntropyLoss()
	optimizer = build_optimizer(cfg, model)
	scheduler = build_scheduler(cfg, optimizer)
	scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")

	train_loader, val_loader = build_loaders(cfg)

	best_val_acc = -math.inf
	# 训练日志（JSON）
	history = {
		"epoch": [],
		"train_loss": [],
		"train_acc": [],
		"val_loss": [],
		"val_acc": [],
		"lr": [],
	}
	for epoch in range(1, cfg.epochs + 1):
		train_loss, train_acc = train_one_epoch(
			model,
			train_loader,
			criterion,
			optimizer,
			device,
			scaler,
			cfg.log_interval,
			epoch,
		)
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)
		if scheduler is not None:
			scheduler.step()

		print(f"epoch {epoch} done - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

		# 记录学习率（取参数组中最大的lr以代表本轮）
		curr_lr = max(pg.get("lr", 0.0) for pg in optimizer.param_groups)
		# 写入TensorBoard
		if writer is not None:
			writer.add_scalar("train/loss", train_loss, epoch)
			writer.add_scalar("train/acc", train_acc, epoch)
			writer.add_scalar("val/loss", val_loss, epoch)
			writer.add_scalar("val/acc", val_acc, epoch)
			writer.add_scalar("opt/lr", curr_lr, epoch)

		# 累计到JSON日志并落盘
		history["epoch"].append(epoch)
		history["train_loss"].append(train_loss)
		history["train_acc"].append(train_acc)
		history["val_loss"].append(val_loss)
		history["val_acc"].append(val_acc)
		history["lr"].append(curr_lr)
		with open(os.path.join(run_dir, "train_log.json"), "w", encoding="utf-8") as f:
			json.dump(history, f, ensure_ascii=False, indent=2)

		is_best = val_acc > best_val_acc
		if is_best:
			best_val_acc = val_acc
		state = {
			"epoch": epoch,
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
			"scheduler": scheduler.state_dict() if scheduler is not None else None,
			"best_val_acc": best_val_acc,
			"config": cfg.__dict__,
		}
		save_checkpoint(state, is_best=is_best, out_dir=cfg.checkpoint_dir)
		# 同步保存一份到 run_dir，便于溯源（可选）
		save_checkpoint(state, is_best=is_best, out_dir=run_dir)

	# 关闭TensorBoard
	if writer is not None:
		writer.flush()
		writer.close()

	print("training finished. best_val_acc=%.4f | logs at: %s" % (best_val_acc, run_dir))


if __name__ == "__main__":
	main()


