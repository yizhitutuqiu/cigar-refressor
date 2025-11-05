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

from cifar_regressor import CifarHierarchicalRegressor
from cifar_regressor.utils import build_train_transform


@dataclass
class TrainConfig:
	# paths
	dataset_root: str
	checkpoint_dir: str

	# encoder/model
	encoder_name: str = "resnet18"
	pretrained_backbone: bool = True
	use_cbam: bool = False

	# heads/hparams
	num_coarse: int = 20
	num_fine: int = 100
	hidden_features: int = 256
	dropout_p: float = 0.1
	use_film: bool = True
	film_hidden: int = 256
	film_use_probs: bool = True

	# train
	batch_size: int = 128
	epochs: int = 30
	learning_rate: float = 3e-4
	backbone_lr_mult: float = 0.1
	weight_decay: float = 5e-2
	num_workers: int = 4
	log_interval: int = 50
	val_split: float = 0.1
	seed: int = 42
	device: str = "cuda"
	mixed_precision: bool = True
	lambda_coarse: float = 0.2

	# scheduler
	scheduler: Dict[str, Any] | None = None

	# augmentation
	aug: Dict[str, Any] | None = None


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


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


def build_transforms(cfg: TrainConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    # train with configurable augmentation
    train_tf = build_train_transform(cfg.aug, image_size=224)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
	train_tf, val_tf = build_transforms(cfg)
	full = Cifar100DualLabels(cfg.dataset_root, split="train", transform=train_tf)
	val_size = int(len(full) * cfg.val_split)
	train_size = len(full) - val_size
	train_ds, val_ds = random_split(full, [train_size, val_size])
	val_ds.dataset.transform = val_tf  # type: ignore
	train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
	return train_loader, val_loader


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
	pred = torch.argmax(logits, dim=1)
	return float((pred == targets).sum().item() / max(1, targets.size(0)))


def build_model(cfg: TrainConfig) -> nn.Module:
	return CifarHierarchicalRegressor(
		pretrained_backbone=cfg.pretrained_backbone,
		encoder_name=cfg.encoder_name,
		use_cbam=cfg.use_cbam,
		num_coarse=cfg.num_coarse,
		num_fine=cfg.num_fine,
		hidden_features=cfg.hidden_features,
		dropout_p=cfg.dropout_p,
		use_film=cfg.use_film,
		film_hidden=cfg.film_hidden,
		film_use_probs=cfg.film_use_probs,
	)


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
	# backbone 分组（ViT 或 ResNet）
	if hasattr(model, "vit"):
		backbone_params = list(model.vit.parameters())
	else:
		backbone_params = list(model.stem.parameters()) + list(model.layer1.parameters()) + list(model.layer2.parameters()) + list(model.layer3.parameters()) + list(model.layer4.parameters())
	head_params = list(model.coarse_head.parameters()) + list(model.fine_head.parameters())
	if getattr(model, "use_cbam", False):
		head_params += list(model.cbam.parameters())
	if getattr(model, "use_film", False):
		head_params += list(model.film.parameters())
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
		return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.scheduler.get("step_size", 10)), gamma=float(cfg.scheduler.get("gamma", 0.1)))
	return None


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion_c: nn.Module, criterion_f: nn.Module, optimizer, device: torch.device, scaler: torch.cuda.amp.GradScaler | None, log_interval: int, epoch: int, lambda_coarse: float) -> Dict[str, float]:
	model.train()
	metrics = {"loss": 0.0, "loss_coarse": 0.0, "loss_fine": 0.0, "acc_coarse": 0.0, "acc_fine": 0.0, "count": 0}
	for step, (images, coarse, fine) in enumerate(loader):
		images = images.to(device, non_blocking=True)
		coarse = coarse.to(device, non_blocking=True)
		fine = fine.to(device, non_blocking=True)
		optimizer.zero_grad(set_to_none=True)
		if scaler is not None:
			with torch.cuda.amp.autocast():
				out = model(images)
				loss_c = criterion_c(out["coarse_logits"], coarse)
				loss_f = criterion_f(out["fine_logits"], fine)
				loss = loss_f + lambda_coarse * loss_c
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			out = model(images)
			loss_c = criterion_c(out["coarse_logits"], coarse)
			loss_f = criterion_f(out["fine_logits"], fine)
			loss = loss_f + lambda_coarse * loss_c
			loss.backward()
			optimizer.step()

		acc_c = accuracy_top1(out["coarse_logits"], coarse)
		acc_f = accuracy_top1(out["fine_logits"], fine)
		bs = images.size(0)
		metrics["loss"] += float(loss.item()) * bs
		metrics["loss_coarse"] += float(loss_c.item()) * bs
		metrics["loss_fine"] += float(loss_f.item()) * bs
		metrics["acc_coarse"] += acc_c * bs
		metrics["acc_fine"] += acc_f * bs
		metrics["count"] += bs
		if (step + 1) % log_interval == 0:
			print(f"epoch {epoch} step {step+1}/{len(loader)} - loss: {metrics['loss']/metrics['count']:.4f} c:{metrics['acc_coarse']/metrics['count']:.3f} f:{metrics['acc_fine']/metrics['count']:.3f}")
	return {k: (metrics[k] / max(1, metrics["count"])) for k in metrics if k != "count"}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion_c: nn.Module, criterion_f: nn.Module, device: torch.device) -> Dict[str, float]:
	model.eval()
	metrics = {"loss": 0.0, "loss_coarse": 0.0, "loss_fine": 0.0, "acc_coarse": 0.0, "acc_fine": 0.0, "count": 0}
	for images, coarse, fine in loader:
		images = images.to(device, non_blocking=True)
		coarse = coarse.to(device, non_blocking=True)
		fine = fine.to(device, non_blocking=True)
		out = model(images)
		loss_c = criterion_c(out["coarse_logits"], coarse)
		loss_f = criterion_f(out["fine_logits"], fine)
		loss = loss_f + loss_c * 0.0  # eval 主指标仍以 fine 为主，汇总时仍输出各项
		acc_c = accuracy_top1(out["coarse_logits"], coarse)
		acc_f = accuracy_top1(out["fine_logits"], fine)
		bs = images.size(0)
		metrics["loss"] += float(loss.item()) * bs
		metrics["loss_coarse"] += float(loss_c.item()) * bs
		metrics["loss_fine"] += float(loss_f.item()) * bs
		metrics["acc_coarse"] += acc_c * bs
		metrics["acc_fine"] += acc_f * bs
		metrics["count"] += bs
	return {k: (metrics[k] / max(1, metrics["count"])) for k in metrics if k != "count"}


def load_config(path: str) -> TrainConfig:
	with open(path, "r", encoding="utf-8") as f:
		cfg = json.load(f)
	return TrainConfig(**cfg)


def main() -> None:
	parser = argparse.ArgumentParser(description="Hierarchical training: coarse+fine with FiLM")
	parser.add_argument("--config", type=str, default="/data/litengmo/ml-test/cifar_regressor/config/hierarchical_default.json")
	parser.add_argument("--gpu", type=int, default=None, help="选择使用的 GPU 编号")
	parser.add_argument("--print-config", action="store_true")
	args = parser.parse_args()

	cfg = load_config(args.config)
	if args.gpu is not None:
		cfg.device = f"cuda:{args.gpu}"

	if args.print_config:
		print("===== Hierarchical Train Config =====")
		print(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2, sort_keys=True))
		print("=====================================")

	set_seed(cfg.seed)
	device = torch.device(cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu")

	# run dir
	run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = os.path.join(cfg.checkpoint_dir, run_id)
	os.makedirs(run_dir, exist_ok=True)
	with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
		json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2, sort_keys=True)
	print(f"Run directory: {run_dir}")

	# tb writer
	writer = None
	try:
		from torch.utils.tensorboard import SummaryWriter  # type: ignore
		writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
	except Exception:
		writer = None

	model = build_model(cfg).to(device)
	criterion_c = nn.CrossEntropyLoss()
	criterion_f = nn.CrossEntropyLoss()
	optimizer = build_optimizer(cfg, model)
	scheduler = build_scheduler(cfg, optimizer)
	scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")

	train_loader, val_loader = build_loaders(cfg)

	best_val = -math.inf
	history = {"epoch": [], "train_loss": [], "train_acc_coarse": [], "train_acc_fine": [], "val_loss": [], "val_acc_coarse": [], "val_acc_fine": [], "lr": []}

	for epoch in range(1, cfg.epochs + 1):
		tr = train_one_epoch(model, train_loader, criterion_c, criterion_f, optimizer, device, scaler, cfg.log_interval, epoch, cfg.lambda_coarse)
		va = evaluate(model, val_loader, criterion_c, criterion_f, device)
		if scheduler is not None:
			scheduler.step()

		curr_lr = max(pg.get("lr", 0.0) for pg in optimizer.param_groups)
		print(f"epoch {epoch} - train_loss: {tr['loss']:.4f} c:{tr['acc_coarse']:.3f} f:{tr['acc_fine']:.3f} | val_loss: {va['loss']:.4f} c:{va['acc_coarse']:.3f} f:{va['acc_fine']:.3f}")

		if writer is not None:
			writer.add_scalar("train/loss", tr["loss"], epoch)
			writer.add_scalar("train/acc_coarse", tr["acc_coarse"], epoch)
			writer.add_scalar("train/acc_fine", tr["acc_fine"], epoch)
			writer.add_scalar("val/loss", va["loss"], epoch)
			writer.add_scalar("val/acc_coarse", va["acc_coarse"], epoch)
			writer.add_scalar("val/acc_fine", va["acc_fine"], epoch)
			writer.add_scalar("opt/lr", curr_lr, epoch)

		history["epoch"].append(epoch)
		history["train_loss"].append(tr["loss"])
		history["train_acc_coarse"].append(tr["acc_coarse"])
		history["train_acc_fine"].append(tr["acc_fine"])
		history["val_loss"].append(va["loss"])
		history["val_acc_coarse"].append(va["acc_coarse"])
		history["val_acc_fine"].append(va["acc_fine"])
		history["lr"].append(curr_lr)
		with open(os.path.join(run_dir, "train_log.json"), "w", encoding="utf-8") as f:
			json.dump(history, f, ensure_ascii=False, indent=2)

		# 主监控指标：val fine acc
		is_best = va["acc_fine"] > best_val
		if is_best:
			best_val = va["acc_fine"]
		state = {
			"epoch": epoch,
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
			"scheduler": scheduler.state_dict() if scheduler is not None else None,
			"best_val_acc_fine": best_val,
			"config": cfg.__dict__,
		}
		os.makedirs(cfg.checkpoint_dir, exist_ok=True)
		torch.save(state, os.path.join(cfg.checkpoint_dir, "last.pth"))
		if is_best:
			torch.save(state, os.path.join(cfg.checkpoint_dir, "best.pth"))
		# 保存到 run_dir 便于溯源
		torch.save(state, os.path.join(run_dir, "last.pth"))
		if is_best:
			torch.save(state, os.path.join(run_dir, "best.pth"))

	if writer is not None:
		writer.flush()
		writer.close()
	print("training finished. best_val_acc_fine=%.4f | logs at: %s" % (best_val, run_dir))


if __name__ == "__main__":
	main()


