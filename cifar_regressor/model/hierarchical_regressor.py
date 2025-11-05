from __future__ import annotations

from typing import Dict, Tuple

import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]
from torch import Tensor  # type: ignore[import-not-found]

# Reuse encoder builders and decoder head from coarse regressor
from .cifar_coarse_regressor import (  # type: ignore
	_make_resnet,
	_make_vit,
	DecoderHead,
)
from .cbam import CBAM  # type: ignore[import-not-found]


class FilmAdapter(nn.Module):
	"""FiLM: produce gamma/beta from coarse signal and modulate features.

	near-identity init to avoid destabilizing pretrained encoders.
	"""

	def __init__(self, signal_dim: int, feature_dim: int, hidden: int = 256, use_probs: bool = True) -> None:
		super().__init__()
		self.use_probs = use_probs
		self.net = nn.Sequential(
			nn.Linear(signal_dim, hidden, bias=True),
			nn.GELU(),
			nn.Linear(hidden, feature_dim * 2, bias=True),
		)
		# near-identity: last layer weights to zero; gamma bias to 0, beta bias to 0
		last = self.net[-1]
		nn.init.zeros_(last.weight)
		nn.init.zeros_(last.bias)

	def forward(self, features: Tensor, coarse_signal: Tensor) -> Tensor:
		gb = self.net(coarse_signal)
		gamma, beta = gb.chunk(2, dim=-1)
		# 1 + gamma ensures starting close to identity
		return (1.0 + gamma) * features + beta


class CifarHierarchicalRegressor(nn.Module):
	"""Shared encoder + dual heads (coarse 20, fine 100) with optional FiLM from coarse to fine.

	Supports ResNet (with optional CBAM) and ViT-S via timm.
	"""

	def __init__(
		self,
		pretrained_backbone: bool = True,
		encoder_name: str = "resnet18",
		use_cbam: bool = False,
		# heads
		num_coarse: int = 20,
		num_fine: int = 100,
		hidden_features: int = 256,
		dropout_p: float = 0.1,
		# FiLM
		use_film: bool = True,
		film_hidden: int = 256,
		film_use_probs: bool = True,
	) -> None:
		super().__init__()
		name_l = encoder_name.lower()
		self.encoder_name = encoder_name
		self.encoder_type = "vit" if "vit" in name_l else "resnet"
		self.use_cbam = bool(use_cbam) if self.encoder_type == "resnet" else False
		self.use_film = bool(use_film)
		self.num_coarse = num_coarse
		self.num_fine = num_fine

		if self.encoder_type == "resnet":
			backbone = _make_resnet(encoder_name=encoder_name, pretrained=pretrained_backbone)
			if not hasattr(backbone, "fc"):
				raise RuntimeError("Unexpected ResNet structure: missing attribute 'fc'")
			feat_dim = backbone.fc.in_features
			self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
			self.layer1 = backbone.layer1
			self.layer2 = backbone.layer2
			self.layer3 = backbone.layer3
			self.layer4 = backbone.layer4
			self.avgpool = backbone.avgpool
			self.cbam = CBAM(feat_dim) if self.use_cbam else nn.Identity()
			self.backbone = backbone
		else:
			vit, feat_dim = _make_vit(encoder_name=encoder_name, pretrained=pretrained_backbone)
			self.vit = vit

		# heads
		self.coarse_head = DecoderHead(in_features=feat_dim, hidden_features=hidden_features, num_classes=num_coarse, dropout_p=dropout_p)
		self.fine_head = DecoderHead(in_features=feat_dim, hidden_features=hidden_features, num_classes=num_fine, dropout_p=dropout_p)
		self.softmax = nn.Softmax(dim=1)

		# FiLM adapter from coarse (20-d) to feature dim
		self.film = FilmAdapter(signal_dim=num_coarse, feature_dim=feat_dim, hidden=film_hidden, use_probs=film_use_probs) if self.use_film else nn.Identity()
		self.film_use_probs = film_use_probs

	def encode(self, x: Tensor) -> Tensor:
		if self.encoder_type == "resnet":
			x = self.stem(x)
			x = self.layer1(x)
			x = self.layer2(x)
			x = self.layer3(x)
			x = self.layer4(x)
			x = self.cbam(x) if isinstance(self.cbam, nn.Module) else x
			x = self.avgpool(x)
			return torch.flatten(x, 1)
		else:
			return self.vit(x)

	def forward(self, x: Tensor) -> Dict[str, Tensor]:
		features = self.encode(x)
		coarse_logits = self.coarse_head(features)
		coarse_probs = self.softmax(coarse_logits)

		if self.use_film:
			signal = coarse_probs if self.film_use_probs else coarse_logits
			features = self.film(features, signal)

		fine_logits = self.fine_head(features)
		fine_probs = self.softmax(fine_logits)
		return {
			"coarse_logits": coarse_logits,
			"coarse_probs": coarse_probs,
			"fine_logits": fine_logits,
			"fine_probs": fine_probs,
		}


