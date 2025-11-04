from typing import Optional, Tuple

import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]
from torch import Tensor  # type: ignore[import-not-found]
from .cbam import CBAM  # type: ignore[import-not-found]


def _make_resnet18(pretrained: bool = False) -> nn.Module:
	"""Create a ResNet18 backbone compatible with different torchvision versions.

	- torchvision>=0.13 uses `weights` argument
	- older versions use `pretrained` boolean
	"""
	try:
		from torchvision.models import resnet18  # type: ignore[import-not-found]
		# Try new API first
		try:
			model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
		except TypeError:
			# Fallback to old API
			model = resnet18(pretrained=pretrained)
		return model
	except Exception as exc:
		raise RuntimeError(f"Failed to construct resnet18: {exc}")


class DecoderHead(nn.Module):
	"""A small MLP decoder head for 20 coarse classes.

	Input: (N, 512)
	Output: (N, 20)
	"""

	def __init__(self, in_features: int = 512, hidden_features: int = 256, num_classes: int = 20, dropout_p: float = 0.1) -> None:
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(in_features, hidden_features),
			nn.GELU(),
			nn.Dropout(p=dropout_p),
			nn.Linear(hidden_features, num_classes),
		)

	def forward(self, x: Tensor) -> Tensor:
		return self.layers(x)


class CifarCoarseRegressor(nn.Module):
	"""ResNet18 backbone + decoder head for CIFAR-100 coarse labels (20 classes).

	The forward returns (logits, probabilities) where probabilities are softmax over classes.
	The class also provides a convenient `sample_topk` method to draw samples from top-k probabilities.
	"""

	def __init__(
		self,
		pretrained_backbone: bool = False,
		num_classes: int = 20,
		dropout_p: float = 0.1,
		hidden_features: int = 256,
		use_cbam: bool = False,
	) -> None:
		super().__init__()
		backbone = _make_resnet18(pretrained=pretrained_backbone)

		# Keep references to feature blocks for custom forward (to insert CBAM before avgpool)
		if not hasattr(backbone, "fc"):
			raise RuntimeError("Unexpected ResNet18 structure: missing attribute 'fc'")
		in_features = backbone.fc.in_features  # typically 512
		self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
		self.layer1 = backbone.layer1
		self.layer2 = backbone.layer2
		self.layer3 = backbone.layer3
		self.layer4 = backbone.layer4
		self.avgpool = backbone.avgpool
		self.use_cbam = use_cbam
		self.cbam = CBAM(in_features) if use_cbam else nn.Identity()

		self.decoder = DecoderHead(
			in_features=in_features,
			hidden_features=hidden_features,
			num_classes=num_classes,
			dropout_p=dropout_p,
		)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
		"""Forward pass.

		Returns
		-------
		logits: Tensor of shape (N, C)
		probs:  Tensor of shape (N, C), softmax over logits
		"""
		# Custom forward to optionally apply CBAM on spatial features
		x = self.stem(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		if self.use_cbam:
			x = self.cbam(x)
		x = self.avgpool(x)
		features = torch.flatten(x, 1)
		logits = self.decoder(features)
		probs = self.softmax(logits)
		return logits, probs

	@torch.no_grad()
	def sample_topk(
		self,
		x: Optional[Tensor] = None,
		logits: Optional[Tensor] = None,
		probs: Optional[Tensor] = None,
		k: int = 5,
		num_samples: int = 1,
	) -> Tensor:
		"""Sample class indices from the top-k probability mass.

		You can provide either an input batch `x` (the model will forward), or
		directly provide `logits` or `probs`.

		Returns a LongTensor of shape (N, num_samples) with sampled class indices.
		"""
		if probs is None:
			if logits is None:
				if x is None:
					raise ValueError("Provide one of x, logits, or probs")
				logits, _probs = self.forward(x)
				probs = _probs
			else:
				probs = self.softmax(logits)

		# top-k mask and renormalize
		values, indices = torch.topk(probs, k=k, dim=1)
		masked = torch.zeros_like(probs)
		masked.scatter_(1, indices, values)
		masked = masked / masked.sum(dim=1, keepdim=True).clamp_min(1e-12)

		# multinomial sampling per row
		# For numerical stability, clamp to avoid zero-prob causing error when all zero
		masked = masked.clamp_min(1e-12)
		return torch.multinomial(masked, num_samples=num_samples, replacement=True)


