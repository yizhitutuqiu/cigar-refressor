from typing import Optional, Tuple

import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]
from torch import Tensor  # type: ignore[import-not-found]
from .cbam import CBAM  # type: ignore[import-not-found]


def _make_resnet(encoder_name: str = "resnet18", pretrained: bool = False) -> nn.Module:
    """Create a ResNet backbone compatible with different torchvision versions.

    Supported: resnet18, resnet34, resnet50.
    If unsupported (e.g., resnet10), falls back to resnet18.
    """
    try:
        from torchvision import models  # type: ignore[import-not-found]
        name = encoder_name.lower()
        if name not in {"resnet18", "resnet34", "resnet50"}:
            name = "resnet18"
        ctor = getattr(models, name)
        # Try new API with weights enum first; fall back to old API
        try:
            weights = None
            if pretrained:
                enum_name_map = {
                    "resnet18": "ResNet18_Weights",
                    "resnet34": "ResNet34_Weights",
                    "resnet50": "ResNet50_Weights",
                }
                enum_name = enum_name_map.get(name)
                weights_enum = getattr(models, enum_name, None) if enum_name else None
                if weights_enum is not None and hasattr(weights_enum, "DEFAULT"):
                    weights = getattr(weights_enum, "DEFAULT")
            model = ctor(weights=weights)
        except TypeError:
            # Older API
            model = ctor(pretrained=pretrained)
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to construct {encoder_name}: {exc}")


def _canonical_vit_name(name: str) -> str:
	name = name.lower().replace("-", "_")
	# 常见别名统一到 timm 的命名
	small_aliases = {"vit_s", "vit_s16", "vit_small", "vit_small_16", "vit_small_patch16_224"}
	base_aliases = {"vit_b", "vit_b16", "vit_base", "vit_base_16", "vit_base_patch16_224"}
	huge_aliases = {"vit_h", "vit_h14", "vit_huge", "vit_huge_14", "vit_huge_patch14_224"}
	if name in small_aliases or ("vit" in name and "small" in name):
		return "vit_small_patch16_224"
	if name in base_aliases or ("vit" in name and "base" in name):
		return "vit_base_patch16_224"
	if name in huge_aliases or ("vit" in name and "huge" in name):
		return "vit_huge_patch14_224"
	return name


def _make_vit(encoder_name: str = "vit_small_patch16_224", pretrained: bool = True) -> Tuple[nn.Module, int]:
	"""Create a ViT backbone via timm. Returns (model, feature_dim).

	Requires `timm` package. The model will output feature vectors (N, D).
	"""
	try:
		import timm  # type: ignore[import-not-found]
	except Exception as exc:
		raise RuntimeError(
			f"ViT encoder requires 'timm'. Please install it (e.g., pip install timm). Error: {exc}"
		)

	model_name = _canonical_vit_name(encoder_name)
	try:
		# num_classes=0 -> head removed; forward returns pooled features
		vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
		feat_dim = getattr(vit, "num_features", None) or getattr(vit, "embed_dim", None)
		if feat_dim is None:
			raise RuntimeError("Failed to infer ViT feature dim (num_features/embed_dim not found)")
		return vit, int(feat_dim)
	except Exception as exc:
		raise RuntimeError(f"Failed to construct ViT '{encoder_name}': {exc}")


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
		encoder_name: str = "resnet18",
	) -> None:
		super().__init__()

		name_l = encoder_name.lower()
		self.encoder_name = encoder_name
		self.encoder_type = "vit" if "vit" in name_l else "resnet"
		self.use_cbam = bool(use_cbam) if self.encoder_type == "resnet" else False

		if self.encoder_type == "resnet":
			backbone = _make_resnet(encoder_name=encoder_name, pretrained=pretrained_backbone)
			if not hasattr(backbone, "fc"):
				raise RuntimeError("Unexpected ResNet structure: missing attribute 'fc'")
			in_features = backbone.fc.in_features
			self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
			self.layer1 = backbone.layer1
			self.layer2 = backbone.layer2
			self.layer3 = backbone.layer3
			self.layer4 = backbone.layer4
			self.avgpool = backbone.avgpool
			self.cbam = CBAM(in_features) if self.use_cbam else nn.Identity()
			self.backbone = backbone  # keep ref for completeness
			features_dim = in_features
		else:
			vit, feat_dim = _make_vit(encoder_name=encoder_name, pretrained=pretrained_backbone)
			self.vit = vit
			features_dim = feat_dim

		self.decoder = DecoderHead(
			in_features=features_dim,
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
		if self.encoder_type == "resnet":
			# CNN path with optional CBAM
			x = self.stem(x)
			x = self.layer1(x)
			x = self.layer2(x)
			x = self.layer3(x)
			x = self.layer4(x)
			if self.use_cbam:
				x = self.cbam(x)
			x = self.avgpool(x)
			features = torch.flatten(x, 1)
		else:
			# ViT path (features already pooled when num_classes=0 in timm)
			features = self.vit(x)
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


