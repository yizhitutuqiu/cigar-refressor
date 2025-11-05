from __future__ import annotations

from typing import Tuple

import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]
from torch import Tensor  # type: ignore[import-not-found]


class ChannelAttention(nn.Module):
	def __init__(self, channels: int, reduction: int = 16) -> None:
		super().__init__()
		hidden = max(1, channels // reduction)
		self.fc1 = nn.Linear(channels, hidden, bias=True)
		self.relu = nn.ReLU(inplace=True)
		self.fc2 = nn.Linear(hidden, channels, bias=True)
		self.sigmoid = nn.Sigmoid()
		# near-identity init: zero weights, positive bias so sigmoid≈0.88
		nn.init.zeros_(self.fc1.weight)
		nn.init.zeros_(self.fc2.weight)
		nn.init.zeros_(self.fc1.bias)
		nn.init.constant_(self.fc2.bias, 2.0)

	def forward(self, x: Tensor) -> Tensor:
		# x: (N, C, H, W)
		N, C, _, _ = x.shape
		avg_pool = torch.mean(x, dim=(2, 3), keepdim=False)  # (N, C)
		max_pool, _ = torch.max(x, dim=2, keepdim=False)
		max_pool, _ = torch.max(max_pool, dim=2, keepdim=False)  # (N, C)
		attn = self.fc2(self.relu(self.fc1(avg_pool))) + self.fc2(self.relu(self.fc1(max_pool)))
		attn = self.sigmoid(attn).view(N, C, 1, 1)
		return x * attn


class SpatialAttention(nn.Module):
	def __init__(self, kernel_size: int = 7) -> None:
		super().__init__()
		padding = (kernel_size - 1) // 2
		self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=True)
		self.sigmoid = nn.Sigmoid()
		# near-identity init
		nn.init.zeros_(self.conv.weight)
		nn.init.constant_(self.conv.bias, 2.0)

	def forward(self, x: Tensor) -> Tensor:
		# x: (N, C, H, W)
		avg = torch.mean(x, dim=1, keepdim=True)
		mx, _ = torch.max(x, dim=1, keepdim=True)
		cat = torch.cat([avg, mx], dim=1)
		attn = self.sigmoid(self.conv(cat))
		return x * attn


class CBAM(nn.Module):
	def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
		super().__init__()
		self.channel = ChannelAttention(channels, reduction)
		self.spatial = SpatialAttention(spatial_kernel)

	def forward(self, x: Tensor) -> Tensor:
		x = self.channel(x)
		x = self.spatial(x)
		return x


