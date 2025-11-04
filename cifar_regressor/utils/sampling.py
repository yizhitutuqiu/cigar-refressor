from typing import Optional

import torch  # type: ignore[import-not-found]
from torch import Tensor  # type: ignore[import-not-found]


@torch.no_grad()
def top_k_sample(probs: Tensor, k: int = 5, num_samples: int = 1) -> Tensor:
	"""Sample indices from top-k of a probability distribution.

	Parameters
	----------
	probs: (N, C) probability tensor, each row sums to 1
	k: keep top-k classes per row for sampling
	num_samples: number of draws per row

	Returns
	-------
	LongTensor of shape (N, num_samples) with sampled class indices
	"""
	if probs.dim() != 2:
		raise ValueError("probs must be 2D (N, C)")
	if k <= 0:
		raise ValueError("k must be positive")
	N, C = probs.shape
	k = min(k, C)
	values, indices = torch.topk(probs, k=k, dim=1)
	masked = torch.zeros_like(probs)
	masked.scatter_(1, indices, values)
	masked = masked / masked.sum(dim=1, keepdim=True).clamp_min(1e-12)
	masked = masked.clamp_min(1e-12)
	return torch.multinomial(masked, num_samples=num_samples, replacement=True)


