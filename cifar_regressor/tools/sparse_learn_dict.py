from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Tuple

import numpy as np  # type: ignore[import-not-found]
from PIL import Image  # type: ignore[import-not-found]
from sklearn.decomposition import MiniBatchDictionaryLearning  # type: ignore[import-not-found]
from sklearn.utils import shuffle  # type: ignore[import-not-found]
import joblib  # type: ignore[import-not-found]


def load_cifar_split(root: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	import pickle
	with open(os.path.join(root, split), "rb") as f:
		data = pickle.load(f, encoding="latin1")
	X = data["data"]  # (N, 3072)
	coarse = np.array(data["coarse_labels"])
	fine = np.array(data["fine_labels"])
	return X, coarse, fine


def extract_random_patches(X: np.ndarray, num_images: int, patches_per_img: int, patch: int, seed: int = 42) -> np.ndarray:
	# X: (N, 3072) raw CIFAR (3x32x32)
	rng = np.random.RandomState(seed)
	N = X.shape[0]
	ids = rng.choice(N, size=min(num_images, N), replace=False)
	patches = []
	for idx in ids:
		img = X[idx].reshape(3, 32, 32).transpose(1, 2, 0)  # (32,32,3)
		for _ in range(patches_per_img):
			y = rng.randint(0, 32 - patch + 1)
			x = rng.randint(0, 32 - patch + 1)
			p = img[y:y+patch, x:x+patch, :].astype(np.float32) / 255.0
			p = (p - p.mean()) / (p.std() + 1e-6)  # normalize patch
			patches.append(p.reshape(-1))
	P = np.stack(patches, axis=0) if patches else np.empty((0, patch*patch*3), dtype=np.float32)
	return P


def main():
	parser = argparse.ArgumentParser(description="Sparse Dictionary Learning on CIFAR-100 (patch-based)")
	parser.add_argument("--dataset_root", type=str, default="./cifar-100-python")
	parser.add_argument("--out_dir", type=str, default="./cifar_regressor/checkpoints/sparse")
	parser.add_argument("--patch", type=int, default=8)
	parser.add_argument("--k", type=int, default=1024, help="dictionary size")
	parser.add_argument("--images", type=int, default=20000, help="num training images sampled")
	parser.add_argument("--patches_per_img", type=int, default=20)
	parser.add_argument("--alpha", type=float, default=1.0, help="L1 sparsity coefficient in dictionary learning")
	parser.add_argument("--iter", type=int, default=2, help="epochs (passes) over the dataset")
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	X, _, _ = load_cifar_split(args.dataset_root, "train")
	P = extract_random_patches(X, num_images=int(args.images), patches_per_img=int(args.patches_per_img), patch=int(args.patch), seed=int(args.seed))
	if P.shape[0] == 0:
		raise RuntimeError("No patches extracted, please check parameters.")
	P = shuffle(P, random_state=args.seed)

	print(f"[INFO] patches: {P.shape}, dict_size={args.k}, alpha={args.alpha}, iter={args.iter}")
	dict_learner = MiniBatchDictionaryLearning(
		n_components=int(args.k),
		alpha=float(args.alpha),
		max_iter=int(args.iter),  # sklearn>=1.2 使用 max_iter
		batch_size=256,
		random_state=args.seed,
		verbose=True,
	)
	D = dict_learner.fit(P).components_.astype(np.float32)  # (K, p)

	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	base = f"dict_p{args.patch}_k{args.k}_{stamp}"
	dict_path = os.path.join(args.out_dir, base + ".joblib")
	joblib.dump({"D": D, "patch": int(args.patch), "k": int(args.k), "alpha": float(args.alpha)}, dict_path)

	meta = {
		"dict_file": dict_path,
		"shape": list(D.shape),
		"patch": int(args.patch),
		"k": int(args.k),
		"alpha": float(args.alpha),
		"images": int(args.images),
		"patches_per_img": int(args.patches_per_img),
	}
	with open(os.path.join(args.out_dir, base + ".json"), "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)
	print(f"[OK] saved dict: {dict_path}")


if __name__ == "__main__":
	main()


