from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np  # type: ignore[import-not-found]
from sklearn.decomposition import sparse_encode  # type: ignore[import-not-found]
import joblib  # type: ignore[import-not-found]


def load_cifar_split(root: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	import pickle
	with open(os.path.join(root, split), "rb") as f:
		data = pickle.load(f, encoding="latin1")
	X = data["data"]
	coarse = np.array(data["coarse_labels"])
	fine = np.array(data["fine_labels"])
	return X, coarse, fine


def extract_all_patches(img: np.ndarray, patch: int, stride: int) -> np.ndarray:
	img = img.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.float32) / 255.0
	H = W = 32
	patches = []
	for y in range(0, H - patch + 1, stride):
		for x in range(0, W - patch + 1, stride):
			p = img[y:y+patch, x:x+patch, :]
			p = (p - p.mean()) / (p.std() + 1e-6)
			patches.append(p.reshape(-1))
	return np.stack(patches, axis=0) if patches else np.empty((0, patch*patch*3), dtype=np.float32)


def image_code(D: np.ndarray, img_vec: np.ndarray, patch: int, stride: int, alpha: float, algo: str) -> np.ndarray:
	P = extract_all_patches(img_vec, patch=patch, stride=stride)
	if P.shape[0] == 0:
		return np.zeros((D.shape[0],), dtype=np.float32)
	Z = sparse_encode(P, D, algorithm=algo, alpha=alpha, n_nonzero_coefs=None)  # (M,K)
	return Z.mean(axis=0).astype(np.float32)


def main():
	parser = argparse.ArgumentParser(description="Evaluate sparse model on CIFAR-100 test set")
	parser.add_argument("--dataset_root", type=str, default="./cifar-100-python")
	parser.add_argument("--model_dir", type=str, required=True, help="directory containing model.joblib from sparse_train")
	parser.add_argument("--output_root", type=str, default="./cifar_regressor/test/sparse")
	args = parser.parse_args()

	os.makedirs(args.output_root, exist_ok=True)
	art = joblib.load(os.path.join(args.model_dir, "model.joblib"))
	Dfile = art["dict_file"]
	dinfo = joblib.load(Dfile)
	D = dinfo["D"]
	patch = int(art["patch"])
	stride = int(art["stride"])
	alpha = float(art["alpha"])
	algo = art["algo"]
	scaler = art["scaler"]
	clf_fine = art["clf_fine"]
	clf_coarse = art["clf_coarse"]

	Xte, cte, fte = load_cifar_split(args.dataset_root, "test")
	N = Xte.shape[0]
	feat = np.zeros((N, D.shape[0]), dtype=np.float32)
	for i in range(N):
		feat[i] = image_code(D, Xte[i], patch=patch, stride=stride, alpha=alpha, algo=algo)
		if (i + 1) % 1000 == 0:
			print(f"[encode] {i+1}/{N}")
	feat = scaler.transform(feat)

	def top1(clf, X, y): return float((clf.predict(X) == y).mean())
	res = {
		"coarse_top1": top1(clf_coarse, feat, cte),
		"fine_top1": top1(clf_fine, feat, fte),
		"dict": {"k": int(D.shape[0]), "patch": patch},
		"encode": {"patch": patch, "stride": stride, "alpha": alpha, "algo": algo},
	}
	out_dir = os.path.join(args.output_root, os.path.basename(args.model_dir))
	os.makedirs(out_dir, exist_ok=True)
	with open(os.path.join(out_dir, "eval_report_sparse.json"), "w", encoding="utf-8") as f:
		json.dump(res, f, ensure_ascii=False, indent=2)
	print(f"[OK] saved to {out_dir}")


if __name__ == "__main__":
	main()


