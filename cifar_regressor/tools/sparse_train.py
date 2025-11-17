from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Tuple, Dict

import numpy as np  # type: ignore[import-not-found]
from PIL import Image  # type: ignore[import-not-found]
from sklearn.decomposition import sparse_encode  # type: ignore[import-not-found]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
from sklearn.svm import LinearSVC  # type: ignore[import-not-found]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]
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
	# img: (32,32,3) float32 [0,1]
	H = W = 32
	patches = []
	for y in range(0, H - patch + 1, stride):
		for x in range(0, W - patch + 1, stride):
			p = img[y:y+patch, x:x+patch, :]
			p = (p - p.mean()) / (p.std() + 1e-6)
			patches.append(p.reshape(-1))
	return np.stack(patches, axis=0) if patches else np.empty((0, patch*patch*3), dtype=np.float32)


def image_code(D: np.ndarray, img_vec: np.ndarray, patch: int, stride: int, alpha: float, algo: str) -> np.ndarray:
	img = img_vec.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.float32) / 255.0
	P = extract_all_patches(img, patch=patch, stride=stride)  # (M, p)
	if P.shape[0] == 0:
		return np.zeros((D.shape[0],), dtype=np.float32)
	# sparse coding per patch, then mean-pool over patches → image code
	Z = sparse_encode(P, D, algorithm=algo, alpha=alpha, n_nonzero_coefs=None)  # (M, K)
	code = Z.mean(axis=0).astype(np.float32)  # average pooling
	return code


def build_dataset_features(D: np.ndarray, X: np.ndarray, patch: int, stride: int, alpha: float, algo: str, limit: int | None = None) -> np.ndarray:
	N = X.shape[0] if limit is None else min(limit, X.shape[0])
	K = D.shape[0]
	feat = np.zeros((N, K), dtype=np.float32)
	for i in range(N):
		feat[i] = image_code(D, X[i], patch=patch, stride=stride, alpha=alpha, algo=algo)
		if (i + 1) % 1000 == 0:
			print(f"[encode] {i+1}/{N}")
	return feat


def main():
	parser = argparse.ArgumentParser(description="Sparse coding + linear classifier on CIFAR-100")
	parser.add_argument("--dataset_root", type=str, default="./cifar-100-python")
	parser.add_argument("--dict_file", type=str, required=True, help="dictionary joblib file from sparse_learn_dict.py")
	parser.add_argument("--out_dir", type=str, default="./cifar_regressor/checkpoints/sparse/run")
	parser.add_argument("--stride", type=int, default=4)
	parser.add_argument("--alpha", type=float, default=0.1, help="sparse_encode alpha")
	parser.add_argument("--algo", type=str, default="omp", choices=["omp", "lasso_lars", "cd", "lars"])
	parser.add_argument("--limit_train", type=int, default=0, help="debug: limit train images")
	parser.add_argument("--limit_test", type=int, default=0, help="debug: limit test images")
	parser.add_argument("--clf", type=str, default="linear_svc", choices=["linear_svc", "logreg"])
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	d = joblib.load(args.dict_file)
	D = d["D"]  # (K,p)
	patch = int(d["patch"])

	Xtr, ctr, ftr = load_cifar_split(args.dataset_root, "train")
	Xte, cte, fte = load_cifar_split(args.dataset_root, "test")

	# Features
	trN = None if args.limit_train <= 0 else int(args.limit_train)
	teN = None if args.limit_test <= 0 else int(args.limit_test)
	feat_tr = build_dataset_features(D, Xtr, patch=patch, stride=args.stride, alpha=args.alpha, algo=args.algo, limit=trN)
	feat_te = build_dataset_features(D, Xte, patch=patch, stride=args.stride, alpha=args.alpha, algo=args.algo, limit=teN)

	# Standardize
	scaler = StandardScaler(with_mean=True, with_std=True)
	feat_tr = scaler.fit_transform(feat_tr)
	feat_te = scaler.transform(feat_te)

	# Classifiers
	def train_clf(y: np.ndarray):
		if args.clf == "linear_svc":
			clf = LinearSVC(C=1.0, max_iter=2000)
		else:
			clf = LogisticRegression(max_iter=2000, n_jobs=-1)
		clf.fit(feat_tr, y[:feat_tr.shape[0]])
		return clf

	clf_fine = train_clf(ftr)
	clf_coarse = train_clf(ctr)

	# Evaluate
	def top1(clf, X, y):
		pred = clf.predict(X)
		return float((pred == y[:X.shape[0]]).mean())

	acc_f = top1(clf_fine, feat_te, fte)
	acc_c = top1(clf_coarse, feat_te, cte)
	print(f"[RESULT] fine_top1={acc_f:.4f} coarse_top1={acc_c:.4f}")

	# Save artifacts
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	base = f"sparse_{os.path.splitext(os.path.basename(args.dict_file))[0]}_{stamp}"
	save_dir = os.path.join(args.out_dir, base)
	os.makedirs(save_dir, exist_ok=True)
	joblib.dump({"scaler": scaler, "clf_fine": clf_fine, "clf_coarse": clf_coarse, "dict_file": args.dict_file, "patch": patch,
				 "stride": int(args.stride), "alpha": float(args.alpha), "algo": args.algo}, os.path.join(save_dir, "model.joblib"))
	with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
		json.dump({"fine_top1": acc_f, "coarse_top1": acc_c, "N_train": feat_tr.shape[0], "N_test": feat_te.shape[0]}, f, ensure_ascii=False, indent=2)
	print(f"[OK] saved sparse model to: {save_dir}")


if __name__ == "__main__":
	main()


