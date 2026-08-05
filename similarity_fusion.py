#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多源相似度矩阵融合（Jaccard + 熵权法）。

输出（写入 data_dir）：
  - drug_fusion_similarity.txt      (n_drugs × n_drugs)
  - target_fusion_similarity.txt    (n_targets × n_targets)

用法：
  python similarity_fusion.py
  python similarity_fusion.py --data-dir data1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

from dataset_utils import (
    TARGET_FUSION_STD,
    DRUG_FUSION_STD,
    fusion_output_paths,
    get_data_dir,
    load_txt_matrix,
)


def jaccard_similarity(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    B = np.zeros((n, n), dtype=np.float64)
    for i in tqdm(range(n), desc="Jaccard"):
        for j in range(i + 1, n):
            if np.sum(A[i]) == 0 and np.sum(A[j]) == 0:
                B[i, j] = 0.0
            else:
                inter = union = 0
                for k in range(A.shape[1]):
                    if A[i, k] == 1 and A[j, k] == 1:
                        inter += 1
                        union += 1
                    elif A[i, k] == 1 or A[j, k] == 1:
                        union += 1
                B[i, j] = inter / union if union else 0.0
    row, col = np.diag_indices_from(B)
    B[row, col] = 1.0
    B += B.T - np.diag(B.diagonal())
    return B


def calculate_entropy(matrix: np.ndarray) -> float:
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized = matrix / row_sums
    entropy = -np.sum(normalized * np.log(normalized + 1e-10), axis=1)
    return float(entropy.mean())


def entropy_weight(similarity_matrices: list[np.ndarray]) -> np.ndarray:
    entropies = [calculate_entropy(m) for m in similarity_matrices]
    weights = [0.0 if e == 0 else 1.0 / e for e in entropies]
    weights = np.array(weights, dtype=np.float64)
    return weights / weights.sum()


def fuse_matrices(matrices: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    fused = np.zeros_like(matrices[0], dtype=np.float64)
    for mat, w in zip(matrices, weights):
        fused += w * mat
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse drug/target similarity matrices.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Data directory (default: ACMPPL_DATA_DIR or ./data).",
    )
    args = parser.parse_args()

    data_dir = get_data_dir(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    print(f"数据目录: {data_dir}")

    # ---------- 药物 ----------
    dd = load_txt_matrix(os.path.join(data_dir, "mat_drug_drug.txt"))
    di = load_txt_matrix(os.path.join(data_dir, "mat_drug_disease.txt"))
    ds = load_txt_matrix(os.path.join(data_dir, "mat_drug_se.txt"))
    dt = load_txt_matrix(os.path.join(data_dir, "mat_drug_protein.txt"))
    n_drugs, n_targets = dt.shape
    drug_chem = load_txt_matrix(os.path.join(data_dir, "Similarity_Matrix_Drugs.txt"))

    if drug_chem.shape != (n_drugs, n_drugs):
        raise ValueError(
            f"Similarity_Matrix_Drugs 形状 {drug_chem.shape} 与药物数 {n_drugs} 不一致"
        )

    print(f"药物数={n_drugs}, 靶点数={n_targets}")
    print("计算药物 Jaccard 相似度 ...")
    drug_mats = [
        jaccard_similarity(dd),
        jaccard_similarity(di),
        jaccard_similarity(ds),
        drug_chem,
    ]
    drug_weights = entropy_weight(drug_mats)
    print("药物权重:", [f"{w:.4f}" for w in drug_weights])
    drug_fused = fuse_matrices(drug_mats, drug_weights)

    # ---------- 靶点 ----------
    ti = load_txt_matrix(os.path.join(data_dir, "mat_protein_disease.txt"))
    tt = load_txt_matrix(os.path.join(data_dir, "mat_protein_protein.txt"))
    target_seq = load_txt_matrix(os.path.join(data_dir, "Similarity_Matrix_Proteins.txt"))

    if target_seq.shape != (n_targets, n_targets):
        raise ValueError(
            f"Similarity_Matrix_Proteins 形状 {target_seq.shape} 与靶点数 {n_targets} 不一致"
        )

    print("计算靶点 Jaccard 相似度 ...")
    target_mats = [
        jaccard_similarity(ti),
        jaccard_similarity(tt),
        target_seq,
    ]
    target_weights = entropy_weight(target_mats)
    print("靶点权重:", [f"{w:.4f}" for w in target_weights])
    target_fused = fuse_matrices(target_mats, target_weights)

    drug_out, target_out = fusion_output_paths(data_dir)
    np.savetxt(drug_out, drug_fused, fmt="%.6f")
    np.savetxt(target_out, target_fused, fmt="%.6f")

    print(f"已保存: {drug_out} {drug_fused.shape}")
    print(f"已保存: {target_out} {target_fused.shape}")
    print("similarity_fusion 完成。")


if __name__ == "__main__":
    main()
