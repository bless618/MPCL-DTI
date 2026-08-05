#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5 折交叉验证数据划分。

固定划分策略：
  - P : RP : RN : LN = 1 : 1 : 1 : 1

输出（位于 data/divide_result/）：
  - index_1.txt
  - train_neg_fold{f}.txt
  - test_neg_fold{f}.txt
  - X{f}.txt, A{f}.txt
"""

import copy
import os
import random

import numpy as np

from dataset_utils import get_data_dir, load_dti_matrix, load_similarity_matrices, load_txt_matrix

FOLD = 5
EXP_DESC = "P:RP:RN:LN=1:1:1:1"

data_dir = get_data_dir()
divide_result_dir = os.path.join(data_dir, "divide_result")
os.makedirs(divide_result_dir, exist_ok=True)

A, m, n = load_dti_matrix(data_dir)
SD, ST, drug_fusion_path, target_fusion_path = load_similarity_matrices(data_dir, m, n)
DDI = load_txt_matrix(os.path.join(data_dir, "mat_drug_drug.txt"))
TTI = load_txt_matrix(os.path.join(data_dir, "mat_protein_protein.txt"))

if SD.shape != (m, m):
    raise ValueError(f"SD 形状不匹配: {SD.shape}, 期望=({m},{m})")
if ST.shape != (n, n):
    raise ValueError(f"ST 形状不匹配: {ST.shape}, 期望=({n},{n})")
if DDI.shape != (m, m):
    raise ValueError(f"DDI 形状不匹配: {DDI.shape}, 期望=({m},{m})")
if TTI.shape != (n, n):
    raise ValueError(f"TTI 形状不匹配: {TTI.shape}, 期望=({n},{n})")

print(f"数据集维度: 药物={m}, 靶点={n}, 图节点={m + n}")
print(f"融合相似度: {os.path.basename(drug_fusion_path)}, {os.path.basename(target_fusion_path)}")
print(f"划分策略: {EXP_DESC}")

# ===================== 样本 5 折划分 =====================
A_flat = A.flatten()
pos_indices = [i for i in range(len(A_flat)) if A_flat[i] == 1]
num_pos = len(pos_indices)
group_size = num_pos // FOLD
if group_size == 0:
    raise ValueError(f"正样本数量不足，无法做 {FOLD} 折划分: num_pos={num_pos}")

random.seed(10)
random.shuffle(pos_indices)
pos_array = np.array(pos_indices[: FOLD * group_size])
grouped_pos = np.reshape(pos_array, (FOLD, group_size))
np.savetxt(os.path.join(divide_result_dir, "index_1.txt"), grouped_pos, fmt="%d")

# ===================== 加载 PU 索引 =====================
for name in ("RP_indices.npy", "RN_indices.npy", "LN_indices.npy"):
    path = os.path.join(data_dir, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"未找到 {path}，请先运行 Metapath_PU_Learning.py 生成 PU 索引。"
        )

RP_flat = np.load(os.path.join(data_dir, "RP_indices.npy"))
RN_flat = np.load(os.path.join(data_dir, "RN_indices.npy"))
LN_flat = np.load(os.path.join(data_dir, "LN_indices.npy"))
print(f"已加载 RP={len(RP_flat)}, RN={len(RN_flat)}, LN={len(LN_flat)}")


def sample_indices(candidates, k, seed):
    if len(candidates) <= k:
        return list(candidates)
    random.seed(seed)
    return list(np.random.permutation(candidates)[:k])


# ===================== 每折采样训练/测试负样本 =====================
for f in range(FOLD):
    test_pos = grouped_pos[f].copy()
    train_pos = np.concatenate([grouped_pos[i] for i in range(FOLD) if i != f])
    num_train_pos = len(train_pos)
    num_test_pos = len(test_pos)

    n_train_rp = num_train_pos
    n_train_rn = num_train_pos
    n_train_ln = num_train_pos

    train_rp = sample_indices(
        [idx for idx in RP_flat if idx not in test_pos],
        n_train_rp,
        10 + f,
    )
    used = set(train_rp)
    train_rn = sample_indices(
        [idx for idx in RN_flat if idx not in test_pos and idx not in used],
        n_train_rn,
        40 + f,
    )
    used.update(train_rn)
    train_ln = sample_indices(
        [idx for idx in LN_flat if idx not in test_pos and idx not in used],
        n_train_ln,
        50 + f,
    )
    train_neg_all = train_rp + train_rn + train_ln

    test_neg = sample_indices(
        [idx for idx in RN_flat if idx not in train_pos and idx not in train_neg_all],
        num_test_pos,
        20 + f,
    )

    np.savetxt(
        os.path.join(divide_result_dir, f"train_neg_fold{f}.txt"),
        np.array(train_neg_all, dtype=int),
        fmt="%d",
    )
    np.savetxt(
        os.path.join(divide_result_dir, f"test_neg_fold{f}.txt"),
        np.array(test_neg, dtype=int),
        fmt="%d",
    )
    print(
        f"第 {f} 折: 训练负样本 {len(train_neg_all)} "
        f"(RP={len(train_rp)}, RN={len(train_rn)}, LN={len(train_ln)}), "
        f"测试负样本 {len(test_neg)}"
    )

print(f"\n数据划分完成 | {EXP_DESC}")

# ===================== 生成 X / A 矩阵（供 GCN 使用） =====================
for f in range(FOLD):
    DTI = copy.deepcopy(A)
    for idx in grouped_pos[f]:
        r = int(idx // n)
        c = int(idx % n)
        DTI[r, c] = 0

    X = np.vstack(
        (
            np.hstack((SD, DTI)),
            np.hstack((DTI.T, ST)),
        )
    )
    adj = np.vstack(
        (
            np.hstack((DDI, DTI)),
            np.hstack((DTI.T, TTI)),
        )
    )
    np.savetxt(os.path.join(divide_result_dir, f"X{f}.txt"), X)
    np.savetxt(os.path.join(divide_result_dir, f"A{f}.txt"), adj)

print(f"X 形状: {X.shape}")
print(f"A 形状: {adj.shape}")
