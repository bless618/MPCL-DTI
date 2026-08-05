#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集路径与矩阵加载工具（自适应药物/靶点数量）。

环境变量：
  ACMPPL_DATA_DIR  数据目录，默认 <project>/data
  ACMPPL_WORK_DIR  项目根目录，默认本文件所在目录

融合相似度标准输出文件名：
  drug_fusion_similarity.txt
  target_fusion_similarity.txt
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DRUG_FUSION_STD = "drug_fusion_similarity.txt"
TARGET_FUSION_STD = "target_fusion_similarity.txt"

DRUG_FUSION_LEGACY = "drug_fusion_similarity_708_708.txt"
TARGET_FUSION_LEGACY = "target_fusion_similarity_3_1512_1512.txt"


def get_data_dir(explicit: str | None = None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    if os.environ.get("ACMPPL_DATA_DIR"):
        return os.path.abspath(os.environ["ACMPPL_DATA_DIR"])
    return os.path.join(PROJECT_ROOT, "data")


def get_work_dir() -> str:
    return os.path.abspath(os.environ.get("ACMPPL_WORK_DIR", PROJECT_ROOT))


def load_txt_matrix(path: str) -> np.ndarray:
    """
    按空白分隔读入数值矩阵。较短行右侧补 0，避免 loadtxt 因列数不一致失败。
    """
    rows: List[np.ndarray] = []
    src_lines: List[int] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if not parts:
                continue
            try:
                rows.append(np.array([float(x) for x in parts], dtype=np.float64))
            except ValueError as exc:
                raise ValueError(f"{path} 第 {line_no} 行含非数值字段: {exc}") from exc
            src_lines.append(line_no)
    if not rows:
        raise ValueError(f"无有效数据行: {path}")

    lens = [len(r) for r in rows]
    n_col = max(lens)
    mode_len = max(set(lens), key=lens.count)
    if len(set(lens)) > 1:
        odd = [src_lines[i] for i, L in enumerate(lens) if L != mode_len]
        print(
            f"警告: {os.path.basename(path)} 各行元素个数不一致 "
            f"(众数={mode_len}, 最大列数={n_col})，已对较短行右侧补 0。"
            f"与众数不同的行号(最多显示30个): {odd[:30]}{'...' if len(odd) > 30 else ''}"
        )
    out = np.zeros((len(rows), n_col), dtype=np.float64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def load_dti_matrix(data_dir: str | None = None) -> Tuple[np.ndarray, int, int]:
    """返回 (label_mat, n_drugs, n_targets)。"""
    data_dir = get_data_dir(data_dir)
    path = os.path.join(data_dir, "mat_drug_protein.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到标签矩阵: {path}")
    label = load_txt_matrix(path)
    n_drugs, n_targets = label.shape
    return label, n_drugs, n_targets


def _candidate_fusion_paths(data_dir: str, kind: str) -> List[str]:
    """kind: 'drug' | 'target'"""
    if kind == "drug":
        patterns = [
            DRUG_FUSION_STD,
            DRUG_FUSION_LEGACY,
            "drug_fusion_similarity_*.txt",
        ]
    else:
        patterns = [
            TARGET_FUSION_STD,
            TARGET_FUSION_LEGACY,
            "target_fusion_similarity_*.txt",
        ]

    seen: List[str] = []
    for pat in patterns:
        full_pat = os.path.join(data_dir, pat)
        matches = sorted(glob.glob(full_pat)) if "*" in pat else ([full_pat] if os.path.isfile(full_pat) else [])
        for p in matches:
            p = os.path.abspath(p)
            if p not in seen:
                seen.append(p)
    return seen


def find_fusion_file(data_dir: str, kind: str, expected_shape: Tuple[int, int]) -> str:
    """
    在 data_dir 中查找形状匹配的融合相似度矩阵文件。
    kind: 'drug' -> (n_drugs, n_drugs), 'target' -> (n_targets, n_targets)
    """
    candidates = _candidate_fusion_paths(data_dir, kind)
    if not candidates:
        raise FileNotFoundError(
            f"未找到{kind}融合相似度文件。请先运行 similarity_fusion.py，"
            f"或在 {data_dir} 下放置 {DRUG_FUSION_STD if kind == 'drug' else TARGET_FUSION_STD}。"
        )

    matched: List[str] = []
    mismatched: List[str] = []
    for path in candidates:
        try:
            mat = load_txt_matrix(path)
        except Exception as exc:
            mismatched.append(f"{os.path.basename(path)} (读取失败: {exc})")
            continue
        if mat.shape == expected_shape:
            matched.append(path)
        else:
            mismatched.append(f"{os.path.basename(path)} (shape={mat.shape}, 期望={expected_shape})")

    if matched:
        # 优先标准文件名
        for path in matched:
            if os.path.basename(path) in (DRUG_FUSION_STD, TARGET_FUSION_STD):
                return path
        return matched[0]

    detail = "\n  - ".join(mismatched) if mismatched else "无候选文件"
    raise FileNotFoundError(
        f"未找到形状为 {expected_shape} 的{kind}融合相似度矩阵。\n"
        f"已检查:\n  - {detail}\n"
        f"请运行: python similarity_fusion.py --data-dir {data_dir}"
    )


def load_similarity_matrices(
    data_dir: str | None,
    n_drugs: int,
    n_targets: int,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """加载 SD、ST，返回 (SD, ST, drug_path, target_path)。"""
    data_dir = get_data_dir(data_dir)
    drug_path = find_fusion_file(data_dir, "drug", (n_drugs, n_drugs))
    target_path = find_fusion_file(data_dir, "target", (n_targets, n_targets))
    sd = load_txt_matrix(drug_path)
    st = load_txt_matrix(target_path)
    return sd, st, drug_path, target_path


def try_get_fusion_files(
    data_dir: str | None = None,
) -> Optional[Tuple[str, str, int, int]]:
    """
    若已存在形状匹配的融合相似度文件，返回 (drug_path, target_path, n_drugs, n_targets)；
    否则返回 None。
    """
    data_dir = get_data_dir(data_dir)
    try:
        _, n_drugs, n_targets = load_dti_matrix(data_dir)
        drug_path = find_fusion_file(data_dir, "drug", (n_drugs, n_drugs))
        target_path = find_fusion_file(data_dir, "target", (n_targets, n_targets))
        return drug_path, target_path, n_drugs, n_targets
    except FileNotFoundError:
        return None


def fusion_output_paths(data_dir: str | None = None) -> Tuple[str, str]:
    data_dir = get_data_dir(data_dir)
    return (
        os.path.join(data_dir, DRUG_FUSION_STD),
        os.path.join(data_dir, TARGET_FUSION_STD),
    )


DRUG_SEQ_FEAT_FILE = "drug_smiles_feat.npy"
PROTEIN_SEQ_FEAT_FILE = "protein_seq_feat.npy"
SEQ_FEAT_DIM = 256


def seq_feature_output_paths(data_dir: str | None = None) -> Tuple[str, str]:
    data_dir = get_data_dir(data_dir)
    return (
        os.path.join(data_dir, DRUG_SEQ_FEAT_FILE),
        os.path.join(data_dir, PROTEIN_SEQ_FEAT_FILE),
    )


def try_get_seq_features(
    data_dir: str | None = None,
    feat_dim: int | None = None,
) -> Optional[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]]]:
    """
    若 drug_smiles_feat.npy 与 protein_seq_feat.npy 均存在则返回
    (drug_path, protein_path, drug_shape, protein_shape)；否则返回 None。

    feat_dim 参数保留兼容，不再参与校验。
    """
    del feat_dim
    data_dir = get_data_dir(data_dir)
    drug_path, prot_path = seq_feature_output_paths(data_dir)
    if not (os.path.isfile(drug_path) and os.path.isfile(prot_path)):
        return None

    drug_shape: Tuple[int, ...] = ()
    prot_shape: Tuple[int, ...] = ()
    try:
        drug = np.load(drug_path, mmap_mode="r")
        prot = np.load(prot_path, mmap_mode="r")
        drug_shape = tuple(int(x) for x in drug.shape)
        prot_shape = tuple(int(x) for x in prot.shape)
    except (OSError, ValueError):
        pass
    return drug_path, prot_path, drug_shape, prot_shape


def describe_seq_features(
    data_dir: str | None = None,
    feat_dim: int | None = None,
) -> str:
    """说明序列特征文件是否存在（用于日志/预检提示）。"""
    del feat_dim
    data_dir = get_data_dir(data_dir)
    drug_path, prot_path = seq_feature_output_paths(data_dir)
    if not os.path.isfile(drug_path):
        return f"缺少 {DRUG_SEQ_FEAT_FILE}"
    if not os.path.isfile(prot_path):
        return f"缺少 {PROTEIN_SEQ_FEAT_FILE}"
    try:
        drug = np.load(drug_path, mmap_mode="r")
        prot = np.load(prot_path, mmap_mode="r")
        return (
            f"已存在: drug={tuple(int(x) for x in drug.shape)}, "
            f"protein={tuple(int(x) for x in prot.shape)}"
        )
    except Exception as exc:
        return f"已存在（无法读取 shape: {exc}）"
