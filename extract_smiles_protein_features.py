#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 data/ 下 CSV 读取药物 SMILES 与蛋白质序列，提取 256 维特征并保存：
  - drug_smiles_feat.npy
  - protein_seq_feat.npy

用法：
  python extract_smiles_protein_features.py
  python extract_smiles_protein_features.py --data-dir data1
  python extract_smiles_protein_features.py --force   # 强制重新提取
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import contextlib

import dgl
import numpy as np
import pandas as pd
import torch
from dgllife.utils import CanonicalAtomFeaturizer, mol_to_bigraph
from rdkit import Chem

from dataset_utils import (
    get_data_dir,
    load_dti_matrix,
    try_get_seq_features,
)
from utils import integer_label_protein, set_seed
from utilsGCN import MolecularGCN, ProteinSequenceEncoder

DRUG_HIDDEN_DIM = 256
PROTEIN_INTERNAL_DIM = 128
PROTEIN_OUTPUT_DIM = 256
FEAT_DIM = 256
MAX_DRUG_NODES = 290
DRUG_BATCH_SIZE = 64

DRUG_CSV_NAMES = ("drug_info.csv", "luo_drug_smiles.csv", "drug_smiles.csv")
PROTEIN_CSV_NAMES = (
    "protein_sequence.csv",
    "protein_info.csv",
    "luo_protein_sequences.csv",
    "protein_sequences.csv",
)
SMILES_COLUMNS = ("SMILES", "smiles", "Smiles", "canonical_smiles", "drug_smiles")
PROTEIN_COLUMNS = ("Protein", "protein", "sequence", "Sequence", "seq", "amino_acid")

logger = logging.getLogger(__name__)


def get_project_dirs(data_dir_arg: str | None = None) -> tuple[str, str]:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = get_data_dir(data_dir_arg)
    return project_root, data_dir


def resolve_csv(data_dir: str, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"未找到 CSV 文件，尝试过: {[os.path.join(data_dir, n) for n in candidates]}"
    )


def pick_column(df: pd.DataFrame, candidates: tuple[str, ...], file_path: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"{file_path} 缺少列 {candidates}，当前列: {list(df.columns)}")


def _empty_mol_graph(feat_size: int) -> dgl.DGLGraph:
    g = dgl.graph((torch.tensor([0], dtype=torch.int64), torch.tensor([0], dtype=torch.int64)), num_nodes=1)
    g.ndata["h"] = torch.zeros(1, feat_size, dtype=torch.float32)
    return g


@contextlib.contextmanager
def _patch_dgl_graph_int64():
    """dgllife 构图时使用 int32 边索引，新版 DGL 要求 int64。"""
    orig_graph = dgl.graph

    def patched_graph(data, *args, **kwargs):
        if isinstance(data, (tuple, list)) and len(data) == 2:
            u, v = data
            u = torch.as_tensor(u, dtype=torch.int64)
            v = torch.as_tensor(v, dtype=torch.int64)
            return orig_graph((u, v), *args, **kwargs)
        return orig_graph(data, *args, **kwargs)

    dgl.graph = patched_graph
    try:
        yield
    finally:
        dgl.graph = orig_graph


def smiles_to_dgl_graph(smi: str, node_featurizer: CanonicalAtomFeaturizer) -> dgl.DGLGraph:
    """RDKit + dgllife 构图（兼容 DGL int64）。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit 无法解析 SMILES: {smi[:80]}")

    if mol.GetNumAtoms() == 0:
        raise ValueError("分子无原子")

    with _patch_dgl_graph_int64():
        g = mol_to_bigraph(
            mol,
            node_featurizer=node_featurizer,
            edge_featurizer=None,
            add_self_loop=True,
        )
    if "h" not in g.ndata:
        raise ValueError("分子图缺少节点特征 ndata['h']")
    for key, val in g.ndata.items():
        if val.dtype == torch.int64:
            g.ndata[key] = val.to(torch.int32)
    return g


def pad_graph_to_max(g: dgl.DGLGraph, max_nodes: int) -> dgl.DGLGraph:
    """将分子图 pad/truncate 到固定节点数，供 MolecularGCN 按 (B, N, D) reshape。"""
    n = g.num_nodes()
    feat_size = g.ndata["h"].shape[1]

    if n > max_nodes:
        g = dgl.node_subgraph(g, torch.arange(max_nodes, dtype=torch.int64))
        n = g.num_nodes()

    if n == max_nodes:
        return dgl.add_self_loop(g)

    new_g = dgl.graph(([], []), num_nodes=max_nodes)
    new_g.ndata["h"] = torch.zeros(max_nodes, feat_size, dtype=g.ndata["h"].dtype)
    new_g.ndata["h"][:n] = g.ndata["h"]

    if g.num_edges() > 0:
        src, dst = g.edges()
        new_g.add_edges(src.long(), dst.long())

    return dgl.add_self_loop(new_g)


def build_drug_graphs(smiles_list: list[str], max_nodes: int = MAX_DRUG_NODES):
    """将 SMILES 列表转为 DGL 分子图；无效 SMILES 用空图占位。"""
    node_featurizer = CanonicalAtomFeaturizer()
    feat_size = node_featurizer.feat_size()
    graphs = []
    invalid = 0

    for smi in smiles_list:
        smi = (smi or "").strip()
        if not smi or smi.lower() == "nan":
            graphs.append(pad_graph_to_max(_empty_mol_graph(feat_size), max_nodes))
            invalid += 1
            continue
        try:
            g = smiles_to_dgl_graph(smi, node_featurizer)
            if g.num_nodes() > max_nodes:
                g = dgl.node_subgraph(g, torch.arange(max_nodes, dtype=torch.int64))
                g = dgl.add_self_loop(g)
            graphs.append(pad_graph_to_max(g, max_nodes))
        except Exception as exc:
            logger.warning("无效 SMILES，使用空图占位: %s | %s", smi[:40], exc)
            graphs.append(pad_graph_to_max(_empty_mol_graph(feat_size), max_nodes))
            invalid += 1

    if invalid:
        logger.warning("共 %d 个无效/空 SMILES 已用空图占位", invalid)
    logger.info(
        "分子图已统一 pad 到 %d 节点 | 成功=%d, 占位=%d",
        max_nodes,
        len(smiles_list) - invalid,
        invalid,
    )
    return graphs, invalid


@torch.no_grad()
def extract_features(data_dir: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    label_path = os.path.join(data_dir, "mat_drug_protein.txt")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"未找到标签矩阵: {label_path}")

    label_mat, n_drugs, n_targets = load_dti_matrix(data_dir)

    drug_csv = resolve_csv(data_dir, DRUG_CSV_NAMES)
    protein_csv = resolve_csv(data_dir, PROTEIN_CSV_NAMES)
    drug_df = pd.read_csv(drug_csv)
    prot_df = pd.read_csv(protein_csv)

    smiles_col = pick_column(drug_df, SMILES_COLUMNS, drug_csv)
    protein_col = pick_column(prot_df, PROTEIN_COLUMNS, protein_csv)

    drug_smiles = drug_df[smiles_col].astype(str).tolist()
    prot_seqs = prot_df[protein_col].astype(str).tolist()

    if len(drug_smiles) != n_drugs:
        raise ValueError(f"药物数量不匹配: SMILES={len(drug_smiles)}, label={n_drugs}")
    if len(prot_seqs) != n_targets:
        raise ValueError(f"靶点数量不匹配: sequences={len(prot_seqs)}, label={n_targets}")

    logger.info("构建 %d 个分子图 ...", n_drugs)
    drug_graph_list, _ = build_drug_graphs(drug_smiles)

    sample_graph = dgl.batch([drug_graph_list[0]]).to(device)
    in_feats = sample_graph.ndata["h"].shape[1]
    hidden_feats = [DRUG_HIDDEN_DIM, DRUG_HIDDEN_DIM, DRUG_HIDDEN_DIM]

    drug_encoder = MolecularGCN(
        in_feats=in_feats,
        dim_embedding=DRUG_HIDDEN_DIM,
        padding=True,
        hidden_feats=hidden_feats,
        activation=None,
    ).to(device)
    drug_encoder.eval()

    drug_feat_chunks: list[torch.Tensor] = []
    for start in range(0, len(drug_graph_list), DRUG_BATCH_SIZE):
        chunk = drug_graph_list[start : start + DRUG_BATCH_SIZE]
        batch_graph = dgl.batch([g.clone() for g in chunk]).to(device)
        node_feats = drug_encoder(batch_graph)
        drug_feat_chunks.append(torch.max(node_feats, dim=1).values.cpu())
    drug_features = torch.cat(drug_feat_chunks, dim=0).numpy()

    prot_ids = [torch.tensor(integer_label_protein(seq), dtype=torch.long) for seq in prot_seqs]
    prot_pad = torch.nn.utils.rnn.pad_sequence(prot_ids, batch_first=True, padding_value=0).to(device)

    protein_encoder = ProteinSequenceEncoder(
        internal_dim=PROTEIN_INTERNAL_DIM,
        output_dim=PROTEIN_OUTPUT_DIM,
        num_heads=4,
        transformer_layers=2,
        padding=True,
    ).to(device)
    protein_encoder.eval()

    protein_features = protein_encoder(prot_pad).cpu().numpy()

    if drug_features.shape != (n_drugs, FEAT_DIM):
        raise ValueError(f"药物特征维度异常: {drug_features.shape}, 期望 ({n_drugs}, {FEAT_DIM})")
    if protein_features.shape != (n_targets, FEAT_DIM):
        raise ValueError(f"蛋白质特征维度异常: {protein_features.shape}, 期望 ({n_targets}, {FEAT_DIM})")

    return drug_features.astype(np.float32), protein_features.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SMILES/protein sequence features.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Data directory (default: ACMPPL_DATA_DIR or ./data).",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if valid feature files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()
    set_seed(args.seed)
    _, data_dir = get_project_dirs(args.data_dir)
    device = torch.device(args.device)

    logger.info("数据目录: %s", data_dir)
    logger.info("设备: %s", device)

    existing = try_get_seq_features(data_dir)
    if existing is not None and not args.force:
        drug_path, prot_path, drug_shape, prot_shape = existing
        logger.info("已存在特征文件，跳过提取:")
        logger.info("  %s %s", drug_path, drug_shape or "(unknown shape)")
        logger.info("  %s %s", prot_path, prot_shape or "(unknown shape)")
        return

    drug_feat, protein_feat = extract_features(data_dir, device)

    drug_out = os.path.join(data_dir, "drug_smiles_feat.npy")
    protein_out = os.path.join(data_dir, "protein_seq_feat.npy")
    np.save(drug_out, drug_feat)
    np.save(protein_out, protein_feat)

    logger.info("已保存: %s %s", drug_out, drug_feat.shape)
    logger.info("已保存: %s %s", protein_out, protein_feat.shape)


if __name__ == "__main__":
    main()
