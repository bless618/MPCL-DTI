#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MPCL-DTI 一键运行脚本。

完整流程：
  Step 0: similarity_fusion.py（若已有融合文件则自动跳过）
          -> drug_fusion_similarity.txt, target_fusion_similarity.txt
  Step 1: Metapath_PU_Learning.py
  Step 2: xiaorongshiyangroup_divide.py（P:RP:RN:LN=1:1:1:1）
  Step 3: extract_smiles_protein_features.py（若已有特征文件则自动跳过）
  Step 4: model_GCN_original_structure.py（5 折）
  Step 5: xiaorongshiyanclassifier.py（5 折）
  Step 6: xiaorongshiyanaggregate.py

示例：
  python run.py
  python run.py --data-dir data1
  python run.py --disable-seq --from-step 4
  python run.py --gcn-epochs 50 --from-step 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import Iterable, List

from dataset_utils import (
    find_fusion_file,
    load_dti_matrix,
    try_get_fusion_files,
    try_get_seq_features,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SCRIPT_SIMILARITY_FUSION = "similarity_fusion.py"
SCRIPT_METAPATH = "Metapath_PU_Learning.py"
SCRIPT_GROUP_DIVIDE = "xiaorongshiyangroup_divide.py"
SCRIPT_EXTRACT_SEQ = "extract_smiles_protein_features.py"
SCRIPT_GCN = "model_GCN_original_structure.py"
SCRIPT_CLASSIFIER = "xiaorongshiyanclassifier.py"
SCRIPT_AGGREGATE = "xiaorongshiyanaggregate.py"

DRUG_CSV_CANDIDATES = ("drug_info.csv", "luo_drug_smiles.csv", "drug_smiles.csv")
PROTEIN_CSV_CANDIDATES = (
    "protein_sequence.csv",
    "protein_info.csv",
    "luo_protein_sequences.csv",
    "protein_sequences.csv",
)

SIMILARITY_FUSION_INPUTS = (
    "mat_drug_protein.txt",
    "mat_drug_drug.txt",
    "mat_drug_disease.txt",
    "mat_drug_se.txt",
    "mat_protein_protein.txt",
    "mat_protein_disease.txt",
    "Similarity_Matrix_Drugs.txt",
    "Similarity_Matrix_Proteins.txt",
)

METAPATH_INPUTS = (
    "mat_drug_protein.txt",
    "mat_drug_drug.txt",
    "mat_drug_disease.txt",
    "mat_drug_se.txt",
    "mat_protein_protein.txt",
    "mat_protein_disease.txt",
)

N_FOLDS = 5
LAST_STEP = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MPCL-DTI pipeline end-to-end.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory relative to project root, or an absolute path (default: data).",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=0,
        choices=list(range(0, LAST_STEP + 1)),
        help="Start from this step (default: 0 = full pipeline).",
    )
    parser.add_argument(
        "--disable-seq",
        action="store_true",
        help="Skip Step 3 and pass --disable_seq to GCN.",
    )
    parser.add_argument(
        "--gcn-epochs",
        type=int,
        default=5000,
        help="GCN training epochs per fold (default: 5000).",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip prerequisite file checks.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for subprocess calls.",
    )
    return parser.parse_args()


def resolve_data_dir(data_dir: str) -> str:
    if os.path.isabs(data_dir):
        return os.path.abspath(data_dir)
    return os.path.join(PROJECT_ROOT, data_dir)


def data_dir_arg(data_dir: str, cli_data_dir: str) -> str:
    """Pass relative path to child scripts when possible."""
    if os.path.isabs(cli_data_dir):
        return data_dir
    return cli_data_dir


def run_cmd(cmd: List[str], desc: str, env: dict) -> None:
    print(f"[{desc}] {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"{desc} failed.")


def check_files(data_dir: str, relative_paths: Iterable[str], step_hint: str) -> None:
    missing = [
        rel for rel in relative_paths
        if not os.path.isfile(os.path.join(data_dir, rel))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing required files for {step_hint} in {data_dir}:\n  - "
            + "\n  - ".join(missing)
        )


def find_any_csv(data_dir: str, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if os.path.isfile(os.path.join(data_dir, name)):
            return name
    return None


def check_fusion_files(data_dir: str) -> None:
    _, n_drugs, n_targets = load_dti_matrix(data_dir)
    drug_path = find_fusion_file(data_dir, "drug", (n_drugs, n_drugs))
    target_path = find_fusion_file(data_dir, "target", (n_targets, n_targets))
    print(f"Fusion files : {os.path.basename(drug_path)}, {os.path.basename(target_path)}")
    print(f"Matrix shape : drugs={n_drugs}, targets={n_targets}")


def preflight(data_dir: str, disable_seq: bool, from_step: int) -> None:
    print("=" * 60)
    print("Preflight checks")
    print("=" * 60)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data dir     : {data_dir}")
    print(f"Disable seq  : {disable_seq}")
    print(f"From step    : {from_step}")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if from_step <= 0:
        fusion = try_get_fusion_files(data_dir)
        if fusion is None:
            check_files(data_dir, SIMILARITY_FUSION_INPUTS, "Step 0 (similarity_fusion)")
        else:
            drug_path, target_path, n_drugs, n_targets = fusion
            print(
                f"Fusion files : {os.path.basename(drug_path)}, "
                f"{os.path.basename(target_path)} (Step 0 will be skipped)"
            )
            print(f"Matrix shape : drugs={n_drugs}, targets={n_targets}")

    if from_step >= 1:
        check_fusion_files(data_dir)

    if from_step <= 1:
        check_files(data_dir, METAPATH_INPUTS, "Metapath / group_divide")

    if from_step == 2:
        check_files(
            data_dir,
            ["RP_indices.npy", "RN_indices.npy", "LN_indices.npy"],
            "Step 2 group_divide (run Metapath_PU_Learning first)",
        )

    if not disable_seq and from_step <= 3:
        seq_feat = try_get_seq_features(data_dir)
        if seq_feat is not None:
            drug_path, prot_path, drug_shape, prot_shape = seq_feat
            print(
                f"Seq features : {os.path.basename(drug_path)} {drug_shape}, "
                f"{os.path.basename(prot_path)} {prot_shape} (Step 3 will be skipped)"
            )
        else:
            drug_csv = find_any_csv(data_dir, DRUG_CSV_CANDIDATES)
            protein_csv = find_any_csv(data_dir, PROTEIN_CSV_CANDIDATES)
            if drug_csv is None:
                raise FileNotFoundError(
                    f"Step 3 需要药物 SMILES CSV，未找到: {DRUG_CSV_CANDIDATES}"
                )
            if protein_csv is None:
                raise FileNotFoundError(
                    f"Step 3 需要蛋白质序列 CSV，未找到: {PROTEIN_CSV_CANDIDATES}"
                )
            print(f"Step 3 CSV   : {drug_csv}, {protein_csv}")

    if not disable_seq and from_step >= 4:
        check_files(
            data_dir,
            ["drug_smiles_feat.npy", "protein_seq_feat.npy"],
            "Step 4 GCN (run Step 3 first, or use --from-step 3 / --disable-seq)",
        )

    fusion_script = os.path.join(PROJECT_ROOT, SCRIPT_SIMILARITY_FUSION)
    if not os.path.isfile(fusion_script):
        raise FileNotFoundError(f"Missing script: {SCRIPT_SIMILARITY_FUSION}")

    print("Preflight checks passed.\n")


def build_env(data_dir: str) -> dict:
    env = os.environ.copy()
    env["ACMPPL_DATA_DIR"] = data_dir
    env["ACMPPL_WORK_DIR"] = PROJECT_ROOT
    return env


def gcn_args(fold: int, disable_seq: bool, gcn_epochs: int) -> List[str]:
    args = ["-f", str(fold), "--epochs", str(gcn_epochs)]
    if disable_seq:
        args.append("--disable_seq")
    return args


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    dd_arg = data_dir_arg(data_dir, args.data_dir)
    env = build_env(data_dir)
    py = args.python

    if not args.skip_preflight:
        preflight(data_dir, args.disable_seq, args.from_step)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_file = os.path.join(PROJECT_ROOT, ".current_run_timestamp.txt")
    with open(ts_file, "w", encoding="utf-8") as f:
        f.write(timestamp)
    print(f"Run timestamp: {timestamp}")
    print(f"Results will be saved under: results_{timestamp}/")
    print("Split policy: P:RP:RN:LN=1:1:1:1\n")

    if args.from_step <= 0:
        fusion = try_get_fusion_files(data_dir)
        if fusion is not None:
            drug_path, target_path, _, _ = fusion
            print("=" * 60)
            print("Step 0: Similarity fusion (skipped — files already exist)")
            print(f"  Using: {os.path.basename(drug_path)}, {os.path.basename(target_path)}")
            print("=" * 60)
        else:
            print("=" * 60)
            print("Step 0: Similarity fusion")
            print("  Output: drug_fusion_similarity.txt, target_fusion_similarity.txt")
            print("=" * 60)
            run_cmd(
                [py, SCRIPT_SIMILARITY_FUSION, "--data-dir", dd_arg],
                "similarity_fusion",
                env,
            )

    if args.from_step <= 1:
        print("\n" + "=" * 60)
        print("Step 1: Metapath-PU Learning")
        print("  Default thresholds: P 99% quantile / U 1% quantile")
        print("=" * 60)
        run_cmd(
            [py, SCRIPT_METAPATH, "--data-dir", dd_arg],
            "Metapath_PU_Learning",
            env,
        )

    if args.from_step <= 2:
        print("\n" + "=" * 60)
        print("Step 2: Data split (5-fold)")
        print("P:RP:RN:LN=1:1:1:1")
        print("=" * 60)
        run_cmd([py, SCRIPT_GROUP_DIVIDE], "group_divide", env)

    if args.from_step <= 3 and not args.disable_seq:
        seq_feat = try_get_seq_features(data_dir)
        print("\n" + "=" * 60)
        if seq_feat is not None:
            drug_path, prot_path, drug_shape, prot_shape = seq_feat
            print("Step 3: Extract SMILES / protein sequence features (skipped — files already exist)")
            print(f"  Using: {os.path.basename(drug_path)} {drug_shape}, {os.path.basename(prot_path)} {prot_shape}")
        else:
            print("Step 3: Extract SMILES / protein sequence features")
            print("  Output: drug_smiles_feat.npy, protein_seq_feat.npy")
            run_cmd(
                [py, SCRIPT_EXTRACT_SEQ, "--data-dir", dd_arg],
                "extract_smiles_protein_features",
                env,
            )
        print("=" * 60)

    if args.from_step <= 4:
        print("\n" + "=" * 60)
        print("Step 4: GCN contrastive learning (5 folds)")
        print(f"  Epochs per fold: {args.gcn_epochs}")
        print("  Output: <data-dir>/predict_result/embedding_cl_seq{fold}.txt")
        print("=" * 60)
        for fold in range(N_FOLDS):
            run_cmd(
                [py, SCRIPT_GCN, *gcn_args(fold, args.disable_seq, args.gcn_epochs)],
                f"GCN fold {fold}",
                env,
            )

    if args.from_step <= 5:
        print("\n" + "=" * 60)
        print("Step 5: MLP classifier (5 folds)")
        print("=" * 60)
        for fold in range(N_FOLDS):
            run_cmd(
                [py, SCRIPT_CLASSIFIER, "-f", str(fold)],
                f"classifier fold {fold}",
                env,
            )

    if args.from_step <= 6:
        print("\n" + "=" * 60)
        print("Step 6: Aggregate results")
        print("=" * 60)
        run_cmd([py, SCRIPT_AGGREGATE], "aggregate", env)

    results_dir = os.path.join(PROJECT_ROOT, f"results_{timestamp}")
    print("\n" + "=" * 60)
    print("Pipeline finished.")
    print(f"Timestamp : {timestamp}")
    print(f"Results   : {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
