import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from dataset_utils import get_data_dir, load_txt_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SemanticAttention(nn.Module):
    """对多条 metapath 矩阵做语义注意力融合"""

    def __init__(self, in_size: int, hidden_size: int = 32):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z: torch.Tensor):
        """
        参数:
            z: [num_paths, N_drug, N_target]
        返回:
            fused: [N_drug, N_target]
        """
        # project -> [num_paths, N_drug, 1] → [num_paths, N_drug]
        w = self.project(z).squeeze(-1)
        # 对每条路径在药物维度 softmax
        beta = torch.softmax(w, dim=1).unsqueeze(-1)  # [num_paths, N_drug, 1]
        attention_product = beta * z                  # [num_paths, N_drug, N_target]
        fused = attention_product.sum(0)              # [N_drug, N_target]
        return fused


def main():
    parser = argparse.ArgumentParser(description="Metapath-PU Learning")
    parser.add_argument("--alpha", type=float, default=None, help="绝对阈值 alpha_PU（与 --beta 同时给出）")
    parser.add_argument("--beta", type=float, default=None, help="绝对阈值 beta_PU")
    parser.add_argument(
        "--p-quantile",
        type=float,
        default=0.99,
        help="P 得分分位数：支持 0~1(如0.99)或0~100(如99)。默认0.99",
    )
    parser.add_argument(
        "--u-quantile",
        type=float,
        default=0.01,
        help="U 得分分位数：支持 0~1(如0.01)或0~100(如1)。默认0.01",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="数据目录（默认 ACMPPL_DATA_DIR 或 ./data）",
    )
    args = parser.parse_args()

    data_dir = get_data_dir(args.data_dir)
    print(f"数据目录: {data_dir}")

    # === 1. 加载矩阵构造 metapath ===
    print("Loading matrices for metapath construction ...")
    DD = load_txt_matrix(os.path.join(data_dir, "mat_drug_drug.txt"))
    DS = load_txt_matrix(os.path.join(data_dir, "mat_drug_se.txt"))
    DI = load_txt_matrix(os.path.join(data_dir, "mat_drug_disease.txt"))
    TT = load_txt_matrix(os.path.join(data_dir, "mat_protein_protein.txt"))
    DT = load_txt_matrix(os.path.join(data_dir, "mat_drug_protein.txt"))
    TI = load_txt_matrix(os.path.join(data_dir, "mat_protein_disease.txt"))

    N_drug, N_target = DT.shape
    print(f"DT shape: {DT.shape} (N_drug={N_drug}, N_target={N_target})")

    DIT = np.dot(DI, TI.T)
    DDT = np.dot(DD, DT)
    DTT = np.dot(DT, TT)
    DSDT = np.dot(np.dot(DS, DS.T), DT)
    DIDT = np.dot(np.dot(DI, DI.T), DT)
    DTDT = np.dot(np.dot(DT, DT.T), DT)
    DITT = np.dot(np.dot(DI, TI.T), TT)
    DDIT = np.dot(np.dot(DD, DI), TI.T)

    # === 2. 语义注意力融合 ===
    print("Running SemanticAttention over 8 metapath matrices ...")
    matrices = [DITT, DDIT, DIDT, DSDT, DIT, DTT, DDT, DTDT]
    z_np = np.stack(matrices, axis=0)
    z = torch.tensor(z_np, dtype=torch.float32, device=device)

    attn_model = SemanticAttention(in_size=N_target).to(device)
    fused_tensor = attn_model(z)
    sum_attention = fused_tensor.detach().cpu().numpy()

    print(f"sum_attention shape: {sum_attention.shape}")

    # === 3. 使用 XJH 的 mat_drug_protein 作为标签矩阵 ===
    print("Loading XJH label matrix as P/U labels ...")
    label_mat = load_txt_matrix(os.path.join(data_dir, "mat_drug_protein.txt"))
    if label_mat.shape != sum_attention.shape:
        raise ValueError(
            f"标签矩阵形状 {label_mat.shape} 与 sum_attention 形状 {sum_attention.shape} 不一致，"
            "请检查药物/靶点的顺序是否完全一致。"
        )

    # P / U 索引
    P_indices = list(zip(*np.where(label_mat == 1)))
    U_indices = list(zip(*np.where(label_mat == 0)))

    P_scores = np.array([sum_attention[i, j] for (i, j) in P_indices])
    U_scores = np.array([sum_attention[i, j] for (i, j) in U_indices])

    print("=" * 60)
    print("  Metapath sum_attention 数值分布统计")
    print("=" * 60)
    print(f"\n【已知正样本 P ({len(P_scores)}个)】")
    print(f"  均值: {P_scores.mean():.6f}")
    print(f"  范围: [{P_scores.min():.6f}, {P_scores.max():.6f}]")
    for q in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {q:3d}% 分位数: {np.percentile(P_scores, q):.6f}")

    print(f"\n【未标注样本 U ({len(U_scores)}个)】")
    print(f"  均值: {U_scores.mean():.6f}")
    print(f"  范围: [{U_scores.min():.6f}, {U_scores.max():.6f}]")
    for q in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {q:3d}% 分位数: {np.percentile(U_scores, q):.6f}")

    # === 4. PU 阈值：绝对值 (--alpha/--beta) 或 分位数 ===
    P_count = len(P_indices)
    if args.alpha is not None and args.beta is not None:
        alpha_PU = float(args.alpha)
        beta_PU = float(args.beta)
        thresh_desc = f"绝对阈值 alpha_PU={alpha_PU}, beta_PU={beta_PU}"
    elif args.alpha is None and args.beta is None:
        # 允许用户传 0~1 或 0~100 两种写法
        p_q = float(args.p_quantile)
        u_q = float(args.u_quantile)
        if 0.0 <= p_q <= 1.0:
            p_q = p_q * 100.0
        if 0.0 <= u_q <= 1.0:
            u_q = u_q * 100.0

        alpha_PU = max(np.percentile(P_scores, p_q), 1e-6)
        beta_PU = max(np.percentile(U_scores, u_q), 1e-6)
        thresh_desc = f"P {p_q:g}% / U {u_q:g}% 分位数"
    else:
        raise ValueError("请同时提供 --alpha 与 --beta，或两者都不提供以使用分位数模式")

    if alpha_PU <= beta_PU:
        print(
            f"警告：alpha_PU ({alpha_PU:.6f}) <= beta_PU ({beta_PU:.6f})，"
            "调整 beta_PU 为 alpha_PU 的 0.9 倍"
        )
        beta_PU = alpha_PU * 0.9
        print(f"调整后的 beta_PU: {beta_PU:.6f}")

    # 在 U 上划分 RP / RN / LN
    RP_list = [(i, j) for (i, j) in U_indices if sum_attention[i, j] >= alpha_PU]
    RN_list = [(i, j) for (i, j) in U_indices if sum_attention[i, j] <= beta_PU]
    LN_list = [
        (i, j)
        for (i, j) in U_indices
        if beta_PU < sum_attention[i, j] < alpha_PU
    ]

    RP_flat = np.array([i * N_target + j for (i, j) in RP_list], dtype=np.int64)
    RN_flat = np.array([i * N_target + j for (i, j) in RN_list], dtype=np.int64)
    LN_flat = np.array([i * N_target + j for (i, j) in LN_list], dtype=np.int64)

    print("=" * 60)
    print("         PU样本分级数量统计结果 (Metapath)")
    print("=" * 60)
    print(thresh_desc)
    print(f"alpha_PU = {alpha_PU:.6f}")
    print(f"beta_PU  = {beta_PU:.6f}")
    print(f"已知正样本 (P):    {P_count}")
    print(f"可靠正样本 (RP):   {len(RP_list)}")
    print(f"疑似负样本 (LN):   {len(LN_list)}")
    print(f"可靠负样本 (RN):   {len(RN_list)}")
    print("=" * 60)
    print(f"总样本数: {P_count + len(RP_list) + len(LN_list) + len(RN_list)}")

    # === 5. 保存为后续流程使用的文件 ===

    # 1) sum_attention 作为新的 Y_fused_Meta
    y_fused_path = os.path.join(data_dir, "Y_fused_Meta.txt")
    np.savetxt(y_fused_path, sum_attention, delimiter="\t", fmt="%.6f")
    print("Y_fused_Meta.txt 已保存:", y_fused_path)

    # 2) RP / RN / LN 索引
    rn_indices_path = os.path.join(data_dir, "RN_indices.npy")
    rp_indices_path = os.path.join(data_dir, "RP_indices.npy")
    ln_indices_path = os.path.join(data_dir, "LN_indices.npy")
    np.save(rn_indices_path, RN_flat)
    np.save(rp_indices_path, RP_flat)
    np.save(ln_indices_path, LN_flat)
    print(f"RN_indices.npy 已保存: {rn_indices_path}（{len(RN_flat)} 个）")
    print(f"RP_indices.npy 已保存: {rp_indices_path}（{len(RP_flat)} 个）")
    print(f"LN_indices.npy 已保存: {ln_indices_path}（{len(LN_flat)} 个）")

    # 3) PU 阈值（供 classifier / grid_search 直接读取，保证前后阈值一致）
    thresholds_path = os.path.join(data_dir, "PU_thresholds.npy")
    np.save(thresholds_path, np.array([alpha_PU, beta_PU], dtype=np.float64))
    print(f"PU_thresholds.npy 已保存: {thresholds_path}")

    print("\nMetapath-PU Learning 完成。")


if __name__ == "__main__":
    main()
