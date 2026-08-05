import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import copy
import sys
import argparse
import os
import math
import logging
from datetime import datetime

from dataset_utils import get_data_dir, load_dti_matrix, load_txt_matrix
from transformer import get_graph_drop_transform

# ===================== 命令行参数解析 =====================
parser = argparse.ArgumentParser()
parser.add_argument('-f', dest="fold", type=int, required=True, help="指定交叉验证折数 (0-9)")
parser.add_argument('--drop_view1', type=float, default=0.1, help="视图1 对称删边/对称特征丢弃概率")
parser.add_argument('--drop_view2', type=float, default=0.1, help="视图2 对称删边/对称特征丢弃概率")
parser.add_argument('--save_subdir', type=str, default='predict_result', help="输出子目录（位于 data/ 下）")
parser.add_argument('--disable_contrast', action='store_true', help="禁用节点级+子图级对比学习")
parser.add_argument('--disable_subgraph_contrast', action='store_true', help="仅禁用子图级对比学习（保留节点级）")
parser.add_argument('--disable_seq', action='store_true', help="禁用 SMILES/序列特征（不做对齐损失，不拼接序列特征）")
parser.add_argument('--epochs', type=int, default=5000, help="训练 epoch 数（默认 5000）")
parser.add_argument('--log-interval', type=int, default=100, help="每隔多少 epoch 打印一次损失（默认 100）")
results = parser.parse_args()
fold = results.fold
epochs = max(1, results.epochs)
log_interval = max(1, results.log_interval)

# 检查fold参数有效性
if fold is None:
    raise ValueError("错误：必须提供 -f 参数指定折数。使用方法: python model_GCN_original.py -f 0")
if fold < 0 or fold > 9:
    raise ValueError(f"错误：fold参数必须在0-9之间，当前值: {fold}")

# ===================== 日志配置 =====================
_proj_root = os.path.dirname(os.path.abspath(__file__))
_log_dir = os.path.join(_proj_root, "logs")
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, f"gcn_structure_fold{fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"GCN 训练开始 | fold={fold} | epochs={epochs} | log_interval={log_interval}")
logger.info(f"日志文件: {_log_file}")

# ===================== 环境与超参数设置 =====================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
LR = 0.0001

# 维度参数（由数据动态推导）
n = None      # 药物数量
m = None      # 靶点数量
in_features = None
out_features = 256
N_HID = 512

# 对比学习超参数
temperature = 0.1
drop_edge_p1 = results.drop_view1   # 视图1：边删除比例
drop_feat_p1 = results.drop_view1   # 视图1：节点删除比例
drop_edge_p2 = results.drop_view2   # 视图2：边删除比例
drop_feat_p2 = results.drop_view2   # 视图2：节点删除比例
k = 5  # 子图近邻数

# 多尺度损失权重（删除了global_loss_weight）
node_loss_weight = 1.0
subgraph_loss_weight = 1.0

# ===================== 维度确定与（可选）序列特征加载 =====================
_proj_root = os.path.dirname(os.path.abspath(__file__))
_data_dir_env = get_data_dir()
label_mat, n, m = load_dti_matrix(_data_dir_env)  # n=药物数, m=靶点数

drug_smiles_feat = None
protein_seq_feat = None
#seq_align_weight = 0.0 if results.disable_seq else 0.1

if not results.disable_seq:
    drug_smiles_feat_path = os.path.join(_data_dir_env, "drug_smiles_feat.npy")
    protein_seq_feat_path = os.path.join(_data_dir_env, "protein_seq_feat.npy")
    if not (os.path.isfile(drug_smiles_feat_path) and os.path.isfile(protein_seq_feat_path)):
        raise FileNotFoundError(
            f"未找到 {drug_smiles_feat_path} 或 {protein_seq_feat_path}，"
            "请先运行 extract_smiles_protein_features.py 生成 SMILES/序列特征，或使用 --disable_seq。"
        )

    drug_smiles_feat_np = np.load(drug_smiles_feat_path)
    protein_seq_feat_np = np.load(protein_seq_feat_path)
    if drug_smiles_feat_np.shape[0] != n or protein_seq_feat_np.shape[0] != m:
        raise ValueError(
            f"序列特征数量与 label 不匹配: drug_smiles_feat={drug_smiles_feat_np.shape}, "
            f"protein_seq_feat={protein_seq_feat_np.shape}, label={label_mat.shape}"
        )
    drug_smiles_feat = torch.from_numpy(drug_smiles_feat_np).float().to(device)
    protein_seq_feat = torch.from_numpy(protein_seq_feat_np).float().to(device)

# ===================== 工具函数 =====================
def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R[R == 0] = 1e-10
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

def get_subgraph_feature(z, adj, k=5):
    """
    使用随机游走找k个一阶邻居，聚合一阶邻居特征
    Args:
        z: 节点嵌入矩阵 [num_nodes, embedding_dim]
        adj: 邻接矩阵 [num_nodes, num_nodes]
        k: 采样邻居数量
    Returns:
        subgraph_feat: 子图特征 [num_nodes, embedding_dim]
    """
    num_nodes = z.size(0)
    device = z.device
    subgraph_feat_list = []
    
    for i in range(num_nodes):
        # 步骤1：找到所有一阶邻居（直接连接的节点，adj[i] > 0）
        neighbors = torch.nonzero(adj[i] > 0, as_tuple=False).squeeze(1)
        
        if len(neighbors) == 0:
            # 如果没有邻居，使用自身特征
            subgraph_feat_list.append(z[i])
            continue
        
        # 步骤2：从一阶邻居中随机采样k个
        if len(neighbors) > k:
            # 随机采样k个邻居
            sampled_idx = torch.randperm(len(neighbors), device=device)[:k]
            sampled_neighbors = neighbors[sampled_idx]
        else:
            # 如果邻居数少于k，使用所有邻居
            sampled_neighbors = neighbors
        
        # 步骤3：聚合这k个一阶邻居的特征（平均池化）
        neighbor_features = z[sampled_neighbors]  # [k, embedding_dim] 或 [len(neighbors), embedding_dim]
        subgraph_feat_i = torch.mean(neighbor_features, dim=0)  # [embedding_dim]
        subgraph_feat_list.append(subgraph_feat_i)
    
    subgraph_feat = torch.stack(subgraph_feat_list, dim=0)  # [num_nodes, embedding_dim]
    return subgraph_feat

def edge_index_to_adj_matrix(edge_index, edge_weight, num_nodes):
    """
    将edge_index和edge_weight转换为稠密邻接矩阵
    用于子图特征提取时保持与GCN编码的一致性
    """
    adj = torch.zeros(num_nodes, num_nodes).to(edge_index.device)
    if edge_weight is not None:
        adj[edge_index[0], edge_index[1]] = edge_weight
    else:
        adj[edge_index[0], edge_index[1]] = 1.0
    return adj

# 删除了get_global_feature函数（全局特征提取）

# ===================== MSE 损失（仿师姐项目的“修复版”实现） =====================
def mean_square_error(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    通用、鲁棒的 MSE 计算，用法对齐彬滢师姐代码中的“修复版”思路：
    - 支持 2D/3D 张量，自动按 batch 维聚合；
    - 先做 L2 归一化，消除特征尺度影响；
    - 先对特征维度求均值，再对 batch 做全局平均；
    - 做基础的维度检查，避免静默错误。
    """
    if v1.shape != v2.shape:
        raise ValueError(f"mean_square_error: shape mismatch {v1.shape} vs {v2.shape}")

    # 统一到 [batch, ..., dim] 形式，最后一维是特征维
    if v1.dim() == 1:
        # [dim] -> [1, dim]
        v1 = v1.unsqueeze(0)
        v2 = v2.unsqueeze(0)
    elif v1.dim() not in (2, 3):
        raise ValueError(f"mean_square_error: only supports 1D/2D/3D tensors, got dim={v1.dim()}")

    # L2 归一化到特征维（最后一维）
    v1_norm = F.normalize(v1, p=2, dim=-1)
    v2_norm = F.normalize(v2, p=2, dim=-1)

    # 逐元素平方差
    diff_sq = (v1_norm - v2_norm) ** 2  # shape 同 v1

    # 对特征维和可能的 seq 维做平均，只保留 batch 维
    if diff_sq.dim() == 3:
        # [batch, seq, dim] -> [batch]
        mse_per_sample = diff_sq.mean(dim=(1, 2))
    elif diff_sq.dim() == 2:
        # [batch, dim] -> [batch]
        mse_per_sample = diff_sq.mean(dim=1)
    else:  # dim == 1（已在上面转成 2D，这里其实进不来）
        mse_per_sample = diff_sq.mean()

    # 最后对 batch 取平均，得到一个标量
    return mse_per_sample.mean()

def info_nce_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    batch_size = z1.size(0)
    labels = torch.arange(batch_size).to(z1.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# 删除了cross_scale_info_nce_loss函数（跨尺度/全局对比损失）

# ===================== 模型定义（手动添加自环，避免权重冲突） =====================
class GCN_PYG(nn.Module):
    def __init__(self, n_feat, n_hid, n_class):
        super(GCN_PYG, self).__init__()
        # 关闭自动添加自环，手动处理
        self.gc1 = GCNConv(in_channels=n_feat, out_channels=n_hid, add_self_loops=False)
        self.gc2 = GCNConv(in_channels=n_hid, out_channels=n_class, add_self_loops=False)

    def forward(self, x, edge_index, edge_weight=None):
        # 手动添加自环（同步处理edge_weight，核心修复）
        num_nodes = x.size(0)
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight,
            fill_value=1.0,  # 自环权重设为1.0
            num_nodes=num_nodes
        )
        
        # 正常前向传播
        x = self.gc1(x, edge_index, edge_weight)
        x = torch.sigmoid(x)
        x = self.gc2(x, edge_index, edge_weight)
        x = torch.tanh(x)
        return x
    
    def encode(self, x, edge_index, edge_weight=None):
        return self.forward(x, edge_index, edge_weight)

class Decoder(nn.Module):
    def forward(self, x):
        y = x.permute(1, 0)
        z = torch.mm(x, y)
        z = torch.relu(z)
        return z

# ===================== 数据加载与预处理 =====================
_div_root = os.path.dirname(os.path.abspath(__file__))
_data_dir = get_data_dir()
A = load_txt_matrix(os.path.join(_data_dir, "divide_result", "A" + str(fold) + ".txt"))
X = load_txt_matrix(os.path.join(_data_dir, "divide_result", "X" + str(fold) + ".txt"))
in_features = X.shape[1]

if A.shape[0] != A.shape[1]:
    raise ValueError(f"A 必须是方阵，当前形状: {A.shape}")
if A.shape[0] != n + m:
    raise ValueError(f"A 维度与药物/靶点总数不匹配: A={A.shape}, n+m={n+m}")
if X.shape[0] != n + m:
    raise ValueError(f"X 行数与药物/靶点总数不匹配: X={X.shape}, n+m={n+m}")

# 图数据转换
edge_index_temp = sp.coo_matrix(A)
edge_weight = copy.deepcopy(edge_index_temp.data) if len(edge_index_temp.data) > 0 else None
edge_index_A = np.vstack((edge_index_temp.row, edge_index_temp.col))
edge_index_A = torch.LongTensor(edge_index_A).to(device)

# 邻接矩阵tensor化
A_tensor_adj = torch.from_numpy(A).float().view(n + m, n + m).to(device)
X_original = torch.from_numpy(X).float().view(n + m, in_features).to(device)
A_tensor = torch.from_numpy(A).float().view(n + m, n + m).to(device)

# 封装原始数据（适配旧版PyG）
data_original = Data(
    x=X_original,
    edge_index=edge_index_A,
    edge_weight=torch.FloatTensor(edge_weight).to(device) if edge_weight is not None else None
).to(device)

# ===================== 模型初始化 =====================
decoder = Decoder().to(device)
G = GCN_PYG(n_feat=in_features, n_hid=N_HID, n_class=out_features).to(device)
G_optimizer = torch.optim.Adam(G.parameters(), lr=LR)
loss_function_E = nn.MSELoss()

# 初始化图增强器：对称删边 + X 分块对称特征置零
transform1 = get_graph_drop_transform(
    drop_edge_p1, drop_feat_p1, n_drugs=n, n_targets=m, symmetric_feat_drop=True
)
transform2 = get_graph_drop_transform(
    drop_edge_p2, drop_feat_p2, n_drugs=n, n_targets=m, symmetric_feat_drop=True
)

# 消融开关：对比学习/子图对比学习
disable_contrast = bool(results.disable_contrast)
disable_subgraph_contrast = bool(results.disable_subgraph_contrast)
if disable_contrast:
    node_loss_weight = 0.0
    subgraph_loss_weight = 0.0
elif disable_subgraph_contrast:
    subgraph_loss_weight = 0.0

# ===================== 训练循环 =====================
for epoch in range(epochs):
    # 图增强（兼容旧版）
    if disable_contrast:
        data1 = data_original
        data2 = data_original
    else:
        data1 = transform1(data_original).to(device)
        data2 = transform2(data_original).to(device)
    
    # GCN编码
    Z1 = G.encode(data1.x, data1.edge_index, data1.edge_weight)
    Z2 = G.encode(data2.x, data2.edge_index, data2.edge_weight)
    
    # 重构损失：基于两路嵌入求均值后的矩阵
    Z_mean = (Z1 + Z2) / 2.0
    A_hat = decoder(Z_mean)
    recon_loss = loss_function_E(A_hat, A_tensor)
    
    # 序列特征对齐损失（SMILES/Protein 与 对比学习嵌入）
    if results.disable_seq:
        seq_align_loss = torch.tensor(0.0, device=device)
    else:
        drug_cl = Z_mean[:n, :]
        target_cl = Z_mean[n:, :]
        common_d_drug = min(drug_cl.size(1), drug_smiles_feat.size(1))
        common_d_prot = min(target_cl.size(1), protein_seq_feat.size(1))
        drug_align_loss = mean_square_error(
            drug_cl[:, :common_d_drug],
            drug_smiles_feat[:, :common_d_drug]
        )
        target_align_loss = mean_square_error(
            target_cl[:, :common_d_prot],
            protein_seq_feat[:, :common_d_prot]
        )
        seq_align_loss = drug_align_loss + target_align_loss
    
    # 多尺度对比损失（仅保留节点级+子图级）
    drug_Z1 = Z1[:n, :]
    drug_Z2 = Z2[:n, :]
    target_Z1 = Z1[n:, :]
    target_Z2 = Z2[n:, :]
    if disable_contrast:
        drug_node_loss = torch.tensor(0.0, device=device)
        target_node_loss = torch.tensor(0.0, device=device)
        node_contrast_loss = torch.tensor(0.0, device=device)
    else:
        drug_node_loss = info_nce_loss(drug_Z1, drug_Z2, temperature)
        target_node_loss = info_nce_loss(target_Z1, target_Z2, temperature)
        node_contrast_loss = drug_node_loss + target_node_loss

    if disable_contrast or disable_subgraph_contrast:
        drug_subgraph_loss = torch.tensor(0.0, device=device)
        target_subgraph_loss = torch.tensor(0.0, device=device)
        subgraph_contrast_loss = torch.tensor(0.0, device=device)
    else:
        # ✅ 修复：使用增强后的邻接矩阵进行子图特征提取（与GCN编码保持一致）
        A1_adj = edge_index_to_adj_matrix(data1.edge_index, data1.edge_weight, num_nodes=n + m)
        A2_adj = edge_index_to_adj_matrix(data2.edge_index, data2.edge_weight, num_nodes=n + m)
        subgraph_feat1 = get_subgraph_feature(Z1, A1_adj, k=k)
        subgraph_feat2 = get_subgraph_feature(Z2, A2_adj, k=k)
        drug_subgraph1 = subgraph_feat1[:n, :]
        drug_subgraph2 = subgraph_feat2[:n, :]
        target_subgraph1 = subgraph_feat1[n:, :]
        target_subgraph2 = subgraph_feat2[n:, :]
        drug_subgraph_loss = info_nce_loss(drug_subgraph1, drug_subgraph2, temperature)
        target_subgraph_loss = info_nce_loss(target_subgraph1, target_subgraph2, temperature)
        subgraph_contrast_loss = drug_subgraph_loss + target_subgraph_loss

    # 删除了全局级对比损失的所有计算逻辑

    # 总对比损失（删除了global_loss_weight相关项）
    total_contrast_loss = (
        node_loss_weight * node_contrast_loss +
        subgraph_loss_weight * subgraph_contrast_loss
    )

    # 总损失与反向传播（加入序列对齐约束）
    total_loss = recon_loss + total_contrast_loss + seq_align_loss
    G_optimizer.zero_grad()
    total_loss.backward()
    G_optimizer.step()

    if epoch % log_interval == 0 or epoch == epochs - 1:
        logger.info(
            "Epoch %d/%d | recon=%.6f | drug_node=%.6f | target_node=%.6f | "
            "drug_subgraph=%.6f | target_subgraph=%.6f | seq_align=%.6f | total=%.6f",
            epoch + 1,
            epochs,
            recon_loss.item(),
            drug_node_loss.item(),
            target_node_loss.item(),
            drug_subgraph_loss.item(),
            target_subgraph_loss.item(),
            seq_align_loss.item(),
            total_loss.item(),
        )

# ===================== 提取并保存 (SMILES/序列 + 对比嵌入) 特征（分类器输入） =====================
# 最终做法：
# - 使用对比学习最后一次迭代得到的两路嵌入 Z1、Z2 的“均值”作为结构特征 Z_mean_final
# - 将 Z_mean_final 拆成药物 / 靶点两部分，与对应的 SMILES / 序列特征拼接
# - 拼接结果作为最终嵌入，既包含结构信息，又包含序列/SMILES 语义信息
G.eval()
with torch.no_grad():
    # 再做一次图增强，获取对比学习视角下的两路嵌入
    if disable_contrast:
        data1_final = data_original
        data2_final = data_original
    else:
        data1_final = transform1(data_original).to(device)
        data2_final = transform2(data_original).to(device)
    Z1_final = G.encode(data1_final.x, data1_final.edge_index, data1_final.edge_weight)
    Z2_final = G.encode(data2_final.x, data2_final.edge_index, data2_final.edge_weight)

    Z_mean_final = (Z1_final + Z2_final) / 2.0

    if results.disable_seq:
        # 只输出结构嵌入（对比/重构得到的 Z_mean_final）
        full_embedding = Z_mean_final
        A_hat_final = decoder(full_embedding)
        drug_final = full_embedding[:n, :]
        target_final = full_embedding[n:, :]
    else:
        drug_cl_final = Z_mean_final[:n, :]
        target_cl_final = Z_mean_final[n:, :]
        # 拼接 SMILES / 序列特征 与 结构对比嵌入
        drug_final = torch.cat([drug_smiles_feat, drug_cl_final], dim=1)
        target_final = torch.cat([protein_seq_feat, target_cl_final], dim=1)
        full_embedding = torch.cat([drug_final, target_final], dim=0)
        A_hat_final = decoder(full_embedding)

# 保存结果（使用拼接后的最终嵌入及其重构矩阵）
result = A_hat_final.data.cpu().numpy()
embedding = full_embedding.data.cpu().numpy()
drug_embedding = drug_final.data.cpu().numpy()
target_embedding = target_final.data.cpu().numpy()

save_dir = os.path.join(_data_dir, results.save_subdir)
os.makedirs(save_dir, exist_ok=True)
np.savetxt(os.path.join(save_dir, f'score_cl_seq{fold}.txt'), result)
np.savetxt(os.path.join(save_dir, f'embedding_cl_seq{fold}.txt'), embedding)
np.savetxt(os.path.join(save_dir, f'drug_embedding_cl_seq{fold}.txt'), drug_embedding)
np.savetxt(os.path.join(save_dir, f'target_embedding_cl_seq{fold}.txt'), target_embedding)

logger.info(
    "训练完成！fold=%d 的 (SMILES/序列 + 对比嵌入) 特征已保存为 *_cl_seq%d.txt，可用于后续实验",
    fold,
    fold,
)