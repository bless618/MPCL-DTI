import copy

import torch
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.transforms import Compose


def _get_edge_attr(data):
    """读取边权重（兼容 edge_weight / edge_attr）。"""
    if getattr(data, "edge_weight", None) is not None:
        return data.edge_weight
    if getattr(data, "edge_attr", None) is not None:
        return data.edge_attr
    return None


def _set_edge_attr(data, edge_attr):
    """写回边权重（优先写 edge_weight，与 GCN 脚本一致）。"""
    if hasattr(data, "edge_weight"):
        data.edge_weight = edge_attr
    else:
        data.edge_attr = edge_attr


class DropFeatures:
    r"""按列随机置零节点特征（非对称，旧版默认行为）。"""

    def __init__(self, p=None, precomputed_weights=True):
        assert 0.0 < p < 1.0, "Dropout probability has to be between 0 and 1, but got %.2f" % p
        self.p = p

    def __call__(self, data):
        drop_mask = (
            torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1)
            < self.p
        )
        data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return "{}(p={})".format(self.__class__.__name__, self.p)


class DropFeaturesSymmetric:
    r"""按 X 分块结构对称置零特征。

    X 分块（n 药物、m 靶点）::
        | SD (n×n)  | DTI (n×m)  |  药物行
        | DTI^T(m×n)| ST (m×m)   |  靶点行

    - SD / ST：对称位置 (i,j) 与 (j,i) 同时置 0
    - DTI / DTI^T：药物行 (d, n+t) 与靶点行 (n+t, d) 同时置 0
    """

    def __init__(self, p, n_drugs, n_targets):
        assert 0.0 < p < 1.0, "Dropout probability has to be between 0 and 1, but got %.2f" % p
        self.p = p
        self.n_drugs = int(n_drugs)
        self.n_targets = int(n_targets)

    def __call__(self, data):
        x = data.x
        n, m = self.n_drugs, self.n_targets
        num_nodes = n + m
        if x.size(0) != num_nodes or x.size(1) != num_nodes:
            raise ValueError(
                f"DropFeaturesSymmetric: 期望 x 为 ({num_nodes}, {num_nodes})，当前 {tuple(x.size())}"
            )

        sd_mask = torch.rand(n, n, device=x.device) < self.p
        sd_mask = sd_mask | sd_mask.t()
        x[:n, :n][sd_mask] = 0

        dti_mask = torch.rand(n, m, device=x.device) < self.p
        x[:n, n:n + m][dti_mask] = 0
        x[n:n + m, :n][dti_mask.t()] = 0

        st_mask = torch.rand(m, m, device=x.device) < self.p
        st_mask = st_mask | st_mask.t()
        x[n:n + m, n:n + m][st_mask] = 0

        return data

    def __repr__(self):
        return "{}(p={}, n_drugs={}, n_targets={})".format(
            self.__class__.__name__, self.p, self.n_drugs, self.n_targets
        )

class DropFeaturesSymmetricColumn:
    r"""
    对称整行列丢弃：随机选中若干列，该列+对应行全部置0，保证总特征矩阵 X=X^T
    X分块结构 (m药物, n靶点):
        | S^D (m×m)  | Y_DTI (m×n)  |
        | Y_DTI^T(n×m)| S^T (n×n)    |
    参数:
        p: 单个列被选中丢弃的概率
        n_drugs: 药物数量 m
        n_targets: 靶点数量 n
    """
    def __init__(self, p, n_drugs, n_targets):
        assert 0.0 < p < 1.0, "Dropout probability has to be between 0 and 1, but got %.2f" % p
        self.p = p
        self.n_drugs = int(n_drugs)
        self.n_targets = int(n_targets)

    def __call__(self, data):
        x = data.x
        m, n = self.n_drugs, self.n_targets
        total_nodes = m + n

        # 形状校验
        if x.size(0) != total_nodes or x.size(1) != total_nodes:
            raise ValueError(
                f"DropFeaturesSymmetricColumn: 期望 x 为 ({total_nodes}, {total_nodes})，当前 {tuple(x.size())}"
            )
        
        # 1. 随机生成哪些列需要被丢弃
        col_drop_mask = torch.rand(total_nodes, device=x.device) < self.p
        drop_col_indices = torch.nonzero(col_drop_mask).squeeze(dim=-1)

        # 2. 丢弃选中列 + 同步丢弃对应行，保证矩阵对称
        if len(drop_col_indices) > 0:
            x[:, drop_col_indices] = 0.0   # 整列清零
            x[drop_col_indices, :] = 0.0   # 对应整行清零

        return data

    def __repr__(self):
        return "{}(p={}, n_drugs={}, n_targets={})".format(
            self.__class__.__name__, self.p, self.n_drugs, self.n_targets
        )
        
class DropEdges:
    r"""Drops edges with probability p.

    force_undirected=True 时对称删边：(i,j) 与 (j,i) 同时保留或同时删除。
    """

    def __init__(self, p, force_undirected=True):
        assert 0.0 < p < 1.0, "Dropout probability has to be between 0 and 1, but got %.2f" % p
        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = _get_edge_attr(data)

        edge_index, edge_attr = dropout_adj(
            edge_index,
            edge_attr,
            p=self.p,
            force_undirected=self.force_undirected,
            training=True,
        )

        data.edge_index = edge_index
        _set_edge_attr(data, edge_attr)
        return data

    def __repr__(self):
        return "{}(p={}, force_undirected={})".format(
            self.__class__.__name__, self.p, self.force_undirected
        )


def get_graph_drop_transform(
    drop_edge_p,
    drop_feat_p,
    symmetric_edge_drop=True,
    n_drugs=None,
    n_targets=None,
    symmetric_feat_drop=False,
    use_column_wise_feat_drop=True,  # 新增开关
):
    transforms = []
    transforms.append(copy.deepcopy)

    if drop_edge_p > 0.0:
        transforms.append(DropEdges(drop_edge_p, force_undirected=symmetric_edge_drop))

    if drop_feat_p > 0.0:
        if symmetric_feat_drop and n_drugs is not None and n_targets is not None:
            if use_column_wise_feat_drop:
                # 使用对称整列丢弃
                transforms.append(DropFeaturesSymmetricColumn(drop_feat_p, n_drugs, n_targets))
            else:
                # 元素级对称丢弃
                transforms.append(DropFeaturesSymmetric(drop_feat_p, n_drugs, n_targets))
        else:
            transforms.append(DropFeatures(drop_feat_p))

    return Compose(transforms)