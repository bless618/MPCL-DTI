import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN

from einops import reduce
from collections import deque

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def mean_square_error(v_d, v_s):
    loss_fct = torch.nn.MSELoss()
    loss = loss_fct(v_d, v_s)
    return loss
    
def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent

# 新增：计算Jaccard相似性矩阵
def compute_jaccard_similarity(adj_matrix):
    
    num_nodes = adj_matrix.size(0)
    jaccard_matrix = torch.eye(num_nodes, device=adj_matrix.device)  # 对角线为1（自身相似）
    
    # 计算每个节点的邻居集合
    neighbors = [set(torch.nonzero(adj_matrix[i], as_tuple=True)[0].tolist()) 
                for i in range(num_nodes)]
    
    # 两两计算Jaccard相似性
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            set_i = neighbors[i]
            set_j = neighbors[j]
            
            intersection = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))
            
            if union == 0:
                jaccard_sim = 0.0
            else:
                jaccard_sim = intersection / union
            
            jaccard_matrix[i, j] = jaccard_sim
            jaccard_matrix[j, i] = jaccard_sim
    
    return jaccard_matrix

# MGDC类保持不变（Jaccard相似性矩阵输入）
class MGDC(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_scales, fusion='attention'):
        super(MGDC, self).__init__()
        self.num_scales = num_scales
        self.fusion = fusion
        self.lin = nn.Linear(in_channels, hidden_channels)
        
        if fusion == 'attention':
            self.attn = nn.Linear(hidden_channels, 1)
        elif fusion == 'weighted':
            self.weights = nn.Parameter(torch.ones(num_scales))
        elif fusion == 'concat':
            self.out_lin = nn.Linear(num_scales * hidden_channels, hidden_channels)
    
    def forward(self, x, diff_matrices):
        x = F.relu(self.lin(x))
        scale_features = []
        
        for i in range(self.num_scales):
            if diff_matrices[i].is_sparse:
                diff_feat = torch.sparse.mm(diff_matrices[i], x)
            else:
                diff_feat = torch.mm(diff_matrices[i], x)
            scale_features.append(diff_feat)
        
        if self.fusion == 'attention':
            stacked = torch.stack(scale_features, dim=1)
            attn_scores = self.attn(stacked).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=1)
            fused_feat = torch.sum(attn_weights.unsqueeze(-1) * stacked, dim=1)
        
        elif self.fusion == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            fused_feat = sum(w * feat for w, feat in zip(weights, scale_features))
        
        elif self.fusion == 'concat':
            concat_feat = torch.cat(scale_features, dim=1)
            fused_feat = self.out_lin(concat_feat)
        
        return fused_feat

# 新增：自适应交互图模块
class AdaptiveInteractionGraph(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def build_adaptive_adjacency(self, edge_weights, batch_size):
        """构建自适应邻接矩阵"""
        weights_tensor = torch.stack(edge_weights).view(batch_size, batch_size)
        
        # 创建二分图邻接矩阵
        adj_matrix = torch.zeros(2 * batch_size, 2 * batch_size, 
                               device=weights_tensor.device)
        
        # 药物节点索引: 0 到 batch_size-1
        # 蛋白质节点索引: batch_size 到 2*batch_size-1
        for i in range(batch_size):
            for j in range(batch_size):
                weight = weights_tensor[i, j]
                # 药物i连接到蛋白质j
                adj_matrix[i, batch_size + j] = weight
                adj_matrix[batch_size + j, i] = weight  # 无向图
        
        return adj_matrix
    
    def forward(self, drug_feat, protein_feat):
        """动态预测边权重，而非依赖标签"""
        batch_size = drug_feat.size(0)
        edge_weights = []
        
        # 计算所有药物-蛋白质对的边权重
        for i in range(batch_size):
            for j in range(batch_size):
                pair_feat = torch.cat([drug_feat[i], protein_feat[j]], dim=-1)
                weight = self.edge_predictor(pair_feat)
                edge_weights.append(weight)
        
        return self.build_adaptive_adjacency(edge_weights, batch_size)

class InteractionNetworkExtractor(nn.Module):
    def __init__(self, drug_feat_dim, protein_feat_dim, hidden_dim=256, num_scales=3, fusion='attention', use_adaptive_graph=True):
        super(InteractionNetworkExtractor, self).__init__()
        self.drug_feat_dim = drug_feat_dim
        self.protein_feat_dim = protein_feat_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.use_adaptive_graph = use_adaptive_graph
        
        # 投影层，确保药物和蛋白质特征维度一致
        self.drug_proj = nn.Linear(drug_feat_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_feat_dim, hidden_dim)
        
        # 自适应交互图模块
        if use_adaptive_graph:
            self.adaptive_graph = AdaptiveInteractionGraph(hidden_dim)
        
        # MGDC模型
        self.mgdc = MGDC(hidden_dim, hidden_dim, num_scales, fusion)
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # 药物和蛋白质特征拼接
        
    def build_interaction_graph(self, drug_features, protein_features, interactions):
        
        batch_size = drug_features.size(0)
        
        # 投影到统一维度
        drug_proj = self.drug_proj(drug_features)  # (batch_size, hidden_dim)
        protein_proj = self.protein_proj(protein_features)  # (batch_size, hidden_dim)
        
        # 构建节点特征矩阵
        node_features = torch.cat([drug_proj, protein_proj], dim=0)  # (2*batch_size, hidden_dim)
        
        if self.use_adaptive_graph:
            # 使用自适应图结构
            adj_matrix = self.adaptive_graph(drug_proj, protein_proj)
            # 将稠密邻接矩阵转换为边索引格式
            edge_indices = adj_matrix.nonzero(as_tuple=False).t()
            edge_index = edge_indices
        else:
            # 使用基于标签的静态图结构
            edge_indices = []
            for i in range(batch_size):
                if interactions[i] > 0.5:  # 存在相互作用
                    # 药物节点i连接到蛋白质节点i
                    edge_indices.append([i, batch_size + i])
                    edge_indices.append([batch_size + i, i])  # 无向图
            
            if len(edge_indices) == 0:
                # 如果没有相互作用，创建自环
                edge_indices = [[i, i] for i in range(2 * batch_size)]
            
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().to(drug_features.device)
        
        return node_features, edge_index
    
    def compute_diffusion_matrices(self, edge_index, num_nodes, alphas=[0.1, 0.3, 0.5]):
        """基于Jaccard相似性矩阵计算多尺度扩散矩阵"""
        # 添加自环
        edge_index_with_self_loops = torch.cat([
            edge_index,
            torch.arange(num_nodes, device=edge_index.device).repeat(2, 1)
        ], dim=1)
        
        # 计算度矩阵
        row, col = edge_index_with_self_loops
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg = deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # 归一化邻接矩阵
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(edge_index_with_self_loops, norm, (num_nodes, num_nodes))
        adj_dense = adj.to_dense()
        
        # 计算Jaccard相似性矩阵
        jaccard_matrix = compute_jaccard_similarity(adj_dense)
        
        # 用Jaccard矩阵替换原始邻接矩阵进行扩散
        diffusion_matrices = []
        for alpha in alphas:
            identity = torch.eye(num_nodes, device=edge_index.device)
            diff_matrix = alpha * identity
            # 迭代扩散：使用Jaccard矩阵而非原始邻接矩阵
            for _ in range(10):
                diff_matrix = alpha * identity + (1 - alpha) * torch.mm(jaccard_matrix, diff_matrix)
            diffusion_matrices.append(diff_matrix)
        
        return diffusion_matrices
    
    def forward(self, drug_features, protein_features, interactions):
        """
        drug_features: (batch_size, drug_feat_dim)
        protein_features: (batch_size, protein_feat_dim)
        interactions: (batch_size,) 相互作用标签
        """
        batch_size = drug_features.size(0)
        
        # 构建相互作用图
        node_features, edge_index = self.build_interaction_graph(drug_features, protein_features, interactions)
        num_nodes = node_features.size(0)
        
        # 计算基于Jaccard相似性的扩散矩阵
        diffusion_matrices = self.compute_diffusion_matrices(edge_index, num_nodes)
        
        # 通过MGDC提取特征（输入为Jaccard衍生的扩散矩阵）
        node_embeddings = self.mgdc(node_features, diffusion_matrices)  # (2*batch_size, hidden_dim)
        
        # 分离药物和蛋白质嵌入
        drug_embeddings = node_embeddings[:batch_size]  # (batch_size, hidden_dim)
        protein_embeddings = node_embeddings[batch_size:]  # (batch_size, hidden_dim)
        
        # 计算相互作用特征
        interaction_features = torch.cat([drug_embeddings, protein_embeddings], dim=1)  # (batch_size, hidden_dim*2)
        interaction_features = self.output_proj(interaction_features)  # (batch_size, hidden_dim)
        
        return interaction_features

class MGNDTI(nn.Module):
    def __init__(self, **config):
        super(MGNDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        drug_layers = config["DRUG"]["LAYERS"]
        drug_num_head = config["DRUG"]["NUM_HEAD"]
        drug_padding = config["DRUG"]["PADDING"]

        protein_layers = config["PROTEIN"]["LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        protein_padding = config["PROTEIN"]["PADDING"]

        mgn_emb_dim = config["MGN"]["EMBEDDING_DIM"]

        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]

        # 获取交互网络配置
        interaction_hidden_dim = config.get("INTERACTION", {}).get("HIDDEN_DIM", 256)
        use_adaptive_graph = config.get("INTERACTION", {}).get("USE_ADAPTIVE_GRAPH", True)

        # 仅保留GCN药物特征提取器
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        
        self.protein_extractor = ProteinCNNTransformer(embedding_dim=protein_emb_dim,
                                               num_head=protein_num_head, layers=protein_layers, padding=protein_padding)
        
        # 添加相互作用网络特征提取器（支持自适应图）
        self.interaction_extractor = InteractionNetworkExtractor(
            drug_feat_dim=drug_hidden_feats[-1] if isinstance(drug_hidden_feats, list) else drug_hidden_feats,
            protein_feat_dim=protein_emb_dim,
            hidden_dim=interaction_hidden_dim,
            use_adaptive_graph=use_adaptive_graph
        )
        
        # Multimodal Gating Network - 更新输入维度以包含相互作用特征
        self.multi_gating_network = MultimodalGatingNetworkWithInteraction(
            mgn_emb_dim, 
            interaction_dim=interaction_hidden_dim
        )
        # MLPDecoder
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, smi_d, bg_d, v_p, interactions=None, mode="train"):
        # Drug Encoder（仅保留GCN输出）
        v_d = self.drug_extractor(bg_d)
        
        # Protein Encoder
        v_p = self.protein_extractor(v_p)
        
        # 提取相互作用网络特征
        v_d_global = reduce(v_d, "b h w -> b w", 'max')  # (batch_size, drug_feat_dim)
        v_p_global = reduce(v_p, "b h w -> b w", 'max')  # (batch_size, protein_emb_dim)
        
        if interactions is None:
            interactions = torch.zeros(v_d_global.size(0), device=v_d_global.device)
        
        v_i = self.interaction_extractor(v_d_global, v_p_global, interactions)
        
        # Multimodal Gating Network - 
        f, v_d, v_p, v_i = self.multi_gating_network(v_d, v_p, v_i)
        
        # Decoder
        score = self.mlp_classifier(f)
        
        if mode == "train":
            return v_d, v_p, v_i, f, score
        elif mode == "eval":
            return v_d, v_p, v_i, score, None

class MultimodalGatingNetworkWithInteraction(nn.Module):
    def __init__(self, dim, interaction_dim=256):
        super(MultimodalGatingNetworkWithInteraction, self).__init__()
        self.gated_g = GLU(dim, dim)
        self.gated_p = GLU(dim, dim)
        
        self.gated_i = GLU(interaction_dim, dim)
        self.tanh = nn.Tanh()
        
        self.interaction_proj = nn.Linear(dim, dim) if dim != interaction_dim else nn.Identity()

    def forward(self, mg, mp, mi):
        mg = self.gated_g(mg)
        mp = self.gated_p(mp)
        mi = self.gated_i(mi)
        
        v_d = reduce(mg, "b h w -> b w", 'max')
        v_p = reduce(mp, "b h w -> b w", 'max')
        v_i = self.interaction_proj(mi)
        
        v_dp = v_d * v_p
        f = self.tanh(torch.cat([v_dp, v_i], dim=-1))
        
        return f, v_d, v_p, v_i


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

class ProteinRetNet(nn.Module):
    def __init__(self, embedding_dim, num_head, layers, padding=True):
        super(ProteinRetNet, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)

        self.retnet = RetNet(layers=layers, hidden_dim=embedding_dim,
                             ffn_size=embedding_dim // 2, heads=num_head, double_v_dim=False)

    def forward(self, v):
        v = self.embedding(v.long())
        v = self.retnet(F.relu(v))
        return v

class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GLU, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        Y = self.W(X) * self.sigmoid(self.V(X))
        return Y

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class ProteinCNNTransformer(nn.Module):
    def __init__(self, embedding_dim, num_head, layers, padding=True):
        super(ProteinCNNTransformer, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)

        self.conv1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_head, dim_feedforward=embedding_dim // 2, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.permute(0, 2, 1)
        v = F.relu(self.conv1(v))
        v = F.relu(self.conv2(v))
        v = v.permute(0, 2, 1)
        v = self.transformer_encoder(v)
        return v


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TargetFeatureExtractor(nn.Module):
    """
    论文 Target feature extraction：
    - 嵌入 P_e (d 维) + 三路多尺度 2×Conv1d（块级残差）→ 各自 GAP 得 V_c1,V_c2,V_c3
    - P_e + PE → Transformer → GAP 得 V_t
    - 拼接 4d 维后经 1D Conv 映射为 F_target ∈ R^{2d}
    """

    @staticmethod
    def _make_two_layer_cnn(d, kernel_size, padding):
        return nn.Sequential(
            nn.Conv1d(d, d, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(d, d, kernel_size=kernel_size, padding=padding),
        )

    def __init__(self, d=128, out_dim=256, num_head=4, layers=2, kernel_sizes=(3, 5, 7), padding=True):
        super().__init__()
        if out_dim != 2 * d:
            raise ValueError(f"out_dim must equal 2*d (got out_dim={out_dim}, d={d})")
        self.d = d
        if padding:
            self.embedding = nn.Embedding(26, d, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, d)

        self.cnn_branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2
            self.cnn_branches.append(self._make_two_layer_cnn(d, k, pad))

        self.pos_encoder = PositionalEncoding(d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=num_head,
            dim_feedforward=d * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fusion_conv = nn.Conv1d(4 * d, out_dim, kernel_size=1)

    def _multi_scale_branch(self, pe, cnn):
        x = pe.transpose(1, 2)
        out = cnn(x) + x
        return out.mean(dim=2)

    def forward(self, v):
        pe = self.embedding(v.long())
        branch_vecs = [self._multi_scale_branch(pe, cnn) for cnn in self.cnn_branches]

        pad_mask = v.eq(0)
        z = self.pos_encoder(pe)
        pt = self.transformer_encoder(z, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1).float()
        vt = (pt * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        x_t = torch.cat(branch_vecs + [vt], dim=-1).unsqueeze(-1)
        return self.fusion_conv(x_t).squeeze(-1)


class ProteinSequenceEncoder(nn.Module):
    """
    GCN 在线序列对齐用蛋白质编码器。
    三路并行 2×Conv1d(k=3/5/7) + Transformer → 256 维；支持可学习 pos_enc。
    """

    _make_two_layer_cnn = staticmethod(TargetFeatureExtractor._make_two_layer_cnn)

    def __init__(
        self,
        internal_dim=128,
        output_dim=256,
        num_heads=4,
        transformer_layers=2,
        padding=True,
    ):
        super().__init__()
        if output_dim != 2 * internal_dim:
            raise ValueError(
                f"output_dim 应为 2*internal_dim，当前 output_dim={output_dim}, internal_dim={internal_dim}"
            )
        self.internal_dim = internal_dim
        self.output_dim = output_dim

        if padding:
            self.embedding = nn.Embedding(26, internal_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, internal_dim)

        self.cnn3 = TargetFeatureExtractor._make_two_layer_cnn(internal_dim, 3, 1)
        self.cnn5 = TargetFeatureExtractor._make_two_layer_cnn(internal_dim, 5, 2)
        self.cnn7 = TargetFeatureExtractor._make_two_layer_cnn(internal_dim, 7, 3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=internal_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.conv_reduce = nn.Conv1d(4 * internal_dim, output_dim, kernel_size=1)

    def _cnn_branch(self, cnn, x):
        x_perm = x.permute(0, 2, 1)
        out = cnn(x_perm) + x_perm
        return out.mean(dim=1)

    def forward(self, v, pos_enc=None):
        x = self.embedding(v.long())
        v_c1 = self._cnn_branch(self.cnn3, x)
        v_c2 = self._cnn_branch(self.cnn5, x)
        v_c3 = self._cnn_branch(self.cnn7, x)

        pad_mask = v.eq(0)
        z = x + pos_enc if pos_enc is not None else x
        pt = self.transformer_encoder(z, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1).float()
        v_t = (pt * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        x_t = torch.cat([v_c1, v_c2, v_c3, v_t], dim=-1).unsqueeze(-1)
        return self.conv_reduce(x_t).squeeze(-1)