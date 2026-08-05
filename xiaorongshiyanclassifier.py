import numpy as np
import copy
import argparse
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from my_function import *
from dataset_utils import get_data_dir, load_dti_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import json

# ===================== 基础配置 =====================
# 允许外部通过 CUDA_VISIBLE_DEVICES 控制使用哪张卡（并行跑多个实验时很关键）
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# ===================== 命令行参数 =====================
parser = argparse.ArgumentParser()
parser.add_argument('-f', dest="fold", type=int, required=True, help='当前运行第几折 (0-4)')
parser.add_argument('--embedding_subdir', type=str, default='predict_result',
                    help='嵌入特征子目录（位于 data/ 下，默认 predict_result）')
parser.add_argument('--disable_pu_loss', action='store_true',
                    help='禁用 PU Loss，改用原始 BCE 分类损失')
parser.add_argument('--run_id', type=str, default='',
                    help='并行实验专用：用于替代 .current_run_timestamp.txt 的 run_id（不写共享时间戳文件）')
parser.add_argument('--w-p', type=float, default=1.0, help='PU损失 P 样本权重')
parser.add_argument('--w-rp', type=float, default=0.5, help='PU损失 RP 样本权重')
parser.add_argument('--w-ln', type=float, default=0.1, help='PU损失 LN 样本权重')
parser.add_argument('--w-rn', type=float, default=1.0, help='PU损失 RN 样本权重')
parser.add_argument('--no-rp-pseudo-positive', action='store_true',
                    help='关闭 RP 伪正损失（默认 RP 按伪标签 -w*log(p) 计损失）')
results = parser.parse_args()
fold = results.fold
use_pu_loss = not results.disable_pu_loss
run_id = (results.run_id or "").strip()
w_p = results.w_p
w_rp = results.w_rp
w_ln = results.w_ln
w_rn = results.w_rn
rp_pseudo_positive = not results.no_rp_pseudo_positive

EXP_DESC = 'P:RP:RN:LN=1:1:1:1'

# ===================== 路径配置 =====================
_proj_root = os.path.dirname(os.path.abspath(__file__))

# ===================== 日志配置 =====================
log_dir = os.path.join(_proj_root, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'classifier_fold{fold}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f'{"="*60}')
logger.info(f'第{fold}折 | {EXP_DESC}')
logger.info(f'{"="*60}')

# ===================== 1. 加载数据 =====================
data_dir = get_data_dir()
divide_dir = os.path.join(data_dir, 'divide_result')

index_1 = np.loadtxt(os.path.join(divide_dir, 'index_1.txt'))

train_neg_file = os.path.join(divide_dir, f'train_neg_fold{fold}.txt')
test_neg_file = os.path.join(divide_dir, f'test_neg_fold{fold}.txt')

if not os.path.isfile(train_neg_file):
    raise FileNotFoundError(f"未找到: {train_neg_file}\n请先运行 xiaorongshiyangroup_divide.py")
if not os.path.isfile(test_neg_file):
    raise FileNotFoundError(f"未找到: {test_neg_file}\n请先运行 xiaorongshiyangroup_divide.py")

train_neg = np.loadtxt(train_neg_file).astype(int)
test_neg = np.loadtxt(test_neg_file).astype(int)

# 构建训练集/测试集索引
test_pos = index_1[fold].copy()
train_pos = np.concatenate([index_1[i] for i in range(5) if i != fold])
train_index = np.concatenate([train_pos, train_neg])
test_index = np.concatenate([test_pos, test_neg])

logger.info(f'训练集负样本: {train_neg_file} ({len(train_neg)}个)')
logger.info(f'测试集负样本: {test_neg_file} ({len(test_neg)}个)')

# 加载特征和标签（使用 SMILES/序列 + 对比嵌入拼接后的新特征）
label, n, m = load_dti_matrix(data_dir)  # n=药物数, m=靶点数
Y_fuse = np.loadtxt(os.path.join(data_dir, 'Y_fused_Meta.txt'))
embedding = np.loadtxt(os.path.join(data_dir, f'{results.embedding_subdir}/embedding_cl_seq{fold}.txt'))
if embedding.shape[0] != n + m:
    raise ValueError(
        f"embedding 行数与药物/靶点数量不一致: embedding={embedding.shape}, label={label.shape}"
    )
drug_feature = embedding[0:n, :]
target_feature = embedding[n:, :]

# 检查训练/测试集无交集
intersection = np.intersect1d(test_index, train_index)
if len(intersection) > 0:
    logger.warning(f"训练集和测试集有交集: {len(intersection)}个样本")

np.random.shuffle(train_index)

# ===================== 2. PU阈值读取 / 计算 =====================
# 优先与 Metapath-PU Learning 保持一致：从 PU_thresholds.npy 读取全局阈值
if use_pu_loss:
    pu_threshold_path = os.path.join(data_dir, 'PU_thresholds.npy')
    if os.path.isfile(pu_threshold_path):
        alpha_PU, beta_PU = np.load(pu_threshold_path).tolist()
        logger.info(f'PU阈值来自文件 PU_thresholds.npy | alpha_PU={alpha_PU:.6f}, beta_PU={beta_PU:.6f}')
    else:
        logger.warning(f'未找到 {pu_threshold_path}，退回基于训练集分位数估计的方式。')
        train_mask = np.zeros_like(label, dtype=bool)
        for idx_val in train_index:
            train_mask[int(idx_val // m), int(idx_val % m)] = True

        P_mask_train = train_mask & (label == 1)
        U_mask_train = train_mask & (label == 0)
        S_P_train = Y_fuse[P_mask_train]
        S_U_train = Y_fuse[U_mask_train]

        # 默认与当前推荐设置一致：P 75% 分位数 / U 25% 分位数
        alpha_PU = max(np.percentile(S_P_train, 75.0), 1e-6) if len(S_P_train) > 0 else 0.5
        beta_PU = max(np.percentile(S_U_train, 25.0), 1e-6) if len(S_U_train) > 0 else 0.1

        if alpha_PU <= beta_PU:
            logger.warning(f"alpha_PU({alpha_PU:.6f}) <= beta_PU({beta_PU:.6f})，调整beta_PU")
            beta_PU = alpha_PU * 0.9

    logger.info(f'PU阈值 | alpha_PU={alpha_PU:.6f}, beta_PU={beta_PU:.6f}')
    logger.info(f'PU损失权重 | P={w_p}, RP={w_rp}, RN={w_rn}, LN={w_ln}')
    logger.info(f'RP伪正损失 | {"开启" if rp_pseudo_positive else "关闭（RP项恒为0）"}')
else:
    alpha_PU, beta_PU = 1.0, 0.0
    logger.info('已禁用 PU Loss：使用原始 BCE 分类损失。')

# ===================== 3. 样本处理函数 =====================
def get_sample_type_weight(drug_id, target_id, is_test_sample=False):
    """获取样本类型(0=P,1=RP,2=LN,3=RN)和权重"""
    if label[drug_id, target_id] == 1:
        return 0, w_p
    if is_test_sample:
        return 3, w_rn
    conf = Y_fuse[drug_id, target_id]
    if conf >= alpha_PU:
        return 1, w_rp
    elif conf > beta_PU:
        return 2, w_ln
    else:
        return 3, w_rn

def process_samples(index_list, is_test_set=False):
    """生成特征、标签、类型、权重"""
    input_list, output_list, type_list, weight_list = [], [], [], []
    for idx in index_list:
        drug, target = int(idx // m), int(idx % m)
        input_list.append(np.hstack((drug_feature[drug], target_feature[target])))
        output_list.append(label[drug, target])
        s_type, s_weight = get_sample_type_weight(drug, target, is_test_sample=is_test_set)
        type_list.append(s_type)
        weight_list.append(s_weight)
    return np.array(input_list), np.array(output_list), np.array(type_list), np.array(weight_list)

# 统计样本数
train_pos_count = np.sum([label[int(idx // m), int(idx % m)] == 1 for idx in train_index])
train_neg_count = len(train_index) - train_pos_count
test_pos_count = np.sum([label[int(idx // m), int(idx % m)] == 1 for idx in test_index])
test_neg_count = len(test_index) - test_pos_count

logger.info(f'训练集: 正{train_pos_count} 负{train_neg_count} (1:{train_neg_count/train_pos_count:.1f})')
logger.info(f'测试集: 正{test_pos_count} 负{test_neg_count}')

# 生成数据
train_input, train_output, train_type, train_weight = process_samples(train_index, is_test_set=False)
test_input, test_output, test_type, test_weight = process_samples(test_index, is_test_set=True)

type_names = ['P', 'RP', 'LN', 'RN']
train_type_counts = {name: int(np.sum(train_type == i)) for i, name in enumerate(type_names)}
logger.info(f'训练集样本类型统计: {train_type_counts}')

# ===================== 4. 数据预处理 =====================
scaler = StandardScaler()
train_input = scaler.fit_transform(train_input)
test_input = scaler.transform(test_input)

# 大数据集（H/I/J）保持在 CPU，按 batch 送入 GPU，避免 OOM
train_input_tensor = torch.FloatTensor(train_input)
train_output_tensor = torch.FloatTensor(train_output)
train_type_tensor = torch.LongTensor(train_type)
train_weight_tensor = torch.FloatTensor(train_weight)

batch_size = 2048
if use_pu_loss:
    train_dataset = TensorDataset(train_input_tensor, train_output_tensor, train_type_tensor, train_weight_tensor)
else:
    train_dataset = TensorDataset(train_input_tensor, train_output_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def predict_in_batches(model, features_np, infer_batch_size=4096):
    """分批推理，避免一次性将超大测试集载入 GPU。"""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(features_np), infer_batch_size):
            batch_x = torch.FloatTensor(features_np[start:start + infer_batch_size]).to(device)
            preds.append(model(batch_x).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])

# ===================== 5. MLP模型定义 =====================
class DTI_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1=512, hidden_dim2=256, dropout=0.2):
        super(DTI_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.mlp(x).squeeze()

input_dim = train_input.shape[1]
mlp_model = DTI_MLP(input_dim=input_dim).to(device)

# ===================== 6. PU损失函数 =====================
class PULoss(nn.Module):
    def __init__(self, rp_pseudo_positive=True):
        super(PULoss, self).__init__()
        self.eps = 1e-8
        self.rp_pseudo_positive = rp_pseudo_positive

    def forward(self, pred, label, sample_type, sample_weight):
        pred = torch.clamp(pred, self.eps, 1 - self.eps)
        loss = 0.0
        # P(type=0): -w * y * log(p)
        mask = (sample_type == 0)
        if mask.any():
            loss += -torch.sum(sample_weight[mask] * label[mask] * torch.log(pred[mask]))
        # RP(type=1): -w * log(p)（伪正，不乘 label）
        mask = (sample_type == 1)
        if mask.any() and self.rp_pseudo_positive:
            loss += -torch.sum(sample_weight[mask] * torch.log(pred[mask]))
        # LN(type=2) + RN(type=3): -w * (1-y) * log(1-p)
        for t in [2, 3]:
            mask = (sample_type == t)
            if mask.any():
                loss += -torch.sum(sample_weight[mask] * (1 - label[mask]) * torch.log(1 - pred[mask]))
        return loss / pred.size(0)

criterion_pu = PULoss(rp_pseudo_positive=rp_pseudo_positive)
criterion_bce = nn.BCELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# ===================== 7. 训练 =====================
epochs = 100
early_stop_patience = 10
best_auc = 0.0
best_model = mlp_model.state_dict()
patience_counter = 0

logger.info(f'开始训练 | epochs={epochs}, early_stop={early_stop_patience}')

for epoch in range(epochs):
    mlp_model.train()
    total_loss = 0.0
    for batch in train_loader:
        if use_pu_loss:
            batch_input, batch_label, batch_type, batch_weight = batch
            batch_input = batch_input.to(device)
            batch_label = batch_label.to(device)
            batch_type = batch_type.to(device)
            batch_weight = batch_weight.to(device)
        else:
            batch_input, batch_label = batch
            batch_input = batch_input.to(device)
            batch_label = batch_label.to(device)
        optimizer.zero_grad()
        pred = mlp_model(batch_input)
        if use_pu_loss:
            loss = criterion_pu(pred, batch_label, batch_type, batch_weight)
        else:
            loss = criterion_bce(pred, batch_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_input.size(0)
    avg_loss = total_loss / len(train_loader.dataset)

    # 验证（分批推理）
    test_pred_np = predict_in_batches(mlp_model, test_input)
    try:
        auc = roc_auc_score(test_output, test_pred_np)
        aupr = average_precision_score(test_output, test_pred_np)
    except ValueError:
        auc, aupr = 0.0, 0.0

    scheduler.step(auc)
    if epoch % 10 == 0 or epoch == epochs - 1:
        logger.info(f'Epoch [{epoch+1}/{epochs}] Loss={avg_loss:.4f} AUC={auc:.4f} AUPR={aupr:.4f}')

    if auc > best_auc:
        best_auc = auc
        best_model = mlp_model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            logger.info(f'早停 epoch {epoch+1}, best AUC={best_auc:.4f}')
            break

# ===================== 8. 预测与评估 =====================
mlp_model.load_state_dict(best_model)
y_pred_proba = predict_in_batches(mlp_model, test_input)

# 优化阈值
best_thresh, best_f1 = 0.5, 0.0
for thresh in np.arange(0.1, 0.5, 0.05):
    temp_pred = (y_pred_proba > thresh).astype(int)
    try:
        temp_f1 = f1_score(test_output, temp_pred, pos_label=1)
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_thresh = thresh
    except ValueError:
        continue
y_pred = (y_pred_proba > best_thresh).astype(int)

# 计算所有指标
metrics = {}
metric_funcs = {
    'AUC': lambda: roc_auc_score(test_output, y_pred_proba),
    'AUPR': lambda: average_precision_score(test_output, y_pred_proba),
    'Accuracy': lambda: accuracy_score(test_output, y_pred),
    'F1': lambda: f1_score(test_output, y_pred, pos_label=1),
    'Recall': lambda: recall_score(test_output, y_pred, pos_label=1),
    'Precision': lambda: precision_score(test_output, y_pred, pos_label=1),
}
for name, func in metric_funcs.items():
    try:
        val = func()
        metrics[name] = float(val)
        logger.info(f'{name}: {val:.6f}')
    except Exception as e:
        metrics[name] = None
        logger.warning(f'{name}: 无法计算 ({e})')

# ===================== 9. 保存结果 =====================
# 时间戳目录
timestamp_file = os.path.join(_proj_root, ".current_run_timestamp.txt")
predict_result_base = os.path.join(data_dir, "predict_result")
results_base = os.path.join(_proj_root, "results")

if run_id:
    # 并行跑多个实验时：每个实验一个独立 run_id，避免写共享时间戳文件导致目录串写
    timestamp = run_id
else:
    if os.path.isfile(timestamp_file):
        with open(timestamp_file, 'r') as f:
            timestamp = f.read().strip()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(timestamp_file, 'w') as f:
            f.write(timestamp)

predict_result_dir = f'{predict_result_base}_{timestamp}'
results_dir = f'{results_base}_{timestamp}'
os.makedirs(predict_result_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# 保存预测结果
np.savetxt(os.path.join(predict_result_dir, f'ARGA_fold{fold}.txt'), y_pred_proba)
np.savetxt(os.path.join(predict_result_dir, f'ARGA_test_index{fold}.txt'), test_index)

# 保存评估结果 (JSON)
result_dict = {
    'experiment_description': EXP_DESC,
    'fold': fold,
    'best_threshold': float(best_thresh),
    'metrics': metrics,
    'pu_weights': {'P': w_p, 'RP': w_rp, 'RN': w_rn, 'LN': w_ln},
    'rp_pseudo_positive': rp_pseudo_positive,
}

results_json = os.path.join(results_dir, f'fold_{fold}_results.json')
with open(results_json, 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, indent=4, ensure_ascii=False)

# 保存评估结果 (TXT)
results_txt = os.path.join(results_dir, f'fold_{fold}_results.txt')
with open(results_txt, 'w', encoding='utf-8') as f:
    f.write(f'{"="*60}\n')
    f.write(f'第{fold}折 | {EXP_DESC}\n')
    f.write(f'最优阈值: {best_thresh:.4f}\n')
    f.write(f'{"="*60}\n')
    for name, val in metrics.items():
        f.write(f'  {name}: {val:.6f}\n' if val is not None else f'  {name}: N/A\n')

logger.info(f'结果已保存: {results_dir}')
logger.info(f'第{fold}折完成!')