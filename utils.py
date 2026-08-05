import os
import random
import numpy as np
import torch
import dgl
import logging

# 蛋白质序列编码映射（保持不变）
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9, "K": 10,
    "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19,
    "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
}
CHARPROTLEN = 25

# SMILES编码映射（虽然模型不再用RetNet处理，但仍需保留输入兼容性）
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
CHARISOSMILEN = 64

def smiles2onehot(smiles, MAX_SMI_LEN=290):
    """
    SMILES序列编码（模型虽不再用RetNet处理，但仍需保留输入格式）
    用于保持数据加载兼容性，实际模型中会忽略该特征
    """
    x = np.zeros(MAX_SMI_LEN, dtype=int)  # 明确指定dtype为int，避免后续转换警告
    for i, ch in enumerate(smiles[:MAX_SMI_LEN]):
        if ch in CHARISOSMISET:
            x[i] = CHARISOSMISET[ch]
        else:
            # 处理未定义字符（用0填充，即padding）
            logging.warning(f"SMILES character {ch} not in CHARISOSMISET, using padding")
            x[i] = 0
    return x

def set_seed(seed=1000):
    """固定随机种子（保持不变）"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时确保所有GPU种子一致
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def graph_collate_func(x):
    """
    批处理函数：适配模型输入格式（bg, protein, smiles, interactions）
    注意：smiles虽未被模型使用，但仍需保留以维持数据加载流程
    """
    # 解包数据元组（确保顺序正确）
    bg_list, protein_list, smiles_list, interactions_list = zip(*x)
    
    # 处理分子图（DGL批量处理）
    bg_batch = dgl.batch(bg_list)
    
    # 处理蛋白质序列（转换为tensor）
    protein_tensor = torch.tensor(np.array(protein_list), dtype=torch.long)  # 明确指定long类型
    
    # 处理SMILES（转换为tensor，仅为保持格式兼容）
    smiles_tensor = torch.tensor(np.array(smiles_list), dtype=torch.long)
    
    # 处理相互作用标签（转换为tensor，float类型用于后续损失计算）
    interactions_tensor = torch.tensor(interactions_list, dtype=torch.float)
    
    return bg_batch, protein_tensor, smiles_tensor, interactions_tensor

def mkdir(path):
    """创建目录（保持不变）"""
    path = path.strip().rstrip("\\")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # 添加exist_ok=True，避免多进程创建时报错

def integer_label_protein(sequence, max_length=1200):
    """
    蛋白质序列整数编码（保持不变）
    """
    encoding = np.zeros(max_length, dtype=int)  # 明确指定int类型
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"Protein character {letter} not in CHARPROTSET, treated as padding"
            )
            encoding[idx] = 0  # 未定义字符用0填充（padding）
    return encoding