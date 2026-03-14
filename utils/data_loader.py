# -*- coding: utf-8 -*-
"""
数据加载统一接口模块

提供词汇表加载、数据集加载、DataLoader 构建等统一入口。
"""

import json
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset

from generate_graph_data_fromJson import CallGraphDatasetFull_Lazy
from utils.data_utils import split_train_val_test, split_train_test

# =============================================================================
# 工具函数
# =============================================================================

def load_dict(path: str) -> dict:
    """加载 JSON 字典"""
    with open(path, 'r') as f:
        return json.load(f)


# =============================================================================
# 词汇表加载
# =============================================================================

def load_vocabs(vocab_dir: str):
    """
    加载词汇表

    Args:
        vocab_dir: 词汇表目录路径

    Returns:
        name2idx, type2idx, behavior2idx, edge_type2idx: 词汇表字典
    """
    vocab_dir = Path(vocab_dir)
    name2idx = load_dict(str(vocab_dir / "name2idx.json"))
    type2idx = load_dict(str(vocab_dir / "type2idx.json"))
    behavior2idx = load_dict(str(vocab_dir / "behavior2idx.json"))
    edge_type2idx = load_dict(str(vocab_dir / "edge_type2idx.json"))
    print('[data_loader] Vocabulary loaded.')
    return name2idx, type2idx, behavior2idx, edge_type2idx


def get_vocab_sizes(vocab: dict) -> tuple:
    """获取词汇表大小"""
    return (
        len(vocab['name2idx']),
        len(vocab['type2idx']),
        len(vocab['behavior2idx']),
        len(vocab['edge_type2idx'])
    )


# =============================================================================
# 数据集加载
# =============================================================================

def load_month_dataset(
    month: str,
    vocab: dict,
    data_paths: dict
) -> tuple:
    """
    加载指定月份的数据集

    Args:
        month: 月份 (如 '2022-01')
        vocab: 词汇表字典
        data_paths: 数据路径字典，包含:
            - benign_root: 良性样本根目录
            - malicious_root: 恶意样本根目录
            - benign_out: 良性样本输出目录
            - malicious_out: 恶意样本输出目录

    Returns:
        根据 split 参数返回:
        - train_val_test: (normal_train, normal_val, normal_test, malicious_train, malicious_val, malicious_test)
        - train_test: (normal_train, normal_test, malicious_train, malicious_test)
        - concat: (train_dataset, normal_test, malicious_test) - 用于测试场景
        - all: (normal_ds, malicious_ds) - 原始数据集
    """
    normal_ds = CallGraphDatasetFull_Lazy(
        root_dir=data_paths['benign_root'],
        output_dir=data_paths['benign_out'],
        name2idx=vocab['name2idx'],
        type2idx=vocab['type2idx'],
        behavior2idx=vocab['behavior2idx'],
        edge_type2idx=vocab['edge_type2idx'],
        fixed_label=0,
        start_month=month,
        end_month=month
    )

    malicious_ds = CallGraphDatasetFull_Lazy(
        root_dir=data_paths['malicious_root'],
        output_dir=data_paths['malicious_out'],
        name2idx=vocab['name2idx'],
        type2idx=vocab['type2idx'],
        behavior2idx=vocab['behavior2idx'],
        edge_type2idx=vocab['edge_type2idx'],
        fixed_label=1,
        start_month=month,
        end_month=month
    )

    
    return normal_ds, malicious_ds

# =============================================================================
# DataLoader 构建
# =============================================================================

def build_dataloaders(
    datasets: dict,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4
) -> dict:
    """
    构建 DataLoader（按月份）

    Args:
        datasets: {month: (normal_ds, malicious_ds)} 格式的字典
        batch_size: 批次大小
        shuffle: 是否打乱（训练集=True，验证/测试集=False）
        num_workers: 工作进程数

    Returns:
        {month: DataLoader} 格式的字典
    """
    loaders = {}
    for month, (normal_ds, malicious_ds) in datasets.items():
        if len(normal_ds) > 0 and len(malicious_ds) > 0:
            loaders[month] = DataLoader(
                ConcatDataset([normal_ds, malicious_ds]),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            )
    return loaders