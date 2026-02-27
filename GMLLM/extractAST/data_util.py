"""
数据处理工具函数
"""

import torch
from torch.utils.data import random_split


def split_train_val_test(normal_ds, malicious_ds, train_ratio=0.8, val_ratio=0.1) -> tuple:
    """
    划分训练/验证/测试集

    Args:
        normal_ds: 正常样本数据集
        malicious_ds: 恶意样本数据集
        train_ratio: 训练集比例，默认 0.8
        val_ratio: 验证集比例，默认 0.1

    Returns:
        (normal_train, normal_val, normal_test,
         malicious_train, malicious_val, malicious_test)
    """
    # 划分正常样本
    n_train = int(train_ratio * len(normal_ds))
    n_val = int(val_ratio * len(normal_ds))
    normal_train, normal_val, normal_test = random_split(
        normal_ds, [n_train, n_val, len(normal_ds) - n_train - n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # 划分恶意样本
    m_train = int(train_ratio * len(malicious_ds))
    m_val = int(val_ratio * len(malicious_ds))
    malicious_train, malicious_val, malicious_test = random_split(
        malicious_ds, [m_train, m_val, len(malicious_ds) - m_train - m_val],
        generator=torch.Generator().manual_seed(42)
    )

    return (normal_train, normal_val, normal_test,
            malicious_train, malicious_val, malicious_test)


def split_train_test(normal_ds, malicious_ds, train_ratio=0.8) -> tuple:
    """
    划分训练/测试集

    Args:
        normal_ds: 正常样本数据集
        malicious_ds: 恶意样本数据集
        train_ratio: 训练集比例，默认 0.8

    Returns:
        (normal_train, normal_test, malicious_train, malicious_test)
    """
    # 划分正常样本
    n_train = int(train_ratio * len(normal_ds))
    normal_train, normal_test = random_split(
        normal_ds, [n_train, len(normal_ds) - n_train],
        generator=torch.Generator().manual_seed(42)
    )

    # 划分恶意样本
    m_train = int(train_ratio * len(malicious_ds))
    malicious_train, malicious_test = random_split(
        malicious_ds, [m_train, len(malicious_ds) - m_train],
        generator=torch.Generator().manual_seed(42)
    )

    return (normal_train, normal_test, malicious_train, malicious_test)
