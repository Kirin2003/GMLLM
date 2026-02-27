"""
基于记忆回放的增量学习实现
- 交替训练: [新任务] -> [记忆库] -> [新任务] -> ...
- 记忆库: 每月抽取最多10个样本，保持正负样本1:1
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from generate_graph_data_fromJson import CallGraphDatasetFull_Lazy
from torch.utils.data import ConcatDataset, Subset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import random
import numpy as np
import time
from month_utils import generate_month_range
from plot_results import plot_monthly_metrics, plot_monthly_incremental_results
from data_util import split_train_val_test, split_train_test

# 从 distinguish_GNN_2 导入模型定义
from distinguish_GNN_2 import GCNWithBehavior, load_dict, set_seed, validate


def select_sample(month_datasets: dict, max_per_month: int = 10) -> list:
    """
    从每个月数据集中随机抽取样本，保持正负样本1:1

    Args:
        month_datasets: {month: (normal_dataset, malicious_dataset)}
        max_per_month: 每月最多抽取样本数（默认10）

    Returns:
        list of (data, label, month) 元组列表
    """
    memory_samples = []

    for month, (normal_ds, malicious_ds) in month_datasets.items():
        # 计算每个类别抽取的数量（保持1:1比例）
        n_select = min(max_per_month // 2, len(normal_ds), len(malicious_ds))

        if n_select == 0:
            continue

        # 随机抽取正负样本索引
        normal_indices = random.sample(range(len(normal_ds)), n_select)
        malicious_indices = random.sample(range(len(malicious_ds)), n_select)

        # 添加正样本 (label=0, benign)
        for idx in normal_indices:
            memory_samples.append((normal_ds[idx], 0, month))

        # 添加负样本 (label=1, malicious)
        for idx in malicious_indices:
            memory_samples.append((malicious_ds[idx], 1, month))

    return memory_samples


class MemoryDataset(Dataset):
    """记忆库数据集封装类"""

    def __init__(self, samples_list):
        """
        Args:
            samples_list: list of (data, label, month)
        """
        super().__init__()
        self.samples = samples_list
        # 为每个样本设置y属性
        for data, label, month in self.samples:
            data.y = torch.tensor([label], dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def get(self, idx):
        data, label, month = self.samples[idx]
        return data


def train_with_CL(model, new_task_loader, memory_loader, optimizer, criterion, device):
    """
    1:1 交替训练
    [新任务batch1] -> [记忆库batch1] -> [新任务batch2] -> [记忆库batch2] -> ...

    Args:
        model: GCNWithBehavior 模型
        new_task_loader: 当前新任务数据加载器
        memory_loader: 记忆库数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备 (cuda/cpu)

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss, correct, total = 0, 0, 0

    new_iter = iter(new_task_loader)
    mem_iter = iter(memory_loader) if memory_loader else None

    while True:
        # 训练新任务batch
        try:
            data = next(new_iter)
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
            total_loss += loss.item() * data.num_graphs
        except StopIteration:
            break

        # 训练记忆库batch (1:1 交替)
        if mem_iter:
            try:
                mem_data = next(mem_iter)
                mem_data = mem_data.to(device)
                optimizer.zero_grad()
                mem_out = model(mem_data)
                mem_loss = criterion(mem_out, mem_data.y)
                mem_loss.backward()
                optimizer.step()

                mem_pred = mem_out.argmax(dim=1)
                correct += (mem_pred == mem_data.y).sum().item()
                total += mem_data.num_graphs
                total_loss += mem_loss.item() * mem_data.num_graphs
            except StopIteration:
                mem_iter = iter(memory_loader)  # 重置，继续循环

    return total_loss / total, correct / total


def train_with_CL_multi_epochs(model, new_task_loader, memory_loader, optimizer,
                                criterion, device, epochs: int = 5):
    """
    多epoch的1:1交替训练（用于每个月的增量训练）

    Args:
        model: GCNWithBehavior 模型
        new_task_loader: 当前新任务数据加载器
        memory_loader: 记忆库数据加载器（可能为空）
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        epochs: 训练轮数

    Returns:
        (avg_loss, accuracy)
    """
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        if memory_loader is not None and len(memory_loader) > 0:
            # 1:1 交替训练
            train_loss, train_acc = train_with_CL(
                model, new_task_loader, memory_loader, optimizer, criterion, device
            )
        else:
            # 仅训练新任务（第一个月没有记忆库）
            from distinguish_GNN_2 import train as basic_train
            train_loss, train_acc = basic_train(
                model, new_task_loader, optimizer, criterion, device
            )

    return train_loss, train_acc


# ============================================================================
# 辅助函数
# ============================================================================

def load_vocabs(vocab_dir: str) -> tuple:
    """加载词汇表"""
    name2idx = load_dict(str(Path(vocab_dir) / "name2idx.json"))
    type2idx = load_dict(str(Path(vocab_dir) / "type2idx.json"))
    behavior2idx = load_dict(str(Path(vocab_dir) / "behavior2idx.json"))
    edge_type2idx = load_dict(str(Path(vocab_dir) / "edge_type2idx.json"))
    print('Load vocab done.')
    return name2idx, type2idx, behavior2idx, edge_type2idx


def load_month_dataset(month: str, vocab: dict, paths: dict) -> tuple:
    """
    加载当月数据集

    Args:
        month: 月份 (如 '2022-01')
        vocab: 词汇表字典
        paths: 数据路径配置

    Returns:
        (normal_dataset, malicious_dataset)
    """
    normal_dataset = CallGraphDatasetFull_Lazy(
        root_dir=paths['benign_root'], output_dir=paths['benign_out'],
        name2idx=vocab['name2idx'], type2idx=vocab['type2idx'],
        behavior2idx=vocab['behavior2idx'], edge_type2idx=vocab['edge_type2idx'],
        fixed_label=0, start_month=month, end_month=month
    )
    malicious_dataset = CallGraphDatasetFull_Lazy(
        root_dir=paths['malicious_root'], output_dir=paths['malicious_out'],
        name2idx=vocab['name2idx'], type2idx=vocab['type2idx'],
        behavior2idx=vocab['behavior2idx'], edge_type2idx=vocab['edge_type2idx'],
        fixed_label=1, start_month=month, end_month=month
    )
    return normal_dataset, malicious_dataset


def build_test_loaders(test_datasets: dict, batch_size: int) -> dict:
    """构建所有月份的测试数据加载器"""
    test_loaders = {}
    for month, (normal_test, malicious_test) in test_datasets.items():
        if len(normal_test) > 0 and len(malicious_test) > 0:
            test_loaders[month] = DataLoader(
                ConcatDataset([normal_test, malicious_test]),
                batch_size=batch_size, shuffle=False, num_workers=4
            )
    return test_loaders


def initialize_model(vocab: dict, device: str, model_config: dict = None):
    """
    初始化模型、优化器和损失函数

    Args:
        vocab: 词汇表
        device: 设备
        model_config: 模型配置

    Returns:
        (model, optimizer, criterion, device)
    """
    if model_config is None:
        model_config = {'lr': 0.005, 'weight_decay': 1e-3}

    device = torch.device(device)
    model = GCNWithBehavior(
        name_vocab_size=len(vocab['name2idx']),
        type_vocab_size=len(vocab['type2idx']),
        behavior_dim=len(vocab['behavior2idx'])
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'], weight_decay=model_config['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion, device


def create_memory_loader(memory_samples: list, batch_size: int):
    """创建记忆库数据加载器"""
    if len(memory_samples) > 0:
        memory_dataset = MemoryDataset(memory_samples)
        return DataLoader(memory_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return None


def train_month(model, month_train_loader, memory_loader, optimizer, criterion,
                device, epochs: int = 5) -> tuple:
    """
    训练当月任务（增量训练模式，memory_loader 必定有内容）

    Args:
        model: 模型
        month_train_loader: 当月训练数据
        memory_loader: 记忆库数据（必有内容）
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        epochs: 训练轮数

    Returns:
        (last_loss, last_acc)
    """
    last_loss, last_acc = 0.0, 0.0

    for epoch in range(1, epochs + 1):
        # 1:1 交替训练
        train_loss, train_acc = train_with_CL(
            model, month_train_loader, memory_loader,
            optimizer, criterion, device
        )

        print(f"  Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        last_loss, last_acc = train_loss, train_acc

    return last_loss, last_acc


def evaluate_all_months(model, test_loaders: dict, device) -> dict:
    """
    在所有已见月份上评估模型

    Returns:
        results: {'month': [...], 'f1': [...], 'acc': [...], ...}
    """
    results = {'month': [], 'f1': [], 'acc': [], 'precision': [], 'recall': []}

    for month in sorted(test_loaders.keys()):
        test_loader = test_loaders[month]
        metrics = validate(model, test_loader, device)
        f1, acc, malicious_recall, malicious_precision = metrics

        results['month'].append(month)
        results['f1'].append(f1)
        results['acc'].append(acc)
        results['precision'].append(malicious_precision)
        results['recall'].append(malicious_recall)

        print(f"{month} | F1: {f1:.4f} | Acc: {acc:.4f} | Precision: {malicious_precision:.4f} | Recall: {malicious_recall:.4f}")

    return results


def save_results(results: dict, output_path: str):
    """保存结果到JSON文件"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def print_memory_stats(memory_samples: list):
    """打印记忆库统计信息"""
    n_benign = sum(1 for _, label, _ in memory_samples if label == 0)
    n_malicious = sum(1 for _, label, _ in memory_samples if label == 1)
    print(f"  Memory bank: {len(memory_samples)} samples ({n_benign} benign, {n_malicious} malicious)")


# ============================================================================
# 主函数
# ============================================================================

def run_continual_learning(
    vocab_dir: str,
    data_paths: dict,
    base_train_months: tuple = ('2022-01', '2023-02'),  # 基础训练月份
    incremental_months: tuple = ('2023-03', '2024-12'),  # 增量学习月份
    incremental_epochs: int = 5,  # 增量学习每轮训练轮数
    batch_size: int = 128,
    memory_per_month: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    pretrained_model_path: str = "./models/base_model_0207.pt"  # 预训练模型路径
):
    """
    运行增量学习训练流程

    流程:
    1. 阶段1: 加载预训练模型 - 直接加载 pretrained_model_path
    2. 阶段2: 基础记忆库 - 从 base_train_months 抽取代表性样本
    3. 阶段3: 增量学习 (incremental_months 每月)
       - 训练: 当月数据 + memory
       - 评估: 2022-01 ~ month+1 的指标
       - 更新memory: 加入当月代表性样本

    Args:
        vocab_dir: 词汇表目录
        data_paths: 数据路径字典
        base_train_months: 基础训练月份范围（用于构建记忆库）
        incremental_months: 增量学习月份范围
        incremental_epochs: 增量学习每轮训练轮数
        batch_size: 批次大小
        memory_per_month: 每月记忆库样本数
        device: 设备
        seed: 随机种子
        pretrained_model_path: 预训练模型路径
    """
    # 设置随机种子
    set_seed(seed)

    # 1. 加载词汇表
    name2idx, type2idx, behavior2idx, edge_type2idx = load_vocabs(vocab_dir)
    vocab = {'name2idx': name2idx, 'type2idx': type2idx,
             'behavior2idx': behavior2idx, 'edge_type2idx': edge_type2idx}

    # 初始化
    model, optimizer, criterion, device = None, None, None, device
    train_datasets = {}  # 训练集 {month: (normal_train, malicious_train)}
    test_datasets = {}    # 测试集 {month: (normal_test, malicious_test)}


    # =========================================================================
    # 阶段2: 构建基础记忆库
    # =========================================================================
    print("\n" + "="*60)
    print("Phase 2: Building base memory bank")
    print("="*60)

    base_start, base_end = base_train_months
    inc_start, inc_end = incremental_months

    # 加载基础训练月份数据（使用8:1:1划分）
    for month in generate_month_range(base_start, base_end):
        normal_ds, malicious_ds = load_month_dataset(month, vocab, data_paths)
        print(f"  {month}: {len(normal_ds)} normal, {len(malicious_ds)} malicious")

        # 划分数据集
        (normal_train, normal_val, normal_test,
         malicious_train, malicious_val, malicious_test) = split_train_val_test(normal_ds, malicious_ds)

        # 存储训练数据用于构建记忆库
        train_datasets[month] = (normal_train, malicious_train)
        # 存储测试集
        test_datasets[month] = (normal_test, malicious_test)

    # 加载增量月份数据（使用8:2划分）
    for month in generate_month_range(inc_start, inc_end):
        normal_ds, malicious_ds = load_month_dataset(month, vocab, data_paths)
        print(f"  {month}: {len(normal_ds)} normal, {len(malicious_ds)} malicious")

        # 划分数据集 8:2
        (normal_train, normal_test,
         malicious_train, malicious_test) = split_train_test(normal_ds, malicious_ds)

        # 训练数据存储用于增量训练
        train_datasets[month] = (normal_train, malicious_train)
        # 测试数据存储
        test_datasets[month] = (normal_test, malicious_test)

    # 构建基础月份记忆库
    memory_samples = []  # 记忆库
    for month in generate_month_range(base_start, base_end):
        if month in train_datasets:
            normal_train, malicious_train = train_datasets[month]
            month_datasets = {month: (normal_train, malicious_train)}
            new_samples = select_sample(month_datasets, max_per_month=memory_per_month)
            memory_samples.extend(new_samples)

    print_memory_stats(memory_samples)

    # =========================================================================
    # 阶段3: 增量学习
    # =========================================================================
    print("\n" + "="*60)
    print("Phase 3: Incremental Learning")
    print("="*60)

    # 收集评估结果
    seen_months_results = {'month': [], 'f1': [], 'acc': [], 'precision': [], 'recall': []}
    future_month_results = {'month': [], 'f1': [], 'acc': [], 'precision': [], 'recall': []}

    for month in generate_month_range(inc_start, inc_end):
        print(f"\n--- Month: {month} ---")

        # 从已划分的数据集中获取当月训练数据
        normal_train, malicious_train = train_datasets[month]
        month_train_dataset = ConcatDataset([normal_train, malicious_train])
        month_train_loader = DataLoader(
            month_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        # 创建记忆库loader
        memory_loader = create_memory_loader(memory_samples, batch_size)

        # 训练: 当月数据 + 记忆库
        print(f"Training with memory replay...")
        train_month(model, month_train_loader, memory_loader,
                   optimizer, criterion, device, incremental_epochs)

        # 保存当月模型
        model_path = f"./models/incremental_model_{month}.pt"
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # 评估: 2022-01 ~ month 的所有已见月份
        # 重新构建包含当月的测试loaders
        seen_test_datasets = {m: test_datasets[m] for m in test_datasets if m <= month}
        seen_test_loaders = build_test_loaders(seen_test_datasets, batch_size)

        print(f"Evaluating on months up to {month}...")
        for test_month in sorted(seen_test_loaders.keys()):
            test_loader = seen_test_loaders[test_month]
            metrics = validate(model, test_loader, device)
            f1, acc, recall, precision = metrics
            print(f"  {test_month}: F1={f1:.4f}, Acc={acc:.4f}")

            # 记录已见月份结果（用于画图）
            seen_months_results['month'].append(test_month)
            seen_months_results['f1'].append(f1)
            seen_months_results['acc'].append(acc)
            seen_months_results['precision'].append(precision)
            seen_months_results['recall'].append(recall)

        # 评估: 当月后一个月
        from month_utils import generate_month_range as gen_month_range
        all_months_list = list(gen_month_range(inc_start, inc_end))
        month_idx = all_months_list.index(month)
        if month_idx + 1 < len(all_months_list):
            next_month = all_months_list[month_idx + 1]
            if next_month in test_datasets:
                print(f"Evaluating on next month {next_month}...")
                next_test_loader = build_test_loaders({next_month: test_datasets[next_month]}, batch_size)
                next_loader = next_test_loader[next_month]
                metrics = validate(model, next_loader, device)
                f1, acc, recall, precision = metrics
                print(f"  {next_month}: F1={f1:.4f}, Acc={acc:.4f}")

                # 记录后一个月结果（用于画图）
                future_month_results['month'].append(next_month)
                future_month_results['f1'].append(f1)
                future_month_results['acc'].append(acc)
                future_month_results['precision'].append(precision)
                future_month_results['recall'].append(recall)

        # 更新记忆库: 加入当月代表性样本
        print(f"Updating memory bank...")
        month_datasets = {month: train_datasets[month]}
        new_samples = select_sample(month_datasets, max_per_month=memory_per_month)
        memory_samples.extend(new_samples)
        print_memory_stats(memory_samples)

    output_dir = "./results/"

    # 保存增量学习评估结果
    seen_results_file = output_dir + "continual_learning_seen_months.json"
    save_results(seen_months_results, seen_results_file)

    future_results_file = output_dir + "continual_learning_future_month.json"
    save_results(future_month_results, future_results_file)

    # 绘制增量学习评估曲线
    print("\nPlotting incremental learning results...")
    plot_monthly_incremental_results(seen_months_results, future_month_results, output_dir)

    return model, seen_months_results, future_month_results


if __name__ == "__main__":

    # ===== 配置参数 =====
    vocab_dir = "/Data2/hxq/datasets/incremental_packages_subset/vocab"
    data_paths = {
        'benign_root': "/Data2/hxq/datasets/incremental_packages_subset/benign",
        'malicious_root': "/Data2/hxq/datasets/incremental_packages_subset/malicious",
        'benign_out': "/Data2/hxq/datasets/incremental_packages_subset/benign_call_processed",
        'malicious_out': "/Data2/hxq/datasets/incremental_packages_subset/malicious_call_processed",
    }

    # 运行增量学习
    run_continual_learning(
        vocab_dir=vocab_dir,
        data_paths=data_paths,
        base_train_months=('2022-01', '2023-02'),  # 基础训练月份
        incremental_months=('2023-03', '2024-12'),  # 增量学习月份
        incremental_epochs=5,  # 增量学习每轮5个epoch
        batch_size=128,
        memory_per_month=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        pretrained_model_path="./models/base_model_0207.pt"  # 加载预训练模型
    )
