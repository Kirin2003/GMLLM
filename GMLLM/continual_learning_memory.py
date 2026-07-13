"""
基于记忆回放的增量学习实现
- 交替训练: [新任务] -> [记忆库] -> [新任务] -> ...
- 记忆库: 每月抽取最多10个样本，保持正负样本1:1
"""

import sys
from pathlib import Path

# 将上级目录加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import yaml
from utils.config_utils import load_config
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
import random
from utils.month_utils import generate_month_range
from utils.data_utils import split_train_val_test, split_train_test
from utils.data_loader import load_vocabs, load_month_dataset, build_dataloaders
import copy

# 从 distinguish_GNN_2 导入模型定义
from distinguish_GNN_2 import GCNWithBehavior, validate
from utils.logger_utils import Logger
log = Logger("continual_learning.log")


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

    def len(self):
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

        log.log(f"  Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        last_loss, last_acc = train_loss, train_acc

    return last_loss, last_acc


def save_results(results: dict, output_path: str):
    """保存结果到JSON文件"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.log(f"\nResults saved to {output_path}")


def print_memory_stats(memory_samples: list):
    """打印记忆库统计信息"""
    n_benign = sum(1 for _, label, _ in memory_samples if label == 0)
    n_malicious = sum(1 for _, label, _ in memory_samples if label == 1)
    log.log(f"  Memory bank: {len(memory_samples)} samples ({n_benign} benign, {n_malicious} malicious)")


# ============================================================================
# 主函数
# ============================================================================

def run_continual_learning_unk(
    vocab_dir: str,
    data_paths: dict,
    base_train_months: tuple = ('2022-01', '2023-02'),
    incremental_months: tuple = ('2023-03', '2024-12'),
    incremental_epochs: int = 5,
    batch_size: int = 128,
    memory_per_month: int = 10,
    use_memory: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    pretrained_model_path: str = "./models/base_model.pt",
    result_file: str = "continual_learning_unk_test_than_train_future_month.json",
    model_save_path: str = "./models/incremental_unk_model_",
    results_dir: str = "../results"
):
    output_dir = Path(results_dir)
    future_results_file = output_dir / result_file
    print(f"Results will be saved to: {future_results_file}")
    """
    方案1: UNK映射（对照组）
    - 使用固定词汇表（基础训练月份构建的词汇表）
    - 新API自动映射到UNK
    - 记录每月发现的新API数量

    Returns:
        model, seen_months_results, future_month_results
    """
    from distinguish_GNN_2 import set_seed
    set_seed(seed)

    log.log("="*60)
    log.log("Running Feature Incremental Learning - Method 1: UNK Mapping (Baseline)")
    log.log("="*60)

    # 1. 加载词汇表（使用固定的词汇表，作为base vocab）
    name2idx, type2idx, behavior2idx, edge_type2idx = load_vocabs(vocab_dir)
    vocab = {'name2idx': name2idx, 'type2idx': type2idx,
             'behavior2idx': behavior2idx, 'edge_type2idx': edge_type2idx}

    base_vocab_size = len(name2idx)
    log.log(f"Base vocabulary size: {base_vocab_size}")


    # 2. 初始化模型
    device = torch.device(device)
    model = GCNWithBehavior(
        name_vocab_size=len(vocab['name2idx']),
        type_vocab_size=len(vocab['type2idx']),
        behavior_dim=len(vocab['behavior2idx'])
    ).to(device)

    # 加载预训练模型
    if os.path.exists(pretrained_model_path):
        log.log(f"Loading pretrained model from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        log.log("Pretrained model loaded successfully")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 3. 加载数据
    train_datasets = {}
    test_datasets = {}
    unseen_test_datasets = {}  # 用于记录增量月份的完整测试集（不划分）

    # 加载基础训练月份数据（使用8:1:1划分）
    base_start, base_end = base_train_months
    for month in generate_month_range(base_start, base_end):
        normal_ds, malicious_ds = load_month_dataset(month, vocab, data_paths)
        log.log(f"  {month}: {len(normal_ds)} normal, {len(malicious_ds)} malicious")

        (normal_train, normal_val, normal_test,
         malicious_train, malicious_val, malicious_test) = split_train_val_test(normal_ds, malicious_ds)

        train_datasets[month] = (normal_train, malicious_train)
        test_datasets[month] = (normal_test, malicious_test)

    # 加载增量月份数据（使用8:2划分）
    inc_start, inc_end = incremental_months
    for month in generate_month_range(inc_start, inc_end):
        normal_ds, malicious_ds = load_month_dataset(month, vocab, data_paths)
        log.log(f"  {month}: {len(normal_ds)} normal, {len(malicious_ds)} malicious")

        unseen_test_datasets[month] = (copy.deepcopy(normal_ds), copy.deepcopy(malicious_ds))
        # # 划分数据集 8:2
        (normal_train, normal_test,
         malicious_train, malicious_test) = split_train_test(normal_ds, malicious_ds)

        train_datasets[month] = (normal_train, malicious_train)
        test_datasets[month] = (normal_test, malicious_test)

    # 4. 构建基础月份记忆库
    memory_samples = []
    if use_memory:
        for month in generate_month_range(base_start, base_end):
            if month in train_datasets:
                normal_train, malicious_train = train_datasets[month]
                month_datasets = {month: (normal_train, malicious_train)}
                new_samples = select_sample(month_datasets, max_per_month=memory_per_month)
                memory_samples.extend(new_samples)

    print_memory_stats(memory_samples)

    future_month_result = {'month': [], 'f1': [], 'acc': [], 'precision': [], 'recall': [], 'train_time': []}
    seen_months_results = {'month': [], 'f1': [], 'acc': [], 'precision': [], 'recall': []}

    # 5. 增量学习
    log.log("\n" + "="*60)
    log.log("Phase 3: Incremental Learning with UNK Mapping")
    log.log("="*60)

    for month in generate_month_range(inc_start, inc_end):
        log.log(f"\n--- Month: {month} ---")

        # ========== 评估阶段：先评估再训练 ==========

        # 评估: 当月数据（用之前的模型评估）
        if month in unseen_test_datasets:
            current_month_loader = build_dataloaders({month: unseen_test_datasets[month]}, batch_size, shuffle=False)
            current_loader = current_month_loader[month]
            log.log(f"Evaluating on current month {month} (before training), {len(current_loader.dataset)} samples...")
            metrics = validate(model, current_loader, device)
            f1, acc, recall, precision = metrics
            log.log(f"  {month}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

            future_month_result['month'].append(month)
            future_month_result['f1'].append(f1)
            future_month_result['acc'].append(acc)
            future_month_result['precision'].append(precision)
            future_month_result['recall'].append(recall)

        # ========== 训练阶段 ==========

        # 从已划分的数据集中获取当月训练数据
        normal_train, malicious_train = train_datasets[month]
        month_train_dataset = ConcatDataset([normal_train, malicious_train])
        month_train_loader = DataLoader(
            month_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        # 创建记忆库loader
        memory_loader = create_memory_loader(memory_samples, batch_size) if use_memory else None

        # 记录训练开始时间
        train_start_time = time.time()

        # 训练: 当月数据 + 记忆库
        if use_memory:
            train_month(model, month_train_loader, memory_loader,
                       optimizer, criterion, device, incremental_epochs)
        else:
            log.log(f"Training without memory replay...")
            train_month(model, month_train_loader, None,
                       optimizer, criterion, device, incremental_epochs)

        # 计算当月训练耗时
        train_elapsed = time.time() - train_start_time
        log.log(f"  Training time: {train_elapsed:.2f}s")
        future_month_result['train_time'].append(train_elapsed)

        # 保存当月模型
        model_path = f"{model_save_path}{month}.pt"
        os.makedirs(os.path.dirname(model_path) or "./models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        log.log(f"Model saved to {model_path}")

        # 评估已见月份（累积测试集：当月及之前所有月份）
        cumulative_test_normal = []
        cumulative_test_malicious = []
        for past_month in generate_month_range(inc_start, month):
            if past_month in test_datasets:
                n, m = test_datasets[past_month]
                cumulative_test_normal.extend(n)
                cumulative_test_malicious.extend(m)

        if cumulative_test_normal and cumulative_test_malicious:
            cumulative_test_dataset = ConcatDataset([cumulative_test_normal, cumulative_test_malicious])
            cumulative_test_loader = DataLoader(cumulative_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            log.log(f"Evaluating on seen months (cumulative test set, {len(cumulative_test_dataset)} samples)...")
            metrics = validate(model, cumulative_test_loader, device)
            f1, acc, recall, precision = metrics
            log.log(f"  Seen months cumulative: {month}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

            seen_months_results['month'].append(month)
            seen_months_results['f1'].append(f1)
            seen_months_results['acc'].append(acc)
            seen_months_results['precision'].append(precision)
            seen_months_results['recall'].append(recall)

        # ========== 更新记忆库 ==========
        if use_memory:
            log.log(f"Updating memory bank...")
            month_datasets = {month: train_datasets[month]}
            new_samples = select_sample(month_datasets, max_per_month=memory_per_month)
            memory_samples.extend(new_samples)
            print_memory_stats(memory_samples)

    # 保存新API统计
    output_dir.mkdir(parents=True, exist_ok=True)

    # 计算并添加 future_month_result 平均值
    if future_month_result['f1']:
        future_month_result['avg_f1'] = sum(future_month_result['f1']) / len(future_month_result['f1'])
        future_month_result['avg_acc'] = sum(future_month_result['acc']) / len(future_month_result['acc'])
        future_month_result['avg_precision'] = sum(future_month_result['precision']) / len(future_month_result['precision'])
        future_month_result['avg_recall'] = sum(future_month_result['recall']) / len(future_month_result['recall'])
        future_month_result['avg_train_time'] = sum(future_month_result['train_time']) / len(future_month_result['train_time'])
        log.log(f"Future month average: F1={future_month_result['avg_f1']:.4f}, Acc={future_month_result['avg_acc']:.4f}, "
                f"Precision={future_month_result['avg_precision']:.4f}, Recall={future_month_result['avg_recall']:.4f}")
        log.log(f"Average training time per month: {future_month_result['avg_train_time']:.2f}s")

    # 添加 seed 信息
    future_month_result['seed'] = seed

    future_results_file = output_dir / result_file
    with open(future_results_file, 'w') as f:
        json.dump(future_month_result, f, indent=2)
    log.log(f"Future month results saved to {future_results_file}")

    # 计算并添加 seen_months_results 平均值
    if seen_months_results['f1']:
        seen_months_results['avg_f1'] = sum(seen_months_results['f1']) / len(seen_months_results['f1'])
        seen_months_results['avg_acc'] = sum(seen_months_results['acc']) / len(seen_months_results['acc'])
        seen_months_results['avg_precision'] = sum(seen_months_results['precision']) / len(seen_months_results['precision'])
        seen_months_results['avg_recall'] = sum(seen_months_results['recall']) / len(seen_months_results['recall'])
        log.log(f"Seen months average: F1={seen_months_results['avg_f1']:.4f}, Acc={seen_months_results['avg_acc']:.4f}, "
                f"Precision={seen_months_results['avg_precision']:.4f}, Recall={seen_months_results['avg_recall']:.4f}")

    # 添加 seed 信息
    seen_months_results['seed'] = seed

    # 保存已见月份结果
    seen_results_file = output_dir / result_file.replace('future_month', 'seen_month')
    with open(seen_results_file, 'w') as f:
        json.dump(seen_months_results, f, indent=2)
    log.log(f"Seen months results saved to {seen_results_file}")

    return model, seen_months_results, future_month_result

if __name__ == "__main__":
    import argparse

    # ===== 解析命令行参数 =====
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/default.yaml', help='配置文件路径')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（覆盖配置文件）')
    args = parser.parse_args()

    # ===== 加载配置文件 =====
    config_path = args.config
    config = load_config(config_path)

    # 数据集路径
    base_path = config['dataset']['base_path']
    vocab_dir = str(Path(base_path) / config['dataset']['vocab_dir'])
    data_paths = {
        'benign_root': str(Path(base_path) / config['dataset']['benign_root']),
        'malicious_root': str(Path(base_path) / config['dataset']['malicious_root']),
        'benign_out': str(Path(base_path) / config['dataset']['benign_out']),
        'malicious_out': str(Path(base_path) / config['dataset']['malicious_out']),
    }

    # 增量学习配置
    cl_config = config.get('continual_learning', {})
    base_train_months = tuple(cl_config.get('base_train_months', ['2022-01', '2023-02']))
    incremental_months = tuple(cl_config.get('incremental_months', ['2023-03', '2024-12']))
    incremental_epochs = cl_config.get('incremental_epochs', 5)
    memory_per_month = cl_config.get('memory_per_month', 10)
    use_memory = cl_config.get('use_memory', True)

    # 训练参数
    batch_size = config['training']['batch_size']
    seed = args.seed if args.seed is not None else config['training']['seed']

    # 设备配置
    device_config = config.get('device', 'auto')
    if device_config == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config

    # 路径配置
    paths_config = config.get('paths', {})
    pretrained_model_path = paths_config.get('pretrained_model', './models/base_model.pt')

    # 结果文件配置 (支持多种配置方式)
    results_config = config.get('results', {})
    result_file = results_config.get('future_month')        

    # 模型保存路径配置 (组合 models_dir 和 prefix)
    models_dir = paths_config.get('models_dir', './models')
    results_dir = paths_config.get('results_dir', '../results')
    model_prefix = paths_config.get('prefix')
    model_save_path = f"{models_dir}/{model_prefix}"

    # 运行增量学习
    run_continual_learning_unk(
        vocab_dir=vocab_dir,
        data_paths=data_paths,
        base_train_months=base_train_months,
        incremental_months=incremental_months,
        incremental_epochs=incremental_epochs,
        batch_size=batch_size,
        memory_per_month=memory_per_month,
        use_memory=use_memory,
        device=device,
        seed=seed,
        pretrained_model_path=pretrained_model_path,
        result_file=result_file,
        model_save_path=model_save_path,
        results_dir=results_dir
    )
