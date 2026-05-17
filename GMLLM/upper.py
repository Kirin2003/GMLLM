"""
累积训练增量学习实现
- 训练：每月用该月之前所有月份的数据训练（包括基础训练集 + 已完成的增量月份）
- 测试：用下一个月数据测试
- 无记忆池，直接使用全部历史数据
- 每次都从头开始训练，epoch 与基础模型相同
- 例如：2023-04用2023-03之前数据训练，测试2023-04；2023-05用2023-03~2023-04训练，测试2023-05
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from utils.month_utils import generate_month_range
from utils.data_utils import split_train_val_test
from utils.data_loader import load_vocabs, load_month_dataset

from distinguish_GNN_2 import GCNWithBehavior, validate, set_seed, train_model
from utils.logger_utils import Logger

log = Logger("accumulate_train.log")


def run_accumulate_train(
    vocab_dir: str,
    data_paths: dict,
    base_train_months: tuple = ('2022-01', '2023-02'),
    test_start_month: str = '2023-03',
    test_end_month: str = '2024-12',
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 128,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    result_file: str = "accumulate_train_results.json",
    model_save_path: str = "./models/accumulate_train_model_"
):
    """
    累积训练：每月用之前所有月份数据训练，测试下一个月（无记忆池）
    每次从头开始训练，使用与基础模型相同的 epoch 数

    Args:
        vocab_dir: 词汇表目录
        data_paths: 数据路径配置
        base_train_months: 基础训练月份范围
        test_start_month: 测试开始月份（第一个增量月份）
        test_end_month: 测试结束月份
        epochs: 训练轮数（与基础模型相同）
        patience: 早停耐心值
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        device: 设备
        seed: 随机种子
        result_file: 结果保存文件名
        model_save_path: 模型保存路径前缀
    """
    output_dir = "../results/"

    set_seed(seed)

    log.log("=" * 60)
    log.log("Running Accumulate Training: Train on all past months, Test on next month")
    log.log("=" * 60)

    # 1. 加载词汇表
    name2idx, type2idx, behavior2idx, edge_type2idx = load_vocabs(vocab_dir)
    vocab = {'name2idx': name2idx, 'type2idx': type2idx,
             'behavior2idx': behavior2idx, 'edge_type2idx': edge_type2idx}

    log.log(f"Vocabulary size: name={len(name2idx)}, type={len(type2idx)}, behavior={len(behavior2idx)}")

    # 2. 加载所有月份数据（所有月份都按 8:1:1 划分）
    all_months_split = {}  # {month: {'train': (n_train, m_train), 'val': (n_val, m_val), 'test': (n_test, m_test)}}

    # 加载基础训练月份
    base_start, base_end = base_train_months
    for month in generate_month_range(base_start, base_end):
        normal_ds, malicious_ds = load_month_dataset(month, vocab, data_paths)
        log.log(f"  {month}: {len(normal_ds)} normal, {len(malicious_ds)} malicious")

        # 8:1:1 划分
        (normal_train, normal_val, normal_test,
         malicious_train, malicious_val, malicious_test) = split_train_val_test(
             normal_ds, malicious_ds, train_ratio, val_ratio)
        all_months_split[month] = {
            'train': (normal_train, malicious_train),
            'val': (normal_val, malicious_val),
            'test': (normal_test, malicious_test)
        }

    # 加载增量月份（也按 8:1:1 划分）
    inc_start = test_start_month
    inc_end = test_end_month
    for month in generate_month_range(inc_start, inc_end):
        normal_ds, malicious_ds = load_month_dataset(month, vocab, data_paths)
        log.log(f"  {month}: {len(normal_ds)} normal, {len(malicious_ds)} malicious")

        # 8:1:1 划分
        (normal_train, normal_val, normal_test,
         malicious_train, malicious_val, malicious_test) = split_train_val_test(
             normal_ds, malicious_ds, train_ratio, val_ratio)
        all_months_split[month] = {
            'train': (normal_train, malicious_train),
            'val': (normal_val, malicious_val),
            'test': (normal_test, malicious_test)
        }

    # 3. 获取测试月份列表
    test_months = list(generate_month_range(inc_start, inc_end))
    log.log(f"\nTest months: {test_months}")

    # 4. 累积训练循环
    results = {'month': [], 'f1': [], 'acc': [], 'precision': [], 'recall': [],
               'train_months': [], 'n_train': [], 'n_val': [], 'n_test': []}

    # 累积训练数据集：从基础训练月份开始，逐步加入
    cumulative_train_normal = []
    cumulative_train_malicious = []
    cumulative_val_normal = []
    cumulative_val_malicious = []

    # 先加入基础训练月份数据
    for month in generate_month_range(base_start, base_end):
        if month in all_months_split:
            n_train, m_train = all_months_split[month]['train']
            n_val, m_val = all_months_split[month]['val']
            cumulative_train_normal.extend(list(n_train))
            cumulative_train_malicious.extend(list(m_train))
            cumulative_val_normal.extend(list(n_val))
            cumulative_val_malicious.extend(list(m_val))

    log.log(f"\n{'=' * 60}")
    log.log("Starting Accumulate Training (No Memory Pool)")
    log.log(f"Epochs per round: {epochs}, Patience: {patience}")
    log.log(f"{'=' * 60}")

    # ===== 测试 2023-03 月份（使用基础模型） =====
    log.log(f"\n{'=' * 60}")
    log.log("Testing 2023-03 with base model")
    log.log(f"{'=' * 60}")

    # 加载 2023-03 月份数据（全部作为测试集）
    test_month_03 = '2023-03'
    normal_ds_03, malicious_ds_03 = load_month_dataset(test_month_03, vocab, data_paths)
    log.log(f"  {test_month_03}: {len(normal_ds_03)} normal, {len(malicious_ds_03)} malicious")

    # 全部作为测试集
    test_dataset_03 = ConcatDataset([normal_ds_03, malicious_ds_03])
    test_loader_03 = DataLoader(test_dataset_03, batch_size=batch_size, shuffle=False, num_workers=4)
    n_test_03 = len(normal_ds_03) + len(malicious_ds_03)
    log.log(f"Test samples: {n_test_03} (normal={len(normal_ds_03)}, malicious={len(malicious_ds_03)})")

    # 加载基础模型
    device = torch.device(device)
    base_model = GCNWithBehavior(
        name_vocab_size=len(name2idx),
        type_vocab_size=len(type2idx),
        behavior_dim=len(behavior2idx)
    ).to(device)
    base_model_path = "/Data2/hxq/GMLLM/GMLLM/models/base_model.pt"
    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    log.log(f"Loaded base model from {base_model_path}")

    # 测试 2023-03
    log.log(f"Evaluating on {test_month_03}...")
    metrics_03 = validate(base_model, test_loader_03, device)
    f1_03, acc_03, recall_03, precision_03 = metrics_03
    log.log(f"  Results: F1={f1_03:.4f}, Acc={acc_03:.4f}, Precision={precision_03:.4f}, Recall={recall_03:.4f}")

    # 保存 2023-03 测试结果
    results_03 = {
        'month': test_month_03,
        'f1': f1_03,
        'acc': acc_03,
        'precision': precision_03,
        'recall': recall_03,
        'n_test': n_test_03,
        'model': 'base_model.pt'
    }
    log.log(f"2023-03 test results: {results_03}")

    # 遍历每个测试月份
    for i, test_month in enumerate(test_months):
        log.log(f"\n{'=' * 40}")
        log.log(f"Train Month: {test_month} -> Test Month: {test_months[i+1] if i+1 < len(test_months) else 'None'}")
        log.log(f"{'=' * 40}")

        # 先把当月数据加入累积数据集，再用累积数据训练
        n_train_cur, m_train_cur = all_months_split[test_month]['train']
        n_val_cur, m_val_cur = all_months_split[test_month]['val']
        cumulative_train_normal.extend(list(n_train_cur))
        cumulative_train_malicious.extend(list(m_train_cur))
        cumulative_val_normal.extend(list(n_val_cur))
        cumulative_val_malicious.extend(list(m_val_cur))

        # 用累积数据训练，测试下一个月
        train_normal = list(cumulative_train_normal)
        train_malicious = list(cumulative_train_malicious)
        val_normal = list(cumulative_val_normal)
        val_malicious = list(cumulative_val_malicious)

        # 获取下一个月作为测试数据
        if i + 1 < len(test_months):
            next_month = test_months[i + 1]
            n_test, m_test = all_months_split[next_month]['test']
            test_dataset = ConcatDataset([n_test, m_test])
            n_test_samples = len(n_test) + len(m_test)
            log.log(f"Test on {next_month}: {n_test_samples} samples")
        else:
            # 最后一个月份，只测试不训练
            test_dataset = None
            n_test_samples = 0
            log.log("Last month - test only, no training")

        n_train_samples = len(train_normal) + len(train_malicious)
        n_val_samples = len(val_normal) + len(val_malicious)

        log.log(f"Train samples: {n_train_samples} (normal={len(train_normal)}, malicious={len(train_malicious)})")
        log.log(f"Val samples: {n_val_samples} (normal={len(val_normal)}, malicious={len(val_malicious)})")
        if n_test_samples > 0:
            log.log(f"Test samples: {n_test_samples} (normal={len(n_test)}, malicious={len(m_test)})")

        # 构建训练和验证数据加载器
        train_dataset = ConcatDataset([train_normal, train_malicious])
        val_dataset = ConcatDataset([val_normal, val_malicious])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 重新初始化模型（从头开始训练）
        device = torch.device(device)
        model = GCNWithBehavior(
            name_vocab_size=len(name2idx),
            type_vocab_size=len(type2idx),
            behavior_dim=len(behavior2idx)
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        # 训练模型（如果还有下一个月需要测试）
        model_path = f"{model_save_path}{test_month}.pt"
        if test_dataset is not None:
            model, best_val_f1 = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epochs=epochs,
                patience=patience,
                model_save_path=model_path
            )
            log.log(f"Best Val F1: {best_val_f1:.4f}")

            # 测试
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            log.log(f"Evaluating on {next_month}...")
            metrics = validate(model, test_loader, device)
            f1, acc, recall, precision = metrics
            log.log(f"  Results: F1={f1:.4f}, Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

            # 保存结果
            results['month'].append(next_month)
            results['f1'].append(f1)
            results['acc'].append(acc)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['train_months'].append(f"{base_start}-{test_month}")
            results['n_train'].append(n_train_samples)
            results['n_val'].append(n_val_samples)
            results['n_test'].append(n_test_samples)

            log.log(f"Model saved to {model_path}")
        else:
            # 最后一个月份，不需要保存模型
            log.log("Skipping training and testing for last month (no next month to test)")

    # 5. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, result_file)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.log(f"\nResults saved to {result_path}")

    return model, results


if __name__ == "__main__":
    import argparse
    from utils.config_utils import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/default.yaml', help='配置文件路径')
    args = parser.parse_args()

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

    # 训练参数
    epochs = config['training']['epochs']
    patience = config['training']['patience']
    batch_size = config['training']['batch_size']
    train_ratio = config['training']['train_ratio']
    val_ratio = config['training']['val_ratio']
    seed = config['training']['seed']

    # 设备配置
    device_config = config.get('device', 'auto')
    if device_config == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config

    # 路径配置
    paths_config = config.get('paths', {})

    results_config = config.get('results', {})
    result_file = results_config.get('accumulate_train', 'accumulate_train_results.json')

    models_dir = paths_config.get('models_dir', './models')
    model_prefix = paths_config.get('prefix', 'upper_model_')
    model_save_path = f"{models_dir}/{model_prefix}"

    test_start = incremental_months[0]
    test_end = incremental_months[1]

    run_accumulate_train(
        vocab_dir=vocab_dir,
        data_paths=data_paths,
        base_train_months=base_train_months,
        test_start_month=test_start,
        test_end_month=test_end,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        device=device,
        seed=seed,
        result_file=result_file,
        model_save_path=model_save_path
    )