"""
每月全量重训基线
- 测试：每个月先用上一轮模型在当前月完整数据上测试，保证当前月测试数据此前没有参与训练
- 更新：测试后，从头初始化模型，并用基础训练月份到当前月的累计训练/验证划分重新训练一次
- 用途：和增量学习的 monthly update time / peak GPU memory 做效率对比
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import time
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from utils.month_utils import generate_month_range
from utils.data_utils import split_train_val_test
from utils.data_loader import load_vocabs, load_month_dataset

from distinguish_GNN_2 import GCNWithBehavior, validate, set_seed, train_model
from utils.logger_utils import Logger

log = Logger("accumulate_train.log")


def is_cuda_device(device) -> bool:
    return isinstance(device, torch.device) and device.type == "cuda"


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 2)


def start_efficiency_measurement(device) -> float:
    if is_cuda_device(device):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def finish_efficiency_measurement(device, start_time: float) -> dict:
    if is_cuda_device(device):
        torch.cuda.synchronize(device)
        peak_allocated = bytes_to_mib(torch.cuda.max_memory_allocated(device))
        peak_reserved = bytes_to_mib(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = None
        peak_reserved = None

    return {
        "update_time": time.perf_counter() - start_time,
        "peak_gpu_memory_allocated_mb": peak_allocated,
        "peak_gpu_memory_reserved_mb": peak_reserved,
    }


def init_model(name2idx, type2idx, behavior2idx, device):
    return GCNWithBehavior(
        name_vocab_size=len(name2idx),
        type_vocab_size=len(type2idx),
        behavior_dim=len(behavior2idx)
    ).to(device)


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
    num_workers: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    pretrained_model_path: str = "./models/default/base_model.pt",
    result_file: str = "accumulate_train_results.json",
    model_save_path: str = "./models/accumulate_train_model_",
    results_dir: str = "../results"
):
    """
    每月全量更新：
    1. 用当前模型先测试当前月完整数据。
    2. 测试后，将当前月训练/验证划分加入累计数据。
    3. 从头初始化模型，用 2022-01 到当前月的累计数据重新训练。

    当前月完整数据只在步骤 1 之后才进入后续训练流程，因此不会出现先训练再测试当前月的泄漏。
    """
    output_dir = results_dir
    set_seed(seed)

    log.log("=" * 60)
    log.log("Running Monthly Full Retraining Baseline")
    log.log("=" * 60)

    name2idx, type2idx, behavior2idx, edge_type2idx = load_vocabs(vocab_dir)
    vocab = {
        'name2idx': name2idx,
        'type2idx': type2idx,
        'behavior2idx': behavior2idx,
        'edge_type2idx': edge_type2idx,
    }
    log.log(f"Vocabulary size: name={len(name2idx)}, type={len(type2idx)}, behavior={len(behavior2idx)}")

    device = torch.device(device)
    base_start, base_end = base_train_months
    test_months = list(generate_month_range(test_start_month, test_end_month))

    all_months_split = {}
    full_test_datasets = {}
    all_months = list(generate_month_range(base_start, base_end)) + test_months
    for month in all_months:
        normal_ds, malicious_ds = load_month_dataset(month, vocab, data_paths)
        log.log(f"  {month}: {len(normal_ds)} normal, {len(malicious_ds)} malicious")
        full_test_datasets[month] = (normal_ds, malicious_ds)

        (normal_train, normal_val, normal_test,
         malicious_train, malicious_val, malicious_test) = split_train_val_test(
             normal_ds, malicious_ds, train_ratio, val_ratio)
        all_months_split[month] = {
            'train': (normal_train, malicious_train),
            'val': (normal_val, malicious_val),
            'test': (normal_test, malicious_test),
        }

    results = {
        'month': [],
        'f1': [],
        'acc': [],
        'precision': [],
        'recall': [],
        'model_train_months_before_test': [],
        'trained_through_after_update': [],
        'n_train': [],
        'n_val': [],
        'n_test': [],
        'full_retrain_time': [],
        'peak_gpu_memory_allocated_mb': [],
        'peak_gpu_memory_reserved_mb': [],
        'seed': seed,
        'time_unit': 'seconds',
        'memory_unit': 'MiB',
    }

    cumulative_train_normal = []
    cumulative_train_malicious = []
    cumulative_val_normal = []
    cumulative_val_malicious = []

    for month in generate_month_range(base_start, base_end):
        n_train, m_train = all_months_split[month]['train']
        n_val, m_val = all_months_split[month]['val']
        cumulative_train_normal.extend(list(n_train))
        cumulative_train_malicious.extend(list(m_train))
        cumulative_val_normal.extend(list(n_val))
        cumulative_val_malicious.extend(list(m_val))

    current_model = init_model(name2idx, type2idx, behavior2idx, device)
    current_model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    current_model_train_end = base_end
    log.log(f"Loaded base model from {pretrained_model_path}")

    os.makedirs(os.path.dirname(model_save_path) or "./models", exist_ok=True)

    for month in test_months:
        log.log("\n" + "=" * 40)
        log.log(f"Month {month}: test first, then full retrain through {month}")
        log.log(f"{'=' * 40}")

        normal_test_full, malicious_test_full = full_test_datasets[month]
        test_dataset = ConcatDataset([normal_test_full, malicious_test_full])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        n_test_samples = len(test_dataset)

        log.log(
            f"Evaluating {month} before using this month for training "
            f"({n_test_samples} samples, model trained through {current_model_train_end})..."
        )
        f1, acc, recall, precision = validate(current_model, test_loader, device)
        log.log(f"  Results: F1={f1:.4f}, Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

        n_train_cur, m_train_cur = all_months_split[month]['train']
        n_val_cur, m_val_cur = all_months_split[month]['val']
        cumulative_train_normal.extend(list(n_train_cur))
        cumulative_train_malicious.extend(list(m_train_cur))
        cumulative_val_normal.extend(list(n_val_cur))
        cumulative_val_malicious.extend(list(m_val_cur))

        train_dataset = ConcatDataset([list(cumulative_train_normal), list(cumulative_train_malicious)])
        val_dataset = ConcatDataset([list(cumulative_val_normal), list(cumulative_val_malicious)])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        n_train_samples = len(train_dataset)
        n_val_samples = len(val_dataset)
        log.log(f"Full retrain samples through {month}: train={n_train_samples}, val={n_val_samples}")

        model = init_model(name2idx, type2idx, behavior2idx, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        model_path = f"{model_save_path}{month}.pt"

        update_start = start_efficiency_measurement(device)
        model, best_val_f1 = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=epochs,
            patience=patience,
            model_save_path=model_path,
        )
        efficiency = finish_efficiency_measurement(device, update_start)
        update_time = efficiency['update_time']
        peak_allocated = efficiency['peak_gpu_memory_allocated_mb']
        peak_reserved = efficiency['peak_gpu_memory_reserved_mb']

        log.log(f"Best Val F1: {best_val_f1:.4f}")
        log.log(f"Full retrain time: {update_time:.2f}s")
        if peak_allocated is not None:
            log.log(
                f"Peak GPU memory: allocated={peak_allocated:.2f} MiB, "
                f"reserved={peak_reserved:.2f} MiB"
            )
        else:
            log.log("Peak GPU memory: N/A (CPU device)")
        log.log(f"Model saved to {model_path}")

        results['month'].append(month)
        results['f1'].append(f1)
        results['acc'].append(acc)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['model_train_months_before_test'].append(f"{base_start}-{current_model_train_end}")
        results['trained_through_after_update'].append(f"{base_start}-{month}")
        results['n_train'].append(n_train_samples)
        results['n_val'].append(n_val_samples)
        results['n_test'].append(n_test_samples)
        results['full_retrain_time'].append(update_time)
        results['peak_gpu_memory_allocated_mb'].append(peak_allocated)
        results['peak_gpu_memory_reserved_mb'].append(peak_reserved)

        current_model = model
        current_model_train_end = month

    if results['f1']:
        results['avg_f1'] = sum(results['f1']) / len(results['f1'])
        results['avg_acc'] = sum(results['acc']) / len(results['acc'])
        results['avg_precision'] = sum(results['precision']) / len(results['precision'])
        results['avg_recall'] = sum(results['recall']) / len(results['recall'])
        results['avg_full_retrain_time'] = sum(results['full_retrain_time']) / len(results['full_retrain_time'])
        allocated_values = [value for value in results['peak_gpu_memory_allocated_mb'] if value is not None]
        reserved_values = [value for value in results['peak_gpu_memory_reserved_mb'] if value is not None]
        if allocated_values:
            results['max_peak_gpu_memory_allocated_mb'] = max(allocated_values)
        if reserved_values:
            results['max_peak_gpu_memory_reserved_mb'] = max(reserved_values)

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, result_file)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.log(f"\nResults saved to {result_path}")

    return current_model, results


if __name__ == "__main__":
    import argparse
    from utils.config_utils import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/default.yaml', help='配置文件路径')
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    base_path = config['dataset']['base_path']
    vocab_dir = str(Path(base_path) / config['dataset']['vocab_dir'])
    data_paths = {
        'benign_root': str(Path(base_path) / config['dataset']['benign_root']),
        'malicious_root': str(Path(base_path) / config['dataset']['malicious_root']),
        'benign_out': str(Path(base_path) / config['dataset']['benign_out']),
        'malicious_out': str(Path(base_path) / config['dataset']['malicious_out']),
    }

    cl_config = config.get('continual_learning', {})
    base_train_months = tuple(cl_config.get('base_train_months', ['2022-01', '2023-02']))
    incremental_months = tuple(cl_config.get('incremental_months', ['2023-03', '2024-12']))

    epochs = config['training']['epochs']
    patience = config['training']['patience']
    batch_size = config['training']['batch_size']
    train_ratio = config['training']['train_ratio']
    val_ratio = config['training']['val_ratio']
    num_workers = config['training'].get('num_workers', 4)
    seed = config['training']['seed']

    device_config = config.get('device', 'auto')
    if device_config == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config

    paths_config = config.get('paths', {})
    results_config = config.get('results', {})
    result_file = results_config.get('accumulate_train', 'accumulate_train_results.json')

    models_dir = paths_config.get('models_dir', './models')
    results_dir = paths_config.get('results_dir', '../results')
    pretrained_model_path = paths_config.get('pretrained_model', './models/default/base_model.pt')
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
        num_workers=num_workers,
        device=device,
        seed=seed,
        pretrained_model_path=pretrained_model_path,
        result_file=result_file,
        model_save_path=model_save_path,
        results_dir=results_dir,
    )
