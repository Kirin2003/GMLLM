#!/usr/bin/env python3
"""
Parallel monthly retraining for distinguish_GNN_2.py.

Simulates monthly retraining: for each month 2024-01 ~ 2024-11,
trains on 2022-01~current_month and tests on the next month.
All months run in parallel across available GPUs.

Usage:
    python parallel_retrain.py --config ./configs/default_with_deepseek.yaml
    python parallel_retrain.py --config ./configs/default_with_deepseek.yaml --gpus 0,1,2,3
    python parallel_retrain.py --config ./configs/default_with_deepseek.yaml --months 2024-01,2024-03  # quick test
"""

import sys
import os
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
import numpy as np

from utils.config_utils import load_config
from utils.month_utils import generate_month_range
from utils.data_utils import split_train_val_test
from utils.data_loader import load_vocabs, load_month_dataset
from utils.paths import LOG_DIR

# Imports from distinguish_GNN_2 happen inside worker to avoid CUDA fork issues


def worker_loop(tasks_q, results_q):
    """Pull tasks from queue until empty, store results."""
    while True:
        try:
            args = tasks_q.get_nowait()
        except:
            break
        res = retrain_worker(*args)
        results_q.put(res)


def retrain_worker(current_month, test_month, config_path, gpu_id, models_root,
                   num_workers_per_loader):
    """
    Train on 2022-01~current_month, test on test_month.

    Args:
        current_month: training up to this month (inclusive), e.g. '2024-01'
        test_month: month to test on, e.g. '2024-02'
        config_path: path to YAML config file
        gpu_id: CUDA device ID to use (-1 for CPU)
        models_root: root directory for model checkpoint saves
        num_workers_per_loader: DataLoader num_workers

    Returns:
        dict with train_month, test_month, training_time, f1, acc, precision, recall
    """
    # Redirect the distinguish_GNN_2 logger to a per-worker file
    # Each spawn'd process gets its own module copy, so no cross-process conflict
    from distinguish_GNN_2 import log, set_seed, GCNWithBehavior, train_model, test_model
    log.log_file = LOG_DIR / f"train_retrain_{current_month}.log"

    device = torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu')

    # ---------- load config ----------
    config = load_config(config_path)

    base_path = config['dataset']['base_path']
    vocab_dir = str(Path(base_path) / config['dataset']['vocab_dir'])
    data_paths = {
        'benign_root': str(Path(base_path) / config['dataset']['benign_root']),
        'malicious_root': str(Path(base_path) / config['dataset']['malicious_root']),
        'benign_out': str(Path(base_path) / config['dataset']['benign_out']),
        'malicious_out': str(Path(base_path) / config['dataset']['malicious_out']),
    }

    epochs = config['training']['epochs']
    patience = config['training']['patience']
    batch_size = config['training']['batch_size']
    seed = config['training']['seed']
    train_ratio = config['training']['train_ratio']
    val_ratio = config['training']['val_ratio']

    set_seed(seed)

    # ---------- load vocab ----------
    name2idx, type2idx, behavior2idx, edge_type2idx = load_vocabs(vocab_dir)
    vocab = {
        'name2idx': name2idx,
        'type2idx': type2idx,
        'behavior2idx': behavior2idx,
        'edge_type2idx': edge_type2idx,
    }

    # ---------- load training data: 2022-01 ~ current_month ----------
    train_datasets = {}
    val_datasets = {}
    log.log(f"Loading training data from 2022-01 to {current_month}...")
    for month in generate_month_range('2022-01', current_month):
        normal_ds, malicious_ds = load_month_dataset(month, vocab, data_paths)

        (normal_train, normal_val, normal_test,
         malicious_train, malicious_val, malicious_test) = split_train_val_test(
             normal_ds, malicious_ds, train_ratio, val_ratio
        )
        train_datasets[month] = (normal_train, malicious_train)
        val_datasets[month] = (normal_val, malicious_val)

    # ---------- load test data ----------
    log.log(f"Loading test data from {test_month}...")
    normal_ds, malicious_ds = load_month_dataset(test_month, vocab, data_paths)
    test_datasets = {test_month: (normal_ds, malicious_ds)}

    # ---------- build dataloaders ----------
    train_dataset = ConcatDataset([
        ConcatDataset([n, m]) for n, m in train_datasets.values()
    ])
    val_dataset = ConcatDataset([
        ConcatDataset([n, m]) for n, m in val_datasets.values()
    ])

    log.log(f"Train samples: {len(train_dataset)}")
    log.log(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers_per_loader
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers_per_loader
    )

    # ---------- init model ----------
    model = GCNWithBehavior(
        name_vocab_size=len(name2idx),
        type_vocab_size=len(type2idx),
        behavior_dim=len(behavior2idx)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    # ---------- train ----------
    model_save_dir = Path(models_root) / current_month
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = str(model_save_dir / 'base_model.pt')

    train_start = time.time()
    model, best_val_f1 = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, criterion=criterion, device=device,
        epochs=epochs, patience=patience, model_save_path=model_save_path
    )
    train_time = time.time() - train_start

    # ---------- test ----------
    test_results = test_model(
        model=model, test_datasets=test_datasets,
        batch_size=batch_size, device=device,
        test_start_month=test_month
    )

    log.log(f"[{current_month}] Done. Train time: {train_time:.2f}s | "
            f"F1: {test_results['f1'][0]:.4f} | Acc: {test_results['acc'][0]:.4f}")

    return {
        'train_month': current_month,
        'test_month': test_month,
        'training_time': round(train_time, 2),
        'f1': test_results['f1'][0],
        'acc': test_results['acc'][0],
        'precision': test_results['precision'][0],
        'recall': test_results['recall'][0],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Parallel monthly retraining for distinguish_GNN_2.py'
    )
    parser.add_argument('--config', type=str, default='./configs/default_with_deepseek.yaml',
                        help='Path to config YAML')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs, e.g. "0,1,2,3". Auto-detect if not set.')
    parser.add_argument('--months', type=str, default='2024-01,2024-12',
                        help='Month range for retraining, e.g. "2024-01,2024-12"')
    parser.add_argument('--models-root', type=str, default=None,
                        help='Root for model checkpoints (default: ./models_retrain)')
    parser.add_argument('--pool-size', type=int, default=None,
                        help='Max concurrent workers. Default = number of GPUs. '
                             'Set higher than GPU count to multiplex multiple workers '
                             'per GPU (watch for OOM).')
    args = parser.parse_args()

    # ---------- detect GPUs ----------
    if args.gpus:
        gpu_list = [int(x) for x in args.gpus.split(',')]
    else:
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print("No CUDA devices found. Falling back to CPU.")
            gpu_list = [-1]  # one CPU worker
        else:
            gpu_list = list(range(gpu_count))
    print(f"Using GPUs: {gpu_list}")

    # ---------- build task list ----------
    start_month, end_month = args.months.split(',')
    months = generate_month_range(start_month, end_month)
    if len(months) < 2:
        raise ValueError("Need at least 2 months (one train, one test).")

    tasks = []
    for i in range(len(months) - 1):
        gpu_id = gpu_list[i % len(gpu_list)]
        tasks.append((months[i], months[i + 1], args.config, gpu_id))

    models_root = args.models_root or './models_retrain'

    # Pool size: user-specified or default to GPU count
    pool_size = args.pool_size if args.pool_size is not None else len(gpu_list)
    pool_size = min(pool_size, len(tasks))
    if pool_size > len(gpu_list):
        print(f"  ⚠  pool_size={pool_size} > GPU count={len(gpu_list)}, "
              f"workers will share GPUs — reduce batch_size if OOM")

    # Reduce DataLoader worker count per task when running multiple jobs
    num_workers_per_loader = max(1, min(2, mp.cpu_count() // (max(pool_size, 1) * 2)))

    print(f"\n=== Parallel Monthly Retraining ===")
    print(f"Config:    {args.config}")
    print(f"GPU pool:  {gpu_list}")
    print(f"Tasks:     {len(tasks)}")
    print(f"DataLoader workers: {num_workers_per_loader} per task")
    print(f"Checkpoints root:   {models_root}")
    print()
    print(f"Pool size: {pool_size} (max concurrent workers)")
    print()

    for cm, tm, _, gid in tasks:
        print(f"  Train {cm} ~ {cm}  |  Test {tm}  |  GPU {gid}")
    print()

    # ---------- parallel execution ----------
    # Use Process (not Pool) so workers are non-daemon and can spawn
    # DataLoader child processes for data loading.
    mp.set_start_method('spawn', force=True)

    task_queue = mp.Queue()
    for cm, tm, cfg, gid in tasks:
        task_queue.put((cm, tm, cfg, gid, models_root, num_workers_per_loader))

    result_queue = mp.Queue()

    processes = []
    for _ in range(pool_size):
        p = mp.Process(target=worker_loop, args=(task_queue, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Collect all results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    results.sort(key=lambda r: r['train_month'])

    output = {
        'month': [],
        'f1': [],
        'acc': [],
        'precision': [],
        'recall': [],
        'training_time': [],
    }
    for r in results:
        output['month'].append(r['test_month'])
        output['f1'].append(r['f1'])
        output['acc'].append(r['acc'])
        output['precision'].append(r['precision'])
        output['recall'].append(r['recall'])
        output['training_time'].append(r['training_time'])

    output['avg_f1'] = float(np.mean(output['f1']))
    output['avg_acc'] = float(np.mean(output['acc']))
    output['avg_precision'] = float(np.mean(output['precision']))
    output['avg_recall'] = float(np.mean(output['recall']))
    output['avg_training_time'] = float(np.mean(output['training_time']))
    output['total_training_time'] = float(np.sum(output['training_time']))
    output['config'] = str(args.config)

    # ---------- save ----------
    config_stem = Path(args.config).stem
    output_path = Path('../results') / f'retrain_results_{config_stem}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"All {len(results)} retrain tasks completed!")
    print(f"Average F1:        {output['avg_f1']:.4f}")
    print(f"Average Accuracy:  {output['avg_acc']:.4f}")
    print(f"Average Precision: {output['avg_precision']:.4f}")
    print(f"Average Recall:    {output['avg_recall']:.4f}")
    print(f"Total train time:  {output['total_training_time']:.1f}s")
    print(f"Results saved to:  {output_path}")


if __name__ == '__main__':
    main()
