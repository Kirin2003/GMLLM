import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from generate_graph_data_fromJson import CallGraphDatasetFull_Lazy
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import random
import numpy as np
import time
from month_utils import generate_month_range
from plot_results import plot_monthly_metrics
from data_util import split_train_val_test

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class GCNWithBehavior(nn.Module):
    def __init__(self, name_vocab_size, type_vocab_size, behavior_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        self.name_emb = nn.Embedding(name_vocab_size, 64)
        self.type_emb = nn.Embedding(type_vocab_size, 16)
        input_dim = 64 + 16 + behavior_dim  # name + type + behaviors
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, data):
        name_feat = self.name_emb(data.x_names)
        type_feat = self.type_emb(data.x_types)
        behavior_feat = data.x_behaviors.float()
        x = torch.cat([name_feat, type_feat, behavior_feat], dim=1)
        x = self.dropout(self.conv1(x, data.edge_index).relu())
        x = self.dropout(self.conv2(x, data.edge_index).relu())

        if x.shape[0] != data.batch.shape[0]:
            min_len = min(x.shape[0], data.batch.shape[0])
            x = x[:min_len]
            batch = data.batch[:min_len]
        else:
            batch = data.batch

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.classifier(x)

def load_dict(path):
    with open(path, 'r') as f:
        return json.load(f)

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data in loader:
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

    return total_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1], zero_division=0)
    malicious_f1 = f1[1]
    malicious_precision = p[1]
    malicious_recall = r[1]
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    benign_acc = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"[Validation] Overall Acc: {acc:.4f} | Malicious F1: {malicious_f1:.4f}")
    print(
        f" └─ Malicious Metrics: Precision: {malicious_precision:.4f} | Recall: {malicious_recall:.4f} (TP:{tp}, FN:{fn})")
    print(f" └─ Benign Accuracy..: {benign_acc:.4f} (TN:{tn}, FP:{fp})")
    return malicious_f1, acc, malicious_recall, malicious_precision


if __name__ == "__main__":

    # =============================================================================
    # 阶段1: 配置和初始化
    # =============================================================================
    vocab_dir = "/Data2/hxq/datasets/incremental_packages_subset/vocab"
    benign_root = "/Data2/hxq/datasets/incremental_packages_subset/benign"
    malicious_root = "/Data2/hxq/datasets/incremental_packages_subset/malicious"
    benign_out = "/Data2/hxq/datasets/incremental_packages_subset/benign_call_processed"
    malicious_out = "/Data2/hxq/datasets/incremental_packages_subset/malicious_call_processed"
    epochs = 60
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    SEED = 42
    set_seed(SEED)

    os.makedirs("models", exist_ok=True)

    # =============================================================================
    # 阶段2: 加载词汇表
    # =============================================================================
    name2idx = load_dict(str(Path(vocab_dir) / "name2idx.json"))
    type2idx = load_dict(str(Path(vocab_dir) / "type2idx.json"))
    behavior2idx = load_dict(str(Path(vocab_dir) / "behavior2idx.json"))
    edge_type2idx = load_dict(str(Path(vocab_dir) / "edge_type2idx.json"))
    print('Load vocab done.')

    # =============================================================================
    # 阶段3: 构建数据集 (按月份循环)
    # =============================================================================
    # 2022-01 ~ 2023-02: 80% 训练, 10% 验证, 10% 测试
    # 2023-03 ~ 2024-12: 100% 按月测试

    train_dataset = None      # 累积训练集
    val_dataset = None        # 验证集
    test_datasets = {}        # 按月份存储测试集: {month: (test_normal, test_malicious)}

    train_ratio = 0.8
    val_ratio = 0.1

    for month in generate_month_range('2022-01', '2024-12'):
        # 加载当月数据集
        normal_dataset = CallGraphDatasetFull_Lazy(
            root_dir=benign_root, output_dir=benign_out,
            name2idx=name2idx, type2idx=type2idx,
            behavior2idx=behavior2idx, edge_type2idx=edge_type2idx,
            fixed_label=0, start_month=month, end_month=month
        )
        malicious_dataset = CallGraphDatasetFull_Lazy(
            root_dir=malicious_root, output_dir=malicious_out,
            name2idx=name2idx, type2idx=type2idx,
            behavior2idx=behavior2idx, edge_type2idx=edge_type2idx,
            fixed_label=1, start_month=month, end_month=month
        )

        if month <= '2023-02':
            # 训练阶段: 80% 训练, 10% 验证, 10% 测试
            (normal_train, normal_val, normal_test,
             malicious_train, malicious_val, malicious_test) = split_train_val_test(
                 normal_dataset, malicious_dataset, train_ratio, val_ratio
             )

            # 累积训练集和验证集
            train_dataset = ConcatDataset([train_dataset, normal_train, malicious_train]) if train_dataset else ConcatDataset([normal_train, malicious_train])
            val_dataset = ConcatDataset([val_dataset, normal_val, malicious_val]) if val_dataset else ConcatDataset([normal_val, malicious_val])

            # 测试集按月存储
            test_datasets[month] = (normal_test, malicious_test)

        else:
            # 测试阶段: 100% 用于测试
            test_datasets[month] = (normal_dataset, malicious_dataset)

    assert train_dataset and len(train_dataset) > 0, "Empty train set."
    assert val_dataset and len(val_dataset) > 0, "Empty val set."

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 按月份生成测试 loader
    test_loaders = {}
    for month, (normal_test, malicious_test) in test_datasets.items():
        if len(normal_test) > 0 and len(malicious_test) > 0:
            test_loaders[month] = DataLoader(
                ConcatDataset([normal_test, malicious_test]),
                batch_size=batch_size, shuffle=False, num_workers=4
            )
        else:
            print(f"[WARNING] {month}: empty test set, skipping.")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test months: {len(test_loaders)} ({list(test_loaders.keys())[0]} ~ {list(test_loaders.keys())[-1]})")

    # =============================================================================
    # 阶段4: 初始化模型、优化器、损失函数
    # =============================================================================
    device = torch.device(device)
    model = GCNWithBehavior(
        name_vocab_size=len(name2idx),
        type_vocab_size=len(type2idx),
        behavior_dim=len(behavior2idx)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0
    history = {'train_loss': [], 'train_acc': [], 'val_f1': [], 'val_acc': [], 'val_precision': [], 'val_recall': []}

    # =============================================================================
    # 阶段5: 训练模型
    # =============================================================================
    train_start = time.time()
    print(f"\nTraining started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, device)
        val_f1, acc, malicious_recall, val_precision = val_metrics

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(malicious_recall)

        print(
            f"Epoch {epoch:03d} | Loss {train_loss:.4f} | TrainAcc {train_acc:.4f} | ValF1 {val_f1:.4f} | ValAcc {acc:.4f} | MalRecall {malicious_recall:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "models/base_model.pt")

    train_time = time.time() - train_start
    print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training done in {train_time:.2f}s ({train_time/60:.2f} min). Best Val F1: {best_val_f1:.4f}")

    # =============================================================================
    # 阶段6: 按月测试
    # =============================================================================
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load("models/base_model.pt", map_location=device))

    val_period_results = {'month': [], 'f1': [], 'acc': [], 'precision': [], 'recall': []}
    future_test_results = {'month': [], 'f1': [], 'acc': [], 'precision': [], 'recall': []}

    print("\n=== Validation Period Test Results (2022-01 ~ 2023-02) ===")
    for month in sorted(test_loaders.keys()):
        test_loader = test_loaders[month]
        metrics = validate(model, test_loader, device)
        f1, acc, malicious_recall, malicious_precision = metrics

        result_entry = {
            'month': month, 'f1': f1, 'acc': acc,
            'precision': malicious_precision, 'recall': malicious_recall
        }

        if month <= '2023-02':
            # 验证期的测试结果
            val_period_results['month'].append(month)
            val_period_results['f1'].append(f1)
            val_period_results['acc'].append(acc)
            val_period_results['precision'].append(malicious_precision)
            val_period_results['recall'].append(malicious_recall)
        else:
            # 未来测试结果
            future_test_results['month'].append(month)
            future_test_results['f1'].append(f1)
            future_test_results['acc'].append(acc)
            future_test_results['precision'].append(malicious_precision)
            future_test_results['recall'].append(malicious_recall)

        print(f"{month} | F1: {f1:.4f} | Acc: {acc:.4f} | Precision: {malicious_precision:.4f} | Recall: {malicious_recall:.4f}")

    # 未来测试结果汇总
    print("\n=== Future Test Results Summary (2023-03 ~ 2024-12) ===")
    if len(future_test_results['month']) > 0:
        avg_f1 = sum(future_test_results['f1']) / len(future_test_results['f1'])
        avg_acc = sum(future_test_results['acc']) / len(future_test_results['acc'])
        avg_prec = sum(future_test_results['precision']) / len(future_test_results['precision'])
        avg_recall = sum(future_test_results['recall']) / len(future_test_results['recall'])
        print(f"Average | F1: {avg_f1:.4f} | Acc: {avg_acc:.4f} | Precision: {avg_prec:.4f} | Recall: {avg_recall:.4f}")

    # =============================================================================
    # 阶段7: 保存测试结果
    # =============================================================================
    results_dir = Path(malicious_out).parent
    with open(results_dir / "val_period_test_results.json", 'w') as f:
        json.dump(val_period_results, f, indent=2)
    with open(results_dir / "future_test_results.json", 'w') as f:
        json.dump(future_test_results, f, indent=2)

    # 合并所有按月结果（方便画整体折线图）
    all_monthly_results = {
        'val_period': val_period_results,
        'future_test': future_test_results
    }
    with open(results_dir / "monthly_test_results.json", 'w') as f:
        json.dump(all_monthly_results, f, indent=2)

    print(f"Training done in {train_time:.2f}s ({train_time/60:.2f} min). Best Val F1: {best_val_f1:.4f}")
    print(f"\nResults saved to {results_dir}/")
    print("  - val_period_test_results.json")
    print("  - future_test_results.json")
    print("  - monthly_test_results.json")

    # =============================================================================
    # 阶段8: 画图
    # =============================================================================
    plot_monthly_metrics(val_period_results, future_test_results, results_dir)
