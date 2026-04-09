import sys
from pathlib import Path

# 将上级目录加入 Python 搜索路径
sys.path.insert(0, '/Data2/hxq/GMLLM')

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
import yaml
from utils.config_utils import load_config
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import random
import numpy as np
import time
from utils.month_utils import generate_month_range
from utils.data_utils import split_train_val_test
from utils.data_loader import load_vocabs, build_dataloaders, load_month_dataset
from utils.logger_utils import Logger

# 创建日志记录器
log = Logger("train_base_model.log")

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


# =============================================================================
# 辅助函数
# =============================================================================

def train_model(model, train_loader, val_loader, optimizer, criterion, device,
                epochs: int, patience: int, model_save_path: str) -> tuple:
    """训练模型"""
    best_val_f1 = 0
    no_improve_count = 0

    train_start = time.time()
    log.log(f"\nTraining started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, device)
        val_f1, acc, malicious_recall, val_precision = val_metrics

        log.log(
            f"Epoch {epoch:03d} | Loss {train_loss:.4f} | TrainAcc {train_acc:.4f} | "
            f"ValF1 {val_f1:.4f} | ValAcc {acc:.4f} | MalRecall {malicious_recall:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            log.log(f"\nEarly stopping at epoch {epoch}! No improvement for {patience} epochs.")
            break

    train_time = time.time() - train_start
    log.log(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.log(f"Training done in {train_time:.2f}s ({train_time/60:.2f} min). Best Val F1: {best_val_f1:.4f}")

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model, best_val_f1


def test_model(model, test_datasets: dict, batch_size: int, device) -> dict:
    """按月测试模型"""
    test_loaders = build_dataloaders(test_datasets, batch_size, shuffle=False)

    log.log(f"Test months: {len(test_loaders)} ({list(test_loaders.keys())[0]} ~ {list(test_loaders.keys())[-1]})")

    future_test_results = {'month': [], 'f1': [], 'acc': [], 'precision': [], 'recall': []}

    log.log("\n=== Test Results ===")
    for month in sorted(test_loaders.keys()):
        if month > '2023-02':
            test_loader = test_loaders[month]
            metrics = validate(model, test_loader, device)
            f1, acc, malicious_recall, malicious_precision = metrics

            future_test_results['month'].append(month)
            future_test_results['f1'].append(f1)
            future_test_results['acc'].append(acc)
            future_test_results['precision'].append(malicious_precision)
            future_test_results['recall'].append(malicious_recall)

            log.log(f"{month} | F1: {f1:.4f} | Acc: {acc:.4f} | "
                    f"Precision: {malicious_precision:.4f} | Recall: {malicious_recall:.4f}")

    if len(future_test_results['month']) > 0:
        avg_f1 = sum(future_test_results['f1']) / len(future_test_results['f1'])
        avg_acc = sum(future_test_results['acc']) / len(future_test_results['acc'])
        avg_prec = sum(future_test_results['precision']) / len(future_test_results['precision'])
        avg_recall = sum(future_test_results['recall']) / len(future_test_results['recall'])
        log.log(f"\n=== Future Test Results Summary ===")
        log.log(f"Average | F1: {avg_f1:.4f} | Acc: {avg_acc:.4f} | "
                f"Precision: {avg_prec:.4f} | Recall: {avg_recall:.4f}")

    return future_test_results


def run_base_model(
    vocab_dir: str,
    data_paths: dict,
    train_months: tuple = ('2022-01', '2023-02'),
    incremental_months: tuple = ('2023-03', '2024-12'),
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 128,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    device: str = "cuda",
    seed: int = 42,
    model_save_path: str = "./models/base_model.pt",
    result_file: str = "test_than_train_future_test_results.json"
):
    """运行基础模型训练和测试流程"""
    set_seed(seed)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 加载词汇表
    name2idx, type2idx, behavior2idx, edge_type2idx = load_vocabs(vocab_dir)
    vocab = {
        'name2idx': name2idx,
        'type2idx': type2idx,
        'behavior2idx': behavior2idx,
        'edge_type2idx': edge_type2idx
    }

    # ========== 构建数据集 ==========
    train_datasets = {}
    val_datasets = {}
    test_datasets = {}

    train_start, train_end = train_months

    for month in generate_month_range(train_start, train_end):
        normal_ds, malicious_ds = load_month_dataset(month, vocab, data_paths)

        (normal_train, normal_val, normal_test,
            malicious_train, malicious_val, malicious_test) = split_train_val_test(
                normal_ds, malicious_ds, train_ratio, val_ratio
        )
        train_datasets[month] = (normal_train, malicious_train)
        val_datasets[month] = (normal_val, malicious_val)
        test_datasets[month] = (normal_test, malicious_test)
    
    # 加载增量月份数据
    inc_start, inc_end = incremental_months
    for month in generate_month_range(inc_start, inc_end):
        normal_ds, malicious_ds = load_month_dataset(month, vocab, data_paths)
        test_datasets[month] = (normal_ds, malicious_ds)

    # 构建累积的训练集、验证集
    train_dataset = ConcatDataset([
        ConcatDataset([normal_train, malicious_train])
        for normal_train, malicious_train in train_datasets.values()
    ])
    val_dataset = ConcatDataset([
        ConcatDataset([normal_val, malicious_val])
        for normal_val, malicious_val in val_datasets.values()
    ])

    assert len(train_dataset) > 0, "Empty train set."
    assert len(val_dataset) > 0, "Empty val set."

    log.log(f"Train samples: {len(train_dataset)}")
    log.log(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型
    device = torch.device(device)
    model = GCNWithBehavior(
        name_vocab_size=len(name2idx),
        type_vocab_size=len(type2idx),
        behavior_dim=len(behavior2idx)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练模型
    model, best_val_f1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
        patience=patience,
        model_save_path=model_save_path
    )

    # 加载最优模型
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    # 测试模型
    test_results = test_model(
        model=model,
        test_datasets=test_datasets,
        batch_size=batch_size,
        device=device
    )

    # 保存结果
    results_dir = Path("../results")
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir / result_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    log.log(f"Results saved to {results_dir}/")

    return model, test_results


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    # 加载配置文件
    config_path = "./configs/default.yaml"
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

    # 训练参数
    epochs = config['training']['epochs']
    patience = config['training']['patience']
    batch_size = config['training']['batch_size']
    seed = config['training']['seed']
    train_ratio = config['training']['train_ratio']
    val_ratio = config['training']['val_ratio']

    # 设备配置
    device_config = config.get('device', 'auto')
    device = "cuda" if torch.cuda.is_available() else "cpu" if device_config == 'auto' else device_config

    # 路径配置
    models_dir = config['paths']['models_dir']

    # 增量学习配置
    cl_config = config.get('continual_learning', {})
    base_train_months = tuple(cl_config.get('base_train_months', ['2022-01', '2023-02']))
    incremental_months = tuple(cl_config.get('incremental_months', ['2023-03', '2024-12']))

    # 结果文件配置
    results_config = config.get('results', {})
    base_model_result_file = results_config.get('base_model', 'test_than_train_future_test_results.json')

    # 运行训练和测试
    run_base_model(
        vocab_dir=vocab_dir,
        data_paths=data_paths,
        train_months=base_train_months,
        incremental_months=incremental_months,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        device=device,
        seed=seed,
        model_save_path=f"{models_dir}/base_model.pt",
        result_file=base_model_result_file
    )