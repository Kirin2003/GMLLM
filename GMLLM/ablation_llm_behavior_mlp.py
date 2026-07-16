"""
LLM behavior features + MLP ablation.

This baseline reads raw call graph JSON files, aggregates nodes[].behaviors into
package-level tabular features, and trains an MLP without PyG graph data.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from utils.config_utils import load_config
from utils.logger_utils import Logger
from utils.month_utils import generate_month_range


log = Logger("ablation_llm_behavior_mlp.log")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def iter_graph_paths(root_dir: str, month: str, call_graph_filename: str) -> Iterable[Path]:
    month_dir = Path(root_dir) / month
    if not month_dir.exists():
        log.log(f"[WARNING] Month directory does not exist: {month_dir}")
        return

    for package_dir in sorted(month_dir.iterdir()):
        if not package_dir.is_dir():
            continue
        graph_path = package_dir / call_graph_filename
        if graph_path.exists():
            yield graph_path


def read_behaviors(graph_path: Path) -> List[str]:
    try:
        with open(graph_path, "r", encoding="utf-8", errors="ignore") as f:
            graph = json.load(f)
    except Exception as exc:
        log.log(f"[WARNING] Failed to read {graph_path}: {exc}")
        return []

    behaviors = []
    for node in graph.get("nodes", []):
        node_behaviors = node.get("behaviors", [])
        if isinstance(node_behaviors, list):
            behaviors.extend(str(b) for b in node_behaviors if b)
    return behaviors


def build_behavior_vocab(
    root_dirs: Sequence[str],
    months: Sequence[str],
    call_graph_filename: str,
) -> Dict[str, int]:
    behavior_set = set()
    for root_dir in root_dirs:
        for month in months:
            for graph_path in iter_graph_paths(root_dir, month, call_graph_filename):
                behavior_set.update(read_behaviors(graph_path))

    behavior2idx = {behavior: idx for idx, behavior in enumerate(sorted(behavior_set))}
    if not behavior2idx:
        raise ValueError("No LLM behaviors found in base training months.")
    return behavior2idx


def vectorize_behaviors(behaviors: Sequence[str], behavior2idx: Dict[str, int], mode: str) -> torch.Tensor:
    x = torch.zeros(len(behavior2idx), dtype=torch.float32)
    for behavior in behaviors:
        idx = behavior2idx.get(behavior)
        if idx is None:
            continue
        if mode == "count":
            x[idx] += 1.0
        else:
            x[idx] = 1.0

    if mode == "frequency":
        total = x.sum()
        if total > 0:
            x = x / total
    return x


class BehaviorFeatureDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        months: Sequence[str],
        label: int,
        behavior2idx: Dict[str, int],
        call_graph_filename: str,
        aggregation: str = "presence",
    ):
        self.samples = []
        for month in months:
            for graph_path in iter_graph_paths(root_dir, month, call_graph_filename):
                behaviors = read_behaviors(graph_path)
                x = vectorize_behaviors(behaviors, behavior2idx, aggregation)
                self.samples.append(
                    {
                        "x": x,
                        "y": torch.tensor(label, dtype=torch.long),
                        "package": graph_path.parent.name,
                        "month": month,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MemoryDataset(Dataset):
    def __init__(self, samples: Sequence[dict]):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def split_dataset(dataset: Dataset, train_ratio: float, val_ratio: float = None):
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
    if val_ratio is None:
        return Subset(dataset, indices[:n_train]), Subset(dataset, indices[n_train:])

    n_val = int(val_ratio * n_total)
    return (
        Subset(dataset, indices[:n_train]),
        Subset(dataset, indices[n_train:n_train + n_val]),
        Subset(dataset, indices[n_train + n_val:]),
    )


def split_train_val_test(normal_ds, malicious_ds, train_ratio: float, val_ratio: float):
    normal_train, normal_val, normal_test = split_dataset(normal_ds, train_ratio, val_ratio)
    malicious_train, malicious_val, malicious_test = split_dataset(malicious_ds, train_ratio, val_ratio)
    return normal_train, normal_val, normal_test, malicious_train, malicious_val, malicious_test


def split_train_test(normal_ds, malicious_ds, train_ratio: float):
    normal_train, normal_test = split_dataset(normal_ds, train_ratio)
    malicious_train, malicious_test = split_dataset(malicious_ds, train_ratio)
    return normal_train, normal_test, malicious_train, malicious_test


def make_concat(normal_ds, malicious_ds):
    return ConcatDataset([normal_ds, malicious_ds])


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class BehaviorMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.5,
        activation: str = "relu",
        num_classes: int = 2,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, optimizer, criterion, device, extra_loss_fn=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if extra_loss_fn is not None:
            loss = loss + extra_loss_fn(model)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        total_loss += loss.item() * y.numel()

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    if not all_labels:
        return 0.0, 0.0, 0.0, 0.0

    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1], zero_division=0
    )
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    benign_acc = tn / (tn + fp) if (tn + fp) > 0 else 0

    log.log(f"[Validation] Overall Acc: {acc:.4f} | Malicious F1: {f1[1]:.4f}")
    log.log(f" └─ Malicious Metrics: Precision: {p[1]:.4f} | Recall: {r[1]:.4f} (TP:{tp}, FN:{fn})")
    log.log(f" └─ Benign Accuracy..: {benign_acc:.4f} (TN:{tn}, FP:{fp})")
    return f1[1], acc, r[1], p[1]


class TabularReplayBuffer:
    def __init__(self, memory_per_month=10, reservoir_capacity=None, class_balanced=True):
        self.memory_per_month = memory_per_month
        self.reservoir_capacity = reservoir_capacity
        self.class_balanced = class_balanced
        self.samples = []
        self.class_samples = {0: [], 1: []}
        self.seen_count = 0
        self.class_seen_count = {0: 0, 1: 0}

    def add_month_random(self, month, normal_train, malicious_train):
        n_select = min(self.memory_per_month // 2, len(normal_train), len(malicious_train))
        if n_select == 0:
            return
        for idx in random.sample(range(len(normal_train)), n_select):
            self.samples.append(normal_train[idx])
        for idx in random.sample(range(len(malicious_train)), n_select):
            self.samples.append(malicious_train[idx])

    def add_month_reservoir(self, month, normal_train, malicious_train):
        for idx in range(len(normal_train)):
            self._add_reservoir(normal_train[idx], label=0)
        for idx in range(len(malicious_train)):
            self._add_reservoir(malicious_train[idx], label=1)

    def _add_reservoir(self, sample, label: int):
        if self.class_balanced:
            capacity = self._class_capacity(label)
            buffer = self.class_samples[label]
            self.class_seen_count[label] += 1
            seen = self.class_seen_count[label]
            self._reservoir_add(buffer, sample, capacity, seen)
            return

        self.seen_count += 1
        self._reservoir_add(self.samples, sample, self.reservoir_capacity, self.seen_count)

    @staticmethod
    def _reservoir_add(buffer, sample, capacity: int, seen_count: int):
        if capacity <= 0:
            return
        if len(buffer) < capacity:
            buffer.append(sample)
            return
        replace_idx = random.randint(0, seen_count - 1)
        if replace_idx < capacity:
            buffer[replace_idx] = sample

    def _class_capacity(self, label: int):
        benign_capacity = self.reservoir_capacity // 2
        malicious_capacity = self.reservoir_capacity - benign_capacity
        return benign_capacity if label == 0 else malicious_capacity

    def all_samples(self):
        if self.reservoir_capacity is not None and self.class_balanced:
            return self.class_samples[0] + self.class_samples[1]
        return self.samples

    def make_loader(self, batch_size: int, num_workers: int):
        samples = self.all_samples()
        if not samples:
            return None
        return make_loader(MemoryDataset(samples), batch_size, shuffle=True, num_workers=num_workers)

    def stats(self):
        samples = self.all_samples()
        n_benign = sum(1 for sample in samples if int(sample["y"]) == 0)
        n_malicious = sum(1 for sample in samples if int(sample["y"]) == 1)
        stats = {"memory_size": len(samples), "benign": n_benign, "malicious": n_malicious}
        if self.reservoir_capacity is not None:
            stats.update(
                {
                    "reservoir_capacity": self.reservoir_capacity,
                    "seen_count": sum(self.class_seen_count.values()) if self.class_balanced else self.seen_count,
                    "class_balanced": self.class_balanced,
                }
            )
        return stats


class TabularContinualStrategy:
    def __init__(self, config: dict, batch_size: int, num_workers: int):
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.name = self.resolve_strategy_name(config)
        replay_config = config.get("replay", {})
        if self.name == "random_per_month_replay":
            self.buffer = TabularReplayBuffer(
                memory_per_month=replay_config.get("memory_per_month", config.get("memory_per_month", 10))
            )
        elif self.name == "reservoir_replay":
            self.buffer = TabularReplayBuffer(
                reservoir_capacity=replay_config.get("reservoir_capacity", 500),
                class_balanced=replay_config.get("class_balanced", True),
            )
        elif self.name == "none":
            self.buffer = None
        elif self.name == "ewc":
            self.buffer = None
            ewc_config = config.get("ewc", {})
            self.ewc_lambda = ewc_config.get("lambda", 1000.0)
            self.fisher_batches = ewc_config.get("fisher_batches")
            self.online = ewc_config.get("online", True)
            self.gamma = ewc_config.get("gamma", 1.0)
            self.prev_params = None
            self.fisher = None
            self.last_train_loader = None
        else:
            raise ValueError(f"Unknown continual learning strategy: {self.name}")

    @staticmethod
    def resolve_strategy_name(config: dict) -> str:
        if "strategy" in config:
            return config.get("strategy")
        return "random_per_month_replay" if config.get("use_memory", True) else "none"

    def before_incremental(self, train_datasets: dict, base_months: Sequence[str]):
        if self.name not in {"random_per_month_replay", "reservoir_replay"}:
            return
        for month in base_months:
            if month not in train_datasets:
                continue
            normal_train, malicious_train = train_datasets[month]
            if self.name == "random_per_month_replay":
                self.buffer.add_month_random(month, normal_train, malicious_train)
            else:
                self.buffer.add_month_reservoir(month, normal_train, malicious_train)
        self.log_stats()

    def train_month(self, model, month_train_loader, optimizer, criterion, device, epochs):
        if self.name == "ewc":
            self.last_train_loader = month_train_loader

        last_loss, last_acc = 0.0, 0.0
        for epoch in range(1, epochs + 1):
            if self.name in {"random_per_month_replay", "reservoir_replay"}:
                memory_loader = self.buffer.make_loader(self.batch_size, self.num_workers)
                train_loss, train_acc = self._train_alternate(
                    model, month_train_loader, memory_loader, optimizer, criterion, device
                )
            else:
                extra_loss_fn = self.penalty if self.name == "ewc" else None
                train_loss, train_acc = train_epoch(
                    model, month_train_loader, optimizer, criterion, device, extra_loss_fn=extra_loss_fn
                )
            log.log(f"  Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            last_loss, last_acc = train_loss, train_acc
        return last_loss, last_acc

    def after_month(self, month, train_datasets, model, device):
        if self.name in {"random_per_month_replay", "reservoir_replay"}:
            if month not in train_datasets:
                return
            normal_train, malicious_train = train_datasets[month]
            if self.name == "random_per_month_replay":
                self.buffer.add_month_random(month, normal_train, malicious_train)
            else:
                self.buffer.add_month_reservoir(month, normal_train, malicious_train)
            self.log_stats()
        elif self.name == "ewc" and self.last_train_loader is not None:
            self._update_fisher(model, self.last_train_loader, device)

    def _train_alternate(self, model, new_task_loader, memory_loader, optimizer, criterion, device):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        mem_iter = iter(memory_loader) if memory_loader else None

        for batch in new_task_loader:
            batch_loss, batch_correct, batch_total = self._train_batch(
                model, batch, optimizer, criterion, device
            )
            total_loss += batch_loss
            correct += batch_correct
            total += batch_total

            if mem_iter is None:
                continue
            try:
                mem_batch = next(mem_iter)
            except StopIteration:
                mem_iter = iter(memory_loader)
                mem_batch = next(mem_iter)

            batch_loss, batch_correct, batch_total = self._train_batch(
                model, mem_batch, optimizer, criterion, device
            )
            total_loss += batch_loss
            correct += batch_correct
            total += batch_total

        if total == 0:
            return 0.0, 0.0
        return total_loss / total, correct / total

    @staticmethod
    def _train_batch(model, batch, optimizer, criterion, device):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)
        total = y.numel()
        correct = (pred == y).sum().item()
        return loss.item() * total, correct, total

    def penalty(self, model):
        if self.prev_params is None or self.fisher is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher:
                loss = loss + (self.fisher[name] * (param - self.prev_params[name]).pow(2)).sum()
        return self.ewc_lambda * loss

    def _update_fisher(self, model, loader, device):
        fisher = {
            name: torch.zeros_like(param, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        model.eval()
        batches = 0
        for batch in loader:
            if self.fisher_batches is not None and batches >= self.fisher_batches:
                break
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            model.zero_grad()
            out = model(x)
            log_probs = F.log_softmax(out, dim=1)
            selected = log_probs.gather(1, y.view(-1, 1)).sum()
            selected.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            batches += 1

        if batches > 0:
            for name in fisher:
                fisher[name] /= batches

        if self.online and self.fisher is not None:
            self.fisher = {name: self.gamma * self.fisher[name] + fisher[name] for name in fisher}
        else:
            self.fisher = fisher
        self.prev_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        model.zero_grad()

    def log_stats(self):
        if not self.buffer:
            return
        stats = self.buffer.stats()
        log.log(
            f"  Memory bank: {stats.get('memory_size', 0)} samples "
            f"({stats.get('benign', 0)} benign, {stats.get('malicious', 0)} malicious)"
        )
        if "seen_count" in stats:
            log.log(
                f"  Reservoir: capacity={stats.get('reservoir_capacity')} "
                f"seen={stats.get('seen_count')} class_balanced={stats.get('class_balanced')}"
            )

    def stats(self):
        stats = {"strategy": self.name}
        if self.buffer:
            stats.update(self.buffer.stats())
        if self.name == "ewc":
            stats.update(
                {
                    "ewc_lambda": self.ewc_lambda,
                    "fisher_params": 0 if self.fisher is None else len(self.fisher),
                    "online": self.online,
                    "gamma": self.gamma,
                }
            )
        return stats


def train_base_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, patience, model_save_path):
    best_val_f1 = -1.0
    no_improve_count = 0
    train_start = time.time()
    os.makedirs(os.path.dirname(model_save_path) or "./models", exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_f1, val_acc, val_recall, val_precision = validate(model, val_loader, device)
        log.log(
            f"Epoch {epoch:03d} | Loss {train_loss:.4f} | TrainAcc {train_acc:.4f} | "
            f"ValF1 {val_f1:.4f} | ValAcc {val_acc:.4f} | MalRecall {val_recall:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            log.log(f"Early stopping at epoch {epoch}. No improvement for {patience} epochs.")
            break

    train_time = time.time() - train_start
    log.log(f"Base training done in {train_time:.2f}s. Best Val F1: {best_val_f1:.4f}")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model, best_val_f1


def load_month_datasets(month: str, behavior2idx: dict, data_paths: dict, call_graph_filename: str, aggregation: str):
    normal_ds = BehaviorFeatureDataset(
        data_paths["benign_root"], [month], 0, behavior2idx, call_graph_filename, aggregation
    )
    malicious_ds = BehaviorFeatureDataset(
        data_paths["malicious_root"], [month], 1, behavior2idx, call_graph_filename, aggregation
    )
    return normal_ds, malicious_ds


def append_average_metrics(results: dict, train_time_key: str = None):
    if not results.get("f1"):
        return
    results["avg_f1"] = sum(results["f1"]) / len(results["f1"])
    results["avg_acc"] = sum(results["acc"]) / len(results["acc"])
    results["avg_precision"] = sum(results["precision"]) / len(results["precision"])
    results["avg_recall"] = sum(results["recall"]) / len(results["recall"])
    if train_time_key and results.get(train_time_key):
        results[f"avg_{train_time_key}"] = sum(results[train_time_key]) / len(results[train_time_key])


def run_behavior_mlp_ablation(
    config: dict,
    seed: int,
):
    set_seed(seed)

    dataset_config = config["dataset"]
    training_config = config["training"]
    model_config = config.get("model", {})
    cl_config = config.get("continual_learning", {})
    paths_config = config.get("paths", {})
    results_config = config.get("results", {})
    features_config = config.get("features", {})

    base_path = Path(dataset_config["base_path"])
    data_paths = {
        "benign_root": str(base_path / dataset_config["benign_root"]),
        "malicious_root": str(base_path / dataset_config["malicious_root"]),
    }
    call_graph_filename = dataset_config.get("call_graph_filename", "call_graph.json")
    aggregation = features_config.get("behavior_aggregation", "presence")
    if aggregation not in {"presence", "count", "frequency"}:
        raise ValueError(f"Unsupported behavior_aggregation: {aggregation}")

    base_train_months = tuple(cl_config.get("base_train_months", ["2022-01", "2023-02"]))
    incremental_months = tuple(cl_config.get("incremental_months", ["2023-03", "2024-12"]))
    base_month_list = generate_month_range(base_train_months[0], base_train_months[1])
    inc_month_list = generate_month_range(incremental_months[0], incremental_months[1])

    log.log("=" * 60)
    log.log("Running LLM Behavior Features + MLP Ablation")
    log.log("=" * 60)
    log.log(f"Call graph file: {call_graph_filename}")
    log.log(f"Behavior aggregation: {aggregation}")
    log.log(f"Base months: {base_train_months}")
    log.log(f"Incremental months: {incremental_months}")

    behavior2idx = build_behavior_vocab(
        [data_paths["benign_root"], data_paths["malicious_root"]],
        base_month_list,
        call_graph_filename,
    )
    log.log(f"Behavior vocab size: {len(behavior2idx)}")

    train_ratio = training_config.get("train_ratio", 0.8)
    val_ratio = training_config.get("val_ratio", 0.1)
    batch_size = training_config.get("batch_size", 128)
    num_workers = training_config.get("num_workers", 4)
    epochs = training_config.get("epochs", 60)
    patience = training_config.get("patience", 10)
    incremental_epochs = cl_config.get("incremental_epochs", 5)

    train_datasets, val_datasets, test_datasets, unseen_test_datasets = {}, {}, {}, {}
    for month in base_month_list:
        normal_ds, malicious_ds = load_month_datasets(
            month, behavior2idx, data_paths, call_graph_filename, aggregation
        )
        log.log(f"  {month}: {len(normal_ds)} normal, {len(malicious_ds)} malicious")
        split = split_train_val_test(normal_ds, malicious_ds, train_ratio, val_ratio)
        normal_train, normal_val, normal_test, malicious_train, malicious_val, malicious_test = split
        train_datasets[month] = (normal_train, malicious_train)
        val_datasets[month] = (normal_val, malicious_val)
        test_datasets[month] = (normal_test, malicious_test)

    for month in inc_month_list:
        normal_ds, malicious_ds = load_month_datasets(
            month, behavior2idx, data_paths, call_graph_filename, aggregation
        )
        log.log(f"  {month}: {len(normal_ds)} normal, {len(malicious_ds)} malicious")
        unseen_test_datasets[month] = (normal_ds, malicious_ds)
        normal_train, normal_test, malicious_train, malicious_test = split_train_test(
            normal_ds, malicious_ds, train_ratio
        )
        train_datasets[month] = (normal_train, malicious_train)
        test_datasets[month] = (normal_test, malicious_test)

    train_dataset = ConcatDataset([
        make_concat(*train_datasets[month])
        for month in base_month_list
    ])
    val_dataset = ConcatDataset([
        make_concat(normal_val, malicious_val)
        for normal_val, malicious_val in val_datasets.values()
    ])
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Empty base train or validation set.")

    device_config = config.get("device", "auto")
    device = "cuda" if torch.cuda.is_available() else "cpu" if device_config == "auto" else device_config
    device = torch.device(device)

    hidden_dims = model_config.get("mlp_hidden_dims", [128, 64])
    dropout = model_config.get("dropout", 0.5)
    activation = model_config.get("activation", "relu")
    num_classes = model_config.get("num_classes", 2)
    model = BehaviorMLP(
        input_dim=len(behavior2idx),
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.get("learning_rate", 0.005),
        weight_decay=training_config.get("weight_decay", 0.001),
    )
    criterion = nn.CrossEntropyLoss()

    models_dir = paths_config.get("models_dir", "./models/deepseek_behavior_mlp")
    model_save_path = str(Path(models_dir) / "base_model.pt")
    train_loader = make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    model, _best_val_f1 = train_base_model(
        model, train_loader, val_loader, optimizer, criterion, device, epochs, patience, model_save_path
    )

    with open(Path(models_dir) / "behavior2idx.json", "w") as f:
        json.dump(behavior2idx, f, indent=2)

    strategy = TabularContinualStrategy(cl_config, batch_size=batch_size, num_workers=num_workers)
    strategy.before_incremental(train_datasets, base_month_list)
    log.log(f"Continual strategy: {strategy.name}")

    model_prefix = paths_config.get("prefix", "mlp_incremental_")
    results_dir = Path(paths_config.get("results_dir", "../results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    future_result = {"month": [], "f1": [], "acc": [], "precision": [], "recall": [], "train_time": []}
    seen_result = {"month": [], "f1": [], "acc": [], "precision": [], "recall": []}
    efficiency_result = {
        "task": "llm_behavior_mlp_monthly_update",
        "strategy": strategy.name,
        "device": str(device),
        "seed": seed,
        "base_train_months": list(base_train_months),
        "incremental_months": list(incremental_months),
        "incremental_epochs": incremental_epochs,
        "batch_size": batch_size,
        "behavior_aggregation": aggregation,
        "behavior_vocab_size": len(behavior2idx),
        "time_unit": "seconds",
        "memory_unit": "MiB",
        "month": [],
        "monthly_update_time": [],
        "peak_gpu_memory_allocated_mb": [],
        "peak_gpu_memory_reserved_mb": [],
        "n_train": [],
    }

    for month in inc_month_list:
        log.log("\n" + "=" * 40)
        log.log(f"Month {month}: test first, then train MLP")
        log.log("=" * 40)

        current_test_dataset = make_concat(*unseen_test_datasets[month])
        current_test_loader = make_loader(current_test_dataset, batch_size, shuffle=False, num_workers=num_workers)
        f1, acc, recall, precision = validate(model, current_test_loader, device)
        future_result["month"].append(month)
        future_result["f1"].append(f1)
        future_result["acc"].append(acc)
        future_result["precision"].append(precision)
        future_result["recall"].append(recall)

        normal_train, malicious_train = train_datasets[month]
        month_train_dataset = make_concat(normal_train, malicious_train)
        month_train_loader = make_loader(month_train_dataset, batch_size, shuffle=True, num_workers=num_workers)

        update_start = start_efficiency_measurement(device)
        strategy.train_month(model, month_train_loader, optimizer, criterion, device, incremental_epochs)
        strategy.after_month(month, train_datasets, model, device)
        efficiency = finish_efficiency_measurement(device, update_start)
        future_result["train_time"].append(efficiency["update_time"])

        efficiency_result["month"].append(month)
        efficiency_result["monthly_update_time"].append(efficiency["update_time"])
        efficiency_result["peak_gpu_memory_allocated_mb"].append(efficiency["peak_gpu_memory_allocated_mb"])
        efficiency_result["peak_gpu_memory_reserved_mb"].append(efficiency["peak_gpu_memory_reserved_mb"])
        efficiency_result["n_train"].append(len(month_train_dataset))

        model_path = Path(models_dir) / f"{model_prefix}{month}.pt"
        torch.save(model.state_dict(), model_path)
        log.log(f"Model saved to {model_path}")
        log.log(f"Strategy stats: {strategy.stats()}")

        cumulative_normal, cumulative_malicious = [], []
        for past_month in generate_month_range(incremental_months[0], month):
            if past_month not in test_datasets:
                continue
            normal_test, malicious_test = test_datasets[past_month]
            cumulative_normal.append(normal_test)
            cumulative_malicious.append(malicious_test)
        if cumulative_normal and cumulative_malicious:
            cumulative_test_dataset = ConcatDataset(cumulative_normal + cumulative_malicious)
            cumulative_test_loader = make_loader(cumulative_test_dataset, batch_size, shuffle=False, num_workers=num_workers)
            f1, acc, recall, precision = validate(model, cumulative_test_loader, device)
            seen_result["month"].append(month)
            seen_result["f1"].append(f1)
            seen_result["acc"].append(acc)
            seen_result["precision"].append(precision)
            seen_result["recall"].append(recall)

    append_average_metrics(future_result, train_time_key="train_time")
    append_average_metrics(seen_result)
    if efficiency_result["monthly_update_time"]:
        efficiency_result["avg_monthly_update_time"] = (
            sum(efficiency_result["monthly_update_time"]) / len(efficiency_result["monthly_update_time"])
        )
        allocated = [v for v in efficiency_result["peak_gpu_memory_allocated_mb"] if v is not None]
        reserved = [v for v in efficiency_result["peak_gpu_memory_reserved_mb"] if v is not None]
        if allocated:
            efficiency_result["max_peak_gpu_memory_allocated_mb"] = max(allocated)
        if reserved:
            efficiency_result["max_peak_gpu_memory_reserved_mb"] = max(reserved)

    future_result["seed"] = seed
    seen_result["seed"] = seed
    efficiency_result["seed"] = seed

    future_file = results_config.get(
        "future_month",
        "CL_unk_behavior_mlp_test_than_train_future_month_deepseek.json",
    )
    efficiency_file = results_config.get(
        "efficiency",
        "CL_unk_behavior_mlp_efficiency_deepseek.json",
    )
    seen_file = future_file.replace("future_month", "seen_month")
    if seen_file == future_file:
        seen_file = future_file.replace(".json", "_seen_month.json")

    with open(results_dir / future_file, "w") as f:
        json.dump(future_result, f, indent=2)
    with open(results_dir / seen_file, "w") as f:
        json.dump(seen_result, f, indent=2)
    with open(results_dir / efficiency_file, "w") as f:
        json.dump(efficiency_result, f, indent=2)

    log.log(f"Future month results saved to {results_dir / future_file}")
    log.log(f"Seen month results saved to {results_dir / seen_file}")
    log.log(f"Efficiency results saved to {results_dir / efficiency_file}")
    return model, seen_result, future_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM behavior features + MLP ablation")
    parser.add_argument("--config", type=str, default="GMLLM/configs/ablations/llm_behavior_mlp_deepseek.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["training"].get("seed", 42)
    run_behavior_mlp_ablation(config, seed)
