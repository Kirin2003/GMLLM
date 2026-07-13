import random
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


Sample = Tuple[object, int, str]


class MemoryDataset(Dataset):
    """Wrap replay samples as a PyG dataset."""

    def __init__(self, samples_list: List[Sample]):
        super().__init__()
        self.samples = samples_list
        for data, label, _month in self.samples:
            data.y = torch.tensor([label], dtype=torch.long)

    def len(self):
        return len(self.samples)

    def get(self, idx):
        data, _label, _month = self.samples[idx]
        return data


class ContinualStrategy:
    name = "base"

    def before_incremental(self, train_datasets, base_months, model, device):
        pass

    def before_month(self, month, train_datasets, model, device):
        pass

    def train_month(self, model, month_train_loader, optimizer, criterion, device, epochs):
        raise NotImplementedError

    def after_month(self, month, train_datasets, model, device):
        pass

    def stats(self) -> dict:
        return {"strategy": self.name}


def train_plain_epoch(model, loader, optimizer, criterion, device, extra_loss_fn=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        if extra_loss_fn is not None:
            loss = loss + extra_loss_fn(model)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
        total_loss += loss.item() * data.num_graphs

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


class NoneStrategy(ContinualStrategy):
    name = "none"

    def __init__(self, log=None):
        self.log = log

    def train_month(self, model, month_train_loader, optimizer, criterion, device, epochs):
        last_loss, last_acc = 0.0, 0.0
        if self.log:
            self.log.log("Training without continual regularization...")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_plain_epoch(
                model, month_train_loader, optimizer, criterion, device
            )
            if self.log:
                self.log.log(f"  Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            last_loss, last_acc = train_loss, train_acc

        return last_loss, last_acc


class ReplayBuffer:
    def add_month(self, month, normal_train, malicious_train):
        raise NotImplementedError

    def make_loader(self, batch_size, num_workers=4):
        raise NotImplementedError

    def stats(self) -> dict:
        return {}


class RandomPerMonthReplayBuffer(ReplayBuffer):
    def __init__(self, memory_per_month=10):
        self.memory_per_month = memory_per_month
        self.samples: List[Sample] = []

    def add_month(self, month, normal_train, malicious_train):
        n_select = min(self.memory_per_month // 2, len(normal_train), len(malicious_train))
        if n_select == 0:
            return

        normal_indices = random.sample(range(len(normal_train)), n_select)
        malicious_indices = random.sample(range(len(malicious_train)), n_select)

        for idx in normal_indices:
            self.samples.append((normal_train[idx], 0, month))
        for idx in malicious_indices:
            self.samples.append((malicious_train[idx], 1, month))

    def make_loader(self, batch_size, num_workers=4):
        if not self.samples:
            return None
        return DataLoader(
            MemoryDataset(self.samples),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def stats(self) -> dict:
        n_benign = sum(1 for _data, label, _month in self.samples if label == 0)
        n_malicious = sum(1 for _data, label, _month in self.samples if label == 1)
        return {
            "memory_size": len(self.samples),
            "benign": n_benign,
            "malicious": n_malicious,
        }


class ReservoirReplayBuffer(ReplayBuffer):
    def __init__(self, capacity=500, class_balanced=True):
        self.capacity = capacity
        self.class_balanced = class_balanced
        self.samples: List[Sample] = []
        self.seen_count = 0
        self.class_samples = {0: [], 1: []}
        self.class_seen_count = {0: 0, 1: 0}

    def add_month(self, month, normal_train, malicious_train):
        for idx in range(len(normal_train)):
            self._add_sample((normal_train[idx], 0, month), label=0)
        for idx in range(len(malicious_train)):
            self._add_sample((malicious_train[idx], 1, month), label=1)

    def _add_sample(self, sample: Sample, label: int):
        if self.class_balanced:
            per_class_capacity = self._class_capacity(label)
            class_buffer = self.class_samples[label]
            self.class_seen_count[label] += 1
            seen = self.class_seen_count[label]
            self._reservoir_add(class_buffer, sample, per_class_capacity, seen)
            return

        self.seen_count += 1
        self._reservoir_add(self.samples, sample, self.capacity, self.seen_count)

    @staticmethod
    def _reservoir_add(buffer: List[Sample], sample: Sample, capacity: int, seen_count: int):
        if capacity <= 0:
            return
        if len(buffer) < capacity:
            buffer.append(sample)
            return
        replace_idx = random.randint(0, seen_count - 1)
        if replace_idx < capacity:
            buffer[replace_idx] = sample

    def _class_capacity(self, label: int):
        benign_capacity = self.capacity // 2
        malicious_capacity = self.capacity - benign_capacity
        return benign_capacity if label == 0 else malicious_capacity

    def _all_samples(self):
        if not self.class_balanced:
            return self.samples
        return self.class_samples[0] + self.class_samples[1]

    def make_loader(self, batch_size, num_workers=4):
        samples = self._all_samples()
        if not samples:
            return None
        return DataLoader(
            MemoryDataset(samples),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def stats(self) -> dict:
        samples = self._all_samples()
        n_benign = sum(1 for _data, label, _month in samples if label == 0)
        n_malicious = sum(1 for _data, label, _month in samples if label == 1)
        seen_count = sum(self.class_seen_count.values()) if self.class_balanced else self.seen_count
        return {
            "memory_size": len(samples),
            "benign": n_benign,
            "malicious": n_malicious,
            "reservoir_capacity": self.capacity,
            "seen_count": seen_count,
            "class_balanced": self.class_balanced,
        }


class ReplayStrategy(ContinualStrategy):
    name = "replay"

    def __init__(self, buffer, batch_size, num_workers=4, log=None, name="replay"):
        self.buffer = buffer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log = log
        self.name = name

    def before_incremental(self, train_datasets, base_months, model, device):
        for month in base_months:
            if month in train_datasets:
                normal_train, malicious_train = train_datasets[month]
                self.buffer.add_month(month, normal_train, malicious_train)
        self._log_stats()

    def train_month(self, model, month_train_loader, optimizer, criterion, device, epochs):
        last_loss, last_acc = 0.0, 0.0
        for epoch in range(1, epochs + 1):
            memory_loader = self.buffer.make_loader(self.batch_size, self.num_workers)
            train_loss, train_acc = self._train_alternate(
                model, month_train_loader, memory_loader, optimizer, criterion, device
            )
            if self.log:
                self.log.log(f"  Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            last_loss, last_acc = train_loss, train_acc
        return last_loss, last_acc

    def after_month(self, month, train_datasets, model, device):
        if month not in train_datasets:
            return
        if self.log:
            self.log.log("Updating replay buffer...")
        normal_train, malicious_train = train_datasets[month]
        self.buffer.add_month(month, normal_train, malicious_train)
        self._log_stats()

    def _train_alternate(self, model, new_task_loader, memory_loader, optimizer, criterion, device):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        mem_iter = iter(memory_loader) if memory_loader else None

        for data in new_task_loader:
            batch_loss, batch_correct, batch_total = self._train_batch(
                model, data, optimizer, criterion, device
            )
            total_loss += batch_loss
            correct += batch_correct
            total += batch_total

            if mem_iter is None:
                continue
            try:
                mem_data = next(mem_iter)
            except StopIteration:
                mem_iter = iter(memory_loader)
                mem_data = next(mem_iter)

            batch_loss, batch_correct, batch_total = self._train_batch(
                model, mem_data, optimizer, criterion, device
            )
            total_loss += batch_loss
            correct += batch_correct
            total += batch_total

        if total == 0:
            return 0.0, 0.0
        return total_loss / total, correct / total

    @staticmethod
    def _train_batch(model, data, optimizer, criterion, device):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        correct = (pred == data.y).sum().item()
        total = data.num_graphs
        return loss.item() * total, correct, total

    def _log_stats(self):
        if not self.log:
            return
        stats = self.buffer.stats()
        self.log.log(
            f"  Memory bank: {stats.get('memory_size', 0)} samples "
            f"({stats.get('benign', 0)} benign, {stats.get('malicious', 0)} malicious)"
        )
        if "seen_count" in stats:
            self.log.log(
                f"  Reservoir: capacity={stats.get('reservoir_capacity')} "
                f"seen={stats.get('seen_count')} class_balanced={stats.get('class_balanced')}"
            )

    def stats(self) -> dict:
        stats = {"strategy": self.name}
        stats.update(self.buffer.stats())
        return stats


class EWCStrategy(ContinualStrategy):
    name = "ewc"

    def __init__(self, ewc_lambda=1000.0, fisher_batches=None, online=True, gamma=1.0, log=None):
        self.ewc_lambda = ewc_lambda
        self.fisher_batches = fisher_batches
        self.online = online
        self.gamma = gamma
        self.log = log
        self.prev_params: Optional[Dict[str, torch.Tensor]] = None
        self.fisher: Optional[Dict[str, torch.Tensor]] = None
        self._last_train_loader = None

    def before_month(self, month, train_datasets, model, device):
        self._last_train_loader = None

    def train_month(self, model, month_train_loader, optimizer, criterion, device, epochs):
        self._last_train_loader = month_train_loader
        last_loss, last_acc = 0.0, 0.0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_plain_epoch(
                model,
                month_train_loader,
                optimizer,
                criterion,
                device,
                extra_loss_fn=self.penalty,
            )
            if self.log:
                self.log.log(f"  Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            last_loss, last_acc = train_loss, train_acc

        return last_loss, last_acc

    def after_month(self, month, train_datasets, model, device):
        if self._last_train_loader is None:
            return
        if self.log:
            self.log.log("Updating EWC Fisher information...")
        new_fisher = self._estimate_fisher(model, self._last_train_loader, device)

        if self.online and self.fisher is not None:
            self.fisher = {
                name: self.gamma * self.fisher[name] + new_fisher[name]
                for name in new_fisher
            }
        else:
            self.fisher = new_fisher

        self.prev_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def penalty(self, model):
        if self.prev_params is None or self.fisher is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.fisher:
                continue
            loss = loss + (self.fisher[name] * (param - self.prev_params[name]).pow(2)).sum()
        return self.ewc_lambda * loss

    def _estimate_fisher(self, model, loader, device):
        model.eval()
        fisher = {
            name: torch.zeros_like(param, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        batches = 0

        for data in loader:
            if self.fisher_batches is not None and batches >= self.fisher_batches:
                break

            data = data.to(device)
            model.zero_grad()
            out = model(data)
            log_probs = torch.nn.functional.log_softmax(out, dim=1)
            selected = log_probs.gather(1, data.y.view(-1, 1)).sum()
            selected.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            batches += 1

        if batches > 0:
            for name in fisher:
                fisher[name] /= batches

        model.zero_grad()
        return fisher

    def stats(self) -> dict:
        param_count = 0 if self.fisher is None else len(self.fisher)
        return {
            "strategy": self.name,
            "ewc_lambda": self.ewc_lambda,
            "fisher_params": param_count,
            "online": self.online,
            "gamma": self.gamma,
        }


def resolve_strategy_name(cl_config):
    if "strategy" in cl_config:
        return cl_config.get("strategy")
    return "random_per_month_replay" if cl_config.get("use_memory", True) else "none"


def build_continual_strategy(cl_config, batch_size, num_workers=4, log=None):
    strategy_name = resolve_strategy_name(cl_config)
    replay_config = cl_config.get("replay", {})
    ewc_config = cl_config.get("ewc", {})

    if strategy_name == "none":
        return NoneStrategy(log=log)

    if strategy_name == "random_per_month_replay":
        memory_per_month = replay_config.get(
            "memory_per_month",
            cl_config.get("memory_per_month", 10),
        )
        return ReplayStrategy(
            RandomPerMonthReplayBuffer(memory_per_month=memory_per_month),
            batch_size=batch_size,
            num_workers=num_workers,
            log=log,
            name=strategy_name,
        )

    if strategy_name == "reservoir_replay":
        return ReplayStrategy(
            ReservoirReplayBuffer(
                capacity=replay_config.get("reservoir_capacity", 500),
                class_balanced=replay_config.get("class_balanced", True),
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            log=log,
            name=strategy_name,
        )

    if strategy_name == "ewc":
        return EWCStrategy(
            ewc_lambda=ewc_config.get("lambda", 1000.0),
            fisher_batches=ewc_config.get("fisher_batches"),
            online=ewc_config.get("online", True),
            gamma=ewc_config.get("gamma", 1.0),
            log=log,
        )

    raise ValueError(f"Unknown continual learning strategy: {strategy_name}")
