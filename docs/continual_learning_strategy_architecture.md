# Continual Learning Strategy Architecture

## 背景

当前 `GMLLM/continual_learning_memory.py` 的增量学习逻辑和主训练流程耦合较紧：

- 主流程直接维护 `memory_samples`。
- 记忆池样本选择固定为“每月随机抽样，正负样本 1:1”。
- replay 训练逻辑固定为当前月 batch 和 memory batch 交替训练。
- `use_memory` 只能表达“是否使用现有 replay”，无法表达 Reservoir Replay、EWC 等不同增量学习策略。

这会导致新增策略时不断修改主循环，容易把数据加载、评估、模型保存、策略状态更新混在一起。

目标是把“按月增量训练流程”和“具体增量学习策略”拆开，让后续新增策略只需要实现策略类，而不是改主训练循环。

## 总体设计

主入口仍保留在 `GMLLM/continual_learning_memory.py`，负责：

- 加载词汇表、模型、预训练权重。
- 加载基础月份和增量月份数据。
- 按月执行“训练前评估 -> 增量训练 -> 保存模型 -> seen months 评估”。
- 保存结果 JSON。

增量学习策略移动到独立模块，例如 `GMLLM/continual_strategies.py`，负责：

- 是否使用 replay。
- replay buffer 如何选样和更新。
- 每个月如何训练模型。
- 是否添加额外正则项，例如 EWC penalty。
- 输出策略相关统计信息。

建议的核心接口：

```python
class ContinualStrategy:
    name: str

    def before_incremental(self, train_datasets, base_months, model, device):
        pass

    def before_month(self, month, train_datasets, model, device):
        pass

    def train_month(self, model, month_train_loader, optimizer, criterion, device, epochs):
        raise NotImplementedError

    def after_month(self, month, train_datasets, model, device):
        pass

    def stats(self) -> dict:
        return {}
```

主循环只依赖 `ContinualStrategy`，不直接关心 memory 或 Fisher 的内部结构。

## 策略与组件

### NoneStrategy

无增量防遗忘策略，只用当前月数据训练。

用途：

- 替代旧配置 `use_memory: false`。
- 作为无 replay、无正则的 baseline。

训练行为：

- 对 `month_train_loader` 正常训练。
- 不维护任何历史状态。

### ReplayStrategy

Replay 类策略的公共训练器，负责把当前月数据和 memory 数据合并到训练过程中。

默认复刻当前行为：

- 当前月 batch 训练一步。
- memory batch 训练一步。
- memory loader 耗尽后重新迭代。

Replay 训练器不负责选样。选样和容量控制由 `ReplayBuffer` 实现。

```python
class ReplayBuffer:
    def add_month(self, month, normal_train, malicious_train):
        raise NotImplementedError

    def make_loader(self, batch_size, num_workers=4):
        raise NotImplementedError

    def stats(self) -> dict:
        return {}
```

### RandomPerMonthReplayBuffer

复刻现有 `select_sample` 行为：

- 每个月最多抽 `memory_per_month` 个样本。
- benign 和 malicious 各抽 `memory_per_month // 2`。
- 新月份样本追加到 memory，历史 memory 不删除。

该策略适合保持旧实验可复现。

### ReservoirReplayBuffer

使用 reservoir sampling 维护固定容量的全局 memory。

默认设计：

- `reservoir_capacity` 表示 memory 总容量。
- 遍历每个月训练样本时维护全局 seen count。
- 第 `n` 个样本：
  - 如果 memory 未满，直接加入。
  - 如果 memory 已满，以 `capacity / n` 的概率进入 memory。
  - 若进入，则随机替换已有样本。

建议默认开启类别均衡：

- `class_balanced: true` 时，为 benign 和 malicious 分别维护一个 reservoir。
- 每类容量为 `reservoir_capacity // 2`。
- 这样可以避免恶意样本较少时被 benign 样本挤出。

### EWCStrategy

EWC 是非 replay 策略，不保存历史样本，不创建 memory loader。

训练 loss：

```python
loss = ce_loss + ewc_lambda * ewc_penalty
```

EWC penalty：

```python
sum(fisher[name] * (param - prev_param[name]) ** 2)
```

每个月训练完成后：

- 使用当月训练 loader 估计 Fisher diagonal。
- 保存当前模型参数快照。
- 下一月训练时使用该 Fisher 和参数快照作为约束。

默认行为：

- 对所有 `requires_grad=True` 的参数启用 EWC。
- Fisher 使用当前月训练数据估计。
- `online: true` 时只维护一份累计 Fisher。
- `gamma` 控制历史 Fisher 衰减，默认 `1.0`。

## 配置设计

建议将旧的 `use_memory` 扩展为显式策略配置：

```yaml
continual_learning:
  enabled: true
  base_train_months: ['2022-01', '2023-02']
  incremental_months: ['2023-03', '2024-12']
  incremental_epochs: 5

  strategy: "random_per_month_replay"

  replay:
    batch_mode: "alternate"
    memory_per_month: 10
    reservoir_capacity: 500
    class_balanced: true

  ewc:
    lambda: 1000.0
    fisher_batches: null
    online: true
    gamma: 1.0
```

支持的 `strategy`：

- `none`
- `random_per_month_replay`
- `reservoir_replay`
- `ewc`

兼容旧配置：

- 未配置 `strategy` 且 `use_memory: true` 时，映射为 `random_per_month_replay`。
- 未配置 `strategy` 且 `use_memory: false` 时，映射为 `none`。
- 旧字段 `memory_per_month` 可以继续读取，并作为 `replay.memory_per_month` 的 fallback。

建议新增 factory：

```python
def build_continual_strategy(cl_config, batch_size, num_workers=4):
    strategy_name = resolve_strategy_name(cl_config)
    if strategy_name == "none":
        return NoneStrategy()
    if strategy_name == "random_per_month_replay":
        return ReplayStrategy(RandomPerMonthReplayBuffer(...))
    if strategy_name == "reservoir_replay":
        return ReplayStrategy(ReservoirReplayBuffer(...))
    if strategy_name == "ewc":
        return EWCStrategy(...)
    raise ValueError(f"Unknown continual learning strategy: {strategy_name}")
```

## 主流程改造

`continual_learning_memory.py` 中的主循环建议改成以下结构：

```python
strategy = build_continual_strategy(cl_config, batch_size, num_workers)

strategy.before_incremental(
    train_datasets=train_datasets,
    base_months=generate_month_range(base_start, base_end),
    model=model,
    device=device,
)

for month in generate_month_range(inc_start, inc_end):
    evaluate_current_month_before_training(...)

    month_train_loader = build_month_train_loader(...)

    strategy.before_month(month, train_datasets, model, device)
    strategy.train_month(
        model=model,
        month_train_loader=month_train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=incremental_epochs,
    )
    strategy.after_month(month, train_datasets, model, device)

    save_model(...)
    evaluate_seen_months(...)
    log_strategy_stats(strategy.stats())
```

这样主流程不再出现：

- `memory_samples`
- `select_sample`
- `create_memory_loader`
- `if use_memory`
- EWC Fisher 或参数快照细节

## 测试计划

### 兼容性测试

- 使用现有 `configs/default.yaml`，未配置 `strategy` 时应保持原 replay 行为。
- 使用现有 `configs/ablations/no_memory.yaml`，`use_memory: false` 应映射到 `none`。
- 结果 JSON 字段保持兼容：`month`、`f1`、`acc`、`precision`、`recall`、`train_time`、平均值和 `seed`。

### Random Replay 测试

- 每个 base month 初始化后 memory 增量不超过 `memory_per_month`。
- 每个 incremental month 训练后追加当月抽样。
- benign 和 malicious 数量保持 1:1，除非某类样本不足。

### Reservoir Replay 测试

- memory 总量永远不超过 `reservoir_capacity`。
- `class_balanced: true` 时，每类 memory 不超过 `reservoir_capacity // 2`。
- seen count 随处理样本数增长。
- 固定随机种子时 reservoir 内容可复现。

### EWC 测试

- 不创建 memory loader。
- 首个月没有 Fisher 时，只使用 CE loss。
- 第二个月开始 loss 包含 EWC penalty。
- 每个月训练后 Fisher 和参数快照被更新。
- `fisher_batches` 配置为整数时，只使用指定数量 batch 估计 Fisher。

## 实施顺序

1. 新增 `GMLLM/continual_strategies.py`，先迁移现有 random replay 行为。
2. 改造 `continual_learning_memory.py` 使用 `build_continual_strategy`，确保旧配置结果可跑通。
3. 增加 `ReservoirReplayBuffer`，新增 reservoir 配置文件做实验。
4. 增加 `EWCStrategy`，新增 EWC 配置文件做实验。
5. 补充最小单元测试或轻量 smoke test，验证 factory、buffer 容量和 EWC penalty。

