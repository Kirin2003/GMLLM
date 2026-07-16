# GMLLM

GMLLM 是一个用于检测恶意 PyPI 包的实验性流水线。整体思路是：静态分析 Python 包源码，提取敏感 API 行为规则和调用图，将调用图转换为图特征，用 GNN 进行恶意包分类，再通过带记忆池的增量学习按月更新模型；最后可以选择用 GNN 可解释性方法提取可疑恶意子图，再交给 LLM 进一步判断。

主要代码位于 `GMLLM/`。当前主流程如下：

```text
Python 包源码
  -> cli_extract.py
  -> generate_graph_data_fromJson.py
  -> distinguish_GNN_2.py
  -> continual_learning_memory.py
  -> 可选：run_autoexplanation_parallel.py + LLM 复核
```

## 环境

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

LLM 调用使用 OpenAI 兼容接口，相关参数写在 `GMLLM/configs/*.yaml` 中。如果没有可用的合成规则，提取阶段会回退到 `GMLLM/rules_fallback.py` 中的本地规则。

## 数据目录

数据集根目录由配置文件中的 `dataset.base_path` 指定。默认配置使用 `${DATA_ROOT}` 作为机器相关的数据盘前缀：

```bash
cp .env.example .env
# 当前服务器数据在 /Data2/hxq 时
DATA_ROOT=/Data2/hxq
# 另一台服务器数据在 /Data1/hxq 时，改成：
DATA_ROOT=/Data1/hxq
```

也可以不写 `.env`，直接在 shell 中 `export DATA_ROOT=/Data1/hxq`。shell 环境变量优先于 `.env`。

期望的源码目录结构：

```text
DATASET_ROOT/
├── benign/
│   ├── 2022-01/
│   │   └── package_name/
│   │       └── *.py
│   └── ...
└── malicious/
    ├── 2022-01/
    │   └── package_name/
    │       └── *.py
    └── ...
```

图特征和词表会写回数据集根目录，例如：

```text
DATASET_ROOT/
├── vocab/
│   ├── name2idx.json
│   ├── type2idx.json
│   ├── behavior2idx.json
│   └── edge_type2idx.json
├── benign_call_processed/
└── malicious_call_processed/
```

## 主流程

以下命令均在仓库根目录下运行。

### 1. 提取 Call Graph

```bash
python GMLLM/cli_extract.py --config GMLLM/configs/default.yaml
```

输入：
- `dataset.base_path`、`dataset.benign_root`、`dataset.malicious_root`
- `llm.model_name`、`llm.base_url`、`llm.api_key_env`
- 可选的合成规则文件，例如 `GMLLM/synth_rules.json`

输出：
- 每个包一个 call graph JSON，文件名通常是 `{model_name}_call_graph.json`
- 提取日志

call graph 的主要结构：

```text
nodes: [{id, type, name, qualified_name, file, context, behaviors}, ...]
links: [{source, target, edge_type}, ...]
```

### 2. 生成图特征

```bash
python GMLLM/generate_graph_data_fromJson.py --config GMLLM/configs/default.yaml
```

输入：
- 上一步生成的 call graph JSON
- `continual_learning.base_train_months`
- `continual_learning.incremental_months`

输出：
- `dataset.vocab_dir` 下的词表 JSON
- `dataset.benign_out` 和 `dataset.malicious_out` 下的 PyTorch Geometric `.pt` 图文件
- 每个月的 `index.json`

图特征包含：
- 函数名/节点名 embedding id
- 节点类型 embedding id
- 敏感行为 one-hot 向量
- 边和边类型

### 3. 训练基础 GNN

```bash
python GMLLM/distinguish_GNN_2.py --config GMLLM/configs/default.yaml
```

输入：
- 词表文件
- 处理后的 `.pt` 图文件
- `training`、`model`、`continual_learning`、`paths` 中的训练配置

输出：
- 基础模型，通常是 `GMLLM/models/base_model.pt`
- 未来月份测试结果 JSON，默认写到 `../results`

模型为 `GCNWithBehavior`：函数名 embedding、类型 embedding、敏感行为向量、两层 GCN、mean pooling 和二分类器。

### 4. 带记忆池的增量学习

```bash
python GMLLM/continual_learning_memory.py --config GMLLM/configs/default.yaml
```

输入：
- `paths.pretrained_model` 指定的基础模型
- 按月处理好的图数据
- `continual_learning.memory_per_month`
- `continual_learning.use_memory`

输出：
- 每个月的增量模型，例如 `GMLLM/models/incremental_unk_model_YYYY-MM.pt`
- future-month 和 seen-month 结果 JSON，默认写到 `../results`

记忆池会从历史月份中按良性/恶意 1:1 抽样，并在训练时交替使用当月 batch 和记忆池 batch。

### 5. Edge + Node Type 消融

```bash
python GMLLM/generate_graph_data_fromJson.py --config GMLLM/configs/ablations/edge_type_deepseek.yaml
python GMLLM/distinguish_GNN_2.py --config GMLLM/configs/ablations/edge_type_deepseek.yaml
python GMLLM/continual_learning_memory.py --config GMLLM/configs/ablations/edge_type_deepseek.yaml
```

该消融关闭 `x_names` 和 `nodes[].behaviors`，只使用 `node_type` embedding 作为节点特征，并通过 GCN 的 `edge_index` 做消息传递。它对应更严格的 structure-only / topology+type baseline，不包含 API/function name 语义。

### 6. LLM Behavior Features + MLP 消融

```bash
python GMLLM/ablation_llm_behavior_mlp.py --config GMLLM/configs/ablations/llm_behavior_mlp_deepseek.yaml
```

该消融直接读取 raw call graph JSON 中的 `nodes[].behaviors`，聚合为包级 tabular feature，再用 MLP 分类。它不使用 `CallGraphDatasetFull_Lazy`、PyG graph data、`edge_index` 或 GCN，用于和主 GNN 流程对比图结构建模组件的必要性。

## 可选解释模块

`GMLLM/run_autoexplanation_parallel.py` 用来并行启动 `GMLLM/auto_explanation.py`。

预期用途：

```text
训练好的 GNN + 包图
  -> GNN 解释 mask
  -> 可疑恶意子图
  -> LLM 判断
```

这部分代码来自其他论文实现，当前仓库中没有完整测试过，可能依赖缺失的 `config.json`、`build_single_graph_ckpt.py`、`extract2Json.py` 或额外的工具模块。它应被视为可选后处理模块，不是主训练流程的必需步骤。

示例命令：

```bash
python GMLLM/run_autoexplanation_parallel.py \
  --num-workers 4 \
  --gpus 0,1 \
  --dataset-type malicious
```

## 脚本说明

| 文件 | 作用 | 输入 | 输出 |
|---|---|---|---|
| `GMLLM/cli_extract.py` | 批量提取 call graph | YAML 配置、包源码目录 | `{model_name}_call_graph.json` |
| `GMLLM/ast_parser.py` | Python AST 解析 | `.py` 文件 | 函数、调用、字面量节点 |
| `GMLLM/graph_builder.py` | 构建项目调用图 | AST 解析结果、检测器 | graph dict |
| `GMLLM/llm_detector.py` | 敏感行为检测 | 节点信息、规则 | behavior labels |
| `GMLLM/rules_fallback.py` | 本地兜底规则 | 函数名/字面量 | 匹配到的行为标签 |
| `GMLLM/prompts.py` | LLM prompt 模板 | 无 | prompt 文本 |
| `GMLLM/llm_client.py` | OpenAI 兼容客户端 | API key、base URL | client 对象 |
| `GMLLM/exporter.py` | 写出 graph JSON | graph dict | 校验后的 JSON |
| `GMLLM/generate_graph_data_fromJson.py` | 生成图特征 | call graph JSON、配置 | vocab 和 `.pt` 文件 |
| `GMLLM/distinguish_GNN_2.py` | 基础 GNN 训练/测试 | vocab、`.pt`、配置 | `base_model.pt`、指标 |
| `GMLLM/continual_learning_memory.py` | 按月增量学习 | base model、`.pt` | 月度模型、指标 |
| `GMLLM/run_autoexplanation_parallel.py` | 可选解释模块并行启动器 | worker/GPU 参数 | worker 日志 |
| `GMLLM/auto_explanation.py` | 可选解释 worker | 外部配置和脚本 | 解释结果 |
| `GMLLM/explain.py` | 可选 GNNExplainer 逻辑 | explainer checkpoint | mask/图/日志 |
| `GMLLM/explainer_main.py` | 可选 explainer 入口 | checkpoint 参数 | 解释日志 |
| `GMLLM/direct_call_llm_local.py` | 直接调用 LLM 的基线 | direct-call 配置、源码包 | 每包 LLM JSON |
| `GMLLM/extract_qwen_results.py` | 汇总 LLM 结果 | 每包 LLM JSON | CSV |
| `GMLLM/calc_metrics.py` | 计算 LLM 检测指标 | CSV | 指标 JSON |
| `GMLLM/calc_token_stats.py` | 统计 token | CSV | token 统计 JSON |
| `GMLLM/test_single_package.py` | 单包调试 | call graph、模型、vocab | 预测结果或 `.pt` |
| `GMLLM/test_seen_months.py` | seen-month 评估实验 | 增量模型 | seen-month 指标 |
| `GMLLM/upper.py` | 累积训练基线 | 历史图数据 | 基线模型和指标 |

## 配置

主要配置文件在 `GMLLM/configs/`。

重要字段：

- `llm`：模型名、接口地址、上下文长度、合成规则相关选项。
- `dataset`：数据集根目录、良性/恶意源码目录、处理后输出目录、词表目录、call graph 文件名。
- `training`：epoch、batch size、学习率、数据划分比例、随机种子。
- `model`：GNN hidden dim、类别数、dropout。
- `continual_learning`：基础训练月份、增量月份、记忆池大小、是否使用记忆池。
- `paths`：模型目录、结果目录、预训练模型路径、checkpoint 前缀。
- `device`：`auto`、`cuda` 或 `cpu`。

当前 `configs/` 保留一个完整的 `default.yaml`，其他配置只写差异，并继续使用现有的 `parent` 继承机制：

```text
configs/
├── default.yaml
├── profiles/
│   ├── deepseek.yaml
│   ├── deepseek_no_memory.yaml
│   ├── llama2.yaml
│   └── qwen2_5.yaml
├── ablations/
│   ├── memory_20.yaml
│   ├── memory_30.yaml
│   └── no_memory.yaml
├── baselines/
│   └── upper.yaml
├── direct_call/
│   ├── deepseek.yaml
│   ├── llama2.yaml
│   └── qwen2_5.yaml
└── archive/
```

模型 profile 通常只需要覆盖这些字段：

```yaml
parent: "../default.yaml"

llm:
  model_name: "deepseek"
  base_url: "http://..."
  context_length: 4096

dataset:
  vocab_dir: "vocab_deepseek"
  benign_out: "benign_call_processed_deepseek"
  malicious_out: "malicious_call_processed_deepseek"
  call_graph_filename: "deepseek_call_graph.json"

paths:
  models_dir: "./models/deepseek"
  results_dir: "../results"
  pretrained_model: "./models/deepseek/base_model.pt"
  prefix: "incremental_"
```

模型输出统一使用：

```text
GMLLM/models/
├── default/
├── deepseek/
├── deepseek_no_memory/
├── llama2/
└── qwen2_5/
```

不要再增加 `models_deepseek/`、`models_llama2/` 这类平铺目录。增量模型命名为 `incremental_YYYY-MM.pt`。

注意：`cli_extract.py` 默认写出 `{model_name}_call_graph.json`，`generate_graph_data_fromJson.py` 会读取配置中的 `dataset.call_graph_filename`。新增 profile 时需要让这两个名称一致。

## 当前输出

仓库中已有的模型输出包括：

```text
GMLLM/models/default/base_model.pt
GMLLM/models/deepseek/base_model.pt
GMLLM/models/deepseek/incremental_2024-01.pt ... incremental_2024-12.pt
GMLLM/models/deepseek_no_memory/incremental_2024-01.pt ... incremental_2024-12.pt
GMLLM/models/llama2/base_model.pt
GMLLM/models/llama2/incremental_2024-01.pt ... incremental_2024-12.pt
```

## 最小复现实验

1. 在 `GMLLM/configs/` 中选定或修改一个配置文件，例如 `default.yaml` 或 `profiles/deepseek.yaml`。
2. 用 `cli_extract.py` 提取 call graph。
3. 用 `generate_graph_data_fromJson.py` 生成词表和 PyG 图特征。
4. 用 `distinguish_GNN_2.py` 训练基础 GNN。
5. 用 `continual_learning_memory.py` 进行按月增量学习。
6. 如需可解释性分析，在补齐外部依赖后再运行解释模块。
