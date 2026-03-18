#!/usr/bin/env python
"""测试配置继承功能"""

import sys
sys.path.insert(0, '.')
from utils.config_utils import load_config

# 测试加载 CL_unk.yaml
config = load_config('GMLLM/configs/CL_unk.yaml')

# 打印关键参数
print('=== 测试配置继承 ===')
model_cfg = config.get('model', {})
train_cfg = config.get('training', {})
cl_cfg = config.get('continual_learning', {})
results_cfg = config.get('results', {})
llm_cfg = config.get('llm', {})

print(f'model.hidden_dim: {model_cfg.get("hidden_dim")}')  # 来自 parent
print(f'training.epochs: {train_cfg.get("epochs")}')        # 来自 parent
print(f'continual_learning.memory_per_month: {cl_cfg.get("memory_per_month")}')  # 覆盖
print(f'continual_learning.use_memory: {cl_cfg.get("use_memory")}')  # 覆盖
print(f'results.future_month: {results_cfg.get("future_month")}')  # 覆盖
print(f'llm.model_name: {llm_cfg.get("model_name")}')  # 来自 parent
print()
print('=== 继承成功! ===')
