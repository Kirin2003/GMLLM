"""配置加载工具，支持 parent 继承机制"""

import yaml
from pathlib import Path
from typing import Dict, Any


def deep_merge(base: Dict, override: Dict) -> Dict:
    """深度合并两个字典，override 的值会覆盖 base 的值"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件，支持 parent 继承机制

    如果配置文件中包含 'parent' 字段，会先加载 parent 配置文件，
    然后与当前配置进行深度合并。当前配置会覆盖 parent 的配置。

    Usage:
        config = load_config("configs/CL_unk.yaml")

    Example:
        # default.yaml
        llm:
          model_name: "qwen3-max"
        training:
          epochs: 60

        # CL_unk.yaml
        parent: "default.yaml"
        training:
          epochs: 30  # 覆盖为30

        # 结果: 合并后的配置
    """
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # 检查是否有 parent 配置
    parent_path = config.pop('parent', None)

    if parent_path:
        # 解析 parent 路径（相对于当前配置文件目录）
        parent_file = (config_path.parent / parent_path).resolve()
        parent_config = load_config(str(parent_file))
        # 深度合并：parent 作为基础，当前配置覆盖
        config = deep_merge(parent_config, config)

    return config


def load_config_with_defaults(config_path: str, default_config: str = None) -> Dict[str, Any]:
    """
    加载配置，可选指定 default_config 作为默认基础配置

    加载顺序（后者覆盖前者）：
    1. default_config (如果指定)
    2. config 中的 parent 配置
    3. config 自身
    """
    config_path = Path(config_path)

    # 首先加载指定的基础配置
    if default_config:
        default_file = (config_path.parent / default_config).resolve()
        config = load_config(str(default_file))
    else:
        config = {}

    # 加载当前配置（可能包含自己的 parent）
    current_config = load_config(str(config_path))

    # 再次合并，确保当前配置优先级最高
    config = deep_merge(config, current_config)

    return config
