"""
项目路径配置
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 日志目录
LOG_DIR = PROJECT_ROOT / "log"

# 结果目录
RESULTS_DIR = PROJECT_ROOT / "results"
