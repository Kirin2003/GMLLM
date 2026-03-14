"""
日志工具模块
"""

import os
from pathlib import Path
from datetime import datetime


# 默认日志目录
LOG_DIR = Path("./log")


class Logger:
    """日志记录器类，每个文件创建一个实例"""

    def __init__(self, log_file: str):
        """
        初始化日志记录器

        Args:
            log_file: 日志文件名（必填）
        """
        self.log_dir = LOG_DIR
        self.log_file = self.log_dir / log_file

        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        """输出日志到文件和控制台"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
