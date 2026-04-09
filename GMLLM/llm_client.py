# -*- coding: utf-8 -*-
"""统一的 LLM 客户端初始化模块"""
from typing import Any

def get_llm_client(api_key: str, base_url: str) -> Any:
    """
    创建 OpenAI 兼容的 LLM 客户端

    Args:
        api_key: API 密钥
        base_url: API 基础 URL

    Returns:
        配置好的 OpenAI 客户端对象
    """
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)
