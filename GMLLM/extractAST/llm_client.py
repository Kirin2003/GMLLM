# -*- coding: utf-8 -*-
"""统一的 LLM 客户端初始化模块"""
from typing import Any, Optional
import os

PROVIDER_BASE_URLS = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai": "https://api.openai.com/v1",
}


def get_llm_client(provider: str = "qwen", model: Optional[str] = None) -> Any:
    """
    根据 provider 创建 LLM 客户端

    Args:
        provider: "qwen" | "openai" | "azure" | "local"
        model: 模型名称（可选）

    Returns:
        配置好的 OpenAI/AzureOpenAI 客户端对象
    """
    if provider not in ("qwen", "openai"):
        raise ValueError(f"Unknown provider: {provider}")
    
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = PROVIDER_BASE_URLS.get(provider)
    return OpenAI(api_key=api_key, base_url=base_url)

    


def get_model_name(provider: str, config: dict) -> str:
    """
    从配置中获取当前 provider 对应的模型名称

    Args:
        provider: 当前使用的 provider
        config: 完整的 LLM 配置字典

    Returns:
        模型名称字符串
    """
    if provider in config and "model" in config[provider]:
        return config[provider]["model"]
    return None
