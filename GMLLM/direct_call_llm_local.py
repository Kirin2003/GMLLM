# -*- coding: utf-8 -*-
"""PyPI 包安全分析工具 - 支持单包和批量分析"""
import csv
import json
import os
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from llm_client import get_llm_client
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 添加项目根目录到 Python 路径

from utils.config_utils import load_config as load_yaml_config
from utils.logger_utils import Logger
from utils.month_utils import generate_month_range


def load_config(config_path: str) -> dict:
    """加载 direct_call_llm 配置文件"""
    return load_yaml_config(config_path)


def resolve_config_path(config_name: str) -> Path:
    """Resolve direct-call config names in the new layout, with legacy fallback."""
    base_dir = Path(__file__).parent / "configs"
    candidates = [
        base_dir / "direct_call" / f"{config_name}.yaml",
        base_dir / f"direct_call_llm_{config_name}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Direct-call config '{config_name}' not found. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="PyPI 包安全分析工具")
    parser.add_argument('--config', '-c', type=str,
                        default='qwen2_5',
                        help='配置文件名（不含路径和后缀），默认 qwen2_5')
    return parser.parse_args()


# 全局变量（由 main() 初始化）
logger = None
CONFIG = None
LLM_CONFIG = None
LLM_MODEL = None
RESULT_FILENAME = None


def get_max_source_chars(llm_config: dict, ratio: float = 0.75) -> int:
    """
    根据模型上下文长度计算源代码最大字符数

    Args:
        llm_config: LLM 配置字典
        ratio: 使用上下文长度的比例（默认 75%，留空间给 prompt 和 response）

    Returns:
        最大字符数
    """
    context_length = llm_config.get("context_length", 125000)
    # 约 1 token ≈ 4 字符
    return int(context_length * ratio * 4)


def get_all_py_files(package_path: Path) -> str:
    """
    读取 package_path 下所有的 .py 文件内容

    Args:
        package_path: 包路径

    Returns:
        拼接后的源代码字符串
    """
    global logger
    all_code = []
    for py_file in package_path.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8', errors='ignore')
            rel_path = py_file.relative_to(package_path)
            all_code.append(f"# --- File: {rel_path} ---\n")
            all_code.append(content)
            all_code.append("\n\n")
        except Exception as e:
            logger.log(f"读取文件失败 {py_file}: {e}")
    return "".join(all_code)


def query_llm_for_verdict(source_code: str) -> Tuple[Optional[str], Optional[dict], bool, Optional[float]]:
    """
    调用 LLM 获取包的判定结果

    Args:
        source_code: 包的源代码

    Returns:
        (LLM响应文本, token使用信息, 是否截断, 耗时秒) 元组
    """
    global logger, LLM_CONFIG, LLM_MODEL

    system_prompt = """You are a PyPI package security auditor. Analyze the provided PyPI package code for malicious behaviors. Respond ONLY with a valid JSON object in this exact format:
{"verdict": "Malicious" or "Benign", "reasoning": "<filename> + <malicious code/function> + <malicious behavior>"}

Examples:
{"verdict": "Malicious", "reasoning": "__init__.py + os.system() + execute remote commands"}
{"verdict": "Benign", "reasoning": "no malicious behavior found"}

Do not output anything before or after the JSON. Your response must be parseable by json.loads().
"""

    api_key = LLM_CONFIG["api_key_env"]
    base_url = LLM_CONFIG["base_url"]
    client = get_llm_client(api_key, base_url)
    truncated = False
    duration = None

    max_chars = get_max_source_chars(LLM_CONFIG)
    if len(source_code) > max_chars:
        truncated = True
        logger.log(f"源代码过长 ({len(source_code)} 字符)，截断至 {max_chars} 字符")
        source_code = source_code[:max_chars]

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": source_code}
            ],
            temperature=0.1,
            timeout=120,
        )
    except Exception as e:
        logger.log(f"LLM 调用失败: {e}")
        duration = time.time() - start_time
        return None, None, truncated, duration

    duration = time.time() - start_time
    logger.log(f"LLM 原始响应: {response}")
    msg = response.choices[0].message
    # 优先用 content，fallback 到 reasoning
    content = msg.content.strip() if msg.content else (msg.reasoning or "").strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
    logger.log(f"LLM 调用成功，耗时: {duration:.2f}s（截断: {truncated}）")
    return content, usage, truncated, duration


def parse_llm_response(response_text: str) -> Dict[str, str]:
    """
    解析 LLM 响应，提取 verdict 和 reasoning

    Args:
        response_text: LLM 原始响应

    Returns:
        包含 verdict 和 reasoning 的字典
    """
    # 尝试直接解析 JSON
    try:
        data = json.loads(response_text.strip())
        return {
            "verdict": data.get("verdict", "").strip(),
            "reasoning": data.get("reasoning", "").strip()
        }
    except json.JSONDecodeError:
        pass

    # 尝试从文本中提取 JSON（处理模型可能在 JSON 前后加了额外文本的情况）
    import re
    json_match = re.search(r'\{[^{}]*"verdict"[^{}]*"reasoning"[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "verdict": data.get("verdict", "").strip(),
                "reasoning": data.get("reasoning", "").strip()
            }
        except json.JSONDecodeError:
            pass

    # Fallback: 旧格式解析（兼容）
    try:
        lines = response_text.strip().split('\n')
        verdict = lines[0].replace("Verdict:", "").strip()
        reasoning = "\n".join(lines[1:]).replace("Reasoning:", "").strip()
    except IndexError:
        verdict = None
        reasoning = ""

    return {
        "verdict": verdict or "",
        "reasoning": reasoning or ""
    }


def analyze_package(package_path: str, output_path: Optional[str] = None) -> Dict[str, str]:
    """
    分析单个包的安全性

    Args:
        package_path: 包路径
        output_path: 输出 JSON 文件路径（可选）

    Returns:
        分析结果字典
    """
    global logger, RESULT_FILENAME
    package_path = Path(package_path)
    package_name = package_path.name

    logger.log(f"正在分析包: {package_name}")

    # 1. 读取所有 Python 源文件
    source_code = get_all_py_files(package_path)
    if not source_code:
        logger.log(f"警告: {package_name} 中没有找到 Python 文件")
        result = {
            "package_name": package_name,
            "verdict": "Error",
            "reasoning": "No Python files found in the package",
            "truncated": False,
            "duration": None,
        }
    else:
        logger.log(f"已读取源代码长度: {len(source_code)} 字符")

        # 2. 调用 LLM 分析
        llm_response, token_usage, truncated, duration = query_llm_for_verdict(source_code)
        if not llm_response:
            result = {
                "package_name": package_name,
                "verdict": "Error",
                "reasoning": "Failed to get response from LLM",
                "truncated": truncated,
                "duration": duration,
            }
        else:
            logger.log(f"LLM 响应: {llm_response[:200]}...")
            if token_usage:
                logger.log(f"Token 使用: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}, total={token_usage['total_tokens']}")

            # 3. 解析响应
            parsed = parse_llm_response(llm_response)
            result = {
                "package_name": package_name,
                "verdict": parsed["verdict"],
                "reasoning": parsed["reasoning"],
                "prompt_tokens": token_usage.get("prompt_tokens") if token_usage else None,
                "completion_tokens": token_usage.get("completion_tokens") if token_usage else None,
                "total_tokens": token_usage.get("total_tokens") if token_usage else None,
                "truncated": truncated,
                "duration": round(duration, 2) if duration else None,
            }

    # 4. 输出结果
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        logger.log(f"结果已保存到: {output_path}")
    else:
        output_path = f"{package_name}_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        logger.log(f"结果已保存到: {output_path}")

    return result


def save_to_csv(results: List[Dict], csv_path: str):
    """
    将结果保存到 CSV 文件

    Args:
        results: 分析结果列表
        csv_path: CSV 文件路径
    """
    csv_file = Path(csv_path)
    fieldnames = ["package_name", "verdict", "reasoning", "prompt_tokens",
                  "completion_tokens", "total_tokens", "month", "type", "truncated", "duration"]

    # 检查文件是否存在，决定是追加还是新建
    file_exists = csv_file.exists()
    mode = 'a' if file_exists else 'w'

    with open(csv_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for r in results:
            row = {
                "package_name": r.get("package_name", ""),
                "verdict": r.get("verdict", ""),
                "reasoning": r.get("reasoning", ""),
                "prompt_tokens": r.get("prompt_tokens", ""),
                "completion_tokens": r.get("completion_tokens", ""),
                "total_tokens": r.get("total_tokens", ""),
                "month": r.get("month", ""),
                "type": r.get("package_type", ""),
                "truncated": r.get("truncated", ""),
                "duration": r.get("duration", ""),
            }
            writer.writerow(row)

    logger.log(f"CSV 已保存到: {csv_path}")


def batch_analyze_packages(dataset_base: str, max_workers: int = 4):
    """
    批量分析数据集下的所有包（并发版本）

    Args:
        dataset_base: 数据集根目录
        max_workers: 并发数，默认 8
    """
    global logger, CONFIG, RESULT_FILENAME
    base_path = Path(dataset_base)

    # 统计信息
    total_packages = 0
    processed = 0
    skipped = 0

    # 收集所有待处理任务
    pending_tasks = []  # (package_dir, json_path, month, package_type, package_name)

    # allowed_months = generate_month_range("2023-03", "2024-12")
    allowed_months = generate_month_range(CONFIG["start_month"], CONFIG["end_month"])
    # for package_type in ["malicious", "benign"]:
    for package_type in ["benign", "malicious"]: # TODO 先只处理 malicious，后续再处理 benign
        type_dir = base_path / package_type

        for month in allowed_months:
            logger.log(f"扫描目录: {month} ({package_type})")
            month_dir = type_dir / month

            for package_dir in sorted(month_dir.iterdir(), key=lambda p: p.name.lower()):
                if not package_dir.is_dir():
                    continue

                total_packages += 1
                package_name = package_dir.name
                json_path = package_dir / RESULT_FILENAME

                # 检查是否已处理
                if json_path.exists():
                    logger.log(f"跳过已处理: {package_name} ({month}/{package_type})")
                    skipped += 1
                    continue

                pending_tasks.append((package_dir, json_path, month, package_type, package_name))

    logger.log(f"总计: {total_packages}, 待处理: {len(pending_tasks)}, 跳过: {skipped}")

    # 并发执行
    def process_task(package_dir, json_path, month, package_type):
        """处理单个包的任务"""
        result = analyze_package(str(package_dir), str(json_path))
        result["month"] = month
        result["package_type"] = package_type
        # 重新保存包含 month 和 package_type 的结果
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        return result

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for package_dir, json_path, month, package_type, package_name in pending_tasks:
            logger.log(f"提交任务: {package_name} ({month}/{package_type})")
            future = executor.submit(process_task, package_dir, json_path, month, package_type)
            futures[future] = package_name

        for future in as_completed(futures):
            package_name = futures[future]
            try:
                future.result()
                processed += 1
                if processed % 10 == 0:
                    logger.log(f"进度: 已处理 {processed}/{len(pending_tasks)}")
            except Exception as e:
                logger.log(f"处理失败 {package_name}: {e}")

    logger.log(f"完成! 总计: {total_packages}, 已处理: {processed}, 跳过: {skipped}")


def main():
    """批量分析数据集"""
    global logger, CONFIG, LLM_CONFIG, LLM_MODEL, RESULT_FILENAME

    args = parse_args()
    _config_name = args.config
    _config_file = resolve_config_path(_config_name)
    CONFIG = load_config(str(_config_file))

    logger = Logger(CONFIG["log_file"])

    # 加载 LLM 配置（内联在 YAML 中）
    LLM_CONFIG = CONFIG["llm"]
    LLM_MODEL = LLM_CONFIG["model"]
    RESULT_FILENAME = CONFIG["result_filename"]

    # 调试日志：确认使用的模型和 base_url
    logger.log(f"[DEBUG] LLM 配置: model={LLM_MODEL}")
    logger.log(f"[DEBUG] base_url: {LLM_CONFIG.get('base_url', 'NOT FOUND')}")

    dataset_base = os.path.expandvars(CONFIG.get("dataset_base", "${DATA_ROOT}/datasets/incremental_packages_dynamic_capping_subset"))
    batch_analyze_packages(dataset_base)


if __name__ == "__main__":
    main()
