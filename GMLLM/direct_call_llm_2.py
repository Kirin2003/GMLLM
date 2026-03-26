# -*- coding: utf-8 -*-
"""PyPI 包安全分析工具 - 支持单包和批量分析"""
import csv
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from llm_client import get_llm_client, get_model_name
import sys
sys.path.insert(0, "/Data2/hxq/GMLLM")  # 添加项目根目录到 Python 路径

from utils.logger_utils import Logger
from utils.month_utils import generate_month_range

logger = Logger("direct_call_llm_2.log")

# 加载 LLM 配置
config_path = Path(__file__).parent / "configs" / "llm_config.json"
with open(config_path, 'r') as f:
    LLM_CONFIG = json.load(f)

LLM_PROVIDER = LLM_CONFIG.get('provider', 'qwen3.5-27b')
LLM_MODEL = get_model_name(LLM_PROVIDER, LLM_CONFIG)

# 调试日志：确认使用的模型和 base_url
logger.log(f"[DEBUG] LLM 配置: provider={LLM_PROVIDER}, model={LLM_MODEL}")
logger.log(f"[DEBUG] base_url: {LLM_CONFIG.get(LLM_PROVIDER, {}).get('base_url', 'NOT FOUND')}")

system_prompt_a = """You are a PyPI package security auditor.
You have been provided with the code of a PyPI package script.
Please carefully analyze the possible malicious behaviors therein and answer the following:
Is this code indicative of potential malicious activity? (Respond only with 'Malicious' or 'Benign')
Provide your reasoning. Keep it as simple as possible.
Response Format (must be in separate lines, use this exact format):
Verdict: <Malicious or Benign>
Reasoning: <filename> + <malicious code/function> + <malicious behavior>
Example: Reasoning: __init__.py + os.system() + execute remote commands
"""


def get_all_py_files(package_path: Path) -> str:
    """
    读取 package_path 下所有的 .py 文件内容

    Args:
        package_path: 包路径

    Returns:
        拼接后的源代码字符串
    """
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


def query_llm_for_verdict(source_code: str) -> Tuple[Optional[str], Optional[dict]]:
    """
    调用 LLM 获取包的判定结果

    Args:
        source_code: 包的源代码

    Returns:
        (LLM响应文本, token使用信息) 元组
    """
    client = get_llm_client(LLM_PROVIDER, LLM_MODEL, LLM_CONFIG)

    max_chars = 262144  # 根据模型上下文限制

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt_a},
                {"role": "user", "content": source_code[:max_chars]}
            ],
            temperature=0.1,
            max_tokens=1024,
            # 【关键修改】通过 extra_body 传入 enable_thinking=False 关闭推理模式
            extra_body={
                "enable_thinking": False
            }
        )
        content = response.choices[0].message.content.strip()
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        return content, usage
    except Exception as e:
        error_msg = str(e)
        logger.log(f"LLM 调用失败: {e}")
        # 检测额度耗尽错误 (403 AllocationQuota.FreeTierOnly)，重新抛出让上层处理
        if "403" in error_msg and ("Quota" in error_msg or "FreeTierOnly" in error_msg or "AllocationQuota" in error_msg):
            raise
        return None, None


def parse_llm_response(response_text: str) -> Dict[str, str]:
    """
    解析 LLM 响应，提取 verdict 和 reasoning

    Args:
        response_text: LLM 原始响应

    Returns:
        包含 verdict 和 reasoning 的字典
    """
    try:
        lines = response_text.strip().split('\n')
        verdict = lines[0].replace("Verdict:", "").strip()
        reasoning = "\n".join(lines[1:]).replace("Reasoning:", "").strip()
    except IndexError:
        verdict = None
        reasoning = ""

    return {
        "verdict": verdict,
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
            "reasoning": "No Python files found in the package"
        }
    else:
        logger.log(f"已读取源代码长度: {len(source_code)} 字符")

        # 2. 调用 LLM 分析
        llm_response, token_usage = query_llm_for_verdict(source_code)
        if not llm_response:
            result = {
                "package_name": package_name,
                "verdict": "Error",
                "reasoning": "Failed to get response from LLM"
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
                  "completion_tokens", "total_tokens", "month", "type"]

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
                "type": r.get("package_type", "")
            }
            writer.writerow(row)

    logger.log(f"CSV 已保存到: {csv_path}")


def batch_analyze_packages(dataset_base: str):
    """
    批量分析数据集下的所有包

    Args:
        dataset_base: 数据集根目录
    """
    import time

    base_path = Path(dataset_base)

    # 统计信息
    total_packages = 0
    processed = 0
    skipped = 0

    # 先统计总数（只统计 2023-03 到 2024-12 的月份）
    # allowed_months = generate_month_range("2023-03", "2024-12")
    allowed_months = generate_month_range("2023-03", "2023-03") # TODO 先只处理到 2023-03，后续再处理后续月份
    # for package_type in ["malicious", "benign"]:
    for package_type in ["malicious"]: # TODO 先只处理 malicious，后续再处理 benign
        type_dir = base_path / package_type
        if not type_dir.exists():
            continue
        for month_dir in type_dir.iterdir():
            if not month_dir.is_dir():
                continue
            if month_dir.name not in allowed_months:
                continue
            packages = [p for p in month_dir.iterdir() if p.is_dir()]
            total_packages += len(packages)

    logger.log(f"总共需要处理 {total_packages} 个包")

    # 遍历处理
    # for package_type in ["malicious", "benign"]:
    for package_type in ["malicious"]: # TODO 先只处理 malicious，后续再处理 benign
        type_dir = base_path / package_type
        if not type_dir.exists():
            continue

        for month_dir in sorted(type_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            month = month_dir.name

            # 只处理 2023-03 到 2024-12 的月份
            # allowed_months = generate_month_range("2023-03", "2024-12")
            allowed_months = generate_month_range("2023-03", "2023-03") # TODO 先只处理到 2023-03，后续再处理后续月份
            if month not in allowed_months:
                continue

            for package_dir in sorted(month_dir.iterdir()):
                if not package_dir.is_dir():
                    continue

                package_name = package_dir.name
                json_path = package_dir / "qwen3_5_27b.json"

                # 检查是否已处理
                if json_path.exists():
                    logger.log(f"跳过已处理: {package_name} ({month}/{package_type})")
                    skipped += 1
                    continue

                logger.log(f"处理中: {package_name} ({month}/{package_type})")
                processed += 1

                try:
                    # 分析包，结果保存到包目录下的 qwen3_5_27b.json
                    result = analyze_package(str(package_dir), str(json_path))
                    result["month"] = month
                    result["package_type"] = package_type

                    # 重新保存包含 month 和 package_type 的结果
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=4, ensure_ascii=False)

                    # 每成功分析一个包后休息10秒
                    time.sleep(10)

                except Exception as e:
                    error_msg = str(e)
                    # 检测额度耗尽错误 (403 AllocationQuota.FreeTierOnly)
                    if "403" in error_msg and ("Quota" in error_msg or "FreeTierOnly" in error_msg or "AllocationQuota" in error_msg):
                        logger.log(f"额度耗尽，停止批量分析: {error_msg}")
                        logger.log(f"完成! 总计: {total_packages}, 已处理: {processed}, 跳过: {skipped}")
                        return

                    logger.log(f"处理失败 {package_name}: {error_msg}")

    logger.log(f"完成! 总计: {total_packages}, 已处理: {processed}, 跳过: {skipped}")


def main():
    """批量分析数据集"""
    dataset_base = "/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset"
    batch_analyze_packages(dataset_base)


if __name__ == "__main__":
    main()
