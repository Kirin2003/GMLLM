# -*- coding: utf-8 -*-
"""PyPI 包安全分析工具 - 支持单包和批量分析"""
import csv
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from llm_client import get_llm_client, get_model_name

# 加载 LLM 配置
config_path = Path(__file__).parent / "configs" / "llm_config.json"
with open(config_path, 'r') as f:
    LLM_CONFIG = json.load(f)

LLM_PROVIDER = LLM_CONFIG.get('provider', 'qwen')
LLM_MODEL = get_model_name(LLM_PROVIDER, LLM_CONFIG)

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
            print(f"读取文件失败 {py_file}: {e}")
    return "".join(all_code)


def query_llm_for_verdict(source_code: str) -> Tuple[Optional[str], Optional[dict]]:
    """
    调用 LLM 获取包的判定结果

    Args:
        source_code: 包的源代码

    Returns:
        (LLM响应文本, token使用信息) 元组
    """
    client = get_llm_client(LLM_PROVIDER, LLM_MODEL)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt_a},
                {"role": "user", "content": source_code}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        content = response.choices[0].message.content.strip()
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        return content, usage
    except Exception as e:
        print(f"LLM 调用失败: {e}")
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

    print(f"正在分析包: {package_name}")

    # 1. 读取所有 Python 源文件
    source_code = get_all_py_files(package_path)
    if not source_code:
        print(f"警告: {package_name} 中没有找到 Python 文件")
        result = {
            "package_name": package_name,
            "verdict": "Error",
            "reasoning": "No Python files found in the package"
        }
    else:
        print(f"已读取源代码长度: {len(source_code)} 字符")

        # 2. 调用 LLM 分析
        llm_response, token_usage = query_llm_for_verdict(source_code)
        if not llm_response:
            result = {
                "package_name": package_name,
                "verdict": "Error",
                "reasoning": "Failed to get response from LLM"
            }
        else:
            print(f"LLM 响应: {llm_response[:200]}...")
            if token_usage:
                print(f"Token 使用: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}, total={token_usage['total_tokens']}")

            # 3. 解析响应
            parsed = parse_llm_response(llm_response)
            result = {
                "package_name": package_name,
                "verdict": parsed["verdict"],
                "reasoning": parsed["reasoning"],
                "token_usage": token_usage
            }

    # 4. 输出结果
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"结果已保存到: {output_path}")
    else:
        output_path = f"{package_name}_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"结果已保存到: {output_path}")

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
            token_usage = r.get("token_usage") or {}
            row = {
                "package_name": r.get("package_name", ""),
                "verdict": r.get("verdict", ""),
                "reasoning": r.get("reasoning", ""),
                "prompt_tokens": token_usage.get("prompt_tokens", ""),
                "completion_tokens": token_usage.get("completion_tokens", ""),
                "total_tokens": token_usage.get("total_tokens", ""),
                "month": r.get("month", ""),
                "type": r.get("package_type", "")
            }
            writer.writerow(row)

    print(f"CSV 已保存到: {csv_path}")


def batch_analyze_packages(dataset_base: str, output_csv: str):
    """
    批量分析数据集下的所有包

    Args:
        dataset_base: 数据集根目录
        output_csv: 输出 CSV 文件路径
    """
    base_path = Path(dataset_base)
    results = []

    # 统计信息
    total_packages = 0
    processed = 0
    skipped = 0

    # 先统计总数
    for package_type in ["malicious", "benign"]:
        type_dir = base_path / package_type
        if not type_dir.exists():
            continue
        for month_dir in type_dir.iterdir():
            if not month_dir.is_dir():
                continue
            packages = [p for p in month_dir.iterdir() if p.is_dir()]
            total_packages += len(packages)

    print(f"总共需要处理 {total_packages} 个包")

    # 遍历处理
    for package_type in ["malicious", "benign"]:
        type_dir = base_path / package_type
        if not type_dir.exists():
            continue

        for month_dir in sorted(type_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            month = month_dir.name

            for package_dir in month_dir.iterdir():
                if not package_dir.is_dir():
                    continue

                package_name = package_dir.name
                json_path = package_dir / "analysis.json"

                # 检查是否已处理
                if json_path.exists():
                    print(f"跳过已处理: {package_name} ({month}/{package_type})")
                    skipped += 1
                    # 读取已有的结果用于 CSV
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            existing_result = json.load(f)
                            existing_result["month"] = month
                            existing_result["package_type"] = package_type
                            results.append(existing_result)
                    except:
                        pass
                    continue

                print(f"处理中: {package_name} ({month}/{package_type})")
                processed += 1

                try:
                    # 分析包，结果保存到包目录下的 analysis.json
                    result = analyze_package(str(package_dir), str(json_path))
                    result["month"] = month
                    result["package_type"] = package_type
                    results.append(result)
                except Exception as e:
                    print(f"处理失败 {package_name}: {e}")
                    results.append({
                        "package_name": package_name,
                        "verdict": "Error",
                        "reasoning": str(e),
                        "month": month,
                        "package_type": package_type,
                        "token_usage": {}
                    })

    # 保存 CSV
    if results:
        save_to_csv(results, output_csv)

    print(f"\n完成! 总计: {total_packages}, 已处理: {processed}, 跳过: {skipped}")


def main():
    """批量分析数据集"""
    dataset_base = "/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset"
    output_csv = "results/summary.csv"
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    batch_analyze_packages(dataset_base, output_csv)


if __name__ == "__main__":
    main()
