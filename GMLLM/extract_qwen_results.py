# -*- coding: utf-8 -*-
"""从qwen3_5_27b.json提取结果到CSV"""
import csv
import json
from pathlib import Path


def extract_qwen_results_to_csv(dataset_base: str, output_csv: str, json_filename: str = "qwen3_5_27b.json"):
    """
    遍历所有包目录下的指定JSON文件，提取信息并保存到CSV

    Args:
        dataset_base: 数据集根目录
        output_csv: 输出CSV文件路径
        json_filename: JSON文件名
    """
    base_path = Path(dataset_base)
    results = []

    # 遍历所有指定的JSON文件
    for json_path in base_path.rglob(json_filename):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 从路径中提取month和package_type
            # 路径格式: {dataset_base}/{package_type}/{month}/{package_name}/{json_filename}
            parts = json_path.parts
            if len(parts) >= 4:
                package_type = parts[-4]  # benign 或 malicious
                month = parts[-2]         # YYYY-MM格式

                results.append({
                    "package_name": data.get("package_name", ""),
                    "month": data.get("month", month),  # 优先用json中的month
                    "verdict": data.get("verdict", ""),
                    "package_type": data.get("package_type", package_type),  # 优先用json中的package_type
                    "prompt_tokens": data.get("prompt_tokens", ""),
                    "completion_tokens": data.get("completion_tokens", ""),
                    "total_tokens": data.get("total_tokens", "")
                })
        except Exception as e:
            print(f"Error processing {json_path}: {e}")

    # 按类别、月份、文件名首字母排序（不区分大小写）
    results.sort(key=lambda x: (
        x.get("package_type", ""),
        x.get("month", ""),
        x.get("package_name", "").lower()
    ))

    # 保存到CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["package_name", "month", "verdict", "package_type",
                                                "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writeheader()
        writer.writerows(results)

    print(f"已提取 {len(results)} 条记录到 {output_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从JSON文件提取LLM分析结果到CSV")
    parser.add_argument('--model', '-m', type=str, required=True,
                        choices=['qwen2_5', 'deepseek', 'llama2'],
                        help='模型名称')
    args = parser.parse_args()

    dataset_base = "/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset"
    output_csv = str(Path(__file__).resolve().parent.parent / "results" / f"direct_call_llm_local_{args.model}.csv")
    json_filename = f"{args.model}.json"
    extract_qwen_results_to_csv(dataset_base, output_csv, json_filename)
