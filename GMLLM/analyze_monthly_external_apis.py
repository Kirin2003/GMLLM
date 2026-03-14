#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析每个月有多少外部API不在词汇表中
支持两种模式：
1. 词汇表不动态更新：每个月直接与初始词汇表比较
2. 词汇表动态更新：词汇表累积之前所有月份的外部API，再比较当月新增
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from analyze_external_apis import extract_external_apis

MALICIOUS_PATH = "/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset/malicious"
VOCAB_PATH = "/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset/vocab/name2idx.json"

# 要分析的月份范围
START_MONTH = "2023-03"
END_MONTH = "2024-12"

def load_vocab(vocab_path):
    """加载词汇表"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return set(vocab.keys())

def get_months_between(start_month, end_month):
    """生成月份列表"""
    months = []
    current = datetime.strptime(start_month, "%Y-%m")
    end = datetime.strptime(end_month, "%Y-%m")

    while current <= end:
        months.append(current.strftime("%Y-%m"))
        # 移到下个月
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    return months

def analyze_month(month, vocab_set):
    """分析一个月的所有包的外部API"""
    month_path = os.path.join(MALICIOUS_PATH, month)

    if not os.path.exists(month_path):
        return {"month": month, "packages": 0, "total_external_apis": 0, "unknown_apis": 0, "unknown_api_set": set(), "all_external_apis": set()}

    packages = os.listdir(month_path)
    all_external_apis = set()
    packages_with_apis = 0

    for pkg in packages:
        pkg_path = os.path.join(month_path, pkg)
        if not os.path.isdir(pkg_path):
            continue

        # 尝试提取外部API
        try:
            external_apis = extract_external_apis(pkg_path)
            if external_apis:
                packages_with_apis += 1
                all_external_apis.update(external_apis)
        except Exception as e:
            # 跳过有问题的包
            continue

    # 找出不在词汇表中的API
    unknown_apis = all_external_apis - vocab_set

    return {
        "month": month,
        "packages": len(packages),
        "packages_with_apis": packages_with_apis,
        "total_external_apis": len(all_external_apis),
        "unknown_apis": len(unknown_apis),
        "unknown_api_set": unknown_apis,
        "all_external_apis": all_external_apis
    }

def main():
    print("=" * 80)
    print("分析恶意包的外部API不在词汇表中的情况")
    print("月份范围: {} ~ {}".format(START_MONTH, END_MONTH))
    print("=" * 80)

    # 加载初始词汇表
    print("\n加载初始词汇表...")
    initial_vocab = load_vocab(VOCAB_PATH)
    print("初始词汇表大小: {} 个条目".format(len(initial_vocab)))

    # 生成月份列表
    months = get_months_between(START_MONTH, END_MONTH)

    # 分析每个月
    results = []
    cumulative_vocab = set(initial_vocab)  # 累积词汇表（动态更新模式）

    for month in months:
        print("\n分析 {} ...".format(month))
        result = analyze_month(month, cumulative_vocab)  # 使用累积词汇表
        results.append(result)

        # 动态更新词汇表：加入当月所有外部API
        cumulative_vocab.update(result["all_external_apis"])

        print("  包数量: {}".format(result['packages']))
        print("  有外部API的包: {}".format(result['packages_with_apis']))
        print("  外部API总数: {}".format(result['total_external_apis']))
        print("  累积词汇表大小: {}".format(len(cumulative_vocab)))
        print("  模式1-不动态更新: 未知API = {}".format(len(result["unknown_api_set"])))

    # ========== 计算两种模式的统计 ==========
    print("\n" + "=" * 80)
    print("模式1: 词汇表不动态更新（每个月与初始词汇表比较）")
    print("=" * 80)

    # 重新计算模式1（不动态更新）
    results_static = []
    for month in months:
        result = analyze_month(month, initial_vocab)
        results_static.append(result)

    # 输出模式1表格
    print("\n{:<10} {:>10} {:>12} {:>15} {:>12}".format(
        "月份", "总包数", "有API的包", "外部API总数", "未知API数"))
    print("-" * 65)

    total_unknown_static = 0
    for r in results_static:
        print("{:<10} {:>10} {:>12} {:>15} {:>12}".format(
            r['month'], r['packages'], r['packages_with_apis'],
            r['total_external_apis'], r['unknown_apis']))
        total_unknown_static += r['unknown_apis']

    print("-" * 65)
    print("{:<10} {:>10} {:>12} {:>15} {:>12}".format(
        "总计",
        sum(r['packages'] for r in results_static),
        sum(r['packages_with_apis'] for r in results_static),
        sum(r['total_external_apis'] for r in results_static),
        total_unknown_static))

    # ========== 模式2: 动态更新 ==========
    print("\n" + "=" * 80)
    print("模式2: 词汇表动态更新（词汇表累积之前月份的外部API）")
    print("=" * 80)

    print("\n{:<10} {:>10} {:>12} {:>15} {:>12}".format(
        "月份", "总包数", "有API的包", "外部API总数", "当月新增API"))
    print("-" * 65)

    total_new_dynamic = 0
    for r in results:
        print("{:<10} {:>10} {:>12} {:>15} {:>12}".format(
            r['month'], r['packages'], r['packages_with_apis'],
            r['total_external_apis'], r['unknown_apis']))
        total_new_dynamic += r['unknown_apis']

    print("-" * 65)
    print("{:<10} {:>10} {:>12} {:>15} {:>12}".format(
        "总计",
        sum(r['packages'] for r in results),
        sum(r['packages_with_apis'] for r in results),
        sum(r['total_external_apis'] for r in results),
        total_new_dynamic))

    # ========== 对比表格 ==========
    print("\n" + "=" * 80)
    print("两种模式对比")
    print("=" * 80)

    print("\n{:<10} {:>15} {:>15} {:>15}".format(
        "月份", "模式1(不更新)", "模式2(动态更新)", "差异"))
    print("-" * 60)

    for r_static, r_dynamic in zip(results_static, results):
        diff = r_static['unknown_apis'] - r_dynamic['unknown_apis']
        print("{:<10} {:>15} {:>15} {:>15}".format(
            r_static['month'],
            r_static['unknown_apis'],
            r_dynamic['unknown_apis'],
            diff))

    print("-" * 60)
    print("{:<10} {:>15} {:>15} {:>15}".format(
        "总计",
        total_unknown_static,
        total_new_dynamic,
        total_unknown_static - total_new_dynamic))

    # ========== 保存详细结果 ==========
    output_data = {
        "initial_vocab_size": len(initial_vocab),
        "final_cumulative_vocab_size": len(cumulative_vocab),
        "months": []
    }

    for r_static, r_dynamic in zip(results_static, results):
        month_data = {
            "month": r_static["month"],
            "packages": r_static["packages"],
            "external_apis": r_static["total_external_apis"],
            "static_unknown": r_static["unknown_apis"],
            "dynamic_new": r_dynamic["unknown_apis"],
            # 模式2下当月新出的API具体列表
            "new_apis": sorted(list(r_dynamic["unknown_api_set"]))
        }
        output_data["months"].append(month_data)

    output_path = "/Data2/hxq/GMLLM/external_api_analysis_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n详细结果（包含每月新出API列表）已保存到: {}".format(output_path))

    # 打印一些新出API的示例
    print("\n" + "=" * 80)
    print("部分月份新出的外部API示例（前10个）")
    print("=" * 80)

    for month_data in output_data["months"][:6]:  # 显示前6个月
        print("\n{} (共{}个新API):".format(
            month_data["month"], len(month_data["new_apis"])))
        for api in month_data["new_apis"][:10]:
            print("  - {}".format(api))
        if len(month_data["new_apis"]) > 10:
            print("  ... 还有{}个".format(len(month_data["new_apis"]) - 10))

if __name__ == "__main__":
    main()
