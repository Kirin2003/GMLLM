#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Qwen 批量分析新增外部API的恶意性，并生成合成规则更新 synth_rules.json
"""
import sys
from pathlib import Path

# 将上级目录加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import time
import re
from pathlib import Path
from openai import OpenAI
from rules_fallback import BEHAVIOR_RULES
from utils.logger_utils import Logger

# 获取所有恶意行为标签
ALL_BEHAVIOR_TAGS = list(BEHAVIOR_RULES.keys())

# 全局日志对象
log = Logger("analyze_api_malicious.log")

# 批量分析时的prompt（英文）
BATCH_PROMPT = """You are a malware analysis expert. Analyze the following Python API calls and determine if they could be used for malicious purposes.

Available malicious behavior tags (choose from these):
{behaviors}

API list (one per line):
{apis}

Only return APIs that have malicious behaviors. Return analysis results in JSON format as an array:
[
    {{
        "name": "<API_NAME>",
        "malicious_behaviors": ["behavior1", "behavior2"],
        "reason": "analysis reason"
    }},
    ...
]

If an API has no malicious behaviors, do not include it in the response. Return empty array [] if all APIs are benign.

Only return JSON array, nothing else."""

def load_api_results(results_path):
    """加载之前分析的结果"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_synth_rules(rules_path):
    """加载现有的 synth_rules.json"""
    with open(rules_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_existing_behavior_names(rules_data):
    """从 synth_rules.json 中提取已有的行为名称"""
    return set(behavior['name'] for behavior in rules_data.get('behaviors', []))

def init_qwen_client():
    """初始化 qwen 客户端"""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        log.log("警告: 未设置 OPENAI_API_KEY 环境变量")
        return None

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return client

def analyze_batch_apis(client, apis, model="qwen-max"):
    """批量分析多个API的恶意性"""
    if client is None:
        return {api: {
            "name": api,
            "malicious_behaviors": [],
            "reason": "No API key"
        } for api in apis}

    # 构建prompt
    apis_str = "\n".join(f"- {api}" for api in apis)
    behaviors_str = ", ".join(ALL_BEHAVIOR_TAGS)

    prompt = BATCH_PROMPT.format(apis=apis_str, behaviors=behaviors_str)

    # 记录输入给LLM的API列表
    log.log(f"[LLM Input] APIs: {apis}")
    log.log(f"[LLM Input] Prompt: {prompt[:500]}...")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a malware analysis expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            timeout=60.0
        )

        result_text = response.choices[0].message.content or ""

        # 记录LLM的输出
        log.log(f"[LLM Output] {result_text[:1000]}...")

        # 初始化结果：所有API默认无恶意行为
        results = {api: {
            "name": api,
            "malicious_behaviors": [],
            "reason": ""
        } for api in apis}

        # 解析JSON结果
        try:
            parsed = json.loads(result_text)
            # LLM只返回有恶意行为的API，更新结果
            for item in parsed:
                if "name" in item:
                    results[item["name"]] = item
        except json.JSONDecodeError:
            # 尝试提取JSON数组部分
            match = re.search(r'\[[\s\S]*\]', result_text)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    for item in parsed:
                        if "name" in item:
                            results[item["name"]] = item
                except:
                    pass

            log.log(f"    解析失败，返回: {result_text[:200]}")

        return results

    except Exception as e:
        return {api: {
            "name": api,
            "malicious_behaviors": [],
            "reason": str(e)
        } for api in apis}

def generate_rule_for_api(api, behaviors):
    """为API和行为生成 synth_rules.json 格式的规则"""
    parts = api.split('.')
    if len(parts) >= 2:
        # 根据API模式生成规则
        if len(parts) == 2:
            rule = f"lambda n: n.startswith(('{parts[0]}.{parts[1]}',))"
        elif len(parts) == 3:
            rule = f"lambda n: n.startswith(('{parts[0]}.{parts[1]}.{parts[2]}',))"
        else:
            prefix = '.'.join(parts[:-1]) + '.'
            rule = f"lambda n: n.startswith(('{prefix}',))"

        why = f"This API ({api}) is associated with {behaviors[0]} behavior in malware."

        return {
            "name": behaviors[0] if behaviors else "unknown",
            "why": why,
            "rule": rule
        }
    return None

def analyze_month_apis(client, apis, existing_behaviors, model="qwen-max", max_apis=None, batch_size=10):
    """批量分析一个月的所有API"""
    apis_to_analyze = apis[:max_apis] if max_apis else apis

    results = {}
    new_rules_suggestions = []

    # 分批处理
    for i in range(0, len(apis_to_analyze), batch_size):
        batch = apis_to_analyze[i:i+batch_size]
        log.log(f"  批量分析 [{i+1}-{min(i+batch_size, len(apis_to_analyze))}/{len(apis_to_analyze)}]...")

        batch_results = analyze_batch_apis(client, batch, model)
        results.update(batch_results)

        # 处理每个结果
        for api, result in batch_results.items():
            behaviors = result.get("malicious_behaviors", [])
            if behaviors:  # 有恶意行为
                log.log(f"    -> 恶意: {api[:40]}... 行为: {behaviors}")

                # 检查是否是 synth_rules.json 中没有的新行为
                for behavior in behaviors:
                    if behavior not in existing_behaviors:
                        rule_suggestion = generate_rule_for_api(api, [behavior])
                        if rule_suggestion:
                            new_rules_suggestions.append({
                                "api": api,
                                "behavior": behavior,
                                "rule": rule_suggestion
                            })
                            log.log(f"    -> 检测到新行为: {behavior}")

        time.sleep(0.5)  # 避免API调用过于频繁

    return results, new_rules_suggestions

def main():
    results_path = "external_api_analysis_results.json"
    rules_path = "synth_rules.json"
    output_path = "/Data2/hxq/GMLLM/api_malicious_analysis_results.json"
    new_rules_path = "/Data2/hxq/GMLLM/new_rules_suggestions.json"

    log.log("=" * 80)
    log.log("使用 Qwen 批量分析新增外部API的恶意性")
    log.log("=" * 80)
    log.log(f"恶意行为标签 ({len(ALL_BEHAVIOR_TAGS)}): {ALL_BEHAVIOR_TAGS[:5]}...")

    # 加载数据
    log.log("\n加载API分析结果...")
    data = load_api_results(results_path)

    # 加载现有的 synth_rules.json
    log.log("\n加载现有的 synth_rules.json...")
    rules_data = load_synth_rules(rules_path)
    existing_behaviors = get_existing_behavior_names(rules_data)
    log.log(f"已有行为数量: {len(existing_behaviors)}")

    # 初始化 qwen 客户端
    log.log("\n初始化 Qwen 客户端...")
    client = init_qwen_client()
    if client is None:
        log.log("错误: 无法初始化 Qwen 客户端，请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    # 按月分析API
    all_malicious_apis = []
    all_results = {}
    all_new_rules = []

    for month_data in data["months"]:
        month = month_data["month"]
        new_apis = month_data["new_apis"]

        log.log("\n" + "=" * 60)
        log.log(f"月份: {month} (共{len(new_apis)}个新API)")
        log.log("=" * 60)

        month_results, new_rules = analyze_month_apis(
            client,
            new_apis,
            existing_behaviors,
            model="qwen-max",
            batch_size=10  # 每次分析10个API
        )

        all_results.update(month_results)
        all_new_rules.extend(new_rules)

        # 统计当月恶意API
        month_malicious = [(api, r) for api, r in month_results.items() if r.get("malicious_behaviors")]
        log.log(f"\n{month}月恶意API统计:")
        log.log(f"  总分析数: {len(month_results)}")
        log.log(f"  恶意API数: {len(month_malicious)}")

        all_malicious_apis.extend([(month, api, r) for api, r in month_malicious])

        # 只保存有恶意行为的API
        malicious_only = {api: r for api, r in all_results.items() if r.get("malicious_behaviors")}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(malicious_only, f, indent=2, ensure_ascii=False)
        break # 目前只分析第一个月，后续可以去掉这个break继续分析其他月份

    # # 最终统计
    # log.log("\n" + "=" * 80)
    # log.log("最终统计")
    # log.log("=" * 80)

    # malicious_count = sum(1 for r in all_results.values() if r.get("is_malicious"))

    # log.log(f"总分析API数: {len(all_results)}")
    # log.log(f"恶意API数: {malicious_count} ({malicious_count/len(all_results)*100:.1f}%)")

    # # 按月份统计
    # log.log("\n按月份统计恶意API:")
    # month_stats = {}
    # for month, api, result in all_malicious_apis:
    #     if month not in month_stats:
    #         month_stats[month] = 0
    #     month_stats[month] += 1

    # for month in sorted(month_stats.keys()):
    #     log.log(f"  {month}: {month_stats[month]}个恶意API")

    # # 新规则建议
    # log.log("\n" + "=" * 80)
    # log.log("新增规则建议 (用于更新 synth_rules.json)")
    # log.log("=" * 80)

    # # 去重规则
    # unique_rules = {}
    # for item in all_new_rules:
    #     key = (item["behavior"], item["rule"]["rule"])
    #     if key not in unique_rules:
    #         unique_rules[key] = item

    # unique_rules_list = list(unique_rules.values())
    # log.log(f"去重后规则数: {len(unique_rules_list)}")

    # # 保存新规则建议
    # new_rules_output = {
    #     "summary": {
    #         "total_analyzed": len(all_results),
    #         "malicious_count": malicious_count,
    #         "new_rules_suggested": len(unique_rules_list)
    #     },
    #     "new_rules": unique_rules_list
    # }

    # with open(new_rules_path, 'w', encoding='utf-8') as f:
    #     json.dump(new_rules_output, f, indent=2, ensure_ascii=False)

    # log.log(f"\n新规则建议已保存到: {new_rules_path}")

    # # 打印示例
    # log.log("\n新规则示例:")
    # for item in unique_rules_list[:10]:
    #     log.log(f"  行为: {item['behavior']}")
    #     log.log(f"    API: {item['api']}")
    #     log.log(f"    规则: {item['rule']['rule']}")
    #     log.log()

    # # 保存最终结果
    # final_output = {
    #     "summary": {
    #         "total_analyzed": len(all_results),
    #         "malicious_count": malicious_count,
    #         "new_rules_suggested": len(unique_rules_list)
    #     },
    #     "detailed_results": all_results,
    #     "malicious_apis_by_month": {
    #         month: [(api, r) for m, api, r in all_malicious_apis if m == month]
    #         for month in set(m for m, _, _ in all_malicious_apis)
    #     },
    #     "new_rules_suggestions": unique_rules_list
    # }

    # with open(output_path, 'w', encoding='utf-8') as f:
    #     json.dump(final_output, f, indent=2, ensure_ascii=False)

    # log.log(f"\n详细结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
