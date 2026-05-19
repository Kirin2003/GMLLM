#!/usr/bin/env python3
"""分析 deepseek_synth_rules.json 和 rules_fallback.py 中行为规则的差异"""

import json

# 读取 deepseek_synth_rules.json
with open('llama-2_synth_rules.json', 'r') as f:
    deepseek_data = json.load(f)

deepseek_names = {b['name'] for b in deepseek_data['behaviors']}

# 从 rules_fallback.py 提取 BEHAVIOR_RULES 的键
# 直接执行该文件中的字典定义
exec(open('rules_fallback.py').read())
fallback_names = set(BEHAVIOR_RULES.keys())

# 分析差异
only_in_deepseek = deepseek_names - fallback_names
only_in_fallback = fallback_names - deepseek_names
common = deepseek_names & fallback_names

print("=" * 60)
print("行为规则对比分析")
print("=" * 60)
print(f"\nsynth_rules.json 中的行为数量: {len(deepseek_names)}")
print(f"rules_fallback.py 中的行为数量: {len(fallback_names)}")
print(f"共同的行为数量: {len(common)}")

print("\n" + "-" * 60)
print(f"仅在 synth_rules.json 中存在 ({len(only_in_deepseek)} 个):")
print("-" * 60)
for name in sorted(only_in_deepseek):
    print(f"  - {name}")

print("\n" + "-" * 60)
print(f"仅在 rules_fallback.py 中存在 ({len(only_in_fallback)} 个):")
print("-" * 60)
for name in sorted(only_in_fallback):
    print(f"  - {name}")

print("\n" + "-" * 60)
print(f"共同的行为 ({len(common)} 个):")
print("-" * 60)
for name in sorted(common):
    print(f"  - {name}")
