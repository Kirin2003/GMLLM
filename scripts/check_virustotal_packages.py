#!/usr/bin/env python3
"""
检查virustotal_scan.csv中的benign包是否都在数据集中存在
"""
import csv
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils.config_utils import load_dotenv

load_dotenv()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/Data2/hxq"))

# 配置路径
CSV_PATH = str(project_root / "results" / "virustotal_scan.csv")
DATA_DIR = str(DATA_ROOT / "datasets" / "incremental_packages_dynamic_capping_subset_compressed" / "benign")

# 读取CSV中的benign包
csv_packages = {}  # {month: set(packages)}
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['label'] == 'benign':
            month = row['month']
            package = row['package']
            if month not in csv_packages:
                csv_packages[month] = set()
            csv_packages[month].add(package)

print("=== CSV中的月份和benign包数量 ===")
for month in sorted(csv_packages.keys()):
    print(f"  {month}: {len(csv_packages[month])} 个包")

# 检查每个月份的包是否在数据集中存在
print("\n=== 检查CSV中的包是否在数据集中存在 ===")

all_exist = True
total_missing = 0

for month in sorted(csv_packages.keys()):
    month_dir = Path(DATA_DIR) / month
    if not month_dir.exists():
        print(f"  {month}: 数据集目录不存在!")
        all_exist = False
        continue

    # 获取数据集中该月的所有包
    dataset_packages = set()
    for f in month_dir.iterdir():
        if f.is_file() and f.suffix == '.gz':
            dataset_packages.add(f.name)

    # 检查CSV中的包是否都在数据集中
    csv_set = csv_packages[month]
    missing = csv_set - dataset_packages

    if missing:
        print(f"  {month}: CSV中有 {len(missing)} 个包在数据集中不存在:")
        for p in sorted(missing)[:10]:  # 只显示前10个
            print(f"    - {p}")
        if len(missing) > 10:
            print(f"    ... 还有 {len(missing) - 10} 个")
        all_exist = False
        total_missing += len(missing)
    else:
        print(f"  {month}: ✓ 所有 {len(csv_set)} 个包都在数据集中存在")

# 额外检查：数据集有但CSV没有的月份
print("\n=== 数据集中有但CSV中没有的月份 ===")
dataset_months = set(d.name for d in Path(DATA_DIR).iterdir() if d.is_dir())
csv_months = set(csv_packages.keys())
missing_months = dataset_months - csv_months

if missing_months:
    for month in sorted(missing_months):
        count = len(list((Path(DATA_DIR) / month).glob("*.gz")))
        print(f"  {month}: 数据集有 {count} 个包，但CSV中没有")
else:
    print("  无")

# 总结
print("\n=== 总结 ===")
if all_exist:
    print("✓ CSV中所有benign包都在数据集中存在")
else:
    print(f"✗ 共有 {total_missing} 个benign包在数据集中不存在")

print(f"\nCSV中的月份: {sorted(csv_months)}")
print(f"数据集中的月份: {sorted(dataset_months)}")
print(f"缺失的月份: {sorted(missing_months)}")
