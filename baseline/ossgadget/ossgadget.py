#!/usr/bin/env python3
"""
使用 ossgadget 分析本地 PyPI 包（通过 HTTP 服务器）
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from utils.logger_utils import Logger

# 全局日志记录器
log = Logger("ossgadget.log")

# 配置
HTTP_SERVER_DIR = "/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset_compressed"
HTTP_SERVER_PORT = 8045
OSSGADGET_PATH = "/Data2/hxq/ossgadget/ossgadget"
OUTPUT_BASE_DIR = "/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset"


def analyze_package(package_path: str):
    """
    分析一个本地 PyPI 包压缩包，保存原始输出到指定位置

    Args:
        package_path: 压缩包路径
    """
    package_path = Path(package_path)

    # 从路径中提取信息
    # 格式: .../malicious/2023-03/aiotoolsbox-1.4.5.tar.gz
    try:
        month = package_path.parent.name  # 2023-03
        category = package_path.parent.parent.name  # malicious 或 benign
        filename = package_path.name  # aiotoolsbox-1.4.5.tar.gz
        # 提取包名和版本: aiotoolsbox-1.4.5.tar.gz -> aiotoolsbox@1.4.5
        name_version = filename.replace(".tar.gz", "").replace(".whl", "").replace(".zip", "")
        if "-" in name_version:
            parts = name_version.rsplit("-", 1)
            package_name = parts[0]
            version = parts[1]
        else:
            package_name = name_version
            version = "unknown"
    except Exception as e:
        log.log(f"解析路径失败: {e}")
        return

    # 构建输出文件路径: .../category/month/package_name/ossgadget.json
    output_dir = Path(OUTPUT_BASE_DIR) / category / month / name_version
    log.log(f'output_dir: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ossgadget.json"

    # 构建 PURL 和 URL
    purl = f"pkg:url/{package_name}@{version}?url=http://localhost:{HTTP_SERVER_PORT}/{category}/{month}/{filename}"

    # 调用 ossgadget，输出 JSON 格式
    command = [
        OSSGADGET_PATH,
        "detect-backdoor",
        purl,
        "-f", "sarifv2",
        "-o", str(output_file)
    ]

    log.log(f"命令: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            log.log(f"ossgadget 错误: {result.stderr}")
            return

        log.log(f"结果已保存到: {output_file}")

    except subprocess.TimeoutExpired:
        log.log("ossgadget 超时")
    except Exception as e:
        log.log(f"错误: {e}")


def analyze_batch(input_dir: str, skip_existing: bool = True):
    """
    批量分析文件夹下的所有包
    """
    input_path = Path(input_dir)
    package_files = []
    for category in ["benign", "malicious"]:
        category_path = input_path / category
        for month_dir in sorted(category_path.iterdir()):
            for pkg_file in sorted(month_dir.glob("*.tar.gz")):
                package_files.append(pkg_file)

    log.log(f"共找到 {len(package_files)} 个包")

    skip_count = 0

    for i, package_file in enumerate(package_files, 1):
        month = package_file.parent.name
        category = package_file.parent.parent.name
        name_version = package_file.name.replace(".tar.gz", "").replace(".whl", "").replace(".zip", "")

        output_file = Path(OUTPUT_BASE_DIR) / category / month / name_version / "ossgadget.json"

        if skip_existing and output_file.exists():
            log.log(f"[{i}/{len(package_files)}] 跳过: {package_file.name}")
            skip_count += 1
            continue

        log.log(f"[{i}/{len(package_files)}] 分析: {category}/{month}/{package_file.name}")
        analyze_package(str(package_file))

    log.log(f"\n完成! 跳过: {skip_count}")


def analyze_missing_packages(csv_path: str):
    """
    读取缺失包 CSV 文件，运行漏掉的包并保存结果

    Args:
        csv_path: missing_packages.csv 文件路径
    """
    import csv

    csv_file = Path(csv_path)
    if not csv_file.exists():
        log.log(f"CSV 文件不存在: {csv_path}")
        return

    # 读取 CSV，跳过 header
    packages = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            packages.append(row)

    log.log(f"共找到 {len(packages)} 个缺失包")

    for i, pkg in enumerate(packages, 1):
        label = pkg['label']  # benign 或 malicious
        month = pkg['month']
        name = pkg['name']
        path = pkg['path']  # 格式: malicious/2023-07/xxx.tar.gz

        # 检查是否已存在 ossgadget.json
        output_file = Path(OUTPUT_BASE_DIR) / label / month / name / "ossgadget.json"
        if output_file.exists():
            log.log(f"[{i}/{len(packages)}] 跳过已存在: {label}/{month}/{name}")
            continue

        # 构建完整包路径
        package_path = Path(HTTP_SERVER_DIR) / path
        if not package_path.exists():
            log.log(f"[{i}/{len(packages)}] 包不存在: {package_path}")
            continue

        log.log(f"[{i}/{len(packages)}] 分析: {label}/{month}/{name}")
        analyze_package(str(package_path))

    log.log(f"\n完成! 共处理 {len(packages)} 个包")


def batch_parse(base_dir: str, start_month: str, end_month: str):
    """Parse all ossgadget.json files and save results to CSV.

    Args:
        base_dir: Base directory containing benign/malicious subdirs
        start_month: Start month in format "YYYY-MM" (inclusive)
        end_month: End month in format "YYYY-MM" (inclusive)

    Output: CSV saved to ../../results/ossgadget_results.csv with columns:
        month, package_name, verdict, label
    """
    import csv
    from utils.month_utils import generate_month_range

    base_dir = Path(base_dir)
    months = generate_month_range(start_month, end_month)

    results_dir = Path("../../results")
    output_csv = results_dir / "ossgadget_results.csv"

    # Find all ossgadget.json files
    targets = []
    for label in ["benign", "malicious"]:
        for month in months:
            month_dir = base_dir / label / month
            if not month_dir.exists():
                continue
            for pkg_dir in month_dir.iterdir():
                if pkg_dir.is_dir():
                    json_file = pkg_dir / "ossgadget.json"
                    if json_file.exists():
                        targets.append((month, label, pkg_dir.name, json_file))

    log.log(f"Found {len(targets)} ossgadget.json files to parse")

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["month", "package_name", "verdict", "label"])

        for month, label, pkg_name, json_file in targets:
            verdict = parse_category(json_file)
            writer.writerow([month, pkg_name, verdict, label])
            log.log(f"[{label}/{month}/{pkg_name}] {verdict}")

    log.log(f"Results saved to {output_csv}")


def parse_category(json_file: Path) -> str:
    """Parse ossgadget SARIF JSON to determine if malicious or benign.

    Counts triggered rules in results array.
    If count >= 3, returns "malicious", otherwise "benign".

    Args:
        json_file: Path to the ossgadget.json file

    Returns:
        "malicious" or "benign"
    """
    try:
        data = json.load(open(json_file))
        runs = data.get("runs", [])
        if not runs:
            return "error"
        results = runs[0].get("results", [])

        # Count results
        rule_count = 0
        for result in results:
            severity = result.get("properties", {}).get("Severity", 0)
            confidence = result.get("properties", {}).get("Confidence", 0)
            if severity >= 2 and confidence >= 2:
                rule_count += 1

        if rule_count >= 5:
            return "malicious"
        return "benign"
    except Exception:
        return "error"


def compute_metrics(csv_path: str = None):
    """Compute precision/recall/f1 for malicious package detection per month.

    Args:
        csv_path: Path to ossgadget_results.csv (default: ../../results/ossgadget_results.csv)

    Output: JSON saved to ../../results/ossgadget.json with format:
        {"month": [...], "f1": [...], "precision": [...], "recall": [...]}
    """
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support

    if csv_path is None:
        csv_path = "../../results/ossgadget_results.csv"

    df = pd.read_csv(csv_path)
    results = []

    for month in sorted(df['month'].unique()):
        month_df = df[df['month'] == month]
        y_true = (month_df['label'] == 'malicious').astype(int)
        y_pred = (month_df['verdict'] == 'malicious').astype(int)

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
        results.append({'month': month, 'precision': p, 'recall': r, 'f1': f1})

    res_df = pd.DataFrame(results)

    output_json = "../../results/ossgadget.json"

    json_data = {
        "month": res_df['month'].tolist(),
        "f1": res_df['f1'].tolist(),
        "precision": res_df['precision'].tolist(),
        "recall": res_df['recall'].tolist()
    }

    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)

    print(res_df.to_string(index=False))
    print(f"\nResults saved to {output_json}")
    return res_df


def main():
    # analyze_batch(HTTP_SERVER_DIR)
    batch_parse(OUTPUT_BASE_DIR, "2023-03", "2024-12")
    compute_metrics()


if __name__ == "__main__":
    main()
