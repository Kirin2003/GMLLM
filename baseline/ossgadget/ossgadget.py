#!/usr/bin/env python3
"""
使用 ossgadget 分析本地 PyPI 包（通过 HTTP 服务器）
"""

import subprocess
import sys
from pathlib import Path


# 配置
HTTP_SERVER_DIR = "/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset_compressed"
HTTP_SERVER_PORT = 8045
OSSGADGET_PATH = "/Data2/hxq/ossgadget/ossgadget"
OUTPUT_BASE_DIR = "/Data2/hxq/datasets/ossgadget_results"


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
        print(f"解析路径失败: {e}")
        return

    # 构建输出文件路径
    output_dir = Path(OUTPUT_BASE_DIR) / category / month
    print(f'output_dir: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{package_name}.json"

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

    print(f"命令: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"ossgadget 错误: {result.stderr}")
            return

        print(f"结果已保存到: {output_file}")

    except subprocess.TimeoutExpired:
        print("ossgadget 超时")
    except Exception as e:
        print(f"错误: {e}")


def analyze_batch(input_dir: str, skip_existing: bool = True):
    """
    批量分析文件夹下的所有包
    """
    input_path = Path(input_dir)
    package_files = []
    for category in ["benign", "malicious"]:
        category_path = input_path / category
        for month_dir in sorted(category_path.iterdir()):
            for pkg_file in month_dir.glob("*.tar.gz"):
                package_files.append(pkg_file)

    print(f"共找到 {len(package_files)} 个包")

    skip_count = 0

    for i, package_file in enumerate(package_files, 1):
        month = package_file.parent.name
        category = package_file.parent.parent.name
        package_name = package_file.name.replace(".tar.gz", "").rsplit("-", 1)[0]

        output_file = Path(OUTPUT_BASE_DIR) / category / month / f"{package_name}.json"

        if skip_existing and output_file.exists():
            print(f"[{i}/{len(package_files)}] 跳过: {package_file.name}")
            skip_count += 1
            continue

        print(f"[{i}/{len(package_files)}] 分析: {category}/{month}/{package_file.name}")
        analyze_package(str(package_file))

    print(f"\n完成! 跳过: {skip_count}")


def analyze_missing_packages(csv_path: str):
    """
    读取缺失包 CSV 文件，运行漏掉的包并保存结果

    Args:
        csv_path: missing_packages.csv 文件路径
    """
    import csv

    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"CSV 文件不存在: {csv_path}")
        return

    # 读取 CSV，跳过 header
    packages = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            packages.append(row)

    print(f"共找到 {len(packages)} 个缺失包")

    for i, pkg in enumerate(packages, 1):
        label = pkg['label']  # benign 或 malicious
        month = pkg['month']
        name = pkg['name']
        path = pkg['path']  # 格式: malicious/2023-07/xxx.tar.gz

        # 构建完整包路径
        package_path = Path(HTTP_SERVER_DIR) / path
        if not package_path.exists():
            print(f"[{i}/{len(packages)}] 包不存在: {package_path}")
            continue

        print(f"[{i}/{len(packages)}] 分析: {label}/{month}/{name}")
        analyze_package(str(package_path))

    print(f"\n完成! 共处理 {len(packages)} 个包")


def main():
    analyze_missing_packages("/Data2/hxq/datasets/scripts/missing_packages.csv")


if __name__ == "__main__":
    main()
