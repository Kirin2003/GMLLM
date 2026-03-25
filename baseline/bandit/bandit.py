"""Bandit scanner for pre-extracted Python source directories."""

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.logger_utils import Logger

logger = Logger("bandit.log")


def scan_directory(
    source_dir: Path,
    output_path: Path,
    timeout: int = 300
) -> Tuple[str, str]:
    """Scan a directory with bandit.

    Args:
        source_dir: Path to the extracted Python source directory
        output_path: Path to save the JSON report
        timeout: Bandit timeout in seconds (default: 300)

    Returns:
        Tuple of (package_name, category)
    """
    source_dir = Path(source_dir)
    output_path = Path(output_path)
    pkg_name = source_dir.name

    if output_path.exists():
        return pkg_name, parse_category(output_path)

    # 第一次运行
    result = _run_bandit(source_dir, output_path, timeout)

    # 如果超时，重试一次
    if result is None:
        logger.log(f"[{pkg_name}] Timeout, retrying...")
        # 删除可能生成的空文件
        if output_path.exists():
            output_path.unlink()
        result = _run_bandit(source_dir, output_path, timeout)

    # 检查结果
    if result is None:
        logger.log(f"[{pkg_name}] Timeout failed twice")
        return pkg_name, "bandit_timeout"

    if result.returncode not in (0, 1):
        logger.log(f"[{pkg_name}] Bandit failed: {result.stderr}")
        return pkg_name, "bandit_fail"

    if output_path.exists():
        return pkg_name, parse_category(output_path)
    return pkg_name, "bandit_fail"


def _run_bandit(source_dir: Path, output_path: Path, timeout: int):
    """Run bandit and return result or None on timeout."""
    try:
        return subprocess.run(
            ['bandit', '-r', str(source_dir), '-f', 'json', '-o', str(output_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return None


def parse_category(json_file: Path) -> str:
    """Parse bandit JSON report to determine if malicious or benign.

    If SEVERITY.HIGH >= 3, returns "malicious".
    Otherwise returns "benign".

    Args:
        json_file: Path to the bandit4mal.json file

    Returns:
        "malicious" or "benign"
    """
    try:
        data = json.load(open(json_file))
        totals = data.get("metrics", {}).get("_totals", {})
        severity = totals.get("SEVERITY.HIGH", 0) + totals.get("SEVERITY.LOW", 0) + totals.get("SEVERITY.MEDIUM", 0) + totals.get("SEVERITY.UNDEFINED", 0)

        if severity >= 3:
            return "malicious"
        return "benign"
    except Exception:
        return "error"


def batch_scan(base_dir: str, start_month: str, end_month: str, timeout: int = None):
    """Scan all packages in {benign,malicious}/{month}/ structure.

    Args:
        base_dir: Base directory containing benign/malicious subdirs
        start_month: Start month in format "YYYY-MM" (inclusive), e.g. "2023-03"
        end_month: End month in format "YYYY-MM" (inclusive), e.g. "2024-12"
        timeout: Bandit timeout in seconds (None = no timeout)

    Output: bandit4mal.json saved in each package directory.
    Supports resume: skips packages that already have bandit4mal.json.
    """
    from utils.month_utils import generate_month_range

    base_dir = Path(base_dir)
    months = generate_month_range(start_month, end_month)

    # Find all package directories
    targets = []
    for label in ["benign", "malicious"]:
        for month in months:
            month_dir = base_dir / label / month
            for pkg_dir in month_dir.iterdir():
                if pkg_dir.is_dir():
                    targets.append((label, pkg_dir))

    # Filter: skip already processed
    targets = [t for t in targets if not (t[1] / "bandit4mal.json").exists()]

    logger.log(f"Found {len(targets)} packages to scan")

    for label, pkg_dir in targets:
        pkg_name = pkg_dir.name
        output_path = pkg_dir / "bandit4mal.json"
        _, category = scan_directory(pkg_dir, output_path, timeout)
        logger.log(f"[{label}/{pkg_name}] {category}")


def batch_parse(base_dir: str, start_month: str, end_month: str):
    """Parse all bandit4mal.json files and save results to CSV.

    Args:
        base_dir: Base directory containing benign/malicious subdirs
        start_month: Start month in format "YYYY-MM" (inclusive)
        end_month: End month in format "YYYY-MM" (inclusive)

    Output: CSV saved to ../results/bandit_results.csv with columns:
        month, package_name, verdict, label
    """
    from utils.month_utils import generate_month_range

    base_dir = Path(base_dir)
    months = generate_month_range(start_month, end_month)

    results_dir = Path("../../results")
    output_csv = results_dir / "bandit_results.csv"

    # Find all bandit4mal.json files
    targets = []
    for label in ["benign", "malicious"]:
        for month in months:
            month_dir = base_dir / label / month
            if not month_dir.exists():
                continue
            for pkg_dir in month_dir.iterdir():
                if pkg_dir.is_dir():
                    json_file = pkg_dir / "bandit4mal.json"
                    if json_file.exists():
                        targets.append((month, label, pkg_dir.name, json_file))

    logger.log(f"Found {len(targets)} bandit4mal.json files to parse")

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["month", "package_name", "verdict", "label"])

        for month, label, pkg_name, json_file in targets:
            verdict = parse_category(json_file)
            writer.writerow([month, pkg_name, verdict, label])
            logger.log(f"[{label}/{month}/{pkg_name}] {verdict}")

    logger.log(f"Results saved to {output_csv}")


def compute_metrics(csv_path: str = None):
    """Compute precision/recall/f1 for malicious package detection per month."""
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support

    if csv_path is None:
        csv_path = "../../results/bandit_results.csv"

    df = pd.read_csv(csv_path)
    results = []

    for month in sorted(df['month'].unique()):
        month_df = df[df['month'] == month]
        y_true = (month_df['label'] == 'malicious').astype(int)
        y_pred = (month_df['verdict'] == 'malicious').astype(int)

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
        results.append({'month': month, 'precision': p, 'recall': r, 'f1': f1})

    res_df = pd.DataFrame(results)

    output_json = "../../results/bandit.json"

    json_data = {
        "month": res_df['month'].tolist(),
        "f1": res_df['f1'].tolist(),
        "precision": res_df['precision'].tolist(),
        "recall": res_df['recall'].tolist()
    }

    import json
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)

    print(res_df.to_string(index=False))
    print(f"\nResults saved to {output_json}")
    return res_df


if __name__ == "__main__":
    batch_parse(
        base_dir="/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset", start_month="2023-03", end_month="2024-12"
    )
    compute_metrics()
