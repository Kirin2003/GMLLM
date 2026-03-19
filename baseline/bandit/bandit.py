"""Bandit scanner for pre-extracted Python source directories."""

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
    timeout: int = None
) -> Tuple[str, str]:
    """Scan a directory with bandit.

    Args:
        source_dir: Path to the extracted Python source directory
        output_path: Path to save the JSON report
        timeout: Bandit timeout in seconds

    Returns:
        Tuple of (package_name, category)
    """
    source_dir = Path(source_dir)
    output_path = Path(output_path)
    pkg_name = source_dir.name

    if output_path.exists():
        return pkg_name, _parse_category(output_path)

    result = subprocess.run(
        ['bandit', '-r', str(source_dir), '-f', 'json', '-o', str(output_path)],
        capture_output=True,
        text=True,
        timeout=timeout
    )

    if result.returncode not in (0, 1):
        logger.log(f"[{pkg_name}] Bandit failed: {result.stderr}")
        return pkg_name, "bandit_fail"

    if output_path.exists():
        return pkg_name, _parse_category(output_path)
    return pkg_name, "bandit_fail"


def _parse_category(json_file: Path) -> str:
    """Parse severity category from bandit JSON report."""
    try:
        data = json.load(open(json_file))
        totals = data.get("metrics", {}).get("_totals", {})
        high = totals.get("SEVERITY.HIGH", 0)
        medium = totals.get("SEVERITY.MEDIUM", 0)
        low = totals.get("SEVERITY.LOW", 0)

        if high > 0:
            return "high"
        if medium > 0:
            return "medium"
        if low > 0:
            return "low"
        return "non"
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


if __name__ == "__main__":
    batch_scan(
        base_dir="/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset", start_month="2023-03", end_month="2024-12"
    )
