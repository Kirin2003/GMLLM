"""Bandit scanner for pre-extracted Python source directories."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Tuple


def scan_directory(
    source_dir: Path,
    output_path: Path,
    timeout: int = 300
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
        logging.error(f"[{pkg_name}] Bandit failed: {result.stderr}")
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


def batch_scan(base_dir: Path, output_dir: Path, timeout: int = 300):
    """Scan all subdirectories in base_dir.

    Args:
        base_dir: Directory containing extracted package folders
        output_dir: Directory to save JSON reports
        timeout: Bandit timeout per package
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = {p.stem for p in output_dir.glob("*.json")}
    targets = [d for d in base_dir.iterdir() if d.is_dir() and d.name not in processed]

    logging.info(f"Found {len(targets)} packages to scan")

    for target in targets:
        pkg_name, category = scan_directory(target, output_dir / f"{target.name}.json", timeout)
        logging.info(f"[{pkg_name}] {category}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    # Example usage
    batch_scan(
        base_dir=Path("path/to/extracted_packages"),
        output_dir=Path("output/bandit_results")
    )
