#!/usr/bin/env python3
"""Batch scan PyPI packages with VirusTotal.

Two-step approach:
1. submit_scan: Submit files for scanning (wait_for_completion=False), save analysis_id
2. get_results: Query results by analysis_id (run tomorrow or later)
"""

import csv
import hashlib
import os
import time
import vt

from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from utils.logger_utils import Logger

# 全局日志对象
logger = Logger("virustotal.log")


def compute_sha256(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def submit_scan(file_path: str, api_key: str, timeout: int = 60) -> dict:
    """Submit a file for scanning (wait_for_completion=False)."""
    sha256_hex = compute_sha256(file_path)

    with vt.Client(api_key, timeout=timeout) as client:
        # Submit file for scanning without waiting for completion
        analysis = client.scan_file(
            open(file_path, "rb"),
            wait_for_completion=False
        )

    return {
        "sha256": sha256_hex,
        "analysis_id": analysis.id,
    }


def get_result(analysis_id: str, api_key: str, timeout: int = 60) -> dict:
    """Get scan result by analysis_id."""
    with vt.Client(api_key, timeout=timeout) as client:
        # Poll until analysis is complete
        analysis = client.get_object(f"/analyses/{analysis_id}")

        # Check if analysis is complete
        while analysis.status != "completed":
            logger.log(f"  Analysis status: {analysis.status}, waiting...")
            time.sleep(10)
            analysis = client.get_object(f"/analyses/{analysis_id}")

        # Get stats directly from analysis
        stats = {
            "malicious": analysis.stats.get("malicious", 0),
            "suspicious": analysis.stats.get("suspicious", 0),
            "undetected": analysis.stats.get("undetected", 0),
            "total": sum(analysis.stats.values()),
        }

        return stats


def submit_scan_main():
    """Submit all packages for scanning."""
    base_dir = "/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset_compressed"
    api_key = os.environ.get("VIRUSTOTAL_API_KEY")
    scan_csv = "/Data2/hxq/GMLLM/results/virustotal_scan.csv"

    # Labels and their directories
    label_dirs = [
        ("benign", f"{base_dir}/benign"),
        #("malicious", f"{base_dir}/malicious"),
    ]

    months = [f"2023-{m:02d}" for m in range(3, 13)] + [f"2024-{m:02d}" for m in range(1, 13)]

    # Init CSV
    if not os.path.exists(scan_csv):
        with open(scan_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["package", "month", "label", "sha256", "analysis_id", "status"],
            )
            writer.writeheader()

    # Load already submitted packages - use (package, month, label) as unique key
    submitted = []
    if os.path.exists(scan_csv):
        with open(scan_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                submitted.append((row["package"], row["month"], row["label"]))
        logger.log(f"Resuming: {len(submitted)} packages already submitted")

    # Collect packages
    packages = []
    for label, label_dir in label_dirs:
        for month in months:
            month_dir = f"{label_dir}/{month}"
            for pkg in os.listdir(month_dir):
                if pkg.endswith(".tar.gz"):
                    packages.append((f"{month_dir}/{pkg}", month, label))

    logger.log(f"Found {len(packages)} packages to submit")

    # Open CSV for appending
    with open(scan_csv, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["package", "month", "label", "sha256", "analysis_id", "status"],
        )

        for i, (pkg_path, month, label) in enumerate(packages):
            pkg_name = pkg_path.split("/")[-1]
            if (pkg_name, month, label) in submitted:
                continue

            logger.log(f"[{i + 1}/{len(packages)}] Submitting {pkg_name} ({month}, {label})...")

            try:
                result = submit_scan(pkg_path, api_key)

                writer.writerow({
                    "package": pkg_name,
                    "month": month,
                    "label": label,
                    "sha256": result["sha256"],
                    "analysis_id": result["analysis_id"],
                    "status": "pending",
                })
                f.flush()

                logger.log(f"  -> analysis_id: {result['analysis_id']}")

            except Exception as e:
                logger.log(f"  -> Error: {e}")
                if "QuotaExceededError" in str(type(e).__name__) or "quota" in str(e).lower():
                    logger.log("Quota exceeded, exiting...")
                    return

            # Rate limit: 4 requests per minute = 15 seconds between requests
            time.sleep(15.5)


def get_results_main():
    """Query results for submitted scans."""
    api_key = os.environ.get("VIRUSTOTAL_API_KEY")
    scan_csv = "/Data2/hxq/GMLLM/results/virustotal_scan.csv"
    result_csv = "/Data2/hxq/GMLLM/results/virustotal.csv"

    # Read submitted scans
    rows = []
    with open(scan_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.log(f"Found {len(rows)} submitted scans")

    # Init result CSV
    if not os.path.exists(result_csv):
        with open(result_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["package", "month", "label", "verdict", "malicious", "suspicious", "undetected", "total", "sha256"],
            )
            writer.writeheader()

    # Load completed packages - use (package, month, label) as unique key
    completed = []
    if os.path.exists(result_csv):
        with open(result_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.append((row["package"], row["month"], row["label"]))
    logger.log(f"Already completed: {len(completed)} packages")

    # Query results
    with open(result_csv, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["package", "month", "label", "verdict", "malicious", "suspicious", "undetected", "total", "sha256"],
        )

        for i, row in enumerate(rows):
            pkg_name = row["package"]
            month = row["month"]
            label = row["label"]
            if (pkg_name, month, label) in completed:
                continue

            analysis_id = row["analysis_id"]
            logger.log(f"[{i + 1}/{len(rows)}] Getting result for {pkg_name} (analysis_id: {analysis_id})...")

            try:
                stats = get_result(analysis_id, api_key)

                total = stats["total"]
                verdict = "MALICIOUS" if stats["malicious"] > 0 else "SUSPICIOUS" if stats["suspicious"] > 0 else "CLEAN"

                writer.writerow({
                    "package": pkg_name,
                    "month": row["month"],
                    "label": row["label"],
                    "verdict": verdict,
                    "malicious": stats["malicious"],
                    "suspicious": stats["suspicious"],
                    "undetected": stats["undetected"],
                    "total": total,
                    "sha256": row["sha256"],
                })
                f.flush()

                logger.log(f"  -> {verdict}: {stats['malicious']}/{total}")

            except Exception as e:
                logger.log(f"  -> Error: {e}")
                if "QuotaExceededError" in str(type(e).__name__) or "quota" in str(e).lower():
                    logger.log("Quota exceeded, exiting...")
                    return

            # Rate limit
            time.sleep(15.5)


def main():
    mode = "submit"  # "submit" or "results"

    if mode == "submit":
        submit_scan_main()
    else:
        get_results_main()


if __name__ == "__main__":
    main()
