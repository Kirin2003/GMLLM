import os
import shutil
import logging
import subprocess
from pathlib import Path
import zipfile
import tarfile
import re
from typing import Union, Tuple, Set
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import json

def extract_package_name(filename: str) -> str:
    name_part = Path(filename).name
    while any(name_part.endswith(suffix) for suffix in ['.gz', '.zip', '.whl', '.tar']):
        new_name_part = Path(name_part).stem
        if new_name_part == name_part:
            break
        name_part = new_name_part

    parts = name_part.split('-')
    name_parts = []
    for part in parts:
        if re.match(r'^[0-9vV]', part):
            break
        name_parts.append(part)
    
    return '-'.join(name_parts) if name_parts else name_part

def safe_extract_archive(archive_path: Path, extract_to: Path, skipped_archives: Set[str]) -> bool:
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        if archive_path.suffixes[-2:] == ['.tar', '.gz']:
            with tarfile.open(archive_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    member_path = (extract_to / member.name).resolve()
                    if not member_path.is_relative_to(extract_to.resolve()):
                        raise Exception(f"unsafe path: {member.name}")
                tar.extractall(path=extract_to)
        elif archive_path.suffix in ('.zip', '.whl'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                for member_info in zip_ref.infolist():
                    member_path = (extract_to / member_info.filename).resolve()
                    if not member_info.is_dir() and not member_path.is_relative_to(extract_to.resolve()):
                        raise Exception(f"unsafe path: {member_info.filename}")
                zip_ref.extractall(path=extract_to)
        else:
            skipped_archives.add(str(archive_path))
            logging.warning(f"[UNSUPPORTED] Unsupported archive format for {archive_path}")
            return False
        return True
    except Exception as e:
        skipped_archives.add(str(archive_path))
        logging.error(f"unzip failed: {archive_path}: {e}")
        shutil.rmtree(extract_to, ignore_errors=True)
        return False

def analyze_severity_from_file(json_file: Path) -> str:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        totals = data.get("metrics", {}).get("_totals", {})
        high = totals.get("SEVERITY.HIGH", 0)
        medium = totals.get("SEVERITY.MEDIUM", 0)
        low = totals.get("SEVERITY.LOW", 0)

        if high > 0:
            return "high"
        elif medium > 0:
            return "medium"
        elif low > 0:
            return "low"
        else:
            return "non"
    except Exception as e:
        logging.warning(f"cannot analysis report: {json_file}: {e}")
        return "error"

def _analyze_archive(
    archive_file: Path,
    pkg_name: str,
    output_base_path: Path,
    max_archive_size: int = 10 * 1024 * 1024,
    bandit_timeout: int = 300
) -> Tuple[str, str]:
    output_report_file = output_base_path / f"{pkg_name}.json"

    if output_report_file.exists():
        category = analyze_severity_from_file(output_report_file)
        return pkg_name, category

    try:
        archive_size = archive_file.stat().st_size
        if archive_size > max_archive_size:
            logging.warning(
                f"[{pkg_name}] skip too large file {archive_file.name} | "
                f"{archive_size / (1024*1024):.2f} MB (>10MB)"
            )
            return pkg_name, "too_large"
    except Exception as e:
        logging.error(f"[{pkg_name}] fail to get file size: {e}")
        return pkg_name, "error"

    extract_to = output_base_path / f"{pkg_name}_source_tmp"
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        if not safe_extract_archive(archive_file, extract_to, set()):
            return pkg_name, "extract_fail"
    except Exception as e:
        logging.error(f"[{pkg_name}] Extraction error: {e}")
        shutil.rmtree(extract_to, ignore_errors=True)
        return pkg_name, "extract_fail"

    bandit_command = [
        'bandit',
        '-r', str(extract_to),
        '-f', 'json',
        '-o', str(output_report_file),
    ]

    try:
        result = subprocess.run(
            bandit_command,
            capture_output=True,
            text=True,
            timeout=bandit_timeout
        )

        if result.returncode == 0:
            logging.info(f"[{pkg_name}] benign")
        elif result.returncode == 1:
            logging.info(f"[{pkg_name}] malicious")
        else:
            logging.error(f"[{pkg_name}] Bandit fatal (ret={result.returncode}): {result.stderr}")
            return pkg_name, "bandit_fail"

        if output_report_file.exists():
            category = analyze_severity_from_file(output_report_file)
            return pkg_name, category
        else:
            logging.error(f"[{pkg_name}] Bandit did not generate report file")
            return pkg_name, "bandit_fail"

    except subprocess.TimeoutExpired:
        logging.error(f"[{pkg_name}] Bandit timeout ({bandit_timeout}s).")
        status = "timeout"
    except Exception as e:
        logging.critical(f"[{pkg_name}] Unexpected error in bandit: {e}", exc_info=True)
        status = "error"

    finally:
        if extract_to.exists():
            try:
                shutil.rmtree(extract_to)
            except Exception as e:
                logging.warning(f"[{pkg_name}] Cleanup failed: {e}")

    return pkg_name, status


def _process_nested_package(
    package_folder: Path,
    output_base_path: Path,
    max_archive_size: int,
    bandit_timeout: int
) -> Tuple[str, str]:
    pkg_name = package_folder.name

    try:
        version_folders = [f for f in package_folder.iterdir() if f.is_dir()]
        if not version_folders:
            logging.warning(f"[{pkg_name}] No version folder found.")
            return pkg_name, "no_version"

        version_folder = version_folders[0]
        archive_files = (
            list(version_folder.glob("*.tar.gz")) +
            list(version_folder.glob("*.zip")) +
            list(version_folder.glob("*.whl"))
        )
        if not archive_files:
            logging.warning(f"[{pkg_name}] No supported archive found.")
            return pkg_name, "no_archive"

        input_archive = archive_files[0]
        return _analyze_archive(input_archive, pkg_name, output_base_path, max_archive_size, bandit_timeout)

    except Exception as e:
        logging.error(f"[{pkg_name}] Error in nested mode: {e}", exc_info=True)
        return pkg_name, "error"


def _process_flat_archive(
    archive_file: Path,
    output_base_path: Path,
    max_archive_size: int,
    bandit_timeout: int
) -> Tuple[str, str]:
    try:
        pkg_name = extract_package_name(archive_file.name)
        if not pkg_name:
            logging.warning(f"cannot extract file name: {archive_file.name}")
            return archive_file.name, "error"
        return _analyze_archive(archive_file, pkg_name, output_base_path, max_archive_size, bandit_timeout)
    except Exception as e:
        logging.error(f"[{archive_file.name}] Error in flat mode: {e}", exc_info=True)
        return archive_file.name, "error"


def process_single_package(
    target: Union[Path, str],
    output_base_path: Path,
    structure_type: str = "nested",
    max_archive_size: int = 10 * 1024 * 1024,
    bandit_timeout: int = 300
) -> Tuple[str, str]:
    target_path = Path(target)
    if structure_type == "nested":
        return _process_nested_package(target_path, output_base_path, max_archive_size, bandit_timeout)
    elif structure_type == "flat":
        return _process_flat_archive(target_path, output_base_path, max_archive_size, bandit_timeout)
    else:
        logging.error(f"Unknown structure_type: {structure_type}")
        return str(target_path), "error"

def batch_run_bandit(
    base_dir: str,
    output_base: str,
    structure_type: str = "nested",
    max_packages: int = None,
    max_workers: int = 4,
    executor_type: str = "process",
):
    output_base_path = Path(output_base)
    output_base_path.mkdir(parents=True, exist_ok=True)
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    func = partial(
        process_single_package,
        output_base_path=output_base_path,
        structure_type=structure_type
    )

    if structure_type == "nested":
        package_folders = [p for p in base_path.iterdir() if p.is_dir()]
        processed_packages = {p.stem for p in output_base_path.glob("*.json")}
        targets = [p for p in package_folders if p.name not in processed_packages]
        total_found = len(package_folders)

    elif structure_type == "flat":
        archive_files = []
        archive_files.extend(base_path.glob("*.tar.gz"))
        archive_files.extend(base_path.glob("*.zip"))
        archive_files.extend(base_path.glob("*.whl"))
        archive_files = sorted(set(archive_files))

        processed_packages = {extract_package_name(f.name) for f in output_base_path.glob("*.json")}
        targets = [
            f for f in archive_files
            if extract_package_name(f.name) not in processed_packages
        ]
        total_found = len(archive_files)

    else:
        raise ValueError("structure_type must be 'nested' or 'flat'")

    if max_packages:
        targets = targets[:max_packages]

    logging.info(f"Found {total_found} items in '{structure_type}' mode.")
    logging.info(f"Skipping {total_found - len(targets)} already processed.")
    logging.info(f"Processing {len(targets)} new items.")

    if not targets:
        logging.warning("No new items to process.")
        return

    results = {
        "high": 0, "medium": 0, "low": 0, "non": 0,
        "already_exists": total_found - len(targets),
        "too_large": 0, "no_version": 0, "no_archive": 0,
        "extract_fail": 0, "bandit_fail": 0, "timeout": 0, "error": 0
    }

    if executor_type == "process":
        executor_class = ProcessPoolExecutor
        logging.info(f"Using ProcessPoolExecutor with {max_workers} workers.")
    elif executor_type == "thread":
        executor_class = ThreadPoolExecutor
        logging.info(f"Using ThreadPoolExecutor with {max_workers} threads.")
    else:
        raise ValueError("executor_type must be 'process' or 'thread'")

    with executor_class(max_workers=max_workers) as executor:
        futures = {executor.submit(func, target): target for target in targets}

        with tqdm(total=len(futures), desc="Scanning Files", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                try:
                    pkg_name, category = future.result()
                    if category in results:
                        results[category] += 1
                    else:
                        results['error'] += 1
                        logging.warning(f"Unknown category: {category} for {pkg_name}")
                except Exception as e:
                    results['error'] += 1
                    logging.error(f"Task raised exception: {e}")

                pbar.set_postfix({
                    'High': results['high'],
                    'Medium': results['medium'],
                    'Low': results['low'],
                    'Clean': results['non'],
                    'Total': sum(results.values())
                })
                pbar.update(1)

    logging.info("statistic result:")
    for k, v in results.items():
        logging.info(f"{k.capitalize()}: {v}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler("bandit_scan_benign.log"),
            logging.StreamHandler()
        ]
    )

    MODE = "flat"  # "nested", "flat", "both"

    CONFIGS = [
        {
            "base_dir": "path/to/nested_packages",
            "output_base": "output/bandit_nested_results",
            "structure_type": "nested",
            "max_workers": 4
        },
        {
            "base_dir": "path/to/flat_archives",
            "output_base": "output/bandit_flat_results",
            "structure_type": "flat",
            "max_workers": 4
        }
    ]

    for config in CONFIGS:
        if MODE == "both" or config["structure_type"] == MODE:
            try:
                logging.info(f"begin to handle {config['structure_type']} mode: {config['base_dir']}")
                batch_run_bandit(**config)
            except Exception as e:
                logging.critical(f"error in  {config['structure_type']} {e}", exc_info=True)

    logging.info("finished all task")