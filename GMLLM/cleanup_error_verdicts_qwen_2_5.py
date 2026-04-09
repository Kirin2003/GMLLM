import json
import os
from pathlib import Path
from typing import Tuple, List


def cleanup_error_verdicts(dataset_path: str, dry_run: bool = False) -> Tuple[int, List[str]]:
    """
    遍历数据集下 malicious/benign 每个月的包目录中的 qwen3_5_27b.json，
    如果 verdict 不是 Malicious 或 Benign 就删除该文件。

    Args:
        dataset_path: 数据集根目录路径
        dry_run: True 则只打印不删除

    Returns:
        (会删除的文件数量, 会被删除的文件路径列表)
    """
    deleted_files = []
    dataset_root = Path(dataset_path)

    for package_type in ["malicious", "benign"]:
        type_dir = dataset_root / package_type
        if not type_dir.exists():
            continue

        for month_dir in sorted(type_dir.iterdir()):
            if not month_dir.is_dir():
                continue

            for package_dir in month_dir.iterdir():
                if not package_dir.is_dir():
                    continue

                json_file = package_dir / "qwen2_5.json"
                if not json_file.exists():
                    continue

                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    verdict = data.get("verdict")
                    if verdict not in ("Malicious", "Benign"):
                        deleted_files.append(str(json_file))
                        print(f"[Dry run] Would delete: {json_file} (verdict: {verdict})")
                        if not dry_run:
                            json_file.unlink()
                            print(f"Deleted: {json_file}")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading {json_file}: {e}")

    return len(deleted_files), deleted_files


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="实际删除文件，默认只预览")
    args = parser.parse_args()

    dataset_path = "/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset"
    count, files = cleanup_error_verdicts(dataset_path, dry_run=not args.execute)
    print(f"\nTotal: {count}")