from ntpath import exists
import os
import json
import tarfile
import zipfile
from pathlib import Path
import shutil
import time
from tqdm import tqdm
import concurrent.futures
import argparse

from transformers import AutoTokenizer

tokenizer = None
SYSTEM_PROMPT ="""You are a PyPI package security auditor. 
    You have been provided with the code of a PyPI package script. 
    Please carefully analyze the possible malicious behaviors therein and answer the following:
    Is this code indicative of potential malicious activity? (Respond only with 'Malicious' or 'Benign')
    Provide your reasoning.
    Response Format:
    Verdict:
    Reasoning:
"""
def init_worker(output_dir: Path, model_id: str):
    global worker_output_file, tokenizer
    
    pid = os.getpid()
    file_path = output_dir / f"token_counts_{pid}.jsonl"
    worker_output_file = open(file_path, 'a', encoding='utf-8')
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"Worker {pid} initialized tokenizer for {model_id}")
    except Exception as e:
        print(f"Worker {pid} failed to initialize tokenizer for {model_id}: {e}")
        tokenizer = None

def count_tokens(source_code: str) -> int:
    global tokenizer
    if not tokenizer:
        return -1
    length = tokenizer.encode(source_code) + tokenizer.encode(SYSTEM_PROMPT)
    return length

def process_package_to_jsonl(package_folder: Path, temp_extract_base: Path) -> bool:
    global worker_output_file
    package_name = package_folder.name
    temp_extract_dir = temp_extract_base / package_name

    try:
        input_archive = None
        version_folders = [f for f in package_folder.iterdir() if f.is_dir()]
        if not version_folders: return False
        version_folder = version_folders[0]
        
        supported_suffixes = ['.whl', '.zip']
        for f in version_folder.iterdir():
            if f.is_file() and (f.name.endswith('.tar.gz') or f.suffix in supported_suffixes):
                input_archive = f
                break
        if not input_archive: return False
        if not safe_extract_archive(input_archive, temp_extract_dir): return False

        source_code = get_all_source_code(temp_extract_dir)
        if not source_code: return False

        token_count = count_tokens(source_code)
        if token_count == -1:
            return False

        result_data = {
            "packagename": package_name,
            "token_count": token_count
        }

        json_line = json.dumps(result_data, ensure_ascii=False)
        worker_output_file.write(json_line + '\n')
        worker_output_file.flush()
        
        return True

    except Exception as e:
        print(f"An unexpected error occurred while processing {package_name}: {e}")
        return False
    finally:
        shutil.rmtree(temp_extract_dir, ignore_errors=True)

def safe_extract_archive(archive_path: Path, extract_to: Path) -> bool:
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        if archive_path.suffixes[-2:] == ['.tar', '.gz']:
            with tarfile.open(archive_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    member_path = (extract_to / member.name).resolve()
                    if not str(member_path).startswith(str(extract_to.resolve())):
                        raise Exception(f"Unsafe path (Zip Slip): {member.name}")
                tar.extractall(path=extract_to)
        elif archive_path.suffix in ('.zip', '.whl'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                for member_info in zip_ref.infolist():
                    member_path = (extract_to / member_info.filename).resolve()
                    if not str(member_path).startswith(str(extract_to.resolve())):
                        raise Exception(f"Unsafe path (Zip Slip): {member_info.filename}")
                zip_ref.extractall(path=extract_to)
        else:
            return False
        return True
    except Exception as e:
        print(f"Extraction failed for {archive_path}: {e}")
        shutil.rmtree(extract_to, ignore_errors=True)
        return False

def get_all_source_code(directory: Path) -> str:
    all_code = []
    for py_file in directory.rglob("*.py"):
        try:
            all_code.append(f"# --- File: {py_file.relative_to(directory)} ---\n")
            all_code.append(py_file.read_text(encoding='utf-8', errors='ignore'))
            all_code.append("\n\n")
        except Exception as e:
            print(f"Failed to read file {py_file}: {e}")
    return "".join(all_code)

def main_pipeline(base_dir: str, results_base_dir: str, model_id: str, max_workers: int = 10):
    base_path = Path(base_dir)
    model_name_safe = model_id.replace("/", "_")
    base_dir = Path(results_base_dir)
    base_dir.mkdir(exist_ok = True)
    results_dir = Path(results_base_dir) / model_name_safe
    temp_extract_base = Path(f"./temp_extraction_{model_name_safe}")
    
    results_dir.mkdir(exist_ok=True)
    temp_extract_base.mkdir(exist_ok=True)

    all_package_folders = [p for p in base_path.iterdir() if p.is_dir()]
    
    print(f"Found {len(all_package_folders)} total packages to process for token counting with model '{model_id}'.")
    if not all_package_folders:
        return

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(results_dir, model_id)
    ) as executor:
        futures = [
            executor.submit(process_package_to_jsonl, folder, temp_extract_base) 
            for folder in all_package_folders
        ]
        
        progress_bar = tqdm(concurrent.futures.as_completed(futures), total=len(all_package_folders), desc=f"Counting tokens for {model_id}")
        
        success_count = 0
        fail_count = 0
        for future in progress_bar:
            if future.result():
                success_count += 1
            else:
                fail_count += 1
            progress_bar.set_postfix_str(f"Success: {success_count}, Failed: {fail_count}")

    print(f"\nToken counting complete for {model_id}. Success: {success_count}, Failed: {fail_count}")
    print(f"Results saved in individual .jsonl files in: {results_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Count tokens for local Hugging Face models.")
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="The Hugging Face identifier for the model (e.g., 'meta-llama/Meta-Llama-3-8B')."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel worker processes."
    )
    args = parser.parse_args()

    BASE_MALICIOUS_DIR = "path/to/malicious_dataset"
    OUTPUT_DIR = "path/to/output_results"

    main_pipeline(BASE_MALICIOUS_DIR, OUTPUT_DIR, model_id=args.model_id, max_workers=args.max_workers)