import os
import json
import tarfile
import zipfile
from pathlib import Path
from openai import AzureOpenAI,RateLimitError
from tqdm import tqdm
import concurrent.futures
from threading import Lock
import shutil
import time
system_prompt_a = """You are a PyPI package security auditor. 
    You have been provided with the code of a PyPI package script. 
    Please carefully analyze the possible malicious behaviors therein and answer the following:
    Is this code indicative of potential malicious activity? (Respond only with 'Malicious' or 'Benign')
    Provide your reasoning.
    Response Format:
    Verdict:
    Reasoning:
"""

AZURE_OPENAI_API_VERSION = "2023-07-01-preview"
AZURE_OPENAI_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "your_deployment_name")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your_api_key_here")
AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL", "https://your-openai-endpoint.com")

worker_output_file = None

def init_worker(output_dir: Path):
    global worker_output_file
    pid = os.getpid()
    file_path = output_dir / f"results_{pid}.jsonl"
    worker_output_file = open(file_path, 'a', encoding='utf-8')

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
        llm_response = query_llm_for_analysis(source_code, package_name)
        if not llm_response: return False

        try:
            lines = llm_response.strip().split('\n')
            verdict = lines[0].replace("Verdict:", "").strip()
            reasoning = "\n".join(lines[1:]).replace("Reasoning:", "").strip()
            result_data = {
                "packagename": package_name,
                "Verdict": verdict,
                "Reasoning": reasoning
            }
        except IndexError:
            result_data = {"packagename": package_name, "raw_response": llm_response}

        json_line = json.dumps(result_data, ensure_ascii=False)
        worker_output_file.write(json_line + '\n')
        worker_output_file.flush()
        
        return True

    except Exception as e:
        print(f"{package_name} fatal:{e}")
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
                    if not member_path.is_relative_to(extract_to.resolve()):
                        raise Exception(f"{member.name}")
                tar.extractall(path=extract_to)
        elif archive_path.suffix in ('.zip', '.whl'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                for member_info in zip_ref.infolist():
                    member_path = (extract_to / member_info.filename).resolve()
                    if not member_info.is_dir() and not member_path.is_relative_to(extract_to.resolve()):
                        raise Exception(f"{member_info.filename}")
                zip_ref.extractall(path=extract_to)
        else:
            return False
        return True
    except Exception as e:
        print(f"unzip failed {archive_path}: {e}")
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
            print(f"read failed {py_file}: {e}")
    return "".join(all_code)

def log_error_package(package_name: str, error_file_path: str):
    with open(error_file_path, 'a', encoding='utf-8') as f:
        f.write(package_name + '\n')

def query_llm_for_analysis(source_code: str,package_name: str) -> str:
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION 
    )
    system_prompt = system_prompt_a

    error1_path = AZURE_OPENAI_DEPLOYMENT + "max_retries"
    error2_path = AZURE_OPENAI_DEPLOYMENT + "other_question"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": source_code}
                ],
                temperature = 0.1,
                max_tokens = 1024
            )
            return response.choices[0].message.content.strip()
            
        except RateLimitError as e:
            wait_time = 60
            print(e)
            time.sleep(wait_time)
                
        except Exception as e:
            print(f"{e}")
            tex =  package_name+" "+str(e)
            log_error_package(tex, error2_path)
            return None

    log_error_package(package_name, error1_path)
    return None


def append_result_to_json(package_name: str, llm_response_text: str, output_file: Path):
    """
    add one LLM response to JSON
    """
    try:
        lines = llm_response_text.strip().split('\n')
        verdict = lines[0].replace("Verdict:", "").strip()
        reasoning = "\n".join(lines[1:]).replace("Reasoning:", "").strip()
        
        new_entry = {
            "packagename": package_name,
            "Verdict": verdict,
            "Reasoning": reasoning
        }
    except IndexError:
        print(f"{llm_response_text}")
        new_entry = {"packagename": package_name, "raw_response": llm_response_text}
        verdict = None

    all_results = []
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning:{output_file}")
                all_results = []
    all_results.append(new_entry)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    return verdict



def main_pipeline(base_dir: str, results_base_dir: str, max_workers: int = 10):
    base_path = Path(base_dir)
    results_dir = Path(results_base_dir)
    temp_extract_base = Path("./temp_extraction")
    results_dir.mkdir(exist_ok=True)
    temp_extract_base.mkdir(exist_ok=True)

    processed_packages = set()
    output_file = Path("XXXX")
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
                processed_packages = {item['packagename'] for item in existing_data if 'packagename' in item}
            except json.JSONDecodeError:
                pass

    all_package_folders = [p for p in base_path.iterdir() if p.is_dir()]
    unprocessed_folders = [p for p in all_package_folders if p.name not in processed_packages]

    if not unprocessed_folders:
        return

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(results_dir,)
    ) as executor:
        futures = [
            executor.submit(process_package_to_jsonl, folder, temp_extract_base) 
            for folder in unprocessed_folders
        ]
        
        progress_bar = tqdm(concurrent.futures.as_completed(futures), total=len(unprocessed_folders), desc="concurrent processing")
        
        success_count = 0
        fail_count = 0
        for future in progress_bar:
            if future.result():
                success_count += 1
            else:
                fail_count += 1
            progress_bar.set_postfix_str(f"success:{success_count}, fail:{fail_count}")

    print(f"\nsuccess:{success_count}, fail:{fail_count}")

if __name__ == '__main__':
    output_file = "output/malicious_results.jsonl"
    BASE_MALICIOUS_DIR = "path/to/malicious_dataset"
    total, malicious, benign ,error= main_pipeline(BASE_MALICIOUS_DIR, output_file)