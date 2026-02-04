import vt
import json
import hashlib
import time
from pathlib import Path
from aiohttp import ClientTimeout
import re
def extract_package_name_from_filename(filename: str) -> str:
    name_part = Path(filename).name
    while '.' in name_part:
        new_name_part = Path(name_part).stem
        if new_name_part == name_part: break
        name_part = new_name_part
    parts = name_part.split('-')
    name_parts = []
    for part in parts:
        if re.match(r'^\d', part):
            break
        name_parts.append(part)
    return '-'.join(name_parts) if name_parts else name_part
def discover_packages(base_dir, results_jsonl):
    base_path = Path(base_dir)
    results_path = Path(results_jsonl)
    package_files = []
    analyzed_packages = set()
    if results_path.exists():
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            pkg_name = data.get("package_name")
                            if pkg_name:
                                analyzed_packages.add(pkg_name)
                        except json.JSONDecodeError:
                            continue  
        except Exception as e:
            print(e)
    all_files = sorted([p for p in base_path.iterdir() if p.is_file()])
    for file_path in all_files:
        if file_path.name.endswith(('.zip', '.tar.gz', '.whl', '.tar')):
            package_name = extract_package_name_from_filename(file_path.name)
            if package_name in analyzed_packages:
                continue  
            package_files.append((package_name, file_path))
            analyzed_packages.add(package_name)
    return package_files
def analyze_local_pypi_package_batch(apikey, base_dir, output_jsonl, start_index=0, count=None, batch_size=10):
    all_files = discover_packages(base_dir, results_jsonl="output/virus_total_scan.jsonl")
    if not all_files:
        print("No new package files to analyze. Exiting.")
        return
    end_index = len(all_files) if count is None else min(start_index + count, len(all_files))
    files_to_process = all_files[start_index:end_index]
    print(f"Processing {len(files_to_process)} files (from index {start_index} to {end_index - 1}).")
    file_mode = 'a'
    results_batch = []
    def check_report_exists(client, sha256):
        try:
            obj = client.get_object(f"/files/{sha256}")
            return True, obj
        except vt.error.APIError as e:
            if e.code == "NotFoundError":
                return False, None
            else:
                return False, e
        except Exception as e:
            return False, e
    try:
        with open(output_jsonl, file_mode, encoding='utf-8') as output_file, vt.Client(apikey, timeout=300) as client:
            for i, (package_name, file_path) in enumerate(files_to_process, start=start_index):
                print(f"\n--- Processing [{i}] {package_name} ---")
                full_path = str(file_path)
                file_sha256 = None
                try:
                    sha256_hash = hashlib.sha256()
                    with open(full_path, "rb") as f_in:
                        for chunk in iter(lambda: f_in.read(4096), b""):
                            sha256_hash.update(chunk)
                    file_sha256 = sha256_hash.hexdigest()
                except Exception as e:
                    print(f"  Error reading file {file_path.name}: {e}")
                    results_batch.append({
                        "index": i, "package_name": package_name, "file_name": file_path.name,
                        "verdict": "error", "error": f"File read error: {str(e)}",
                        "sha256": None, "raw_response": None
                    })
                    continue
                obj = None
                source = None
                exists, result = check_report_exists(client, file_sha256)
                if exists:
                    obj = result
                    source = "existing_report"
                    print(f" Found existing report in VirusTotal.")
                else:
                    if result is not None:
                        print(f" Unexpected error during lookup: {result}")
                        results_batch.append({
                            "index": i, "package_name": package_name, "file_name": file_path.name,
                            "verdict": "error", "error": f"Lookup failed: {str(result)}",
                            "sha256": file_sha256, "raw_response": None
                        })
                        continue  
                    print(f" No existing report found. Uploading {file_path.name}...")
                    try:
                        with open(full_path, "rb") as f_upload:
                            analysis = client.scan_file(f_upload, wait_for_completion=True)
                        print(f"  Scan wait completed. Status: {analysis.status}")
                        if analysis.status != "completed":
                            raise Exception(f"Analysis status not completed: {analysis.status}")
                        max_retries = 6
                        wait_seconds = 15
                        for attempt in range(max_retries):
                            try:
                                obj = client.get_object(f"/files/{file_sha256}")
                                source = "uploaded_report"
                                print(f" Report available after upload.")
                                break
                            except vt.error.APIError as e:
                                if e.code == "NotFoundError" and attempt < max_retries - 1:
                                    print(f"  Report not ready, waiting {wait_seconds}s...")
                                    time.sleep(wait_seconds)
                                else:
                                    raise e
                            except Exception as e:
                                raise e
                        else:
                            raise Exception(f"Report not available after {max_retries} retries.")
                    except Exception as e:
                        print(f"  Error during upload/analysis: {e}")
                        results_batch.append({
                            "index": i, "package_name": package_name, "file_name": file_path.name,
                            "verdict": "error", "error": str(e), "sha256": file_sha256, "raw_response": None
                        })
                        continue
                if obj is not None:
                    try:
                        stats = obj.last_analysis_stats
                        malicious_count = stats.get("malicious", 0)
                        total_engines = sum(stats.values())
                        verdict = "malicious" if malicious_count > 0 else \
                                  "suspicious" if stats.get("suspicious", 0) > 0 else "clean"
                        results_batch.append({
                            "index": i,
                            "package_name": package_name,
                            "file_name": file_path.name,
                            "verdict": verdict,
                            "malicious_count": malicious_count,
                            "total_engines": total_engines,
                            "sha256": file_sha256,
                            "source": source,  
                            "raw_response": obj.to_dict()
                        })
                        print(f"  Result: {verdict} ({malicious_count}/{total_engines})")
                    except Exception as e:
                        print(f"  Error parsing report: {e}")
                        results_batch.append({
                            "index": i, "package_name": package_name, "file_name": file_path.name,
                            "verdict": "error", "error": f"Parsing error: {str(e)}",
                            "sha256": file_sha256, "raw_response": None
                        })
                if len(results_batch) >= batch_size:
                    print(f"--- Writing batch of {len(results_batch)} results ---")
                    for entry in results_batch:
                        output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    output_file.flush()
                    results_batch.clear()
                print("--- Waiting for 16 seconds (rate limit) ---")
                time.sleep(16)
            if results_batch:
                print(f"--- Writing final batch of {len(results_batch)} results ---")
                for entry in results_batch:
                    output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                output_file.flush()
                results_batch.clear()
    except IOError as e:
        print(f"Error opening output file {output_jsonl}: {e}")
    except Exception as e:
        print(f"Unexpected error in batch processing: {e}")
    print(f"\nBatch processing completed. Results saved to {output_jsonl}")
def run_analysis(api_key, base_dir, output_file, start_index, count, batch_size):
    print(f"--- Starting analysis task for {output_file} ---")
    analyze_local_pypi_package_batch(
        apikey=api_key,
        base_dir=base_dir,
        output_jsonl=output_file,
        start_index=start_index,
        count=count,
        batch_size=batch_size
    )