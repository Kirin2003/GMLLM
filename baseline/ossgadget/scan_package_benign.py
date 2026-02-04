import os
import re
import shutil
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path
from tqdm.contrib.concurrent import process_map
import socket
import sys

NUM_PROCESSES = 6
INPUT_DIR = Path("path/to/input_packages")
TEMP_ROOT_DIR = Path("path/to/temp_dir")
OUTPUT_DIR = Path("path/to/output_results")
WORKER_ROOT_DIR = Path("path/to/worker_tmp")
BASE_PORT = 8010


def clean_ansi_codes(text: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def find_tasks(root_dir: Path):
    tasks = []
    patterns = ["*.tar.gz", "*.whl", "*.zip"]
    
    for pattern in patterns:
        tasks.extend(root_dir.glob(pattern))

    tasks = [f for f in tasks if f.is_file()]

    return sorted(tasks)

def extract_package_name_from_filename(filename: str) -> str:
    name_part = Path(filename).name
    while '.' in name_part:
        new_name_part = Path(name_part).stem
        if new_name_part == name_part:
             break
        name_part = new_name_part

    parts = name_part.split('-')
    name_parts = []
    for part in parts:
        if re.match(r'^\d', part):
            break
        name_parts.append(part)

    if not name_parts:
        return name_part
        
    return '-'.join(name_parts)


def process_package(package_file: Path,worker_config:dict):
    worker_id = worker_config['id']
    worker_dir = Path(worker_config['dir'])
    port = worker_config['port']

    filename = package_file.name
    package_name = extract_package_name_from_filename(filename)
    version_name = "1.0.0"
    
    copied_file_path = None
    try:
        copied_file_path = worker_dir / filename
        shutil.copy(package_file, copied_file_path)

        url = f"http://localhost:{port}/{filename}"
        purl = f"pkg:url/{package_name}@{version_name}?url={url}"
        output_file = OUTPUT_DIR / f"{filename}.txt"
        command = ["ossgadget", "detect-backdoor", purl]
        
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        
        stdout_clean = clean_ansi_codes(result.stdout)
        stderr_clean = clean_ansi_codes(result.stderr)
        full_output = f"--- STDOUT ---\n{stdout_clean}\n\n--- STDERR ---\n{stderr_clean}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_output)
        print(f"[Worker {worker_id}] finished.")

    except Exception as e:
        print(e)
    finally:
        if copied_file_path and copied_file_path.exists():
            os.remove(copied_file_path)



def main():
    if not INPUT_DIR.is_dir():
        return
    OUTPUT_DIR.mkdir(exist_ok=True)
    if WORKER_ROOT_DIR.exists():
        shutil.rmtree(WORKER_ROOT_DIR)
    WORKER_ROOT_DIR.mkdir()

    tasks = find_tasks(INPUT_DIR)
    tasks = sorted(tasks)
    if not tasks:
        return

    worker_configs = [
        {
            'id': i,
            'dir': str(WORKER_ROOT_DIR / f'worker_{i}'),
            'port': BASE_PORT + i
        }
        for i in range(NUM_PROCESSES)
    ]

    server_processes = []
    for config in worker_configs:
        server_dir = Path(config['dir'])
        server_dir.mkdir()
        server_cmd = ["python3", "-m", "http.server", str(config['port'])]
        proc = subprocess.Popen(
            server_cmd,
            cwd=server_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        server_processes.append(proc)
    
    time.sleep(3)

    all_servers_ok = True
    for i, proc in enumerate(server_processes):
        port = worker_configs[i]['port']
        if proc.poll() is not None:
            all_servers_ok = False
            continue

        try:
            with socket.create_connection(("localhost", port), timeout=2):
                print(f"port {port}")
        except (socket.timeout, ConnectionRefusedError):
            all_servers_ok = False

    if not all_servers_ok:
        for proc in server_processes:
            proc.terminate()
            proc.wait()
        sys.exit(1)

    task_args = []
    for i, package_file in enumerate(tasks):
        worker_config = worker_configs[i % NUM_PROCESSES]
        task_args.append((package_file, worker_config))

    # Create and run the process pool.
    try:
        print(f"Found {len(tasks)} packages. Distributing tasks to {NUM_PROCESSES} workers...")
        list(process_map(
            process_package,
            [args[0] for args in task_args],
            [args[1] for args in task_args],
            max_workers=NUM_PROCESSES,
            chunksize=1,
            desc="Processing packages",
            unit="pkg"
        ))
    except Exception as e:
        print(e)
    finally:
        for proc in server_processes:
            proc.terminate()
            proc.wait()
        try:
            shutil.rmtree(WORKER_ROOT_DIR)
        except Exception as e:
            print(f"[WARNING]: {e}")

if __name__ == "__main__":
    main()