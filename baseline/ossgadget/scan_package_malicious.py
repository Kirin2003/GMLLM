import os
import re
import shutil
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path
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

def find_tasks(root_dir: Path,completed_packages:set):
    tasks = []
    package_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    print(len(package_dirs))
    for package_dir in package_dirs:
        version_dirs = sorted([d for d in package_dir.iterdir() if d.is_dir()])
        if not version_dirs:
            continue
        
        first_version_dir = version_dirs[0]
        
        package_files = list(first_version_dir.glob('*.tar.gz')) + \
                        list(first_version_dir.glob('*.whl')) + \
                        list(first_version_dir.glob('*.zip'))
        
        if package_files:
            filename = package_files[0].name
            last_dash_index = filename.rfind('-')
            if last_dash_index > 0:
                package_key = filename[:last_dash_index]
            else:
                package_key = filename.split('.')[0]
            if package_key in completed_packages:
                continue
            tasks.append(package_files[0])
    return tasks


def get_completed_packages(output_dir: Path) -> set:
    completed = set()
    for file_path in output_dir.glob("*.txt"):
        stem = file_path.stem
        last_dash_index = stem.rfind('-')
        if last_dash_index > 0:
            package_key = stem[:last_dash_index]
            completed.add(package_key)
        else:
            completed.add(stem)
    print(len(completed))
    return completed

def process_package(package_file: Path,worker_config:dict):
    worker_id = worker_config['id']
    worker_dir = Path(worker_config['dir'])
    port = worker_config['port']
    
    package_name = package_file.parent.parent.name
    version_name = package_file.parent.name
    filename = package_file.name
    
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
    cp = get_completed_packages(OUTPUT_DIR)
    tasks = find_tasks(INPUT_DIR,cp)
    tasks = sorted(tasks)
    if not tasks:
        return
    print(len(tasks))
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
                print(f"Port: {port}")
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

    try:
        with Pool(processes=NUM_PROCESSES) as pool:
            pool.starmap(process_package, task_args)
    finally:
        pool.close()
        pool.join()
        for proc in server_processes:
            proc.terminate()
            proc.wait()
        shutil.rmtree(WORKER_ROOT_DIR)

if __name__ == "__main__":
    main()