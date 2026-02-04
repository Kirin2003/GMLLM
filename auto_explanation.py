import subprocess
import os
import time
import datetime
import json
from tqdm import tqdm

import argparse

with open("config.json", 'r') as f:
    config = json.load(f)
# --- NEW: Add Argument Parser for Parallelization ---
parser = argparse.ArgumentParser(description="Run the explanation pipeline in parallel.")
parser.add_argument("--worker-id", type=int, default=0, help="ID of this worker (e.g., 0, 1, 2...).")
parser.add_argument("--total-workers", type=int, default=1, help="Total number of workers.")
parser.add_argument("--gpu-id", type=int,default=config["cuda_device_id"], help="GPU device ID to use.")
parser.add_argument("--dataset-type", type=str, required=True, choices=['malicious', 'normal'], help="Type of dataset to process.")

args = parser.parse_args()

# Use the arguments
WORKER_ID = args.worker_id
TOTAL_WORKERS = args.total_workers


BASE_GRAPH_DATA_DIR = config["base_graph_data_dir"]
DATASET_TYPE = args.dataset_type
if DATASET_TYPE == "malicious":
    HANDLED_DIR = os.path.join(BASE_GRAPH_DATA_DIR, config["handled_malicious_dir"])
    EXPLAIN_LOGS_ROOT_DIR = config["explain_logs_malicious"]
else:
    HANDLED_DIR = os.path.join(BASE_GRAPH_DATA_DIR, config["handled_normal_dir"])
    EXPLAIN_LOGS_ROOT_DIR = config["explain_logs_normal"]

# CKPT_DIR = config["checkpoint_dir"]
CKPT_DIR = os.path.join(config["checkpoint_dir"], f"worker_{WORKER_ID}") 
MODEL_WEIGHTS_FILE = config["model_weights_file"]
CUDA_DEVICE_ID = args.gpu_id
VOCAB_DIR = os.path.join(BASE_GRAPH_DATA_DIR, config["vocab_dir"])

EXPLAINER_CKPT_FILENAME = "single_graph_ckpt.pth.tar"


# 各个脚本的路径
BUILD_CKPT_SCRIPT = "build_single_graph_ckpt.py"
EXPLAINER_MAIN_SCRIPT = "explainer_main.py"
EXTRACT_TO_JSON_SCRIPT = "extract2Jsontrue.py" 

# --- 日志文件配置 ---
# 创建一个带时间戳的日志文件名
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# GLOBAL_LOG_FILE = f"explanation_normal_{timestamp}.log"
GLOBAL_LOG_FILE = f"explanation_normal_{timestamp}_worker_{WORKER_ID}.log"
GLOBAL_LOG_PATH = os.path.join(EXPLAIN_LOGS_ROOT_DIR, GLOBAL_LOG_FILE)


# --- 辅助函数：将输出同时打印到控制台和日志文件 ---
def log_message(message, level="INFO"):
    log_entry = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}"
    print(message) # 打印到控制台
    with open(GLOBAL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")


PROCESSED_FILE = os.path.join(EXPLAIN_LOGS_ROOT_DIR, "processed.log")  # 记录已处理的 file_base_name
def read_processed_files():
    if not os.path.exists(PROCESSED_FILE):
        return set()
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def write_processed_file(file_base_name):
    with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
        f.write(file_base_name + "\n")
processed_set = read_processed_files()
log_message(f"Loaded {len(processed_set)} already processed files from {PROCESSED_FILE}")


# --- 确保必要的目录存在 ---
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(EXPLAIN_LOGS_ROOT_DIR, exist_ok=True)
# 初始化日志文件
with open(GLOBAL_LOG_PATH, "w", encoding="utf-8") as f:
    f.write(f"--- Explanation Automation Log Start: {timestamp} ---\n")
log_message(f"Global log file created at: {GLOBAL_LOG_PATH}")

# --- 遍历 .pt 文件并执行流程 ---
if not os.path.exists(HANDLED_DIR):
    log_message(f"Error: Directory not found: {HANDLED_DIR}", level="ERROR")
    exit()

# 获取所有 .pt 文件，并按文件名排序，确保处理顺序
# pt_files = [
#     "cpulib.pt",
#     "cpuad.pt",
#     "cpucandy.pt",
#     "edre.pt" 
# ]
pt_files = [f for f in os.listdir(HANDLED_DIR) if f.endswith(".pt")]
pt_files = sorted(pt_files)

if not pt_files:
    log_message(f"No .pt files found in {HANDLED_DIR}", level="WARNING")
    exit()

# # 过滤：只保留未处理的
remaining_files = []
for f in pt_files:
    file_base_name = os.path.splitext(f)[0]
    if file_base_name not in processed_set:
        remaining_files.append(f)

log_message(f"Total files: {len(pt_files)}")
log_message(f"Already processed: {len(processed_set)}")
log_message(f"Remaining to process: {len(remaining_files)}")
if not remaining_files:
    log_message("All files have been processed. Nothing to do.")
    exit()

num_files = len(remaining_files)
files_per_worker = (num_files + TOTAL_WORKERS - 1) // TOTAL_WORKERS
start_index = WORKER_ID * files_per_worker
end_index = min((WORKER_ID + 1) * files_per_worker, num_files)

my_pt_files = remaining_files[start_index:end_index]

log_message(f"Total files available: {num_files}")
log_message(f"Worker {WORKER_ID}/{TOTAL_WORKERS} assigned {len(my_pt_files)} files to process (from index {start_index} to {end_index} of remaining files).")


with tqdm(total=len(my_pt_files), desc="Processing .pt files", unit="file", ncols=100, colour='green') as pbar:
    for pt_filename in my_pt_files:
        # 提取文件基础名 (不带 .pt 后缀)，例如 "15cent"
        file_base_name = os.path.splitext(pt_filename)[0] 
        
        # 构建当前图的日志目录，例如 "./explain_logs/graph_15cent"
        current_graph_log_dir = os.path.join(EXPLAIN_LOGS_ROOT_DIR, f"graph_{file_base_name}")
        if os.path.exists(current_graph_log_dir):
            log_message(f"Output directory for {file_base_name} already exists. Skipping.")
            pbar.update(1)
            continue # Skips to the next file
        os.makedirs(current_graph_log_dir, exist_ok=True)

        log_message(f"\n--- Processing {pt_filename} ({file_base_name}) ---")
        success = True
        # --- Step 1: Run build_single_graph_ckpt.py ---
        # build_single_graph_ckpt.py 内部会根据 file_base_name 构建完整的 .pt 路径
        ckpt_path = os.path.join(CKPT_DIR,EXPLAINER_CKPT_FILENAME)
        vocab_path = os.path.join(BASE_GRAPH_DATA_DIR,VOCAB_DIR)
        build_ckpt_cmd = [
            "python", BUILD_CKPT_SCRIPT,
            "--pt_filename", file_base_name, # 传递文件名基础 (不带 .pt 后缀)
            "--model_weights", MODEL_WEIGHTS_FILE,
            "--vocab_dir", vocab_path,
            "--output_ckpt_dir", ckpt_path,
            "--device", f"cuda:{CUDA_DEVICE_ID}",
            "--handled_dir",HANDLED_DIR
        ]
        log_message(f"Executing: {' '.join(build_ckpt_cmd)}")
        try:
            result = subprocess.run(build_ckpt_cmd, check=True, capture_output=True, text=True)
            log_message("build_single_graph_ckpt.py completed successfully.")
            log_message(f"STDOUT:\n{result.stdout.strip()}")
            # if result.stderr:
                # log_message(f"STDERR:\n{result.stderr.strip()}", level="WARNING")
        except subprocess.CalledProcessError as e:
            log_message(f"Error running {BUILD_CKPT_SCRIPT} for {pt_filename}:", level="ERROR")
            log_message(f"STDOUT:\n{e.stdout.strip()}", level="ERROR")
            log_message(f"STDERR:\n{e.stderr.strip()}", level="ERROR")
            log_message(f"Skipping {pt_filename} due to error in {BUILD_CKPT_SCRIPT}.", level="ERROR")
            success = False
        
        if not success:
            pbar.update(1)
            continue

        # --- Step 2: Run explainer_main.py ---
        explainer_main_cmd = [
            "python", EXPLAINER_MAIN_SCRIPT,
            "--ckptdir", CKPT_DIR, # Explainer 会从这个目录加载检查点
            "--logdir", current_graph_log_dir, # Explainer 的输出日志和 .npy 掩码文件会存到这里
            "--graph-mode",
            "--graph-idx", "0", # 每次只处理一个图，所以总是0
            "--mask-act", "sigmoid",
            "--cuda", str(CUDA_DEVICE_ID),
            "--gpu",
            "--no-writer"
        ]
        log_message(f"Executing: {' '.join(explainer_main_cmd)}")
        start_time = time.perf_counter()
        try:
            result = subprocess.run(explainer_main_cmd, check=True, capture_output=True, text=True)
            elapsed = time.perf_counter() - start_time
            log_message(
                f"[EXPLAINER_RUNTIME] graph={pt_filename} time_ms={elapsed*1000:.2f}"
            )
            for line in result.stdout.splitlines():
                if "[EXPLAINER_STEPS]" in line:
                    log_message(line) 
                    break

            log_message("explainer_main.py completed successfully.")
            # log_message(f"STDOUT:\n{result.stdout.strip()}")
            if result.stderr:
                log_message(f"STDERR:\n{result.stderr.strip()}", level="WARNING")
        except subprocess.CalledProcessError as e:
            log_message(f"Error running {EXPLAINER_MAIN_SCRIPT} for {pt_filename}:", level="ERROR")
            log_message(f"STDOUT:\n{e.stdout.strip()}", level="ERROR")
            log_message(f"STDERR:\n{e.stderr.strip()}", level="ERROR")
            log_message(f"Skipping {pt_filename} due to error in {EXPLAINER_MAIN_SCRIPT}.", level="ERROR")
            success = False
        
        if not success:
            pbar.update(1)
            continue

        # --- Step 3: Run extract2Json.py ---
        # extract2Json.py 内部会根据 file_base_name 构建所有必要的输入文件路径和输出目录
        extract_json_cmd = [
            "python", EXTRACT_TO_JSON_SCRIPT,
            "--graph_base_filename", file_base_name,
            "--log_dir",EXPLAIN_LOGS_ROOT_DIR,
            "--base_explain_logs",BASE_GRAPH_DATA_DIR
            # ex: "--k_edge_threshold", "0.1", "--max_elements_saved", "100"
        ]
        log_message(f"Executing: {' '.join(extract_json_cmd)}")
        try:
            result = subprocess.run(extract_json_cmd, check=True, capture_output=True, text=True)
            log_message("extract2Json.py completed successfully.")
            log_message(f"STDOUT:\n{result.stdout.strip()}")
            if result.stderr:
                log_message(f"STDERR:\n{result.stderr.strip()}", level="WARNING")
        except subprocess.CalledProcessError as e:
            log_message(f"Error running {EXTRACT_TO_JSON_SCRIPT} for {pt_filename}:", level="ERROR")
            log_message(f"STDOUT:\n{e.stdout.strip()}", level="ERROR")
            log_message(f"STDERR:\n{e.stderr.strip()}", level="ERROR")
            log_message(f"Skipping {pt_filename} due to error in {EXTRACT_TO_JSON_SCRIPT}.", level="ERROR")
            success = False
        
        if not success:
            pbar.update(1)
            continue
        
        write_processed_file(file_base_name)
        log_message(f"--- Finished processing {pt_filename} ({file_base_name}) ---")
        pbar.set_postfix(file=file_base_name)
        pbar.update(1)
        time.sleep(0.5)

log_message(f"\n--- Worker {WORKER_ID} completed all assigned tasks ---")