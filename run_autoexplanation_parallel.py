import subprocess
import argparse
import sys
import os

def main():
    """
    A master script to launch multiple worker processes for the explanation pipeline,
    with GPU allocation and dataset type selection.
    """
    parser = argparse.ArgumentParser(
        description="Launch N parallel workers for the GNN explanation task."
    )
    parser.add_argument(
        "-n", "--num-workers",
        type=int,
        required=True,
        help="The total number of parallel workers to launch."
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="A comma-separated list of GPU IDs to use (e.g., '0,1,2,3')."
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        choices=['malicious', 'normal'],
        help="The type of dataset to process ('malicious' or 'normal')."
    )
    args = parser.parse_args()

    num_workers = args.num_workers
    if num_workers <= 0:
        print("Error: Number of workers must be a positive integer.", file=sys.stderr)
        exit(1)
        
    # --- NEW: Process the list of GPU IDs ---
    gpu_ids = [int(gid.strip()) for gid in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    print(f"Available GPUs for workers: {gpu_ids}")

    worker_script = "auto_explanation.py"

    if not os.path.exists(worker_script):
        print(f"Error: Worker script '{worker_script}' not found.", file=sys.stderr)
        exit(1)

    print(f"--- Starting {num_workers} parallel workers for '{args.dataset_type}' dataset ---")

    processes = []
    for worker_id in range(num_workers):
        # --- NEW: Assign a GPU to this worker using round-robin ---
        assigned_gpu = gpu_ids[worker_id % num_gpus]

        command = [
            sys.executable,
            worker_script,
            "--worker-id", str(worker_id),
            "--total-workers", str(num_workers),
            "--gpu-id", str(assigned_gpu), # Pass the assigned GPU ID
            "--dataset-type", args.dataset_type # Pass the dataset type
        ]
        
        print(f"  > Launching Worker {worker_id} on GPU {assigned_gpu} with command: {' '.join(command)}")
        
        log_file_path = f"./rebuttal_50percent/worker_{worker_id}_output_malicious.log"
        with open(log_file_path, 'w') as log_file:
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((process, log_file_path))

    print(f"\n--- All {num_workers} workers launched. Waiting for them to complete... ---")
    print("You can monitor the individual worker logs (e.g., worker_0_output.log) for progress.")

    # Wait for all processes to finish
    for process, log_file_path in processes:
        process.wait()
        if process.returncode == 0:
            print(f"  > Worker with log '{log_file_path}' finished successfully.")
        else:
            print(f"  > Worker with log '{log_file_path}' finished with an error (return code {process.returncode}). Please check the log for details.")

    print("\n--- All parallel processes have completed. ---")


if __name__ == "__main__":
    main()
    #python run_autoexplanation_parallel.py --num-workers 8 --gpus 0,1 --dataset-type normal
    #python run_autoexplanation_parallel.py --num-workers 8 --gpus 0,1 --dataset-type malicious

    #nohup python run_autoexplanation_parallel.py --num-workers 8 --gpus 0,1 --dataset-type normal > rebuttal_human_benign.log 2>&1 &