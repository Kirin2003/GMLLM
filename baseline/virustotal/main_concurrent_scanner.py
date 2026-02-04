import multiprocessing
import vt_scanner

# basedir for malicious or benign
BASE_DIRECTORY = "path/to/package_data"
CONFIGS = [
    {
    "api_key": "your_api_key_here",
    "output_file": "output/vt_results_part1.jsonl",
    },
# Add more configs as needed
]

def worker(config,i):
    vt_scanner.run_analysis(
        api_key=config["api_key"],
        base_dir=BASE_DIRECTORY,
        output_file=config["output_file"],
        start_index=i,
        count=50,
        batch_size=5)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":
    print(f"Starting {len(CONFIGS)} analysis tasks in batches of 5...")

    batch_size = 5
    i = 0  
    hei = 0
    for batch in chunks(CONFIGS, batch_size):
        print(f"Starting batch with {len(batch)} tasks...")
        processes = []  
        for config in batch:
            process = multiprocessing.Process(target=worker, args=(config, i,))
            i += 50
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        print(f"Batch completed.")
        break

    print("\n--- All analysis tasks have completed! ---")
