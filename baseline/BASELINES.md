# Baselines

This doc lists the baseline scripts included with the project and how to run them. Each baseline is isolated and **does not require** the main GNN pipeline.

> **Placeholders**: All paths/API keys are shown as `path/to/data` or `your_api_key`. Replace them locally. Do **not** commit secrets.

## B0 — LLM Full‑Code Judgement (Azure OpenAI)
Classify a package by feeding **all source code** to an Azure model.

- **File**: `baseline_LLM_malicious.py`  
- **Inputs**: nested packages root; Azure envs `AZURE_OPENAI_API_KEY`, `ENDPOINT_URL`, `DEPLOYMENT_NAME`, `AZURE_OPENAI_API_VERSION`.  
- **Outputs**: per‑worker JSONL files in `results_base_dir`; each line has `{packagename, Verdict, Reasoning}`.  
- **Run**:
  ```bash
  pip install openai tqdm
  export AZURE_OPENAI_API_KEY=... ENDPOINT_URL=... DEPLOYMENT_NAME=... AZURE_OPENAI_API_VERSION=...
  # edit BASE_MALICIOUS_DIR and results path variables ("XXXX") near the bottom
  python baselin_LLM_malicious.py
  ```
  *Notes*: The script retries on rate limits; it extracts archives safely and streams results to JSONL.

## B1 — Bandit Static Scanner
Run [Bandit](https://bandit.readthedocs.io/) on extracted sources.

- **File**: `bandit_scanner.py`  
- **Inputs**: either a nested packages root (each package in its own folder with one archive inside) or a flat folder of archives.  
- **Outputs**: one JSON report per package in `output_base`, plus console/log stats.  
- **Run**:
  ```bash
  pip install bandit tqdm
  # edit CONFIGS at bottom: base_dir/output_base/structure_type ("nested" or "flat")
  python bandit_scanner.py
  ```

## B2   — OSSGadget `detect-backdoor` Wrapper
Use Microsoft OSSGadget to scan packages via a local HTTP server for each worker.

- **File**: `scan_package_malicious.py`  
- **Inputs**: nested packages root; `ossgadget` CLI in PATH.  
- **Outputs**: one `.txt` per package with STDOUT/STDERR of the tool; ANSI cleaned.  
- **Run**:
  ```bash
  # install https://github.com/microsoft/OSSGadget (detect-backdoor)
  # edit INPUT_DIR / OUTPUT_DIR / WORKER_ROOT_DIR placeholders ("XXXX"); adjust NUM_PROCESSES/BASE_PORT if needed
  python scan_package_malicious.py
  ```

## B3 — VirusTotal Batch Scanner
Batch-scan artifacts with VirusTotal.

- **Files**: `main_concurrent_scanner.py`, `vt_scanner.py`  
- **Inputs**: a flat folder of archives (`.whl/.zip/.tar.gz`); VT API keys.  
- **Outputs**: JSONL files like `vt_results_part*.jsonl`.  
- **Run**:
  1) Edit `BASE_DIRECTORY` in `main_concurrent_scanner.py` and `CONFIGS` (set individual `api_key`, `output_file`).  
  2) In `vt_scanner.py`, the call to `discover_packages(..., results_jsonl="XXXX")` is used to skip processed packages — set it to an existing results file or leave `"XXXX"` if you don't want skipping.  
  3) Execute:
  ```bash
  pip install vt tqdm aiohttp
  python main_concurrent_scanner.py
  ```

## B4 — LLM on Call‑Graph Report (Local/OAI‑compatible)
Summarize `call_graph.json` into a textual report and ask a local/OpenAI‑compatible endpoint for a verdict.

- **File**: `LLama3_call_function.py`  
- **Inputs**: dataset root where each package folder contains a `call_graph.json`; OAI‑compatible endpoint/base URL and API key.  
- **Outputs**: a JSONL results file (path set by the `output_file` placeholder).  
- **Run**:
  ```bash
  pip install openai
  # edit base_url/api_key and base_dir/output_file placeholders ("XXXX") in the script
  python LLama3_call_function.py
  ```

## B5 — Token Budget Estimators
Quickly estimate prompt sizes before calling LLMs.

- **`cal_token.py`** — counts tokens using `tiktoken` for an Azure/OpenAI model name (default `"gpt-35-turbo"`).  
  **Inputs**: a directory of packages (nested structure).  
  **Outputs**: per‑worker JSONL files `token_counts_*.jsonl` under the given output dir.  
  **Run**:
  ```bash
  # edit BASE_MALICIOUS_DIR / OUTPUT_DIR in the script (placeholders "XXXX")
  python cal_token.py
  ```

- **`cal_token_local.py`** — counts tokens using a local HuggingFace tokenizer.  
  **Inputs**: same as above; `--model-id` selects a tokenizer (e.g., `meta-llama/Meta-Llama-3-8B`).  
  **Outputs**: per‑worker JSONL under `OUTPUT_DIR/<model-id>/`.  
  **Run**:
  ```bash
  python cal_token_local.py --model-id meta-llama/Meta-Llama-3-8B --max-workers 8
  # edit BASE_MALICIOUS_DIR / OUTPUT_DIR in the script
  ```
---

### Evaluation Notes
- Use the **same split and vocab** as the main method when comparing (where applicable).  
