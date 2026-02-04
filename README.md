# Supplementary Material for Anonymous Review

This repository provides an end-to-end pipeline for detecting sensitive/malicious behavior in Python packages using **static analysis**, **LLM-synthesized ruleset**, **graph construction**, **GNN training**, **GNNExplainer** and **LLM subgraph review**.

The project is organized as three modules plus a shared `requirements.txt`:

> **Privacy/Double-blind note**: All sample configs use `XXXX` placeholders instead of absolute local paths.

---

## 0) Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
The pipeline runs offline. If no API key is present, it automatically falls back to the local rule base.

## 1) Build call_graph.json (extract stage)
### 1.1 Configure
Edit `extractAST/extract_config.json` (keep XXXX placeholders or change to your paths):
```bash
{
  "paths": {
    "src_dir": "XXXX",
    "out_dir": "XXXX",
    "synth_rules_out": "synth_rules.json",
    "cache_file": "detector_cache.json"
  },
  "llm": {
    "provider": "openai\azure\local",
    "model_name": "Model you want to use",
    "temperature": 0.0,
    "max_retries": 3,
    "timeout_s": 30.0,
    "auto_synthesize": true
  },
  "detector": {
    "use_rule_fallback": true
  }
}
```

### `extract_config.json` — field meanings

**paths**
- `src_dir`: Default source root (used when `--src` is not provided).
- `out_dir`: Default output root (used when `--out` is not provided).
- `synth_rules_out`: Path to save synthesized rules (`synth_rules.json`); reused if present.
- `cache_file`: Cache file for LLM calls (kept for backward compatibility; rarely used in the current flow).

**llm** *(no API keys here; keys are read from environment variables)*
- `provider`: LLM provider (`openai` or `azure`).
- `model_name`: Model used for **rule synthesis** (not per-node labeling).
- `temperature`: Sampling temperature (recommend `0.0` for stability).
- `max_retries`: Retry times on synthesis failure.
- `timeout_s`: Timeout per API request (seconds).
- `auto_synthesize`: If an API key is detected, first **synthesize rules → apply** for this run.

**detector**
- `use_rule_fallback`: If synthesis is unavailable or fails, fall back to local rules in `rules_fallback.py`.

- With API keys set (OpenAI or Azure), the extractor will first ask the LLM to synthesize a ruleset via PROMPT_COMM, save it to synth_rules.json, safely compile each lambda n: ..., and apply it to the whole project.

- If API is missing or synthesis fails, it falls back to rules_fallback.py.

- We do not call the LLM per node at labeling time; we only use the LLM once to generate rules.

### 1.2 Run extractor
**With API (recommended path: synthesize → apply → label locally)**
```bash
# Choose one provider
export OPENAI_API_KEY=...            # or: export AZURE_OPENAI_API_KEY=...
python extractAST/cli_extract.py \
  --config-extract extractAST/extract_config.json \
  --src /path/to/one_python_project \
  --out /path/to/output_dir_for_this_project
# → writes call_graph.json into the given --out directory
# → writes synth_rules.json at paths.synth_rules_out (if enabled)
```
**Without API (fully offline; local fallback rules)**
```bash
python extractAST/cli_extract.py \
  --config-extract extractAST/extract_config.json \
  --src /path/to/one_python_project \
  --out /path/to/output_dir_for_this_project
```
**Output schema (high-level):**

- nodes: [{ id, name, qualified_name, type, file, context, behaviors }, ... ]

- links: [{ source, target, edge_type }, ... ]

If synth_rules.json exists, it will be reused and can be versioned for reproducibility.

## 2) Organize data for training
Create a dataset root with two subfolders (each package in its own subfolder containing `call_graph.json`):
```bash
DATA_ROOT/
├─ benign_call/
│  ├─ pkgA/ call_graph.json
│  ├─ pkgB/ call_graph.json
│  └─ ...
└─ malicious_call/
   ├─ pkgC/ call_graph.json
   ├─ pkgD/ call_graph.json
   └─ ...
```
Tip: when running `cli_extract.py`, set `--out` to `.../benign_call/pkgA` etc.

## 3) Build vocab & PyG tensors
```bash
python generate_graph_data_fromJson.py \
  --normal-root    /path/to/DATA_ROOT/benign_call \
  --malicious-root /path/to/DATA_ROOT/malicious_call \
  --normal-out     /path/to/DATA_ROOT/benign_handled \
  --malicious-out  /path/to/DATA_ROOT/malicious_handled \
  --vocab-dir      /path/to/DATA_ROOT/vocab
```
This step:

- scans both roots,

- builds and saves `vocab/name2idx.json`, `vocab/type2idx.json`, `vocab/edge_type2idx.json`, `vocab/behavior2idx.json`,

- creates `.pt` samples and an index under `*_handled/`.

**Repro tip**: Always train and infer with the same vocab/*.json to avoid OOV drift.

## 4) Train the GNN
```bash
python trainGNN.py \
  --vocab-dir     /path/to/DATA_ROOT/vocab \
  --benign-root   /path/to/DATA_ROOT/benign_call \
  --malicious-root /path/to/DATA_ROOT/malicious_call \
  --benign-out    /path/to/DATA_ROOT/benign_handled \
  --malicious-out /path/to/DATA_ROOT/malicious_handled \
  --device cuda
```
- Model: `GCNWithBehavior` (embeddings for `name/type` + one-hot `behaviors`).

- Reports Acc/F1/malicious recall; saves best weights.

## 5) Explain: GNNExplainer batch
Use the scripts under `GNNExplainer/` (paths in `GNNExplainer/config.json` use `XXXX` placeholders).
**Parallel driver**:
```bash
python GNNExplainer/run_autoexplanation_parallel.py \
  --num-workers 4 \
  --gpus 0,1 \
  --dataset-type normal
```

**Single worker**:
```bash
python GNNExplainer/auto_explanation.py \
  --worker-id 0 --total-workers 1 \
  --gpu-id 0 \
  --dataset-type normal
```

Pipeline per sample:

1. `build_single_graph_ckpt.py` → single-graph checkpoint (`model_state_dict` + `cg_dict`)

2. `explainer_main.py` → masked adjacency `masked_adj_*.npy`

3. `extract2Json.py` → JSON explanation from mask + call_graph

## 6) LLM judgement on subgraphs
The module `call_LLM/` turns masked subgraphs into LLM assessments.

Configure `call_LLM/config_LLM.json` (keep `XXXX` placeholders):

- `provider`: `"azure" | "openai" | "local"` (OpenAI-compatible base URL like Ollama)

- `paths.root_directory`: explainer logs root with `graph_*` folders

- `paths.call_graph_base_path`: root containing `<pkg>/call_graph.json`

- `mask_filename`: default masked adjacency file name

- plus concurrency and timeouts

Run
```bash
# example: OpenAI
export OPENAI_API_KEY=...
python call_LLM/llm_subgraph_runner.py
```

Output: `llm_results.json` with fields `{ name, verdict, reasoning, mitigation }`, resumable and parallel.


## 7) Repro/Determinism tips
- Use the same vocab folder across training/inference.

- When using CUDA, enabling full deterministic ops may incur a small performance drop.

## 8) Baselines
For how to run the baseline scripts, see BASELINES.md.

---
### Dataset and Supplementary Material

To ensure stability and long-term access, the complete dataset and a snapshot of the source code for this project are permanently archived on Zenodo. This supplementary material is provided for the purpose of anonymous peer review.

**You can access the archive here:**

* **[Anonymous Zenodo Repository for Review](https://zenodo.org/records/17182081?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlhODQ1YTA1LTZmNWItNDM4Ni04MGQxLTFjOTdkZTE2NTczOSIsImRhdGEiOnt9LCJyYW5kb20iOiJhNzIyZGQ1M2NhM2U1NzBlNTVlZjE2ZDljYjNjODEzMSJ9.IksQTU47sii4gEmaZ1uuDcH9I3W6A4L6LGZsfxB9wABvcxtiDb4gzZ8SNMlzeZY-S0p05V749xW8R2tydux5pQ)**

---
