# -*- coding: utf-8 -*-
import os
import json
import re
import numpy as np
import asyncio
from openai import AzureOpenAI,OpenAI
from tqdm.asyncio import tqdm as asyncio_tqdm # 使用tqdm的异步版本
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# --- 路径配置 ---
ROOT_DIRECTORY = '/media/pt/malicious/filterData/explain_logs_malicious'
CALL_GRAPH_BASE_PATH = '/media/pt/malicious/filterData/malicious_call'
MASK_FILENAME = 'masked_adj_syn1_base_h64_o20_explainnode_idx_0graph_idx_0.npy'
OUTPUT_JSON_FILE = "./LLama3_our_malicious.json"



AZURE_OPENAI_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "llama3:8b")

# --- 模型与任务配置 ---
TOP_K_COUNT = 10
MAX_NODE_NAME_LENGTH = 250
START_INDEX = 0  # 处理目录的起始索引
END_INDEX = None # 处理目录的结束索引 (None表示到最后)

# --- 性能配置 ---
# 设置并发请求的数量。请根据您API账户的速率限制（Rate Limit）进行调整。
# 10-20是一个比较安全和高效的起点。
MAX_WORKERS = 20 
# 单个API请求的超时时间（秒）
API_REQUEST_TIMEOUT = 100  

client = OpenAI(
    base_url ='http://localhost:11434/v1',
    api_key = 'ollama',
)

LOG_FILE = "LLama3malicious.log"

def log_to_file(message: str):
    """将调试信息追加到日志文件中"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")

def extract_top_k_subgraph(mask_path, call_graph_path, top_k, max_name_length):
    """
    从给定的调用图和注意力掩码中提取Top-K子图结构。
    """
    try:
        masked_adj = np.load(mask_path)
        N = masked_adj.shape[0]
        with open(call_graph_path, 'r', encoding='utf-8') as f:
            cg = json.load(f)
    except FileNotFoundError as e:
        return f"错误：读取文件失败。{e}"
    except Exception as e:
        return f"处理文件时发生错误: {e}"

    try:
        id2name_str = {node['id']: node.get('qualified_name') or node.get('name') or node['id'] for node in cg['nodes']}
    except KeyError:
        return f"错误: 文件 {call_graph_path} 中的某个节点缺少 'id' 字段，已跳过。"

    scored_edges = []
    for link in cg['links']:
        src_id, dst_id = link['source'], link['target']
        try:
            src_idx = int(src_id.split('_')[-1])
            dst_idx = int(dst_id.split('_')[-1])
        except (ValueError, IndexError):
            continue

        if src_idx < N and dst_idx < N:
            score = masked_adj[src_idx, dst_idx]
            if score > 0:
                scored_edges.append({
                    "score": score,
                    "source": id2name_str.get(src_id, src_id),
                    "target": id2name_str.get(dst_id, dst_id)
                })

    sorted_edges = sorted(scored_edges, key=lambda x: x['score'], reverse=True)
    top_k_edges = sorted_edges[:top_k]
    
    if not top_k_edges:
        return None

    renaming_map, rename_counter = {}, 1
    processed_edges = []
    for edge in top_k_edges:
        p_source, p_target = edge['source'], edge['target']
        for name in [('source', p_source), ('target', p_target)]:
            if len(name[1]) > max_name_length:
                if name[1] not in renaming_map:
                    new_name = f"Filtered_Long_Node_{rename_counter}"
                    renaming_map[name[1]] = new_name
                    rename_counter += 1
                if name[0] == 'source':
                    p_source = renaming_map[name[1]]
                else:
                    p_target = renaming_map[name[1]]
        processed_edges.append((p_source, p_target))

    final_nodes_set = {node for edge in processed_edges for node in edge}
    final_sorted_nodes = sorted(list(final_nodes_set))

    output_lines = [f"- {n}" for n in final_sorted_nodes]
    output_str = "Subgraph Nodes:\n" + "\n".join(output_lines)
    output_lines = [f"- {u} → {v}" for u, v in processed_edges]
    output_str += "\n\nSubgraph Edges:\n" + "\n".join(output_lines)
        
    if renaming_map:
        output_str += "\n\n[Analysis Note]:\n- Some nodes with abnormally long names were detected and renamed for clarity (e.g., 'Filtered_Long_Node_1').\n- These abnormally named nodes and their connections are highly suspicious."
    
    return output_str


def call_llm(sub_graph: str, package_name: str):
    """异步调用LLM API，并包含错误处理"""
    try:
        completion = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": """You are a PyPI package security auditor. 
                 You have been provided with the 'high-attention subgraph' structure of a PyPI package script. Based solely on this subgraph structure, please answer the following:
                 Is this structure indicative of potential malicious activity? (Respond only with 'Malicious' or 'Benign'). 
                 Provide your reasoning.
                 If mitigation is needed, identify the highest priority nodes or calls for input validation or permission checks.
                 Response Format:
                 Verdict:
                 Reasoning:
                 Mitigation:"""},
                {"role": "user", "content": sub_graph},
            ],
            timeout=API_REQUEST_TIMEOUT,
        )
        return completion.model_dump()
    except Exception as e:
        log_to_file(f"API call for package '{package_name}' failed. Error: {e}")
        return {"package_name": package_name,"error": str(e)} # 返回错误信息


def parse_llm_response(entry_name: str, api_response: dict):
    """
    解析LLM API的回复，提取关键内容，并确保输出字典的key顺序固定。
    """
    if not api_response or "error" in api_response:
        return {
            "name": entry_name,
            "verdict": "API_ERROR",
            "reasoning": api_response.get("error", "Unknown API error"),
            "mitigation": "N/A"
        }

    try:
        content = api_response['choices'][0]['message']['content']
    except (KeyError, IndexError):
        log_to_file(f"Error parsing content for {entry_name}. Response: {api_response}")
        return {
            "name": entry_name,
            "verdict": "PARSE_ERROR",
            "reasoning": "Failed to extract 'content' from API response.",
            "mitigation": "N/A",
            "raw_content": api_response
        }

    verdict = (re.search(r'Verdict:\s*(.*)', content, re.IGNORECASE) or [None, "Not Found"])[1].strip()
    reasoning = (re.search(r'Reasoning:\s*(.*?)(?=Mitigation:|$)', content, re.DOTALL | re.IGNORECASE) or [None, "Not Found"])[1].strip()
    mitigation = (re.search(r'Mitigation:\s*(.*)', content, re.DOTALL | re.IGNORECASE) or [None, "Not Found"])[1].strip()

    final_entry = {
        "name": entry_name,
        "verdict": verdict,
        "reasoning": reasoning,
        "mitigation": mitigation
    }
    
    return final_entry


def main():
    """主执行函数，包含并发调度和断点续传逻辑"""
    processed_packages = set()
    existing_results = []

    if os.path.exists(OUTPUT_JSON_FILE):
        print(f"检测到输出文件 '{OUTPUT_JSON_FILE}'，加载已有结果以避免重复...")
        try:
            with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                if isinstance(existing_results, list):
                    processed_packages.update(item['name'] for item in existing_results if 'name' in item)
                    print(f"成功加载 {len(processed_packages)} 条已有记录。")
                else:
                    existing_results = [] # 如果文件格式不对，则重置
        except (json.JSONDecodeError, IOError) as e:
            log_to_file(f"读取或解析 '{OUTPUT_JSON_FILE}' 失败: {e}。将创建新文件。")
            existing_results = []


    if not os.path.isdir(ROOT_DIRECTORY):
        print(f"错误: 根目录 '{ROOT_DIRECTORY}' 不存在！")
        return

    all_dirs = sorted(os.listdir(ROOT_DIRECTORY))
    target_dirs = all_dirs[START_INDEX:END_INDEX]
    
    tasks_to_run = []
    print("\n扫描目录，准备新任务...")
    for item_name in tqdm(target_dirs, desc="Preparing Tasks"):
        if not item_name.startswith("graph_"): continue
        package_name = item_name[6:]

        if package_name in processed_packages:
            continue

        mask_path = os.path.join(ROOT_DIRECTORY, item_name, MASK_FILENAME)
        cg_path = os.path.join(CALL_GRAPH_BASE_PATH, package_name, 'call_graph.json')

        if not (os.path.exists(mask_path) and os.path.exists(cg_path)):
            log_to_file(f"Skipping {package_name}: Missing required files.")
            continue
        
        subgraph_data = extract_top_k_subgraph(mask_path, cg_path, TOP_K_COUNT, MAX_NODE_NAME_LENGTH)
        
        if subgraph_data and "错误:" not in subgraph_data:
            # 将异步函数调用和其参数打包成一个任务
            tasks_to_run.append({'subgraph': subgraph_data, 'package_name': package_name})
        else:
            log_to_file(f"Skipping {package_name}: Could not extract valid subgraph data.")

    if not tasks_to_run:
        print("\n所有包均已处理完毕，没有新任务。程序结束。")
        return

    print(f"\n发现 {len(tasks_to_run)} 个新任务，开始执行并发API调用...")
    new_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(call_llm, task['subgraph'], task['package_name']): task for task in tasks_to_run}
        
        for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run), desc="Calling LLM API"):
            task_info = future_to_task[future]
            package_name = task_info['package_name']
            
            try:
                api_response = future.result()
                parsed_data = parse_llm_response(package_name, api_response)
                new_results.append(parsed_data)
                log_to_file(parsed_data)
            except Exception as exc:
                log_to_file(f"Package {package_name} generated an exception: {exc}")
                new_results.append(parse_llm_response(package_name, {"error": str(exc)}))

            
    print("\n所有API调用完成，正在处理和保存结果...")
    
    final_results = existing_results + new_results
    try:
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
        print(f"\n成功！结果已更新至 {OUTPUT_JSON_FILE}")
    except IOError as e:
        print(f"致命错误：无法将最终结果写入文件！错误: {e}")

    benign_count = sum(1 for res in final_results if str(res.get('verdict')).lower() == 'benign')
    malicious_count = sum(1 for res in final_results if str(res.get('verdict')).lower() == 'malicious')
    error_count = len(final_results) - benign_count - malicious_count
    print("\n" + "="*50 + "\n最终统计:\n" + f"  - 总共处理的包 (在文件中): {len(final_results)}\n" + f"  - 本次运行新增: {len(new_results)}\n" + f"  - 判定为良性 (Benign): {benign_count}\n" + f"  - 判定为恶意 (Malicious): {malicious_count}\n" + f"  - 其他 (错误/无法判断): {error_count}\n" + "="*50)


if __name__ == "__main__":
    main()