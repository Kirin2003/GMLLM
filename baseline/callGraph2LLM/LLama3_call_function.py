import json
import os
from pathlib import Path
import time
from openai import OpenAI, Timeout

LLM_MODEL = os.getenv("DEPLOYMENT_NAME", "llama3:8b")
client = OpenAI(
base_url='http://localhost:8000',
api_key='your_api_key_here',
)


SYSTEM_PROMPT = """You are an expert cybersecurity analyst specializing in static code analysis of Python packages. Your task is to determine if a software package is malicious based on its call graph structure and detected behaviors.
I will provide you with a "Static Call Graph Analysis Report". This report lists all functions, classes, and modules, their properties, and their connections.
Be cautious not to label a package as malicious based on a single suspicious behavior without considering the broader context of the entire report.
Based on the entire report, you must make a final verdict.
**Your response MUST strictly follow this format:**
VERDICT: [BENIGN/MALICIOUS]
REASON: [Provide a concise, one-sentence explanation for your verdict, focusing on the most critical evidence you found.]"""


def generate_textual_representation(graph_data: dict) -> str:
    """Converts a call graph from JSON into a structured textual representation."""
    nodes = graph_data.get('nodes', [])
    links = graph_data.get('links', [])
    if not nodes:
        return "Report Error: No nodes found in call graph."
    node_map = {node['id']: node for node in nodes}
    source_to_links = {}
    for link in links:
        source_id = link.get('source')
        if source_id not in source_to_links:
            source_to_links[source_id] = []
        source_to_links[source_id].append(link)
    report_parts = []
    for node in nodes:
        node_name = node.get('qualified_name', node.get('name', 'Unnamed Node'))
        report_parts.append(f"### Node: {node_name}\n")
        report_parts.append(f"- **Type**: `{node.get('type', 'unknown')}`\n")
        behaviors = node.get('behaviors', [])
        if behaviors:
            behaviors_str = ", ".join(f"`{b}`" for b in behaviors)
            report_parts.append(f"- **Detected Behaviors**: {behaviors_str}\n")
        report_parts.append("- **Connections**:\n")
        outgoing_links = source_to_links.get(node.get('id'), [])
        if not outgoing_links:
            report_parts.append("  - *None*\n")
        else:
            for link in outgoing_links:
                target_id = link.get('target')
                if target_id in node_map:
                    target_node = node_map[target_id]
                    target_name = target_node.get('qualified_name', target_node.get('name', 'Unnamed Node'))
                    target_type = target_node.get('type', 'unknown')
                    edge_type = link.get('edge_type', 'related to').replace('_', ' ').upper()
                    report_parts.append(f"  - `--> [{edge_type}] -->` **{target_name}** (Type: `{target_type}`)\n")
        report_parts.append("\n---\n")
    return "".join(report_parts)

def call_local_llm(report_text: str, package_name: str) -> dict:
    """Queries the local LLM service for analysis and returns a dictionary."""
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": report_text},
            ],
            timeout=100,
        )
        return completion.model_dump()
    except Timeout as e:
        print(f" -> FAILED (API call for package '{package_name}' timed out. Error: {e})")
        return {"package_name": package_name, "error": f"TimeoutError: {e}"}
    except Exception as e:
        print(f" -> FAILED (API call for package '{package_name}' failed. Error: {e})")
        return {"package_name": package_name, "error": str(e)}

def parse_llm_response(response_content: str) -> dict:
    """
    Parses the raw text response from the LLM to extract the verdict and reason.
    If parsing fails, it returns the original content in the 'reason' field.
    """
    verdict = "UNKNOWN"
    reason = "Could not parse the LLM response."

    found_verdict = False
    lines = response_content.strip().split('\n')
    parsed_reason_parts = []

    for line in lines:
        if line.upper().startswith("VERDICT:"):
            verdict = line[len("VERDICT:"):].strip()
            found_verdict = True
        elif line.upper().startswith("REASON:"):
            parsed_reason_parts.append(line[len("REASON:"):].strip())
        else:
            parsed_reason_parts.append(line)

    if found_verdict:
        reason = " ".join(parsed_reason_parts) if parsed_reason_parts else "No reason provided."
    else:
        reason = response_content

    return {"verdict": verdict, "reason": reason}

def main():
    base_dir = Path("path/to/call_graph_dataset")
    output_file = "output/llama3_scan_results.jsonl"

    if not base_dir.exists():
        print(f"Error: Base directory not found at '{base_dir}'")
        return

    processed_packages = set()
    if os.path.exists(output_file):
        print(f"Found existing results file: '{output_file}'. Resuming job.")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "verdict" in data and ("MALICIOUS" in data["verdict"] or "BENIGN" in data["verdict"]):
                        processed_packages.add(data["name"])
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(processed_packages)} successfully completed packages. Will skip them.")
    else:
        print("No existing results file found. Starting a new job.")

    print(f"Starting analysis of packages in: '{base_dir}'")
    print(f"Using local LLM model: '{LLM_MODEL}'")
    
    package_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    total_packages = len(package_dirs)
    print(f"Found {total_packages} total package directories.")

    for i, package_dir in enumerate(package_dirs):
        package_name = package_dir.name
        
        if package_name in processed_packages:
            continue

        call_graph_path = package_dir / "call_graph.json"
        
        print(f"[{i+1}/{total_packages}] Processing: {package_name}...", end='', flush=True)

        if not call_graph_path.exists():
            print(" -> SKIPPED (call_graph.json not found)")
            continue
        
        try:
            with open(call_graph_path, 'r', encoding='utf-8') as f:
                graph_json = json.load(f)

            report_text = generate_textual_representation(graph_json)
            
            llm_result = call_local_llm(report_text, package_name)
            
            output_record = {"name": package_name}

            if "error" in llm_result:
                output_record["verdict"] = "Error: LLM_API_FAILED"
                output_record["reason"] = llm_result["error"]
            else:
                response_content = llm_result.get('choices', [{}])[0].get('message', {}).get('content', '')
                if not response_content:
                    output_record["verdict"] = "Error: LLM_EMPTY_RESPONSE"
                    output_record["reason"] = "LLM returned an empty message."
                    print(" -> FAILED (LLM returned empty response)")
                else:
                    parsed_data = parse_llm_response(response_content)
                    output_record["verdict"] = parsed_data["verdict"]
                    output_record["reason"] = parsed_data["reason"]
                    print(f" -> Done. Verdict: {output_record['verdict']}")

            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_record) + '\n')

        except Exception as e:
            print(f" -> FAILED (Critical script error: {e})")
            error_record = {
                "name": package_name,
                "verdict": "Error: SCRIPT_CRASH",
                "reason": str(e)
            }
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_record) + '\n')

    final_count = len(processed_packages) + sum(1 for line in open(output_file) if line.strip()) - len(processed_packages)
    print(f"\nAnalysis complete. Total results in '{output_file}': {final_count}")

if __name__ == "__main__":
    main()