import csv
import json
import statistics
import yaml
from collections import defaultdict
from pathlib import Path


def resolve_direct_call_config(model: str) -> Path:
    base_dir = Path(__file__).parent / "configs"
    candidates = [
        base_dir / "direct_call" / f"{model}.yaml",
        base_dir / f"direct_call_llm_{model}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Direct-call config for '{model}' not found. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def calculate_token_stats(input_csv: str, output_json: str, start_month: str = None, end_month: str = None):
    """Calculate token statistics per month for LLM calls.

    Args:
        input_csv: Path to input CSV with columns: month, prompt_tokens, completion_tokens, total_tokens
        output_json: Path to output JSON file
        start_month: Start month for incremental analysis (format: YYYY-MM)
        end_month: End month for incremental analysis (format: YYYY-MM)
    """
    # Read CSV
    data = []
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # Group by month
    monthly = defaultdict(list)
    for row in data:
        month = row['month']
        # Only include rows with valid token data
        try:
            prompt_tokens = int(row['prompt_tokens']) if row['prompt_tokens'] else None
            completion_tokens = int(row['completion_tokens']) if row['completion_tokens'] else None
            total_tokens = int(row['total_tokens']) if row['total_tokens'] else None
        except (ValueError, TypeError):
            continue

        if total_tokens is not None:
            monthly[month].append({
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens
            })

    # Calculate stats per month
    months = sorted(monthly.keys())

    # Filter months if incremental range specified
    if start_month and end_month:
        months = [m for m in months if start_month <= m <= end_month]

    result = {
        "month": months,
        "prompt_tokens": {"min": [], "max": [], "avg": [], "std": []},
        "completion_tokens": {"min": [], "max": [], "avg": [], "std": []},
        "total_tokens": {"min": [], "max": [], "avg": [], "std": []}
    }

    for month in months:
        tokens = monthly[month]
        prompt_list = [t['prompt_tokens'] for t in tokens if t['prompt_tokens'] is not None]
        completion_list = [t['completion_tokens'] for t in tokens if t['completion_tokens'] is not None]
        total_list = [t['total_tokens'] for t in tokens]

        for name, lst, key in [
            ("prompt_tokens", prompt_list, "prompt_tokens"),
            ("completion_tokens", completion_list, "completion_tokens"),
            ("total_tokens", total_list, "total_tokens")
        ]:
            if lst:
                result[key]["min"].append(min(lst))
                result[key]["max"].append(max(lst))
                result[key]["avg"].append(sum(lst) / len(lst))
                result[key]["std"].append(statistics.stdev(lst) if len(lst) > 1 else 0)
            else:
                result[key]["min"].append(0)
                result[key]["max"].append(0)
                result[key]["avg"].append(0)
                result[key]["std"].append(0)

    # Calculate overall stats
    all_prompt = []
    all_completion = []
    all_total = []
    for month in months:
        for t in monthly[month]:
            if t['prompt_tokens'] is not None:
                all_prompt.append(t['prompt_tokens'])
            if t['completion_tokens'] is not None:
                all_completion.append(t['completion_tokens'])
            all_total.append(t['total_tokens'])

    result["overall"] = {
        "prompt_tokens": {
            "min": min(all_prompt) if all_prompt else 0,
            "max": max(all_prompt) if all_prompt else 0,
            "avg": sum(all_prompt) / len(all_prompt) if all_prompt else 0,
            "std": statistics.stdev(all_prompt) if len(all_prompt) > 1 else 0,
            "count": len(all_prompt)
        },
        "completion_tokens": {
            "min": min(all_completion) if all_completion else 0,
            "max": max(all_completion) if all_completion else 0,
            "avg": sum(all_completion) / len(all_completion) if all_completion else 0,
            "std": statistics.stdev(all_completion) if len(all_completion) > 1 else 0,
            "count": len(all_completion)
        },
        "total_tokens": {
            "min": min(all_total) if all_total else 0,
            "max": max(all_total) if all_total else 0,
            "avg": sum(all_total) / len(all_total) if all_total else 0,
            "std": statistics.stdev(all_total) if len(all_total) > 1 else 0,
            "count": len(all_total)
        }
    }

    # Save
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved to {output_json}")
    print(f"Months: {len(months)}")
    print(f"Total valid records: {len(all_total)}")
    print(f"\nOverall token stats:")
    print(f"  Prompt tokens: min={result['overall']['prompt_tokens']['min']}, "
          f"max={result['overall']['prompt_tokens']['max']}, "
          f"avg={result['overall']['prompt_tokens']['avg']:.2f}, "
          f"std={result['overall']['prompt_tokens']['std']:.2f}")
    print(f"  Completion tokens: min={result['overall']['completion_tokens']['min']}, "
          f"max={result['overall']['completion_tokens']['max']}, "
          f"avg={result['overall']['completion_tokens']['avg']:.2f}, "
          f"std={result['overall']['completion_tokens']['std']:.2f}")
    print(f"  Total tokens: min={result['overall']['total_tokens']['min']}, "
          f"max={result['overall']['total_tokens']['max']}, "
          f"avg={result['overall']['total_tokens']['avg']:.2f}, "
          f"std={result['overall']['total_tokens']['std']:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="计算每月token统计信息")
    parser.add_argument('--model', '-m', type=str, required=True,
                        choices=['qwen2_5', 'deepseek', 'llama2'],
                        help='模型名称')
    parser.add_argument('--continual', '-c', action='store_true',
                        help='增量模式：从配置文件读取月份范围，只计算该范围内的统计')
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent / "results"
    input_csv = str(results_dir / f"direct_call_llm_local_{args.model}.csv")
    output_json = str(results_dir / f"direct_call_llm_local_{args.model}_token_stats.json")

    start_month = None
    end_month = None

    if args.continual:
        # Load config file
        config_path = resolve_direct_call_config(args.model)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        start_month = config.get('incremental_start_month')
        end_month = config.get('incremental_end_month')
        print(f"Incremental mode: {start_month} to {end_month}")

    calculate_token_stats(input_csv, output_json, start_month, end_month)
