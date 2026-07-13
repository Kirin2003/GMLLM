import csv
import json
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

def calculate_metrics(input_csv: str, output_json: str, start_month: str = None, end_month: str = None):
    """Calculate precision/recall/f1 for malicious package detection.

    Args:
        input_csv: Path to input CSV with columns: month, verdict, package_type
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
        monthly[row['month']].append(row)

    # Calculate metrics per month
    months = sorted(monthly.keys())

    # Filter months if incremental range specified
    if start_month and end_month:
        months = [m for m in months if start_month <= m <= end_month]

    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []

    for month in months:
        rows = monthly[month]
        tp = fp = fn = tn = 0
        for row in rows:
            pred = row['verdict'].strip()
            actual = row['package_type'].strip()
            if pred == 'Malicious' and actual == 'malicious':
                tp += 1
            elif pred == 'Benign' and actual == 'benign':
                tn += 1
            elif pred != 'Benign' and actual == 'benign':
                fp += 1
            elif pred != 'Malicious' and actual == 'malicious':
                fn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        accuracy_list.append(acc)

    # Calculate averages
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_f1 = sum(f1_list) / len(f1_list)
    avg_accuracy = sum(accuracy_list) / len(accuracy_list)

    # Build result
    result = {
        "month": months,
        "f1": f1_list,
        "precision": precision_list,
        "recall": recall_list,
        "accuracy": accuracy_list,
        "avg_f1": avg_f1,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_accuracy": avg_accuracy
    }

    # Save
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved to {output_json}")
    print(f"Months: {len(months)}")
    print(f"Avg precision: {avg_precision:.4f}")
    print(f"Avg recall: {avg_recall:.4f}")
    print(f"Avg f1: {avg_f1:.4f}")
    print(f"Avg accuracy: {avg_accuracy:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="计算LLM检测结果的precision/recall/f1指标")
    parser.add_argument('--model', '-m', type=str, required=True,
                        choices=['qwen2_5', 'deepseek', 'llama2'],
                        help='模型名称')
    parser.add_argument('--continual', '-c', action='store_true',
                        help='增量模式：从配置文件读取月份范围，只计算该范围内的平均指标')
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent / "results"
    input_csv = str(results_dir / f"direct_call_llm_local_{args.model}.csv")
    output_json = str(results_dir / f"direct_call_llm_local_{args.model}.json")

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

    calculate_metrics(input_csv, output_json, start_month, end_month)
