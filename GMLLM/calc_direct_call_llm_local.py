import csv
import json
from collections import defaultdict

def calculate_metrics(input_csv: str, output_json: str):
    """Calculate precision/recall/f1 for malicious package detection.

    Args:
        input_csv: Path to input CSV with columns: month, verdict, package_type
        output_json: Path to output JSON file
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
    precision_list = []
    recall_list = []
    f1_list = []

    for month in months:
        rows = monthly[month]
        tp = fp = fn = 0
        for row in rows:
            pred = row['verdict'].strip()
            actual = row['package_type'].strip()
            if pred == 'Malicious' and actual == 'malicious':
                tp += 1
            elif pred == 'Malicious' and actual == 'benign':
                fp += 1
            elif pred in ('Benign', 'Error') and actual == 'malicious':
                fn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

    # Calculate averages
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_f1 = sum(f1_list) / len(f1_list)

    # Build result
    result = {
        "month": months,
        "f1": f1_list,
        "precision": precision_list,
        "recall": recall_list,
        "avg_f1": avg_f1,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall
    }

    # Save
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved to {output_json}")
    print(f"Months: {len(months)}")
    print(f"Avg precision: {avg_precision:.4f}")
    print(f"Avg recall: {avg_recall:.4f}")
    print(f"Avg f1: {avg_f1:.4f}")


if __name__ == "__main__":
    # qwen 3_5_27b results
    calculate_metrics(
        '/Data2/hxq/GMLLM/results/direct_call_llm_local.csv',
        '/Data2/hxq/GMLLM/results/direct_call_llm_local.json'
    )
    # qwen2_5 results
    calculate_metrics(
        '/Data2/hxq/GMLLM/results/direct_call_llm_local_qwen2_5.csv',
        '/Data2/hxq/GMLLM/results/direct_call_llm_local_qwen2_5.json'
    )