# -*- coding: utf-8 -*-
"""测试 2023-12 模型在 2024-01 数据上的表现，记录预测与真实标签不一致的包"""
import sys
from pathlib import Path
from datetime import datetime

import torch
import json

# 复用 test_single_package.py 中的工具
sys.path.insert(0, str(Path(__file__).parent))
from test_single_package import load_vocabs, load_model, build_graph_from_json, predict_package

# 路径配置
MODEL_PATH = Path("/Data2/hxq/GMLLM/GMLLM/models/incremental_unk_model_2023-12.pt")
VOCAB_DIR = Path("/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset/vocab")
MALICIOUS_DATA = Path("/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset/malicious/2024-01")
BENIGN_DATA = Path("/Data2/hxq/datasets/incremental_packages_dynamic_capping_subset/benign/2024-01")
OUTPUT_FILE = Path("/Data2/hxq/GMLLM/GMLLM/mismatches_202312_model_202401.json")


def test_dataset(data_dir: Path, true_label: int, name2idx: dict, type2idx: dict,
                 behavior2idx: dict, edge_type2idx: dict, model, device: str):
    """遍历数据集目录，收集预测与真实标签不一致的包"""
    mismatches = []
    total = 0

    if not data_dir.exists():
        print(f"Warning: {data_dir} does not exist, skipping.")
        return mismatches, 0

    for package_dir in sorted(data_dir.iterdir()):
        if not package_dir.is_dir():
            continue

        call_graph_path = package_dir / "call_graph.json"
        if not call_graph_path.exists():
            continue

        total += 1

        try:
            data = build_graph_from_json(
                call_graph_path, name2idx, type2idx, behavior2idx, edge_type2idx, label=true_label
            )
            result = predict_package(model, data, device)

            if result["pred_class"] != true_label:
                mismatches.append({
                    "package_name": package_dir.name,
                    "true_label": "malicious" if true_label == 1 else "benign",
                    "predicted_label": result["prediction"],
                    "pred_class": result["pred_class"],
                    "prob_benign": result["prob_benign"],
                    "prob_malicious": result["prob_malicious"]
                })
        except Exception as e:
            print(f"Error processing {package_dir.name}: {e}")

    return mismatches, total


def main():
    print("=" * 60)
    print("测试 2023-12 模型在 2024-01 数据上的表现")
    print(f"模型: {MODEL_PATH}")
    print(f"时间: {datetime.now().isoformat()}")
    print("=" * 60)

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 1. 加载 vocab
    print("\n[1/4] 加载词汇表...")
    name2idx, type2idx, behavior2idx, edge_type2idx = load_vocabs(VOCAB_DIR)
    print(f"  Vocab 大小: name={len(name2idx)}, type={len(type2idx)}, "
          f"behavior={len(behavior2idx)}, edge={len(edge_type2idx)}")

    # 2. 加载模型
    print("\n[2/4] 加载模型...")
    model = load_model(
        model_path=MODEL_PATH,
        name_vocab_size=len(name2idx),
        type_vocab_size=len(type2idx),
        behavior_dim=len(behavior2idx),
        device=device
    )
    print(f"  模型加载成功: {MODEL_PATH}")

    # 3. 测试恶意包
    print("\n[3/4] 测试恶意包 (malicious/2024-01)...")
    malicious_mismatches, malicious_total = test_dataset(
        MALICIOUS_DATA, true_label=1,
        name2idx=name2idx, type2idx=type2idx,
        behavior2idx=behavior2idx, edge_type2idx=edge_type2idx,
        model=model, device=device
    )
    print(f"  总数: {malicious_total}, 误判(预测为benign): {len(malicious_mismatches)}")

    # 4. 测试良性包
    print("\n[4/4] 测试良性包 (benign/2024-01)...")
    benign_mismatches, benign_total = test_dataset(
        BENIGN_DATA, true_label=0,
        name2idx=name2idx, type2idx=type2idx,
        behavior2idx=behavior2idx, edge_type2idx=edge_type2idx,
        model=model, device=device
    )
    print(f"  总数: {benign_total}, 误判(预测为malicious): {len(benign_mismatches)}")

    # 5. 计算恶意类的 precision/recall/f1
    tp = malicious_total - len(malicious_mismatches)  # malicious predicted as malicious
    fn = len(malicious_mismatches)                    # malicious predicted as benign
    fp = len(benign_mismatches)                       # benign predicted as malicious

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 6. 保存结果
    all_mismatches = malicious_mismatches + benign_mismatches
    total_packages = malicious_total + benign_total
    total_mismatches = len(all_mismatches)

    results = {
        "model_path": str(MODEL_PATH),
        "test_data": {
            "malicious": {"path": str(MALICIOUS_DATA), "total": malicious_total, "mismatches": len(malicious_mismatches)},
            "benign": {"path": str(BENIGN_DATA), "total": benign_total, "mismatches": len(benign_mismatches)}
        },
        "summary": {
            "total_packages": total_packages,
            "total_mismatches": total_mismatches,
            "accuracy": f"{(total_packages - total_mismatches) / total_packages * 100:.2f}%" if total_packages > 0 else "N/A",
            "malicious_class": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4)
            }
        },
        "mismatches": all_mismatches
    }

    print("\n" + "=" * 60)
    print("结果汇总")
    print(f"  总包数: {total_packages}")
    print(f"  误判数: {total_mismatches}")
    print(f"  准确率: {results['summary']['accuracy']}")
    print(f"  恶意类 - Precision: {results['summary']['malicious_class']['precision']:.4f}, "
          f"Recall: {results['summary']['malicious_class']['recall']:.4f}, "
          f"F1: {results['summary']['malicious_class']['f1']:.4f}")
    print(f"  结果保存至: {OUTPUT_FILE}")
    print("=" * 60)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    main()
