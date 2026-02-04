"""
诊断脚本：分析为什么指定月份的模型recall为0
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np

from distinguish_GNN_2 import GCNWithBehavior, load_dict, set_seed
from generate_graph_data_fromJson import CallGraphDatasetFull_Lazy


class Logger:
    """同时输出到控制台和文件"""
    def __init__(self, txt_path):
        self.txt_path = Path(txt_path)
        self.lines = []

    def print(self, *args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(msg, **kwargs)
        self.lines.append(msg)

    def save(self):
        self.txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.lines))


@torch.no_grad()
def detailed_validate(model, loader, device, month_name, logger, results_dict):
    """详细验证并输出诊断信息"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for data in loader:
        data = data.to(device)
        out = model(data)
        probs = F.softmax(out, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(out.argmax(dim=1).cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1], zero_division=0
    )

    num_benign = (all_labels == 0).sum()
    num_malicious = (all_labels == 1).sum()

    # 输出到日志
    logger.print(f"\n{'='*60}")
    logger.print(f"月份: {month_name}")
    logger.print(f"{'='*60}")
    logger.print(f"【样本分布】良性: {num_benign}, 恶意: {num_malicious}")
    logger.print(f"【混淆矩阵】TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    logger.print(f"【分类指标】恶意 Precision: {p[1]:.4f}, Recall: {r[1]:.4f}, F1: {f1[1]:.4f}")

    # 预测概率分布
    benign_probs = all_probs[all_labels == 0, 0]
    malicious_probs = all_probs[all_labels == 1, 0]
    logger.print(f"【概率分布】恶意样本P(良性)均值: {malicious_probs.mean():.4f}, 良性样本P(良性)均值: {benign_probs.mean():.4f}")

    # 诊断结论
    if num_malicious == 0:
        logger.print(f"⚠️ 该月没有恶意样本！")
    elif tp == 0:
        logger.print(f"❌ 所有恶意样本都被判为良性 (FN={fn})")
        if malicious_probs.mean() > 0.8:
            logger.print(f"   → 模型极度倾向于将恶意样本判为良性")
    else:
        logger.print(f"✓ 部分恶意样本被正确检测 (TP={tp})")

    # 保存到JSON
    results_dict[month_name] = {
        'sample_counts': {
            'benign': int(num_benign),
            'malicious': int(num_malicious)
        },
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'metrics': {
            'benign_precision': float(p[0]), 'benign_recall': float(r[0]),
            'malicious_precision': float(p[1]), 'malicious_recall': float(r[1]),
            'malicious_f1': float(f1[1])
        },
        'prob_distribution': {
            'benign_mean_p_benign': float(benign_probs.mean()),
            'malicious_mean_p_benign': float(malicious_probs.mean())
        }
    }


def main():
    # 配置
    model_path = "/Data2/hxq/datasets/incremental_packages_subset/best_model.pt"
    vocab_dir = "/Data2/hxq/datasets/incremental_packages_subset/vocab"
    benign_root = "/Data2/hxq/datasets/incremental_packages_subset/benign"
    malicious_root = "/Data2/hxq/datasets/incremental_packages_subset/malicious"
    benign_out = "/Data2/hxq/datasets/incremental_packages_subset/benign_call_processed"
    malicious_out = "/Data2/hxq/datasets/incremental_packages_subset/malicious_call_processed"

    target_months = ["2023-07", "2023-09", "2023-11", "2024-01", "2024-03"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 输出路径
    output_dir = Path("/Data2/hxq/GMLLM/GMLLM/extractAST")
    txt_path = output_dir / "diagnosis_results.txt"
    json_path = output_dir / "diagnosis_results.json"

    logger = Logger(txt_path)
    all_results = {}

    set_seed(42)
    logger.print(f"设备: {device}, 目标月份: {target_months}")

    # 加载词汇表和模型
    name2idx = load_dict(str(Path(vocab_dir) / "name2idx.json"))
    type2idx = load_dict(str(Path(vocab_dir) / "type2idx.json"))
    behavior2idx = load_dict(str(Path(vocab_dir) / "behavior2idx.json"))
    edge_type2idx = load_dict(str(Path(vocab_dir) / "edge_type2idx.json"))

    model = GCNWithBehavior(
        name_vocab_size=len(name2idx),
        type_vocab_size=len(type2idx),
        behavior_dim=len(behavior2idx)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.print(f"模型加载完成: {model_path}")

    # 分析目标月份
    for month in target_months:
        logger.print(f"\n{'#'*60}")
        logger.print(f"## 分析月份: {month}")
        logger.print(f"{'#'*60}")

        normal_dataset = CallGraphDatasetFull_Lazy(
            root_dir=benign_root, output_dir=benign_out,
            name2idx=name2idx, type2idx=type2idx,
            behavior2idx=behavior2idx, edge_type2idx=edge_type2idx,
            fixed_label=0, start_month=month, end_month=month
        )
        malicious_dataset = CallGraphDatasetFull_Lazy(
            root_dir=malicious_root, output_dir=malicious_out,
            name2idx=name2idx, type2idx=type2idx,
            behavior2idx=behavior2idx, edge_type2idx=edge_type2idx,
            fixed_label=1, start_month=month, end_month=month
        )

        if len(normal_dataset) == 0 and len(malicious_dataset) == 0:
            logger.print(f"数据为空，跳过")
            continue

        # 100% 用于测试（不做划分）
        test_dataset = ConcatDataset([normal_dataset, malicious_dataset])

        if len(test_dataset) == 0:
            logger.print(f"测试集为空，跳过")
            continue

        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
        detailed_validate(model, test_loader, device, month, logger, all_results)

    # 保存结果
    logger.print(f"\n{'='*60}")
    logger.print(f"诊断完成！")
    logger.print(f"结果已保存到: {txt_path} 和 {json_path}")
    logger.save()

    # 保存JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
