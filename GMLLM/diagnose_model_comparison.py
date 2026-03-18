# -*- coding: utf-8 -*-
"""
诊断脚本：比较baseline模型和UNK模型在给定月份的输出差异

功能：
1. 找出两个模型输出结果不一样的样本
2. 显示正确的结果是什么
3. 统计UNK模型相比于baseline模型，恶意样本和良性样本的正确数提升了多少
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
import argparse
from datetime import datetime

from distinguish_GNN_2 import GCNWithBehavior, load_dict, set_seed
from utils.data_loader import load_vocabs, load_month_dataset
from utils.data_utils import split_train_test
from utils.logger_utils import Logger as Log

# 全局日志对象
log = None


def get_prev_month(month: str) -> str:
    """获取给定月份的前一个月"""
    year, mon = month.split('-')
    year, mon = int(year), int(mon)
    if mon == 1:
        return f"{year-1}-12"
    else:
        return f"{year:04d}-{mon-1:02d}"


@torch.no_grad()
def evaluate_model_detailed(model, loader, device):
    """评估模型，返回预测结果、标签和概率"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_indices = []

    idx = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        probs = F.softmax(out, dim=1)
        preds = out.argmax(dim=1)

        batch_size = data.y.shape[0]
        all_indices.extend(range(idx, idx + batch_size))
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        idx += batch_size

    return {
        'preds': np.array(all_preds),
        'labels': np.array(all_labels),
        'probs': np.array(all_probs),
        'indices': np.array(all_indices)
    }


def analyze_disagreement(results_base, results_unk):
    """分析两个模型预测不一致的样本"""
    preds_base = results_base['preds']
    preds_unk = results_unk['preds']
    labels = results_base['labels']
    probs_base = results_base['probs']
    probs_unk = results_unk['probs']

    # 找出预测不一致的样本
    disagreement_mask = preds_base != preds_unk
    disagreement_indices = np.where(disagreement_mask)[0]

    log.log(f"\n{'='*70}")
    log.log(f"模型预测不一致分析")
    log.log(f"{'='*70}")
    log.log(f"总样本数: {len(preds_base)}")
    log.log(f"预测不一致的样本数: {len(disagreement_indices)}")

    if len(disagreement_indices) == 0:
        log.log("没有预测不一致的样本")
        return [], []

    # 分类统计
    benign_disagreements = []
    malicious_disagreements = []

    for idx in disagreement_indices:
        label = labels[idx]
        pred_base = preds_base[idx]
        pred_unk = preds_unk[idx]
        prob_base = probs_base[idx]
        prob_unk = probs_unk[idx]

        # 判断正确的结果和错误的结果
        if label == 0:  # 良性样本
            # 正确结果是0（良性）
            if pred_base == 0 and pred_unk == 1:
                # baseline正确，UNK错误
                detail = {
                    'idx': idx,
                    'label': 'benign',
                    'correct': 0,
                    'baseline_pred': 0,
                    'unk_pred': 1,
                    'baseline_correct': True,
                    'unk_correct': False,
                    'baseline_p_benign': float(prob_base[0]),
                    'unk_p_benign': float(prob_unk[0])
                }
                benign_disagreements.append(detail)
            elif pred_base == 1 and pred_unk == 0:
                # baseline错误，UNK正确
                detail = {
                    'idx': idx,
                    'label': 'benign',
                    'correct': 0,
                    'baseline_pred': 1,
                    'unk_pred': 0,
                    'baseline_correct': False,
                    'unk_correct': True,
                    'baseline_p_benign': float(prob_base[0]),
                    'unk_p_benign': float(prob_unk[0])
                }
                benign_disagreements.append(detail)
        else:  # 恶意样本
            # 正确结果是1（恶意）
            if pred_base == 1 and pred_unk == 0:
                # baseline正确，UNK错误
                detail = {
                    'idx': idx,
                    'label': 'malicious',
                    'correct': 1,
                    'baseline_pred': 1,
                    'unk_pred': 0,
                    'baseline_correct': True,
                    'unk_correct': False,
                    'baseline_p_malicious': float(prob_base[1]),
                    'unk_p_malicious': float(prob_unk[1])
                }
                malicious_disagreements.append(detail)
            elif pred_base == 0 and pred_unk == 1:
                # baseline错误，UNK正确
                detail = {
                    'idx': idx,
                    'label': 'malicious',
                    'correct': 1,
                    'baseline_pred': 0,
                    'unk_pred': 1,
                    'baseline_correct': False,
                    'unk_correct': True,
                    'baseline_p_malicious': float(prob_base[1]),
                    'unk_p_malicious': float(prob_unk[1])
                }
                malicious_disagreements.append(detail)

    # 打印详细统计
    log.log(f"\n--- 良性样本不一致分析 (共{len(benign_disagreements)}个) ---")
    base_correct_benign = sum(1 for d in benign_disagreements if d['baseline_correct'])
    unk_correct_benign = sum(1 for d in benign_disagreements if d['unk_correct'])
    log.log(f"  Baseline正确, UNK错误: {base_correct_benign}")
    log.log(f"  Baseline错误, UNK正确: {unk_correct_benign}")

    log.log(f"\n--- 恶意样本不一致分析 (共{malicious_disagreements}个) ---")
    log.log(f"  共{len(malicious_disagreements)}个不一致")

    # 打印不一致样本详情（限制数量）
    if benign_disagreements:
        log.log(f"\n良性样本不一致详情 (前10个):")
        for detail in benign_disagreements[:10]:
            log.log(f"  样本_idx={detail['idx']}: 正确={detail['correct']}, "
                        f"Baseline预测={detail['baseline_pred']}({'✓' if detail['baseline_correct'] else '✗'}), "
                        f"UNK预测={detail['unk_pred']}({'✓' if detail['unk_correct'] else '✗'})")

    if malicious_disagreements:
        log.log(f"\n恶意样本不一致详情 (前10个):")
        for detail in malicious_disagreements[:10]:
            log.log(f"  样本_idx={detail['idx']}: 正确={detail['correct']}, "
                        f"Baseline预测={detail['baseline_pred']}({'✓' if detail['baseline_correct'] else '✗'}), "
                        f"UNK预测={detail['unk_pred']}({'✓' if detail['unk_correct'] else '✗'})")

    return benign_disagreements, malicious_disagreements


def calculate_improvement(results_base, results_unk):
    """计算UNK模型相比baseline模型的正确数提升"""
    preds_base = results_base['preds']
    preds_unk = results_unk['preds']
    labels = results_base['labels']

    # 分别统计良性样本和恶意样本
    benign_mask = labels == 0
    malicious_mask = labels == 1

    # Baseline模型
    base_benign_correct = (preds_base[benign_mask] == labels[benign_mask]).sum()
    base_malicious_correct = (preds_base[malicious_mask] == labels[malicious_mask]).sum()

    # UNK模型
    unk_benign_correct = (preds_unk[benign_mask] == labels[benign_mask]).sum()
    unk_malicious_correct = (preds_unk[malicious_mask] == labels[malicious_mask]).sum()

    # 计算提升
    benign_improvement = unk_benign_correct - base_benign_correct
    malicious_improvement = unk_malicious_correct - base_malicious_correct

    # 总样本数
    total_benign = benign_mask.sum()
    total_malicious = malicious_mask.sum()

    log.log(f"\n{'='*70}")
    log.log(f"模型正确数对比")
    log.log(f"{'='*70}")
    log.log(f"\n【良性样本】总数: {total_benign}")
    log.log(f"  Baseline正确: {base_benign_correct} ({base_benign_correct/total_benign*100:.2f}%)")
    log.log(f"  UNK正确:      {unk_benign_correct} ({unk_benign_correct/total_benign*100:.2f}%)")
    log.log(f"  提升:         {benign_improvement:+d}")

    log.log(f"\n【恶意样本】总数: {total_malicious}")
    log.log(f"  Baseline正确: {base_malicious_correct} ({base_malicious_correct/total_malicious*100:.2f}%)")
    log.log(f"  UNK正确:      {unk_malicious_correct} ({unk_malicious_correct/total_malicious*100:.2f}%)")
    log.log(f"  提升:         {malicious_improvement:+d}")

    # 计算整体准确率
    total_samples = len(labels)
    base_total_correct = (preds_base == labels).sum()
    unk_total_correct = (preds_unk == labels).sum()
    log.log(f"\n【总体】总数: {total_samples}")
    log.log(f"  Baseline正确: {base_total_correct} ({base_total_correct/total_samples*100:.2f}%)")
    log.log(f"  UNK正确:      {unk_total_correct} ({unk_total_correct/total_samples*100:.2f}%)")
    log.log(f"  提升:         {unk_total_correct - base_total_correct:+d}")

    return {
        'benign': {
            'total': int(total_benign),
            'base_correct': int(base_benign_correct),
            'unk_correct': int(unk_benign_correct),
            'improvement': int(benign_improvement)
        },
        'malicious': {
            'total': int(total_malicious),
            'base_correct': int(base_malicious_correct),
            'unk_correct': int(unk_malicious_correct),
            'improvement': int(malicious_improvement)
        },
        'total': {
            'total': int(total_samples),
            'base_correct': int(base_total_correct),
            'unk_correct': int(unk_total_correct),
            'improvement': int(unk_total_correct - base_total_correct)
        }
    }


def run_diagnosis(target_months, models_dir, vocab_dir, data_paths, output_dir, device):
    """运行诊断分析"""
    set_seed(42)

    # 加载词汇表
    name2idx, type2idx, behavior2idx, edge_type2idx = load_vocabs(vocab_dir)
    vocab = {
        'name2idx': name2idx,
        'type2idx': type2idx,
        'behavior2idx': behavior2idx,
        'edge_type2idx': edge_type2idx
    }

    # 模型路径
    base_model_path = Path(models_dir) / "base_model_0207.pt"

    global log
    log = Log(f"diagnosis_{'_'.join(target_months)}.log")

    log.log(f"="*70)
    log.log(f"模型诊断报告")
    log.log(f"="*70)
    log.log(f"目标月份: {target_months}")
    log.log(f"设备: {device}")
    log.log(f"Baseline模型: {base_model_path}")

    all_month_results = {}

    for month in target_months:
        prev_month = get_prev_month(month)
        unk_model_path = Path(models_dir) / f"incremental_unk_model_{prev_month}.pt"

        log.log(f"\n{'#'*70}")
        log.log(f"## 月份: {month} (UNK模型: incremental_unk_model_{prev_month}.pt)")
        log.log(f"{'#'*70}")

        if not unk_model_path.exists():
            log.log(f"警告: UNK模型不存在 - {unk_model_path}")
            continue

        # 加载数据
        normal_ds, malicious_ds = load_month_dataset(month, vocab, data_paths)

        if len(normal_ds) == 0 and len(malicious_ds) == 0:
            log.log(f"该月没有数据，跳过")
            continue

        # 8:2 划分，测试集为 20%
        _, normal_test, _, malicious_test = split_train_test(normal_ds, malicious_ds, train_ratio=0.8)
        test_dataset = ConcatDataset([normal_test, malicious_test])
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

        log.log(f"数据加载完成: 良性={len(normal_test)}, 恶意={len(malicious_test)}, 总计={len(test_dataset)}")

        # 加载模型
        # Baseline模型
        model_base = GCNWithBehavior(
            name_vocab_size=len(name2idx),
            type_vocab_size=len(type2idx),
            behavior_dim=len(behavior2idx)
        ).to(device)
        model_base.load_state_dict(torch.load("./models/base_model_0207.pt", map_location=device))
        model_base.eval()

        # UNK模型
        model_unk = GCNWithBehavior(
            name_vocab_size=len(name2idx),
            type_vocab_size=len(type2idx),
            behavior_dim=len(behavior2idx)
        ).to(device)
        model_unk.load_state_dict(torch.load(unk_model_path, map_location=device))
        model_unk.eval()

        log.log(f"模型加载完成")

        # 评估两个模型
        results_base = evaluate_model_detailed(model_base, test_loader, device)
        results_unk = evaluate_model_detailed(model_unk, test_loader, device)

        # 分析不一致
        benign_disc, malicious_disc = analyze_disagreement(results_base, results_unk)

        # 计算提升
        improvement = calculate_improvement(results_base, results_unk)

        # 保存月度结果
        all_month_results[month] = {
            'unk_model': f"incremental_unk_model_{prev_month}.pt",
            'sample_counts': {
                'benign': len(normal_test),
                'malicious': len(malicious_test)
            },
            'improvement': improvement,
            'disagreement_counts': {
                'benign': len(benign_disc),
                'malicious': len(malicious_disc)
            }
        }

    # 汇总统计
    log.log(f"\n{'='*70}")
    log.log(f"汇总统计")
    log.log(f"{'='*70}")

    total_benign_improvement = 0
    total_malicious_improvement = 0

    for month, result in all_month_results.items():
        log.log(f"\n{month}:")
        log.log(f"  良性样本提升: {result['improvement']['benign']['improvement']:+d} "
                    f"({result['improvement']['benign']['base_correct']} -> {result['improvement']['benign']['unk_correct']})")
        log.log(f"  恶意样本提升: {result['improvement']['malicious']['improvement']:+d} "
                    f"({result['improvement']['malicious']['base_correct']} -> {result['improvement']['malicious']['unk_correct']})")
        total_benign_improvement += result['improvement']['benign']['improvement']
        total_malicious_improvement += result['improvement']['malicious']['improvement']

    log.log(f"\n总提升:")
    log.log(f"  良性样本: {total_benign_improvement:+d}")
    log.log(f"  恶意样本: {total_malicious_improvement:+d}")

    # 保存JSON结果
    json_path = output_dir / f"diagnosis_{'_'.join(target_months)}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_month_results, f, indent=2, ensure_ascii=False)

    log.log(f"\n结果已保存到: {log.log_file} 和 {json_path}")

    return all_month_results


def main():
    parser = argparse.ArgumentParser(description='诊断baseline和UNK模型的输出差异')
    parser.add_argument('--months', nargs='+', required=True,
                        help='要诊断的月份列表，例如: 2023-03 2023-04')
    parser.add_argument('--models_dir', type=str,
                        default='./models',
                        help='模型目录')
    parser.add_argument('--output_dir', type=str,
                        default='./results',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备: cuda 或 cpu')

    args = parser.parse_args()

    # 数据路径配置
    vocab_dir = "/Data2/hxq/datasets/incremental_packages_subset/vocab"
    data_paths = {
        'benign_root': "/Data2/hxq/datasets/incremental_packages_subset/benign",
        'malicious_root': "/Data2/hxq/datasets/incremental_packages_subset/malicious",
        'benign_out': "/Data2/hxq/datasets/incremental_packages_subset/benign_call_processed",
        'malicious_out': "/Data2/hxq/datasets/incremental_packages_subset/malicious_call_processed",
    }

    # 检查设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")

    print(f"诊断月份: {args.months}")
    print(f"模型目录: {args.models_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {device}")

    results = run_diagnosis(
        target_months=args.months,
        models_dir=args.models_dir,
        vocab_dir=vocab_dir,
        data_paths=data_paths,
        output_dir=Path(args.output_dir),
        device=device
    )

    print("\n诊断完成!")
    for month, result in results.items():
        print(f"\n{month}:")
        print(f"  恶意样本提升: {result['improvement']['malicious']['improvement']:+d}")
        print(f"  良性样本提升: {result['improvement']['benign']['improvement']:+d}")


if __name__ == "__main__":
    main()
