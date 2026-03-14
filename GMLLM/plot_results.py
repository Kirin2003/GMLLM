"""Plotting utilities for GNN experiment results."""
import matplotlib.pyplot as plt
from pathlib import Path


def plot_complete_metrics_curve(val_period_results: dict, future_test_results: dict, results_dir: Path):
    """
    绘制完整的F1/Precision/Recall曲线对比图（单张图）。

    Args:
        val_period_results: 验证期结果 {'month': [...], 'f1': [...], 'precision': [...], 'recall': [...]}
        future_test_results: 未来测试结果 {'month': [...], 'f1': [...], 'precision': [...], 'recall': [...]}
        results_dir: 保存图片的目录
    """
    # 合并所有月份和指标
    all_months = val_period_results['month'] + future_test_results['month']
    all_f1 = val_period_results['f1'] + future_test_results['f1']
    all_prec = val_period_results['precision'] + future_test_results['precision']
    all_recall = val_period_results['recall'] + future_test_results['recall']

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(all_months, all_f1, 'b-o', label='F1', markersize=4, linewidth=1.5)
    ax.plot(all_months, all_prec, 'g-s', label='Precision', markersize=4, linewidth=1.5)
    ax.plot(all_months, all_recall, 'r--^', label='Recall', markersize=4, linewidth=1.5)

    # 标注训练/验证期和未来测试期的分界线
    val_len = len(val_period_results['month'])
    if val_len > 0:
        ax.axvline(x=val_len - 0.5, color='purple', linestyle='--', alpha=0.7, linewidth=2,
                   label='Train/Val End')

    ax.set_title('Complete F1/Precision/Recall Curves', fontsize=14)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'complete_metrics_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - complete_metrics_curve.png")


def plot_monthly_metrics(val_period_results: dict, future_test_results: dict, results_dir: Path):
    """
    绘制按月测试指标折线图（4个子图）。

    Args:
        val_period_results: 验证期结果 {'month': [...], 'f1': [...], 'precision': [...], 'recall': [...]}
        future_test_results: 未来测试结果 {'month': [...], 'f1': [...], 'precision': [...], 'recall': [...]}
        results_dir: 保存图片的目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 验证期结果
    val_months = val_period_results['month']
    axes[0, 0].plot(val_months, val_period_results['f1'], 'b-o', label='F1')
    axes[0, 0].plot(val_months, val_period_results['precision'], 'g--s', label='Precision')
    axes[0, 0].plot(val_months, val_period_results['recall'], 'r--^', label='Recall')
    axes[0, 0].set_title('Validation Period (2022-01 ~ 2023-02)')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1.05)

    # 未来测试结果
    future_months = future_test_results['month']
    axes[0, 1].plot(future_months, future_test_results['f1'], 'b-o', label='F1')
    axes[0, 1].plot(future_months, future_test_results['precision'], 'g--s', label='Precision')
    axes[0, 1].plot(future_months, future_test_results['recall'], 'r--^', label='Recall')
    axes[0, 1].set_title('Future Test (2023-03 ~ 2024-12)')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1.05)

    # 完整 F1/Precision/Recall 曲线对比
    all_months = val_months + future_months
    all_f1 = val_period_results['f1'] + future_test_results['f1']
    all_prec = val_period_results['precision'] + future_test_results['precision']
    all_recall = val_period_results['recall'] + future_test_results['recall']

    ax1 = axes[1, 0]
    ax1.plot(all_months, all_f1, 'b-o', label='F1', markersize=4)
    ax1.plot(all_months, all_prec, 'g-s', label='Precision', markersize=4)
    ax1.plot(all_months, all_recall, 'r--^', label='Recall', markersize=4)
    ax1.axvline(x='2023-02', color='purple', linestyle='--', alpha=0.7, label='Train/Val End')
    ax1.set_title('Overall Performance')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 1.05)

    # 未来测试 F1 单独展示
    axes[1, 1].bar(range(len(future_months)), future_test_results['f1'], color='steelblue', alpha=0.7)
    axes[1, 1].set_xticks(range(len(future_months)))
    axes[1, 1].set_xticklabels(future_months, rotation=45, ha='right')
    axes[1, 1].set_title('Monthly F1 (Future Test)')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].axhline(y=sum(future_test_results['f1']) / len(future_test_results['f1']),
                       color='red', linestyle='--', label='Average')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(results_dir / 'monthly_test_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - monthly_test_metrics.png")


def plot_monthly_incremental_results(seen_months_results: dict, future_month_results: dict, results_dir: Path):
    """
    绘制增量学习评估曲线。

    Args:
        seen_months_results: 已见月份评估结果 {'month': [...], 'f1': [...], 'acc': [...], 'precision': [...], 'recall': [...]}
        future_month_results: 后一个月评估结果 {'month': [...], 'f1': [...], 'acc': [...], 'precision': [...], 'recall': [...]}
        results_dir: 保存图片的目录
    """
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 已见月份 F1 曲线
    ax1 = axes[0, 0]
    ax1.plot(seen_months_results['month'], seen_months_results['f1'], 'b-o', label='F1', markersize=4)
    ax1.plot(seen_months_results['month'], seen_months_results['acc'], 'g--s', label='Accuracy', markersize=4)
    ax1.set_title('Seen Months - F1 & Accuracy')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # 已见月份 Precision/Recall 曲线
    ax2 = axes[0, 1]
    ax2.plot(seen_months_results['month'], seen_months_results['precision'], 'g-o', label='Precision', markersize=4)
    ax2.plot(seen_months_results['month'], seen_months_results['recall'], 'r--s', label='Recall', markersize=4)
    ax2.set_title('Seen Months - Precision & Recall')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # 后一个月 F1/Accuracy 曲线
    ax3 = axes[1, 0]
    ax3.plot(future_month_results['month'], future_month_results['f1'], 'b-o', label='F1', markersize=4)
    ax3.plot(future_month_results['month'], future_month_results['acc'], 'g--s', label='Accuracy', markersize=4)
    ax3.set_title('Next Month - F1 & Accuracy')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)

    # 后一个月 Precision/Recall 曲线
    ax4 = axes[1, 1]
    ax4.plot(future_month_results['month'], future_month_results['precision'], 'g-o', label='Precision', markersize=4)
    ax4.plot(future_month_results['month'], future_month_results['recall'], 'r--s', label='Recall', markersize=4)
    ax4.set_title('Next Month - Precision & Recall')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Score')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'incremental_learning_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - incremental_learning_metrics.png")


if __name__ == "__main__":
    # 测试代码：可以从JSON文件加载结果并绘图
    import json

    results_dir = Path("/Data2/hxq/datasets/incremental_packages_subset")

    with open(results_dir / "val_period_test_results.json", 'r') as f:
        val_period_results = json.load(f)
    with open(results_dir / "future_test_results.json", 'r') as f:
        future_test_results = json.load(f)

    plot_complete_metrics_curve(val_period_results, future_test_results, results_dir)
    print(f"Plots saved to {results_dir}")
