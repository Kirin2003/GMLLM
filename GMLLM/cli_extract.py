from __future__ import annotations
import sys
from pathlib import Path

# 将上级目录加入 Python 搜索路径
sys.path.insert(0, '/Data2/hxq/GMLLM')

import argparse
from pathlib import Path
from ast_parser import parse_file
import json
import yaml
from utils.config_utils import load_config
from graph_builder import ProjectGraphBuilder
from llm_detector import LLMBehaviorDetector
from exporter import save_call_graph
from utils.month_utils import generate_month_range
from utils.logger_utils import Logger
log = Logger("batch_extract.log")

def extract_call_graph(detector: LLMBehaviorDetector,
                       src_path: Path | str,
                       out_path: Path | str) -> dict:
    """
    从一个Python包中提取call graph。

    Args:
        detector: 预配置的检测器（用于批量处理时复用规则）
        src_path: Python项目/包源码目录
        out_path: 输出文件完整路径

    Returns:
        dict: call graph数据 (包含nodes和links)
    """
    src_path = Path(src_path)
    out_path = Path(out_path)

    # 构建图
    gb = ProjectGraphBuilder()
    for py in sorted(src_path.rglob("*.py")):
        if py.name in {"cli_extract.py", "graph_builder.py", "ast_parser.py", "llm_detector.py", "exporter.py"}:
            continue
        try:
            mod = parse_file(py)
            gb.add_module(mod)
        except Exception as e:
            print(f"[warn] failed to parse {py}: {e}")

    # 生成并保存call graph
    gb.attach_behaviors(detector)
    graph = gb.to_jsonable()
    save_call_graph(graph, out_path)

    print(f"[ok] wrote {out_path} with {len(graph['nodes'])} nodes and {len(graph['links'])} links.")
    return graph


import time


def batch_extract_call_graphs(detector: LLMBehaviorDetector, base_path: Path | str, model_name: str):
    """
    批量处理所有月份的恶意包和良性包，生成call graph。

    Args:
        detector: 预配置的检测器
        base_path: incremental_packages 的父目录
        model_name: 模型名称（用于生成文件名）
    """
    base_path = Path(base_path)

    log.log("=" * 60)
    log.log("开始批量提取 Call Graph")
    log.log(f"Base path: {base_path}")
    log.log("=" * 60)

    log.log("\n开始处理所有包...")
    overall_start = time.time()
    total_packages = 0

    pkg_types = ["benign", "malicious"]

    for pkg_type in pkg_types:
        type_dir = base_path / pkg_type

        log.log(f"\n{'='*60}")
        log.log(f"正在处理 {pkg_type} 包")
        log.log("=" * 60)

        # 获取该类型下的所有月份目录
        months = generate_month_range("2022-01", "2024-12")

        for month_str in months:
            month_dir = type_dir / month_str

            if not month_dir.exists():
                log.log(f"  {month_str}: 目录不存在，跳过")
                continue

            log.log(f"\n  正在处理 {month_str} 月的 {pkg_type} 包")

            # 获取所有包目录
            packages = [p for p in month_dir.iterdir() if p.is_dir()]
            month_pkg_count = 0
            month_start = time.time()

            for pkg in packages:
                pkg_start = time.time()
                log.log(f"    处理包: {pkg.name}")

                out_path = pkg / f"{model_name}_call_graph.json"
                try:
                    # 传入已配置好的 detector，复用规则
                    extract_call_graph(
                        detector=detector,
                        src_path=pkg,
                        out_path=out_path,
                    )
                    log.log(f"    完成: {pkg.name} (耗时: {time.time() - pkg_start:.1f}s)")
                except Exception as e:
                    log.log(f"    失败: {pkg.name} - {e}")

                month_pkg_count += 1
                total_packages += 1

            month_time = time.time() - month_start
            log.log(f"\n  {month_str} 月 {pkg_type} 包处理完成: {month_pkg_count} 个包, 耗时: {month_time:.1f}s")

    total_time = time.time() - overall_start
    log.log(f"\n{'='*60}")
    log.log("批量处理完成!")
    log.log(f"总共处理了 {total_packages} 个包")
    log.log(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}分钟)")
    log.log("=" * 60)


if __name__ == "__main__":
    # 读取配置文件
    config_path = "./configs/default.yaml"
    config = load_config(config_path)

    llm_config = config.get("llm", {})
    dataset_config = config.get("dataset", {})

    model_name = llm_config.get("model_name", "qwen3-max")
    synth_rules = llm_config.get("synth_rules", False)
    synth_path = Path(f"./{model_name}_synth_rules.json")
    pkg_path = dataset_config.get("base_path", "")

    log.log("=" * 60)
    log.log("开始批量提取 Call Graph")
    log.log(f"Model: {model_name}")
    log.log(f"Dataset: {pkg_path}")
    log.log("=" * 60)

    # 创建 detector
    detector = LLMBehaviorDetector(
        model_name=model_name,
        use_rule_fallback=True,
    )
    if synth_rules:
        log.log(f"正在使用 {model_name} 合成规则...")
        try:
            obj = detector.synthesize_rules()
            synth_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

            log.log(f"[ok] 规则合成成功，保存在 {synth_path}")
        except Exception as e:
            log.log(f"[warn] 规则合成失败，将使用 fallback 规则: {e}")

    log.log("加载规则...")
    try:
        detector.load_synth_rules(synth_path)
        log.log(f"[ok] 规则加载成功: {synth_path}")
    except Exception as e:
        log.log(f"[warn] 规则加载失败，将使用 fallback 规则: {e}")

    log.log(f"使用 {model_name} 处理所有包...")
    overall_start = time.time()
    batch_extract_call_graphs(
        detector=detector,
        base_path=pkg_path,
        model_name=model_name,
    )
    total_time = time.time() - overall_start
    log.log(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f}min)")