from __future__ import annotations
import argparse
from pathlib import Path
from ast_parser import parse_file
import json
from graph_builder import ProjectGraphBuilder
from llm_detector import LLMBehaviorDetector
from exporter import save_call_graph

def extract_call_graph(src_path: Path | str, out_path: Path | str,
                       model_name: str = "gpt-4o",
                       cache_path: Path | str | None = None,
                       detector: LLMBehaviorDetector | None = None) -> dict:
    """
    从一个Python包中提取call graph。

    Args:
        src_path: Python项目/包源码目录
        out_path: 输出目录
        model_name: LLM模型名称 (默认 "gpt-4o")
        cache_path: 可选的缓存文件路径
        detector: 可选的预配置检测器（用于批量处理时复用规则）

    Returns:
        dict: call graph数据 (包含nodes和links)
    """
    src_path = Path(src_path)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

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

    # 如果没有传入检测器，创建一个新的（每次都会尝试合成规则）
    if detector is None:
        detector = LLMBehaviorDetector(
            model_name=model_name,
            use_rule_fallback=True,
            cache_path=cache_path,
        )
        # 自动规则合成
        try:
            synth_out = out_path / "synth_rules.json"
            obj = detector.synthesize_rules()
            synth_out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] synthesized rules to {synth_out}")
            detector.load_synth_rules(synth_out)
        except Exception as e:
            print(f"[warn] synthesize failed ({e}); will proceed with fallback rules.")

    # 生成并保存call graph
    gb.attach_behaviors(detector)
    graph = gb.to_jsonable()
    save_call_graph(graph, out_path)

    print(f"[ok] wrote {out_path / 'call_graph.json'} with {len(graph['nodes'])} nodes and {len(graph['links'])} links.")
    return graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-extract", type=Path, help="Path to extract_config.json")
    ap.add_argument("--src", type=Path, required=True, help="Path to a Python project directory")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for call_graph.json")
    ap.add_argument("--model_name", default="mock-llm")
    ap.add_argument("--synth-rules", action="store_true", help="If API is available, synthesize rules first and apply for this run")
    ap.add_argument("--synth-rules-out", type=Path, default=Path("synth_rules.json"), help="Where to save synthesized rules JSON")
    
    ap.add_argument("--no-fallback", action="store_true", help="kept for interface compatibility")
    ap.add_argument("--cache", type=Path, default=None, help="optional JSON cache file for detections")
    args = ap.parse_args()

    if args.config_extract and args.config_extract.exists():
        try:
            _cfg = json.loads(args.config_extract.read_text(encoding="utf-8"))
            _paths = _cfg.get("paths", {})
            _llm = _cfg.get("llm", {})
            _det = _cfg.get("detector", {})
            # Paths
            if not args.src and _paths.get("src_dir"): args.src = _paths["src_dir"]
            if not args.out and _paths.get("out_dir"): args.out = _paths["out_dir"]
            if getattr(args, "synth_rules_out", None) is not None and _paths.get("synth_rules_out"):
                args.synth_rules_out = Path(_paths["synth_rules_out"])
            if getattr(args, "cache", None) is None and _paths.get("cache_file"):
                args.cache = Path(_paths["cache_file"])
            # LLM options
            if getattr(args, "model_name", None) in (None, "mock-llm") and _llm.get("model_name"):
                args.model_name = _llm["model_name"]
            if hasattr(args, "temperature") and _llm.get("temperature") is not None:
                args.temperature = float(_llm["temperature"])
            if hasattr(args, "max_retries") and _llm.get("max_retries") is not None:
                args.max_retries = int(_llm["max_retries"])
            if hasattr(args, "timeout_s") and _llm.get("timeout_s") is not None:
                args.timeout_s = float(_llm["timeout_s"])
            # detector fallback
            if hasattr(args, "no_fallback") and _det.get("use_rule_fallback") is False:
                args.no_fallback = True
            # auto synthesize toggle
            AUTO_SYNTH_FROM_CFG = bool(_llm.get("auto_synthesize", True))
        except Exception as _e:
            print(f"[warn] failed to read extract config: {_e}")
            AUTO_SYNTH_FROM_CFG = True
    else:
        AUTO_SYNTH_FROM_CFG = True
    gb = ProjectGraphBuilder()
    for py in sorted(args.src.rglob("*.py")):
        if py.name in {"cli_extract.py", "graph_builder.py", "ast_parser.py", "llm_detector.py", "exporter.py"}:
            continue
        try:
            mod = parse_file(py)
            gb.add_module(mod)
        except Exception as e:
            print(f"[warn] failed to parse {py}: {e}")
    detector = LLMBehaviorDetector(model_name=args.model_name, use_rule_fallback=not args.no_fallback, cache_path=args.cache)

    # Auto synthesize-then-apply if requested/available
    def _api_available():
        import os
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"))
    auto_synth = (AUTO_SYNTH_FROM_CFG if 'AUTO_SYNTH_FROM_CFG' in globals() else True)
    print('auto_synth=', auto_synth, 'args.synth_rules:', args.synth_rules)
    if auto_synth or args.synth_rules:
        try:
            obj = detector.synthesize_rules()
            args.synth_rules_out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] synthesized rules to {args.synth_rules_out}")
            detector.load_synth_rules(args.synth_rules_out)
        except Exception as e:
            print(f"[warn] synthesize failed ({e}); will proceed with fallback rules.")

    # Auto synthesize-then-apply if requested and API likely available
    def _api_available():
        import os
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"))

    auto_synth = args.synth_rules or _api_available()
    print('auto_synth:', auto_synth)
    if auto_synth:
        try:
            # LLM自动规则生成
            obj = detector.synthesize_rules()
            args.synth_rules_out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] synthesized rules to {args.synth_rules_out}")
            detector.load_synth_rules(args.synth_rules_out)
        except Exception as e:
            # 没有LLM API Key，回退
            print(f"[warn] synthesize failed ({e}); will proceed with fallback rules.")
    else:
        try:
            obj = detector.synthesize_rules()
            args.synth_rules_out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] synthesized rules to {args.synth_rules_out}")
        except Exception as e:
            print(f"[warn] failed to synthesize rules: {e}")
        return
    gb.attach_behaviors(detector)
    graph = gb.to_jsonable()
    save_call_graph(graph, args.out)
    print(f"[ok] wrote {args.out/'call_graph.json'} with {len(graph['nodes'])} nodes and {len(graph['links'])} links.")
import time
from datetime import datetime

LOG_FILE = Path("/Data2/hxq/GMLLM/GMLLM/extractAST/batch_extract.log")

def log(msg: str):
    """输出日志到文件和控制台"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")


def batch_extract_call_graphs(base_path: Path | str, model_name: str = "qwen3-max"):
    """
    批量处理所有月份的恶意包和良性包，生成call graph。
    只在开始时合成一次规则，然后复用。

    Args:
        base_path: incremental_packages 的父目录
        model_name: LLM模型名称
    """
    base_path = Path(base_path)
    log_file_path = Path("/Data2/hxq/GMLLM/GMLLM/extractAST/batch_extract.log")
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 清空日志文件
    log_file_path.write_text("", encoding="utf-8")

    log("=" * 60)
    log("开始批量提取 Call Graph")
    log(f"Base path: {base_path}")
    log(f"Model: {model_name}")
    log("=" * 60)

    # 只合成一次规则，复用于所有包
    log("\n[1/2] 正在合成规则...")
    detector = LLMBehaviorDetector(
        model_name=model_name,
        use_rule_fallback=True,
    )
    try:
        obj = detector.synthesize_rules()
        synth_path = Path("/Data2/hxq/GMLLM/GMLLM/extractAST/synth_rules.json")
        synth_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        detector.load_synth_rules(synth_path)
        log(f"[ok] 规则合成成功，保存在 {synth_path}")
    except Exception as e:
        log(f"[warn] 规则合成失败，将使用 fallback 规则: {e}")

    log("\n[2/2] 开始处理所有包...")
    overall_start = time.time()
    total_packages = 0

    # 获取所有月份目录 (yyyy-mm 格式)
    months = sorted([d for d in base_path.iterdir() if d.is_dir() and len(d.name) == 7 and d.name[4] == "-"])

    for month in months:
        month_str = month.name
        log(f"\n{'='*60}")
        log(f"正在处理 {month_str} 月")
        log("=" * 60)

        for pkg_type in ["malicious", "benign"]:
            pkg_dir = month / pkg_type
            if not pkg_dir.exists():
                log(f"  {pkg_type}: 目录不存在，跳过")
                continue

            log(f"\n  正在处理 {month_str} 月的 {pkg_type} 包")

            # 获取所有包目录
            packages = [p for p in pkg_dir.iterdir() if p.is_dir()]
            month_pkg_count = 0
            month_start = time.time()

            for pkg in sorted(packages):
                pkg_start = time.time()
                log(f"    处理包: {pkg.name}")

                out_path = pkg
                try:
                    # 传入已配置好的 detector，复用规则
                    extract_call_graph(
                        src_path=pkg,
                        out_path=out_path,
                        detector=detector,
                    )
                    log(f"    完成: {pkg.name} (耗时: {time.time() - pkg_start:.1f}s)")
                except Exception as e:
                    log(f"    失败: {pkg.name} - {e}")

                month_pkg_count += 1
                total_packages += 1

            month_time = time.time() - month_start
            log(f"\n  {month_str} 月 {pkg_type} 包处理完成: {month_pkg_count} 个包, 耗时: {month_time:.1f}s")

    total_time = time.time() - overall_start
    log(f"\n{'='*60}")
    log("批量处理完成!")
    log(f"总共处理了 {total_packages} 个包")
    log(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}分钟)")
    log("=" * 60)


if __name__ == "__main__":
    batch_extract_call_graphs(
        base_path="/Data2/hxq/datasets/incremental_packages",
        model_name="qwen3-max",
    )
    # extract_call_graph(src_path="/Data2/hxq/datasets/incremental_packages/2022-01/malicious/AadhaarCrypt-1.0",
    #                    out_path="/Data2/hxq/datasets/incremental_packages/2022-01/malicious/AadhaarCrypt-1.0",
    #                    model_name="qwen3-max")