from __future__ import annotations
import json
import re
from pathlib import Path

# 用于清理字符串中的无效代理字符
INVALID_SURROGATE_PATTERN = re.compile(r'[\ud800-\udfff]')


def _clean_invalid_surrogates(obj):
    """递归清理对象中的无效代理字符"""
    if isinstance(obj, str):
        # 移除无效的代理字符
        return INVALID_SURROGATE_PATTERN.sub('', obj)
    elif isinstance(obj, dict):
        return {k: _clean_invalid_surrogates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_invalid_surrogates(item) for item in obj]
    else:
        return obj


REQUIRED_NODE_FIELDS = {"id", "type", "name", "qualified_name", "file", "behaviors"}
REQUIRED_LINK_FIELDS = {"source", "target", "edge_type"}
def quick_validate(graph: dict):
    assert "nodes" in graph and "links" in graph, "graph must contain 'nodes' and 'links'"
    for n in graph["nodes"]:
        missing = REQUIRED_NODE_FIELDS - set(n.keys())
        assert not missing, f"node missing fields: {missing}"
        assert isinstance(n["behaviors"], list), "node.behaviors must be a list"
    for e in graph["links"]:
        missing = REQUIRED_LINK_FIELDS - set(e.keys())
        assert not missing, f"link missing fields: {missing}"
def save_call_graph(graph: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    quick_validate(graph)
    # 清理无效代理字符后写入
    clean_graph = _clean_invalid_surrogates(graph)
    (out_dir / "call_graph.json").write_text(json.dumps(clean_graph, ensure_ascii=False, indent=2), encoding="utf-8")