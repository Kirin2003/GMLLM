"""测试单个包的恶意性检测模型"""
from pathlib import Path
from distinguish_GNN_2 import GCNWithBehavior, load_dict
import json
import torch
import torch.nn as nn
from torch_geometric.data import Data


def load_vocabs(vocab_dir: Path):
    """加载词汇表（复用 load_dict）"""
    return (
        load_dict(vocab_dir / "name2idx.json"),
        load_dict(vocab_dir / "type2idx.json"),
        load_dict(vocab_dir / "behavior2idx.json"),
        load_dict(vocab_dir / "edge_type2idx.json"),
    )


def load_model(model_path: Path, name_vocab_size: int, type_vocab_size: int, behavior_dim: int, device: str = "cpu"):
    """加载最佳模型"""
    model = GCNWithBehavior(
        name_vocab_size=name_vocab_size,
        type_vocab_size=type_vocab_size,
        behavior_dim=behavior_dim
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def build_graph_from_json(call_graph_path: Path, name2idx: dict, type2idx: dict,
                          behavior2idx: dict, edge_type2idx: dict, label: int = None) -> Data:
    """从 call_graph.json 构建 PyG Data 对象"""
    with open(call_graph_path, 'r', encoding='utf-8', errors='ignore') as f:
        graph = json.load(f)

    name_ids, type_ids, behavior_feats = [], [], []
    for node in graph.get('nodes', []):
        name = node.get('qualified_name') or node.get('name', 'unknown_name')
        node_type = node.get('type', 'unknown_type')
        behaviors = node.get('behaviors', [])

        name_ids.append(name2idx.get(name, name2idx.get('unknown_name', 0)))
        type_ids.append(type2idx.get(node_type, type2idx.get('unknown_type', 0)))

        behavior_vec = torch.zeros(len(behavior2idx))
        for b in behaviors:
            if b in behavior2idx:
                behavior_vec[behavior2idx[b]] = 1
        behavior_feats.append(behavior_vec)

    id_map = {node['id']: i for i, node in enumerate(graph.get('nodes', []))}
    edges, edge_attrs = [], []
    for link in graph.get('links', []):
        src = id_map.get(link.get('source'))
        tgt = id_map.get(link.get('target'))
        edge_type = link.get('edge_type', 'unknown')
        if src is not None and tgt is not None:
            edges.append([src, tgt])
            edge_attrs.append(edge_type2idx.get(edge_type, -1))

    data = Data(
        x_names=torch.tensor(name_ids, dtype=torch.long),
        x_types=torch.tensor(type_ids, dtype=torch.long),
        x_behaviors=torch.stack(behavior_feats),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(edge_attrs, dtype=torch.long),
        name=call_graph_path.parent.name
    )
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
    return data


def predict_package(model: nn.Module, data: Data, device: str = "cpu") -> dict:
    """对单个包进行预测"""
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1).item()
        probs = torch.softmax(out, dim=1)
        prob_malicious = probs[0][1].item()
        prob_benign = probs[0][0].item()

    return {
        "prediction": "malicious" if pred == 1 else "benign",
        "pred_class": pred,
        "prob_benign": prob_benign,
        "prob_malicious": prob_malicious,
        "package_name": data.name
    }


def test_single_package(package_json_path: str, model_path: str, vocab_dir: str, device: str = "cpu"):
    """
    测试单个包

    Args:
        package_json_path: call_graph.json 文件路径
        model_path: best_model.pt 模型路径
        vocab_dir: vocab 目录路径
        device: 设备 (cuda/cpu)
    """
    package_path = Path(package_json_path)
    model_path = Path(model_path)
    vocab_dir = Path(vocab_dir)

    print(f"Package: {package_path.parent.name}")
    print(f"Model: {model_path}")
    print(f"Vocabs: {vocab_dir}")
    print(f"Device: {device}")
    print("-" * 50)

    # 1. 加载 vocab（复用 load_dict）
    name2idx, type2idx, behavior2idx, edge_type2idx = load_vocabs(vocab_dir)
    print(f"  Vocab sizes: name={len(name2idx)}, type={len(type2idx)}, behavior={len(behavior2idx)}, edge={len(edge_type2idx)}")

    # 2. 加载模型（复用 GCNWithBehavior）
    model = load_model(
        model_path=model_path,
        name_vocab_size=len(name2idx),
        type_vocab_size=len(type2idx),
        behavior_dim=len(behavior2idx),
        device=device
    )
    print(f"  Model loaded successfully")

    # 3. 构建图
    data = build_graph_from_json(package_path, name2idx, type2idx, behavior2idx, edge_type2idx)
    print(f"  Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")

    # 4. 预测
    result = predict_package(model, data, device)
    print("-" * 50)
    print(f"  Prediction: {result['prediction']} (class={result['pred_class']})")
    print(f"  Probability - Benign: {result['prob_benign']:.4f}, Malicious: {result['prob_malicious']:.4f}")

    return result


def convert_package_to_pt(call_graph_path: str, vocab_dir: str, output_path: str = None, label: int = None):
    """
    将单个包的 call_graph.json 转换为 .pt 文件

    Args:
        call_graph_path: call_graph.json 文件路径
        vocab_dir: vocab 目录路径
        output_path: 输出的 .pt 文件路径 (可选，默认保存在同目录下)
        label: 标签，0=benign, 1=malicious (可选)

    Returns:
        Data: PyG Data 对象
    """
    call_graph_path = Path(call_graph_path)
    vocab_dir = Path(vocab_dir)

    print(f"Converting: {call_graph_path}")
    print(f"Package: {call_graph_path.parent.name}")

    # 1. 加载 vocab（复用 load_dict）
    name2idx, type2idx, behavior2idx, edge_type2idx = load_vocabs(vocab_dir)
    print(f"  Vocab sizes: name={len(name2idx)}, type={len(type2idx)}, behavior={len(behavior2idx)}, edge={len(edge_type2idx)}")

    # 2. 构建图
    data = build_graph_from_json(call_graph_path, name2idx, type2idx, behavior2idx, edge_type2idx, label)
    print(f"  Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")

    # 3. 保存为 .pt
    if output_path is None:
        output_path = call_graph_path.parent / f"{call_graph_path.parent.name}.pt"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    print(f"  Saved to: {output_path}")

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test or convert a single package")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test a single package")
    test_parser.add_argument("--package", required=True, help="Path to call_graph.json")
    test_parser.add_argument("--model", required=True, help="Path to best_model.pt")
    test_parser.add_argument("--vocab", required=True, help="Path to vocab directory")
    test_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert call_graph.json to .pt")
    convert_parser.add_argument("--package", required=True, help="Path to call_graph.json")
    convert_parser.add_argument("--vocab", required=True, help="Path to vocab directory")
    convert_parser.add_argument("--output", help="Output .pt file path (optional)")
    convert_parser.add_argument("--label", type=int, choices=[0, 1], help="Label: 0=benign, 1=malicious (optional)")

    args = parser.parse_args()

    if args.command == "test":
        result = test_single_package(args.package, args.model, args.vocab, args.device)
    elif args.command == "convert":
        data = convert_package_to_pt(args.package, args.vocab, args.output, args.label)
    else:
        parser.print_help()
