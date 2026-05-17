import sys
from pathlib import Path

# 将上级目录加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'

import json
import time
import torch
from pathlib import Path
import argparse
import yaml
from utils.config_utils import load_config
from tqdm import tqdm
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from utils.month_utils import generate_month_range
from sentence_transformers import SentenceTransformer

# 加载 GTE 模型 (local_files_only=True 强制只读本地缓存，避免验证请求)
gte_model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)

def generate_name_embeddings(name2idx, batch_size=64, chunk_size=1000):
    import numpy as np
    import torch
    from tqdm import tqdm

    # 1. 确保在 [CPU] 上开辟内存，绝对不给 GPU 尝试申请 96GB 的机会
    total_count = len(name2idx)
    embedding_dim = 768  # GTE-base 的固定维度
    print(f"Creating RAM-based matrix for {total_count} names...")
    
    # 使用 memmap 或简单的 numpy (在系统内存里，不在显存里)
    # 160万 * 768 * 4 bytes ≈ 4.6 GB RAM
    final_embeddings_np = np.zeros((total_count, embedding_dim), dtype=np.float32)

    # 2. 准备模型：显式指定 fp16 且关闭梯度
    gte_model.to('cuda')
    gte_model.half()
    gte_model.eval()

    names = [name for name in name2idx.keys() if name != 'unknown_name']
    
    # 3. 核心循环
    with torch.inference_mode(): # 彻底关闭梯度计算，省显存
        for chunk_start in tqdm(range(0, len(names), chunk_size), desc="Encoding"):
            chunk_end = min(chunk_start + chunk_size, len(names))
            chunk_names = [str(n) for n in names[chunk_start:chunk_end]]

            # 这里的 encode 必须极其干净
            chunk_embs = gte_model.encode(
                chunk_names,
                batch_size=batch_size,
                convert_to_numpy=True,    # 算出后立刻回传 CPU
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # 4. 直接映射到 CPU 矩阵，跳过中间 Tensor 转换
            for i, name in enumerate(chunk_names):
                idx = name2idx[name]
                final_embeddings_np[idx] = chunk_embs[i]

            # 强制清理当前 chunk 显存
            del chunk_embs
            torch.cuda.empty_cache()

    # 5. 返回 CPU 上的 Tensor (4.6 GB)
    # 显式指定 device='cpu'，防止它默认去 cuda 申请空间
    name_embeddings = torch.from_numpy(final_embeddings_np).to('cpu')
    
    print(f"Done! Memory efficiency: {name_embeddings.element_size() * name_embeddings.nelement() / 1024**3:.2f} GB in RAM")
    return name_embeddings

class CallGraphDatasetFull_Lazy(Dataset):
    def __init__(self, root_dir, output_dir=None, fixed_label=None,
                 name2idx=None, type2idx=None, edge_type2idx=None, behavior2idx=None,
                 name_embeddings=None,
                 transform=None, pre_transform=None, start_month='2022-01', end_month='2024-12'):
        self.root_dir = Path(root_dir)
        self.start_month = start_month
        self.end_month = end_month
        self.month_dirs = [self.root_dir / month for month in generate_month_range(start_month, end_month)]
        self.output_dir = Path(output_dir) if output_dir else self.root_dir / 'processed'
        self.fixed_label = fixed_label
        assert fixed_label is not None, "Must provide fixed_label"
        assert name2idx and type2idx and edge_type2idx and behavior2idx, "Must provide name2idx, type2idx, edge_type2idx, and behavior2idx"
        self.name2idx = name2idx
        self.type2idx = type2idx
        self.edge_type2idx = edge_type2idx
        self.behavior2idx = behavior2idx
        self.name_embeddings = name_embeddings  # GTE embeddings for names

        # 调用父类初始化（必须在访问数据之前）
        super().__init__(self.output_dir, transform=transform, pre_transform=pre_transform)

        # 收集所有月份的数据
        self._load_all_months_data()

    def _get_missing_months(self):
        """检查哪些月份的数据尚未处理"""
        return [m for m in generate_month_range(self.start_month, self.end_month)
                if not (self.output_dir / m / 'index_gte.json').exists()]

    def _load_all_months_data(self):
        """加载所有月份的数据路径"""
        self.graph_paths_by_month = {}  # {month: [path1, path2, ...]}
        self.all_graph_paths = []       # 所有图的扁平列表

        for month_dir in self.month_dirs:
            month_output_dir = self.output_dir / month_dir.name
            index_file = month_output_dir / 'index_gte.json'

            if not index_file.exists():
                continue

            with open(index_file, 'r') as f:
                graph_paths = json.load(f)

            # 存储相对路径
            self.graph_paths_by_month[month_dir.name] = [
                f"{month_dir.name}/{p}" for p in graph_paths
            ]
            self.all_graph_paths.extend(self.graph_paths_by_month[month_dir.name])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['index_gte.json']
    def download(self):
        pass

    def process(self):
        # 检查是否所有月份都已处理，避免重复处理
        missing_months = self._get_missing_months()

        if not missing_months:
            return

        print(f"Missing months detected: {missing_months}. Processing...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for month_dir in self.month_dirs:
            if not month_dir.exists():
                print(f"[WARNING] Month directory {month_dir} does not exist, skipping.")
                continue

            # 为每个月创建子目录
            month_output_dir = self.output_dir / month_dir.name
            month_output_dir.mkdir(parents=True, exist_ok=True)

            print('Start processing month directory:', month_dir)
            graph_paths = []
            for folder in tqdm(os.listdir(month_dir), desc=f"Processing {month_dir.name}"):
                call_graph_file = month_dir / folder / 'call_graph.json'
                if not call_graph_file.exists():
                    continue
                try:
                    with open(call_graph_file, 'r', encoding='utf-8', errors='ignore') as f:
                        graph = json.load(f)
                    label = self.fixed_label
                    name_ids, type_ids, behavior_feats, node_raw_attrs = [], [], [], []
                    for idx, node in enumerate(graph.get('nodes', [])):
                        name = node.get('qualified_name') or node.get('name', 'unknown_name')
                        node_type = node.get('type', 'unknown_type')
                        behaviors = node.get('behaviors', [])
                        name_ids.append(self.name2idx.get(name, self.name2idx['unknown_name']))
                        type_ids.append(self.type2idx.get(node_type, self.type2idx['unknown_type']))
                        behavior_vec = torch.zeros(len(self.behavior2idx))
                        for b in behaviors:
                            if b in self.behavior2idx:
                                behavior_vec[self.behavior2idx[b]] = 1
                        behavior_feats.append(behavior_vec)
                        node_raw_attrs.append({
                            'id': node.get('id', f'missing_id_{idx}'),
                            'name': name,
                            'type': node_type,
                            'file': node.get('file', ''),
                            'behaviors': behaviors
                        })
                    assert len(name_ids) == len(type_ids) == len(behavior_feats), "Node features are misaligned"
                    id_map = {node['id']: i for i, node in enumerate(graph.get('nodes', []))}
                    edges, edge_attrs = [], []
                    for link in graph.get('links', []):
                        src = id_map.get(link.get('source'))
                        tgt = id_map.get(link.get('target'))
                        edge_type = link.get('edge_type', 'unknown')
                        if src is not None and tgt is not None:
                            edges.append([src, tgt])
                            edge_attrs.append(self.edge_type2idx.get(edge_type, -1))
                    if len(name_ids) == 0 or len(behavior_feats) == 0:
                        print(f"[WARNING] Skipping {folder}: empty node list or no behaviors.")
                        continue

                    # 使用 GTE embeddings 替换原始 name indices
                    x_names_tensor = torch.tensor(name_ids, dtype=torch.long)
                    x_names_gte = self.name_embeddings[x_names_tensor]  # (num_nodes, embedding_dim)

                    data = Data(
                        x_names=x_names_gte,  # 使用 GTE embeddings
                        x_types=torch.tensor(type_ids, dtype=torch.long),
                        x_behaviors=torch.stack(behavior_feats),
                        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long),
                        edge_attr=torch.tensor(edge_attrs, dtype=torch.long),
                        y=torch.tensor([label], dtype=torch.long),
                        name=folder,
                        node_raw_attrs=node_raw_attrs,
                        graph_raw=graph,
                        num_nodes=len(name_ids)
                    )
                    save_path = month_output_dir / f"{folder}_gte.pt"
                    torch.save(data, save_path)
                    graph_paths.append(f"{folder}_gte.pt")
                except Exception as e:
                    print(f"[ERROR] Failed to process {folder}: {e}")

            # 每个月份生成独立的index_gte.json
            with open(month_output_dir / 'index_gte.json', 'w') as f:
                json.dump(graph_paths, f, indent=2)
            print(f"  {month_dir.name}: {len(graph_paths)} graphs processed.")

        # 重新加载所有月份的数据
        self._load_all_months_data()

    def len(self):
        return len(self.all_graph_paths)

    def get(self, idx):
        rel_path = self.all_graph_paths[idx]
        return torch.load(self.output_dir / rel_path, map_location='cpu', weights_only=False)

    def get_by_month(self, month):
        """获取指定月份的所有图"""
        if month not in self.graph_paths_by_month:
            return []
        return [self.get(i) for i in range(len(self.all_graph_paths))
                if self.all_graph_paths[i].startswith(f"{month}/")]
def build_global_vocab(root_dirs, start_month='2022-01', end_month='2023-02'):
    print('Start building global vocab...')
    name_set, type_set, edge_type_set, behavior_set = set(), set(), set(), set()
    for root_dir in root_dirs:
        for month in generate_month_range(start_month, end_month):
            print('Scanning month directory:', month)
            month_dir = Path(root_dir) / month
            for folder in tqdm(os.listdir(month_dir)):
                json_path = month_dir / folder / "call_graph.json"
                if not json_path.exists():
                    with open('error_log.txt', 'a') as log_f:
                        log_f.write(f"[MISSING] {json_path}\n")
                try:
                    with open(json_path, 'r') as f:
                        graph = json.load(f)
                    for node in graph.get("nodes", []):
                        name = node.get("qualified_name") or node.get("name", "unknown_name")
                        type_ = node.get("type", "unknown_type")
                        name_set.add(name)
                        type_set.add(type_)
                        for b in node.get("behaviors", []):
                            behavior_set.add(b)
                    for link in graph.get("links", []):
                        edge_type_set.add(link.get("edge_type", "unknown"))
                except Exception as e:
                    print(f"[ERROR] Failed to parse {json_path}: {e}")
    name_list = sorted(name_set - {'unknown_name'})
    name2idx = {'unknown_name': 0, **{name: i + 1 for i, name in enumerate(name_list)}}
    type_list = sorted(type_set - {'unknown_type'})
    type2idx = {'unknown_type': 0, **{t: i + 1 for i, t in enumerate(type_list)}}
    edge_type2idx = {et: i for i, et in enumerate(sorted(edge_type_set))}
    behavior2idx = {b: i for i, b in enumerate(sorted(behavior_set))}
    return name2idx, type2idx, edge_type2idx, behavior2idx
def clean_dir(path):
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".pt") or f == "index_gte.json":
                os.remove(Path(path) / f)


if __name__ == "__main__":
    # 加载配置文件
    config_path = "./configs/default_gte.yaml"
    config = load_config(config_path)

    # 从配置中读取路径
    base_path = config['dataset']['base_path']
    normal_root = str(Path(base_path) / config['dataset']['benign_root'])
    malicious_root = str(Path(base_path) / config['dataset']['malicious_root'])
    normal_out = str(Path(base_path) / config['dataset']['benign_out'])
    malicious_out = str(Path(base_path) / config['dataset']['malicious_out'])
    VOCAB_DIR = str(Path(base_path) / config['dataset']['vocab_dir'])

    # 从配置中读取时间范围
    cl_config = config.get('continual_learning', {})
    base_train_months = cl_config.get('base_train_months', ['2022-01', '2023-02'])
    incremental_months = cl_config.get('incremental_months', ['2023-03', '2024-12'])
    vocab_start_month = base_train_months[0]
    vocab_end_month = base_train_months[1]
    data_start_month = base_train_months[0]
    data_end_month = incremental_months[1]

    print("\n[1/4] Building global vocab...")
    vocab_start = time.time()
    # name2idx, type2idx, edge_type2idx, behavior2idx = build_global_vocab([normal_root, malicious_root], start_month=vocab_start_month, end_month=vocab_end_month)
    name2idx, type2idx, edge_type2idx, behavior2idx = build_global_vocab([malicious_root], start_month=vocab_start_month, end_month=vocab_end_month)
    vocab_time = time.time() - vocab_start
    print(f"  build_global_vocab completed in {vocab_time:.2f}s")
    os.makedirs(VOCAB_DIR, exist_ok=True)
    with open(str(Path(VOCAB_DIR) / "name2idx.json"), 'w') as f:
        json.dump(name2idx, f, indent=2)
    with open(str(Path(VOCAB_DIR) / "type2idx.json"), 'w') as f:
        json.dump(type2idx, f, indent=2)
    with open(str(Path(VOCAB_DIR) / "edge_type2idx.json"), 'w') as f:
        json.dump(edge_type2idx, f, indent=2)
    with open(str(Path(VOCAB_DIR) / "behavior2idx.json"), 'w') as f:
        json.dump(behavior2idx, f, indent=2)

    print("\n[2/4] Generating name embeddings with GTE...")
    name_emb_start = time.time()
    name_embeddings = generate_name_embeddings(name2idx)
    torch.save(name_embeddings, str(Path(VOCAB_DIR) / "name_embeddings.pt"))
    name_emb_time = time.time() - name_emb_start
    print(f"  Name embeddings saved to {VOCAB_DIR}/name_embeddings.pt")
    print(f"  Shape: {name_embeddings.shape}")

    clean_dir(normal_out)
    clean_dir(malicious_out)

    print("\n[3/4] Processing benign packages...")
    benign_start = time.time()
    _ = CallGraphDatasetFull_Lazy(
        root_dir=normal_root,
        output_dir=normal_out,
        name2idx=name2idx,
        type2idx=type2idx,
        edge_type2idx=edge_type2idx,
        behavior2idx=behavior2idx,
        name_embeddings=name_embeddings,
        fixed_label=0,
        start_month=data_start_month,
        end_month=data_end_month
    )
    benign_time = time.time() - benign_start
    print(f"  CallGraphDatasetFull_Lazy (benign) completed in {benign_time:.2f}s")

    print("\n[4/4] Processing malicious packages...")
    malicious_start = time.time()
    _ = CallGraphDatasetFull_Lazy(
        root_dir=malicious_root,
        output_dir=malicious_out,
        name2idx=name2idx,
        type2idx=type2idx,
        edge_type2idx=edge_type2idx,
        behavior2idx=behavior2idx,
        name_embeddings=name_embeddings,
        fixed_label=1,
        start_month=data_start_month,
        end_month=data_end_month
    )
    malicious_time = time.time() - malicious_start
    print(f"  CallGraphDatasetFull_Lazy (malicious) completed in {malicious_time:.2f}s")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  build_global_vocab:       {vocab_time:.2f}s")
    print(f"  generate_name_embeddings: {name_emb_time:.2f}s")
    print(f"  CallGraphDatasetFull_Lazy (benign):    {benign_time:.2f}s")
    print(f"  CallGraphDatasetFull_Lazy (malicious): {malicious_time:.2f}s")
    print(f"  Total:                    {vocab_time + name_emb_time + benign_time + malicious_time:.2f}s")
    print("=" * 60)