import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import Subset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ---------- 模型定义 ----------

class GCNWithBehavior(nn.Module):
    def __init__(self, name_vocab_size, type_vocab_size, behavior_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        self.name_emb = nn.Embedding(name_vocab_size, 64)
        self.type_emb = nn.Embedding(type_vocab_size, 16)
        input_dim = 64 + 16 + behavior_dim  # name + type + behaviors
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, data):
        name_feat = self.name_emb(data.x_names)
        type_feat = self.type_emb(data.x_types)
        behavior_feat = data.x_behaviors.float()
        # nobehavior
        # behavior_feat = torch.zeros_like(behavior_feat)  

        x = torch.cat([name_feat, type_feat, behavior_feat], dim=1)

        x = self.dropout(self.conv1(x, data.edge_index,data.edge_weight).relu())
        x = self.dropout(self.conv2(x, data.edge_index,data.edge_weight).relu())
        
        #no behavior
        # x = self.dropout(self.conv1(x, data.edge_index).relu())
        # x = self.dropout(self.conv2(x, data.edge_index).relu())

        if x.shape[0] != data.batch.shape[0]:
            min_len = min(x.shape[0], data.batch.shape[0])
            x = x[:min_len]
            batch = data.batch[:min_len]
            # print(f"[WARN] Mismatched x vs batch → Truncated to {min_len}")
        else:
            batch = data.batch

        x = global_mean_pool(x, batch)
        x = self.dropout(x) 
        return self.classifier(x)


class GCNWithBehaviorExpandable(nn.Module):
    """
    支持动态扩容的GCN模型
    当发现新API时，可以扩展embedding层的词汇表大小
    """
    def __init__(self, name_vocab_size, type_vocab_size, behavior_dim,
                 hidden_dim=64, num_classes=2, name_emb_dim=64, type_emb_dim=16):
        super().__init__()
        self.name_emb_dim = name_emb_dim
        self.type_emb_dim = type_emb_dim
        self.name_emb = nn.Embedding(name_vocab_size, name_emb_dim)
        self.type_emb = nn.Embedding(type_vocab_size, type_emb_dim)
        self.behavior_dim = behavior_dim
        self.hidden_dim = hidden_dim

        input_dim = name_emb_dim + type_emb_dim + behavior_dim
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.7)

    def expand_vocab(self, new_name_size, new_type_size, device):
        """
        动态扩展embedding层

        Args:
            new_name_size: 新的name词汇表大小
            new_type_size: 新的type词汇表大小
            device: 设备
        """
        old_name_size = self.name_emb.num_embeddings
        old_type_size = self.type_emb.num_embeddings

        # 扩展name embedding
        if new_name_size > old_name_size:
            new_name_emb = nn.Embedding(new_name_size, self.name_emb_dim).to(device)
            with torch.no_grad():
                # 复制旧权重
                new_name_emb.weight[:old_name_size] = self.name_emb.weight
                # 新行初始化为UNK的权重（第一行）
                new_name_emb.weight[old_name_size:] = self.name_emb.weight[0]
            self.name_emb = new_name_emb
            print(f"[EXPAND] name_emb: {old_name_size} -> {new_name_size}")

        # 扩展type embedding
        if new_type_size > old_type_size:
            new_type_emb = nn.Embedding(new_type_size, self.type_emb_dim).to(device)
            with torch.no_grad():
                new_type_emb.weight[:old_type_size] = self.type_emb.weight
                new_type_emb.weight[old_type_size:] = self.type_emb.weight[0]
            self.type_emb = new_type_emb
            print(f"[EXPAND] type_emb: {old_type_size} -> {new_type_size}")

    def expand_gcn_layers(self, new_name_size, new_type_size, device):
        """
        扩展GCN卷积层的输入维度
        """
        old_input_dim = self.name_emb_dim + self.type_emb_dim + self.behavior_dim
        new_input_dim = self.name_emb_dim + self.type_emb_dim + self.behavior_dim

        if old_input_dim == new_input_dim:
            return

        # 创建新的卷积层
        self.conv1 = GCNConv(new_input_dim, self.hidden_dim).to(device)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim).to(device)
        print(f"[EXPAND] GCN layers input dim: {old_input_dim} -> {new_input_dim}")

    def forward(self, data):
        name_feat = self.name_emb(data.x_names)
        type_feat = self.type_emb(data.x_types)
        behavior_feat = data.x_behaviors.float()

        x = torch.cat([name_feat, type_feat, behavior_feat], dim=1)

        x = self.dropout(self.conv1(x, data.edge_index, data.edge_weight).relu())
        x = self.dropout(self.conv2(x, data.edge_index, data.edge_weight).relu())

        if x.shape[0] != data.batch.shape[0]:
            min_len = min(x.shape[0], data.batch.shape[0])
            x = x[:min_len]
            batch = data.batch[:min_len]
        else:
            batch = data.batch

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.classifier(x)


class GCNWithLLMFeature(nn.Module):
    """
    结合预训练LLM特征的GCN模型
    将LLM提取的API语义特征与原有特征拼接
    """
    def __init__(self, name_vocab_size, type_vocab_size, behavior_dim,
                 llm_feature_dim: int = 768, llm_proj_dim: int = 32,
                 hidden_dim: int = 64, num_classes: int = 2,
                 use_llm_only: bool = False, use_llm_concat: bool = True,
                 freeze_llm: bool = True):
        """
        Args:
            name_vocab_size: name词汇表大小
            type_vocab_size: type词汇表大小
            behavior_dim: 行为特征维度
            llm_feature_dim: LLM特征维度（默认768 for GraphCodeBERT）
            llm_proj_dim: LLM特征投影后的维度
            hidden_dim: 隐藏层维度
            num_classes: 分类类别数
            use_llm_only: 是否只使用LLM特征（替换原有embedding）
            use_llm_concat: 是否拼接LLM特征
            freeze_llm: 是否冻结LLM特征
        """
        super().__init__()
        self.use_llm_only = use_llm_only
        self.use_llm_concat = use_llm_concat
        self.freeze_llm = freeze_llm
        self.llm_feature_dim = llm_feature_dim

        # 原有embedding层
        self.name_emb = nn.Embedding(name_vocab_size, 64)
        self.type_emb = nn.Embedding(type_vocab_size, 16)

        # LLM特征处理
        if use_llm_only:
            input_dim = llm_feature_dim
        elif use_llm_concat:
            self.llm_proj = nn.Linear(llm_feature_dim, llm_proj_dim)
            input_dim = 64 + 16 + llm_proj_dim + behavior_dim
        else:
            input_dim = 64 + 16 + behavior_dim

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, data, llm_features: torch.Tensor = None):
        """
        Args:
            data: 图数据
            llm_features: 预提取的LLM特征 [num_nodes, llm_feature_dim]
        """
        name_feat = self.name_emb(data.x_names)
        type_feat = self.type_emb(data.x_types)
        behavior_feat = data.x_behaviors.float()

        if self.use_llm_only and llm_features is not None:
            x = llm_features
        elif self.use_llm_concat and llm_features is not None:
            llm_feat = self.llm_proj(llm_features)
            x = torch.cat([name_feat, type_feat, llm_feat, behavior_feat], dim=1)
        else:
            # fallback到原始特征
            x = torch.cat([name_feat, type_feat, behavior_feat], dim=1)

        x = self.dropout(self.conv1(x, data.edge_index, data.edge_weight).relu())
        x = self.dropout(self.conv2(x, data.edge_index, data.edge_weight).relu())

        if x.shape[0] != data.batch.shape[0]:
            min_len = min(x.shape[0], data.batch.shape[0])
            x = x[:min_len]
            batch = data.batch[:min_len]
        else:
            batch = data.batch

        x = global_mean_pool(x, batch)
        x = self.dropout(x) 
        return self.classifier(x)


# ---------- 工具函数 ----------
def load_dict(path):
    with open(path, 'r') as f:
        return json.load(f)


# ---------- 训练 ----------
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
        total_loss += loss.item() * data.num_graphs

    return total_loss / total, correct / total


# ---------- 验证 ----------
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
    
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1], zero_division=0)


    malicious_f1 = f1[1]
    malicious_precision = p[1]
    malicious_recall = r[1]
    
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    
    # 打印更清晰的日志
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    benign_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"[Validation] Overall Acc: {acc:.4f} | Malicious F1: {malicious_f1:.4f}")
    print(f" └─ Malicious Metrics: Precision: {malicious_precision:.4f} | Recall: {malicious_recall:.4f} (TP:{tp}, FN:{fn})")
    print(f" └─ Benign Accuracy..: {benign_acc:.4f} (TN:{tn}, FP:{fp})")
    
    # 返回我们最关心的指标：恶意样本的召回率和F1分数
    return malicious_f1, acc, malicious_recall

