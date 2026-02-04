import torch
import numpy as np
import json
import os
from torch_geometric.data import Data
from distinguish_GNN_2 import GCNWithBehavior    
from torch_geometric.loader import DataLoader
import argparse


# def load_dict(path):
#     with open(path, 'r') as f:
#         return json.load(f)
# name2idx = load_dict("/media/pt/malicious/LLM_Malicious_Pypi/graph_data_Pypi/vocab/name2idx.json")
# type2idx = load_dict("/media/pt/malicious/LLM_Malicious_Pypi/graph_data_Pypi/vocab/type2idx.json")
# behavior2idx = load_dict("/media/pt/malicious/LLM_Malicious_Pypi/graph_data_Pypi/vocab/behavior2idx.json")

def build_and_save_single_ckpt(pt_graph_path, model_ckpt_path,output_ckpt_path,deviceCuda):
    #加载单图 Data 对象
    if torch.cuda.is_available():
        device = torch.device(deviceCuda)  
    else:
        device = torch.device('cpu')
    data: Data = torch.load(pt_graph_path,weights_only=False)
    # device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    print(device)
    data = data.to(device)
    
    #恢复训练好的 GNN 模型
    ckpt = torch.load(model_ckpt_path, map_location=device)
    model = GCNWithBehavior(        # 与训练时初始化的参数一致
        name_vocab_size=len(name2idx),
        type_vocab_size=len(type2idx),
        behavior_dim=len(behavior2idx),
        hidden_dim=64, 
        num_classes=2 )  
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    
    # 构造稠密邻接矩阵 adj（N×N）并扩维成 1×N×N
    N = data.num_nodes
    adj_dense = torch.zeros((N, N), dtype=torch.float32, device=device)
    # PyG 的 edge_index 里，行 0 表示源节点索引，行 1 表示目标节点索引
    src, dst = data.edge_index
    adj_dense[src, dst] = 1.0
    adj_np = adj_dense.cpu().numpy()[np.newaxis, :, :]  # shape = [1, N, N]
    
    # 构造原始节点特征 feat（N×F）并扩维成 1×N×F
    name_feat = data.x_names 
    type_feat = data.x_types 
    behavior_feat = data.x_behaviors.float() 
    #设置节点数量（根据name特征的长度）
    data.num_nodes = data.x_names.size(0)
    with torch.no_grad():
        name_emb = model.name_emb(name_feat)  
        type_emb = model.type_emb(type_feat)  
        x = torch.cat([name_emb, type_emb, behavior_feat], dim=1)  
    data.x = x
    feat_np = data.x.cpu().numpy()[np.newaxis, :, :]   # shape = [1, N, F]
    x_names_np     = data.x_names.cpu().numpy()[np.newaxis, :]       # [1, N]
    x_types_np     = data.x_types.cpu().numpy()[np.newaxis, :]       # [1, N]
    x_behaviors_np = data.x_behaviors.cpu().numpy()[np.newaxis, :, :]

    label_np = np.array([ int(data.y.item()) ], dtype=np.int64)  # shape = [1]
    
    # 让模型前向一次得到 pred，再扩维成 [1, 2]
    single_loader = DataLoader([data],batch_size=1)
    with torch.no_grad():
        for batch_data in single_loader:
            batch_data = batch_data.to(device)
            out = model(batch_data)   
            prob = torch.softmax(out.view(-1), dim=0).cpu().numpy() 
    pred_np = prob[np.newaxis,np.newaxis,  :]             # shape = [1,1, 2]
    
    train_idx = list(range(N))
    
    # cg_dict = {
    #     'adj':       adj_np.astype(np.float32),
    #     'feat':      feat_np.astype(np.float32),
    #     'label':     label_np.astype(np.int64),
    #     'pred':      pred_np.astype(np.float32),
    #     'train_idx': train_idx
    # }
    cg_dict = {
        'adj':            adj_np.astype(np.float32),      # [1, N, N]
        'feat':           feat_np.astype(np.float32),     # [1, N, F]  ← 给 Explainer 用
        'name_idx':       x_names_np.astype(np.int64),    # [1, N]
        'type_idx':       x_types_np.astype(np.int64),    # [1, N]
        'behavior_feats': x_behaviors_np.astype(np.float32), # [1, N, behavior_dim]
        'label':          label_np.astype(np.int64),      # [1]
        'pred':           pred_np.astype(np.float32),     # [1, 1, 2]
        'train_idx':      train_idx
    }



    # 模型权重 + cg_dict 保存到 single_graph_ckpt.pth.tar
    new_ckpt = {
        'model_state_dict': model.state_dict(),
        'cg_dict':          cg_dict
    }
    torch.save(new_ckpt, output_ckpt_path)
    print(f" 单图 checkpoint 已保存到：{output_ckpt_path}")


if __name__ == "__main__":
    # pt_graph_path    = "/home/pt/LLM_Malicious_Pypi/graph_data_Pypi/handled_malicious/15cent.pt"      # 单张图文件
    # parser = argparse.ArgumentParser(description="Build single graph checkpoint for GNNExplainer.")
    # parser.add_argument("--pt_filename", type=str, required=True,
    #                     help="Filename of the .pt graph (e.g., 15cent.pt, NOT the full path).")
    
    # args = parser.parse_args()
    # # BASE_HANDLED_MALICIOUS_DIR = "/media/pt/malicious/LLM_Malicious_Pypi/graph_data_Pypi/handled_malicious_all"
    # BASE_HANDLED_MALICIOUS_DIR = "/media/pt/malicious/LLM_Malicious_Pypi/graph_data_Pypi/handled_malicious_all"
    # pt_graph_path = os.path.join(BASE_HANDLED_MALICIOUS_DIR, args.pt_filename+".pt")

    # model_ckpt_path  = "/home/pt/new_best_model.pth" # 训练好权重文件
    # output_ckpt_path = "/home/pt/Explainer_Malicious/checkpoint/single_graph_ckpt.pth.tar"
    # build_and_save_single_ckpt(pt_graph_path, model_ckpt_path, output_ckpt_path)



    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_filename", required=True)
    parser.add_argument("--model_weights", required=True)
    parser.add_argument("--vocab_dir", required=True)
    parser.add_argument("--output_ckpt_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--handled_dir", required=True)
    args = parser.parse_args()

    with open(os.path.join(args.vocab_dir, 'name2idx.json'), 'r') as f: name2idx = json.load(f)
    with open(os.path.join(args.vocab_dir, 'type2idx.json'), 'r') as f: type2idx = json.load(f)
    with open(os.path.join(args.vocab_dir, 'behavior2idx.json'), 'r') as f: behavior2idx = json.load(f)
    pt_graph_path = os.path.join(args.handled_dir, args.pt_filename+".pt")
    build_and_save_single_ckpt( pt_graph_path , args.model_weights , args.output_ckpt_dir,args.device)

