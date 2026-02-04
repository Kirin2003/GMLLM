""" explainer_main.py

     Main user interface for the explainer module.
"""
import numpy as np
import argparse
import os

import sklearn.metrics as metrics

from tensorboardX import SummaryWriter

import pickle
import shutil
import torch

# import models
from distinguish_GNN_2 import GCNWithBehavior
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from explainer import explain
import json
from torch_geometric.utils import dense_to_sparse
def load_dict(path):
    with open(path, 'r') as f:
        return json.load(f)
# name2idx = load_dict("/media/pt/malicious/LLM_Malicious_Pypi/graph_data_Pypi/vocab/name2idx.json")
# type2idx = load_dict("/media/pt/malicious/LLM_Malicious_Pypi/graph_data_Pypi/vocab/type2idx.json")
# behavior2idx = load_dict("/media/pt/malicious/LLM_Malicious_Pypi/graph_data_Pypi/vocab/behavior2idx.json")
name2idx = load_dict("/media/pt/malicious/filterData/ablation/humanRule/vocab/name2idx.json")
type2idx = load_dict("/media/pt/malicious/filterData/ablation/humanRule/vocab/type2idx.json")
behavior2idx = load_dict("/media/pt/malicious/filterData/ablation/humanRule/vocab/behavior2idx.json")
import torch
from torch_geometric.data import Data

class GCNWithBehavior_Wrapper(torch.nn.Module):
    """
    这个 Wrapper 用来让 Explainer 直接调用 `forward(x, adj)`，
    它会在内部把 x, adj 还原成 PyG 的 Data 对象，再交给 GCNWithBehavior。
    """
    def __init__(self, base_model,cg_dict):
        super().__init__()
        self.base_model = base_model
        device = next(base_model.parameters()).device

        # cg_dict["name_idx"]: [1, N]
        self.name_idx = torch.from_numpy(cg_dict["name_idx"][0]).long().to(device)        # [N]
        # cg_dict["type_idx"]: [1, N]
        self.type_idx = torch.from_numpy(cg_dict["type_idx"][0]).long().to(device)        # [N]
        # cg_dict["behavior_feats"]: [1, N, behavior_dim]
        self.behavior_feats = torch.from_numpy(cg_dict["behavior_feats"][0]).float().to(device)  # [N, behavior_dim]


    def forward(self, x_tensor, adj_tensor, **kwargs):
        if isinstance(adj_tensor, torch.Tensor) and adj_tensor.dim() == 3:
            adj = adj_tensor.squeeze(0)  # 从 [1, N, N] 变成 [N, N]
        elif isinstance(adj_tensor, np.ndarray) and adj_tensor.ndim == 3:
            adj = torch.from_numpy(adj_tensor).to(self.base_model.name_emb.weight.device)
            adj = adj.squeeze(0)
        else:
            adj = adj_tensor
            if isinstance(adj, np.ndarray):
                adj = torch.from_numpy(adj).to(self.base_model.name_emb.weight.device)
        adj = adj.to(self.base_model.name_emb.weight.device)

        # mask = (adj > 0)
        # src, dst = torch.nonzero(mask, as_tuple=True)
        # edge_index = torch.stack([src, dst], dim=0)  # [2, #edges]
        mask_mat = adj  # adj 自身就是对称、浮点的掩码 * 原始 adj
        src, dst = torch.nonzero(mask_mat > 0, as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0)      # [2, E]
        # 对应的 edge_weight 就是掩码后的值
        edge_weight = mask_mat[src, dst] 

        N = self.name_idx.size(0)
        batch = torch.zeros(N, dtype=torch.long, device=self.name_idx.device)

        data = Data(
            x_names=self.name_idx,       # [N]
            x_types=self.type_idx,       # [N]
            x_behaviors=self.behavior_feats,  # [N, behavior_dim]
            edge_index=edge_index,       # [2, #edges]
            edge_weight=edge_weight, 
            batch=batch                  # [N]
        )
        pred = self.base_model(data)
        return pred, adj
    
def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    parser_utils.parse_optimizer(parser)

    parser.add_argument("--clean-log", action="store_true", help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--explain-node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=False,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )
    parser.add_argument(
        "--explainer-suffix",
        dest="explainer_suffix",
        help="suffix added to the explainer log",
    )

    # TODO: Check argument usage
    parser.set_defaults(
        logdir="log",
        ckptdir="ckpt",
        dataset="syn1",
        opt="adam",  
        opt_scheduler="none",
        cuda="0",
        lr=0.1,
        clip=2.0,
        batch_size=20,
        num_epochs=100,
        hidden_dim=64,
        output_dim=20,
        num_gc_layers=3,
        dropout=0.7,
        method="base",
        name_suffix="",
        explainer_suffix="",
        align_steps=1000,
        explain_node=None,
        graph_idx=-1,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
    )
    return parser.parse_args()


def main():
    # Load a configuration
    prog_args = arg_parse()

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")
        return 

    # Configure the logging directory 
    if prog_args.writer:
        path = os.path.join(prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
        if os.path.isdir(path) and prog_args.clean_log:
           print('Removing existing log dir: ', path)
           if not input("Are you sure you want to remove this directory? (y/n): ").lower().strip()[:1] == "y": sys.exit(1)
           shutil.rmtree(path)
        writer = SummaryWriter(path)
    else:
        writer = None

    # Load a model checkpoint
    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt["cg_dict"] # get computation graph
    # input_dim = cg_dict["feat"].shape[2] 
    input_dim   = cg_dict["behavior_feats"].shape[2] + 64 + 16
    num_classes = cg_dict["pred"].shape[2]
    print("Loaded model from {}".format(prog_args.ckptdir))
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    # Determine explainer mode
    graph_mode = (
        prog_args.graph_mode
        or prog_args.multigraph_class >= 0
        or prog_args.graph_idx >= 0
    )
    name_vocab_size = len(name2idx)
    type_vocab_size = len(type2idx)
    behavior_dim = len(behavior2idx) 

    # build model
    print("Method: ", prog_args.method)
    base_model = GCNWithBehavior(
        name_vocab_size=name_vocab_size,
        type_vocab_size=type_vocab_size,
        behavior_dim=behavior_dim,
        hidden_dim=64,
        num_classes=num_classes
    )
    if prog_args.gpu:
        base_model = base_model.cuda()
    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    base_model.load_state_dict(ckpt["model_state_dict"])
    
    print("Using GCNWithBehavior (loaded weights).")
    model = GCNWithBehavior_Wrapper(base_model,cg_dict) 
    
    if prog_args.gpu:
        model = model.cuda()
    # Create explainer
    explainer = explain.Explainer(
        model=model,
        adj=cg_dict["adj"],
        feat=cg_dict["feat"],
        label=cg_dict["label"],
        pred=cg_dict["pred"],
        train_idx=cg_dict["train_idx"],
        args=prog_args,
        writer=writer,
        print_training=True,
        graph_mode=graph_mode,
        graph_idx=prog_args.graph_idx,
    )

    # TODO: API should definitely be cleaner
    # Let's define exactly which modes we support 
    # We could even move each mode to a different method (even file)
    if prog_args.explain_node is not None:
        explainer.explain(prog_args.explain_node, unconstrained=False)
    elif graph_mode:
        if prog_args.multigraph_class >= 0:
            print(cg_dict["label"])
            # only run for graphs with label specified by multigraph_class
            labels = cg_dict["label"].numpy()
            graph_indices = []
            for i, l in enumerate(labels):
                if l == prog_args.multigraph_class:
                    graph_indices.append(i)
                if len(graph_indices) > 30:
                    break
            print(
                "Graph indices for label ",
                prog_args.multigraph_class,
                " : ",
                graph_indices,
            )
            explainer.explain_graphs(graph_indices=graph_indices)

        elif prog_args.graph_idx == -1:
            # just run for a customized set of indices
            explainer.explain_graphs(graph_indices=[1, 2, 3, 4])
        else:
            explainer.explain(
                node_idx=0,
                graph_idx=prog_args.graph_idx,
                graph_mode=True,
                unconstrained=False,
            )
            # io_utils.plot_cmap_tb(writer, "tab20", 20, "tab20_cmap")
    else:
        if prog_args.multinode_class >= 0:
            print(cg_dict["label"])
            # only run for nodes with label specified by multinode_class
            labels = cg_dict["label"][0]  # already numpy matrix

            node_indices = []
            for i, l in enumerate(labels):
                if len(node_indices) > 4:
                    break
                if l == prog_args.multinode_class:
                    node_indices.append(i)
            print(
                "Node indices for label ",
                prog_args.multinode_class,
                " : ",
                node_indices,
            )
            explainer.explain_nodes(node_indices, prog_args)

        else:
            # explain a set of nodes
            masked_adj = explainer.explain_nodes_gnn_stats(
                range(400, 700, 5), prog_args
            )

if __name__ == "__main__":
    main()

