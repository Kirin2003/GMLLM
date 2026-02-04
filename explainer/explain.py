import math
import time
import os

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import pandas as pd
import tensorboardX.utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import grad

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN

import pdb

import utils.io_utils as io_utils
import utils.train_utils as train_utils
import utils.graph_utils as graph_utils

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

class Explainer:
    def __init__(
        self,
        model,
        adj,
        feat,
        label,
        pred,
        train_idx,
        args,
        writer=None,
        print_training=True,
        graph_mode=False,
        graph_idx=False,
    ):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.n_hops = args.num_gc_layers
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        self.neighborhoods = None if self.graph_mode else graph_utils.neighborhoods(adj=self.adj, n_hops=self.n_hops, use_cuda=use_cuda)
        self.args = args
        self.writer = writer
        self.print_training = print_training

        # ==== 新增：定义小图的可视化阈值 ====
        self.NODE_VIS_THRESHOLD = 5 # 例如，节点数小于等于5被视为小图
        self.EDGE_VIS_THRESHOLD = 5 # 例如，边数小于等于5被视为小图


    # Main method
    def explain(
        self, node_idx, graph_idx=0, graph_mode=False, unconstrained=False, model="exp"
    ):
        """Explain a single node prediction
        """
        # index of the query node in the new adj
        if graph_mode:
            node_idx_new = node_idx
            sub_adj = self.adj[graph_idx]
            sub_feat = self.feat[graph_idx, :]
            sub_label = self.label[graph_idx]
            neighbors = np.asarray(range(self.adj.shape[0]))
        else:
            print("node label: ", self.label[graph_idx][node_idx])
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                node_idx, graph_idx
            )
            print("neigh graph idx: ", node_idx, node_idx_new)
            sub_label = np.expand_dims(sub_label, axis=0)

        # sub_adj = np.expand_dims(sub_adj, axis=0)
        # sub_feat = np.expand_dims(sub_feat, axis=0)

        # adj   = torch.tensor(sub_adj, dtype=torch.float)
        # x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        # label = torch.tensor(sub_label, dtype=torch.long)

        # 确保 sub_adj 和 sub_feat 是 tensor 并添加 batch dimension [1, N, N] 和 [1, N, D]
        sub_adj_tensor = torch.tensor(sub_adj, dtype=torch.float).unsqueeze(0)
        sub_feat_tensor = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float).unsqueeze(0)
        label = torch.tensor(sub_label, dtype=torch.long)

        # ==== 新增：判断当前子图是否为小图（用于可视化阈值调整） ====
        current_sub_nodes = sub_adj_tensor.size()[1]
        current_sub_edges = torch.sum(sub_adj_tensor[0] > 0).item() # 假设 sub_adj_tensor 是二值的

        is_current_graph_small = False
        if current_sub_nodes <= self.NODE_VIS_THRESHOLD or current_sub_edges <= self.EDGE_VIS_THRESHOLD:
            is_current_graph_small = True
            print(f"Current subgraph for visualization is small (Nodes: {current_sub_nodes}, Edges: {current_sub_edges}). Visualization threshold will be adjusted.")
        # ==========================================================



        if self.graph_mode:
            pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
            print("Graph predicted label: ", pred_label)
        else:
            pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)
            print("Node predicted label: ", pred_label[node_idx_new])

        explainer = ExplainModule(
            adj=sub_adj_tensor,
            x=sub_feat_tensor,
            model=self.model,
            label=label,
            args=self.args,
            writer=self.writer,
            graph_idx=self.graph_idx,
            graph_mode=self.graph_mode,
            ###
            is_small_graph=is_current_graph_small, # <-- 传递给 ExplainModule
        )
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()


        # gradient baseline
        if model == "grad":
            explainer.zero_grad()
            # pdb.set_trace()
            # 在这里，由于 ExplainerModule._masked_adj() 会计算 self.current_masked_adj
            # 并且我们现在直接从 self.current_masked_adj.grad 获取梯度，
            # 这里需要确保在 log_adj_grad 被调用前，self.current_masked_adj 已经经过了一次前向计算
            # 所以这里直接调用 explainModule 的 forward 就可以
            ypred_dummy, _ = explainer(node_idx_new, unconstrained=unconstrained) # 确保 current_masked_adj 被计算
            # adj_grad = explainer.adj_feat_grad(node_idx_new, pred_label[node_idx_new])[0] #
            # masked_adj = adj_grad + adj_grad.t()
            adj_grad, _ = explainer.log_adj_grad(node_idx_new, pred_label[node_idx_new], epoch)
            masked_adj = (adj_grad + adj_grad.t()) * 0.5
            masked_adj = nn.functional.sigmoid(masked_adj)
            masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
        else:
            explainer.train()
            begin_time = time.time()
            best_loss = float("inf")
            no_improve = 0
            patience = 10       
            min_epochs = 20
            for epoch in range(self.args.num_epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred, adj_atts = explainer(node_idx_new, unconstrained=unconstrained)
                loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
                loss.backward()

                explainer.optimizer.step()
                if explainer.scheduler is not None:
                    explainer.scheduler.step()

                mask_density = explainer.mask_density()

                cur_loss = loss.item()
                if cur_loss + 1e-6 < best_loss:
                    best_loss = cur_loss
                    no_improve = 0
                else:
                    no_improve += 1

                if self.print_training:
                    print(
                        "epoch: ",
                        epoch,
                        "; loss: ",
                        loss.item(),
                        "; mask density: ",
                        mask_density.item(),
                        "; pred: ",
                        ypred,
                    )
                single_subgraph_label = sub_label.squeeze()

                if epoch + 1 >= min_epochs and no_improve >= patience:
                    print(f"[EXPLAINER_STEPS] steps={epoch+1} final_loss={best_loss:.4f}")
                    break

                if self.writer is not None:
                    self.writer.add_scalar("mask/density", mask_density, epoch)
                    self.writer.add_scalar(
                        "optimization/lr",
                        explainer.optimizer.param_groups[0]["lr"],
                        epoch,
                    )
                    if epoch % 25 == 0:
                        explainer.log_mask(epoch)
                        explainer.log_masked_adj(
                            node_idx_new, epoch, label=single_subgraph_label
                        )
                        explainer.log_adj_grad(
                            node_idx_new, pred_label, epoch, label=single_subgraph_label
                        ) #

                    # if epoch == 0:
                    #     if self.model.att:
                    #         # explain node
                    #         print("adj att size: ", adj_atts.size())
                    #         adj_att = torch.sum(adj_atts[0], dim=2)
                    #         # adj_att = adj_att[neighbors][:, neighbors]
                    #         node_adj_att = adj_att * adj.float().cuda()
                    #         io_utils.log_matrix(
                    #             self.writer, node_adj_att[0], "att/matrix", epoch
                    #         )
                    #         node_adj_att = node_adj_att[0].cpu().detach().numpy()
                    #         G = io_utils.denoise_graph(
                    #             node_adj_att,
                    #             node_idx_new,
                    #             threshold=3.8,  # threshold_num=20,
                    #             max_component=True,
                    #         )
                    #         io_utils.log_graph(
                    #             self.writer,
                    #             G,
                    #             name="att/graph",
                    #             identify_self=not self.graph_mode,
                    #             nodecolor="label",
                    #             edge_vmax=None,
                    #             args=self.args,
                    #         )
                if model != "exp":
                    break

            print("finished training in ", time.time() - begin_time)
            if model == "exp":
                masked_adj = (
                    explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )
            else:
                adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
                masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()

        fname = 'masked_adj_' + io_utils.gen_explainer_prefix(self.args) + (
                'node_idx_'+str(node_idx)+'graph_idx_'+str(self.graph_idx)+'.npy')
        with open(os.path.join(self.args.logdir, fname), 'wb') as outfile:
            np.save(outfile, np.asarray(masked_adj.copy()))
            print("Saved adjacency matrix to ", fname)
        return masked_adj


    # NODE EXPLAINER
    def explain_nodes(self, node_indices, args, graph_idx=0):
        """
        Explain nodes

        Args:
            - node_indices  :  Indices of the nodes to be explained
            - args          :  Program arguments (mainly for logging paths)
            - graph_idx     :  Index of the graph to explain the nodes from (if multiple).
        """
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx) for node_idx in node_indices
        ]
        ref_idx = node_indices[0]
        ref_adj = masked_adjs[0]
        curr_idx = node_indices[1]
        curr_adj = masked_adjs[1]
        new_ref_idx, _, ref_feat, _, _ = self.extract_neighborhood(ref_idx)
        new_curr_idx, _, curr_feat, _, _ = self.extract_neighborhood(curr_idx)

        G_ref = io_utils.denoise_graph(ref_adj, new_ref_idx, ref_feat, threshold=0.1)
        denoised_ref_feat = np.array(
            [G_ref.nodes[node]["feat"] for node in G_ref.nodes()]
        )
        denoised_ref_adj = nx.to_numpy_matrix(G_ref)
        # ref center node
        ref_node_idx = list(G_ref.nodes()).index(new_ref_idx)

        G_curr = io_utils.denoise_graph(
            curr_adj, new_curr_idx, curr_feat, threshold=0.1
        )
        denoised_curr_feat = np.array(
            [G_curr.nodes[node]["feat"] for node in G_curr.nodes()]
        )
        denoised_curr_adj = nx.to_numpy_matrix(G_curr)
        # curr center node
        curr_node_idx = list(G_curr.nodes()).index(new_curr_idx)

        P, aligned_adj, aligned_feat = self.align(
            denoised_ref_feat,
            denoised_ref_adj,
            ref_node_idx,
            denoised_curr_feat,
            denoised_curr_adj,
            curr_node_idx,
            args=args,
        )
        io_utils.log_matrix(self.writer, P, "align/P", 0)

        G_ref = nx.convert_node_labels_to_integers(G_ref)
        io_utils.log_graph(self.writer, G_ref, "align/ref")
        G_curr = nx.convert_node_labels_to_integers(G_curr)
        io_utils.log_graph(self.writer, G_curr, "align/before")

        P = P.cpu().detach().numpy()
        aligned_adj = aligned_adj.cpu().detach().numpy()
        aligned_feat = aligned_feat.cpu().detach().numpy()

        aligned_idx = np.argmax(P[:, curr_node_idx])
        print("aligned self: ", aligned_idx)
        G_aligned = io_utils.denoise_graph(
            aligned_adj, aligned_idx, aligned_feat, threshold=0.5
        )
        io_utils.log_graph(self.writer, G_aligned, "mask/aligned")

        # io_utils.log_graph(self.writer, aligned_adj.cpu().detach().numpy(), new_curr_idx,
        #        'align/aligned', epoch=1)

        return masked_adjs


    def explain_nodes_gnn_stats(self, node_indices, args, graph_idx=0, model="exp"):
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx, model=model)
            for node_idx in node_indices
        ]
        # pdb.set_trace()
        graphs = []
        feats = []
        adjs = []
        pred_all = []
        real_all = []
        for i, idx in enumerate(node_indices):
            new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
            G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat, threshold_num=20)
            pred, real = self.make_pred_real(masked_adjs[i], new_idx)
            pred_all.append(pred)
            real_all.append(real)
            denoised_feat = np.array([G.nodes[node]["feat"] for node in G.nodes()])
            denoised_adj = nx.to_numpy_matrix(G)
            graphs.append(G)
            feats.append(denoised_feat)
            adjs.append(denoised_adj)
            io_utils.log_graph(
                self.writer,
                G,
                "graph/{}_{}_{}".format(self.args.dataset, model, i),
                identify_self=True,
                args=self.args
            )

        pred_all = np.concatenate((pred_all), axis=0)
        real_all = np.concatenate((real_all), axis=0)

        auc_all = roc_auc_score(real_all, pred_all)
        precision, recall, thresholds = precision_recall_curve(real_all, pred_all)

        plt.switch_backend("agg")
        plt.plot(recall, precision)
        plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")

        plt.close()

        auc_all = roc_auc_score(real_all, pred_all)
        precision, recall, thresholds = precision_recall_curve(real_all, pred_all)

        plt.switch_backend("agg")
        plt.plot(recall, precision)
        plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")

        plt.close()

        with open("log/pr/auc_" + self.args.dataset + "_" + model + ".txt", "w") as f:
            f.write(
                "dataset: {}, model: {}, auc: {}\n".format(
                    self.args.dataset, "exp", str(auc_all)
                )
            )

        return masked_adjs

    # GRAPH EXPLAINER
    def explain_graphs(self, graph_indices):
        """
        Explain graphs.
        """
        masked_adjs = []

        for graph_idx in graph_indices:
            masked_adj = self.explain(node_idx=0, graph_idx=graph_idx, graph_mode=True)
            G_denoised = io_utils.denoise_graph(
                masked_adj,
                0,
                threshold_num=20,
                feat=self.feat[graph_idx],
                max_component=False,
            )
            label = self.label[graph_idx]
            io_utils.log_graph(
                self.writer,
                G_denoised,
                "graph/graphidx_{}_label={}".format(graph_idx, label),
                identify_self=False,
                nodecolor="feat",
                args=self.args
            )
            masked_adjs.append(masked_adj)

            G_orig = io_utils.denoise_graph(
                self.adj[graph_idx],
                0,
                feat=self.feat[graph_idx],
                threshold=None,
                max_component=False,
            )

            io_utils.log_graph(
                self.writer,
                G_orig,
                "graph/graphidx_{}".format(graph_idx),
                identify_self=False,
                nodecolor="feat",
                args=self.args
            )

        # plot cmap for graphs' node features
        io_utils.plot_cmap_tb(self.writer, "tab20", 20, "tab20_cmap")

        return masked_adjs

    def log_representer(self, rep_val, sim_val, alpha, graph_idx=0):
        """ visualize output of representer instances. """
        rep_val = rep_val.cpu().detach().numpy()
        sim_val = sim_val.cpu().detach().numpy()
        alpha = alpha.cpu().detach().numpy()
        sorted_rep = sorted(range(len(rep_val)), key=lambda k: rep_val[k])
        print(sorted_rep)
        topk = 5
        most_neg_idx = [sorted_rep[i] for i in range(topk)]
        most_pos_idx = [sorted_rep[-i - 1] for i in range(topk)]
        rep_idx = [most_pos_idx, most_neg_idx]

        if self.graph_mode:
            pred = np.argmax(self.pred[0][graph_idx], axis=0)
        else:
            pred = np.argmax(self.pred[graph_idx][self.train_idx], axis=1)
        print(metrics.confusion_matrix(self.label[graph_idx][self.train_idx], pred))
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(5, 3), dpi=600)
        for i in range(2):
            for j in range(topk):
                idx = self.train_idx[rep_idx[i][j]]
                print(
                    "node idx: ",
                    idx,
                    "; node label: ",
                    self.label[graph_idx][idx],
                    "; pred: ",
                    pred,
                )

                idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                    idx, graph_idx
                )
                G = nx.from_numpy_matrix(sub_adj)
                node_colors = [1 for i in range(G.number_of_nodes())]
                node_colors[idx_new] = 0
                # node_color='#336699',

                ax = plt.subplot(2, topk, i * topk + j + 1)
                nx.draw(
                    G,
                    pos=nx.spring_layout(G),
                    with_labels=True,
                    font_size=4,
                    node_color=node_colors,
                    cmap=plt.get_cmap("Set1"),
                    vmin=0,
                    vmax=8,
                    edge_vmin=0.0,
                    edge_vmax=1.0,
                    width=0.5,
                    node_size=25,
                    alpha=0.7,
                )
                ax.xaxis.set_visible(False)
        fig.canvas.draw()
        self.writer.add_image(
            "local/representer_neigh", tensorboardX.utils.figure_to_image(fig), 0
        )

    def representer(self):
        """
        experiment using representer theorem for finding supporting instances.
        https://papers.nips.cc/paper/8141-representer-point-selection-for-explaining-deep-neural-networks.pdf
        """
        self.model.train()
        self.model.zero_grad()
        adj = torch.tensor(self.adj, dtype=torch.float)
        x = torch.tensor(self.feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(self.label, dtype=torch.long)
        if self.args.gpu:
            adj, x, label = adj.cuda(), x.cuda(), label.cuda()

        preds, _ = self.model(x, adj)
        preds.retain_grad()
        self.embedding = self.model.embedding_tensor
        loss = self.model.loss(preds, label)
        loss.backward()
        self.preds_grad = preds.grad
        pred_idx = np.expand_dims(np.argmax(self.pred, axis=2), axis=2)
        pred_idx = torch.LongTensor(pred_idx)
        if self.args.gpu:
            pred_idx = pred_idx.cuda()
        self.alpha = self.preds_grad


    # Utilities
    def extract_neighborhood(self, node_idx, graph_idx=0):
        """Returns the neighborhood of a given ndoe."""
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_feat = self.feat[graph_idx, neighbors]
        sub_label = self.label[graph_idx][neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

    def align(
        self, ref_feat, ref_adj, ref_node_idx, curr_feat, curr_adj, curr_node_idx, args
    ):
        """ Tries to find an alignment between two graphs.
        """
        ref_adj = torch.FloatTensor(ref_adj)
        curr_adj = torch.FloatTensor(curr_adj)

        ref_feat = torch.FloatTensor(ref_feat)
        curr_feat = torch.FloatTensor(curr_feat)

        P = nn.Parameter(torch.FloatTensor(ref_adj.shape[0], curr_adj.shape[0]))
        with torch.no_grad():
            nn.init.constant_(P, 1.0 / ref_adj.shape[0])
            P[ref_node_idx, :] = 0.0
            P[:, curr_node_idx] = 0.0
            P[ref_node_idx, curr_node_idx] = 1.0
        opt = torch.optim.Adam([P], lr=0.01, betas=(0.5, 0.999))
        for i in range(args.align_steps):
            opt.zero_grad()
            feat_loss = torch.norm(P @ curr_feat - ref_feat)

            aligned_adj = P @ curr_adj @ torch.transpose(P, 0, 1)
            align_loss = torch.norm(aligned_adj - ref_adj)
            loss = feat_loss + align_loss
            loss.backward()  # Calculate gradients
            self.writer.add_scalar("optimization/align_loss", loss, i)
            print("iter: ", i, "; loss: ", loss)
            opt.step()

        return P, aligned_adj, P @ curr_feat

    def make_pred_real(self, adj, start):
        # house graph
        if self.args.dataset == "syn1" or self.args.dataset == "syn2":
            # num_pred = max(G.number_of_edges(), 6)
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()

            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start][start + 3] > 0:
                real[start][start + 3] = 10
            if real[start][start + 4] > 0:
                real[start][start + 4] = 10
            if real[start + 1][start + 4]:
                real[start + 1][start + 4] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        # cycle graph
        elif self.args.dataset == "syn4":
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()
            # pdb.set_trace()
            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start + 3][start + 4] > 0:
                real[start + 3][start + 4] = 10
            if real[start + 4][start + 5] > 0:
                real[start + 4][start + 5] = 10
            if real[start][start + 5]:
                real[start][start + 5] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        return pred, real


class ExplainModule(nn.Module):
    def __init__(
        self,
        adj,
        x,
        model,
        label,
        args,
        graph_idx=0,
        writer=None,
        use_sigmoid=True,
        graph_mode=False,
        is_small_graph=False,# <--新增参数
    ):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.graph_idx = graph_idx
        self.args = args
        self.writer = writer
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid
        self.graph_mode = graph_mode
        self.is_small_graph = is_small_graph # <-- 存储参数

        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )

        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy="constant")
        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.scheduler, self.optimizer = train_utils.build_optimizer(args, params)

        self.coeffs = {
                "size": 0.005,
                "feat_size": 1.0,
                "ent": 1.0,
                "feat_ent": 0.1,
                "grad": 0,
                "lap": 1.0,
            }

        # 新增用于存储当前 masked_adj 的属性
        self.current_masked_adj = None

        # # 这个部分决定了GNNExplainer优化时的稀疏性惩罚
        # # 它和可视化阈值可以独立设置
        # NODE_COUNT_THRESHOLD_OPTIM = 5 # GNNExplainer优化时的节点数阈值
        # EDGE_COUNT_THRESHOLD_OPTIM = 5 # GNNExplainer优化时的边数阈值

        # current_num_nodes = adj.size()[1] # N
        # current_num_edges = torch.sum(self.adj[0] > 0).item() # 假设 adj 是二值的，或 >0 代表有边

        # if current_num_nodes <= NODE_COUNT_THRESHOLD_OPTIM or current_num_edges <= EDGE_COUNT_THRESHOLD_OPTIM :
        #     print(f"Detected small graph for optimization (Nodes: {current_num_nodes}, Edges: {current_num_edges}). Setting mask penalties to 0.")
        #     self.coeffs["size"] = 0.0
        #     self.coeffs["feat_size"] = 0.0
        #     self.coeffs["ent"] = 0.0 
        #     self.coeffs["feat_ent"] = 0.0


    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                # mask[0] = 2
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    # 新增辅助函数，用于计算对称掩码
    def _sym_mask(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        return (sym_mask + sym_mask.t()) / 2

    def _masked_adj(self):
        # 1) 拿到原始邻接矩阵（常量）
        adj = self.adj.cuda() if self.args.gpu else self.adj

        # 2) detach + clone → 确保 leaf 张量，然后开启梯度
        adj = adj.clone().detach().requires_grad_(True)

        # 3) 计算对称掩码（sigmoid(edge_mask_logits)）
        sym_mask = self._sym_mask()

        # 4) 做元素乘得到被掩的邻接，并保存到 self.current_masked_adj
        masked_adj = adj * sym_mask
        self.current_masked_adj = masked_adj * self.diag_mask # 保存最终送入模型的邻接
        self.current_masked_adj.retain_grad()
        # 5) 返还最终用于前向的 masked_adj
        return self.current_masked_adj

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu() # 调用 _masked_adj 会更新 self.current_masked_adj
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def forward(self, node_idx, unconstrained=False, mask_features=True, marginalize=False):
        x = self.x.cuda() if self.args.gpu else self.x

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = (
                torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask
            )
            # 在 unconstrained 模式下，也要更新 current_masked_adj 以便后续梯度获取
            self.current_masked_adj = self.masked_adj.clone().detach().requires_grad_(True)
        else:
            self.masked_adj = self._masked_adj() # 这里会调用新的 _masked_adj，并更新 self.current_masked_adj

            if mask_features:
                feat_mask = (
                    torch.sigmoid(self.feat_mask)
                    if self.use_sigmoid
                    else self.feat_mask
                )
                if marginalize:
                    std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                    mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                    z = torch.normal(mean=mean_tensor, std=std_tensor)
                    x = x + z * (1 - feat_mask)
                else:
                    x = x * feat_mask
        
        # 确保模型调用是解包形式：ypred, adj_att = self.model(x, adj)
        # 这里使用 self.current_masked_adj 作为输入
        if self.current_masked_adj.requires_grad: # 只有当它是可求导时才 retain
            self.current_masked_adj.retain_grad()
        if x.requires_grad:
            x.retain_grad()

        # 确保模型调用是解包形式：ypred, adj_att = self.model(x, adj)
        ypred, adj_att = self.model(x, self.current_masked_adj)
        if self.graph_mode:
            res = nn.Softmax(dim=0)(ypred[0])
        else:
            node_pred = ypred[self.graph_idx, node_idx, :]
            res = nn.Softmax(dim=0)(node_pred)
        return res, adj_att

    # def adj_feat_grad(self, node_idx, pred_label_node):
    #     # 这个函数现在只需要返回 self.current_masked_adj.grad，
    #     # 因为 loss.backward() 会在 ExplainModule.explain() 的主循环中被调用，
    #     # 且 self.current_masked_adj 在 forward 中已设置为可求导。
        
    #     # 确保在调用此函数之前，已经执行过一次 forward 和 backward
    #     adj_grad_tensor = self.current_masked_adj.grad
    #     x_grad_tensor = self.x.grad # x 的梯度会正常回传到 self.x.grad

    #     if adj_grad_tensor is None:
    #         raise RuntimeError("masked_adj.grad is None – check requires_grad and leaf status. Was ExplainModule.forward() and loss.backward() called before this?")
    #     if x_grad_tensor is None: # 同样检查 x 的梯度
    #         raise RuntimeError("x.grad is None – check requires_grad and leaf status. Was ExplainModule.forward() and loss.backward() called before this?")

    #     # 取绝对值并取第一张图（batch dim=1）
    #     # adj_grad = torch.abs(adj_grad_tensor)[0] 是原先的，直接返回两个 tensor
    #     return adj_grad_tensor, x_grad_tensor

    def loss(self, pred, pred_label, node_idx, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
            gt_label_node = self.label if self.graph_mode else self.label[0][node_idx]
            logit = pred[gt_label_node]
            pred_loss = -torch.log(logit)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs["size"] * torch.sum(mask)

        # pre_mask_sum = torch.sum(self.feat_mask)
        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        feat_mask_ent = - feat_mask             \
                        * torch.log(feat_mask)  \
                        - (1 - feat_mask)       \
                        * torch.log(1 - feat_mask)

        feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)

        # laplacian
        D = torch.diag(torch.sum(self.current_masked_adj[0], 0)) # 使用 self.current_masked_adj
        m_adj = self.current_masked_adj if self.graph_mode else self.current_masked_adj[self.graph_idx] # 使用 self.current_masked_adj
        L = D - m_adj
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        if self.graph_mode:
            lap_loss = 0
        else:
            lap_loss = (self.coeffs["lap"]
                * (pred_label_t @ L @ pred_label_t)
                / self.adj.numel()
            )

        # grad
        # adj
        # adj_grad, x_grad = self.adj_feat_grad(node_idx, pred_label_node)
        # adj_grad = adj_grad[self.graph_idx]
        # x_grad = x_grad[self.graph_idx]
        # if self.args.gpu:
        #    adj_grad = adj_grad.cuda()
        # grad_loss = self.coeffs['grad'] * -torch.mean(torch.abs(adj_grad) * mask)

        # feat
        # x_grad_sum = torch.sum(x_grad, 1)
        # grad_feat_loss = self.coeffs['featgrad'] * -torch.mean(x_grad_sum * mask)

        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss + feat_mask_ent_loss
        if self.writer is not None:
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
            self.writer.add_scalar("optimization/feat_size_loss", feat_size_loss, epoch)
            self.writer.add_scalar("optimization/mask_ent_loss", mask_ent_loss, epoch)
            self.writer.add_scalar(
                "optimization/feat_mask_ent_loss", feat_mask_ent_loss, epoch # 修正了变量名
            )
            # self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
            self.writer.add_scalar("optimization/lap_loss", lap_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)
        return loss

    def log_mask(self, epoch):
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(4, 3), dpi=400)
        plt.imshow(self.mask.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/mask", tensorboardX.utils.figure_to_image(fig), epoch
        )

        # fig = plt.figure(figsize=(4,3), dpi=400)
        # plt.imshow(self.feat_mask.cpu().detach().numpy()[:,np.newaxis], cmap=plt.get_cmap('BuPu'))
        # cbar = plt.colorbar()
        # cbar.solids.set_edgecolor("face")

        # plt.tight_layout()
        # fig.canvas.draw()
        # self.writer.add_image('mask/feat_mask', tensorboardX.utils.figure_to_image(fig), epoch)
        io_utils.log_matrix(
            self.writer, torch.sigmoid(self.feat_mask), "mask/feat_mask", epoch
        )

        fig = plt.figure(figsize=(4, 3), dpi=400)
        # use [0] to remove the batch dim
        plt.imshow(self.masked_adj[0].cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/adj", tensorboardX.utils.figure_to_image(fig), epoch
        )

        if self.args.mask_bias:
            fig = plt.figure(figsize=(4, 3), dpi=400)
            # use [0] to remove the batch dim
            plt.imshow(self.mask_bias.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
            cbar = plt.colorbar()
            cbar.solids.set_edgecolor("face")

            plt.tight_layout()
            fig.canvas.draw()
            self.writer.add_image(
                "mask/bias", tensorboardX.utils.figure_to_image(fig), epoch
            )

    def _calculate_dynamic_threshold(self, weights_tensor, percentile=90):
        flat_weights = weights_tensor.flatten()
        
        positive_weights = flat_weights[flat_weights > 1e-9] # Using a small epsilon

        if len(positive_weights) == 0:
            return 0.0 
        
        threshold = np.percentile(positive_weights.cpu().detach().numpy(), percentile)
        return float(threshold)

    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        # --- 1) 重新计算 masked_adj 和前向输出 ---
        masked = self._masked_adj()  # 这会返回当前的 masked_adj [1, N, N]
        ypred, _ = self.model(self.x.cuda() if self.args.gpu else self.x,
                              masked)
        # 取出我们关心的那个 logit
        if self.graph_mode:
            logit = F.softmax(ypred[0], dim=0)[pred_label]
        else:
            logit = F.softmax(ypred[self.graph_idx, node_idx], dim=0)[pred_label]

        # --- 2) 计算负 log 概率作为 loss ---
        loss = -torch.log(logit+ 1e-9)

        # --- 3) 用 grad() 显式算梯度，不用 .grad 属性 ---
        # retain_graph=True 如果后面还要继续用这个计算图
        adj_grad_tensor = grad(loss, masked, retain_graph=True)[0]  # shape [1, N, N]
        # x_grad_tensor   = grad(loss, self.x, retain_graph=True)[0]  # shape [1, N, F]
        x_grad_tensor = None

        # --- 4) 取绝对值并对称化，再取 batch 维度 0 ---
        adj_grad = torch.abs(adj_grad_tensor[0])
        adj_grad = (adj_grad + adj_grad.t()) * 0.5

        # ==== 新增：根据 is_small_graph 设定可视化阈值 ====
        # grad_vis_threshold = 0.0 if self.is_small_graph else 0.0003
        if self.is_small_graph:
            grad_vis_threshold = 0.0 # 小图：阈值为0
        else:
            grad_vis_threshold = self._calculate_dynamic_threshold(adj_grad, percentile=85)
            print(f"Dynamically calculated grad_vis_threshold for adj_grad: {grad_vis_threshold:.4f}")

        # ============================================
        # if not np.any(adj_grad > 0.0003):
        #     print("[Warning] no edge passed threshold; skipping log_graph.")
        # --- 5) 可视化 / 保存 ---
        if self.graph_mode:
            G = io_utils.denoise_graph(
                adj_grad.cpu().detach().numpy(),
                node_idx,
                feat=(self.x[0].cpu().detach().numpy() 
                      if isinstance(self.x, torch.Tensor) else None),
                threshold=grad_vis_threshold, # <-- 使用新的梯度可视化阈值
                # threshold=0,#0.0003
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        else:
            G = io_utils.denoise_graph(
                adj_grad.cpu().detach().numpy(),
                node_idx,
                # threshold_num=12,
                feat=self.x[0].cpu().detach().numpy(), # Pass current subgraph features
                threshold=grad_vis_threshold, # <-- 使用新的梯度可视化阈值
                max_component=True
            )
            io_utils.log_graph(
                self.writer, G, name="grad/graph", epoch=epoch, args=self.args
            )
        return adj_grad, x_grad_tensor


    def log_masked_adj(self, node_idx, epoch, name="mask/graph", label=None):
        # use [0] to remove the batch dim
        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        # 如果是小图，阈值为0，否则使用一个默认值（例如0.2）
        # vis_threshold = 0.0 if self.is_small_graph else 0.05
        if self.is_small_graph:
            vis_threshold = 0.0 
        else:
            vis_threshold = self._calculate_dynamic_threshold(self.masked_adj[0], percentile=85)
            print(f"Dynamically calculated vis_threshold for masked_adj: {vis_threshold:.4f}")

        print("Now the threshold is ")
        print(vis_threshold)
        # ============================================
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj,
                node_idx,
                feat=self.x[0].cpu().detach().numpy(),
                threshold=vis_threshold,
                # threshold=0,   threshold_num=20, 0.2
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=False,
                nodecolor="feat",
                epoch=epoch,
                label_node_feat=True,
                edge_vmax=None,
                args=self.args,
            )
        else:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, feat=self.x[0].cpu().detach().numpy(), # Pass current subgraph features
                threshold=vis_threshold, # <-- 使用新的可视化阈值
                max_component=True
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=True,
                nodecolor="label",
                epoch=epoch,
                edge_vmax=None,
                args=self.args,
            )