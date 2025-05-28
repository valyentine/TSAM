import networkx as nx
import igraph
import pickle
from typing import Tuple, Dict, List, Any
import numpy
import numpy as np
import torch
import os
import random
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


def pickle_read(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def pickle_save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def dijkstra_shortest_path(G, source):
    return nx.single_source_dijkstra_path_length(G, source)


def all_pairs_dijkstra_shortest_path(G):
    return {n: dijkstra_shortest_path(G, n) for n in G}


def shortest_path_dijkstra_distance(data: Data):
    G = to_networkx(data)
    node_paths = all_pairs_dijkstra_shortest_path(G)
    return node_paths


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, logger, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.countering = False
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.logging = logger

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.countering = False
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score <= self.best_score + self.delta:
            self.countering = True
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.countering = False
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        """Saves model when validation loss decrease."""
        if self.verbose:
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print(f'Validation decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), os.path.join(self.path, f"checkpoint_{epoch}.pt"))
        self.val_loss_min = val_loss


def LoadGraphData(root):
    ig_g = igraph.Graph().Read_Edgelist(root, directed=False)

    return ig_g


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def compute_similarity(feature_vec: torch.Tensor, attention_limit) -> torch.Tensor:
    """
        :param feature_vec: node feature matrix
        :param attention_limit: edge feature matrix
        :return: torch.Tensor, Dynamic Attention matrix
    """
    featureMat_norm = feature_vec / feature_vec.norm(dim=1, keepdim=True)

    cos_similar_mat = torch.matmul(featureMat_norm, featureMat_norm.T)
    cos_similar_mat = cos_similar_mat - torch.diag(cos_similar_mat.diag())
    for idx in range(cos_similar_mat.size(0)):
        _, indices = torch.topk(cos_similar_mat[idx], attention_limit[idx])
        cos_similar_mat[idx, indices] = 2

    cos_similar_mat = (cos_similar_mat == 2).int()

    return cos_similar_mat


def get_node_path(file_path, data):
    root = os.path.dirname(file_path)
    num_nodes = data.x.shape[0]

    if not os.path.exists(root):
        os.makedirs(root)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            node_path = pickle.load(f)
    else:
        node_paths = shortest_path_dijkstra_distance(data)
        node_path = numpy.zeros((num_nodes, num_nodes), dtype=int)
        for src in node_paths:
            for dst in node_paths[src]:
                node_path[src, dst] = node_paths[src][dst]
        node_path = node_path + node_path.T

        with open(file_path, 'wb') as f:
            pickle.dump(node_path, f)

    return node_path
