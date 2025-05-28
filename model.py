from typing import Union
import torch
from torch import nn
from torch_geometric.data import Data
from model_layers import EncoderLayer, SpatialEncoding, CentralityEncoding, EgoEncoding
from utils import compute_similarity

class Graphormer(nn.Module):
    def __init__(self, args, count, max_path, ptr):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param input_edge_dim: input dimension of edge features
        :param edge_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max out degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()
        self.device = torch.device('cpu' if args.gpu_index < 0 else 'cuda:{}'.format(args.gpu_index))
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim

        self.max_path_distance = max_path
        self.degree_count = count
        self.ptr = ptr

        self.node_in_lin = nn.Linear(self.input_dim, self.hidden_dim)

        self.edge_in_lin = nn.Linear(self.input_dim, self.hidden_dim)

        self.centrality_encoding_large = CentralityEncoding(
            max_in_degree=self.degree_count,
            max_out_degree=self.degree_count,
            node_dim=self.hidden_dim
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=self.max_path_distance,
        )

        self.centrality_encoding = EgoEncoding(
            max_degree=self.degree_count,
            node_dim=self.hidden_dim
        )

        self.layers = nn.ModuleList([
            EncoderLayer(
                node_dim=self.hidden_dim,
                edge_dim=self.hidden_dim,
                num_heads=self.num_heads,
                max_path_distance=self.max_path_distance) for _ in range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(self.hidden_dim, self.input_dim)

        self.node_out_lin2 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, data: Union[Data],node_paths, sparse_mask,attention_limit) -> {torch.Tensor, torch.Tensor}:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """

        x = data.x.float()

        rank = data.rank
        ptr = data.ptr

        x = self.node_in_lin(x)

        x = self.centrality_encoding_large(x, rank)
        b = self.spatial_encoding(x, node_paths, sparse_mask)
        c = self.centrality_encoding(x, rank, sparse_mask)

        for layer in self.layers:
            x = layer(x, b, c, ptr, sparse_mask)

        x = self.node_out_lin(x)

        featureMat = x.clone().detach()
        cos_similar_mat = compute_similarity(featureMat, attention_limit).to(self.device)

        x = self.node_out_lin2(x)

        x = torch.sigmoid(x)

        return x, featureMat, cos_similar_mat

    def loss(self, score, adj, gamma: float = 1, mean: bool = True):
        adj = adj.to(self.device)
        score = score.to(self.device)
        tmp = 1 + torch.mul(score.unsqueeze(1), adj.to(self.device))
        tmp = torch.prod(tmp.pow(-1), dim=0)
        loss1 = tmp.mean() if mean else tmp.sum()
        loss2 = score.mean() if mean else score.sum()

        return loss1 + gamma * loss2
