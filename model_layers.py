#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/4/17 20:40
"""
from typing import Tuple
import torch
from torch import nn
import os


class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max out degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, rank:torch.Tensor) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_index: edge_index of graph (adjacency list)
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        x += self.z_in[rank] + self.z_out[rank] # 将每个节点度的排名作为索引，挑选z_in或z_out的每行，形成每个节点的嵌入

        return x


class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.max_path_distance = max_path_distance
        self.b = nn.Parameter(torch.randn(self.max_path_distance))

    def forward(self, x: torch.Tensor, node_path, sparse_mask = None) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param paths: pairwise node paths
        :return: torch.Tensor, spatial Encoding matrix
        """
        spatial_matrix = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device) # (num_nodes, num_nodes)

        if sparse_mask is not None:
            sparse_attention_mask = sparse_mask
            sparse_attention_mask = sparse_attention_mask.to(next(self.parameters()).device)
        else:
            sparse_attention_mask = torch.ones(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        nonzero_indices = torch.nonzero(sparse_mask, as_tuple=False)
        nonzero_index = nonzero_indices.cpu().numpy()

        for index in nonzero_index:
            spatial_matrix[index[0]][index[1]] = self.b[min(node_path[index[0]][index[1]], self.max_path_distance) - 1]

        spatial_matrix = spatial_matrix * sparse_attention_mask

        return spatial_matrix


class EgoEncoding(nn.Module):
    def __init__(self, max_degree: int, node_dim: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.max_degree = max_degree
        self.node_dim = node_dim
        self.c = nn.Parameter(torch.randn(self.max_degree))

    def forward(self, x: torch.Tensor, rank, sparse_mask = None) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param rank: degree of nodes
        :return: torch.Tensor, spatial Encoding matrix
        """

        if sparse_mask is not None:
            sparse_attention_mask = sparse_mask.to(next(self.parameters()).device)
        else:
            sparse_attention_mask = torch.ones(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        rank_clamped = torch.clamp(rank, max=self.max_degree - 1)
        central_matrix = self.c[rank_clamped].unsqueeze(1) * sparse_attention_mask

        return central_matrix


class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self,
                x: torch.Tensor,
                b: torch.Tensor,
                c: torch.Tensor,
                ptr=None,
                sparse_mask = None,
                ) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        batch_mask_neg_inf = torch.full(size=(x.shape[0], x.shape[0]), fill_value=-1e6).to(next(self.parameters()).device)
        batch_mask_zeros = torch.zeros(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        if sparse_mask is not None:
            sparse_attention_mask = sparse_mask
            sparse_attention_mask_zeros = torch.where(sparse_attention_mask == 0, torch.tensor(0), torch.tensor(1)).to(next(self.parameters()).device)
        else:
            sparse_attention_mask_zeros = torch.ones(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        if len(ptr) == 1:
            batch_mask_neg_inf = torch.ones(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)
            batch_mask_zeros += 1
        else:
            # 批图的mask,邻接矩阵以对角阵组合

            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        a = self.compute_a(key, query, ptr)
        a = a + b + c
        a[batch_mask_zeros == 0] = -1e6
        a[sparse_attention_mask_zeros == 0] = -1e6
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros * sparse_attention_mask_zeros # e^(-inf) ——> 0

        x = softmax.mm(value)

        return x

    def compute_a(self, key, query, ptr=None):
        "Query-Key product(normalization)"
        if type(ptr) == type(None):
            a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        else:
            a = torch.zeros((query.shape[0], query.shape[0]), device=key.device)
            for i in range(len(ptr) - 1):
                a[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = query[ptr[i]:ptr[i + 1]].mm(
                    key[ptr[i]:ptr[i + 1]].transpose(0, 1)) / query.size(-1) ** 0.5

        return a


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: torch.Tensor,
                b: torch.Tensor,
                c: torch.Tensor,
                ptr=None,
                sparse_mask = None,
                ) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """

        return self.linear(
            torch.cat([
                attention_head(x, b, c, ptr,sparse_mask) for i, attention_head in enumerate(self.heads)
            ], dim=-1)
        )


class EncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads, max_path_distance):
        """
        :param node_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param num_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads

        self.attention = MultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=num_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)
        self.ff = nn.Linear(node_dim, node_dim)

    def forward(self,
                x: torch.Tensor,
                b: torch.Tensor,
                c: torch.Tensor,
                ptr=None,
                sparse_mask = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """

        x_prime = self.attention(self.ln_1(x), b, c, ptr, sparse_mask) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new