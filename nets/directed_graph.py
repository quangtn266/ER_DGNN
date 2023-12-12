#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 19:49:17 2019

@author: quangtn
"""

from typing import Tuple, List
from collections import defaultdict

import numpy as np
from scipy import special


from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# For NTU RGB+D, assume node 21 (centre of chest)
# is the "centre of gravity" mentioned in the paper

#num_nodes = 51
epsilon = 1e-6

# Directed edges: (source, target), see
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf
# for node IDs, and reduce index to 0-based
directed_edges = [(i-1, j-1) for i, j in [
            (0,1), (1,2),(2,3),(3,4),  (9,8), (8,7),(7,6), (6,5),
            (43,44),(44,45),(45,46),
            (10,11),(11,12),(12,13), (16,13), (14,15),(15,16),(17,16),(18,17),
            (19,20),(20,21),(21,22), (19,24),(24,23),(23,22),
            (28,27),(27,26), (26,25),  (28,29),(29,30),(30,25), (46,47),
            (31,32),(32,33),(33,34), (35,34), (36,35), (37,36),
            (42,41),(42,31),(38,37),
            (41,40), (49,45), (38,39), (39,40),(4,5),
            (45,49),(43,50),(50,49),(49,48),(48,47)    # Add self loop for Node 21 (the centre) to avoid singular matrices
]]


# NOTE: for now, let's not add self loops since the paper didn't mention this
# self_loops = [(i, i) for i in range(num_nodes)]


def build_digraph_adj_list(edges: List[Tuple]) -> np.ndarray:
    graph = defaultdict(list)
    for source, target in edges:
        graph[source].append(target)
    return graph


def normalize_incidence_matrix(im: np.ndarray, full_im: np.ndarray, flag) -> np.ndarray:
    # NOTE:
    # 1. The paper assumes that the Incidence matrix is square,
    #    so that the normalized form A @ (D ** -1) is viable.
    #    However, if the incidence matrix is non-square, then
    #    the above normalization won't work.
    #    For now, move the term (D ** -1) to the front
    # 2. It's not too clear whether the degree matrix of the FULL incidence matrix
    #    should be calculated, or just the target/source IMs.
    #    However, target/source IMs are SINGULAR matrices since not all nodes
    #    have incoming/outgoing edges, but the full IM as described by the paper
    #    is also singular, since ±1 is used for target/source nodes.
    #    For now, we'll stick with adding target/source IMs.
    degree_mat = full_im.sum(-1) * np.eye(len(full_im))
    # Since all nodes should have at least some edge, degree matrix is invertible
    if flag == 1:
        inv_degree_mat = np.linalg.inv(degree_mat)
        return (inv_degree_mat @ im) + epsilon
    return im + epsilon


def build_digraph_incidence_matrix(num_nodes: int, edges: List[Tuple], flag) -> np.ndarray:
    # NOTE: For now, we won't consider all possible edges
    # max_edges = int(special.comb(num_nodes, 2))
    max_edges = len(edges)
    source_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    target_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    for edge_id, (source_node, target_node) in enumerate(edges):
        source_graph[source_node, edge_id] = 1.
        target_graph[target_node, edge_id] = 1.
    full_graph = source_graph + target_graph
    source_graph = normalize_incidence_matrix(source_graph, full_graph,flag)
    target_graph = normalize_incidence_matrix(target_graph, full_graph,flag)
    return source_graph, target_graph


def build_digraph_adj_matrix(num_nodes, edges: List[Tuple]) -> np.ndarray:
    graph = np.zeros((num_nodes, num_nodes), dtype='float32')
    for edge in edges:
        graph[edge] = 1
    graph = normalize_incidence_matrix(graph,graph)
    return graph

def PCA_(x):

    standardizedData = StandardScaler().fit_transform(x)

    pca = PCA(51)
        
    principalComponents = pca.fit_transform(X = standardizedData)
    
    return principalComponents

def Graph(num_nodes):
        num_nodes = num_nodes
        edges = directed_edges

        # Incidence matrices
        source_m, target_m = build_digraph_incidence_matrix(num_nodes, edges,1)
        
        
        return source_m, target_m

# TODO:
# Check whether self loop should be added inside the graph
# Check incidence matrix size


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    num_nodes=51
    source_M,target_M = Graph(num_nodes)

    plt.imshow(source_M, cmap='gray')
    plt.show()
    plt.imshow(target_M, cmap='gray')
    plt.show()

