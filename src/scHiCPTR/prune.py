import numpy as np
import pandas as pd
import igraph
import heapq
from typing import Optional, Union
from scipy.sparse import  csr_matrix


def edge_between_score(
    g: igraph.Graph
) -> list:
    score_list = g.edge_betweenness(directed=True, weights='weight')
    norm = max(score_list)
    score_list = [score_list[i]/norm for i in range(len(score_list))]
    return score_list

def common_neighbor_score(
    g: igraph.Graph
) -> list:
    edges = g.get_edgelist()
    score_list = g.similarity_dice(pairs=edges, loops=True)
    norm = max(score_list)
    score_list = [score_list[i]/norm for i in range(len(score_list))]
    return score_list

def cal_similarity(
    g: igraph.Graph,
    lambda_: float,
) -> csr_matrix:
    between = edge_between_score(g)
    common_neighbor = common_neighbor_score(g)
    score_list = [(lambda_*common_neighbor[i] + (1-lambda_)*between[i]) for i in range(len(between)) ]
    return score_list

def get_edge_index(
    g: igraph.Graph,
    lambda_: float=0.5,
    r: float = 0.5, 
) -> list:
    n = int(g.ecount()*r)
    score = cal_similarity(g, lambda_)
    edges = g.get_edgelist()
    index = [edges[i] for i in (heapq.nsmallest(n, range(len(score)), score.__getitem__))]
    return index

def get_mst_edge_index(
    adj: Union[csr_matrix, np.ndarray],
    mst: Union[csr_matrix, np.ndarray],
    groups: Optional[pd.DataFrame] = None,
) -> list:
    global_index_list = []

    if isinstance(adj, csr_matrix):
        adj = adj.todense()
    if isinstance(mst, csr_matrix):
        mst = mst.todense()

    membership = groups['louvain'].cat.codes.values
    for i in range(len(adj)):
        cluster_i = membership[i]
        for j in range(i, len(adj)):
            cluster_j = membership[j]
            if adj[i,j] == 0 or cluster_i == cluster_j or mst[cluster_i, cluster_j] > 0 or mst[cluster_j, cluster_i] > 0:
                continue
            else:
                global_index_list.append((i,j))        
    return global_index_list

def get_his_edge_index(
    adj: Union[csr_matrix, np.ndarray],
    groups: Optional[pd.DataFrame] = None,
    lambda_: float=0.5,
    r: float = 0.5, 
    mode: Optional[str] = None,  
) -> list:
    global_index_list = []

    if isinstance(adj, csr_matrix):
        adj = adj.todense()

    g = igraph.Graph.Weighted_Adjacency(matrix=adj)
    g.vs['global_index'] = [i for i in range(len(adj))] 
    
    if mode == 'cluster':
        if groups is None:
            raise ValueError('missing key parameter \'groups\'')
        vc = igraph.VertexClustering(
            g, membership=groups['louvain'].cat.codes.values,
        )
        nc = vc.sizes()
        for i in range(len(nc)):
            sg = vc.subgraph(i)
            index = get_edge_index(sg, lambda_=lambda_, r=r)
            index = [(sg.vs['global_index'][index[i][0]], sg.vs['global_index'][index[i][1]]) for i in range(len(index))]
            global_index_list.extend(index)
    else:
        index = get_edge_index(g, lambda_=lambda_, r=r)
        global_index_list = [(g.vs['global_index'][index[i][0]], g.vs['global_index'][index[i][1]]) for i in range(len(index))]

    return global_index_list

def prune_graph(
    adj: Union[csr_matrix, np.ndarray],
    edges: list,
) -> csr_matrix:
    
    if isinstance(adj, csr_matrix):
        adjacency = adj.todense()
    else:
        adjacency = adj.copy()
    for edge in edges:
        # if the edge is the cut edge, ignore it
        if np.sum(adjacency, axis=1)[edge[0]] == adjacency[edge[0], edge[1]] or np.sum(adjacency, axis=1)[edge[1]] == adjacency[edge[1], edge[0]]:
            continue
        # if sum(adjacency[edge[0], ]) == adjacency[edge[0], edge[1]] or sum(sum(adjacency[edge[1], :])) == adjacency[edge[1], edge[0]]:
        #     continue
        adjacency[edge[0], edge[1]] = 0
        adjacency[edge[1], edge[0]] = 0

    return csr_matrix(adjacency)
