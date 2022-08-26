from numba.cuda import args
import numpy as np
import pandas as pd
import louvain
import igraph
import logging
from . import args, embs, prune, dpt
import scipy as sp
from typing import List, Optional, NamedTuple, Union
from natsort import natsorted
from scipy.sparse import spmatrix, csr_matrix, coo_matrix
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree
import warnings
warnings.filterwarnings("ignore")


def get_indices_distances_from_dense_matrix(D, n_neighbors: int):
    sample_range = np.arange(D.shape[0])[:, None]
    indices = np.argpartition(D, n_neighbors-1, axis=1)[:, :n_neighbors]
    indices = indices[sample_range, np.argsort(D[sample_range, indices])]
    distances = D[sample_range, indices]
    return indices, distances

def get_sparse_matrix_from_indices_distances_numpy(indices, distances, n_obs, n_neighbors):
    n_nonzero = n_obs * n_neighbors
    indptr = np.arange(0, n_nonzero + 1, n_neighbors)
    D = csr_matrix((
        distances.copy().ravel(),  # copy the data, otherwise strange behavior here
        indices.copy().ravel(),
        indptr,
    ), shape=(n_obs, n_obs))
    D.eliminate_zeros()
    return D

def get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors):
    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = coo_matrix((vals, (rows, cols)),
                                      shape=(n_obs, n_obs))
    result.eliminate_zeros()
    return result.tocsr()

def compute_connectivities_umap(
    knn_indices, knn_dists,
    n_obs, n_neighbors, set_op_mix_ratio=1.0,
    local_connectivity=1.0,
):
    """\
    This is from umap.fuzzy_simplicial_set [McInnes18]_.

    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """
    from umap.umap_ import fuzzy_simplicial_set

    X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    distances = get_sparse_matrix_from_indices_distances_umap(
        knn_indices, knn_dists, n_obs, n_neighbors
    )

    return distances, connectivities.tocsr()

def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    # if g.vcount() != adjacency.shape[0]:
    #     logg.warning(
    #         f'The constructed graph has only {g.vcount()} nodes. '
    #         'Your adjacency matrix contained redundant nodes.'
    #     )
    return g

def get_connectivities_tree(connectivities):
    inverse_connectivities = connectivities.copy()
    inverse_connectivities.data = 1./inverse_connectivities.data
    connectivities_tree = minimum_spanning_tree(inverse_connectivities)
    connectivities_tree_indices = [
        connectivities_tree[i].nonzero()[1]
        for i in range(connectivities_tree.shape[0])]
    connectivities_tree = sp.sparse.lil_matrix(connectivities.shape, dtype=float)
    for i, neighbors in enumerate(connectivities_tree_indices):
        if len(neighbors) > 0:
            connectivities_tree[i, neighbors] = connectivities[i, neighbors]
    return connectivities_tree.tocsr()

def get_sparse_from_igraph(graph, weight_attr=None):
    from scipy.sparse import csr_matrix

    edges = graph.get_edgelist()
    if weight_attr is None:
        weights = [1] * len(edges)
    else:
        weights = graph.es[weight_attr]
    if not graph.is_directed():
        edges.extend([(v, u) for u, v in edges])
        weights.extend(weights)
    shape = graph.vcount()
    shape = (shape, shape)
    if len(edges) > 0:
        return csr_matrix((weights, zip(*edges)), shape=shape)
    else:
        return csr_matrix(shape)

class Trajectory:
    """\
    """
    def __init__(self, embedding: np.ndarray):
        self._data: np.ndarray = embedding
        self._distacnes_knn: Union[np.ndarray, csr_matrix, None] = None
        self._distacnes_umap: Union[np.ndarray, csr_matrix, None] = None
        self._trajectory: pd.DataFrame = pd.DataFrame()
        self._connectivities_umap: Union[np.ndarray, csr_matrix, None] = None
        self._connectivities_prune: Union[np.ndarray, csr_matrix, None] = None
        self._connectivities_cluster: Union[np.ndarray, csr_matrix, None] = None
        self._inter_es: Union[np.ndarray, csr_matrix, None] = None
        self._connectivities_tree: Union[np.ndarray, csr_matrix, None] = None
        self._HIS_index: Union[list, None] = None
        self._MST_inex: Union[list, None] = None
    
    def _compute_graph(
        self,
        n_neighbors: int = 15,
        metric: Union[str, None] = 'euclidean'  
    ) -> None:
        """\
        """
        X = self._data
        if n_neighbors > X.shape[0]:
            n_neighbors = 1 + int(0.5*X.shape[0])
        self.n_neighbors = n_neighbors
        # neighbor search
        # use_dense_distances = (metric == 'euclidean' and self._data.shape[0] < 8912) 
        # if use_dense_distances:
        distances = pairwise_distances(X, metric=metric)
        knn_indices, knn_distances = get_indices_distances_from_dense_matrix(distances, n_neighbors)
        self._distacnes_knn = get_sparse_matrix_from_indices_distances_numpy(knn_indices, knn_distances, X.shape[0], n_neighbors)
        self._distacnes_umap, self._connectivities_umap = compute_connectivities_umap(knn_indices, knn_distances, X.shape[0], n_neighbors)
        return

    def _detect_communities(
        self,
        resolution: float = 0.5,
        random_state: Union[None, int] = 0,
        use_weights: bool = False,
        partition_type = None,
        **Kwargs,
    ) -> None:
        if self._connectivities_umap is None:
            logging.error("No KNN graph!")
        adjacency = self._connectivities_umap
        g = get_igraph_from_adjacency(adjacency, directed = True)
        if use_weights:
            weights = np.array(g.es["weight"]).astype(np.float64)
        else:
            weights = None
        # louvain
        if partition_type == None:
            partition_type = louvain.RBConfigurationVertexPartition
        if resolution is not None:
            Kwargs['resolution_parameter'] = resolution
        if use_weights:
            Kwargs['weights'] = weights
        # if version.parse(louvain.__version__) < version.parse('0.7.0'):
        #     louvain.set_rng_seed(random_state)
        part = louvain.find_partition(
            g, partition_type,
            **Kwargs,
        )
        groups = np.array(part.membership)
        self.louvain_resolution = resolution
        self.louvain_random_state = random_state
        self._trajectory['louvain'] = pd.Categorical(
            values = groups.astype('U'),
            categories = natsorted(map(str, np.unique(groups)))
        )
    
    # MST on PAGA-connectivities-based cluster graph
    def _abstract_cluster_tree(
        self,
    ) -> None:
        ones = self._distacnes_umap
        ones.data = np.ones(len(ones.data))
        g = get_igraph_from_adjacency(ones, directed=True)
        vc = igraph.VertexClustering(
            g,
            membership=self._trajectory['louvain'].cat.codes.values,
        )
        ns = vc.sizes()
        n = sum(ns)
        es_inner_cluster = [vc.subgraph(i).ecount() for i in range(len(ns))]
        cg = vc.cluster_graph(combine_edges='sum')
        inter_es = get_sparse_from_igraph(cg, weight_attr='weight')
        es = np.array(es_inner_cluster) + inter_es.sum(axis=1).A1
        inter_es = inter_es + inter_es.T  # \epsilon_i + \epsilon_j
        connectivities = inter_es.copy()
        expected_n_edges = inter_es.copy()
        inter_es = inter_es.tocoo()
        for i, j, v in zip(inter_es.row, inter_es.col, inter_es.data):
            expected_random_null = (es[i]*ns[j] + es[j]*ns[i])/(n - 1)
            if expected_random_null != 0:
                scaled_value = v / expected_random_null
            else:
                scaled_value = 1
            if scaled_value > 1:
                scaled_value = 1
            connectivities[i, j] = scaled_value
            expected_n_edges[i, j] = expected_random_null 

        self.ns = ns
        self._inter_es = inter_es.tocsr
        self._connectivities_cluster = connectivities
        self._connectivities_tree = get_connectivities_tree(connectivities)   

    def _prune_graph(
        self,
        lambda_: float = 0.5,
        threshold: float = 0.5,
        louvain_res: float = 0.5,
        **Kwargs,
    ):
        self._prune_HIS(lambda_=lambda_, threshold=1-threshold)
        self._prune_MST(res=louvain_res, **Kwargs)
        his_index = self._HIS_index
        mst_index = self._MST_inex
        edges_index = list(set(his_index+mst_index))
        self._connectivities_prune = prune.prune_graph(self._connectivities_umap, edges=edges_index)
        return
    
    def _prune_HIS(
        self,
        lambda_: float = 0.5,
        threshold: float = 0.5,
    ) :
        if self._connectivities_umap is None:
            logging.error("No KNN graph!")
        adjacency = self._connectivities_umap
        self._HIS_index = prune.get_his_edge_index(adj=adjacency, lambda_=lambda_, r=threshold)   
        return 

    def _prune_MST(
        self,
        res: float = 0.5,
        **Kwargs,
    ) -> list:
        if self._connectivities_umap is None:
            logging.error("No KNN graph!")
        self._detect_communities(resolution=res, **Kwargs)
        self._abstract_cluster_tree()
        self._MST_inex = prune.get_mst_edge_index(self._connectivities_umap, self._connectivities_tree, self._trajectory)
        return

def scHiCPTR(
    data: np.ndarray,
    imp_method = "convolution_randomwalk",
    chromsize: dict = None,
    resolution: int = 1000000,
    imp_args = None,
    emb_method = "schicptr",
    emb_args = None,
    embedding: np.ndarray = None,
    start: int = 0,
    knn_args: dict = None,
    prune_args: dict = None,
) -> Trajectory:
    
    """Pipeline of imputation, embedding and pseudotime inference :
    data:       np.ndarray, the path to the cell files. 
    imp_method: str, the imputation method, it can be "convolution_randomwalk" or None(means no imputation)
    chromsize:  dict, the size of each chromosome used.
    resolution: int, bin size
    imp_args:   dict, the parameter for imputation if the "imputation_method" is not None, including 
                pad (int) for convolution, 
                rp (float) for random walk,
                prct(float) for binaryzation,
                ncpus(int) for the multiprocessing.
                If it is set to None by default, they will be 3,0.5,20 and 10, respectively.
    emb_method: str, the embedding method,it can be "schicptr", "PCA" or "PPCA". 
                "schicptr" :combing PCA and UMAP i.e. the pipeline proposed by scHiCPTR
                "PCA" : using PCA only to reduce dimension, 
                "PPCA" : replacing the imputation PCA with probabilistic PCA. (imp_method should be set to None)
    emb_args:   dict, the parameters for embedding, including
                pca_ncomponent (int),
                umap_neighbour (int),
                umap_mindist (float),
                umap_ncomponent (int),
                umap_metric (str),
                ppca_sigma (float)
                If it is set to None by default, they will be 120, 50, 0.5, 20, 'euclidean' and 50, respectively.
    embedding:  np.ndarry, with 2 dimensions, the 1st shape is the number of cells and the 2nd shape is the
                dimension of embeddings. the "data" will be ignored if this parameter is not None
    start:      int, the root cell for trajectory inference
    knn_args:   dict, the parameters for knn graph construction, including
                n_neighbors (int)
                metric (str)
                If it is set to None, they will be 100 and 'euclidean', respectively.
    prune_args: dict, the parameters for graph pruning, including
                lambda_ (float)
                threshold (float)
                louvain_res (float)
                If it is set to None, they will be 0.5, 0.7 and 0.5, respectively.
    """

    if data is None and embedding is None:
        logging.error("Missing key parameter '\embedding'\ ")
    
    if (chromsize is None) and (embedding is None):
        logging.warning("Using default chromsize parameters!")
        chromsize = args.chromsize
    
    if knn_args is None:
        logging.warning("Using default knn parameters!")
        knn_args = args.knn_args

    if prune_args is None:
        logging.warning("Using default pruning parameters!")
        prune_args = args.prune_args
    
    if embedding is None:
        embedding = embs.impute_embedding(data, chromsize=chromsize, res=resolution,imp_method=imp_method,
                                        imp_args=imp_args, emb_method=emb_method, emb_args=emb_args)
    tra = Trajectory(embedding)
    tra._compute_graph(**knn_args)
    tra._prune_graph(**prune_args)
    tra._trajectory['pseudotime'] = dpt.compute_dpt(tra._connectivities_prune, start=start)
    return tra
