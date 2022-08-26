import os
import sys
import time
import logging
import numpy as np
from . import args
import platform
import anndata as ad
# import torch
# import torch.nn.functional as F
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from typing import Optional
from pathlib import Path
from sklearn.decomposition import PCA
from .ppca import PPCA

if platform.system().lower() == 'windows':
    import umap.umap_ as umap
if platform.system().lower() == 'linux':
    import umap

def neighbor_ave_cpu(A, pad):
	if pad==0:
		return A
	ngene, _ = A.shape
	ll = pad * 2 + 1
	B, C, D, E = [np.zeros((ngene + ll, ngene + ll)) for i in range(4)]
	B[(pad + 1):(pad + ngene + 1), (pad + 1):(pad + ngene + 1)] = A[:]
	F = B.cumsum(axis = 0).cumsum(axis = 1)
	C[ll :, ll:] = F[:-ll, :-ll]
	D[ll:, :] = F[:-ll, :]
	E[:, ll:] = F[:, :-ll]
	return (np.around(F + C - D - E, decimals=8)[ll:, ll:] / float(ll * ll))

def random_walk_cpu(A, rp):
	ngene, _ = A.shape
	A = A - np.diag(np.diag(A))
	A = A + np.diag(np.sum(A, axis=0) == 0)
	P = np.divide(A, np.sum(A, axis = 0))
	Q = np.eye(ngene)
	I = np.eye(ngene)
	for i in range(30):
		Q_new = (1 - rp) * I + rp * np.dot(Q, P)
		delta = np.linalg.norm(Q - Q_new)
		Q = Q_new.copy()
		if delta < 1e-6:
			break
	return Q

def impute_cpu(args):
	cell, c, ngene, pad, rp = args
	D = np.loadtxt(cell + '_chr' + c + '.txt')
	A = csr_matrix((D[:, 2], (D[:, 0], D[:, 1])), shape = (ngene, ngene)).toarray()
	A = np.log2(A + A.T + 1)
	A = neighbor_ave_cpu(A, pad)
	if rp==-1:
		Q = A[:]
	else:
		Q = random_walk_cpu(A, rp)
	return [cell, Q.reshape(ngene*ngene)]

def impute_embedding(
    data: np.ndarray,
    chromsize: dict,
    res: int = 1000000,
    imp_method: str = None,
    imp_args = None,
    emb_method = None,
    emb_args = None,
):
    """impute and embed scHi-C data:
    data:np.ndarray, the path to the cell files. 
    chromsize: dict, the size of each chromosome used.
    res: int, bin size
    imp_method: str, the imputation method, it can be "convolution_randomwalk" or None(means no imputation)
    imp_args:   dict, the parameter for imputation if the "imputation_method" is not None, including 
                pad(int) for convolution, 
                rp(float) for random walk,
                prct(float) for binaryzation and ncpus(int) for the multiprocessing.
    emb_method: str, the embedding method,it can be "schicptr", "PCA" or "PPCA". 
                "schicptr" :combing PCA and UMAP i.e. the pipeline proposed by scHiCPTR
                "PCA" : using PCA only to reduce dimension, 
                "PPCA" : replacing the imputation PCA with probabilistic PCA. (imp_method should be set to None)
    emb_args: dict, the parameters for embedding, including:
                pca_ncomponent(int)
                umap_neighbour(int)
                umap_mindist(float)
                umap_ncomponent(int)
                umap_metric(str)
                ppca_sigma(float)
    """
    if imp_method not in args.imp_method_list:
        print("Error: Invalid method for imputation, use \"convolution_randomwalk\" or None")
        sys.exit(-1)

    if emb_method not in args.emb_method_list:
        print("Error: Invalid method for embedding, use one of thses:\"PCA\", \"PPCA\", \"schicptr\"")
        sys.exit(-1)

    if data is None:
        logging.error("Missing key parameter '\data'\!")

    if chromsize is None:
        logging.error("Missing key parameter '\chromsize'\!")

    if imp_method is None:
        logging.warning("No imputation is used") #ignore the imp_args
        pad, rp, prct = 1, -1, -1
    else:
        print("Imputing with", imp_method)
        if imp_args is None:
            imp_args = args.imp_args
            logging.warning("Using default imputaion parameters")
        pad, rp, prct = imp_args["pad"], imp_args["rp"], imp_args["prct"]
    ncpus = imp_args["ncpus"]


    if emb_args is None:
        emb_args = args.emb_args
        logging.warning("Using default embedding parameters")

    matrix = []
    for i, c in enumerate(chromsize):
        ngene = int(chromsize[c] / res) + 1
        start_time = time.time()
        paras = [[cell, c, ngene, pad, rp] for cell in data]
        p = Pool(ncpus)
        result = p.map(impute_cpu, paras)
        p.close()
        index = {x[0]: j for j, x in enumerate(result)}
        Q_concat = np.array([result[index[x]][1] for x in data])
        if prct > -1:
            thres = np.percentile(Q_concat, 100 - prct, axis=1)
            Q_concat = (Q_concat > thres[:, None])
        end_time = time.time()
        print('chromosome', c, ' load and imputation completed,', 'take', end_time - start_time, 'seconds')
        ndim = int(min(Q_concat.shape) * 0.2) - 1
        if emb_method == "PPCA":
            ## PPCA use PPCA to reduce the dimension firstly instead of PCA.
            ppca_obj = PPCA(q = ndim, sigma = emb_args['ppca_sigma'])
            ppca_obj.fit(Q_concat.T, em = False)
            R_reduce = ppca_obj.transform().T
        else:
            pca = PCA(n_components=ndim)
            R_reduce = pca.fit_transform(Q_concat)
        matrix.append(R_reduce)
        #print(c)
    matrix = np.concatenate(matrix, axis=1)

    print("Embedding with", emb_method)
    pca = PCA(n_components=min(matrix.shape) - 1)
    data = pca.fit_transform(matrix)
    data = data[:, 0:emb_args['pca_ncomponent']]
    if emb_method == "schicptr":
        fit = umap.UMAP(n_neighbors=emb_args['umap_neighbour'], 
                        min_dist=emb_args['umap_mindist'],
                        n_components=emb_args['umap_ncomponent'],
                        metric=emb_args['umap_metric'])
        mapper = fit.fit_transform(data) 
    else:
        mapper = data
    print("Embedding completed")
    return mapper
