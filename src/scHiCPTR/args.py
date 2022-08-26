
mm9dim = [197195432,181748087,159599783,155630120,152537259,149517037,152524553,131738871,124076172,129993255,121843856,121257530,120284312,125194864,103494974,98319150,95272651,90772031,61342430,166650296]
chrom =  [str(i+1) for i in range(19)] + ['X']
chromsize = {chrom[i]:mm9dim[i] for i in range(len(chrom))}

## default args
## feature args
imp_args = {"pad": 3, "rp": 0.5, "prct": 20, "ncpus": 10}
emb_args = {"pca_ncomponent":120,
                "umap_neighbour":100,
                "umap_mindist":0.5,
                "umap_ncomponent":20,
                "umap_metric":"euclidean",
                "ppca_sigma":50}

## KNN args
knn_args = {"n_neighbors": 100, "metric":'euclidean'}

## pruning args
prune_args = {"lambda_": 0.5, "threshold": 0.7, "louvain_res": 0.5}

imp_method_list = ["convolution_randomwalk", None]

emb_method_list = ["PCA", "PPCA", "schicptr"]