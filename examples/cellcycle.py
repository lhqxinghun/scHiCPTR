import numpy as np
from scHiCPTR import scHiCPTR, eva, vis



####### example code for taking contact matrices as input #######
# if __name__ == '__main__':

# ##For the given small dataset "scHiCPTR/data/matrices/1M.zip", please unzip it before use
#     ## 1. Prepare dataset
#     rawpath = "../data/matrices/1M"
#     network = np.array([rawpath + "/cell_" + str(i) for i in list(range(140, 165))+list(range(440, 465))+list(range(700, 725))+list(range(1020, 1045))], dtype=np.str)

#     mm9dim = [197195432,181748087,159599783,155630120,152537259,149517037,152524553,131738871,124076172,129993255,121843856,121257530,120284312,125194864,103494974,98319150,95272651,90772031,61342430,166650296]
#     chrom =  [str(i+1) for i in range(19)] + ['X']
#     chromsize = {chrom[i]:mm9dim[i] for i in range(len(chrom))}

#     ## 2. Imputation and embedding args for the small dataset
#     imp_args = None # e.g. default
#     emb_args = {"pca_ncomponent":50, "umap_neighbour":15, "umap_mindist":0.5, "umap_ncomponent":5, "umap_metric":"euclidean"}
#     ## 3. KNN args
#     knn_args = {"n_neighbors": 20, "metric":'euclidean'}
#     ## 4. Pruning args
#     prune_args = {"lambda_": 0.5, "threshold": 0.7, "louvain_res": 0.5}

#     ## 5. Calculate pseudotime
#     trajectory = scHiCPTR.scHiCPTR(data=network, chromsize=chromsize, resolution=1000000, emb_method="schicptr", 
#                                     emb_args=emb_args, embedding=None, knn_args=knn_args, prune_args=prune_args, start=0)
#     result = trajectory._trajectory 
#     result.to_csv('../data/results/pseudotime_test.csv')




###### example code for taking embedding vectors as input #######
if __name__ == '__main__':
    
    ## 1. Load embedding vector
    embeds = np.loadtxt('../data/embeds/cellcycle_250k.txt' , dtype=np.str)
    
    ## 2. KNN args
    knn_args = {"n_neighbors": 100, "metric":'euclidean'}

    ## 3. Pruning args
    prune_args = {"lambda_": 0.5, "threshold": 0.7, "louvain_res": 0.5}

    ## 4. Calculate pseudotime.
    trajectory = scHiCPTR.scHiCPTR(data=None, embedding=embeds, knn_args=knn_args, prune_args=prune_args, start=270)
    result = trajectory._trajectory 
    result.to_csv('../data/results/pseudotime_cellcycle.csv')

    ## EValuate results with AUC
    true_phases = ['G1', 'ES', 'MS', 'G2']    # the true order of the pahses, i.e.  between which successive phases, the AUC will be calculated
    true_cell_phases =  ['G1']*280+['ES']*303+['MS']*262+['G2']*326 #the ground truth phases for each cell
    eva.evaluation_ranks_AUC(true_phases=true_phases, true_cell_phases=true_cell_phases, predict_order=result, key='pseudotime')

     ## EValuate results with Kendall rank coefficient
    true_order = [0]*280+[1]*303+[2]*262+[3]*326 #the true order for the cells, cells in the same phase are regarded as the same order
    eva.kendall_rank_coefficient(true_order=true_order,predict_order=result, key='pseudotime')

    ## Save the visualized pseudotime as pseudotime.jpg to the given path
    vis.vis_trajectory(trajectory, save_path='../data/results/pseudotime_cellcycle.jpg')
