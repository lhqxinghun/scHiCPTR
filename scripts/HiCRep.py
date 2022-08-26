import numpy as np
import pandas as pd
from scHiCTools import scHiCs,scatter
import matplotlib.pylab as plt

###load data
cellList =  [('path to the sparse cell files/HiCToolInput_sparse/cell_' + str(i+1)) for i in range(1171)] 
label = list(np.array(['G1']*280+['ES']*303+['MS']*262+['G2']*326))
loaded_data = scHiCs(cellList,
                                                reference_genome='mm9',
                                                resolution=1000000,
                                                #sparse=False,
                                                #keep_n_strata=10,
                                                format='shortest_score',
                                                adjust_resolution=False,
                                                #header=0,
                                                chromosomes='except Y',
                                                #store_full_map=True,
                                                operations=['convolution, logarithm'],
                                                log_base=1,
                                                kernel_shape = 3
)

###embedding
embs, disMat = loaded_data.learn_embedding(
    dim=2,
    similarity_method='HiCRep',
    embedding_method='MDS',
    n_strata=None, aggregation='median',
    return_distance=True
)
scatter(embs, dimension="2D", point_size=3,sty='default', label=label)
plt.show()


###calculate angle
mean = np.mean(embs, axis=0)
angle = np.array([math.atan2(embs[i][1]-mean[1],embs[i][0]-mean[0]) for i in range(1171)])
np.savetxt("path to angleschicTools_cellcycle_result_500k_old.txt", angle)
