import os
import numpy as np
import pandas as pd

MCM = pd.read_csv("../DATA/Hi-Cmaps/MCM.txt", delim_whitespace=True, index_col=0)
CDD = pd.read_csv("../DATA/Hi-Cmaps/CDD.txt", delim_whitespace=True, index_col=0)
MCM_CDD = pd.concat([MCM, CDD], axis=1)

#get the 1171 cells for analysis
path = "/home/wu/BioinfoCode/wu_Hi-C/data/cellcycle/raw/1M/chrMat"
cell_nm = os.listdir(path)

data = MCM_CDD.loc[cell_nm].sort_index()
np.savetxt("./data/MCM_CDD.txt", data.values)
np.savetxt("./data/MCM.txt", MCM.loc[cell_nm].sort_index().values)
np.savetxt("./data/CDD.txt", CDD.loc[cell_nm].sort_index().values)

data = np.loadtxt("./data/MCM_CDD.txt")
reslut = CIRCLET_CORE.CIRCLET(data, s=270)
df = pd.DataFrame({'dpt_pseudotime': reslut['Trajectory'], 'cell_cycle_stages': ['G1']*280+['ES']*303+['MS']*262+['G2']*326})
stages = ['G1', 'ES', 'MS', 'G2']
eva.evaluation_ranks(stages, df, 'cellcycle', key='dpt_pseudotime')
eva.kendall_rank_coefficient(predict_order = df, dataset='cellcycle')
