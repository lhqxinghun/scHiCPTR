import numpy as np
from schicluster import *

celllist = np.loadtxt('path to sampleList/SampleList.txt' , dtype=np.str)
mm9dim = [197195432,181748087,159599783,155630120,152537259,149517037,152524553,131738871,124076172,129993255,121843856,121257530,120284312,125194864,103494974,98319150,95272651,90772031,61342430,166650296]
chrom =  [str(i+1) for i in range(19)] + ['X']
chromsize = {chrom[i]:mm9dim[i] for i in range(len(chrom))}

cluster, embedding = hicluster_cpu(celllist, chromsize, nc = 4, res=1000000, pad=3, rp=0.5, prct=20, ncpus=4)