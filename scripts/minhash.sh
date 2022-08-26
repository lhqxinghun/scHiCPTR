#!/bin/bash

scHicClusterMinHash -m cellcycle_1M.scool --numberOfHashFunctions 5000 --numberOfClusters 4 --umap_n_components 5 --clusterMethod kmeans -o clusters_minhash_kmeans.txt --createScatterPlot "cluster_minhash.png" --threads 20
