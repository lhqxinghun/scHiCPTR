# scHiCPTR

## 1. Introduction
scHiCPTR is an unsupervised pseudotime inference pipeline through dual graph refinement for single cell Hi-C data. It reconciles pseudotime inference in the case of circular and bifurcating topology.

## 2. Installation & example

**2.1 OS**
- ubuntu 18.04

**2.2 Required Python Packages**

Make sure all the packages listed in the *requirements.txt* are installed.

- Python>=3.6
- numpy>=1.18.0
- pandas
- igraph>=0.9.8
- matplotlib>=3.3.2



**2.3 Install from Github**

First, download the folder *scHiCPTR*.

```
$ git clone https://github.com/lhqxinghun/scHiCPTR
```
Second, install the package with following command:

```
$ cd scHiCPTR
$ pip install .
```

**2.4 Or install from PyPI**

```
$ pip install scHiCPTR
```

**2.5 run example **

```
$ cd scHiCPTR/examples
$ python cellcycle.py
```

## 3. Usage
### 3.1 Input parameters

- **data**: np.ndarray, it lists the paths of contact matrices for each cell, e.g.['path to cell_1','path_to_cell_2',...].
- **imp_method**:str, the imputation method, it can be "convolution_randomwalk" or None(means no imputation)
- **chromsize**:dict, it specifies the size for each chromosome and according to the reference genome, e.g. for mm9, it shall be {'1':197195432,'2':181748087,...'X':166650296}. Elements in it correspond to the contact matrics of chromosomes for a cell. (If it is set to None, the default chromsome size will be defined by mm9)
- **resolution**:int, the binsize of the contact matrices.(default 1000000)
- **imp_args**: dict, the parameter for imputation if the "imputation_method" is not None, including pad (int) for convolution, 
                rp (float) for random walk,
                prct(float) for binaryzation,
                ncpus(int) for the multiprocessing.
    
    If it is set to None by default, they will be 3,0.5,20 and 10, respectively.

- **emb_method**: str, the embedding method,it can be "schicptr", "PCA" or "PPCA".

    *"schicptr"*：combing PCA and UMAP i.e. the pipeline in manucript of scHiCPTR
     
    *"PCA"*: using PCA only to reduce dimension
     
    *"PPCA"* : replacing the imputation PCA with probabilistic PCA. (imp_method should be set to None)

- **emb_args**:   dict, the parameters for embedding, including
                pca_ncomponent (int),
                umap_neighbour (int),
                umap_mindist (float),
                umap_ncomponent (int),
                umap_metric (str),
                ppca_sigma (float)
            
    If it is set to None by default, they will be 120, 100, 0.5, 20, 'euclidean' and 50, respectively.


- **embedding**:np.ndarray with 2 dimensions, the 1st shape is the number of cells and the 2nd shape is the dimension of embeddings. Note that **data** will be ignored if this parameter is valid.
- **start**:int, the root cell for trajectory inference. (default 0)
- **knn_args**:dict, including the 'n_neighbors' and 'metric' for knn graph construction. (default 100 and 'euclidean', respectively)
- **prune_args**:dict, including the 'lambda_'for mixture of HIS, 'threshold' for purning, and 'louvain_res' for louvain clustering. (default 0.5, 0.7 and 0.5, respectively)

### 3.2 Input with contact matrices
**Data preparation** Sparse matrices: For each cell, each intra-chromosomal contact matrice is stored in a .txt file. The file contain three columns separated by tab, i.e. sparse format of the contact matrix . The name of the file need to be in the format of 'cell_{id}_chr{nm}.txt'. {id} is the number of cells (like 1, 2, 3,...) and {nm} is the number of chromosome (like 1, 2, ..., X).

**Parameters setting** To perform scHiCPTR on contact matrices, the **embedding** should be set to None, the **data** indicates the path for each cell, and the **chromsize** and **resolution** should be specified accordingly.

Note that we have provided an example of input matrices in /data/matrices/1M, and the example code for this can be found at /example/cellcycle.py

### 3.3 Input with embeded vectors
the **embedding** should be an np.ndarray with 2 dimensions, the 1st dimension is the number of cells and the 2nd dimension is the dimension of embeddings. the **data**,**chromsize**, and **resolution** will be ignored.

we have also provided an example of input embeddings in /data/embeds, and the example code for this can be found at /example/cellcycle.py

### 3.4 Alternative operations in the software
The scHiCPTR allows users to replace the imputation and embedding components if they need to. Take two cases here as examples 
#### Using PC space instead of UMAP
For this case, just set the **emb_method** to "PCA", and adjust the "pca_component" in **emb_args** to target dimension

#### Using probabilistic PCA for preprocessing
Since probabilistic PCA has an ability to handle missing values, users can replace the imputation and pre-dimensionalization step with it. This can be down by setting the **imp_method** to None and **emb_method** to "PPCA".
