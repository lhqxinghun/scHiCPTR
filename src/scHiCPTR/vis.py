import os
import numpy as np
import pandas as pd
import scanpy as sc
import collections.abc as cabc
import matplotlib.pyplot as plt
from types import MappingProxyType
from typing import Optional, Union, Mapping, Sequence, Any, List
from pathlib import Path
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like, Colormap
from . import scHiCPTR

# visualize trajectory or cluster-level graph
def vis_trajectory(
    tra: scHiCPTR.Trajectory,
    save_path: str=None,
    plot_cluster_graph: bool = False,
    plot_trajectory: bool = True,
):
    """\
    Plot the cluster-level graph or trajectory graph.
    This uses ForceAtlas2 or igraph's layout algorithms for most layouts [Csardi06]_.
    The implementation references https://github.com/theislab/scanpy. 

    Parameters
    ----------
    tra
        Class Trajectory.
    save_path
        The dir for saving figures.
    plot_cluster_graph
        IF 'False', do not plot cluster-level graph.
    plot_trajectory
        IF 'True', plot trajectory graph.
    """
    coords = cluster_graph(tra, plot=plot_cluster_graph)
    df = tra._trajectory['pseudotime'].values
    cm = plt.cm.get_cmap('jet')
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax1 = ax.scatter(coords[:, 0], coords[:, 1], s=8, c=df, alpha=.9, cmap=cm)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(ax1)
    fig.tight_layout()
    if save_path == None:
        save_path = os.getcwd()
    fig_path = save_path
    fig.savefig(fig_path, dpi=350)
    # plt.show()
    return

# plot cluster-level graph
def cluster_graph(
    tra: scHiCPTR.Trajectory,
    plot: bool = True,
) -> np.array:
    adata = tra2adata(tra)
    sc.tl.paga(adata, groups='louvain')
    sc.pl.paga(adata, color='louvain', plot=plot)
    sc.tl.draw_graph(adata, init_pos='paga')
    return adata.obsm['X_draw_graph_fa']

# transfer class Trajectory to class AnnData
def tra2adata(tra: scHiCPTR.Trajectory)-> AnnData:
    adata = AnnData(tra._data)
    adata.uns['neighbors'] = {}
    adata.uns['louvain'] = {}
    adata.obsp['distances'] = tra._distacnes_umap
    adata.obsp['connectivities'] = tra._connectivities_umap
    adata.obs = tra._trajectory
    return adata

