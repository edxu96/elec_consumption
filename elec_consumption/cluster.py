"""Functions for clustering analysis."""
import itertools
from typing import List, Optional

from loguru import logger
import networkx as nx
from networkx.utils.misc import pairwise
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns

__all__ = ['init_graph_weight']


def mst_corr(df: DataFrame) -> nx.Graph:
    """Init complete graph with correlations and exec min span tree.

    Args:
        df: original longitudinal in wide format.

    Returns:
        Min spanning tree based on pearson correlation coefficients.
    """
    corr_long = df.corr().stack().to_frame()
    corr_long.index.names = ['source', 'target']
    corr_long.columns = ['corr']
    corr_long.reset_index(inplace=True)

    # Remove redundant correlation values.
    corr_long = corr_long[corr_long['target'] > corr_long['source']]

    graph = nx.from_pandas_edgelist(corr_long, edge_attr='corr')

    tree_generator = nx.maximum_spanning_edges(graph, weight='corr')
    tree = nx.from_edgelist(tree_generator)
    return tree


def keep_cluster(tree: nx.Graph, n_clusters: int) -> DataFrame:
    """Remove edges with low correlations to have clusters.

    Warning:
        The operation usually results in isolated nodes.

    Args:
        tree: [description]
        n_clusters: [description]

    Returns:
        DataFrame: [description]
    """
    nodes = list(tree.nodes)
    edges = nx.to_pandas_edgelist(tree)
    for i in range(n_clusters):
        idx = edges['corr'].idxmin()
        print(
            f"Edge from {edges.loc[idx, 'source']} ",
            f"to {edges.loc[idx, 'target']} "
            f"with weight {edges.loc[idx, 'corr']} is removed."
        )
        edges.drop(idx, inplace=True)

    graph = nx.from_pandas_edgelist(edges)
    graph.add_nodes_from(nodes)
    return graph


def draw_graph(g: nx.Graph, pos: dict = None):
    """Plot graph using given coordinates.

    Args:
        g: graph to be plotted.
        pos: coordinates of nodes. Defaults to None.
    """
    nx.draw(
        g, pos, with_labels=True, font_weight='bold',
        node_size=600, node_color='lightblue',
    )


def gather_nodes_info(g: nx.Graph) -> DataFrame:
    """Collect info about nodes in a graph.

    Note:
        Nodes with high degrees are expected to be centers of clusters.

    Args:
        g: some graph.

    Returns:
        Degrees of nodes.
    """
    res = pd.DataFrame.from_dict(dict(g.degree()), orient='index')
    res.columns = ['degree']
    res.sort_values('degree', ascending=False, inplace=True)
    return res


def find_weak_edge(tree: nx.Graph, nodes: list) -> List[tuple]:
    """Find weakest edge between given list of nodes.

    Args:
        tree: [description]
        nodes: [description]

    Returns:
        List of edges to be removed.
    """
    if not nx.is_tree(tree):
        logger.warning('The passed graph is not a tree.')

    pairs = itertools.combinations(nodes, r=2)
    edges_to_remove = []
    for source, target in pairs:
        paths = nx.shortest_simple_paths(tree, source, target)
        paths = list(paths)
        if len(paths) > 1:
            logger.warning('More than one path in tree.')
        path = paths[0]
        edges_path = list(pairwise(path))

        weights = [tree.edges[edge]['corr'] for edge in edges_path]
        idx = np.argmin(weights)
        try:
            if len(idx) > 1:
                logger.warning(
                    f'There are multiple edges with min weights: {idx}.'
                )
        except TypeError:
            pass

        idx_edge = edges_path[idx]
        edges_to_remove.append(idx_edge)

    for edge in edges_to_remove:
        try:
            tree.remove_edge(*edge)
        except nx.NetworkXError:
            pass

    return edges_to_remove


def plot_corr_mat(
    g: nx.Graph, representatives: list, df: DataFrame,
    num_emptys: Optional[int] = 5
):
    """Plot heatmap for re-indexed correlation matrix.

    Warning:
        There is no check implemented, so make sure arguments match.

    Args:
        g: clustered graph.
        representatives: all the representative entities.
        df: original data set.
        num_emptys: number of empty columns to separate blocks. Default
            to be 5.
    """
    idx = []
    emptys = ['nan'] * num_emptys
    for re in representatives:
        idx_new = list(nx.node_connected_component(g, re))
        idx_new.sort()
        idx = idx + idx_new + emptys

    # Remove last several empty columns and rows.
    for i in range(num_emptys):
        idx.pop()

    corr_mat = df.corr()
    corr_mat = corr_mat.reindex(idx, fill_value=0)
    corr_mat = corr_mat.reindex(idx, axis="columns", fill_value=0)

    sns.heatmap(
        corr_mat.values, center=0, cmap="bwr",
        xticklabels=False, yticklabels=False,
    )


def inspect_nan_corr(mat: DataFrame) -> List[tuple]:
    """Inspect list of tuples whose correlations are NaN.

    Args:
        mat: correlation matrix.

    Returns:
        Coordinates of NaN entries.
    """
    res = mat.fillna(100)
    res = res.stack()
    return res[res > 1].index.to_list()
