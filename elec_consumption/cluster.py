"""Functions for clustering analysis."""
import itertools
from typing import Dict, List, Optional, Tuple

from loguru import logger
import networkx as nx
from networkx.utils.misc import pairwise
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns

__all__ = ['init_graph_weight']


def mst_corr(df: DataFrame) -> Tuple[nx.Graph, nx.Graph]:
    """Init complete graph with correlations and exec min span tree.

    Warning:
        NaN entries are filled with 0 automatically.

    Args:
        df: original longitudinal in wide format.

    Returns:
        Min spanning tree based on pearson correlation coefficients.
    """
    corr_long = df.corr()

    # Set NaN entries to 0
    nan_entries = inspect_nan_corr(corr_long)
    if nan_entries:
        for entry in nan_entries:
            corr_long.loc[entry[0], entry[1]] = 0

    corr_long = corr_long.stack().to_frame()
    corr_long.index.names = ['source', 'target']
    corr_long.columns = ['corr']
    corr_long.reset_index(inplace=True)

    # Remove redundant correlation values.
    corr_long = corr_long[corr_long['target'] > corr_long['source']]

    graph = nx.from_pandas_edgelist(corr_long, edge_attr='corr')

    tree_generator = nx.maximum_spanning_edges(graph, weight='corr')
    tree = nx.from_edgelist(tree_generator)
    return tree, graph


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


def remove_weak_edges(tree: nx.Graph, representatives: list) -> dict:
    """Remove weakest edges between representative entities one-by-one.

    Args:
        tree: [description]
        representatives: [description]

    Returns:
        List of edges to be removed.
    """
    if not nx.is_tree(tree):
        logger.warning('The passed graph is not a tree.')

    pairs = itertools.combinations(representatives, r=2)
    for source, target in pairs:
        try:
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
            if idx_edge in tree.edges:
                tree.remove_edge(*idx_edge)
                logger.debug(
                    f"Edge {idx_edge[0]}-{idx_edge[1]} has been removed."
                )
        except nx.NetworkXNoPath:  # No path between given pair.
            pass

    components = {}
    for re in representatives:
        components[re] = list(nx.node_connected_component(tree, re))
        components[re].sort()

    return components


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
    df = mat.fillna(100)
    df = df.stack()
    res = df[df > 1].index.to_list()
    if res:
        logger.warning(f'There are NaN entries: {res}.')
    return res


def get_labels(components: Dict[int, List[int]]) -> List[int]:
    """Get integer labels for a set of components.

    Warning:
        Names of units must be integer in this function, because labels
        are sorted according to those integer values.

    Args:
        components: a set of components keyed by representative units.

    Returns:
        Integer labels with 0 being the minimum value.
    """
    units = [item for sublist in components.values() for item in sublist]
    logger.info(f"There are {len(units)} units.")

    label = 0
    labels = [0] * len(units)
    for key in components.keys():
        for i in components[key]:
            labels[i] = label
        label += 1

    return labels
