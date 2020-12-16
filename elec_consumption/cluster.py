"""Functions for clustering analysis."""
import networkx as nx
from pandas.core.frame import DataFrame

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
    nx.draw(
        g, pos, with_labels=True, font_weight='bold',
        node_size=600, node_color='lightblue',
    )
