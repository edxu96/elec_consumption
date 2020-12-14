# AUTOGENERATED! DO NOT EDIT! File to edit: index.ipynb (unless otherwise specified).

__all__ = ['cal_inertia']

# Cell
def cal_inertia(n_clusters: int) -> float:
    """Calculate sum of distances to closest cluster centre.

    Args:
        n_clusters: number of clusters.

    Returns:
        Sum of distances to closest cluster centre.
    """
    km = TimeSeriesKMeans(
        n_clusters=n_clusters, verbose=False, random_state=_seed)
    km.fit_predict(mts)
    return km.inertia_
