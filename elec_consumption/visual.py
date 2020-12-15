"""Func to visualise clustering results."""
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [4 * 1.61803398875, 4]


def plot_silhouette(
    n_clusters: int, cluster_labels: np.array, data: np.ndarray,
):
    """Conduct Silhouette analysis for cluster results.

    Args:
        n_clusters: number of clusters.
        cluster_labels: cluster labels corresponds to series in `data`.
        data: repeated measured data.
    """
    # Create a subplot with 2 row and 1 columns
    _, ax = plt.subplots(1, 1)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


def plot_silhouette_2d(
    n_clusters: int, cluster_labels: np.array, data: np.ndarray,
    centers: np.ndarray
):    
    """Visualise clusters in 2-D using Silhouette analysis.

    Args:
        n_clusters: number of clusters.
        cluster_labels: cluster labels corresponds to series in `data`.
        data: repeated measured data.
        centers: centers of clusters.
    """
    _, ax = plt.subplots(1, 1)

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax.scatter(
        data[:, 0], data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
        c=colors, edgecolor='k'
    )

    # Labeling the clusters
    # Draw white circles at cluster centers
    ax.scatter(
        centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1,
        s=200, edgecolor='k',
    )

    for i, c in enumerate(centers):
        ax.scatter(
            c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k'
        )

    ax.set_title("2-D visualization of clustered time series")
    ax.set_xlabel("Feature space for the 1st feature")
    ax.set_ylabel("Feature space for the 2nd feature")
