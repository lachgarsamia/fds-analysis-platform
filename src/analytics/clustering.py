"""PCA projection + clustering over ensemble feature vectors (M3.1.2).

sklearn is a new dependency introduced specifically for this milestone
(pyproject.toml) -- kept out of the app's earlier, lighter dependency set
until an actual feature needed it, per this project's own "don't add
until it's earned" convention (ROADMAP.md §7.3).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Fixed for reproducibility -- the DoD's clustering result (does it align
# with candle count?) must not depend on which run of KMeans's random
# initialization happened to land first.
RANDOM_STATE = 42

# Spec's own DoD sanity check is a 1-vs-2-candle split, i.e. 2 groups --
# used as the default rather than an arbitrary "reasonable-sounding"
# number, though callers can ask for more.
DEFAULT_N_CLUSTERS = 2


def run_pca(feature_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """(n_scenarios, n_components) projection. Features are on very
    different scales (fractions in [0,1] vs temperatures in the hundreds
    vs threshold-crossing seconds) -- standardizing each feature to zero
    mean/unit variance first keeps large-magnitude features (raw °C
    curves) from dominating the projection just because of their units,
    not because they're more informative."""
    if feature_matrix.shape[0] == 0:
        return np.zeros((0, n_components))
    standardized = _standardize(feature_matrix)
    n_components = min(n_components, feature_matrix.shape[0], feature_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return pca.fit_transform(standardized)


def run_clustering(feature_matrix: np.ndarray, n_clusters: int = DEFAULT_N_CLUSTERS) -> np.ndarray:
    """Cluster label per scenario (row of feature_matrix), same
    standardization as run_pca so clustering and the 2D scatter it's
    plotted against are consistent with each other."""
    if feature_matrix.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    n_clusters = min(n_clusters, feature_matrix.shape[0])
    standardized = _standardize(feature_matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    return kmeans.fit_predict(standardized)


def _standardize(feature_matrix: np.ndarray) -> np.ndarray:
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std == 0] = 1.0  # a constant feature column must not divide by zero
    return (feature_matrix - mean) / std


def cluster_alignment(cluster_labels: np.ndarray, factor_values: list) -> float:
    """Fraction of scenarios whose cluster label matches the majority
    cluster for their own factor value (e.g. candle count) -- the DoD's
    "clusters align with at least one interpretable factor" sanity check,
    made into an actual number rather than an eyeballed scatter plot.
    1.0 = every cluster is "pure" w.r.t. this factor; chance level for a
    2-cluster/2-value split is ~0.5.
    """
    if len(cluster_labels) == 0:
        return 0.0
    cluster_labels = np.asarray(cluster_labels)
    factor_values = np.asarray(factor_values)
    correct = 0
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        values_in_cluster = factor_values[mask]
        counts = np.bincount(values_in_cluster.astype(int))
        correct += counts.max()
    return correct / len(cluster_labels)
