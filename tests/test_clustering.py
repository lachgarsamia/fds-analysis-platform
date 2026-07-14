import os

import numpy as np
import pytest

from analytics.clustering import (
    DEFAULT_N_CLUSTERS,
    RANDOM_STATE,
    cluster_alignment,
    run_clustering,
    run_pca,
)
from analytics.features import build_feature_index, build_feature_matrix
from load_data import SIM_ROOT

requires_real_dataset = pytest.mark.skipif(
    not os.path.isdir(SIM_ROOT), reason="real fds/sim/ dataset not present"
)


def _two_obvious_blobs(n_per_blob=10, seed=0):
    rng = np.random.default_rng(seed)
    blob_a = rng.normal(loc=0.0, scale=0.1, size=(n_per_blob, 5))
    blob_b = rng.normal(loc=10.0, scale=0.1, size=(n_per_blob, 5))
    return np.vstack([blob_a, blob_b])


class TestRunPCA:
    def test_output_shape(self):
        matrix = _two_obvious_blobs()
        result = run_pca(matrix, n_components=2)
        assert result.coords.shape == (20, 2)

    def test_empty_matrix_returns_empty(self):
        result = run_pca(np.zeros((0, 5)))
        assert result.coords.shape == (0, 2)
        assert result.explained_variance_ratio.shape == (2,)

    def test_deterministic_across_calls(self):
        matrix = _two_obvious_blobs()
        a = run_pca(matrix)
        b = run_pca(matrix)
        assert np.allclose(a.coords, b.coords)
        assert np.allclose(a.explained_variance_ratio, b.explained_variance_ratio)

    def test_n_components_capped_by_n_features_and_n_samples(self):
        # 1 sample, 3 features -- can't ask for 2 PCA components.
        result = run_pca(np.array([[1.0, 2.0, 3.0]]), n_components=2)
        assert result.coords.shape[0] == 1

    def test_explained_variance_ratio_sums_to_at_most_one(self):
        # Two obvious, well-separated blobs -- almost all variance should
        # be captured by 2 components over only 5 features.
        matrix = _two_obvious_blobs()
        result = run_pca(matrix, n_components=2)
        assert result.explained_variance_ratio.shape == (2,)
        assert 0.0 < result.explained_variance_ratio.sum() <= 1.0 + 1e-9
        # A real, non-placeholder number -- the two obvious blobs are
        # separated along one dominant axis, so PC1 alone should explain
        # a large majority of the variance.
        assert result.explained_variance_ratio[0] > 0.5


class TestRunClustering:
    def test_separates_two_obvious_blobs(self):
        matrix = _two_obvious_blobs()
        labels = run_clustering(matrix, n_clusters=2)
        first_half = labels[:10]
        second_half = labels[10:]
        assert len(set(first_half)) == 1
        assert len(set(second_half)) == 1
        assert first_half[0] != second_half[0]

    def test_deterministic_across_calls(self):
        matrix = _two_obvious_blobs()
        a = run_clustering(matrix)
        b = run_clustering(matrix)
        assert (a == b).all()

    def test_empty_matrix_returns_empty(self):
        labels = run_clustering(np.zeros((0, 5)))
        assert labels.shape == (0,)

    def test_n_clusters_capped_by_n_samples(self):
        labels = run_clustering(np.array([[1.0, 2.0]]), n_clusters=5)
        assert labels.shape == (1,)

    def test_default_n_clusters_is_two(self):
        assert DEFAULT_N_CLUSTERS == 2


class TestClusterAlignment:
    def test_perfect_alignment_is_one(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        factor = [0, 0, 0, 1, 1, 1]
        assert cluster_alignment(labels, factor) == 1.0

    def test_no_alignment_is_chance_level(self):
        # Each cluster has an even 50/50 split of the factor -- worst case.
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        factor = [0, 1, 0, 1, 0, 1, 0, 1]
        assert cluster_alignment(labels, factor) == 0.5

    def test_empty_input_returns_zero(self):
        assert cluster_alignment(np.array([]), []) == 0.0

    def test_partial_alignment(self):
        # cluster 0: [0,0,0,1] -> majority 0 (3/4); cluster 1: [1,1] -> majority 1 (2/2)
        labels = np.array([0, 0, 0, 0, 1, 1])
        factor = [0, 0, 0, 1, 1, 1]
        assert cluster_alignment(labels, factor) == 5 / 6


@requires_real_dataset
class TestClusteringRealData:
    """The DoD's own sanity check, made concrete and pinned: does a 2-way
    KMeans clustering over the real 24-scenario feature space actually
    align with candle count better than with a factor it has no obvious
    physical reason to track (door width)? Verified once interactively
    (83.3% candle alignment vs. 50.0%, i.e. chance level, for door) and
    pinned here so a future change to features.py/clustering.py can't
    silently break the milestone's own claim without a visible failure."""

    def test_candle_count_alignment_beats_door_alignment(self):
        from data_provider import load_simulation_data
        sim_data = load_simulation_data()
        if sim_data.is_demo:
            pytest.skip("real dataset not present")

        features = build_feature_index(sim_data.manifest, sim_data.store, sim_data.timesteps_per_second)
        matrix, case_indices = build_feature_matrix(features)
        labels = run_clustering(matrix, n_clusters=2)

        by_case = {f.case_index: f for f in features}
        candles = [by_case[ci].candles for ci in case_indices]
        doors = [by_case[ci].door for ci in case_indices]

        candle_alignment = cluster_alignment(labels, candles)
        door_alignment = cluster_alignment(labels, doors)

        assert candle_alignment == pytest.approx(0.8333, abs=0.01)
        assert door_alignment == pytest.approx(0.5, abs=0.01)
        assert candle_alignment > door_alignment, \
            "clustering should track candle count meaningfully better than an unrelated factor"
