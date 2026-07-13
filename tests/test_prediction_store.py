import json
import os

import numpy as np
import pytest

from prediction_store import PredictionSource


class FakeRealStore:
    def get_extent(self, scenario_index, key=None):
        return [0.0, 1.0, 0.0, 0.5]


class TestPredictionSourceAbsent:
    def test_not_available_when_predictions_dir_missing(self, tmp_path):
        source = PredictionSource(FakeRealStore(), predictions_dir=str(tmp_path / "nope"))
        assert source.is_available is False
        assert source.case_indices == []

    def test_not_available_when_manifest_missing(self, tmp_path):
        source = PredictionSource(FakeRealStore(), predictions_dir=str(tmp_path))
        assert source.is_available is False

    def test_disabled_ignores_a_present_directory(self, tmp_path):
        manifest = {"cases": {"5": {"folder": "x", "n_frames": 10}}}
        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest, f)
        np.save(tmp_path / "5.npy", np.zeros((10, 4, 4)))
        source = PredictionSource(FakeRealStore(), predictions_dir=str(tmp_path), enabled=False)
        assert source.is_available is False


class TestPredictionSourcePresent:
    def _make_source(self, tmp_path, case_indices=(5, 12)):
        manifest = {"cases": {}}
        for ci in case_indices:
            arr = np.full((10, 4, 4), float(ci))
            np.save(tmp_path / f"{ci}.npy", arr)
            manifest["cases"][str(ci)] = {"folder": f"case_{ci}", "n_frames": 10}
        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest, f)
        return PredictionSource(FakeRealStore(), predictions_dir=str(tmp_path))

    def test_is_available_and_case_indices_sorted(self, tmp_path):
        source = self._make_source(tmp_path, case_indices=(12, 5))
        assert source.is_available is True
        assert source.case_indices == [5, 12]

    def test_get_returns_the_right_array(self, tmp_path):
        source = self._make_source(tmp_path, case_indices=(5, 12))
        data = source.get(12)
        assert data.shape == (10, 4, 4)
        assert np.all(data == 12.0)

    def test_is_cached_true_for_known_case_false_otherwise(self, tmp_path):
        source = self._make_source(tmp_path, case_indices=(5,))
        assert source.is_cached(5) is True
        assert source.is_cached(99) is False

    def test_get_extent_delegates_to_real_store(self, tmp_path):
        source = self._make_source(tmp_path, case_indices=(5,))
        assert source.get_extent(5) == [0.0, 1.0, 0.0, 0.5]

    def test_get_unknown_case_raises(self, tmp_path):
        source = self._make_source(tmp_path, case_indices=(5,))
        with pytest.raises(KeyError):
            source.get(999)
