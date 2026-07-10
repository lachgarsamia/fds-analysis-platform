"""Unit tests for manifest.py (M2.1): folder-name parsing, case_index
assignment, and manifest.json persistence.

Uses synthetic empty folders under tmp_path (scan_scenarios only reads
folder *names*, not their contents) rather than the real fds/sim/ dataset,
matching this suite's existing convention of not depending on data that
isn't committed to the repo (see tests/fixtures/, .gitignore's fds/sim/).
"""

import os

import numpy as np
import pytest

import manifest as manifest_module
from manifest import (
    scan_scenarios, save_manifest, load_manifest, get_manifest,
    factor_counts, data_matrix_from_manifest, ScenarioEntry,
)


def _make_folders(root, names):
    for name in names:
        os.makedirs(os.path.join(root, name))


class TestScanScenarios:
    def test_case_index_matches_sorted_folder_order(self, tmp_path):
        names = ["c2_d1_vod2_voc1", "c1_d0_vod0_voc0", "c1_d1_vod1_voc0"]
        _make_folders(tmp_path, names)

        entries = scan_scenarios(str(tmp_path))
        assert [e.folder for e in entries] == sorted(names)
        assert [e.case_index for e in entries] == list(range(len(names)))

    def test_factor_indices_derived_from_actual_values_present(self, tmp_path):
        """Only c1/c2 (2 levels) and d0 (1 level) exist here -- indices must
        reflect that, not an assumed N_CANDLES=2/N_DOORS=2."""
        _make_folders(tmp_path, ["c1_d0_vod0_voc0", "c2_d0_vod1_voc0"])
        entries = scan_scenarios(str(tmp_path))
        by_folder = {e.folder: e for e in entries}

        assert by_folder["c1_d0_vod0_voc0"].candles == 0
        assert by_folder["c2_d0_vod1_voc0"].candles == 1
        assert by_folder["c1_d0_vod0_voc0"].door == 0
        assert by_folder["c2_d0_vod1_voc0"].door == 0  # only one door level present
        assert by_folder["c1_d0_vod0_voc0"].vod == 0
        assert by_folder["c2_d0_vod1_voc0"].vod == 1

    def test_matches_real_dataset_factor_structure(self, tmp_path):
        """Full 2x2x3x2 factorial, matching the real fds/sim/ naming."""
        names = [f"c{c}_d{d}_vod{v}_voc{o}"
                 for c in (1, 2) for d in (0, 1) for v in (0, 1, 2) for o in (0, 1)]
        _make_folders(tmp_path, names)
        entries = scan_scenarios(str(tmp_path))
        assert len(entries) == 24
        assert factor_counts(entries) == (2, 2, 3, 2)

    def test_unrecognized_folder_name_is_skipped_not_fatal(self, tmp_path):
        _make_folders(tmp_path, ["c1_d0_vod0_voc0", "not_a_scenario_folder"])
        entries = scan_scenarios(str(tmp_path))
        assert [e.folder for e in entries] == ["c1_d0_vod0_voc0"]

    def test_path_is_absolute(self, tmp_path):
        _make_folders(tmp_path, ["c1_d0_vod0_voc0"])
        entries = scan_scenarios(str(tmp_path))
        assert os.path.isabs(entries[0].path)


class TestDataMatrixFromManifest:
    def test_matches_old_build_data_matrix_on_full_factorial(self, tmp_path):
        from scenario_store import build_data_matrix
        names = [f"c{c}_d{d}_vod{v}_voc{o}"
                 for c in (1, 2) for d in (0, 1) for v in (0, 1, 2) for o in (0, 1)]
        _make_folders(tmp_path, names)
        entries = scan_scenarios(str(tmp_path))

        old = build_data_matrix(2, 2, 3, 2)
        new = data_matrix_from_manifest(entries)
        assert np.array_equal(old, new)


class TestManifestPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        _make_folders(tmp_path, ["c1_d0_vod0_voc0", "c2_d1_vod2_voc1"])
        entries = scan_scenarios(str(tmp_path))
        manifest_path = str(tmp_path / "manifest.json")

        save_manifest(entries, manifest_path)
        loaded = load_manifest(manifest_path)

        assert loaded == entries

    def test_get_manifest_writes_file_on_first_call(self, tmp_path):
        _make_folders(tmp_path, ["c1_d0_vod0_voc0"])
        manifest_path = str(tmp_path / "manifest.json")
        assert not os.path.exists(manifest_path)

        entries = get_manifest(str(tmp_path), manifest_path)
        assert os.path.exists(manifest_path)
        assert len(entries) == 1

    def test_get_manifest_loads_from_disk_without_rescanning(self, tmp_path, monkeypatch):
        _make_folders(tmp_path, ["c1_d0_vod0_voc0"])
        manifest_path = str(tmp_path / "manifest.json")
        get_manifest(str(tmp_path), manifest_path)  # first call: scans + writes

        scan_calls = []
        real_scan = manifest_module.scan_scenarios
        monkeypatch.setattr(manifest_module, "scan_scenarios",
                             lambda root: scan_calls.append(root) or real_scan(root))

        get_manifest(str(tmp_path), manifest_path)
        assert scan_calls == [], "a present manifest.json should be loaded, not rescanned"

    def test_get_manifest_force_regenerate_rescans(self, tmp_path, monkeypatch):
        _make_folders(tmp_path, ["c1_d0_vod0_voc0"])
        manifest_path = str(tmp_path / "manifest.json")
        get_manifest(str(tmp_path), manifest_path)

        scan_calls = []
        real_scan = manifest_module.scan_scenarios
        monkeypatch.setattr(manifest_module, "scan_scenarios",
                             lambda root: scan_calls.append(root) or real_scan(root))

        get_manifest(str(tmp_path), manifest_path, force_regenerate=True)
        assert scan_calls == [str(tmp_path)]

    def test_get_manifest_regenerates_on_corrupted_file(self, tmp_path):
        _make_folders(tmp_path, ["c1_d0_vod0_voc0"])
        manifest_path = str(tmp_path / "manifest.json")
        with open(manifest_path, "w") as f:
            f.write("not valid json")

        entries = get_manifest(str(tmp_path), manifest_path)  # must not raise
        assert len(entries) == 1

    def test_get_manifest_defaults_path_under_sim_root(self, tmp_path):
        _make_folders(tmp_path, ["c1_d0_vod0_voc0"])
        get_manifest(str(tmp_path))
        assert os.path.exists(os.path.join(str(tmp_path), "manifest.json"))
