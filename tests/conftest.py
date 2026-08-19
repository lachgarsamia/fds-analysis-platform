"""Pytest configuration and shared fixtures for FDS Visualizer tests."""

import os
import pytest
from PyQt5 import QtWidgets


# Must set before any Qt imports in the test modules themselves
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication for all tests."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


@pytest.fixture
def fixtures_dir():
    """Path to the test fixtures directory."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "c1_d0_vod0_voc0")


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path):
    """Redirect every QSettings(org, app) construction (main_window.py's
    self.settings, the only such call site) to a per-test scratch file --
    without this, tests read/write the real, persisted
    ~/Library/Preferences/com.fzjuelich.FDSSLCFVisualizer.plist, so a test
    run can inherit stale state from the developer's last manual session
    and can leave the real app's saved preferences altered (window
    geometry, ui_scale, interpolation, ...) after the suite exits.

    setDefaultFormat() is required before setPath(), not just belt-and-
    suspenders: on macOS, QSettings(org, app) defaults to NativeFormat
    (CFPreferences-backed), and setPath() has no effect on NativeFormat --
    only IniFormat honors a custom search path."""
    from PyQt5 import QtCore
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path))
