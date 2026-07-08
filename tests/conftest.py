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
