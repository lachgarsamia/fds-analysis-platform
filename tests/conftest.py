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
def _isolated_qsettings(tmp_path, monkeypatch):
    """Redirect every QSettings(org, app) construction (main_window.py's
    self.settings, the only such call site) to a per-test scratch file --
    without this, tests read/write the real, persisted
    ~/Library/Preferences/com.fzjuelich.FDSSLCFVisualizer.plist, so a test
    run can inherit stale state from the developer's last manual session
    and can leave the real app's saved preferences altered (window
    geometry, ui_scale, interpolation, ...) after the suite exits.

    setPath() alone is not enough: on macOS, QSettings(org, app) --
    Qt's two-argument convenience constructor, the exact form
    main_window.py uses -- *always* resolves to NativeFormat
    (CFPreferences-backed) whenever a native format exists on the
    platform, regardless of setDefaultFormat(); only the zero-argument
    QSettings() (via QCoreApplication's org/app name) or an explicit
    QSettings(IniFormat, ...) call honor it. setPath() has no effect on
    NativeFormat either way. Confirmed directly: a setDefaultFormat() +
    setPath()-only version of this fixture (this function's first,
    incorrect implementation, briefly committed) left
    QSettings('FZJuelich', 'FDSSLCFVisualizer').fileName() still pointing
    at the real .plist -- silently masked in the full suite by
    test_bilinear_is_the_fresh_install_interpolation_default's own
    now-redundant QSettings.value() monkeypatch, until removing that
    workaround surfaced it.

    So the two-arg call itself has to be rewritten: wrapping __init__ to
    force IniFormat/UserScope whenever called with exactly two plain
    string args (organization, application) -- main_window.py's call
    shape -- routes it through setPath()'s redirect; any other
    construction form (explicit format/scope, a path-based QSettings like
    test_kiosk_polish.py's own scratch instance) passes through
    unchanged."""
    from PyQt5 import QtCore
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path))
    real_init = QtCore.QSettings.__init__

    def _init(self, *args, **kwargs):
        if len(args) == 2 and not kwargs and all(isinstance(a, str) for a in args):
            real_init(self, QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, *args)
        else:
            real_init(self, *args, **kwargs)

    monkeypatch.setattr(QtCore.QSettings, "__init__", _init)
