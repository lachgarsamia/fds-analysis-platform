"""Unit tests for export.py: AnimationExporter (M1.5)."""

import os
import time

import numpy as np
from export import AnimationExporter, ffmpeg_available


def _run_and_wait(exporter, timeout_ms=10000):
    """Start the exporter, block until it truly finishes, then flush the
    Qt event loop so cross-thread queued signals (finished_ok/error/
    cancelled) are actually delivered before the caller inspects results."""
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance()
    exporter.start()
    exporter.wait(timeout_ms)
    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.005)


class TestAnimationExporter:
    def test_gif_export_succeeds_and_produces_a_file(self, qapp, tmp_path):
        data = np.random.uniform(20, 300, size=(15, 49, 101)).astype(np.float32)
        output = str(tmp_path / "out.gif")
        exporter = AnimationExporter(
            data, output, fps=4, cmap="gist_heat", vmin=20.0, vmax=300.0,
            interpolation="nearest", start=0, end=15,
        )
        results = {}
        progress = []
        exporter.progress.connect(lambda done, total: progress.append((done, total)))
        exporter.finished_ok.connect(lambda p: results.update(status="ok", path=p))
        exporter.error.connect(lambda m: results.update(status="error", message=m))

        _run_and_wait(exporter)

        assert results.get("status") == "ok"
        assert results["path"] == output
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0
        assert progress[-1] == (15, 15)

    def test_gif_export_partial_range(self, qapp, tmp_path):
        """fps/range are actually honored, not just accepted and ignored."""
        data = np.random.uniform(20, 300, size=(50, 49, 101)).astype(np.float32)
        output = str(tmp_path / "partial.gif")
        exporter = AnimationExporter(
            data, output, fps=4, cmap="gist_heat", vmin=20.0, vmax=300.0,
            interpolation="nearest", start=10, end=20,
        )
        progress = []
        exporter.progress.connect(lambda done, total: progress.append((done, total)))
        _run_and_wait(exporter)
        assert progress[-1] == (10, 10), "must render exactly [start, end), i.e. 10 frames"

    def test_cancel_leaves_no_partial_file(self, qapp, tmp_path):
        data = np.random.uniform(20, 300, size=(300, 49, 101)).astype(np.float32)
        output = str(tmp_path / "cancelled.gif")
        exporter = AnimationExporter(
            data, output, fps=4, cmap="gist_heat", vmin=20.0, vmax=300.0,
            interpolation="nearest", start=0, end=300,
        )
        results = {}
        progress = []
        exporter.progress.connect(lambda done, total: progress.append(done))
        exporter.cancelled.connect(lambda: results.update(status="cancelled"))
        exporter.finished_ok.connect(lambda p: results.update(status="ok"))

        from PyQt5 import QtWidgets
        app = QtWidgets.QApplication.instance()
        exporter.start()
        deadline = time.perf_counter() + 5.0
        while len(progress) < 5 and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert len(progress) >= 5, "export should have started rendering before we cancel it"
        exporter.request_cancel()
        exporter.wait(10000)
        for _ in range(20):
            app.processEvents()
            time.sleep(0.01)

        assert results.get("status") == "cancelled"
        assert not os.path.exists(output), "cancel must never leave a partial file at the destination"
        # the temp working directory must also have been cleaned up, not
        # just the final destination left absent
        assert not any(f.startswith("fdsvis_export_") for f in os.listdir(os.path.dirname(output) or "."))

    def test_mp4_without_ffmpeg_errors_cleanly_no_partial_file(self, qapp, tmp_path, monkeypatch):
        """Exercises the MP4 code path's error handling directly (this
        environment has no ffmpeg -- ffmpeg_available() is False, and the
        real UI never offers .mp4 in that case, but the exporter itself
        must still fail safely rather than crash if reached some other way)."""
        data = np.random.uniform(20, 300, size=(5, 49, 101)).astype(np.float32)
        output = str(tmp_path / "out.mp4")
        exporter = AnimationExporter(
            data, output, fps=4, cmap="gist_heat", vmin=20.0, vmax=300.0,
            interpolation="nearest", start=0, end=5,
        )
        results = {}
        exporter.error.connect(lambda m: results.update(status="error", message=m))
        exporter.finished_ok.connect(lambda p: results.update(status="ok"))

        _run_and_wait(exporter)

        assert results.get("status") == "error"
        assert not os.path.exists(output)

    def test_ffmpeg_available_matches_shutil_which(self, qapp):
        import shutil
        assert ffmpeg_available() == (shutil.which("ffmpeg") is not None)
