"""
export.py
---------
Background MP4/GIF animation export (M1.5).

Frames are rendered offscreen through a dedicated Figure/FigureCanvasAgg
-- never the live on-screen canvas -- so exporting never disturbs
playback, and the export itself runs on a background QThread so the GUI
thread stays responsive while it works. Cancel is cooperative (a
threading.Event checked between frames), matching the app's existing
cooperative-stop philosophy elsewhere (see simulation_controller.py's
history). A partial file is never left behind: frames are written to a
temp path and only moved to the requested destination on full,
uncancelled completion; any cancellation or error just discards the
temp directory.
"""

import os
import shutil
import tempfile
import threading

from PyQt5 import QtCore
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def ffmpeg_available() -> bool:
    """Whether ffmpeg is on PATH -- determines MP4 vs GIF (Pillow) fallback,
    per M1.5's spec."""
    return shutil.which("ffmpeg") is not None


class AnimationExporter(QtCore.QThread):
    """Renders data[start:end] to output_path as MP4 (ffmpeg) or GIF
    (Pillow), at fps, using the same colormap/clim/interpolation as the
    live view so the export matches what the user was looking at."""

    progress = QtCore.pyqtSignal(int, int)  # (frames_done, frames_total)
    finished_ok = QtCore.pyqtSignal(str)    # output_path
    error = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, data, output_path: str, fps: int, cmap: str,
                 vmin: float, vmax: float, interpolation: str,
                 start: int = 0, end: int = None):
        super().__init__()
        self._data = data
        self._output_path = output_path
        self._fps = max(1, fps)
        self._cmap = cmap
        self._vmin = vmin
        self._vmax = vmax
        self._interpolation = interpolation
        self._start = max(0, start)
        self._end = end if end is not None else data.shape[0]
        self._cancel_event = threading.Event()

    def request_cancel(self):
        self._cancel_event.set()

    def run(self):
        is_mp4 = self._output_path.lower().endswith(".mp4")
        tmp_dir = tempfile.mkdtemp(prefix="fdsvis_export_")
        tmp_path = os.path.join(tmp_dir, os.path.basename(self._output_path))
        try:
            if is_mp4:
                self._export_mp4(tmp_path)
            else:
                self._export_gif(tmp_path)

            if self._cancel_event.is_set():
                self.cancelled.emit()
                return

            shutil.move(tmp_path, self._output_path)
            self.finished_ok.emit(self._output_path)
        except Exception as e:  # noqa: BLE001 - never let the export thread die silently
            self.error.emit(f"Export failed: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _make_figure(self):
        fig = Figure(dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.subplots_adjust(top=0.97, bottom=0.03, left=0.02, right=0.95)
        image = ax.imshow(
            self._data[self._start], cmap=self._cmap, interpolation=self._interpolation,
            aspect="auto", vmin=self._vmin, vmax=self._vmax,
        )
        colorbar = fig.colorbar(image, fraction=0.04, pad=0.02)
        colorbar.set_label("Temperature (°C)")
        return fig, canvas, image

    def _export_mp4(self, tmp_path: str):
        from matplotlib.animation import FFMpegWriter
        fig, canvas, image = self._make_figure()
        total = self._end - self._start
        writer = FFMpegWriter(fps=self._fps)
        with writer.saving(fig, tmp_path, dpi=100):
            for i in range(self._start, self._end):
                if self._cancel_event.is_set():
                    return
                image.set_data(self._data[i])
                writer.grab_frame()
                self.progress.emit(i - self._start + 1, total)

    def _export_gif(self, tmp_path: str):
        from PIL import Image
        fig, canvas, image = self._make_figure()
        total = self._end - self._start
        frames = []
        for i in range(self._start, self._end):
            if self._cancel_event.is_set():
                return
            image.set_data(self._data[i])
            canvas.draw()
            width, height = canvas.get_width_height()
            frame = Image.frombuffer(
                "RGBA", (width, height), canvas.buffer_rgba(), "raw", "RGBA", 0, 1
            ).convert("RGB")
            frames.append(frame)
            self.progress.emit(i - self._start + 1, total)

        if self._cancel_event.is_set():
            return
        duration_ms = round(1000 / self._fps)
        frames[0].save(
            tmp_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0
        )
