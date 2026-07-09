"""Standalone FPS benchmark for M1.3.3's blitting change.

Not a pytest test (bench_rendering.py doesn't match test_*.py). Run
directly (offscreen, no real display needed):

    QT_QPA_PLATFORM=offscreen python3 tests/bench_rendering.py

Compares per-frame draw cost: blit_update() (M1.3.3) vs. the previous
full draw_idle() path, on real scenario data.
"""

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt5 import QtWidgets  # noqa: E402

N_FRAMES = 120


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from data_provider import load_simulation_data
    from main_window import MainWindow

    sim_data = load_simulation_data()
    window = MainWindow(sim_data)
    window.show()
    app.processEvents()

    data = window.controller.store.get(window.controller.current_case_index())
    n = min(N_FRAMES, data.shape[0])
    frames = [data[i] for i in range(n)]

    # Warm up (first draw is always slower -- font/layout caching etc.)
    window._redraw(frames[0])
    app.processEvents()

    t0 = time.perf_counter()
    for frame in frames:
        window.heatmap.set_data(frame)
        window.canvas.blit_update(window.heatmap)
    blit_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for frame in frames:
        window.heatmap.set_data(frame)
        window.canvas.draw()  # synchronous full draw, the pre-M1.3.3 path
    full_draw_elapsed = time.perf_counter() - t0

    window.close()

    blit_fps = n / blit_elapsed
    full_fps = n / full_draw_elapsed
    speedup = full_draw_elapsed / blit_elapsed if blit_elapsed > 0 else float("inf")

    print(f"frames: {n}")
    print(f"blit_update:      {blit_elapsed:.4f}s total  ({blit_elapsed/n*1000:.3f} ms/frame, {blit_fps:.1f} fps)")
    print(f"full draw() (old): {full_draw_elapsed:.4f}s total  ({full_draw_elapsed/n*1000:.3f} ms/frame, {full_fps:.1f} fps)")
    print(f"speedup: {speedup:.1f}x  (M1.3.3 DoD target: >=5x)")


if __name__ == "__main__":
    main()
