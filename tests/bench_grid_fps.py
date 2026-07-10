"""Standalone FPS benchmark for M2.2's DoD: "2x2 grid, 4 different
scenarios, synced playback >=15 fps on dev machine."

Not a pytest test (bench_grid_fps.py doesn't match test_*.py). Run
directly (offscreen, no real display needed):

    QT_QPA_PLATFORM=offscreen python3 tests/bench_grid_fps.py

Drives MainWindow._on_time_changed -- the real per-tick "redraw every
visible cell" path a live playback session uses -- across a 2x2 grid of
4 distinct real scenarios, and reports fps. Per M1.3.3's own benchmark
note, offscreen rendering numbers are a lower/rougher bound than a real
display, not an exact match -- kept consistent with that existing
convention (tests/bench_rendering.py) rather than assumed equivalent.
"""

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt5 import QtWidgets  # noqa: E402

N_TICKS = 120
TARGET_FPS = 15


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from data_provider import load_simulation_data
    from main_window import MainWindow

    sim_data = load_simulation_data()
    window = MainWindow(sim_data)
    window.show()
    app.processEvents()

    if sim_data.is_demo:
        print("Real dataset not present (fds/sim/ missing) -- demo mode has no "
              "manifest to pick 4 distinct scenarios from. Nothing to benchmark.")
        window.close()
        return

    window._set_grid_layout("2x2")
    cells = window.view_grid.visible_cells()
    # 4 clearly distinct scenarios, not adjacent case_indices, so this
    # doesn't accidentally benchmark a degenerate "1 scenario in 4 cells" case.
    distinct_cases = [0, 6, 12, 18]
    for cell, case_index in zip(cells, distinct_cases):
        cell.scenario_combo.setCurrentIndex(case_index)
    deadline = time.perf_counter() + 5.0
    while window._pending_cell_prefetches and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()

    n_frames = min(N_TICKS, window._current_n_frames)

    # Warm up (first draw is always slower).
    window._on_time_changed(0)
    app.processEvents()

    t0 = time.perf_counter()
    for i in range(n_frames):
        window._on_time_changed(i % window._current_n_frames)
    elapsed = time.perf_counter() - t0

    window.close()

    fps = n_frames / elapsed if elapsed > 0 else float("inf")
    print(f"cells: {len(cells)}, scenarios: {distinct_cases}, ticks: {n_frames}")
    print(f"2x2 synced playback: {elapsed:.4f}s total ({elapsed/n_frames*1000:.3f} ms/tick, {fps:.1f} fps)")
    print(f"DoD target: >={TARGET_FPS} fps -- {'PASS' if fps >= TARGET_FPS else 'FAIL'}")


if __name__ == "__main__":
    main()
