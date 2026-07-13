"""Ensemble analytics dock: PCA scatter + clustering (M3.1.2).

Static matplotlib -- computed once at construction from the already-loaded
feature index, no per-frame redraw, so it never touches the playback tick
path (_on_time_changed) at all. That's the actual mechanism behind the
DoD's "panel doesn't degrade playback" claim, not just an assumption: this
widget has no connection to TimeController whatsoever.
"""

from __future__ import annotations

import matplotlib.cm as mpl_cm
from PyQt5 import QtCore, QtWidgets

from analytics.clustering import DEFAULT_N_CLUSTERS, cluster_alignment, run_clustering, run_pca
from analytics.features import build_feature_matrix
from widgets import MplCanvas

# Cycled by a scenario's `candles` factor index so an unexpected extra
# candle-count level (a future dataset) degrades to reusing markers
# rather than crashing.
CANDLE_MARKERS = ("o", "^", "s", "D", "P")
CLUSTER_CMAP_NAME = "tab10"


class AnalyticsPanelDock(QtWidgets.QDockWidget):
    scenario_activated = QtCore.pyqtSignal(int)  # case_index, same convention as ExperimentBrowserDock

    def __init__(self, features: list, parent=None):
        super().__init__("Ensemble Analytics", parent)
        self.setObjectName("analyticsPanelDock")
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        self._features = sorted(features, key=lambda f: f.case_index)
        self._case_indices: list = []
        self._coords = None

        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.canvas = MplCanvas(root)
        self.canvas.setAccessibleName("Ensemble PCA scatter plot")
        self.canvas.setToolTip(
            "Each point is one scenario, projected to 2D by PCA over its "
            "feature curves. Color = cluster, marker shape = candle count. "
            "Hover for details, click to load into the active view."
        )
        layout.addWidget(self.canvas, 1)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAccessibleName("Analytics panel scenario readout")
        layout.addWidget(self.status_label)

        self.setWidget(root)

        self.ax = self.canvas.fig.add_subplot(111)
        self._plot()

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def _plot(self):
        matrix, case_indices = build_feature_matrix(self._features)
        self._case_indices = case_indices
        if matrix.shape[0] == 0:
            self.status_label.setText("No scenarios to analyze.")
            return

        self._coords = run_pca(matrix, n_components=2)
        labels = run_clustering(matrix, n_clusters=DEFAULT_N_CLUSTERS)
        by_case = {f.case_index: f for f in self._features}

        colors = mpl_cm.get_cmap(CLUSTER_CMAP_NAME)

        for i, case_index in enumerate(case_indices):
            entry = by_case[case_index]
            marker = CANDLE_MARKERS[entry.candles % len(CANDLE_MARKERS)]
            self.ax.scatter(
                self._coords[i, 0], self._coords[i, 1],
                c=[colors(labels[i] % 10)], marker=marker, s=70,
                edgecolors="white", linewidths=0.5,
            )

        self.ax.set_xlabel("PC1")
        self.ax.set_ylabel("PC2")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.fig.subplots_adjust(top=0.95, bottom=0.1, left=0.08, right=0.97)
        self.canvas.draw_idle()

        candles = [by_case[ci].candles for ci in case_indices]
        alignment = cluster_alignment(labels, candles)
        self.status_label.setText(
            f"{len(case_indices)} scenarios, {DEFAULT_N_CLUSTERS} clusters "
            f"({alignment * 100:.0f}% match candle count). Hover a point for details."
        )

    def _nearest_case_index(self, event) -> int:
        if self._coords is None or event.xdata is None or event.ydata is None:
            return None
        deltas = self._coords - [event.xdata, event.ydata]
        distances_sq = (deltas ** 2).sum(axis=1)
        nearest = int(distances_sq.argmin())
        # A generous-but-bounded pick radius in data units, scaled to the
        # plot's own spread, so hovering empty space between points
        # doesn't claim the nearest one regardless of distance.
        spread = max(self._coords.max(axis=0) - self._coords.min(axis=0)) or 1.0
        if distances_sq[nearest] ** 0.5 > 0.05 * spread:
            return None
        return self._case_indices[nearest]

    def _on_hover(self, event):
        if event.inaxes != self.ax:
            return
        case_index = self._nearest_case_index(event)
        if case_index is None:
            return
        by_case = {f.case_index: f for f in self._features}
        entry = by_case[case_index]
        self.status_label.setText(f"{entry.folder} (case {case_index})")

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        case_index = self._nearest_case_index(event)
        if case_index is not None:
            self.scenario_activated.emit(case_index)
