"""Ensemble analytics dock: PCA scatter + clustering (M3.1.2).

Static matplotlib -- computed once from a feature index, no per-frame
redraw, so it never touches the playback tick path (_on_time_changed) at
all. That's the actual mechanism behind the DoD's "panel doesn't degrade
playback" claim, not just an assumption: this widget has no connection to
TimeController whatsoever.

The feature index itself (build_feature_index() over all 24 scenarios) is
NOT computed here -- it's supplied later via load_features(), computed by
MainWindow on a background thread (_AnalyticsFeatureWorker below) and only
once the panel is actually shown. That split exists because of a real bug
(fixed post-3fd87cf, see main_window.py's _build_analytics_panel): building
the feature index touches ScenarioStore for every scenario, so doing it
eagerly at MainWindow construction silently warmed the cache for all 24
scenarios before the window was even visible -- a startup-latency
regression, and also the reason two prefetch/cache-miss race-condition
regression tests started failing (their premise that a specific scenario
starts uncached no longer held).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.lines as mlines
from PyQt5 import QtCore, QtWidgets

from analytics.clustering import DEFAULT_N_CLUSTERS, cluster_alignment, run_clustering, run_pca
from analytics.features import build_feature_index, build_feature_matrix
from browser import FACTOR_LABELS
from widgets import MplCanvas

# Cycled by a scenario's `candles` factor index so an unexpected extra
# candle-count level (a future dataset) degrades to reusing markers
# rather than crashing.
CANDLE_MARKERS = ("o", "^", "s", "D", "P")
CLUSTER_CMAP_NAME = "tab10"


class AnalyticsPanelDock(QtWidgets.QDockWidget):
    scenario_activated = QtCore.pyqtSignal(int)  # case_index, same convention as ExperimentBrowserDock

    def __init__(self, features: list = None, parent=None):
        """features=None (the lazy-load path used by MainWindow) shows a
        "Loading..." placeholder instead of plotting -- call load_features()
        once the real feature index is ready. Passing a list plots it
        immediately, unchanged from before (existing tests construct this
        widget with a ready-made feature list and expect the plot done
        synchronously; that contract is preserved)."""
        super().__init__("Ensemble Analytics", parent)
        self.setObjectName("analyticsPanelDock")
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        self._features = sorted(features, key=lambda f: f.case_index) if features is not None else []
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
        self.ax.set_facecolor(MplCanvas.PLOT_BG)
        if features is None:
            self.status_label.setText("Loading ensemble analytics...")
        else:
            self._plot()

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def load_features(self, features: list):
        """Populates a panel constructed with features=None -- MainWindow's
        deferred/background load calls this once _AnalyticsFeatureWorker
        finishes. Clears the "Loading..." placeholder and plots for real."""
        self._features = sorted(features, key=lambda f: f.case_index)
        self._plot()

    def _plot(self):
        matrix, case_indices = build_feature_matrix(self._features)
        self._case_indices = case_indices
        if matrix.shape[0] == 0:
            self.status_label.setText("No scenarios to analyze.")
            return

        pca_result = run_pca(matrix, n_components=2)
        self._coords = pca_result.coords
        variance = pca_result.explained_variance_ratio
        labels = run_clustering(matrix, n_clusters=DEFAULT_N_CLUSTERS)
        by_case = {f.case_index: f for f in self._features}

        colors = mpl.colormaps[CLUSTER_CMAP_NAME]

        for i, case_index in enumerate(case_indices):
            entry = by_case[case_index]
            marker = CANDLE_MARKERS[entry.candles % len(CANDLE_MARKERS)]
            self.ax.scatter(
                self._coords[i, 0], self._coords[i, 1],
                c=[colors(labels[i] % 10)], marker=marker, s=70,
                edgecolors="white", linewidths=0.5,
            )

        self.ax.set_title("Ensemble PCA — scenario clustering by fire behavior", fontsize=11, fontweight="bold")
        self.ax.set_xlabel(self._axis_label("PC1", variance, 0))
        self.ax.set_ylabel(self._axis_label("PC2", variance, 1))
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        candles = [by_case[ci].candles for ci in case_indices]
        alignment = cluster_alignment(labels, candles)
        self._add_legends(labels, candles, colors)

        caption = (
            f"{len(case_indices)} scenarios, {DEFAULT_N_CLUSTERS} clusters — "
            f"{alignment * 100:.0f}% match candle count."
        )
        self.canvas.fig.text(0.5, 0.01, caption, ha="center", va="bottom",
                              fontsize=8, style="italic", color="#555555")

        self.canvas.fig.subplots_adjust(top=0.90, bottom=0.13, left=0.09, right=0.97)
        self.canvas.draw_idle()

        self.status_label.setText(
            f"{len(case_indices)} scenarios, {DEFAULT_N_CLUSTERS} clusters "
            f"({alignment * 100:.0f}% match candle count). Hover a point for details."
        )

    @staticmethod
    def _axis_label(name: str, variance, index: int) -> str:
        """"PC1 (62% variance explained)" -- the real number from this
        fit, not a placeholder; falls back to a bare name if this
        component wasn't computed (e.g. only 1 scenario -> 1 component)."""
        if index < len(variance):
            return f"{name} ({variance[index] * 100:.0f}% variance explained)"
        return name

    def _add_legends(self, labels, candles: list, colors) -> None:
        """Two separate legends -- point color encodes cluster, marker
        shape encodes candle count -- rather than a single legend trying
        to enumerate every (cluster, candle) combination actually present."""
        unique_clusters = sorted(set(int(label) for label in labels))
        cluster_handles = [
            mlines.Line2D(
                [], [], color=colors(cluster_id % 10), marker="o", linestyle="None",
                markersize=8, markeredgecolor="white", markeredgewidth=0.5,
                label=f"Cluster {cluster_id}",
            )
            for cluster_id in unique_clusters
        ]
        cluster_legend = self.ax.legend(
            handles=cluster_handles, loc="upper left", title="Cluster",
            fontsize=8, title_fontsize=8, framealpha=0.9,
        )
        self.ax.add_artist(cluster_legend)  # kept on-axes once the 2nd legend() call below replaces the "current" legend

        unique_candles = sorted(set(candles))
        candle_handles = [
            mlines.Line2D(
                [], [], color="#555555", marker=CANDLE_MARKERS[candle_count % len(CANDLE_MARKERS)],
                linestyle="None", markersize=8,
                label=FACTOR_LABELS["candles"].get(candle_count, f"{candle_count} candles"),
            )
            for candle_count in unique_candles
        ]
        self.ax.legend(
            handles=candle_handles, loc="upper right", title="Candles",
            fontsize=8, title_fontsize=8, framealpha=0.9,
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


class _AnalyticsFeatureWorker(QtCore.QThread):
    """One-shot background computation of the ensemble feature index for
    the analytics panel. Mirrors simulation_controller.py's
    _PrefetchWorker: fire-and-forget, kept alive by the owner (MainWindow)
    until its own finished_ok/error signal fires. Lives here rather than
    in simulation_controller.py because it's UI-adjacent analytics
    plumbing, not SimulationController's scenario-parameter/prefetch
    domain -- see that module's own "zero Qt-widget knowledge" framing."""

    finished_ok = QtCore.pyqtSignal(list)  # ScenarioFeatures list
    error = QtCore.pyqtSignal(str)

    def __init__(self, manifest, store, fps):
        super().__init__()
        self._manifest = manifest
        self._store = store
        self._fps = fps

    def run(self):
        try:
            features = build_feature_index(self._manifest, self._store, self._fps)
            self.finished_ok.emit(features)
        except Exception as e:  # noqa: BLE001 - never let a worker crash silently
            self.error.emit(f"Failed to build ensemble analytics: {e}")
