"""Compare & Discover workspace (Analysis section consolidation Phase 3),
an Analysis-page tab.

A thin QTabWidget wrapper, not a rewrite: hosts the existing pairwise
comparison (AdvancedComparePanel), parallel-coordinates
(ParallelCoordinatesPanel), ensemble-envelope (EnsemblePanel), and
PCA/clustering (AnalyticsPanelDock's re-hosted widget) panels as four
modes of one "how do scenarios compare?" workspace, instead of four
same-level tabs previously scattered across different groups. Every
child's own construction, store access, lazy-load convention, and
SelectionBus wiring is completely unchanged -- only the tab-level
presentation is consolidated, the same pattern already proven for
Hazard & Tenability (hazard_tenability_panel.py) and Assistant + Ask
(assistant_query_panel.py).

Any of the four children may be absent (a generic non-factorial study has
no PCA/clustering; fewer than 2 scenarios means no pairwise comparison) --
only supplied children get a tab, same "only supplied surfaces get a tab"
rule the outer AnalysisPage already follows.
"""

from __future__ import annotations

from PyQt5 import QtWidgets


class CompareDiscoverPanel(QtWidgets.QWidget):
    def __init__(self, pairwise: QtWidgets.QWidget = None,
                 parallel: QtWidgets.QWidget = None,
                 ensemble: QtWidgets.QWidget = None,
                 clustering: QtWidgets.QWidget = None, parent=None):
        super().__init__(parent)
        self.pairwise_widget = pairwise
        self.parallel_widget = parallel
        self.ensemble_widget = ensemble
        self.clustering_widget = clustering

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        for widget, label in ((pairwise, "Pairwise"),
                             (parallel, "Parallel coordinates"),
                             (ensemble, "Ensemble"),
                             (clustering, "Clustering")):
            if widget is not None:
                self.tabs.addTab(widget, label)
        layout.addWidget(self.tabs, 1)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        """All children load on first show, not just the one currently in
        view -- switching modes later must never reveal a blank panel.
        The re-hosted AnalyticsPanelDock widget has no ensure_loaded of its
        own (its one-shot background load is triggered by main_window at
        the Analysis-page level, unchanged by this nesting) -- guarded."""
        for widget in (self.pairwise_widget, self.parallel_widget,
                      self.ensemble_widget, self.clustering_widget):
            if widget is not None and hasattr(widget, "ensure_loaded"):
                widget.ensure_loaded()
