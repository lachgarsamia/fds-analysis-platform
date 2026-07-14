import pytest

from analytics.features import ScenarioFeatures, CURVE_POINTS
from analytics_panel import AnalyticsPanelDock


def _fake_features(n=6):
    features = []
    for i in range(n):
        candles = i % 2
        # Two obviously-separated blobs in curve-space, so clustering has
        # a real signal to find rather than being at the mercy of noise.
        base = 0.0 if i < n // 2 else 50.0
        features.append(ScenarioFeatures(
            case_index=i,
            folder=f"scenario_{i}",
            candles=candles, door=0, vod=0, voc=0,
            max_temp_curve=[base + i] * CURVE_POINTS,
            hot_area_fraction_curve=[0.1] * CURVE_POINTS,
            spatial_mean_curve=[base] * CURVE_POINTS,
            time_to_100c_s=1.0, time_to_300c_s=2.0, time_to_600c_s=None,
        ))
    return features


class TestAnalyticsPanelDock:
    def test_builds_and_plots_without_crashing(self, qapp):
        dock = AnalyticsPanelDock(_fake_features())
        assert dock.ax is not None
        assert len(dock.ax.collections) > 0, "scatter should have added at least one collection"

    def test_status_label_reports_scenario_count_and_alignment(self, qapp):
        dock = AnalyticsPanelDock(_fake_features())
        assert "6 scenarios" in dock.status_label.text()
        assert "2 clusters" in dock.status_label.text()

    def test_empty_features_does_not_crash(self, qapp):
        dock = AnalyticsPanelDock([])
        assert "No scenarios" in dock.status_label.text()

    def test_hover_near_a_point_updates_status_label_with_its_scenario(self, qapp):
        dock = AnalyticsPanelDock(_fake_features())

        class FakeEvent:
            inaxes = dock.ax
            xdata = dock._coords[0, 0]
            ydata = dock._coords[0, 1]

        dock._on_hover(FakeEvent())
        assert dock._case_indices[0] is not None
        expected_folder = next(f.folder for f in dock._features if f.case_index == dock._case_indices[0])
        assert expected_folder in dock.status_label.text()

    def test_hover_far_from_any_point_does_not_change_label(self, qapp):
        dock = AnalyticsPanelDock(_fake_features())
        dock.status_label.setText("unchanged")

        class FakeEventFar:
            inaxes = dock.ax
            xdata = 10_000.0
            ydata = 10_000.0

        dock._on_hover(FakeEventFar())
        assert dock.status_label.text() == "unchanged"

    def test_hover_outside_axes_is_ignored(self, qapp):
        dock = AnalyticsPanelDock(_fake_features())
        dock.status_label.setText("unchanged")

        class FakeEventOutside:
            inaxes = None
            xdata = None
            ydata = None

        dock._on_hover(FakeEventOutside())
        assert dock.status_label.text() == "unchanged"

    def test_click_near_a_point_emits_scenario_activated(self, qapp):
        dock = AnalyticsPanelDock(_fake_features())
        received = []
        dock.scenario_activated.connect(received.append)

        class FakeEvent:
            inaxes = dock.ax
            xdata = dock._coords[2, 0]
            ydata = dock._coords[2, 1]

        dock._on_click(FakeEvent())
        assert received == [dock._case_indices[2]]

    def test_click_far_from_any_point_emits_nothing(self, qapp):
        dock = AnalyticsPanelDock(_fake_features())
        received = []
        dock.scenario_activated.connect(received.append)

        class FakeEventFar:
            inaxes = dock.ax
            xdata = 10_000.0
            ydata = 10_000.0

        dock._on_click(FakeEventFar())
        assert received == []

    def test_marker_assigned_per_candle_count_not_uniform(self, qapp):
        dock = AnalyticsPanelDock(_fake_features())
        paths = [c.get_paths() for c in dock.ax.collections]
        # Each scatter() call (one per scenario) creates its own
        # PathCollection with one marker path -- confirm at least two
        # distinct marker shapes were used across the 6 (candles 0/1) scenarios.
        markers_used = {tuple(p[0].vertices.round(3).flatten()) for p in paths if p}
        assert len(markers_used) >= 2

    def test_has_a_title(self, qapp):
        dock = AnalyticsPanelDock(_fake_features())
        assert dock.ax.get_title() != ""

    def test_axis_labels_include_real_explained_variance_percentages(self, qapp):
        dock = AnalyticsPanelDock(_fake_features())
        xlabel, ylabel = dock.ax.get_xlabel(), dock.ax.get_ylabel()
        assert xlabel.startswith("PC1 (") and "% variance explained" in xlabel
        assert ylabel.startswith("PC2 (") and "% variance explained" in ylabel
        # Not a placeholder -- extract the number and confirm it's a real,
        # non-trivial percentage (the two obvious blobs in _fake_features
        # are separated along a dominant axis, so PC1 should explain a lot).
        pct = float(xlabel.split("(")[1].split("%")[0])
        assert pct > 10.0

    def test_has_cluster_and_candle_legends(self, qapp):
        dock = AnalyticsPanelDock(_fake_features())
        legends = dock.ax.findobj(match=lambda a: hasattr(a, "get_texts") and callable(a.get_texts))
        # Two separate legend() calls (_add_legends) -- one for cluster
        # color, one for candle-count marker shape -- both must survive on
        # the axes (add_artist() for the first, so the second legend()
        # call doesn't silently remove it).
        titles = {leg.get_title().get_text() for leg in legends if hasattr(leg, "get_title")}
        assert "Cluster" in titles
        assert "Candles" in titles

    def test_candle_legend_uses_plain_language_labels(self, qapp):
        dock = AnalyticsPanelDock(_fake_features())
        legends = [a for a in dock.ax.findobj(match=lambda a: hasattr(a, "get_texts") and callable(a.get_texts))
                   if hasattr(a, "get_title") and a.get_title().get_text() == "Candles"]
        assert len(legends) == 1
        labels = {t.get_text() for t in legends[0].get_texts()}
        # FACTOR_LABELS wording ("1 candle"/"2 candles"), same as the
        # experiment browser -- not a bare "0"/"1" factor index.
        assert labels <= {"1 candle", "2 candles"}
        assert any("candle" in label for label in labels)

    def test_figure_has_a_plain_language_caption(self, qapp):
        dock = AnalyticsPanelDock(_fake_features())
        texts = [t.get_text() for t in dock.canvas.fig.texts]
        assert any("6 scenarios" in t and "match candle count" in t for t in texts)
