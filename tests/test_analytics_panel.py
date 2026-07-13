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
