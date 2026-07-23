"""V6-M4 Context Panel data layer: gather_context() against a plain fake
app object (no QApplication needed -- context.py is Qt-free)."""

from __future__ import annotations

import context as ctx_mod
import devices as dv
import velocity as vel
import measure as mz
from zone_stats import Zone
from selection import Selection
from insight import Insight
from evidence_notebook import EvidenceNotebook
import graph_model as gm


class _Panel:
    """A minimal stand-in for any *_panel with the one live attribute
    context.py reads off it."""
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class _App:
    """A plain fake MainWindow: only the attributes context.py duck-types
    through are present, exactly mirroring graph_panel._gather()'s own
    `live(attr, sub)` closure."""
    def __init__(self, **panels):
        for k, v in panels.items():
            setattr(self, k, v)


def _device(scenario=0, position=(1.0, 1.0)):
    d = dv.Device(id="d1", name="TC-01", type="thermocouple", scenario=scenario, position=position)
    d.results = {"time_s": [0.0], "temperature_C": [20.0], "max_temperature_C": 20.0,
                "heating_rate_C_per_s": 0.0, "threshold_times_s": {}, "basis": "x"}
    return d


def _probe(scenario=0, position=(1.0, 1.0)):
    p = vel.VectorProbe(id="p1", name="VP-01", scenario=scenario, position=position)
    p.mark_gated("Requires the M-SIM cluster re-run")
    return p


class TestRelatedDevicesAndProbes:
    def test_filters_devices_by_scenario(self):
        app = _App(device_panel=_Panel(_devices=[_device(scenario=0), _device(scenario=1)]))
        out = ctx_mod.gather_context(app, Selection(scenario=0))
        assert len(out["devices"]) == 1 and out["devices"][0].scenario == 0

    def test_no_scenario_selected_returns_all(self):
        app = _App(device_panel=_Panel(_devices=[_device(scenario=0), _device(scenario=1)]))
        out = ctx_mod.gather_context(app, Selection())
        assert len(out["devices"]) == 2

    def test_filters_probes_by_scenario(self):
        app = _App(velocity_panel=_Panel(_probes=[_probe(scenario=0), _probe(scenario=2)]))
        out = ctx_mod.gather_context(app, Selection(scenario=2))
        assert len(out["probes"]) == 1 and out["probes"][0].scenario == 2

    def test_missing_panel_yields_empty_list_not_error(self):
        out = ctx_mod.gather_context(_App(), Selection(scenario=0))
        assert out["devices"] == [] and out["probes"] == []


class TestRelatedNotebook:
    def _notebook_with(self, location):
        nb = EvidenceNotebook()
        nb.add(Insight(statement="hot spot", time_s=5.0, location=location))
        return nb

    def test_no_point_returns_all_entries(self):
        app = _App(evidence_dock=_Panel(notebook=self._notebook_with((1.0, 1.0))))
        out = ctx_mod.gather_context(app, Selection(scenario=0))
        assert len(out["notebook"]) == 1

    def test_point_filters_by_proximity(self):
        app = _App(evidence_dock=_Panel(notebook=self._notebook_with((1.0, 1.0))))
        near = ctx_mod.gather_context(app, Selection(point=(1.1, 1.0)))
        far = ctx_mod.gather_context(app, Selection(point=(9.0, 9.0)))
        assert len(near["notebook"]) == 1
        assert len(far["notebook"]) == 0


class TestRelatedZonesAndMeasurements:
    def test_point_inside_zone_matches(self):
        z = Zone("doorway", 0.0, 2.0, 0.0, 2.0)
        app = _App(zone_panel=_Panel(_zones=[z]))
        inside = ctx_mod.gather_context(app, Selection(point=(1.0, 1.0)))
        outside = ctx_mod.gather_context(app, Selection(point=(9.0, 9.0)))
        assert len(inside["zones"]) == 1 and len(outside["zones"]) == 0

    def test_measurement_near_point_matches(self):
        m = mz.Measurement("probe", [(1.0, 1.0)])
        app = _App(measurement_panel=_Panel(_measurements=[m]))
        near = ctx_mod.gather_context(app, Selection(point=(1.05, 1.0)))
        far = ctx_mod.gather_context(app, Selection(point=(9.0, 9.0)))
        assert len(near["measurements"]) == 1 and len(far["measurements"]) == 0


class TestRelatedGraphNodes:
    def test_scenario_node_matches(self):
        g = gm.Graph()
        g.add(gm.Node("scenario:0", "scenario", "case_0", scenario=0))
        g.add(gm.Node("scenario:1", "scenario", "case_1", scenario=1))
        app = _App(graph_panel=_Panel(_graph=g))
        out = ctx_mod.gather_context(app, Selection(scenario=0))
        assert [n.id for n in out["graph_nodes"]] == ["scenario:0"]

    def test_no_graph_yet_is_empty(self):
        out = ctx_mod.gather_context(_App(), Selection(scenario=0))
        assert out["graph_nodes"] == []


class TestRelatedSessions:
    def test_session_referencing_scenario_is_found(self, tmp_path):
        import session_store
        from session import build_session_dict

        class DummyCell:
            def __init__(self, case_index):
                self.cell_type = "slice"
                self.case_index = case_index
                self.quantity_key = type("K", (), {"quantity": "TEMPERATURE"})()

        session = build_session_dict("grid_1x1", [DummyCell(3)], 0, 0, False, "fds_fire", False)
        session_store.save_session(str(tmp_path), session)
        app = _App(_sessions_dir=str(tmp_path))
        found = ctx_mod.gather_context(app, Selection(scenario=3))
        not_found = ctx_mod.gather_context(app, Selection(scenario=99))
        assert len(found["sessions"]) == 1
        assert len(not_found["sessions"]) == 0

    def test_no_scenario_selected_returns_no_sessions(self, tmp_path):
        app = _App(_sessions_dir=str(tmp_path))
        out = ctx_mod.gather_context(app, Selection())
        assert out["sessions"] == []

    def test_missing_sessions_dir_is_never_fatal(self):
        app = _App(_sessions_dir="/nonexistent/path/for/sure")
        out = ctx_mod.gather_context(app, Selection(scenario=0))
        assert out["sessions"] == []


class TestSchema:
    def test_every_value_is_always_a_list(self):
        out = ctx_mod.gather_context(_App(), Selection())
        for key in ("devices", "probes", "notebook", "zones", "measurements",
                   "graph_nodes", "sessions"):
            assert isinstance(out[key], list)
