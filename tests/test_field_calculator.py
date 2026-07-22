"""V6-M1: the Field Calculator safe expression engine."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import field_calculator as fc  # noqa: E402


def _resolver(**arrays):
    """A resolver mapping registry keys to arrays for evaluate()."""
    return lambda key: arrays[key]


class TestExpressionMath:
    def test_temperature_minus_scalar(self):
        t = np.array([[[20.0, 120.0], [320.0, 20.0]]])   # (1,2,2)
        out = fc.evaluate("Temperature - 20", _resolver(TEMPERATURE=t))
        np.testing.assert_allclose(out, t - 20)

    def test_arithmetic_and_power(self):
        v = np.array([[[0.0, 2.0], [4.0, 0.0]]])
        out = fc.evaluate("0.5 * 1.2 * Velocity ** 2", _resolver(VELOCITY=v))
        np.testing.assert_allclose(out, 0.5 * 1.2 * v ** 2)

    def test_gradient_matches_synthetic(self):
        # a field with a pure horizontal ramp -> constant in-plane gradient
        z, x = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
        field = (2.0 * x).astype(float)[None, :, :]        # dT/dx = 2, dT/dz = 0
        out = fc.evaluate("gradient(Temperature)", _resolver(TEMPERATURE=field))
        # interior gradient magnitude == 2 (np.gradient central difference)
        assert out[0, 1, 1] == pytest.approx(2.0)

    def test_rate_matches_temporal_derivative(self):
        # a field that ramps linearly in time at 10 units/frame; fps=2 -> 20/s
        frames = np.stack([np.full((2, 2), 10.0 * i) for i in range(5)])
        out = fc.evaluate("rate(Temperature)", _resolver(TEMPERATURE=frames), fps=2)
        np.testing.assert_allclose(out, np.full_like(frames, 20.0))

    def test_where_needs_no_python_execution(self):
        t = np.array([[[20.0, 400.0], [350.0, 50.0]]])
        out = fc.evaluate("where(Temperature > 300, 1, 0)", _resolver(TEMPERATURE=t))
        np.testing.assert_allclose(out, [[[0, 1], [1, 0]]])

    def test_multi_dependency(self):
        t = np.array([[[100.0, 200.0]]])
        v = np.array([[[1.0, 2.0]]])
        out = fc.evaluate("Temperature * Velocity", _resolver(TEMPERATURE=t, VELOCITY=v))
        np.testing.assert_allclose(out, t * v)


class TestSecurity:
    @pytest.mark.parametrize("expr", [
        "import os",
        "__import__('os')",
        "Temperature.__class__",
        "os.system('ls')",
        "open('x')",
        "eval('1+1')",
        "exec('x=1')",
        "().__class__.__bases__",
        "Temperature; import os",
        "lambda: 1",
        "[x for x in range(3)]",
    ])
    def test_unsafe_expressions_are_rejected(self, expr):
        with pytest.raises(fc.CalculatorError):
            fc.validate(expr)

    def test_unknown_quantity_rejected(self):
        with pytest.raises(fc.CalculatorError):
            fc.validate("NotAQuantity + 1")

    def test_non_whitelisted_function_rejected(self):
        with pytest.raises(fc.CalculatorError):
            fc.validate("sin(Temperature)")     # sin is not whitelisted

    def test_evaluate_revalidates(self):
        # evaluate() must itself reject unsafe input even if called directly
        with pytest.raises(fc.CalculatorError):
            fc.evaluate("__import__('os')", _resolver())


class TestDependenciesAndUnits:
    def test_dependencies_extracted(self):
        assert fc.dependencies("Temperature - 20") == ["TEMPERATURE"]
        assert fc.dependencies("Temperature * Velocity") == ["TEMPERATURE", "VELOCITY"]

    def test_unit_inference(self):
        assert fc.infer_unit("Temperature - 20") == "°C"
        assert fc.infer_unit("gradient(Temperature)") == "°C/m"
        assert fc.infer_unit("rate(Temperature)") == "°C/s"
        assert fc.infer_unit("Temperature * Velocity") == "derived"


class TestCalculatedFieldRegistration:
    def teardown_method(self):
        fc.clear()

    def test_make_and_register_appears_in_registry(self):
        from registry import QUANTITY_REGISTRY, get_quantity
        field = fc.make_field("Temperature Rise", "Temperature - 20")
        assert field.dependencies == ("TEMPERATURE",) and field.unit == "°C"
        assert "calculated field" in field.basis
        fc.register(field)
        assert "Temperature Rise" in QUANTITY_REGISTRY
        q = get_quantity("Temperature Rise")
        assert q.calculated and q.expression == "Temperature - 20" and q.kind == "derived"

    def test_make_field_rejects_empty_name_and_no_deps(self):
        with pytest.raises(fc.CalculatorError):
            fc.make_field("", "Temperature - 20")
        with pytest.raises(fc.CalculatorError):
            fc.make_field("Const", "1 + 2")     # no quantity dependency

    def test_serialization_roundtrip(self):
        field = fc.make_field("Thermal Gradient", "gradient(Temperature)")
        back = fc.CalculatedField.from_dict(field.to_dict())
        assert back.name == field.name and back.expression == field.expression

    def test_clear_removes_from_registry(self):
        from registry import QUANTITY_REGISTRY
        fc.register(fc.make_field("X", "Temperature * 2"))
        assert "X" in QUANTITY_REGISTRY
        fc.clear()
        assert "X" not in QUANTITY_REGISTRY and fc.all_fields() == []
