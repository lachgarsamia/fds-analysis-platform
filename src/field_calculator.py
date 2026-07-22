"""Field Calculator (V6-M1): a deterministic scientific expression engine.

A researcher writes an expression over existing quantities -- ``Temperature - 20``,
``gradient(Temperature)``, ``rate(Temperature)``, ``0.5*1.2*Velocity**2`` -- and
gets a new field that behaves like any registered quantity: it appears in the
registry, the QuantityProvider computes it, and it plots/exports like the rest.

**Safety is the whole point.** Expressions are parsed with Python's ``ast`` and
validated against a strict whitelist; they are *never* executed with ``eval`` or
``exec``. Only arithmetic/comparison operators, a fixed set of numeric functions,
numeric literals, and *registered quantity names* are allowed. Imports, attribute
access, and any other call are rejected before evaluation.

Every calculated field stores its expression (the provenance/basis), its
dependencies, and a unit, so results are reproducible, traceable, and exportable.

Pure NumPy, Qt-free. Extends derived_quantities/registry rather than replacing
them: a calculated field is registered *into* QUANTITY_REGISTRY (no parallel
representation) and the QuantityProvider evaluates it via evaluate().
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from registry import QUANTITY_REGISTRY, QuantityInfo, get_quantity


class CalculatorError(ValueError):
    """Raised on an invalid or unsafe expression. The message is user-facing."""


# Whitelisted numeric functions. min/max are elementwise (numpy semantics);
# gradient/rate are field operators that need the (t, z, x) stack and fps.
_ALLOWED_FUNCS = {"abs", "sqrt", "clip", "log", "exp", "min", "max",
                  "where", "gradient", "rate"}

# Whitelisted AST node types. Comparison operators are included solely so the
# whitelisted where() function is usable (e.g. where(Temperature > 300, 1, 0));
# they are pure/deterministic. Anything not here is rejected.
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Compare, ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq,
)


# --- identifier <-> registry key mapping ---------------------------------
def _ident_to_key() -> dict:
    """Map a friendly identifier (spaces -> underscores, case-insensitive) to a
    registry key, e.g. Temperature -> TEMPERATURE, Dynamic_Pressure ->
    DYNAMIC PRESSURE. Only non-gated quantities are usable as inputs."""
    out = {}
    for key, q in QUANTITY_REGISTRY.items():
        if not q.gated:
            out[key.replace(" ", "_").upper()] = key
    return out


def _key_for_ident(ident: str) -> str:
    key = _ident_to_key().get(ident.upper())
    if key is None:
        raise CalculatorError(f"unknown quantity: '{ident}'")
    return key


# --- parse / validate / dependencies -------------------------------------
def _parse(expression: str) -> ast.Expression:
    try:
        return ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"could not parse expression: {e.msg}") from e


def validate(expression: str) -> None:
    """Raise CalculatorError unless `expression` is a safe field expression:
    only whitelisted nodes/operators, only whitelisted function calls, and only
    registered non-gated quantity names as variables."""
    tree = _parse(expression)
    idents = _ident_to_key()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise CalculatorError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
                raise CalculatorError("only whitelisted functions may be called")
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_FUNCS:
                continue                       # a function reference (checked above)
            if node.id.upper() not in idents:
                raise CalculatorError(
                    f"'{node.id}' is not a known quantity (use a registered "
                    "quantity name or a whitelisted function)")


def dependencies(expression: str) -> List[str]:
    """The registry keys the expression depends on (sorted, unique). Assumes the
    expression already validated."""
    tree = _parse(expression)
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_FUNCS:
            keys.add(_key_for_ident(node.id))
    return sorted(keys)


# --- evaluation ----------------------------------------------------------
def _fn_min(a, b):
    return np.minimum(a, b)


def _fn_max(a, b):
    return np.maximum(a, b)


def _fn_gradient(field, fps):
    """Per-frame in-plane spatial gradient magnitude of a (t, z, x) field."""
    a = np.asarray(field, dtype=float)
    gz = np.gradient(a, axis=1) if a.shape[1] >= 2 else np.zeros_like(a)
    gx = np.gradient(a, axis=2) if a.shape[2] >= 2 else np.zeros_like(a)
    return np.sqrt(gz ** 2 + gx ** 2)


def _fn_rate(field, fps):
    """Temporal derivative d(field)/dt of a (t, z, x) field."""
    a = np.asarray(field, dtype=float)
    return np.gradient(a, axis=0) * fps if a.shape[0] >= 2 else np.zeros_like(a)


_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
           ast.Div: np.divide, ast.Pow: np.power}
_CMPOPS = {ast.Gt: np.greater, ast.Lt: np.less, ast.GtE: np.greater_equal,
           ast.LtE: np.less_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}


def evaluate(expression: str, resolve: Callable[[str], np.ndarray], fps: int = 1) -> np.ndarray:
    """Evaluate a validated expression. `resolve(key)` returns the field array
    for a registered quantity key. Never executes raw code -- it walks the AST
    and applies whitelisted numpy operations only."""
    validate(expression)
    tree = _parse(expression)
    env = {ident: resolve(key) for key, ident in
           ((_key_for_ident(i), i) for i in _dep_idents(tree))}

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise CalculatorError("only numeric literals are allowed")
            return float(node.value)
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = _ev(node.operand)
            return -operand if isinstance(node.op, ast.USub) else +operand
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise CalculatorError("chained comparisons are not supported")
            return _CMPOPS[type(node.ops[0])](_ev(node.left), _ev(node.comparators[0]))
        if isinstance(node, ast.Call):
            name = node.func.id
            args = [_ev(a) for a in node.args]
            return _apply(name, args, fps)
        raise CalculatorError(f"disallowed syntax: {type(node).__name__}")

    return np.asarray(_ev(tree), dtype=float)


def _dep_idents(tree) -> list:
    return sorted({n.id for n in ast.walk(tree)
                   if isinstance(n, ast.Name) and n.id not in _ALLOWED_FUNCS})


def _apply(name, args, fps):
    if name == "abs":
        return np.abs(args[0])
    if name == "sqrt":
        return np.sqrt(args[0])
    if name == "log":
        return np.log(args[0])
    if name == "exp":
        return np.exp(args[0])
    if name == "clip":
        return np.clip(args[0], args[1], args[2])
    if name == "where":
        return np.where(args[0], args[1], args[2])
    if name == "min":
        return _fn_min(args[0], args[1])
    if name == "max":
        return _fn_max(args[0], args[1])
    if name == "gradient":
        return _fn_gradient(args[0], fps)
    if name == "rate":
        return _fn_rate(args[0], fps)
    raise CalculatorError(f"unknown function: {name}")     # unreachable after validate


# --- unit inference (best-effort) ----------------------------------------
def infer_unit(expression: str) -> str:
    """A best-effort unit for the result, or 'derived' when ambiguous. Honest,
    not a full dimensional analysis: a single-quantity arithmetic expression
    carries that quantity's unit; gradient/rate append /m and /s."""
    deps = dependencies(expression)
    if len(deps) != 1:
        return "derived"
    base = get_quantity(deps[0]).unit
    tree = _parse(expression)
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    if "gradient" in calls:
        return f"{base}/m"
    if "rate" in calls:
        return f"{base}/s"
    if calls & {"log", "exp"}:
        return "derived"
    return base


# --- the calculated-field object + registry integration ------------------
@dataclass(frozen=True)
class CalculatedField:
    name: str                       # display name and registry key, e.g. "Temperature Rise"
    expression: str
    unit: str
    dependencies: Tuple[str, ...]
    basis: str

    def to_dict(self) -> dict:
        return {"name": self.name, "expression": self.expression, "unit": self.unit}

    @classmethod
    def from_dict(cls, d: dict) -> "CalculatedField":
        return make_field(d["name"], d["expression"], d.get("unit"))


def make_field(name: str, expression: str, unit: str = None) -> CalculatedField:
    """Build (and validate) a CalculatedField from a name + expression. Raises
    CalculatorError on an invalid/unsafe expression or empty name."""
    name = (name or "").strip()
    if not name:
        raise CalculatorError("a calculated field needs a name")
    validate(expression)
    deps = dependencies(expression)
    if not deps:
        raise CalculatorError("expression must reference at least one quantity")
    return CalculatedField(
        name=name, expression=expression.strip(),
        unit=(unit.strip() if unit and unit.strip() else infer_unit(expression)),
        dependencies=tuple(deps),
        basis=f"calculated field: {expression.strip()} (deps: {', '.join(deps)})")


# Calculated fields registered into QUANTITY_REGISTRY this session.
CALCULATED: dict = {}


def register(field: CalculatedField) -> str:
    """Register a calculated field *into* the quantity registry (no parallel
    representation). It inherits its primary dependency's colormap/scale so it
    displays sensibly. Returns the registry key (its name)."""
    primary = get_quantity(field.dependencies[0])
    QUANTITY_REGISTRY[field.name] = QuantityInfo(
        name=field.name, label=field.name, unit=field.unit,
        cmap=primary.cmap, vmin=primary.vmin,
        slider_min=primary.slider_min, slider_max=primary.slider_max,
        slider_default=primary.slider_default, kind="derived",
        interpretation=field.basis, expression=field.expression, calculated=True)
    CALCULATED[field.name] = field
    return field.name


def unregister(name: str) -> None:
    CALCULATED.pop(name, None)
    q = QUANTITY_REGISTRY.get(name)
    if q is not None and getattr(q, "calculated", False):
        QUANTITY_REGISTRY.pop(name, None)


def clear() -> None:
    """Remove every calculated field from the registry (e.g. before restoring a
    session, or for test isolation)."""
    for name in list(CALCULATED):
        unregister(name)


def get_field(name: str):
    return CALCULATED.get(name)


def all_fields() -> list:
    return list(CALCULATED.values())
