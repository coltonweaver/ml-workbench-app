"""The `workbench` helper: payloads, and publishing them through the kernel."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from workbench import graph_payload, landscape_payload, show_slice

SOLUTIONS = Path(__file__).resolve().parents[2] / "content" / "m01-backprop" / "solutions"


def load_solution(name: str):
    spec = importlib.util.spec_from_file_location(f"sol_{name}", SOLUTIONS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def Value():
    return load_solution("engine").Value


def test_graph_payload_is_topological(Value) -> None:
    a, b = Value(2.0), Value(-3.0)
    a.label = "a"
    L = a * b + a
    L.backward()
    p = graph_payload(L, title="L")
    assert p["kind"] == "graph" and p["has_backward"] is True
    assert len(p["nodes"]) == 4  # a, b, a*b, L
    assert p["root"] == len(p["nodes"]) - 1, "the root comes last in topological order"
    position = {node_id: i for i, node_id in enumerate(p["order"])}
    for child, parent in p["edges"]:
        assert position[child] < position[parent]
    a_node = next(n for n in p["nodes"] if n["label"] == "a")
    assert a_node["data"] == 2.0 and a_node["grad"] == -2.0  # dL/da = b + 1
    json.dumps(p, allow_nan=False)


def test_graph_before_backward(Value) -> None:
    p = graph_payload(Value(1.0) * 3)
    assert p["has_backward"] is False


def test_graph_size_limit(Value) -> None:
    x = Value(1.0)
    for _ in range(500):
        x = x + 1
    with pytest.raises(ValueError, match="more than"):
        graph_payload(x)


def test_landscape_1d_with_path() -> None:
    f = lambda w: (w - 7) ** 2  # noqa: E731
    p = landscape_payload(f, (0, 10), path=[10, 9.4, 8.92], n=11)
    assert p["dims"] == 1 and len(p["x"]) == 11 and p["line"][7] == 0.0
    assert p["path"][0] == [10.0, 9.0]
    json.dumps(p, allow_nan=False)


def test_landscape_2d_grid_orientation() -> None:
    f = lambda v: v[0] + 10 * v[1]  # noqa: E731
    p = landscape_payload(f, ((0, 1), (0, 2)), path=[[1, 2]], n=3)
    assert p["dims"] == 2
    assert p["grid"][0] == [0.0, 0.5, 1.0], "grid rows run along x at fixed y"
    assert p["grid"][2][0] == 20.0
    assert p["path"] == [[1.0, 2.0, 21.0]]


def test_non_finite_values_become_null() -> None:
    p = landscape_payload(lambda x: math.inf if x > 0 else math.nan if x < 0 else 1.0, (-1, 1), n=3)
    assert p["line"] == [None, 1.0, None]
    json.dumps(p, allow_nan=False)


def test_show_slice_restores_parameters(Value) -> None:
    params = [Value(0.5), Value(-1.0), Value(2.0)]
    loss = lambda: sum(((p - 1) ** 2 for p in params), Value(0.0))  # noqa: E731
    handle = show_slice(loss, params, span=0.5, n=5)
    assert [p.data for p in params] == [0.5, -1.0, 2.0]
    grid = handle.payload["grid"]
    assert grid[2][2] == pytest.approx(loss().data), "the centre of the slice is the current loss"
    assert handle.payload["path"] == [[0.0, 0.0, pytest.approx(loss().data)]]


def test_instruments_reach_the_protocol(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from test_sidecar import Sidecar

    sidecar = Sidecar(tmp_path / "ws")
    try:
        events = sidecar.request(op="execute", code=(
            "from workbench import show_landscape\n"
            "h = show_landscape(lambda w: (w - 7) ** 2, (0, 10), n=5)\n"
            "h.add(10)\n"
            "h.add(9.4)\n"
        ))
        shown = [e for e in events if e["type"] == "display_data"]
        updates = [e for e in events if e["type"] == "update_display_data"]
        assert len(shown) == 1 and len(updates) == 2, [e["type"] for e in events]
        mime = "application/vnd.workbench.landscape+json"
        assert shown[0]["data"][mime]["dims"] == 1
        assert updates[-1]["display_id"] == shown[0]["display_id"]
        assert len(updates[-1]["data"][mime]["path"]) == 2
    finally:
        sidecar.close()


def test_tensor_summary_shapes_and_stats() -> None:
    import numpy as np
    from workbench import tensor_summary

    s = tensor_summary([[1.0, 2.0], [3.0, float("nan")]], "t", None, note="n")
    assert s["shape"] == [2, 2] and s["count"] == 4 and s["non_finite"] == 1
    assert s["stats"]["min"] == 1.0 and s["stats"]["max"] == 3.0 and s["stats"]["mean"] == 2.0
    assert s["note"] == "n" and s["values"] == [1.0, 2.0, 3.0]
    a = tensor_summary(np.zeros((3, 4, 5)), "z")
    assert a["shape"] == [3, 4, 5] and a["stats"]["zeros"] == 60 and "values" not in a


def test_payloads_carry_explain_with_provenance() -> None:
    from workbench import graph_payload, landscape_payload

    class V:
        def __init__(self, data, prev=(), op=""):
            self.data, self.grad, self._prev, self._op = data, 0.0, prev, op

    g = graph_payload(V(3.0, (V(1.0), V(2.0)), "+"))
    values = g["explain"][0]
    assert values["shape"] == [3] and values["provenance"]["file"] == "test_instruments.py"
    assert "graph_payload" in values["provenance"]["code"]
    land = landscape_payload(lambda v: v[0] ** 2 + v[1] ** 2, ((-1, 1), (-2, 2)), n=5)
    surface = land["explain"][0]
    assert surface["shape"] == [5, 5] and surface["stats"]["min"] == 0.0
    assert surface["axes"] == ["x in [-1, 1]", "y in [-2, 2]"]
