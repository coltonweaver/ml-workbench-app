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


# ---- language-model instruments -------------------------------------------------

def test_training_dashboard_downsamples_and_flushes(monkeypatch) -> None:
    import numpy as np
    from workbench import lm

    monkeypatch.setattr(lm, "MAX_POINTS", 10)
    published = []
    monkeypatch.setattr(lm.TrainingDashboard, "update", lambda self, payload=None: published.append(1))
    dash = lm.show_training("t", every=1e9)
    for step in range(25):
        dash.log(step, loss=np.float64(3.0 - step * 0.1), lr=0.1, tokens=64,
                 grad_norms={"W": 2.0}, accuracy=0.5)
    assert len(dash.payload["points"]) <= 10 and dash.payload["stride"] > 1
    assert dash.payload["points"][-1]["step"] == 24, "the newest point is always kept"
    assert dash.payload["points"][-1]["metrics"] == {"accuracy": 0.5}
    assert "tokens_per_sec" in dash.payload["points"][-1]
    published.clear()
    dash.done()
    assert published and dash.payload["done"] is True
    assert dash.payload["explain"][0]["name"] == "training loss"


def test_grad_norms_accepts_arrays_values_and_names() -> None:
    import numpy as np
    from workbench import grad_norms

    class P:
        def __init__(self, g):
            self.grad = g
    norms = grad_norms({"a": P(np.array([3.0, 4.0])), "b": P(0.5), "none": P(None)})
    assert norms == {"a": 5.0, "b": 0.5}
    assert grad_norms([P(np.ones(4))]) == {"param 0": 2.0}


def test_embedding_payload_pca() -> None:
    import numpy as np
    from workbench.lm import embedding_payload, token_class

    rng = np.random.default_rng(0)
    x = np.zeros((5, 3))
    x[:, 0] = np.arange(5) * 10  # all variance along one axis
    x[:, 1] = rng.normal(size=5) * 0.01
    p = embedding_payload(x, list(".abcd"))
    assert p["kind"] == "embedding" and len(p["points"]) == 5
    assert p["explained_variance"][0] > 0.99
    assert p["classes"][:2] == ["special", "vowel"]
    assert token_class("Hello") == "upper" and token_class(" ") == "space" and token_class("7") == "digit"
    with pytest.raises(ValueError):
        embedding_payload(np.zeros((3, 2)), ["a"])


def test_attention_payload_shapes_and_rerun() -> None:
    import numpy as np
    from workbench import lm

    a = np.full((2, 3, 3), 1 / 3)
    p = lm.attention_payload(a, ["a", "b", "c"])
    assert len(p["layers"]) == 1 and len(p["layers"][0]["heads"]) == 2
    with pytest.raises(ValueError):
        lm.attention_payload(a, ["a", "b"])
    calls = []

    def model(text):
        calls.append(text)
        n = len(text)
        return list(text), np.tril(np.ones((1, 2, n, n))) / np.arange(1, n + 1)[:, None]

    h = lm.show_attention(model, "hey", title="t")
    callback = h.payload["callback"]
    lm.rerun(callback, "hello")
    assert calls == ["hey", "hello"] and h.payload["tokens"] == list("hello")
    assert h.payload["prompt"] == "hello"
    with pytest.raises(KeyError):
        lm.rerun("attention-999", "x")


def test_matrix_payload() -> None:
    import numpy as np
    from workbench.lm import matrix_payload

    p = matrix_payload(np.array([[1, 2], [3, np.inf]]), ["r0", "r1"], ["c0", "c1"], log_scale=True)
    assert p["values"] == [[1.0, 2.0], [3.0, None]] and p["log_scale"] is True
    with pytest.raises(ValueError):
        matrix_payload(np.zeros((200, 2)))


def test_as_array_handles_engine_tensors_and_values(Value) -> None:
    import numpy as np
    from workbench.lm import as_array

    class T:
        def __init__(self, d):
            self.data = np.asarray(d)
    assert as_array(T([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert as_array([Value(1.0), Value(2.0)]).tolist() == [1.0, 2.0]
    assert as_array([[1, 2], [3, 4]]).shape == (2, 2)


def test_images_payload_layouts() -> None:
    import base64
    import numpy as np
    from workbench.vision import images_payload

    p = images_payload(np.arange(12, dtype=float).reshape(1, 1, 3, 4), labels=["a"])
    img = p["images"][0]
    assert (img["width"], img["height"], img["channels"]) == (4, 3, 1)
    px = np.frombuffer(base64.b64decode(img["pixels"]), dtype=np.uint8)
    assert px[0] == 0 and px[-1] == 255 and len(px) == 12
    assert images_payload(np.zeros((5, 3, 8, 8)))["images"][0]["channels"] == 3
    assert len(images_payload(np.zeros((8, 8)))["images"]) == 1
    assert len(images_payload(np.zeros((4, 2, 5, 5)))["images"]) == 8, "filters show each input channel"
    g = images_payload(np.stack([np.zeros((2, 2)), np.ones((2, 2))]), normalize="global")
    assert base64.b64decode(g["images"][0]["pixels"]) == bytes([0] * 4)
    with pytest.raises(ValueError):
        images_payload(np.zeros((3, 4, 4)), labels=["x"])


def test_shapes_dataset() -> None:
    import numpy as np
    from workbench.data import SHAPE_CLASSES, shapes

    X, Y = shapes(120, seed=1)
    assert X.shape == (120, 1, 32, 32) and X.dtype == np.float32 and Y.dtype == np.int64
    assert 0 <= X.min() and X.max() <= 1 and set(Y) <= set(range(len(SHAPE_CLASSES)))
    X2, Y2 = shapes(120, seed=1)
    assert np.array_equal(X, X2) and np.array_equal(Y, Y2), "same seed, same data"
