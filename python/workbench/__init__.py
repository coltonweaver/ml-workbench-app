"""Instruments: send what your code computed to the app's visual widgets.

Use these from the kernel (Run, or code you execute there)::

    from workbench import show_graph, show_landscape, show_slice

    show_graph(loss)                                 # the computation graph below loss
    show_landscape(f, (0, 10), path=descend(...))    # f of one number
    h = show_landscape(f, ((-6, 6), (-6, 6)))        # f of a list of two numbers
    h.add([x, y])                                    # extend the path live
    show_slice(lambda: loss(model), model.parameters())

Everything is published as a Jupyter display with a ``application/vnd.workbench.*``
MIME type. The app draws it in the Instruments tab, and a widget updates in
place when its handle publishes again.
"""

from __future__ import annotations

import inspect
import math
import os
import random
from collections.abc import Callable, Iterable, Sequence
from typing import Any

GRAPH_MIME = "application/vnd.workbench.graph+json"
LANDSCAPE_MIME = "application/vnd.workbench.landscape+json"
MAX_GRAPH_NODES = 400

__all__ = ["show_graph", "show_landscape", "show_slice", "graph_payload", "landscape_payload",
           "tensor_summary"]

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---- publishing -------------------------------------------------------------

class Handle:
    """Returned by every show_* call. ``update`` redraws the same widget."""

    def __init__(self, mime: str, payload: dict[str, Any]) -> None:
        self.mime = mime
        self.payload = payload
        self._display = _publish(mime, payload, handle=None)

    def update(self, payload: dict[str, Any] | None = None) -> None:
        if payload is not None:
            self.payload = payload
        _publish(self.mime, self.payload, handle=self._display)


def _publish(mime: str, payload: dict[str, Any], handle: Any) -> Any:
    try:
        from IPython.display import display
    except ImportError:  # pragma: no cover - only outside the kernel
        return None
    data = {mime: payload, "text/plain": f"<{payload.get('kind')}: {payload.get('title') or ''}>"}
    if handle is None:
        return display(data, raw=True, display_id=True)
    handle.update(data, raw=True)
    return handle


def _provenance() -> dict[str, Any] | None:
    """Where the show_* call came from: the first frame outside this package."""
    for frame in inspect.stack(context=1)[1:]:
        filename = os.path.abspath(frame.filename)
        if os.path.dirname(filename) == _PACKAGE_DIR:
            continue
        line = (frame.code_context or [""])[0].strip()
        return {"file": os.path.basename(frame.filename), "line": frame.lineno,
                "function": frame.function, "code": line}
    return None


def _flatten(values: Any) -> tuple[list[int], list[float]]:
    """Shape and finite entries of a nested list, array or Value-likes."""
    if hasattr(values, "shape") and hasattr(values, "ravel"):
        shape = [int(d) for d in values.shape]
        flat = [_num(v) for v in values.ravel().tolist()]
        return shape, [v for v in flat if v is not None]
    shape: list[int] = []
    probe = values
    while isinstance(probe, (list, tuple)):
        shape.append(len(probe))
        probe = probe[0] if probe else None
    flat: list[float] = []

    def walk(v: Any) -> None:
        if isinstance(v, (list, tuple)):
            for item in v:
                walk(item)
        else:
            n = _num(v)
            if n is not None:
                flat.append(n)

    walk(values)
    return shape, flat


def tensor_summary(values: Any, name: str, provenance: dict[str, Any] | None = None,
                   **extra: Any) -> dict[str, Any]:
    """What "Explain this tensor" sends to the tutor: shape, statistics and
    where the tensor came from. Works on arrays, nested lists and Values."""
    shape, flat = _flatten(values)
    count = 1
    for d in shape:
        count *= d
    summary: dict[str, Any] = {"name": name, "shape": shape, "count": count,
                               "non_finite": count - len(flat)}
    if flat:
        mean = sum(flat) / len(flat)
        var = sum((v - mean) ** 2 for v in flat) / len(flat)
        summary["stats"] = {
            "min": min(flat), "max": max(flat), "mean": mean, "std": math.sqrt(var),
            "zeros": sum(1 for v in flat if v == 0.0),
        }
        if len(flat) <= 16:
            summary["values"] = flat
    if provenance is not None:
        summary["provenance"] = provenance
    summary.update(extra)
    return summary


def _num(x: Any) -> float | None:
    """A JSON-safe float: NaN and infinities become null."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        try:
            v = float(x.data)  # a Value
        except (AttributeError, TypeError, ValueError):
            return None
    return v if math.isfinite(v) else None


# ---- computation graph -------------------------------------------------------

def graph_payload(root: Any, title: str | None = None) -> dict[str, Any]:
    """Serialize the graph below ``root``.

    Works with any Value-like object that has ``data``, ``grad``, ``_prev``
    and ``_op`` (the engine from module 1). An optional ``label`` attribute is
    shown on the node.
    """
    order: list[Any] = []
    seen: set[int] = set()
    stack: list[tuple[Any, bool]] = [(root, False)]
    # Iterative post-order DFS: children before parents, like backward().
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        if len(seen) > MAX_GRAPH_NODES:
            raise ValueError(
                f"the graph has more than {MAX_GRAPH_NODES} nodes; show a smaller expression"
            )
        stack.append((node, True))
        for child in getattr(node, "_prev", ()):
            if id(child) not in seen:
                stack.append((child, False))
    ids = {id(n): i for i, n in enumerate(order)}
    nodes = [
        {
            "id": ids[id(n)],
            "op": str(getattr(n, "_op", "") or ""),
            "label": str(getattr(n, "label", "") or ""),
            "data": _num(getattr(n, "data", None)),
            "grad": _num(getattr(n, "grad", 0.0)),
        }
        for n in order
    ]
    edges = [
        [ids[id(child)], ids[id(n)]]
        for n in order
        for child in getattr(n, "_prev", ())
    ]
    root_grad = _num(getattr(root, "grad", 0.0)) or 0.0
    where = _provenance()
    return {
        "explain": [
            tensor_summary([n["data"] for n in nodes], "node values", where,
                           note="every value in the graph, inputs first, root last"),
            tensor_summary([n["grad"] for n in nodes], "node gradients", where,
                           note="d(root)/d(node); all zero until root.backward() runs"),
        ],
        "kind": "graph",
        "title": title,
        "nodes": nodes,
        "edges": edges,
        "order": list(range(len(order))),  # ids are assigned in topological order
        "root": ids[id(root)],
        "has_backward": root_grad != 0.0,
    }


def show_graph(root: Any, title: str | None = None) -> Handle:
    """Draw the computation graph below ``root``. Call ``root.backward()``
    first to see gradients as well as values."""
    return Handle(GRAPH_MIME, graph_payload(root, title))


# ---- loss landscape ----------------------------------------------------------

Range = tuple[float, float]


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def landscape_payload(
    f: Callable[..., Any],
    ranges: Range | tuple[Range, Range],
    path: Iterable[Any] | None = None,
    n: int | None = None,
    title: str | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate ``f`` over a 1D range ``(lo, hi)`` or a 2D box
    ``((xlo, xhi), (ylo, yhi))``. In 2D, ``f`` takes a list ``[x, y]``."""
    two_d = isinstance(ranges[0], (tuple, list))
    if two_d:
        (xlo, xhi), (ylo, yhi) = ranges  # type: ignore[misc]
        n = n or 61
        xs, ys = _linspace(xlo, xhi, n), _linspace(ylo, yhi, n)
        grid = [[_num(f([x, y])) for x in xs] for y in ys]
        payload = {"dims": 2, "x": xs, "y": ys, "grid": grid}
    else:
        lo, hi = ranges  # type: ignore[misc]
        n = n or 201
        xs = _linspace(float(lo), float(hi), n)
        payload = {"dims": 1, "x": xs, "line": [_num(f(x)) for x in xs]}
    surface = payload.get("grid", payload.get("line"))
    payload.update({
        "explain": [tensor_summary(
            surface, "loss surface", _provenance(),
            axes=[f"{lab} in [{r[0]}, {r[1]}]" for lab, r in zip(
                labels or (["x", "y"] if two_d else ["x"]),
                ranges if two_d else [ranges])])],
        "kind": "landscape",
        "title": title,
        "labels": list(labels) if labels else (["x", "y"] if two_d else ["x"]),
        "path": [],
    })
    for point in path or ():
        _append_point(payload, f, point)
    return payload


def _append_point(payload: dict[str, Any], f: Callable[..., Any], point: Any) -> None:
    if payload["dims"] == 2:
        x, y = (float(v) for v in point)
        payload["path"].append([_num(x), _num(y), _num(f([x, y]))])
    else:
        x = float(point)
        payload["path"].append([_num(x), _num(f(x))])


class LandscapeHandle(Handle):
    def __init__(self, payload: dict[str, Any], f: Callable[..., Any]) -> None:
        self._f = f
        super().__init__(LANDSCAPE_MIME, payload)

    def add(self, point: Any) -> None:
        """Append one point (a number in 1D, [x, y] in 2D) to the drawn path."""
        _append_point(self.payload, self._f, point)
        self.update()

    def clear_path(self) -> None:
        self.payload["path"] = []
        self.update()


def show_landscape(
    f: Callable[..., Any],
    ranges: Range | tuple[Range, Range],
    path: Iterable[Any] | None = None,
    n: int | None = None,
    title: str | None = None,
    labels: Sequence[str] | None = None,
) -> LandscapeHandle:
    """Draw ``f`` over a range, with an optional path (e.g. the list that
    ``descend`` returns). Use the returned handle's ``add`` to draw a path
    live while an optimizer runs."""
    return LandscapeHandle(landscape_payload(f, ranges, path, n, title, labels), f)


def show_slice(
    loss: Callable[[], Any],
    params: Sequence[Any],
    span: float = 1.0,
    dims: int = 2,
    n: int | None = None,
    seed: int = 0,
    title: str | None = None,
) -> LandscapeHandle:
    """Slice the loss around the current parameters.

    ``params`` are Value-like objects with a mutable ``data``. ``loss()``
    recomputes the loss from them. Picks ``dims`` random directions (each the
    length of the parameter vector, so a step of 1 is "as far as the weights
    themselves") and evaluates the loss at ``params + a·d1 (+ b·d2)`` for a, b
    in [-span, span]. Parameters are restored afterwards.
    """
    if dims not in (1, 2):
        raise ValueError("dims must be 1 or 2")
    rng = random.Random(seed)
    base = [float(p.data) for p in params]
    norm = math.sqrt(sum(v * v for v in base)) or 1.0

    def direction() -> list[float]:
        d = [rng.gauss(0.0, 1.0) for _ in base]
        dn = math.sqrt(sum(v * v for v in d)) or 1.0
        return [v * norm / dn for v in d]

    dirs = [direction() for _ in range(dims)]

    def at(coords: Sequence[float]) -> float | None:
        for i, p in enumerate(params):
            p.data = base[i] + sum(c * d[i] for c, d in zip(coords, dirs))
        try:
            return _num(loss())
        finally:
            for p, v in zip(params, base):
                p.data = v

    if dims == 2:
        f = at
        ranges: Any = ((-span, span), (-span, span))
        labels = ["direction 1", "direction 2"]
    else:
        f = lambda a: at([a])  # noqa: E731
        ranges = (-span, span)
        labels = ["direction 1"]
    payload = landscape_payload(f, ranges, [[0.0, 0.0]] if dims == 2 else [0.0], n,
                                title or "loss around the current parameters", labels)
    return LandscapeHandle(payload, f)


# Language-model instruments live in their own module; re-exported here.
from .lm import (  # noqa: E402
    grad_norms, pca, rerun, show_attention, show_embeddings, show_matrix, show_training,
    token_class,
)

__all__ += ["grad_norms", "pca", "show_attention", "show_embeddings", "show_matrix",
            "show_training", "token_class"]
