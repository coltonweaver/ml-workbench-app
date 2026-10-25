"""General charts: an x-y plot (lines and points, optional log axes) and a
timeline (lanes of spans, like a Gantt chart, with metrics and an optional
plot underneath). Scaling-law fits, pipeline schedules and the serving
simulator are drawn with these.

    show_plot([series("loss", N, L, kind="scatter"), series("fit", N, fit(N))],
              log_x=True, log_y=True, x_label="parameters", y_label="loss")
    show_timeline(["GPU 0", "GPU 1"], [span(0, 0, 1, "F0"), span(1, 1, 2, "F0")])
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import Handle, _provenance, tensor_summary
from .lm import as_array

PLOT_MIME = "application/vnd.workbench.plot+json"
TIMELINE_MIME = "application/vnd.workbench.timeline+json"
MAX_SERIES_POINTS = 5000
MAX_SPANS = 5000
MAX_LANES = 256


def series(name: str, x: Any, y: Any, kind: str = "line") -> dict[str, Any]:
    """One series for ``show_plot``: kind "line" or "scatter"."""
    if kind not in ("line", "scatter"):
        raise ValueError(f"kind must be 'line' or 'scatter', not {kind!r}")
    xs, ys = as_array(x).reshape(-1), as_array(y).reshape(-1)
    if len(xs) != len(ys):
        raise ValueError(f"series {name!r}: {len(xs)} x values but {len(ys)} y values")
    if len(xs) > MAX_SERIES_POINTS:
        step = int(np.ceil(len(xs) / MAX_SERIES_POINTS))
        xs, ys = xs[::step], ys[::step]

    def clean(a):
        return [float(v) if np.isfinite(v) else None for v in a.tolist()]

    return {"name": str(name), "kind": kind, "x": clean(xs), "y": clean(ys)}


def plot_payload(all_series: Sequence[dict[str, Any]], title: str | None = None,
                 x_label: str | None = None, y_label: str | None = None,
                 log_x: bool = False, log_y: bool = False) -> dict[str, Any]:
    all_series = list(all_series)
    for s in all_series:
        for axis, log in (("x", log_x), ("y", log_y)):
            if log and any(v is not None and v <= 0 for v in s[axis]):
                raise ValueError(f"series {s['name']!r} has {axis} values ≤ 0, which a log axis can't show")
    ys = [v for s in all_series for v in s["y"] if v is not None]
    return {
        "kind": "plot",
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "log_x": bool(log_x),
        "log_y": bool(log_y),
        "series": all_series,
        "explain": [tensor_summary(np.asarray(ys or [0.0]), "plot y values", _provenance(),
                                   series=[s["name"] for s in all_series])],
    }


def show_plot(all_series: Sequence[dict[str, Any]], title: str | None = None,
              x_label: str | None = None, y_label: str | None = None,
              log_x: bool = False, log_y: bool = False) -> Handle:
    """Lines and scatter points on shared axes; build each with ``series``."""
    return Handle(PLOT_MIME, plot_payload(all_series, title, x_label, y_label, log_x, log_y))


def span(lane: int, start: float, end: float, label: str | None = None,
         kind: str | None = None) -> dict[str, Any]:
    """One bar on a timeline: lane index, start and end time, an optional
    label drawn on it, and a ``kind`` that picks its color (spans of the
    same kind share one)."""
    if end < start:
        raise ValueError(f"span ends ({end}) before it starts ({start})")
    return {"lane": int(lane), "start": float(start), "end": float(end), "label": label, "kind": kind}


def timeline_payload(lanes: Sequence[str], spans: Sequence[dict[str, Any]], title: str | None = None,
                     unit: str = "s", metrics: Mapping[str, float] | None = None,
                     below: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    lanes = [str(l) for l in lanes]
    spans = list(spans)
    if len(lanes) > MAX_LANES or len(spans) > MAX_SPANS:
        raise ValueError(f"at most {MAX_LANES} lanes and {MAX_SPANS} spans")
    for s in spans:
        if not 0 <= s["lane"] < len(lanes):
            raise ValueError(f"span on lane {s['lane']}, but there are {len(lanes)} lanes")
    durations = np.asarray([s["end"] - s["start"] for s in spans] or [0.0])
    return {
        "kind": "timeline",
        "title": title,
        "unit": unit,
        "lanes": lanes,
        "spans": spans,
        "metrics": {str(k): float(v) for k, v in (metrics or {}).items()},
        "below": list(below or []),
        "explain": [tensor_summary(durations, "span durations", _provenance(), unit=unit)],
    }


def show_timeline(lanes: Sequence[str], spans: Sequence[dict[str, Any]], title: str | None = None,
                  unit: str = "s", metrics: Mapping[str, float] | None = None,
                  below: Sequence[dict[str, Any]] | None = None) -> Handle:
    """Lanes of spans (build each with ``span``), headline ``metrics`` above,
    and optional ``series`` plotted on the same time axis below."""
    return Handle(TIMELINE_MIME, timeline_payload(lanes, spans, title, unit, metrics, below))
