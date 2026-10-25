"""Instruments for language models: training dashboard, embedding explorer,
attention viewer and a plain heatmap.

    dash = show_training("bigram")
    for step in range(n):
        ...
        dash.log(step, loss=loss, lr=lr, grad_norms=grad_norms(params), tokens=batch_tokens)
    dash.done()

    show_embeddings(C, labels=itos)                 # rows of C, one per token
    show_attention(lambda text: model.attention(text), "hello there")
    show_matrix(counts, row_labels=itos, col_labels=itos)
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from . import Handle, _num, _provenance, tensor_summary

TRAINING_MIME = "application/vnd.workbench.training+json"
EMBEDDING_MIME = "application/vnd.workbench.embedding+json"
ATTENTION_MIME = "application/vnd.workbench.attention+json"
MATRIX_MIME = "application/vnd.workbench.matrix+json"

MAX_POINTS = 1500
MAX_MATRIX = 128
MAX_EMBEDDINGS = 4000
MAX_TOKENS = 64

# Instruments the app can call back into: attention re-runs on a new prompt.
_callbacks: dict[str, Callable[[str], None]] = {}
_ids = itertools.count(1)


def as_array(x: Any) -> np.ndarray:
    """A NumPy array from an array, a torch tensor, the course's Tensor or
    Value, or (nested) lists of any of those."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):  # torch
        return x.detach().cpu().float().numpy()
    data = getattr(x, "data", None)
    if isinstance(data, np.ndarray):  # Tensor
        return data
    if isinstance(data, (int, float)):  # Value
        return np.asarray(float(data))
    if isinstance(x, (list, tuple)):
        return np.array([as_array(v) for v in x], dtype=float)
    return np.asarray(x, dtype=float)


def _clean(a: np.ndarray, digits: int = 6) -> list:
    """Nested lists with NaN/inf as None, rounded to keep payloads small."""
    a = np.asarray(a, dtype=float)
    out = np.round(a, digits).astype(object)
    out[~np.isfinite(a)] = None
    return out.tolist()


def rerun(callback: str, prompt: str) -> None:
    """Called by the app (through the kernel) when the learner types a new prompt."""
    fn = _callbacks.get(callback)
    if fn is None:
        raise KeyError("this instrument is from an earlier kernel session; run the code again")
    fn(prompt)


# ---- training dashboard -------------------------------------------------------

def grad_norms(params: Mapping[str, Any] | Sequence[Any]) -> dict[str, float]:
    """L2 norm of each parameter's gradient. Pass a dict ``{name: param}`` to
    name them; a list is named by position. Works with the course's Tensor
    and Value classes and with torch parameters."""
    items = params.items() if isinstance(params, Mapping) else (
        (f"param {i}", p) for i, p in enumerate(params))
    out: dict[str, float] = {}
    for name, p in items:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        out[str(name)] = float(np.linalg.norm(as_array(g)))
    return out


class TrainingDashboard(Handle):
    """Loss curves, gradient norms per layer, learning rate and tokens/sec."""

    def __init__(self, title: str | None, every: float) -> None:
        self._every = every
        self._last_publish = 0.0
        self._last_log: float | None = None
        self._where = _provenance()
        payload = {"kind": "training", "title": title, "points": [], "stride": 1,
                   "explain": [], "done": False}
        super().__init__(TRAINING_MIME, payload)

    def log(self, step: int, loss: Any = None, val_loss: Any = None, lr: Any = None,
            grad_norms: Mapping[str, Any] | None = None, tokens: int | None = None,
            **metrics: Any) -> None:
        """Record one step. ``tokens`` is how many tokens the step processed;
        the dashboard turns it into tokens/sec using the time between logs."""
        now = time.perf_counter()
        point: dict[str, Any] = {"step": int(step)}
        for key, value in (("loss", loss), ("val_loss", val_loss), ("lr", lr)):
            if value is not None:
                point[key] = _num(as_array(value).reshape(-1)[0])
        if grad_norms:
            point["grad_norms"] = {k: _num(v) for k, v in grad_norms.items()}
        if tokens is not None and self._last_log is not None and now > self._last_log:
            point["tokens_per_sec"] = tokens / (now - self._last_log)
        extra = {k: _num(v) for k, v in metrics.items()}
        if extra:
            point["metrics"] = extra
        self._last_log = now
        points = self.payload["points"]
        points.append(point)
        if len(points) > MAX_POINTS:
            # Keep the curve's shape: drop every other point, keep the latest.
            self.payload["points"] = points[::2] + ([points[-1]] if len(points) % 2 == 0 else [])
            self.payload["stride"] *= 2
        if now - self._last_publish >= self._every:
            self._flush(now)

    def _flush(self, now: float | None = None) -> None:
        self._last_publish = now or time.perf_counter()
        losses = [p["loss"] for p in self.payload["points"] if p.get("loss") is not None]
        if losses:
            self.payload["explain"] = [tensor_summary(
                losses, "training loss", self._where,
                note=f"one entry per logged step (every {self.payload['stride']} step(s) kept)")]
        self.update()

    def done(self) -> None:
        """Publish whatever hasn't been drawn yet."""
        self.payload["done"] = True
        self._flush()


def show_training(title: str | None = None, every: float = 0.25) -> TrainingDashboard:
    """A live training dashboard. Call ``log`` each step and ``done`` at the
    end. Redraws at most every ``every`` seconds so logging stays cheap."""
    return TrainingDashboard(title, every)


# ---- embedding explorer ---------------------------------------------------------

def token_class(label: str) -> str:
    """A coarse class for coloring character and word tokens."""
    if label in (".", "<s>", "</s>", "<|endoftext|>", "\n") or label.startswith("<"):
        return "special"
    stripped = label.strip()
    if not stripped:
        return "space"
    if stripped.isdigit():
        return "digit"
    if len(stripped) == 1 and stripped.isalpha():
        return "vowel" if stripped.lower() in "aeiou" else "consonant"
    if stripped.isalpha():
        return "upper" if stripped[0].isupper() else "word"
    return "punctuation"


def pca(x: np.ndarray, k: int = 2) -> tuple[np.ndarray, list[float]]:
    """Project rows of x onto their top-k principal components. Returns the
    projection and the fraction of variance each component explains."""
    x = np.asarray(x, dtype=float)
    centered = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    var = s ** 2
    total = var.sum() or 1.0
    k = min(k, vt.shape[0])
    proj = centered @ vt[:k].T
    if proj.shape[1] < 2:
        proj = np.pad(proj, ((0, 0), (0, 2 - proj.shape[1])))
    return proj, [float(v / total) for v in var[:k]]


def embedding_payload(vectors: Any, labels: Sequence[str] | None = None,
                      classes: Sequence[str] | None = None, method: str = "pca",
                      title: str | None = None) -> dict[str, Any]:
    x = as_array(vectors)
    if x.ndim != 2:
        raise ValueError(f"expected a 2D array (one row per token), got shape {list(x.shape)}")
    if x.shape[0] > MAX_EMBEDDINGS:
        raise ValueError(f"{x.shape[0]} rows is more than the explorer draws ({MAX_EMBEDDINGS})")
    labels = [str(l) for l in labels] if labels is not None else [str(i) for i in range(x.shape[0])]
    if len(labels) != x.shape[0]:
        raise ValueError(f"{len(labels)} labels for {x.shape[0]} rows")
    explained: list[float] | None = None
    if method == "pca":
        proj, explained = pca(x)
    elif method == "umap":
        try:
            import umap  # type: ignore[import-not-found]
        except ImportError:
            raise ValueError("method='umap' needs the umap-learn package; use 'pca'") from None
        proj = umap.UMAP(n_components=2, random_state=0).fit_transform(x)
    else:
        raise ValueError("method must be 'pca' or 'umap'")
    return {
        "kind": "embedding",
        "title": title,
        "method": method,
        "points": _clean(proj[:, :2]),
        "labels": labels,
        "classes": list(classes) if classes is not None else [token_class(l) for l in labels],
        "explained_variance": explained,
        "dim": int(x.shape[1]),
        "explain": [tensor_summary(x, "embedding matrix", _provenance(),
                                   note=f"{x.shape[0]} tokens × {x.shape[1]} dimensions; "
                                        f"the plot is a {method.upper()} projection")],
    }


def show_embeddings(vectors: Any, labels: Sequence[str] | None = None,
                    classes: Sequence[str] | None = None, method: str = "pca",
                    title: str | None = None) -> Handle:
    """Scatter the rows of ``vectors`` (one per token) in 2D, colored by
    token class. ``labels`` names each row (e.g. your ``itos``)."""
    return Handle(EMBEDDING_MIME, embedding_payload(vectors, labels, classes, method, title))


# ---- attention viewer -------------------------------------------------------------

def attention_payload(attn: Any, tokens: Sequence[str], title: str | None = None,
                      prompt: str | None = None, layer_names: Sequence[str] | None = None,
                      callback: str | None = None) -> dict[str, Any]:
    """``attn`` has shape (heads, T, T) or (layers, heads, T, T); row i is
    how much position i attends to each position."""
    a = as_array(attn)
    if a.ndim == 2:
        a = a[None, None]
    elif a.ndim == 3:
        a = a[None]
    if a.ndim != 4 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"expected (heads, T, T) or (layers, heads, T, T), got {list(as_array(attn).shape)}")
    if a.shape[-1] != len(tokens):
        raise ValueError(f"{len(tokens)} tokens for attention over {a.shape[-1]} positions")
    if len(tokens) > MAX_TOKENS:
        a = a[..., -MAX_TOKENS:, -MAX_TOKENS:]
        tokens = list(tokens)[-MAX_TOKENS:]
    names = list(layer_names) if layer_names else [f"layer {i}" for i in range(a.shape[0])]
    return {
        "kind": "attention",
        "title": title,
        "prompt": prompt,
        "tokens": [str(t) for t in tokens],
        "layers": [{"name": names[i], "heads": [_clean(h, 4) for h in a[i]]} for i in range(a.shape[0])],
        "callback": callback,
        "explain": [tensor_summary(a, "attention weights", _provenance(),
                                   note="layers × heads × query position × key position; "
                                        "each row sums to 1")],
    }


def show_attention(source: Any, prompt: str | Sequence[str] | None = None,
                   title: str | None = None, layer_names: Sequence[str] | None = None) -> Handle:
    """Per-head attention heatmaps.

    Either pass weights and tokens directly, ``show_attention(attn, tokens)``,
    or a function from a prompt to ``(tokens, attn)`` plus a starting prompt,
    ``show_attention(fn, "hello")``. With a function, the viewer has a text
    field: type a new prompt and it re-runs your model on it.
    """
    if not callable(source):
        if prompt is None:
            raise ValueError("pass the tokens: show_attention(attn, tokens)")
        return Handle(ATTENTION_MIME, attention_payload(source, list(prompt), title,
                                                        layer_names=layer_names))
    if not isinstance(prompt, str):
        raise ValueError("with a function, pass the starting prompt as a string")
    callback = f"attention-{next(_ids)}"
    tokens, attn = source(prompt)
    handle = Handle(ATTENTION_MIME, attention_payload(attn, tokens, title, prompt, layer_names, callback))

    def again(text: str) -> None:
        toks, weights = source(text)
        handle.update(attention_payload(weights, toks, title, text, layer_names, callback))

    _callbacks[callback] = again
    return handle


# ---- heatmap ----------------------------------------------------------------------

def matrix_payload(m: Any, row_labels: Sequence[str] | None = None,
                   col_labels: Sequence[str] | None = None, title: str | None = None,
                   log_scale: bool = False) -> dict[str, Any]:
    a = as_array(m)
    if a.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {list(a.shape)}")
    if max(a.shape) > MAX_MATRIX:
        raise ValueError(f"shape {list(a.shape)} is too big to draw; slice it to at most {MAX_MATRIX}")
    return {
        "kind": "matrix",
        "title": title,
        "values": _clean(a),
        "rows": [str(l) for l in row_labels] if row_labels is not None else None,
        "cols": [str(l) for l in col_labels] if col_labels is not None else None,
        "log_scale": bool(log_scale),
        "explain": [tensor_summary(a, "matrix", _provenance())],
    }


def show_matrix(m: Any, row_labels: Sequence[str] | None = None,
                col_labels: Sequence[str] | None = None, title: str | None = None,
                log_scale: bool = False) -> Handle:
    """A heatmap of a 2D array, e.g. bigram counts. Hover a cell for its value."""
    return Handle(MATRIX_MIME, matrix_payload(m, row_labels, col_labels, title, log_scale))
