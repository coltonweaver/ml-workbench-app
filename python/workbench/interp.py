"""Interpretability instruments: a feature browser with activation-highlighted
text, and a logit attribution view (per-token heatmap plus an optional
attribution graph).

    show_features([feature_card(7, contexts, idx, decode, up, down, density)])
    show_attribution(tokens, ["emb", "a0", "m0"], values, target=(5, "e"))
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from . import Handle, _provenance, tensor_summary
from .lm import _clean, as_array

FEATURES_MIME = "application/vnd.workbench.features+json"
ATTRIBUTION_MIME = "application/vnd.workbench.attribution+json"
MAX_FEATURES = 256
MAX_EXAMPLES = 40
MAX_CONTEXT = 64
MAX_TOKENS = 128
MAX_COMPONENTS = 64
MAX_GRAPH_NODES = 200
MAX_GRAPH_EDGES = 600


def _token_values(pairs: Any, decode: Callable[[int], str] | None) -> list[dict[str, Any]]:
    if pairs is None:
        return []
    ids, values = pairs
    out = []
    for i, v in zip(as_array(ids).reshape(-1).tolist(), as_array(values).reshape(-1).tolist()):
        out.append({"token": decode(int(i)) if decode else str(int(i)), "value": float(v)})
    return out


def feature_card(feature: int, contexts: Sequence[dict[str, Any]], idx: Any,
                 decode: Callable[[int], str] | None = None, up: Any = None, down: Any = None,
                 density: float | None = None, label: str | None = None) -> dict[str, Any]:
    """One feature's entry for ``show_features``.

    ``contexts`` are dicts with "row", "start", "pos" and "acts" (as
    ``top_contexts`` returns them); ``idx`` is the (B, T) token array they
    index into. ``up`` and ``down`` are optional (token ids, values) pairs
    for the tokens the feature promotes and suppresses.
    """
    tokens = as_array(idx).astype(int)
    examples = []
    for c in contexts[:MAX_EXAMPLES]:
        row, start, pos = int(c["row"]), int(c["start"]), int(c["pos"])
        start = max(start, pos + 1 - MAX_CONTEXT)
        ids = tokens[row, start:pos + 1].tolist()
        acts = [float(a) for a in c["acts"]][-len(ids):]
        examples.append({
            "tokens": [decode(i) if decode else str(i) for i in ids],
            "acts": acts,
            "highlight": len(ids) - 1,
        })
    peak = max((max(e["acts"]) for e in examples if e["acts"]), default=0.0)
    return {
        "id": int(feature),
        "label": label,
        "density": None if density is None else float(density),
        "max_act": float(peak),
        "examples": examples,
        "up": _token_values(up, decode),
        "down": _token_values(down, decode),
    }


def features_payload(cards: Sequence[dict[str, Any]], title: str | None = None) -> dict[str, Any]:
    cards = list(cards)
    if len(cards) > MAX_FEATURES:
        raise ValueError(f"{len(cards)} features is too many to browse; pass at most {MAX_FEATURES}")
    acts = [a for c in cards for e in c["examples"] for a in e["acts"]]
    return {
        "kind": "features",
        "title": title,
        "features": cards,
        "explain": [tensor_summary(np.asarray(acts or [0.0]), "feature activations", _provenance(),
                                   features=len(cards))],
    }


def show_features(cards: Sequence[dict[str, Any]], title: str | None = None) -> Handle:
    """A feature (or neuron) browser: pick a feature, see the contexts where it
    fires most with each token shaded by its activation, and the tokens it
    pushes up and down."""
    return Handle(FEATURES_MIME, features_payload(cards, title))


def attribution_payload(tokens: Sequence[str], components: Sequence[str], values: Any,
                        target: tuple[int, str] | None = None,
                        nodes: Sequence[tuple[int, int, float]] | None = None,
                        edges: Sequence[tuple[int, int, int, int, float]] | None = None,
                        title: str | None = None) -> dict[str, Any]:
    v = as_array(values)
    if v.ndim != 2 or v.shape != (len(components), len(tokens)):
        raise ValueError(f"values should be (components, tokens) = ({len(components)}, {len(tokens)}),"
                         f" got {list(v.shape)}")
    if len(tokens) > MAX_TOKENS or len(components) > MAX_COMPONENTS:
        raise ValueError(f"at most {MAX_TOKENS} tokens and {MAX_COMPONENTS} components")
    graph = None
    if nodes is not None:
        nodes, edges = list(nodes)[:MAX_GRAPH_NODES], list(edges or [])[:MAX_GRAPH_EDGES]
        ids = {(int(i), int(s)): n for n, (i, s, _) in enumerate(nodes)}
        graph = {
            "nodes": [{"component": int(i), "pos": int(s), "effect": float(e)} for i, s, e in nodes],
            "edges": [{"source": ids[(int(i), int(s))], "target": ids[(int(j), int(t))], "weight": float(w)}
                      for i, s, j, t, w in edges if (int(i), int(s)) in ids and (int(j), int(t)) in ids],
        }
    return {
        "kind": "attribution",
        "title": title,
        "tokens": [str(t) for t in tokens],
        "components": [str(c) for c in components],
        "values": _clean(v),
        "target": None if target is None else {"pos": int(target[0]), "token": str(target[1])},
        "graph": graph,
        "explain": [tensor_summary(v, "attribution", _provenance(), layout="(components, tokens)")],
    }


def show_attribution(tokens: Sequence[str], components: Sequence[str], values: Any,
                     target: tuple[int, str] | None = None,
                     nodes: Sequence[tuple[int, int, float]] | None = None,
                     edges: Sequence[tuple[int, int, int, int, float]] | None = None,
                     title: str | None = None) -> Handle:
    """Logit attribution: ``values[c, t]`` is component c's contribution at
    token t, drawn as a diverging heatmap under the tokens. With ``nodes``
    (component, pos, effect) and ``edges`` (component, pos, component, pos,
    weight), also draws the attribution graph, components bottom to top."""
    return Handle(ATTRIBUTION_MIME, attribution_payload(tokens, components, values, target, nodes, edges,
                                                        title))
