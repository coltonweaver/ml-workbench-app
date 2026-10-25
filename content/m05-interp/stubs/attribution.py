"""Attribution graphs: which components, at which positions, feed which,
on the way to one prediction.

Lesson 5.9."""
import torch
from m05.hooks import run
from m05.lens import components


def node_names(model):
    """Hook points whose outputs are added into the residual stream, with a
    short label: [("embed", "emb"), ("blocks.0.attn_out", "a0"), ...]."""
    raise NotImplementedError('lesson 5.9')


def attribution_graph(model, idx, pos, token, freeze_attention=True):
    """Gradient × activation attribution for the logit of ``token`` at
    ``pos`` on one prompt idx (1, T).

    Nodes are (component, position). A node's effect is a · ∂logit/∂a: the
    first-order estimate of how much the logit would drop without it. The
    edge from node A to a later node B is a_A · ∂(a_B · g_B)/∂a_A, where g_B
    is B's gradient held fixed: how much of B's effect comes through A.
    Every layer norm's scale is frozen (detached) at its value on this
    prompt, so the norms act as linear maps; without that, a · ∂logit/∂a
    summed over a norm's inputs is 0, since a norm ignores its input's size.
    With ``freeze_attention``, attention patterns are detached too, so
    effects flow only through what heads move, not through where they look.

    Returns {"labels": [...], "effects": (N, T) tensor,
             "edges": {(i, s, j, t): weight}} with node i at position s
    feeding node j at position t.
    """
    raise NotImplementedError('lesson 5.9')


def prune(graph, keep_nodes=12, keep_edges=24):
    """The ``keep_nodes`` nodes with the largest |effect| and the strongest
    ``keep_edges`` edges among them. Returns (nodes, edges): nodes are
    (i, pos, effect) and edges (i, s, j, t, weight)."""
    raise NotImplementedError('lesson 5.9')
