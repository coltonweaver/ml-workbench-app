"""Attribution graphs: which components, at which positions, feed which,
on the way to one prediction."""

import torch

from m05.hooks import run
from m05.lens import components


def node_names(model):
    """Hook points whose outputs are added into the residual stream, with a
    short label: [("embed", "emb"), ("blocks.0.attn_out", "a0"), ...]."""
    out = [("embed", "emb")]
    for i in range(len(model.blocks)):
        out += [(f"blocks.{i}.attn_out", f"a{i}"), (f"blocks.{i}.mlp_out", f"m{i}")]
    return out


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
    idx = torch.as_tensor(idx, dtype=torch.long)
    names = node_names(model)
    acts = {}

    def keep(name):
        def fn(t):
            t.retain_grad()
            acts[name] = t
            return t
        return fn

    patch = {name: keep(name) for name, _ in names}
    patch["ln_final.scale"] = torch.Tensor.detach
    for i in range(len(model.blocks)):
        patch[f"blocks.{i}.ln1.scale"] = torch.Tensor.detach
        patch[f"blocks.{i}.ln2.scale"] = torch.Tensor.detach
        if freeze_attention:
            patch[f"blocks.{i}.attn.pattern"] = torch.Tensor.detach
    logits = run(model, idx, patch=patch)
    logits[0, pos, token].backward(retain_graph=True)
    grads = {n: acts[n].grad[0].detach().clone() for n, _ in names}
    effects = torch.stack([(acts[n][0] * grads[n]).sum(-1).detach() for n, _ in names])

    T = idx.shape[1]
    edges = {}
    for j, (bname, _) in enumerate(names):
        if j == 0:
            continue
        for t in range(T):
            target = (acts[bname][0, t] * grads[bname][t]).sum()
            sources = [acts[n] for n, _ in names[:j]]
            gs = torch.autograd.grad(target, sources, retain_graph=True, allow_unused=True)
            for i, (src, g) in enumerate(zip(sources, gs)):
                if g is None:
                    continue
                w = (src[0] * g[0]).sum(-1).detach()
                for s in range(t + 1):
                    if w[s] != 0:
                        edges[(i, s, j, t)] = float(w[s])
    return {"labels": [label for _, label in names], "effects": effects, "edges": edges}


def prune(graph, keep_nodes=12, keep_edges=24):
    """The ``keep_nodes`` nodes with the largest |effect| and the strongest
    ``keep_edges`` edges among them. Returns (nodes, edges): nodes are
    (i, pos, effect) and edges (i, s, j, t, weight)."""
    eff = graph["effects"]
    N, T = eff.shape
    flat = eff.abs().flatten()
    top = torch.argsort(flat, descending=True, stable=True)[:keep_nodes].tolist()
    nodes = [(k // T, k % T, float(eff.flatten()[k])) for k in top]
    kept = {(i, s) for i, s, _ in nodes}
    edges = [(i, s, j, t, w) for (i, s, j, t), w in graph["edges"].items()
             if (i, s) in kept and (j, t) in kept]
    edges.sort(key=lambda e: -abs(e[4]))
    return nodes, edges[:keep_edges]
