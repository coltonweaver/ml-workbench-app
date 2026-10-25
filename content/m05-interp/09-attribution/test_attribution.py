import torch

from m03.torch_gpt import TorchGPT
from m05.attribution import attribution_graph, node_names, prune
from m05.hooks import run_with_cache
from m05.lens import direct_logit_attribution


def setup():
    torch.manual_seed(5)
    m = TorchGPT(10, 8, n_embd=16, n_head=2, n_layer=2)
    with torch.no_grad():
        m.ln_f.weight.uniform_(0.5, 1.5)
    return m, torch.tensor([[1, 4, 2, 7, 3]])


def test_node_names():
    m, _ = setup()
    assert node_names(m) == [("embed", "emb"), ("blocks.0.attn_out", "a0"), ("blocks.0.mlp_out", "m0"),
                             ("blocks.1.attn_out", "a1"), ("blocks.1.mlp_out", "m1")]


def test_effects():
    m, idx = setup()
    g = attribution_graph(m, idx, 3, 6)
    assert g["labels"] == ["emb", "a0", "m0", "a1", "m1"]
    assert g["effects"].shape == (5, 5)
    assert torch.all(g["effects"][:, 4] == 0), "position 4 comes after the target position"
    with torch.no_grad():
        _, cache = run_with_cache(m, idx)
        names, dla, _ = direct_logit_attribution(m, cache, 3, 6)
    assert torch.isclose(g["effects"][4, 3], dla[names.index("mlp 1")], atol=1e-5), (
        "the last MLP at the target position only acts directly, so its effect is its DLA "
        "(with the final layer norm's scale frozen)")


def test_edges_respect_the_architecture():
    m, idx = setup()
    g = attribution_graph(m, idx, 3, 6)
    assert g["edges"], "there should be edges"
    for (i, s, j, t), w in g["edges"].items():
        assert i < j and s <= t, "edges go up the layers and never back in time"
        if i in (1, 2, 3) and j in (2, 4) and j == i + 1:
            assert s == t, "an MLP only reads its own position"
    assert any(s < t for (i, s, j, t) in g["edges"]), "attention moves information between positions"
    assert (2, 3, 4, 3) in g["edges"]


def test_prune():
    m, idx = setup()
    g = attribution_graph(m, idx, 3, 6)
    nodes, edges = prune(g, keep_nodes=6, keep_edges=5)
    assert len(nodes) == 6 and len(edges) <= 5
    mags = [abs(e) for _, _, e in nodes]
    assert mags == sorted(mags, reverse=True)
    kept = {(i, s) for i, s, _ in nodes}
    assert all((i, s) in kept and (j, t) in kept for i, s, j, t, _ in edges)
    ws = [abs(e[4]) for e in edges]
    assert ws == sorted(ws, reverse=True)
