import torch

from m03.torch_gpt import TorchGPT
from m05.hooks import run_with_cache
from m05.lens import components, direct_logit_attribution, logit_lens


def setup():
    torch.manual_seed(1)
    m = TorchGPT(11, 10, n_embd=16, n_head=2, n_layer=3)
    with torch.no_grad():
        m.ln_f.weight.uniform_(0.5, 1.5)
        m.ln_f.bias.uniform_(-0.5, 0.5)
    idx = torch.randint(0, 11, (2, 7))
    logits, cache = run_with_cache(m, idx)
    return m, idx, logits, cache


@torch.no_grad()
def test_logit_lens_layers():
    m, idx, logits, cache = setup()
    ll = logit_lens(m, cache)
    assert ll.shape == (4, 2, 7, 11), "after the embedding and after each of 3 blocks"
    assert torch.allclose(ll[-1], logits, atol=1e-5), "the last layer of the lens is the real output"
    assert torch.allclose(ll[0], m.lm_head(m.ln_f(cache["embed"])), atol=1e-5)
    assert torch.allclose(ll[1], m.lm_head(m.ln_f(cache["blocks.0.resid_post"])), atol=1e-5)


@torch.no_grad()
def test_components_sum_to_the_stream():
    m, idx, logits, cache = setup()
    parts = components(m, cache)
    assert [n for n, _ in parts] == ["embed", "attn 0", "mlp 0", "attn 1", "mlp 1", "attn 2", "mlp 2"]
    total = sum(t for _, t in parts)
    assert torch.allclose(total, cache["blocks.2.resid_post"], atol=1e-5)


@torch.no_grad()
def test_dla_adds_up_to_the_logit():
    m, idx, logits, cache = setup()
    for batch, pos, token in [(0, 6, 3), (1, 2, 0), (1, 6, 10)]:
        names, contribs, const = direct_logit_attribution(m, cache, pos, token, batch=batch)
        assert len(names) == 7 and contribs.shape == (7,)
        assert torch.isclose(contribs.sum() + const, logits[batch, pos, token], atol=1e-4)


@torch.no_grad()
def test_dla_of_one_component():
    m, idx, logits, cache = setup()
    names, contribs, _ = direct_logit_attribution(m, cache, 4, 5)
    c = cache["blocks.1.mlp_out"][0, 4]
    sigma = cache["ln_final.scale"][0, 4]
    expected = ((c - c.mean()) / sigma * m.ln_f.weight) @ m.lm_head.weight[5]
    assert torch.isclose(contribs[names.index("mlp 1")], expected, atol=1e-5)
