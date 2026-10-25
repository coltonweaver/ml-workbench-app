import torch

from m03.torch_gpt import TorchGPT
from m05.hooks import hook_names, layer_norm, run, run_with_cache


def model_and_input():
    torch.manual_seed(0)
    m = TorchGPT(13, 12, n_embd=16, n_head=4, n_layer=2)
    return m, torch.randint(0, 13, (2, 9))


@torch.no_grad()
def test_same_logits_as_the_model():
    m, idx = model_and_input()
    assert torch.allclose(run(m, idx), m(idx), atol=1e-5)


@torch.no_grad()
def test_cache_has_every_point_in_order():
    m, idx = model_and_input()
    _, cache = run_with_cache(m, idx)
    assert list(cache) == hook_names(m)
    assert cache["blocks.1.attn.pattern"].shape == (2, 4, 9, 9)
    assert cache["blocks.0.attn.z"].shape == (2, 9, 4, 4)
    assert cache["blocks.0.ln1.scale"].shape == (2, 9, 1)
    assert cache["final"].shape == (2, 9, 16)


@torch.no_grad()
def test_the_residual_stream_adds_up():
    m, idx = model_and_input()
    _, c = run_with_cache(m, idx)
    assert torch.allclose(c["blocks.0.resid_pre"], c["embed"])
    for i in range(2):
        assert torch.allclose(c[f"blocks.{i}.resid_mid"], c[f"blocks.{i}.resid_pre"] + c[f"blocks.{i}.attn_out"])
        assert torch.allclose(c[f"blocks.{i}.resid_post"], c[f"blocks.{i}.resid_mid"] + c[f"blocks.{i}.mlp_out"])
    assert torch.allclose(c["blocks.1.resid_pre"], c["blocks.0.resid_post"])


@torch.no_grad()
def test_attention_pieces():
    m, idx = model_and_input()
    _, c = run_with_cache(m, idx)
    p = c["blocks.0.attn.pattern"]
    assert torch.allclose(p.sum(-1), torch.ones(2, 4, 9), atol=1e-6)
    assert torch.all(torch.triu(p, diagonal=1) == 0), "no attention to the future"
    attn = m.blocks[0].attn
    assert torch.allclose(c["blocks.0.attn_out"], attn.o(c["blocks.0.attn.z"].reshape(2, 9, 16)), atol=1e-6)
    assert torch.allclose(c["blocks.0.attn_out"], attn(m.blocks[0].ln1(c["blocks.0.resid_pre"])), atol=1e-5)


@torch.no_grad()
def test_layer_norm_by_hand():
    m, _ = model_and_input()
    with torch.no_grad():
        m.ln_f.weight.uniform_(0.5, 1.5)
        m.ln_f.bias.uniform_(-1, 1)
    x = torch.randn(3, 5, 16) * 4 + 2
    seen = []
    out = layer_norm(m.ln_f, x, lambda s: seen.append(s) or s)
    assert torch.allclose(out, m.ln_f(x), atol=1e-5)
    assert torch.allclose(seen[0], torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5), atol=1e-5)


@torch.no_grad()
def test_patches_apply_and_are_cached():
    m, idx = model_and_input()
    cache = {}
    run(m, idx, patch={"blocks.0.attn_out": torch.zeros_like}, cache=cache)
    assert torch.all(cache["blocks.0.attn_out"] == 0), "the cache holds the patched value"
    assert torch.allclose(cache["blocks.0.resid_mid"], cache["blocks.0.resid_pre"])
    logits = run(m, idx, patch={"final": lambda t: t * 0})
    assert torch.allclose(logits, torch.zeros_like(logits)), "lm_head has no bias"
