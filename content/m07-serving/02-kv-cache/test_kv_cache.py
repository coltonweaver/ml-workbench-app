import torch

from m03.torch_gpt import TorchGPT
from m07.kv_cache import KVCache, forward_with_cache, generate_cached, kv_cache_bytes


def model():
    torch.manual_seed(1)
    return TorchGPT(13, 16, n_embd=32, n_head=4, n_layer=3)


@torch.no_grad()
def test_prefill_matches_the_model():
    m = model()
    idx = torch.randint(0, 13, (2, 6))
    cache = KVCache(3)
    assert torch.allclose(forward_with_cache(m, idx, cache), m(idx), atol=1e-5)
    assert cache.length == 6 and cache.k[2].shape == (2, 4, 6, 8)


@torch.no_grad()
def test_incremental_matches_full_forward():
    m = model()
    idx = torch.randint(0, 13, (2, 10))
    full = m(idx)
    cache = KVCache(3)
    forward_with_cache(m, idx[:, :4], cache)
    part = forward_with_cache(m, idx[:, 4:7], cache)
    assert torch.allclose(part, full[:, 4:7], atol=1e-5), "several new tokens at once, masked causally"
    for t in range(7, 10):
        one = forward_with_cache(m, idx[:, t:t + 1], cache)
        assert torch.allclose(one[:, 0], full[:, t], atol=1e-5), "positions continue from the cache"
    assert cache.length == 10


@torch.no_grad()
def test_generate_cached_equals_greedy():
    m = model()
    idx = torch.randint(0, 13, (3, 4))
    out = generate_cached(m, idx, 9)
    ref = idx
    for _ in range(9):
        ref = torch.cat([ref, m(ref)[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(out, ref)


def test_cache_bytes():
    # a 7B-class model: 32 layers, 4096 wide, 4096 tokens, fp16
    assert kv_cache_bytes(32, 4096, 4096) == 2 * 32 * 4096 * 4096 * 2
    assert kv_cache_bytes(2, 8, 10, batch=3, bytes_per_value=4) == 2 * 2 * 8 * 10 * 3 * 4
