import torch

from m03.torch_gpt import TorchGPT
from m07.decoding import (arithmetic_intensity, cached_generation_flops, critical_batch, forward_flops,
                          generate_naive, naive_generation_flops, step_time)


@torch.no_grad()
def test_generate_naive_is_greedy():
    torch.manual_seed(0)
    m = TorchGPT(11, 8, n_embd=16, n_head=2, n_layer=1)
    idx = torch.tensor([[1, 2, 3]])
    out = generate_naive(m, idx, 7)
    assert out.shape == (1, 10) and torch.equal(out[:, :3], idx)
    ref = idx
    for _ in range(7):
        ref = torch.cat([ref, m(ref[:, -8:])[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(out, ref), "crop to block_size, then take the argmax"


def test_forward_flops():
    assert forward_flops(1000, 2, 8, 1) == 2000 + 4 * 2 * 1 * 8
    assert forward_flops(1000, 2, 8, 10) - forward_flops(1000, 2, 8, 9) == 64, "attention grows with context"


def test_cache_saves_quadratic_work():
    args = (10_000, 2, 16)
    naive = naive_generation_flops(*args, prompt=4, new=3)
    # steps reprocess 4, 5, 6 tokens at positions 1..n
    expected = sum(sum(forward_flops(*args, t) for t in range(1, n + 1)) for n in (4, 5, 6))
    assert naive == expected
    cached = cached_generation_flops(*args, prompt=4, new=3)
    assert cached == sum(forward_flops(*args, t) for t in range(1, 7)), "each token processed once"
    assert naive_generation_flops(*args, 100, 100) > 50 * cached_generation_flops(*args, 100, 100)


def test_roofline():
    assert arithmetic_intensity(1, 2) == 1.0, "batch 1, fp16: one FLOP per byte"
    assert arithmetic_intensity(64, 1) == 128
    # 7B params in fp16 on 1e15 FLOP/s and 1e12 B/s
    t1 = step_time(7e9, 1, 2, 1e15, 1e12)
    assert abs(t1 - 0.014) < 1e-9, "memory-bound: 14 GB of weights at 1 TB/s"
    assert step_time(7e9, 8, 2, 1e15, 1e12) == t1, "eight sequences cost the same as one"
    assert critical_batch(2, 1e15, 1e12) == 1000
    assert step_time(7e9, 2000, 2, 1e15, 1e12) > t1 * 1.9, "past the critical batch, compute-bound"
