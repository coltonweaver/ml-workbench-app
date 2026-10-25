import torch

from m03.torch_gpt import TorchGPT
from m07.quant import (dequantize, dequantize_int4, linear_weight_bytes, quantize_int4, quantize_int8,
                       quantize_model)


def test_int8_by_hand():
    w = torch.tensor([[1.0, -0.5, 0.25], [0.0, 2.0, -4.0]])
    q, scale = quantize_int8(w)
    assert q.dtype == torch.int8 and scale.shape == (2, 1)
    assert torch.allclose(scale[:, 0], torch.tensor([1 / 127, 4 / 127]))
    assert q.tolist() == [[127, -64, 32], [0, 64, -127]]


def test_int8_error_bound():
    torch.manual_seed(0)
    w = torch.randn(32, 64) * torch.logspace(-2, 1, 32)[:, None]
    q, scale = quantize_int8(w)
    err = (dequantize(q, scale) - w).abs()
    assert torch.all(err <= scale / 2 + 1e-7), "rounding error is at most half a step"
    assert q.abs().max() == 127, "the largest weight in each row uses the full range"


def test_int4_groups():
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    w[0, 0] = 50.0                       # an outlier
    q, scale = quantize_int4(w, group=32)
    assert scale.shape == (8, 2) and q.abs().max() <= 7
    back = dequantize_int4(q, scale, group=32)
    assert (back - w)[0, 32:].abs().max() < 0.5, "the outlier only coarsens its own group"
    assert (back - w).abs().max() > (dequantize(*quantize_int8(w)) - w).abs().max(), "4 bits are coarser than 8"


@torch.no_grad()
def test_quantized_model():
    torch.manual_seed(0)
    m = TorchGPT(20, 16, n_embd=64, n_head=4, n_layer=2)
    idx = torch.randint(0, 20, (2, 10))
    ref = m(idx)
    q8, q4 = quantize_model(m, 8), quantize_model(m, 4)
    e8, e4 = (q8(idx) - ref).abs().max(), (q4(idx) - ref).abs().max()
    assert e8 < 0.05 and e4 < 0.5 and e8 < e4
    assert torch.equal(m(idx), ref), "quantize a copy, not the original"
    assert not torch.equal(q8.blocks[0].mlp.fc.weight, m.blocks[0].mlp.fc.weight)


def test_bytes():
    m = TorchGPT(20, 16, n_embd=64, n_head=4, n_layer=2)
    n = sum(mod.weight.numel() for mod in m.modules() if isinstance(mod, torch.nn.Linear))
    assert linear_weight_bytes(m, 16) == 2 * n
    rows = sum(mod.weight.shape[0] for mod in m.modules() if isinstance(mod, torch.nn.Linear))
    assert linear_weight_bytes(m, 8) == n + 2 * rows
    assert linear_weight_bytes(m, 4) == n / 2 + 2 * n / 32
