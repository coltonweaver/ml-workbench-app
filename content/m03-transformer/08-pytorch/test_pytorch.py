import numpy as np
import torch

from m03.gpt import GPT
from m03.torch_gpt import TorchGPT, load_from_engine, pick_device

rng = np.random.default_rng(6)


def models():
    engine = GPT(11, 8, n_embd=16, n_head=2, n_layer=2, seed=3)
    for p in engine.parameters():
        p.data = p.data + rng.normal(size=p.shape) * 0.2
    torch_model = TorchGPT(11, 8, n_embd=16, n_head=2, n_layer=2).double()
    assert load_from_engine(torch_model, engine) is torch_model
    return engine, torch_model


def test_logits_match():
    engine, tm = models()
    idx = rng.integers(0, 11, size=(3, 8))
    ours = engine(idx).data
    theirs = tm(torch.tensor(idx)).detach().numpy()
    assert theirs.shape == (3, 8, 11)
    assert np.allclose(ours, theirs, atol=1e-10), (
        f"max difference {np.abs(ours - theirs).max():.2e}: in float64 the two should agree to ~1e-15")


def test_loss_and_gradients_match():
    engine, tm = models()
    idx, tgt = rng.integers(0, 11, size=(2, 8)), rng.integers(0, 11, size=(2, 8))
    loss = engine.loss(idx, tgt)
    loss.backward()
    logits, tloss = tm(torch.tensor(idx), torch.tensor(tgt))
    assert logits.shape == (2, 8, 11)
    tloss.backward()
    assert np.isclose(float(loss.data), tloss.item(), atol=1e-12)
    pairs = [
        ("wte", engine.wte.grad, tm.wte.weight.grad.numpy()),
        ("attn Wq", engine.blocks[0].attn.Wq.grad, tm.blocks[0].attn.q.weight.grad.numpy().T),
        ("mlp W1", engine.blocks[1].mlp.W1.grad, tm.blocks[1].mlp.fc.weight.grad.numpy().T),
        ("ln_f gamma", engine.ln_f.gamma.grad, tm.ln_f.weight.grad.numpy()),
        ("lm_head", engine.lm_head.grad, tm.lm_head.weight.grad.numpy().T),
    ]
    for name, ours, theirs in pairs:
        assert np.allclose(ours, theirs, atol=1e-10), f"{name} gradients differ"


def test_device():
    assert pick_device() in ("mps", "cpu")
