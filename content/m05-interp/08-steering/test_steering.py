import torch

from m03.torch_gpt import TorchGPT
from m05.hooks import run, run_with_cache
from m05.steering import add_vector, generate, mean_resid, steered_logits, steering_vector


def model():
    torch.manual_seed(3)
    return TorchGPT(12, 8, n_embd=16, n_head=2, n_layer=2)


@torch.no_grad()
def test_mean_resid_and_vector():
    m = model()
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b = torch.tensor([[7, 8, 9]])
    _, ca = run_with_cache(m, a)
    _, cb = run_with_cache(m, b)
    assert torch.allclose(mean_resid(m, a, 1), ca["blocks.1.resid_pre"][:, -1].mean(0))
    assert torch.allclose(mean_resid(m, a, 0, pos=1), ca["blocks.0.resid_pre"][:, 1].mean(0))
    v = steering_vector(m, a, b, 1)
    assert torch.allclose(v, ca["blocks.1.resid_pre"][:, -1].mean(0) - cb["blocks.1.resid_pre"][0, -1])


def test_add_vector():
    f = add_vector(torch.tensor([1.0, 2.0]), 3.0)
    assert torch.equal(f(torch.zeros(1, 2, 2)), torch.tensor([[[3.0, 6.0], [3.0, 6.0]]]))


@torch.no_grad()
def test_steering_one_token_into_another():
    m = model()
    a, b = torch.tensor([[3]]), torch.tensor([[5]])
    v = steering_vector(m, a, b, 0)
    assert torch.allclose(steered_logits(m, b, 0, v, 1.0), m(a), atol=1e-5), (
        "for one-token prompts, adding (a - b) at layer 0 turns b into a")
    assert torch.allclose(steered_logits(m, b, 0, v, 0.0), m(b), atol=1e-6)


@torch.no_grad()
def test_generate():
    m = model()
    prompt = torch.tensor([[1, 2]])
    out = generate(m, prompt, 4)
    ref = prompt.clone()
    for _ in range(4):
        ref = torch.cat([ref, m(ref)[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(out, ref), "temperature 0 is greedy"
    long = torch.randint(0, 12, (1, 11))
    assert generate(m, long, 3).shape == (1, 14), "crop the context to block_size"
    v = torch.randn(16) * 3
    steered = generate(m, prompt, 4, layer=1, vec=v, scale=2.0)
    ref = prompt.clone()
    for _ in range(4):
        logits = run(m, ref, patch={"blocks.1.resid_pre": lambda t: t + 2.0 * v})
        ref = torch.cat([ref, logits[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(steered, ref), "steer on every step"
    s1 = generate(m, prompt, 6, temperature=1.0, seed=4)
    assert torch.equal(s1, generate(m, prompt, 6, temperature=1.0, seed=4)), "same seed, same sample"
