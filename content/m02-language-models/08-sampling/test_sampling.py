import numpy as np

from m02.sampling import generate, sample_next


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def frequencies(logits, temperature, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    draws = [sample_next(logits, temperature, rng) for _ in range(n)]
    return np.bincount(draws, minlength=len(logits)) / n


LOGITS = np.array([2.0, 1.0, 0.0, -1.0])


def test_greedy():
    rng = np.random.default_rng(0)
    assert all(sample_next(LOGITS, 0, rng) == 0 for _ in range(20))
    assert isinstance(sample_next(LOGITS, 1.0, rng), int)


def test_temperature_one_is_softmax():
    f = frequencies(LOGITS, 1.0)
    assert np.allclose(f, softmax(LOGITS), atol=0.015), f"expected about {softmax(LOGITS)}, got {f}"


def test_temperature_scales_logits():
    hot, cold = frequencies(LOGITS, 2.0), frequencies(LOGITS, 0.5)
    assert np.allclose(hot, softmax(LOGITS / 2.0), atol=0.015), "T = 2 divides the logits by 2"
    assert np.allclose(cold, softmax(LOGITS / 0.5), atol=0.015), "T = 0.5 doubles them"
    assert hot[0] < softmax(LOGITS)[0] < cold[0], "higher temperature, flatter distribution"


def test_large_logits_are_fine():
    f = frequencies(np.array([1000.0, 999.0]), 1.0, n=4000)
    assert np.all(np.isfinite(f)) and abs(f[0] - softmax(np.array([1.0, 0.0]))[0]) < 0.03


def test_generate_grows_context_and_stops():
    seen = []

    def next_logits(context):
        seen.append(list(context))
        # always "context length", then the stop token 0 after 3 new tokens
        nxt = 0 if len(context) >= 4 else len(context) + 1
        logits = np.full(10, -1e9)
        logits[nxt] = 0.0
        return logits

    out = generate(next_logits, [7], 1.0, np.random.default_rng(0))
    assert out == [2, 3, 4], "the stop token ends generation and isn't included"
    assert seen[0] == [7] and seen[1] == [7, 2], "each new token is fed back in"


def test_generate_max_new():
    out = generate(lambda c: np.array([-1e9, 0.0]), [], 1.0, np.random.default_rng(0), max_new=6)
    assert out == [1] * 6
