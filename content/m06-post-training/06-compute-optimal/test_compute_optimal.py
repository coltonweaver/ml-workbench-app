import numpy as np

from m06.scaling import chinchilla_loss, flops, isoflop_minimum, optimal_allocation

# The fit from Hoffmann et al. (2022), approach 3.
P = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)


def test_flops():
    assert flops(1e9, 2e10) == 1.2e20


def test_chinchilla_loss():
    assert np.isclose(chinchilla_loss(1e9, 1e10, **P), 1.69 + 406.4 / 1e9 ** 0.34 + 410.7 / 1e10 ** 0.28)


def test_allocation_is_the_minimum():
    for C in [1e18, 1e21, 1e24]:
        N, D = optimal_allocation(C, P["A"], P["B"], P["alpha"], P["beta"])
        assert np.isclose(6 * N * D, C), "spend exactly the budget"
        Ns = N * np.logspace(-1, 1, 201)
        Ls = chinchilla_loss(Ns, C / (6 * Ns), **P)
        assert np.isclose(Ns[Ls.argmin()], N, rtol=0.02), "a grid search over N agrees"


def test_bigger_budgets_want_bigger_models_and_more_data():
    N1, D1 = optimal_allocation(1e20, P["A"], P["B"], P["alpha"], P["beta"])
    N2, D2 = optimal_allocation(1e22, P["A"], P["B"], P["alpha"], P["beta"])
    assert N2 > N1 and D2 > D1
    assert np.isclose(np.log(N2 / N1) / np.log(100), P["beta"] / (P["alpha"] + P["beta"]))


def test_isoflop_minimum():
    N = np.logspace(6, 9, 9)
    L = 0.05 * (np.log(N) - np.log(3e7)) ** 2 + 2.0
    assert np.isclose(isoflop_minimum(N, L), 3e7, rtol=1e-6)
