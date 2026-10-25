import os

import numpy as np
import pytest
import torch

from m08.fault import (Crash, expected_goodput, stragglers, sync_step_times, train, with_backup_workers)


def test_resume_is_exact(tmp_path):
    full = train(str(tmp_path / "a.pt"), 30, 5)
    path = str(tmp_path / "b.pt")
    with pytest.raises(Crash):
        train(path, 30, 5, crash_at=17)
    assert os.path.exists(path) and not os.path.exists(path + ".tmp"), "the last checkpoint is whole"
    assert torch.load(path, weights_only=False)["step"] == 15
    resumed = train(path, 30, 5)
    assert all(torch.equal(a, b) for a, b in zip(full, resumed)), (
        "restore the model, optimizer and both RNGs, and the run is bit-for-bit the same")


def test_sync_and_backups():
    times = np.array([[1.0, 1.0, 1.0], [1.1, 1.2, 0.9], [1.0, 5.0, 1.0]])
    assert sync_step_times(times).tolist() == [1.1, 5.0, 1.0]
    assert with_backup_workers(times, 1).tolist() == [1.0, 1.2, 1.0], "ignore the slowest rank each step"


def test_stragglers():
    rng = np.random.default_rng(0)
    times = rng.normal(1.0, 0.05, size=(8, 50))
    times[5] *= 2.0
    times[2, 10] = 30.0           # one slow step is not a straggler
    assert stragglers(times) == [5]


def test_goodput():
    g = expected_goodput(mtbf_hours=10, checkpoint_minutes=1, interval_minutes=30, restart_minutes=10)
    assert g == pytest.approx((1 - 1 / 31) * (1 - (10 / 60 + 0.25) / 10))
    often = expected_goodput(10, 1, 5, 10)
    rarely = expected_goodput(10, 1, 300, 10)
    best = max(expected_goodput(10, 1, i, 10) for i in range(5, 300))
    assert best > often and best > rarely, "checkpoint neither too often nor too rarely"
