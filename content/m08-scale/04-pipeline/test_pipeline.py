import pytest
import torch

from m08.pipeline import bubble_fraction, gpipe_schedule, ideal_bubble, makespan, pipeline_worker
from workbench.dist import launch


def test_schedule_order():
    spans = gpipe_schedule(2, 2, t_forward=1, t_backward=2)
    by = {(s, label): (a, b) for s, a, b, label, _ in spans}
    assert by[(0, "F0")] == (0, 1) and by[(1, "F0")] == (1, 2)
    assert by[(0, "F1")] == (1, 2) and by[(1, "F1")] == (2, 3)
    assert by[(1, "B1")] == (3, 5), "backward starts at the last stage after every forward"
    assert by[(0, "B1")] == (5, 7) and by[(1, "B0")] == (5, 7) and by[(0, "B0")] == (7, 9)
    assert makespan(spans) == 9
    assert {k for *_, k in spans} == {"forward", "backward"}


def test_bubble():
    for p, m in [(4, 1), (4, 8), (8, 32)]:
        spans = gpipe_schedule(p, m)
        assert bubble_fraction(spans, p) == pytest.approx(ideal_bubble(p, m))
    assert ideal_bubble(4, 1) == 0.75, "one micro-batch: only one stage works at a time"
    assert ideal_bubble(4, 64) < 0.05


def test_pipeline_forward_across_processes():
    results = launch(pipeline_worker, 3, (4,))
    assert results[0] is None and results[1] is None
    out, full = results[2]
    assert torch.allclose(out, full, atol=1e-12)
