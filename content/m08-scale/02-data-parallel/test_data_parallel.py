import numpy as np
import torch

from m08.data_parallel import ddp_worker, shard, single_process
from workbench.dist import launch


def test_shard():
    b = np.arange(10)
    parts = [shard(b, r, 3) for r in range(3)]
    assert sorted(np.concatenate(parts).tolist()) == list(range(10)), "together the shards are the batch"
    assert all(len(set(a) & set(c)) == 0 for i, a in enumerate(parts) for c in parts[i + 1:])


def test_matches_one_big_batch():
    results = launch(ddp_worker, 4, (20,))
    ref = single_process(20)
    for params, seen in results:
        assert seen == 20 * 64 // 4, "each rank sees a quarter of every batch"
        for a, b in zip(params, ref):
            assert torch.allclose(a, b, atol=1e-10), "data parallel training is the same math as one process"
    for params, _ in results[1:]:
        assert all(torch.equal(a, b) for a, b in zip(params, results[0][0])), "ranks stay identical"
