import pytest
import torch

from m07.paged import BlockAllocator, contiguous_waste, gather


def test_allocate_append_release():
    a = BlockAllocator(6, 4)
    assert a.allocate("x", 5) == [0, 1] and a.n_free == 4
    assert a.allocate("y", 4) == [2]
    a.append("y")                       # y's block is full: takes a new one
    assert a.tables["y"] == [2, 3]
    for _ in range(3):
        a.append("x")                   # 6, 7, 8 tokens fit in 2 blocks
    assert a.tables["x"] == [0, 1] and a.lengths["x"] == 8
    a.append("x")
    assert a.tables["x"] == [0, 1, 4]
    assert a.wasted_slots() == (12 - 9) + (8 - 5)
    a.release("x")
    assert a.n_free == 4 and "x" not in a.tables


def test_out_of_memory_takes_nothing():
    a = BlockAllocator(3, 4)
    a.allocate("x", 8)
    with pytest.raises(MemoryError):
        a.allocate("y", 5)
    assert a.n_free == 1 and "y" not in a.tables
    assert a.can_allocate(4) and not a.can_allocate(5)
    a.allocate("y", 1)
    for _ in range(3):
        a.append("y")
    with pytest.raises(MemoryError):
        a.append("y")


def test_slots_and_gather():
    a = BlockAllocator(4, 2)
    a.allocate("x", 1)
    a.allocate("y", 2)
    a.append("x")
    a.append("x")
    assert a.tables["x"] == [0, 2] and a.slot("x", 2) == (2, 0) and a.slot("y", 1) == (1, 1)
    pool = torch.zeros(4, 2, 3)
    for p in range(3):
        b, o = a.slot("x", p)
        pool[b, o] = p + 1
    assert gather(pool, a, "x")[:, 0].tolist() == [1, 2, 3]
    assert gather(pool, a, "y").shape == (2, 3)


def test_contiguous_waste():
    assert contiguous_waste([10, 200, 37], 512) == 3 * 512 - 247
