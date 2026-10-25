"""Checkpointing, recovering from failures, and stragglers."""

import os

import numpy as np
import torch


def save_checkpoint(path, model, opt, step, rng):
    """Write model, optimizer, step and RNG state atomically: to a temporary
    file first, then rename over ``path``, so a crash mid-write never leaves
    a half-written checkpoint."""
    tmp = f"{path}.tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, opt, rng):
    """Restore everything save_checkpoint saved; return the step."""
    state = torch.load(path, weights_only=False)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    rng.bit_generator.state = state["rng"]
    torch.set_rng_state(state["torch_rng"])
    return state["step"]


class Crash(Exception):
    pass


def train(path, steps, every, crash_at=None, seed=0):
    """Train a small model with dropout and Adam (so the optimizer and RNG
    state matter), checkpointing every ``every`` steps. Resumes from
    ``path`` if it exists. Raises Crash at step ``crash_at``. Returns the
    final parameters."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    rng = np.random.default_rng(seed)
    step = load_checkpoint(path, model, opt, rng) if os.path.exists(path) else 0
    while step < steps:
        if step == crash_at:
            raise Crash(f"crashed at step {step}")
        x = torch.tensor(rng.normal(size=(16, 4)), dtype=torch.float32)
        loss = ((model(x) - x.sum(1, keepdim=True)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % every == 0:
            save_checkpoint(path, model, opt, step, rng)
    return [p.detach().clone() for p in model.parameters()]


def sync_step_times(times):
    """Synchronous data parallelism waits for the slowest rank every step.
    ``times`` is (ranks, steps); returns the per-step times."""
    return np.asarray(times).max(axis=0)


def with_backup_workers(times, backups):
    """Proceed once all but ``backups`` ranks have finished each step (Chen
    et al., 2016): the per-step time is the (ranks − backups)-th fastest."""
    t = np.sort(np.asarray(times), axis=0)
    return t[len(t) - 1 - backups]


def stragglers(times, factor=1.5):
    """Ranks whose median step time is more than ``factor`` times the median
    over all ranks."""
    t = np.asarray(times)
    per_rank = np.median(t, axis=1)
    return [int(r) for r in np.flatnonzero(per_rank > factor * np.median(per_rank))]


def expected_goodput(mtbf_hours, checkpoint_minutes, interval_minutes, restart_minutes):
    """Fraction of time doing useful work with periodic checkpoints: lose
    the checkpoint cost each interval, and on each failure the restart time
    plus half an interval of redone work."""
    interval_h, ckpt_h, restart_h = interval_minutes / 60, checkpoint_minutes / 60, restart_minutes / 60
    overhead = ckpt_h / (interval_h + ckpt_h)
    lost_per_failure = restart_h + interval_h / 2
    return (1 - overhead) * (1 - lost_per_failure / mtbf_hours)
