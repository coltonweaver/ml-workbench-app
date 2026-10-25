"""Checkpointing, recovering from failures, and stragglers.

Lesson 8.6."""

import os

import numpy as np
import torch


def save_checkpoint(path, model, opt, step, rng):
    """Write model, optimizer, step and RNG state atomically: to a temporary
    file first, then rename over ``path``, so a crash mid-write never leaves
    a half-written checkpoint."""
    raise NotImplementedError("lesson 8.6")


def load_checkpoint(path, model, opt, rng):
    """Restore everything save_checkpoint saved; return the step."""
    raise NotImplementedError("lesson 8.6")


class Crash(Exception):
    pass


def train(path, steps, every, crash_at=None, seed=0):
    """Train a small model with dropout and Adam (so the optimizer and RNG
    state matter), checkpointing every ``every`` steps. Resumes from
    ``path`` if it exists. Raises Crash at step ``crash_at``. Returns the
    final parameters."""
    raise NotImplementedError("lesson 8.6")


def sync_step_times(times):
    """Synchronous data parallelism waits for the slowest rank every step.
    ``times`` is (ranks, steps); returns the per-step times."""
    raise NotImplementedError("lesson 8.6")


def with_backup_workers(times, backups):
    """Proceed once all but ``backups`` ranks have finished each step (Chen
    et al., 2016): the per-step time is the (ranks − backups)-th fastest."""
    raise NotImplementedError("lesson 8.6")


def stragglers(times, factor=1.5):
    """Ranks whose median step time is more than ``factor`` times the median
    over all ranks."""
    raise NotImplementedError("lesson 8.6")


def expected_goodput(mtbf_hours, checkpoint_minutes, interval_minutes, restart_minutes):
    """Fraction of time doing useful work with periodic checkpoints: lose
    the checkpoint cost each interval, and on each failure the restart time
    plus half an interval of redone work."""
    raise NotImplementedError("lesson 8.6")
