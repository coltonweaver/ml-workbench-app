"""Mixed precision: 16-bit math with 32-bit safety nets.

Lesson 8.5."""

import torch


def spacing(x, dtype):
    """The gap between ``x`` and the next representable number in ``dtype``."""
    raise NotImplementedError("lesson 8.5")


def update_survives(weight, update, dtype):
    """Does weight + update differ from weight when stored in ``dtype``?"""
    raise NotImplementedError("lesson 8.5")


class LossScaler:
    """Dynamic loss scaling for fp16 (Micikevicius et al., 2018): multiply
    the loss by ``scale`` so small gradients don't underflow; if any
    gradient overflows to inf or NaN, skip the step and halve the scale;
    after ``growth_interval`` good steps in a row, double it."""

    def __init__(self, scale=2.0 ** 16, growth_interval=2000):
        raise NotImplementedError("lesson 8.5")

    def scale_loss(self, loss):
        raise NotImplementedError("lesson 8.5")

    def unscale(self, grads):
        """Divide gradients by the scale in place; return True if any is
        not finite."""
        raise NotImplementedError("lesson 8.5")

    def update(self, found_inf):
        raise NotImplementedError("lesson 8.5")


def fp16_gradient(true_grad, scale):
    """What survives of ``true_grad`` (a float32 tensor) when the backward
    pass runs in fp16 with the loss multiplied by ``scale``: cast scaled to
    fp16, then back to fp32 and unscaled."""
    raise NotImplementedError("lesson 8.5")


def train_step(params, grads_fp16, master, opt, scaler):
    """One mixed-precision step: fp16 ``grads_fp16`` (still scaled) are
    copied into the fp32 ``master`` weights' grads, unscaled and checked;
    on overflow the step is skipped. Then the fp16 ``params`` are refreshed
    from the master copy. Returns True if the step was taken."""
    raise NotImplementedError("lesson 8.5")
