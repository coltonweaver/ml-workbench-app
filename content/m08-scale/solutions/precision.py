"""Mixed precision: 16-bit math with 32-bit safety nets."""

import torch


def spacing(x, dtype):
    """The gap between ``x`` and the next representable number in ``dtype``."""
    t = torch.tensor(x, dtype=dtype)
    return (torch.nextafter(t, torch.tensor(float("inf"), dtype=dtype)) - t).item()


def update_survives(weight, update, dtype):
    """Does weight + update differ from weight when stored in ``dtype``?"""
    w = torch.tensor(weight, dtype=dtype)
    return bool((w + torch.tensor(update, dtype=dtype)) != w)


class LossScaler:
    """Dynamic loss scaling for fp16 (Micikevicius et al., 2018): multiply
    the loss by ``scale`` so small gradients don't underflow; if any
    gradient overflows to inf or NaN, skip the step and halve the scale;
    after ``growth_interval`` good steps in a row, double it."""

    def __init__(self, scale=2.0 ** 16, growth_interval=2000):
        self.scale = scale
        self.growth_interval = growth_interval
        self.good_steps = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale(self, grads):
        """Divide gradients by the scale in place; return True if any is
        not finite."""
        bad = False
        for g in grads:
            g /= self.scale
            bad |= not torch.isfinite(g).all().item()
        return bad

    def update(self, found_inf):
        if found_inf:
            self.scale /= 2
            self.good_steps = 0
        else:
            self.good_steps += 1
            if self.good_steps == self.growth_interval:
                self.scale *= 2
                self.good_steps = 0


def fp16_gradient(true_grad, scale):
    """What survives of ``true_grad`` (a float32 tensor) when the backward
    pass runs in fp16 with the loss multiplied by ``scale``: cast scaled to
    fp16, then back to fp32 and unscaled."""
    return (true_grad * scale).half().float() / scale


def train_step(params, grads_fp16, master, opt, scaler):
    """One mixed-precision step: fp16 ``grads_fp16`` (still scaled) are
    copied into the fp32 ``master`` weights' grads, unscaled and checked;
    on overflow the step is skipped. Then the fp16 ``params`` are refreshed
    from the master copy. Returns True if the step was taken."""
    for m, g in zip(master, grads_fp16):
        m.grad = g.float()
    found_inf = scaler.unscale([m.grad for m in master])
    scaler.update(found_inf)
    if found_inf:
        return False
    opt.step()
    with torch.no_grad():
        for p, m in zip(params, master):
            p.copy_(m.half())
    return True
