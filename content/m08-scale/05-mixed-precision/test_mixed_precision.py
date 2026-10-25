import torch

from m08.precision import LossScaler, fp16_gradient, spacing, train_step, update_survives


def test_spacing_and_lost_updates():
    assert spacing(1.0, torch.float16) == 2 ** -10
    assert spacing(1.0, torch.bfloat16) == 2 ** -7
    assert spacing(1.0, torch.float32) == 2 ** -23
    assert not update_survives(1.0, 1e-3, torch.bfloat16), "a small update to a bf16 weight vanishes"
    assert update_survives(1.0, 1e-3, torch.float32)


def test_fp16_underflow_and_scaling():
    g = torch.tensor([1e-8, 1e-3])
    assert fp16_gradient(g, 1.0)[0] == 0, "1e-8 is below fp16's smallest subnormal"
    assert torch.allclose(fp16_gradient(g, 2.0 ** 16), g, rtol=1e-2), "scaled up first, it survives"


def test_scaler_logic():
    s = LossScaler(scale=1024.0, growth_interval=3)
    assert s.scale_loss(torch.tensor(2.0)).item() == 2048
    grads = [torch.tensor([1024.0, 2048.0])]
    assert s.unscale(grads) is False and torch.equal(grads[0], torch.tensor([1.0, 2.0]))
    assert s.unscale([torch.tensor([float("inf")])]) is True
    s.update(True)
    assert s.scale == 512 and s.good_steps == 0
    for _ in range(3):
        s.update(False)
    assert s.scale == 1024, "doubles after growth_interval good steps"


def test_train_step_skips_overflow():
    master = [torch.nn.Parameter(torch.tensor([1.0, 2.0]))]
    params = [torch.tensor([1.0, 2.0]).half()]
    opt = torch.optim.SGD(master, lr=0.5)
    s = LossScaler(scale=4.0)
    assert train_step(params, [torch.tensor([4.0, -8.0]).half()], master, opt, s)
    assert torch.allclose(master[0].data, torch.tensor([0.5, 3.0])) and params[0].dtype == torch.float16
    assert torch.allclose(params[0].float(), master[0].data)
    before = master[0].data.clone()
    assert not train_step(params, [torch.tensor([float("inf"), 1.0]).half()], master, opt, s)
    assert torch.equal(master[0].data, before) and s.scale == 2.0, "skip the step, halve the scale"
