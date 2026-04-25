import math

import torch

from sdm.losses.wristband_gaussian import (
    W2ToStandardNormalSq,
    WristbandGaussianLoss,
)


def test_w2_to_standard_normal_zero_for_gaussian():
    torch.manual_seed(0)
    x = torch.randn(2048, 8)
    val = float(W2ToStandardNormalSq(x))
    # Sample W2 to N(0,I) is small but not exactly zero with finite samples.
    assert val < 0.5


def test_w2_to_standard_normal_large_for_shifted():
    x = torch.randn(2048, 8) + 5.0
    val = float(W2ToStandardNormalSq(x))
    assert val > 50.0  # roughly d * mu^2 = 8 * 25 = 200


def test_wristband_loss_finite_and_differentiable():
    torch.manual_seed(0)
    x = torch.randn(64, 8, requires_grad=True)
    loss_fn = WristbandGaussianLoss()  # uncalibrated
    out = loss_fn(x)
    assert torch.isfinite(out.total)
    out.total.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_wristband_loss_separates_gaussian_from_uniform():
    torch.manual_seed(0)
    loss_fn = WristbandGaussianLoss()
    g = torch.randn(256, 8)
    # Stretched uniform with same support as N(0,I) at d=8 (||x||~sqrt(d)~2.8):
    u = (torch.rand(256, 8) - 0.5) * 6.0
    g_loss = float(loss_fn(g).total)
    u_loss = float(loss_fn(u).total)
    assert u_loss > g_loss


def test_wristband_loss_calibrated_zero_mean_total_on_null():
    torch.manual_seed(0)
    loss_fn = WristbandGaussianLoss(calibration_shape=(64, 8), calibration_reps=128)
    accum = 0.0
    for _ in range(32):
        x = torch.randn(64, 8)
        accum += float(loss_fn(x).total)
    mean = accum / 32.0
    # After calibration the total should hover near 0 on null samples.
    assert abs(mean) < 1.0
