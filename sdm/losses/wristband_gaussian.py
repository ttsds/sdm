"""Wristband Gaussian loss for forcing latents toward N(0, I).

Direct port of `C_WristbandGaussianLoss` and its helpers from
`mvparakhin/ml-tidbits` (MIT-licensed, Copyright (c) 2025 Mikhail Parakhin).
Renamed to sdm style and stripped of unrelated code in EmbedModels.py.
The upstream documentation in /tmp/ml-tidbits/docs/wristband.md describes
the math in detail.

Used as an *ablation* in sdm finetuning — applied at the TTSDS-output layer,
the loss makes those latents directly suitable for closed-form W2/Frechet
distances against N(0, I) reference noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from scipy.special import gammaln, iv, ive

__all__ = [
    "GaussianLossConfig",
    "S_LossComponents",
    "SpectralNeumannCoefficients",
    "W2ToStandardNormalSq",
    "WristbandGaussianLoss",
]


@dataclass
class GaussianLossConfig:
    """Plumbing config for applying `WristbandGaussianLoss` in a training loop."""

    enabled: bool = False
    weight: float = 0.1
    beta: float = 8.0
    lambda_rad: float = 0.1
    lambda_mom: float = 1.0
    # moment="w2" requires torch.linalg.eigvalsh which hangs XLA compilation
    # for hours on TPU (Jacobi rotation lowering); use "kl_diag" by default.
    moment: str = "kl_diag"
    calibrate: bool = True  # disable if batch shape varies a lot


def _eps_for_dtype(dtype: torch.dtype, large: bool = False) -> float:
    eps = torch.finfo(dtype).eps
    return math.sqrt(eps) if large else eps


@dataclass(frozen=True)
class SpectralNeumannCoefficients:
    lam_0: float
    lam_1: float
    a_k: torch.Tensor


def _log_bessel_ive(order: float, c: float) -> float:
    val = float(ive(order, c))
    if val > 0.0 and math.isfinite(val):
        return math.log(val)
    val = float(iv(order, c))
    if val > 0.0 and math.isfinite(val):
        return math.log(val) - c
    return float(-c + order * math.log(c / 2.0) - gammaln(order + 1.0))


def _angular_eigenvalue_l(d: int, beta: float, alpha: float, ell: int) -> float:
    if d < 3:
        raise ValueError("Spectral Neumann path requires d >= 3.")
    nu = 0.5 * (d - 2)
    c = 2.0 * beta * (alpha**2)
    log_prefactor = float(gammaln(nu + 1.0) + nu * (math.log(2.0) - math.log(c)))
    log_lambda = log_prefactor + _log_bessel_ive(nu + ell, c)
    if log_lambda < math.log(float(torch.finfo(torch.float64).tiny)):
        return 0.0
    lam = math.exp(log_lambda)
    if not math.isfinite(lam) or lam < 0.0:
        raise FloatingPointError(f"Invalid angular eigenvalue: {lam}.")
    return lam


def _build_spectral_neumann_coefficients(
    d: int,
    beta: float,
    alpha: float,
    k_modes: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SpectralNeumannCoefficients:
    if k_modes < 1:
        raise ValueError("k_modes must be >= 1.")
    lam_0 = _angular_eigenvalue_l(d, beta, alpha, ell=0)
    lam_1 = _angular_eigenvalue_l(d, beta, alpha, ell=1)
    beta_t = torch.as_tensor(beta, device=device, dtype=dtype)
    k_range = torch.arange(k_modes, device=device, dtype=dtype)
    pref = torch.sqrt(torch.pi / beta_t)
    a_k = pref * torch.where(
        k_range == 0,
        torch.ones_like(k_range),
        2.0 * torch.exp(-(torch.pi**2) * k_range.square() / (4.0 * beta_t)),
    )
    return SpectralNeumannCoefficients(lam_0=lam_0, lam_1=lam_1, a_k=a_k)


def W2ToStandardNormalSq(x: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    """Squared 2-Wasserstein distance from sample Gaussian fit to N(0, I)."""
    if x.ndim < 2:
        raise ValueError(f"Expected x.ndim>=2 with shape (..., B, d), got {tuple(x.shape)}")
    b = x.shape[-2]
    d = x.shape[-1]
    if b < 2:
        raise ValueError("Need B>=2 for covariance (denominator B-1).")

    work_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    xw = x.to(dtype=work_dtype)

    mu = xw.mean(dim=-2, keepdim=True)
    xc = xw - mu
    mu2 = mu.squeeze(-2).square().sum(dim=-1)
    denom = float(b - 1)

    if d <= b:
        m = (xc.transpose(-1, -2) @ xc) / denom
        m_dim = d
    else:
        m = (xc @ xc.transpose(-1, -2)) / denom
        m_dim = b
    m = 0.5 * (m + m.transpose(-1, -2))

    eig = torch.linalg.eigvalsh(m).clamp_min(0.0)
    sqrt_eig = torch.sqrt(eig + _eps_for_dtype(eig.dtype))
    bw2 = (sqrt_eig - 1.0).square().sum(dim=-1)
    if d > m_dim:
        bw2 = bw2 + (d - m_dim)
    loss = mu2 + bw2

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError("reduction must be one of {'none','mean','sum'}")


class S_LossComponents(NamedTuple):
    total: torch.Tensor
    rep: torch.Tensor
    rad: torch.Tensor
    ang: torch.Tensor
    mom: torch.Tensor


class WristbandGaussianLoss:
    """Batch loss encouraging N(0, I) via wristband repulsion. See upstream docs."""

    def __init__(
        self,
        *,
        beta: float = 8.0,
        alpha: float | None = None,
        angular: str = "chordal",
        reduction: str = "per_point",
        spectral: bool = False,
        k_modes: int = 6,
        lambda_rad: float = 0.1,
        lambda_ang: float = 0.0,
        moment: str = "w2",
        lambda_mom: float = 1.0,
        calibration_shape: tuple[int, ...] | None = None,
        calibration_reps: int = 1024,
        calibration_device: str | torch.device = "cpu",
        calibration_dtype: torch.dtype = torch.float32,
    ):
        if beta <= 0:
            raise ValueError("beta must be > 0")
        if angular not in ("chordal", "geodesic"):
            raise ValueError("angular must be 'chordal' or 'geodesic'")
        if reduction not in ("per_point", "global"):
            raise ValueError("reduction must be 'per_point' or 'global'")
        if moment not in ("mu_only", "kl_diag", "kl_full", "jeff_diag", "jeff_full", "w2"):
            raise ValueError("invalid moment penalty type")
        if int(k_modes) < 1:
            raise ValueError("k_modes must be >= 1")
        if spectral and angular != "chordal":
            raise ValueError("spectral=True currently supports only angular='chordal'")
        if spectral and reduction != "global":
            raise ValueError("spectral=True currently supports only reduction='global'")
        if spectral and lambda_ang != 0.0:
            raise ValueError("spectral=True currently supports only lambda_ang=0")

        self.beta = float(beta)
        self.angular = angular
        self.reduction = reduction
        self.spectral = bool(spectral)
        self.k_modes = int(k_modes)

        if alpha is None:
            alpha = math.sqrt(1.0 / 12.0) if angular == "chordal" else math.sqrt(2.0 / (3.0 * math.pi**2))
        self.alpha = float(alpha)
        self.beta_alpha2 = self.beta * (self.alpha**2)

        self.lambda_rad = float(lambda_rad)
        self.lambda_ang = float(lambda_ang)
        self.moment = moment
        self.lambda_mom = float(lambda_mom)
        self.eps = 1.0e-12
        self.clamp_cos = 1.0e-6
        self._spectral_cache: dict = {}
        self._spectral_cache_signature: tuple | None = None
        self._q_cache: dict = {}
        self._calibration_shape = (
            tuple(int(v) for v in calibration_shape[-2:])
            if calibration_shape is not None
            else None
        )

        self.mean_rep = self.mean_rad = self.mean_ang = self.mean_mom = 0.0
        self.std_rep = self.std_rad = self.std_ang = self.std_mom = 1.0
        self.std_total = 1.0

        if calibration_shape is not None:
            self._calibrate(
                calibration_shape, calibration_reps, calibration_device, calibration_dtype
            )

    def _spectral_coefficients(self, d: int, device: torch.device, dtype: torch.dtype):
        if d < 3:
            raise ValueError("spectral=True requires d >= 3")
        signature = (self.beta, self.alpha, self.k_modes)
        if self._spectral_cache_signature != signature:
            self._spectral_cache.clear()
            self._spectral_cache_signature = signature
        key = (int(d), device, dtype)
        coeffs = self._spectral_cache.get(key)
        if coeffs is None:
            coeffs = _build_spectral_neumann_coefficients(
                d=int(d),
                beta=self.beta,
                alpha=self.alpha,
                k_modes=self.k_modes,
                device=device,
                dtype=dtype,
            )
            self._spectral_cache[key] = coeffs
        return coeffs

    def _moment_penalty(self, xw: torch.Tensor) -> torch.Tensor:
        batch_shape = xw.shape[:-2]
        if self.lambda_mom == 0.0:
            return xw.new_zeros(batch_shape)

        n = int(xw.shape[-2])
        d = int(xw.shape[-1])
        n_f, d_f = float(n), float(d)
        eps = self.eps

        if self.moment == "w2":
            return W2ToStandardNormalSq(xw, reduction="none") / d_f

        mu = xw.mean(dim=-2)
        if self.moment == "mu_only":
            return mu.square().mean(dim=-1)

        xc = xw - mu[..., None, :]
        if self.moment == "jeff_diag":
            var = xc.square().sum(dim=-2) / (n_f - 1.0)
            v = var + eps
            inv_v = v.reciprocal()
            mu2 = mu.square()
            return 0.25 * (v + inv_v + mu2 + mu2 * inv_v - 2.0).mean(dim=-1)

        if self.moment == "kl_diag":
            var = xc.square().sum(dim=-2) / (n_f - 1.0)
            return 0.5 * (var + mu.square() - 1.0 - torch.log(var + eps)).mean(dim=-1)

        eps_cov = (
            max(eps, 1.0e-6)
            if xw.dtype == torch.float32
            else max(eps, float(torch.finfo(xw.dtype).eps))
        )
        if self.moment == "jeff_full":
            cov = (xc.transpose(-1, -2) @ xc) / (n_f - 1.0)
            eye = torch.eye(d, device=xw.device, dtype=xw.dtype)
            cov = cov + eps_cov * eye
            chol, _ = torch.linalg.cholesky_ex(cov)
            tr = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            inv_cov = torch.cholesky_solve(eye, chol)
            tr_inv = inv_cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            mu_col = mu[..., :, None]
            sol_mu = torch.cholesky_solve(mu_col, chol)
            mu_inv_mu = (mu_col * sol_mu).sum(dim=(-2, -1))
            mu2_sum = mu.square().sum(dim=-1)
            return 0.25 * (tr + tr_inv + mu2_sum + mu_inv_mu - 2.0 * d_f) / d_f

        # kl_full
        eye = torch.eye(d, device=xw.device, dtype=xw.dtype)
        cov = (xc.transpose(-1, -2) @ xc) / (n_f - 1.0) + eps_cov * eye
        chol, _ = torch.linalg.cholesky_ex(cov)
        diag = chol.diagonal(dim1=-2, dim2=-1)
        logdet = 2.0 * torch.log(diag).sum(dim=-1)
        tr = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        mu2 = mu.square().sum(dim=-1)
        return 0.5 * (tr + mu2 - d_f - logdet) / d_f

    def _wristband_map(self, xw: torch.Tensor):
        d_f = float(xw.shape[-1])
        s = xw.square().sum(dim=-1)
        u = xw * torch.rsqrt(s.clamp_min(self.eps))[..., :, None]
        a_df = s.new_tensor(0.5 * d_f)
        tiny = float(torch.finfo(s.dtype).tiny)
        t = torch.special.gammainc(a_df, 0.5 * (s + tiny))
        return u, t

    def _normal_icdf(self, u: torch.Tensor) -> torch.Tensor:
        eps_u = max(self.eps, float(torch.finfo(u.dtype).eps))
        u = u.clamp(eps_u, 1.0 - eps_u)
        return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)

    def _target_quantiles(self, n: int, device: torch.device, dtype: torch.dtype):
        key = (int(n), device, dtype)
        pair = self._q_cache.get(key)
        if pair is None:
            q_u = (torch.arange(int(n), device=device, dtype=dtype) + 0.5) / float(n)
            q_g = math.sqrt(2.0) * torch.erfinv(2.0 * q_u - 1.0)
            self._q_cache[key] = (q_u, q_g)
            return q_u, q_g
        return pair

    def _radial_loss(self, z: torch.Tensor, *, gaussian_input: bool) -> torch.Tensor:
        z_sorted = z.sort(dim=-1).values
        q_u, q_g = self._target_quantiles(int(z.shape[-1]), z.device, z.dtype)
        if gaussian_input:
            u_sorted = 0.5 * (1.0 + torch.erf(z_sorted / math.sqrt(2.0)))
            g_sorted = z_sorted
        else:
            u_sorted = z_sorted
            g_sorted = self._normal_icdf(z_sorted)
        loss_u = 12 * (u_sorted - q_u).square().mean(dim=-1)
        loss_g = (g_sorted - q_g).square().mean(dim=-1)
        return 0.5 * (loss_u + loss_g)

    def _angular_exponent(self, u: torch.Tensor) -> torch.Tensor:
        g = (u @ u.transpose(-1, -2)).clamp(-1.0, 1.0)
        if self.angular == "chordal":
            e_ang = (2.0 * self.beta_alpha2) * (g - 1.0)
            e_ang.diagonal(dim1=-2, dim2=-1).zero_()
            return e_ang
        theta = torch.acos(g.clamp(-1.0 + self.clamp_cos, 1.0 - self.clamp_cos))
        ang2 = theta.square()
        ang2.diagonal(dim1=-2, dim2=-1).zero_()
        return -self.beta_alpha2 * ang2

    def _angular_uniformity(self, e_ang: torch.Tensor, n_f: float) -> torch.Tensor:
        if self.reduction == "per_point":
            row_sum = torch.exp(e_ang).sum(dim=-1) - 1.0
            mean_k = row_sum / (n_f - 1.0)
            return torch.log(mean_k + self.eps).mean(dim=-1) / self.beta
        total = torch.exp(e_ang).sum(dim=(-2, -1)) - n_f
        mean_k = total / (n_f * (n_f - 1.0))
        return torch.log(mean_k + self.eps) / self.beta

    def _pairwise_repulsion(
        self, e_ang: torch.Tensor, t: torch.Tensor, n_f: float
    ) -> torch.Tensor:
        tc = t[..., :, None]
        tr = t[..., None, :]
        diff0 = tc - tr
        diff1 = tc + tr
        diff2 = diff1 - 2.0
        if self.reduction == "per_point":
            row_sum = torch.exp(torch.addcmul(e_ang, diff0, diff0, value=-self.beta)).sum(dim=-1)
            row_sum += torch.exp(torch.addcmul(e_ang, diff1, diff1, value=-self.beta)).sum(dim=-1)
            row_sum += torch.exp(torch.addcmul(e_ang, diff2, diff2, value=-self.beta)).sum(dim=-1)
            row_sum -= 1.0
            mean_k = row_sum / (3.0 * n_f - 1.0)
            return torch.log(mean_k + self.eps).mean(dim=-1) / self.beta
        total = torch.exp(torch.addcmul(e_ang, diff0, diff0, value=-self.beta)).sum(dim=(-2, -1))
        total += torch.exp(torch.addcmul(e_ang, diff1, diff1, value=-self.beta)).sum(dim=(-2, -1))
        total += torch.exp(torch.addcmul(e_ang, diff2, diff2, value=-self.beta)).sum(dim=(-2, -1))
        total -= n_f
        mean_k = total / (3.0 * n_f * n_f - n_f)
        return torch.log(mean_k + self.eps) / self.beta

    def _spectral_repulsion(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        d = int(u.shape[-1])
        n_f, d_f = float(u.shape[-2]), float(d)
        coeffs = self._spectral_coefficients(d, u.device, u.dtype)
        a_k = coeffs.a_k.to(device=t.device, dtype=t.dtype)
        k_range = torch.arange(int(a_k.shape[0]), device=t.device, dtype=t.dtype)
        cos_mat = torch.cos(torch.pi * t[..., :, None] * k_range)
        c_0k = cos_mat.mean(dim=-2)
        c_1k = (math.sqrt(d_f) / n_f) * (u.transpose(-1, -2) @ cos_mat)

        lam_0_t = torch.as_tensor(coeffs.lam_0, device=t.device, dtype=t.dtype)
        lam_1_t = torch.as_tensor(coeffs.lam_1, device=t.device, dtype=t.dtype)

        e_total = lam_0_t * (a_k * c_0k.square()).sum(dim=-1)
        e_total += lam_1_t * (a_k * c_1k.square()).sum(dim=(-2, -1))
        norm_const = torch.clamp_min(lam_0_t * a_k[0], self.eps)
        return torch.log(torch.clamp_min(e_total / norm_const, self.eps)) / self.beta

    def _calibrate(self, shape, reps, device, dtype):
        if len(shape) < 2:
            raise ValueError(f"Expected shape with at least 2 dimensions, got {tuple(shape)}")
        n, d = int(shape[-2]), int(shape[-1])
        if n < 2 or d < 1 or reps < 2:
            raise ValueError("invalid calibration parameters")

        all_rep, all_rad, all_ang, all_mom = [], [], [], []
        with torch.no_grad():
            for _ in range(int(reps)):
                x_gauss = torch.randn(int(n), int(d), device=device, dtype=dtype)
                rep, rad, ang, mom = self._compute(x_gauss)
                all_rep.append(rep)
                all_rad.append(rad)
                all_ang.append(ang)
                all_mom.append(mom)

        eps_cal = float(_eps_for_dtype(dtype, True))

        def _stats(vals):
            t = torch.stack(vals)
            return t, float(t.mean()), math.sqrt(max(float(t.var(unbiased=True)), eps_cal))

        t_rep, self.mean_rep, self.std_rep = _stats(all_rep)
        t_rad, self.mean_rad, self.std_rad = _stats(all_rad)
        t_ang, self.mean_ang, self.std_ang = _stats(all_ang)
        t_mom, self.mean_mom, self.std_mom = _stats(all_mom)

        n_rep = (t_rep - self.mean_rep) / self.std_rep
        n_rad = (t_rad - self.mean_rad) / self.std_rad
        n_ang = self.lambda_ang * (t_ang - self.mean_ang) / self.std_ang
        n_mom = self.lambda_mom * (t_mom - self.mean_mom) / self.std_mom
        total = n_rep + self.lambda_rad * n_rad + n_ang + n_mom
        self.std_total = math.sqrt(max(float(total.var(unbiased=True)), eps_cal))

    def _compute(self, x: torch.Tensor):
        if x.ndim < 2:
            raise ValueError(f"Expected x.ndim>=2, got {tuple(x.shape)}")
        n = int(x.shape[-2])
        d = int(x.shape[-1])
        batch_shape = x.shape[:-2]
        if n < 2 or d < 1:
            z = x.sum(dim=(-2, -1)) * 0.0
            return z, z, z, z

        wdtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        xw = x.to(wdtype)
        n_f = float(n)

        mom_pen = self._moment_penalty(xw)
        if d == 1:
            z = xw.new_zeros(batch_shape)
            rep_loss = self._radial_loss(xw.squeeze(-1), gaussian_input=True)
            return rep_loss, z, z, mom_pen

        u, t = self._wristband_map(xw)
        rad_loss = xw.new_zeros(batch_shape)
        if self.lambda_rad != 0.0:
            rad_loss = self._radial_loss(t, gaussian_input=False)

        if self.spectral:
            rep_loss = self._spectral_repulsion(u, t)
            ang_loss = xw.new_zeros(batch_shape)
        else:
            e_ang = self._angular_exponent(u)
            ang_loss = xw.new_zeros(batch_shape)
            if self.lambda_ang != 0.0:
                ang_loss = self._angular_uniformity(e_ang, n_f)
            rep_loss = self._pairwise_repulsion(e_ang, t, n_f)

        return rep_loss, rad_loss, ang_loss, mom_pen

    def __call__(self, x: torch.Tensor) -> S_LossComponents:
        if self._calibration_shape is not None:
            if x.ndim < 2:
                raise ValueError("calibrated loss expects x.ndim>=2")
            shape = tuple(int(v) for v in x.shape[-2:])
            if shape != self._calibration_shape:
                raise ValueError(
                    f"calibrated loss expects x.shape[-2:]=={self._calibration_shape}, got {shape}"
                )
        comp_rep, comp_rad, comp_ang, comp_mom = self._compute(x)
        norm_rep = (comp_rep - self.mean_rep) / self.std_rep
        norm_rad = (comp_rad - self.mean_rad) / self.std_rad
        norm_ang = (comp_ang - self.mean_ang) / self.std_ang
        norm_mom = (comp_mom - self.mean_mom) / self.std_mom
        total = (
            norm_rep
            + self.lambda_rad * norm_rad
            + self.lambda_ang * norm_ang
            + self.lambda_mom * norm_mom
        ) / self.std_total
        return S_LossComponents(
            total.mean(),
            norm_rep.mean(),
            norm_rad.mean(),
            norm_ang.mean(),
            norm_mom.mean(),
        )
