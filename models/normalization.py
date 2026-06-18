# models/normalization.py
# ─────────────────────────────────────────────────────────────────────────────
# Drop-in replacements for RevIN.
# Every class exposes the same two-call interface:
#
#   x = norm(x, 'norm')              # before model
#   x = norm(x, 'denorm', slice)     # after model
#
# So tsAMD.py needs zero changes beyond swapping the class.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn


# ── 1. RevIN (original baseline) ─────────────────────────────────────────────
class RevIN(nn.Module):
    """
    Original RevIN from AMD paper.
    Per-sample, per-channel mean/std normalization.
    Learnable affine parameters (gamma, beta).
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode, target_slice=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        return x

    def _get_statistics(self, x):
        dim = tuple(range(1, x.ndim - 1))
        self.mean  = x.mean(dim=dim, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=dim, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = (x - self.affine_bias[target_slice]) / (self.affine_weight[target_slice] + self.eps**2)
        x = x * self.stdev[:, :, target_slice] + self.mean[:, :, target_slice]
        return x


# ── 2. Global Z-Score Normalization ──────────────────────────────────────────
class GlobalZScore(nn.Module):
    """
    Z-score using statistics computed across the ENTIRE input window.
    Single mean and std per sample (not per channel).
    Simpler than RevIN — no learnable parameters.

    Good for: datasets where all channels share the same scale
    Risk for: datasets with channels on very different scales (e.g. Dengue
              mixes disease counts with temperature and emissions)
    """
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode, target_slice=None):
        # x: [B, seq_len, C]
        if mode == 'norm':
            self.mean  = x.mean(dim=(1, 2), keepdim=True).detach()
            self.stdev = torch.sqrt(
                x.var(dim=(1, 2), keepdim=True, unbiased=False) + self.eps
            ).detach()
            return (x - self.mean) / self.stdev

        elif mode == 'denorm':
            # stdev and mean are [B,1,1] so slicing on dim-2 still works
            return x * self.stdev + self.mean


# ── 3. Robust Normalization (Median / IQR) ────────────────────────────────────
class RobustNorm(nn.Module):
    """
    (x - median) / IQR   per channel per sample.
    Resistant to outlier spikes — useful for Dengue outbreak data
    where a sudden surge should not distort the normalization.

    Good for: epidemiological data with sudden spikes (Dengue)
    Risk for: data with very small IQR (near-constant channels)
    """
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode, target_slice=None):
        # x: [B, seq_len, C]
        if mode == 'norm':
            # compute per-channel median and IQR across time
            self.median = x.median(dim=1, keepdim=True).values.detach()  # [B,1,C]
            q75 = torch.quantile(x, 0.75, dim=1, keepdim=True).detach()  # [B,1,C]
            q25 = torch.quantile(x, 0.25, dim=1, keepdim=True).detach()
            self.iqr = (q75 - q25).clamp(min=self.eps)                   # [B,1,C]
            return (x - self.median) / self.iqr

        elif mode == 'denorm':
            if target_slice is not None:
                return x * self.iqr[:, :, target_slice] + self.median[:, :, target_slice]
            return x * self.iqr + self.median


# ── 4. Min-Max Normalization ──────────────────────────────────────────────────
class MinMaxNorm(nn.Module):
    """
    (x - min) / (max - min)  →  scales each channel to [0, 1].
    Fully invertible, preserves shape of the signal.
    No learnable parameters.

    Good for: Exchange Rate — bounded daily changes, no extreme outliers
    Risk for: test data with values outside training min/max range
    """
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode, target_slice=None):
        # x: [B, seq_len, C]
        if mode == 'norm':
            self.xmin = x.min(dim=1, keepdim=True).values.detach()   # [B,1,C]
            self.xmax = x.max(dim=1, keepdim=True).values.detach()   # [B,1,C]
            self.rng  = (self.xmax - self.xmin).clamp(min=self.eps)
            return (x - self.xmin) / self.rng

        elif mode == 'denorm':
            if target_slice is not None:
                return x * self.rng[:, :, target_slice] + self.xmin[:, :, target_slice]
            return x * self.rng + self.xmin


# ── 5. Log + RevIN (Two-stage) ────────────────────────────────────────────────
class LogRevIN(nn.Module):
    """
    Two-stage:  log(x + shift)  →  RevIN normalization.

    Stage 1 — log compress:  reduces multiplicative variance
              (e.g. Dengue cases grow multiplicatively during outbreaks)
    Stage 2 — RevIN:         per-sample mean/std centering with learnable affine

    Good for: Dengue and any data with multiplicative seasonality or
              heavy right-skewed distributions.
    Risk for: data with zero or negative values (shift handles zeros).
    """
    def __init__(self, num_features, eps=1e-5, affine=True, shift=1.0):
        super().__init__()
        self.shift = shift
        self.revin = RevIN(num_features, eps=eps, affine=affine)

    def forward(self, x, mode, target_slice=None):
        if mode == 'norm':
            # ensure all values are positive before log
            self.shift_val = max(self.shift, float(-x.min().item()) + 1.0) \
                             if x.min().item() <= 0 else self.shift
            x = torch.log(x + self.shift_val)
            return self.revin(x, 'norm')

        elif mode == 'denorm':
            x = self.revin(x, 'denorm', target_slice)
            return torch.exp(x) - self.shift_val


# ── 6. Adaptive Normalization ─────────────────────────────────────────────────
class AdaptiveNorm(nn.Module):
    """
    Learnable weighted combination of instance norm and batch norm.
    From: 'Non-stationary Transformers' (Liu et al. 2022).

    alpha (learned) controls how much instance-level vs batch-level
    statistics dominate — adapts automatically per dataset.

    Good for: non-stationary data where the best normalization
              strategy changes across channels or time periods.
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        # learnable blend weight per channel
        self.alpha = nn.Parameter(torch.ones(num_features) * 0.5)
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias   = nn.Parameter(torch.zeros(num_features))
        self.affine = affine

    def forward(self, x, mode, target_slice=None):
        # x: [B, seq_len, C]
        if mode == 'norm':
            alpha = torch.sigmoid(self.alpha).unsqueeze(0).unsqueeze(0)  # [1,1,C]

            # instance statistics (per sample)
            inst_mean = x.mean(dim=1, keepdim=True)
            inst_std  = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)

            # batch statistics (across batch)
            batch_mean = x.mean(dim=(0, 1), keepdim=True)
            batch_std  = torch.sqrt(x.var(dim=(0, 1), keepdim=True, unbiased=False) + self.eps)

            # weighted blend
            self.mean  = (alpha * inst_mean  + (1 - alpha) * batch_mean).detach()
            self.stdev = (alpha * inst_std   + (1 - alpha) * batch_std ).detach()

            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.weight + self.bias
            return x

        elif mode == 'denorm':
            if self.affine:
                w = self.weight[target_slice] if target_slice else self.weight
                b = self.bias[target_slice]   if target_slice else self.bias
                x = (x - b) / (w + self.eps**2)
            if target_slice is not None:
                return x * self.stdev[:, :, target_slice] + self.mean[:, :, target_slice]
            return x * self.stdev + self.mean


# ── 7. No Normalization (ablation baseline) ───────────────────────────────────
class NoNorm(nn.Module):
    """
    Pass-through — no normalization applied.
    Use ONLY for ablation study to confirm normalization matters.
    WARNING: likely to produce NaN on longer pred_len (as seen in ETTh2 experiments).
    """
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()

    def forward(self, x, mode, target_slice=None):
        return x


# ── Registry — maps name string → class ──────────────────────────────────────
NORM_REGISTRY = {
    'revin'    : RevIN,
    'zscore'   : GlobalZScore,
    'robust'   : RobustNorm,
    'minmax'   : MinMaxNorm,
    'log_revin': LogRevIN,
    'adaptive' : AdaptiveNorm,
    'none'     : NoNorm,
}


def get_norm(norm_type, num_features, eps=1e-5, affine=True):
    """
    Factory function.
    Usage:
        from models.normalization import get_norm
        self.rev_norm = get_norm(args.norm_type, num_features)
    """
    norm_type = norm_type.lower()
    if norm_type not in NORM_REGISTRY:
        raise ValueError(
            f"Unknown norm_type '{norm_type}'. "
            f"Choose from: {list(NORM_REGISTRY.keys())}"
        )
    cls = NORM_REGISTRY[norm_type]
    # NoNorm and GlobalZScore don't use affine
    if cls in (NoNorm, GlobalZScore):
        return cls(num_features, eps=eps)
    return cls(num_features, eps=eps, affine=affine)