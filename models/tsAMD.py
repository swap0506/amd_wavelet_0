import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import RevIN, DDI, MDM
from models.tsmoe import AMS


class AMD(nn.Module):
    def __init__(self, input_shape, pred_len, n_block, dropout, patch, k, c,
                 alpha, target_slice, norm=True, layernorm=True,
                 lifting_levels=3, lifting_kernel_size=7,
                 regu_details=0.0, regu_approx=0.0):
        super(AMD, self).__init__()

        self.target_slice = target_slice
        self.norm = norm
        self.seq_len = input_shape[0]

        if self.norm:
            self.rev_norm = RevIN(input_shape[-1])

        # MDM: adaptive multi-level wavelet decomposition
        # FIX: use correct keyword args that MDM.__init__ actually accepts
        self.pastmixing = MDM(
            input_shape,
            lifting_levels=lifting_levels,
            lifting_kernel_size=lifting_kernel_size,
            regu_details=regu_details,
            regu_approx=regu_approx,
        )

        self.fc_blocks = nn.ModuleList([
            DDI(input_shape, dropout=dropout, patch=patch,
                alpha=alpha, layernorm=layernorm)
            for _ in range(n_block)
        ])

        self.moe = AMS(input_shape, pred_len, ff_dim=1024,
                       dropout=dropout, num_experts=4, top_k=2)

        # Project coarsest approximation (seq_len // 2^levels) up to seq_len
        # so it can serve as time_embedding for AMS gating
        coarse_len = self.seq_len // (2 ** lifting_levels)
        self.gate_proj = nn.Linear(coarse_len, self.seq_len)

    def _moving_avg(self, x, kernel_size=25):
        """Subtract a simple moving-average trend before wavelet decomposition."""
        # x: [B, C, L]
        pad = kernel_size // 2
        x_padded = F.pad(x, (pad, pad), mode='replicate')
        trend = F.avg_pool1d(x_padded, kernel_size=kernel_size,
                             stride=1, padding=0)
        # align length
        trend = trend[:, :, :x.shape[-1]]
        seasonal = x - trend
        return seasonal, trend

    def forward(self, x):
        # x: [B, seq_len, feature_num]

        if self.norm:
            x = self.rev_norm(x, 'norm')

        x = torch.transpose(x, 1, 2)
        # x: [B, feature_num, seq_len]

        # --- Step 1: Trend / seasonal split ---
        # Give the wavelet only the oscillatory part; keep trend for residual
        seasonal, trend = self._moving_avg(x)

        # --- Step 2: Adaptive wavelet decomposition on seasonal only ---
        seasonal_out, wavelet_regu = self.pastmixing(seasonal)
        # seasonal_out: [B, C, L]

        # Re-add trend so DDI and AMS see the full signal
        x = seasonal_out + trend

        # --- Step 3: Build time_embedding from the COARSEST approximation ---
        # The encoder_levels store the last approx as the coarsest signal.
        # We re-run just the encoder to grab it (cheap — no decoder).
        with torch.no_grad():
            coarse = seasonal
            for enc in self.pastmixing.encoder_levels:
                coarse, _, _ = enc(coarse)
        # coarse: [B, C, seq_len // 2^levels]  — most compressed, trend-dominant
        # Project to seq_len so AMS gating receives a proper [B, C, seq_len] tensor
        time_embedding = self.gate_proj(coarse)   # [B, C, seq_len]

        # --- Step 4: DDI blocks refine the signal ---
        for fc_block in self.fc_blocks:
            x = fc_block(x)

        # --- Step 5: AMS mixture-of-experts forecasting ---
        x, moe_loss = self.moe(x, time_embedding)

        # Fold wavelet regularization into total loss
        if wavelet_regu is not None:
            moe_loss = moe_loss + wavelet_regu

        x = torch.transpose(x, 1, 2)
        # x: [B, pred_len, feature_num]

        if self.norm:
            x = self.rev_norm(x, 'denorm', self.target_slice)

        if self.target_slice:
            x = x[:, :, self.target_slice]

        return x, moe_loss