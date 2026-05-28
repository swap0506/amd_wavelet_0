import torch
import torch.nn as nn

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

        if self.norm:
            self.rev_norm = RevIN(input_shape[-1])

        # MDM: adaptive wavelet decomposition (first step, replaces avg-pool MDM)
        self.pastmixing = MDM(
            input_shape,
            k=lifting_levels,
            c=c,
            lifting_kernel_size=lifting_kernel_size,
            regu_details=regu_details,
            regu_approx=regu_approx,
            layernorm=layernorm,
        )

        self.fc_blocks = nn.ModuleList([
            DDI(input_shape, dropout=dropout, patch=patch,
                alpha=alpha, layernorm=layernorm)
            for _ in range(n_block)
        ])

        self.moe = AMS(input_shape, pred_len, ff_dim=2048,
                       dropout=dropout, num_experts=8, top_k=2)

    def forward(self, x):
        # x: [B, seq_len, feature_num]

        if self.norm:
            x = self.rev_norm(x, 'norm')

        x = torch.transpose(x, 1, 2)
        # x: [B, feature_num, seq_len]

        # Step 1: Adaptive wavelet decomposition (replaces avg-pool MDM)
        x, wavelet_regu = self.pastmixing(x)
        # x: [B, C, L] — wavelet-reconstructed signal used as time embedding
        time_embedding = x   # pass the wavelet output to AMS

        # Step 2: DDI blocks refine the same signal
        for fc_block in self.fc_blocks:
            x = fc_block(x)

        # Step 3: AMS mixture-of-experts forecasting
        x, moe_loss = self.moe(x, time_embedding)

        # fold wavelet regularization into the total loss
        if wavelet_regu is not None:
            moe_loss = moe_loss + wavelet_regu

        x = torch.transpose(x, 1, 2)
        # x: [B, pred_len, feature_num]

        if self.norm:
            x = self.rev_norm(x, 'denorm', self.target_slice)

        if self.target_slice:
            x = x[:, :, self.target_slice]

        return x, moe_loss