import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def normalization(channels: int):
    return nn.InstanceNorm1d(num_features=channels)


from layers.LiftingScheme import LiftingScheme, InverseLiftingScheme


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, target_slice=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias   = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean  = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / (self.affine_weight + self.eps * self.eps)[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x


class AdaWaveletBlock(nn.Module):
    def __init__(self, configs, input_size):
        super(AdaWaveletBlock, self).__init__()
        self.regu_details = configs.regu_details
        self.regu_approx  = configs.regu_approx
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingScheme(configs.enc_in, k_size=configs.lifting_kernel_size,
                                     input_size=input_size)
        self.norm_x = normalization(configs.enc_in)
        self.norm_d = normalization(configs.enc_in)

    def forward(self, x):
        (c, d) = self.wavelet(x)
        x = c

        r = None
        if self.regu_approx + self.regu_details != 0.0:
            if self.regu_details:
                rd = self.regu_details * d.abs().mean()
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(c.mean(), x.mean(), p=2)
            if self.regu_approx == 0.0:
                r = rd
            elif self.regu_details == 0.0:
                r = rc
            else:
                r = rd + rc

        x = self.norm_x(x)
        d = self.norm_d(d)
        return x, r, d


class InverseAdaWaveletBlock(nn.Module):
    def __init__(self, configs, input_size):
        super(InverseAdaWaveletBlock, self).__init__()
        self.inverse_wavelet = InverseLiftingScheme(
            configs.enc_in, input_size=input_size,
            kernel_size=configs.lifting_kernel_size
        )

    def forward(self, c, d):
        return self.inverse_wavelet(c, d)


class ChannelWiseAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention applied at the
    coarsest approximation level.  Learns which channels (variables)
    matter most before passing to the AMS gating network.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, L]
        # Global average pool over time → [B, C]
        scale = x.mean(dim=-1)
        # Attention weights → [B, C]
        scale = self.fc(scale)
        # Recalibrate → [B, C, L]
        return x * scale.unsqueeze(-1)


class MDM(nn.Module):
    """
    Multi-level wavelet MDM with Channel-Wise Attention (CWA) at the
    coarsest encoder level.
    """
    def __init__(self, input_shape, lifting_levels=3, lifting_kernel_size=7,
                 regu_details=0.0, regu_approx=0.0, **kwargs):
        super().__init__()
        seq_len = input_shape[0]
        enc_in  = input_shape[1]
        self.levels = lifting_levels

        class _Cfg:
            pass
        cfg = _Cfg()
        cfg.enc_in              = enc_in
        cfg.lifting_kernel_size = lifting_kernel_size
        cfg.regu_details        = regu_details
        cfg.regu_approx         = regu_approx

        self.encoder_levels = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()

        size = seq_len
        for level in range(lifting_levels):
            self.encoder_levels.append(AdaWaveletBlock(cfg, size))
            size = size // 2

        # CWA applied after the coarsest approximation (bottom encoder level)
        # FIX: helps model learn channel dominance at the most compressed scale
        self.cwa = ChannelWiseAttention(enc_in)

        for _ in range(lifting_levels - 1, -1, -1):
            self.decoder_levels.append(InverseAdaWaveletBlock(cfg, size))
            size = size * 2

    def forward(self, x):
        # x: [B, C, L]
        coeffs     = []
        regu_total = None
        approx     = x

        for idx, enc in enumerate(self.encoder_levels):
            approx, r, d = enc(approx)
            coeffs.append(d)
            if r is not None:
                regu_total = r if regu_total is None else regu_total + r

        # Apply CWA at the coarsest approximation (after all encoder levels)
        approx = self.cwa(approx)

        for dec, d in zip(self.decoder_levels, reversed(coeffs)):
            approx = dec(approx, d)

        return approx, regu_total   # (reconstructed_x, regularisation_loss)


class DDI(nn.Module):
    def __init__(self, input_shape, dropout=0.2, patch=12, alpha=0.0, layernorm=True):
        super(DDI, self).__init__()
        self.input_shape = input_shape
        if alpha > 0.0:
            self.ff_dim = 2 ** math.ceil(math.log2(self.input_shape[-1]))
            self.fc_block = nn.Sequential(
                nn.Linear(self.input_shape[-1], self.ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim, self.input_shape[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.n_history = 1
        self.alpha  = alpha
        self.patch  = patch

        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(self.input_shape[0] * self.input_shape[-1])
        self.norm1 = nn.BatchNorm1d(self.n_history * patch * self.input_shape[-1])
        if self.alpha > 0.0:
            self.norm2 = nn.BatchNorm1d(self.patch * self.input_shape[-1])

        self.agg       = nn.Linear(self.n_history * self.patch, self.patch)
        self.dropout_t = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, feature_num, seq_len]
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)

        output = torch.zeros_like(x)
        output[:, :, :self.n_history * self.patch] = x[:, :, :self.n_history * self.patch].clone()

        for i in range(self.n_history * self.patch, self.input_shape[0], self.patch):
            inp  = output[:, :, i - self.n_history * self.patch: i]
            inp  = self.norm1(torch.flatten(inp, 1, -1)).reshape(inp.shape)
            inp  = F.gelu(self.agg(inp))
            inp  = self.dropout_t(inp)

            tmp = inp + x[:, :, i: i + self.patch]
            res = tmp

            if self.alpha > 0.0:
                tmp = self.norm2(torch.flatten(tmp, 1, -1)).reshape(tmp.shape)
                tmp = torch.transpose(tmp, 1, 2)
                tmp = self.fc_block(tmp)
                tmp = torch.transpose(tmp, 1, 2)
            output[:, :, i: i + self.patch] = res + self.alpha * tmp

        return output