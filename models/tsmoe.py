import torch
import torch.nn as nn


class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, noise_epsilon=1e-5):
        super(TopKGating, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        self.num_experts = num_experts
        self.w_noise = nn.Parameter(torch.zeros(num_experts, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

    def decompostion_tp(self, x, alpha=10):
        # x: [batch_size, num_experts]
        output = torch.zeros_like(x)
        kth_largest_val, _ = torch.kthvalue(x, self.num_experts - self.top_k + 1)
        kth_largest_mat = kth_largest_val.unsqueeze(1).expand(-1, self.num_experts)
        mask = x < kth_largest_mat
        x = self.softmax(x)
        output[mask]  = alpha * torch.log(x[mask] + 1)
        output[~mask] = alpha * (torch.exp(x[~mask]) - 1)
        return output

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.gate(x)
        clean_logits = x

        if self.training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits

        logits = self.decompostion_tp(logits)
        gates  = self.softmax(logits)
        return gates


class Expert(nn.Module):
    """3-layer expert with residual shortcut for better gradient flow."""
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.2):
        super(Expert, self).__init__()
        # Main path
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),   # extra capacity layer
            
        )
        # Residual shortcut: project input directly to output_dim
        # FIX: gives gradient a direct path, prevents vanishing
        self.shortcut = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class AMS(nn.Module):
    def __init__(self, input_shape, pred_len, ff_dim=1024, dropout=0.2,
                 loss_coef=1.0, num_experts=4, top_k=2):
        super(AMS, self).__init__()
        # input_shape[0] = seq_len, input_shape[1] = feature_num
        self.num_experts = num_experts
        self.top_k       = top_k
        self.pred_len    = pred_len
        self.loss_coef   = loss_coef
        assert self.top_k <= self.num_experts

        self.gating = TopKGating(input_shape[0], num_experts, top_k)

        self.experts = nn.ModuleList([
            Expert(input_shape[0], pred_len, hidden_dim=ff_dim, dropout=dropout)
            for _ in range(num_experts)
        ])

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, time_embedding):
        # x, time_embedding: [batch_size, feature_num, seq_len]
        batch_size  = x.shape[0]
        feature_num = x.shape[1]

        # [feature_num, batch_size, seq_len]
        x              = torch.transpose(x, 0, 1)
        time_embedding = torch.transpose(time_embedding, 0, 1)

        output = torch.zeros(feature_num, batch_size, self.pred_len, device=x.device)

        # Collect importance across ALL features before computing loss
        # FIX: average importance over features → stable load-balancing loss
        all_importance = torch.zeros(self.num_experts, self.pred_len, device=x.device)

        for i in range(feature_num):
            inp       = x[i]            # [batch_size, seq_len]
            time_info = time_embedding[i]

            # gates: [batch_size, num_experts]
            gates = self.gating(time_info)

            # expert_outputs: [num_experts, batch_size, pred_len]
            expert_outputs = torch.stack(
                [self.experts[j](inp) for j in range(self.num_experts)], dim=0
            )
            # → [batch_size, num_experts, pred_len]
            expert_outputs = expert_outputs.permute(1, 0, 2)

            # gates expanded: [batch_size, num_experts, pred_len]
            gates_exp = gates.unsqueeze(-1).expand_as(expert_outputs)

            # weighted sum → [batch_size, pred_len]
            batch_output = (gates_exp * expert_outputs).sum(dim=1)
            output[i] = batch_output

            # accumulate importance (sum over batch, keep per-expert per-pred_len)
            all_importance += gates_exp.sum(0)   # [num_experts, pred_len]

        # Average over features → single scalar loss
        # FIX: divide by feature_num to decouple loss scale from dataset width
        avg_importance = all_importance / feature_num
        load_loss = self.loss_coef * self.cv_squared(avg_importance.mean(-1))

        # [batch_size, feature_num, pred_len]
        output = torch.transpose(output, 0, 1)
        return output, load_loss