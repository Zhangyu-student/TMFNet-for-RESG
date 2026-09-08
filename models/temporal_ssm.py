from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveDiagonalSSMBlock(nn.Module):
    """A compact input-dependent diagonal state-space block.

    This implementation is intentionally pure PyTorch. It supports variable-length
    sequences and reliability-conditioned state updates without requiring the
    compiled mamba_ssm package.
    """

    def __init__(
        self,
        dim: int,
        state_dim: int = 8,
        expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        hidden = dim * expansion

        self.norm = nn.LayerNorm(dim)
        self.input_proj = nn.Linear(dim, dim * 2)
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.delta_proj = nn.Linear(dim, dim)
        self.b_proj = nn.Linear(dim, state_dim)
        self.c_proj = nn.Linear(dim, state_dim)

        # A is constrained to be negative for stable state transitions.
        init = torch.arange(1, state_dim + 1, dtype=torch.float32).log()
        self.log_a = nn.Parameter(init.repeat(dim, 1))
        self.d = nn.Parameter(torch.ones(dim))

        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        quality: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x [N,T,C], quality [N,T,1], valid_mask [N,T]
        valid = valid_mask.unsqueeze(-1)
        residual = x * valid.to(x.dtype)
        z = self.norm(residual)
        u, gate = self.input_proj(z).chunk(2, dim=-1)
        u = u * valid.to(u.dtype)
        u = u + self.local_conv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u) * valid.to(u.dtype)
        gate = torch.sigmoid(gate)

        n, t, c = u.shape
        state = torch.zeros(n, c, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        a = -torch.exp(self.log_a).to(dtype=x.dtype)  # [C,D]

        for index in range(t):
            u_t = u[:, index]
            q_t = quality[:, index].clamp(0.0, 1.0)
            valid_t = valid_mask[:, index:index + 1]

            delta = F.softplus(self.delta_proj(u_t))
            delta = delta * (0.25 + 0.75 * q_t)
            transition = torch.exp(delta.unsqueeze(-1) * a.unsqueeze(0))

            b_t = torch.tanh(self.b_proj(u_t)).unsqueeze(1)
            c_t = torch.tanh(self.c_proj(u_t)).unsqueeze(1)
            candidate = transition * state + (1.0 - transition) * b_t * u_t.unsqueeze(-1)
            candidate = q_t.unsqueeze(-1) * candidate + (1.0 - q_t.unsqueeze(-1)) * state
            state = torch.where(valid_t.unsqueeze(-1), candidate, state)

            y_t = (state * c_t).sum(dim=-1) + self.d * u_t
            y_t = y_t * gate[:, index]
            y_t = torch.where(valid_t, y_t, torch.zeros_like(y_t))
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = self.out_proj(self.out_norm(y))
        result = residual + self.drop(y)
        return result * valid.to(result.dtype)


class SelectiveSSMStack(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        state_dim: int = 8,
        expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SelectiveDiagonalSSMBlock(
                    dim=dim,
                    state_dim=state_dim,
                    expansion=expansion,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        quality: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, quality, valid_mask)
        return x


class BidirectionalQualityTemporalSSM(nn.Module):
    """Bidirectional quality-conditioned temporal fusion.

    The single shared quality map controls state updates and is normalized
    directly for final temporal aggregation. There is no extra selection gate.
    """

    def __init__(
        self,
        channels: int,
        depth: int = 2,
        state_dim: int = 8,
        expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.forward_ssm = SelectiveSSMStack(channels, depth, state_dim, expansion, dropout)
        self.backward_ssm = SelectiveSSMStack(channels, depth, state_dim, expansion, dropout)
        self.merge = nn.Sequential(
            nn.LayerNorm(channels * 3),
            nn.Linear(channels * 3, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    @staticmethod
    def _reverse_valid(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Reverse valid tokens in place while keeping invalid positions zero."""
        n, t = valid_mask.shape
        positions = torch.arange(t, device=x.device).unsqueeze(0).expand(n, -1)
        invalid = torch.full_like(positions, t)
        sorted_positions = torch.where(valid_mask, positions, invalid).sort(dim=1).values
        ranks = valid_mask.long().cumsum(dim=1) - 1
        reverse_ranks = valid_mask.sum(dim=1, keepdim=True) - 1 - ranks
        gather_positions = sorted_positions.gather(1, reverse_ranks.clamp_min(0)).clamp_max(t - 1)
        gather_shape = [n, t] + [1] * (x.ndim - 2)
        gather_index = gather_positions.view(*gather_shape).expand_as(x)
        reversed_x = x.gather(1, gather_index)
        mask_shape = [n, t] + [1] * (x.ndim - 2)
        return torch.where(valid_mask.view(*mask_shape), reversed_x, torch.zeros_like(reversed_x))

    def forward(
        self,
        features: torch.Tensor,
        quality: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # features [B,T,C,H,W], quality [B,T,1,H,W]
        b, t, c, h, w = features.shape
        x = features.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        q = quality.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, 1)
        mask = valid_mask[:, None, None, :].expand(b, h, w, t).reshape(b * h * w, t)

        forward_state = self.forward_ssm(x, q, mask)

        x_rev = self._reverse_valid(x, mask)
        q_rev = self._reverse_valid(q, mask)
        backward_rev = self.backward_ssm(x_rev, q_rev, mask)
        backward_state = self._reverse_valid(backward_rev, mask)

        states = self.merge(torch.cat([x, forward_state, backward_state], dim=-1))
        valid = mask.unsqueeze(-1).to(states.dtype)
        quality = q * valid
        quality_sum = quality.sum(dim=1, keepdim=True)
        uniform = valid / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = torch.where(
            quality_sum > 1e-6,
            quality / quality_sum.clamp_min(1e-6),
            uniform,
        )
        fused = (states * weights).sum(dim=1)

        fused = fused.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        weights_map = weights.reshape(b, h, w, t, 1).permute(0, 3, 4, 1, 2).contiguous()
        states_map = states.reshape(b, h, w, t, c).permute(0, 3, 4, 1, 2).contiguous()
        return fused, weights_map, states_map
