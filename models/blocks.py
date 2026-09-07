from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 32) -> int:
    """Return the largest valid GroupNorm group count up to max_groups."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.GELU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualDetailBlock(nn.Module):
    """Lightweight local detail block used by the shared encoder and decoder."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.pw1 = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.pw2 = nn.Conv2d(hidden, channels, 1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden, hidden, 1), nn.Sigmoid())
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = F.gelu(self.pw1(y))
        y = self.dw(y)
        y = y * self.gate(y)
        y = self.pw2(F.gelu(y))
        return x + self.beta * y


class SharedFrameEncoder(nn.Module):
    """Shared CNN encoder for each temporal observation.

    Returns features at H/2, H/4 and H/8.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 48) -> None:
        super().__init__()
        b = base_channels
        self.stem = nn.Sequential(
            ConvGNAct(in_channels, b, 3, 1),
            ResidualDetailBlock(b),
        )
        self.level1 = nn.Sequential(
            ConvGNAct(b, b, 3, 2),
            ResidualDetailBlock(b),
        )
        self.level2 = nn.Sequential(
            ConvGNAct(b, b * 2, 3, 2),
            ResidualDetailBlock(b * 2),
        )
        self.level3 = nn.Sequential(
            ConvGNAct(b * 2, b * 4, 3, 2),
            ResidualDetailBlock(b * 4),
            ResidualDetailBlock(b * 4),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        f1 = self.level1(x)
        f2 = self.level2(f1)
        f3 = self.level3(f2)
        return f1, f2, f3


class FeatureQualityEstimator(nn.Module):
    """Estimate reliability, where 1 means reliable and 0 means degraded.

    The estimator compares each observation with a temporal reference feature.
    """

    def __init__(self, channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(channels * 2, hidden_channels, 3, 1),
            ResidualDetailBlock(hidden_channels),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # features: [B,T,C,H,W], valid_mask: [B,T]
        b, t, c, h, w = features.shape
        mask = valid_mask[:, :, None, None, None].to(features.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        reference = (features * mask).sum(dim=1, keepdim=True) / denom
        difference = torch.abs(features - reference)
        quality = self.net(torch.cat([features, difference], dim=2).reshape(b * t, c * 2, h, w))
        quality = torch.sigmoid(quality).reshape(b, t, 1, h, w)
        return quality * mask


class HierarchicalWeightRefiner(nn.Module):
    """Refine coarse temporal logits with local features and reliability."""

    def __init__(self, channels: int, hidden_channels: int = 24) -> None:
        super().__init__()
        self.local_score = nn.Sequential(
            ConvGNAct(channels, hidden_channels, 3, 1),
            ResidualDetailBlock(hidden_channels),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        quality: torch.Tensor,
        coarse_weights: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # features [B,T,C,H,W], quality [B,T,1,H,W]
        b, t, c, h, w = features.shape
        local = self.local_score(features.reshape(b * t, c, h, w)).reshape(b, t, 1, h, w)
        coarse = F.interpolate(
            coarse_weights.reshape(b * t, 1, coarse_weights.shape[-2], coarse_weights.shape[-1]),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, t, 1, h, w)
        logits = local + torch.log(quality.clamp_min(1e-5)) + torch.log(coarse.clamp_min(1e-5))
        mask = valid_mask[:, :, None, None, None]
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        fused = (features * weights).sum(dim=1)
        return fused, weights


class DirectReconstructionDecoder(nn.Module):
    def __init__(self, base_channels: int, out_channels: int) -> None:
        super().__init__()
        b = base_channels
        self.merge3 = nn.Sequential(
            ConvGNAct(b * 4 + b * 2, b * 4),
            ResidualDetailBlock(b * 4),
        )
        self.up2 = nn.Sequential(
            ConvGNAct(b * 4, b * 2),
            ResidualDetailBlock(b * 2),
        )
        self.merge2 = nn.Sequential(
            ConvGNAct(b * 2 + b, b * 2),
            ResidualDetailBlock(b * 2),
        )
        self.up1 = nn.Sequential(
            ConvGNAct(b * 2, b),
            ResidualDetailBlock(b),
        )
        self.out_head = nn.Sequential(
            ConvGNAct(b, b),
            ResidualDetailBlock(b),
            nn.Conv2d(b, out_channels, 3, padding=1),
        )
    def forward(
        self,
        deep: torch.Tensor,
        skip2: torch.Tensor,
        skip1: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        x = F.interpolate(deep, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.merge3(torch.cat([x, skip2], dim=1))
        x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = self.merge2(torch.cat([x, skip1], dim=1))
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        x = self.up1(x)
        return torch.tanh(self.out_head(x))
