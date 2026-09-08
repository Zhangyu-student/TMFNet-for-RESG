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
    """Estimate reliability with a non-invertible consistency prior.

    A robust temporal median defines the reference. Larger feature disagreement
    always lowers the prior. The learned branch may suppress suspicious regions
    further, but cannot turn a strong temporal outlier into high reliability.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 32,
        temperature: float = 1.5,
        learned_strength: float = 0.25,
        quality_floor: float = 0.02,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0.0 <= learned_strength <= 1.0:
            raise ValueError("learned_strength must be in [0,1].")
        if not 0.0 <= quality_floor < 1.0:
            raise ValueError("quality_floor must be in [0,1).")
        self.temperature = float(temperature)
        self.learned_strength = float(learned_strength)
        self.quality_floor = float(quality_floor)
        self.net = nn.Sequential(
            ConvGNAct(channels * 2, hidden_channels, 3, 1),
            ResidualDetailBlock(hidden_channels),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    @staticmethod
    def _masked_temporal_median(
        features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the per-channel median of valid frames as [B,1,C,H,W]."""
        b, _, c, h, w = features.shape
        mask = valid_mask[:, :, None, None, None]
        sorted_features = features.masked_fill(~mask, torch.inf).sort(dim=1).values
        counts = valid_mask.sum(dim=1)
        lower = ((counts - 1) // 2).view(b, 1, 1, 1, 1).expand(b, 1, c, h, w)
        upper = (counts // 2).view(b, 1, 1, 1, 1).expand(b, 1, c, h, w)
        lower_value = sorted_features.gather(1, lower)
        upper_value = sorted_features.gather(1, upper)
        return 0.5 * (lower_value + upper_value)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # features: [B,T,C,H,W], valid_mask: [B,T]
        b, t, c, h, w = features.shape
        mask = valid_mask[:, :, None, None, None].to(features.dtype)
        # The reference is a statistic, not a trainable route. Detaching it
        # avoids retaining the temporal sort graph and keeps memory overhead low.
        reference = self._masked_temporal_median(features.detach(), valid_mask)
        difference = torch.abs(features - reference) * mask

        # Normalize disagreement per sample so the meaning is stable across
        # feature scales. Detaching the prior prevents the encoder from
        # collapsing its features merely to maximize quality.
        distance = difference.mean(dim=2, keepdim=True)
        valid_pixels = valid_mask.sum(dim=1).to(features.dtype) * float(h * w)
        distance_scale = distance.sum(dim=(1, 3, 4), keepdim=True)
        distance_scale = distance_scale / valid_pixels.view(b, 1, 1, 1, 1).clamp_min(1.0)
        relative_distance = (distance / distance_scale.clamp_min(1e-4)).detach()
        consistency_prior = torch.exp(-self.temperature * relative_distance)
        consistency_prior = consistency_prior.clamp(min=self.quality_floor, max=1.0) * mask

        learned = self.net(
            torch.cat([features, difference], dim=2).reshape(b * t, c * 2, h, w)
        )
        learned = torch.sigmoid(learned).reshape(b, t, 1, h, w)
        modulation = 1.0 - self.learned_strength + self.learned_strength * learned
        return consistency_prior * modulation * mask


class HierarchicalWeightRefiner(nn.Module):
    """Fuse a scale directly from the shared quality map.

    There is no additional local scoring branch. The shared quality is
    normalized only for feature fusion and optionally blended with the
    upsampled coarse-scale weights.
    """

    def __init__(
        self,
        coarse_blend: float = 0.25,
    ) -> None:
        super().__init__()
        if not 0.0 <= coarse_blend <= 1.0:
            raise ValueError("coarse_blend must be in [0,1].")
        self.coarse_blend = coarse_blend

    def forward(
        self,
        features: torch.Tensor,
        quality: torch.Tensor,
        coarse_weights: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # features [B,T,C,H,W], quality [B,T,1,H,W]
        b, t, c, h, w = features.shape
        mask = valid_mask[:, :, None, None, None].to(features.dtype)
        quality = quality * mask
        quality_sum = quality.sum(dim=1, keepdim=True)
        uniform = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        local_weights = torch.where(
            quality_sum > 1e-6,
            quality / quality_sum.clamp_min(1e-6),
            uniform,
        )

        coarse = F.interpolate(
            coarse_weights.reshape(b * t, 1, coarse_weights.shape[-2], coarse_weights.shape[-1]),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, t, 1, h, w)
        coarse = coarse * mask
        coarse = coarse / coarse.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = (1.0 - self.coarse_blend) * local_weights + self.coarse_blend * coarse
        weights = weights * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
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
