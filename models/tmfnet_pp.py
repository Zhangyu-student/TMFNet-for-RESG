from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    DirectReconstructionDecoder,
    FeatureQualityEstimator,
    HierarchicalWeightRefiner,
    SharedFrameEncoder,
)
from .temporal_ssm import BidirectionalQualityTemporalSSM


class TMFNetPlusPlus(nn.Module):
    """TMFNet++ for variable-length blind multi-temporal cloud removal.

    Main components:
      1. Shared multi-scale spatial encoder.
      2. Feature-consistency reliability estimation.
      3. Bidirectional quality-conditioned selective SSM at the bottleneck.
      4. Hierarchical state-guided temporal weight refinement.
      5. Direct clear-image reconstruction decoder.
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        base_channels: int = 48,
        temporal_depth: int = 2,
        state_dim: int = 8,
        temporal_expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_channels != output_channels:
            raise ValueError("The current dataset pipeline requires input_channels == output_channels.")
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels

        self.encoder = SharedFrameEncoder(input_channels, base_channels)
        self.quality1 = FeatureQualityEstimator(base_channels, hidden_channels=24)
        self.quality2 = FeatureQualityEstimator(base_channels * 2, hidden_channels=32)
        self.quality3 = FeatureQualityEstimator(base_channels * 4, hidden_channels=48)

        self.temporal_fusion = BidirectionalQualityTemporalSSM(
            channels=base_channels * 4,
            depth=temporal_depth,
            state_dim=state_dim,
            expansion=temporal_expansion,
            dropout=dropout,
        )
        self.refine2 = HierarchicalWeightRefiner(base_channels * 2, hidden_channels=32)
        self.refine1 = HierarchicalWeightRefiner(base_channels, hidden_channels=24)
        self.decoder = DirectReconstructionDecoder(base_channels, output_channels)

    @staticmethod
    def _default_mask(x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        return torch.ones(b, t, dtype=torch.bool, device=x.device)

    def forward(
        self,
        observations: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        if observations.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(observations.shape)}")
        b, t, c, h, w = observations.shape
        if c != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {c}")
        if valid_mask is None:
            valid_mask = self._default_mask(observations)
        valid_mask = valid_mask.to(device=observations.device, dtype=torch.bool)
        if valid_mask.shape != (b, t):
            raise ValueError("valid_mask must have shape [B,T].")
        if torch.any(valid_mask.sum(dim=1) == 0):
            raise ValueError("Each sample must contain at least one valid observation.")

        flat = observations.reshape(b * t, c, h, w)
        f1, f2, f3 = self.encoder(flat)
        f1 = f1.reshape(b, t, *f1.shape[1:])
        f2 = f2.reshape(b, t, *f2.shape[1:])
        f3 = f3.reshape(b, t, *f3.shape[1:])

        q1 = self.quality1(f1, valid_mask)
        q2 = self.quality2(f2, valid_mask)
        q3 = self.quality3(f3, valid_mask)

        deep, w3, states = self.temporal_fusion(f3, q3, valid_mask)
        skip2, w2 = self.refine2(f2, q2, w3, valid_mask)
        skip1, w1 = self.refine1(f1, q1, w2, valid_mask)

        output = self.decoder(deep, skip2, skip1, output_size=(h, w))

        if not return_aux:
            return output

        full_weights = F.interpolate(
            w1.reshape(b * t, 1, w1.shape[-2], w1.shape[-1]),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, t, 1, h, w)
        full_weights = full_weights * valid_mask[:, :, None, None, None].to(full_weights.dtype)
        full_weights = full_weights / full_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        aux: Dict[str, torch.Tensor] = {
            "quality_s1": q1,
            "quality_s2": q2,
            "quality_s3": q3,
            "weights_s1": w1,
            "weights_s2": w2,
            "weights_s3": w3,
            "weights_full": full_weights,
            "temporal_states": states,
        }
        return output, aux


# Backward-friendly alias for scripts that expect the original class name.
TemporalMambaFusionNet = TMFNetPlusPlus
