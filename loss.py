from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def ssim_value(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    channels = x.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, x.device, x.dtype)
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channels)
    sigma_x = F.conv2d(x * x, window, padding=window_size // 2, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(y * y, window, padding=window_size // 2, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return score.mean()


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def spectral_angle_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dot = (prediction * target).sum(dim=1)
    denom = prediction.norm(dim=1) * target.norm(dim=1)
    cosine = (dot / denom.clamp_min(1e-6)).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cosine).mean()


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        gradient_weight: float = 0.1,
        ssim_weight: float = 0.1,
        sam_weight: float = 0.02,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.ssim_weight = ssim_weight
        self.sam_weight = sam_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prediction_01 = ((prediction + 1.0) * 0.5).clamp(0.0, 1.0)
        target_01 = ((target + 1.0) * 0.5).clamp(0.0, 1.0)
        components: Dict[str, torch.Tensor] = {
            "l1": F.l1_loss(prediction, target),
            "gradient": gradient_loss(prediction_01, target_01),
            "ssim": 1.0 - ssim_value(prediction_01, target_01),
            "sam": spectral_angle_loss(prediction_01, target_01),
        }

        total = (
            self.l1_weight * components["l1"]
            + self.gradient_weight * components["gradient"]
            + self.ssim_weight * components["ssim"]
            + self.sam_weight * components["sam"]
        )
        components["total"] = total
        return total, components


# Original name retained for compatibility.
MultiLoss = ReconstructionLoss
