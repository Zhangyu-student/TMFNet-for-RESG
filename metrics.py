from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DEFAULT_REFLECTANCE_SCALE = 10000.0
DEFAULT_REFLECTANCE_MAX = 2000.0


def tensor_to_numpy01(tensor: torch.Tensor) -> np.ndarray:
    array = ((tensor.detach().float().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
    return array.permute(1, 2, 0).numpy()


def tensor_to_reflectance(
    tensor: torch.Tensor,
    reflectance_scale: float = DEFAULT_REFLECTANCE_SCALE,
    reflectance_max: float = DEFAULT_REFLECTANCE_MAX,
) -> np.ndarray:
    """Convert a model-domain [-1,1] tensor to clipped reflectance [0,max]."""
    if reflectance_scale <= 0 or reflectance_max <= 0:
        raise ValueError("Reflectance scale and maximum must be positive.")
    reflectance = ((tensor.detach().float().cpu() + 1.0) * 0.5) * reflectance_scale
    reflectance = reflectance.clamp(0.0, reflectance_max)
    return reflectance.permute(1, 2, 0).numpy()


def tensor_to_original_tmfnet_rgb(
    tensor: torch.Tensor,
    reflectance_scale: float = DEFAULT_REFLECTANCE_SCALE,
    reflectance_max: float = DEFAULT_REFLECTANCE_MAX,
) -> np.ndarray:
    """Reproduce the original TMFNet new_multi RGB conversion exactly."""
    reflectance = tensor_to_reflectance(tensor, reflectance_scale, reflectance_max)
    if reflectance.shape[-1] < 3:
        if reflectance.shape[-1] == 1:
            reflectance = np.repeat(reflectance, 3, axis=-1)
        else:
            reflectance = np.concatenate([reflectance, reflectance[..., -1:]], axis=-1)
    rgb = reflectance[..., :3]
    rgb = rgb - np.min(rgb)
    rgb_max = float(np.max(rgb))
    rgb = np.full_like(rgb, 255.0) if rgb_max == 0 else 255.0 * rgb / rgb_max
    rgb = np.nan_to_num(rgb, nan=float(np.nanmean(rgb)))
    return rgb.astype(np.uint8)


def calculate_sam(target: np.ndarray, prediction: np.ndarray) -> float:
    dot = np.sum(target * prediction, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    prediction_norm = np.linalg.norm(prediction, axis=-1)
    valid = (target_norm > 1e-8) & (prediction_norm > 1e-8)
    if not np.any(valid):
        return 0.0
    cosine = np.clip(dot[valid] / (target_norm[valid] * prediction_norm[valid]), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).mean())


def image_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dataset_type: str = "new_multi",
    reflectance_scale: float = DEFAULT_REFLECTANCE_SCALE,
    reflectance_max: float = DEFAULT_REFLECTANCE_MAX,
    metric_mode: str = "original_tmfnet",
) -> Dict[str, float]:
    if dataset_type == "new_multi":
        if metric_mode == "original_tmfnet":
            pred = tensor_to_original_tmfnet_rgb(
                prediction, reflectance_scale, reflectance_max
            ).astype(np.float32)
            gt = tensor_to_original_tmfnet_rgb(
                target, reflectance_scale, reflectance_max
            ).astype(np.float32)
            data_range = 255.0
            mae_value = float(torch.mean(torch.abs(prediction - target)).item())
        elif metric_mode == "reflectance_2000":
            pred = tensor_to_reflectance(prediction, reflectance_scale, reflectance_max)
            gt = tensor_to_reflectance(target, reflectance_scale, reflectance_max)
            data_range = reflectance_max
            mae_value = float(np.mean(np.abs(gt - pred)))
        else:
            raise ValueError(
                "metric_mode must be 'original_tmfnet' or 'reflectance_2000'."
            )
    else:
        pred = tensor_to_numpy01(prediction)
        gt = tensor_to_numpy01(target)
        data_range = 1.0
        mae_value = float(np.mean(np.abs(gt - pred)))
    return {
        "psnr": float(peak_signal_noise_ratio(gt, pred, data_range=data_range)),
        "ssim": float(structural_similarity(gt, pred, channel_axis=-1, data_range=data_range)),
        "sam": calculate_sam(gt, pred),
        "mae": mae_value,
    }


def temporal_fusion_statistics(
    weights: torch.Tensor,
    quality: torch.Tensor,
) -> Dict[str, float]:
    """Summarize valid temporal maps shaped [T,1,H,W].

    Effective frames is 1 for single-frame selection and approaches T for
    uniform participation. Normalized entropy follows the same 0..1 scale.
    """
    if weights.ndim != 4 or quality.shape != weights.shape:
        raise ValueError("weights and quality must share shape [T,1,H,W].")
    temporal_length = int(weights.shape[0])
    if temporal_length < 1:
        raise ValueError("At least one valid temporal map is required.")
    w = weights.detach().float().clamp_min(0.0)
    w = w / w.sum(dim=0, keepdim=True).clamp_min(1e-8)
    entropy = -(w * w.clamp_min(1e-8).log()).sum(dim=0)
    if temporal_length > 1:
        entropy = entropy / math.log(temporal_length)
    else:
        entropy = torch.ones_like(entropy)
    effective_frames = 1.0 / w.square().sum(dim=0).clamp_min(1e-8)
    return {
        "mean_quality": float(quality.detach().float().mean().item()),
        "mean_max_weight": float(w.max(dim=0).values.mean().item()),
        "weight_entropy": float(entropy.mean().item()),
        "effective_frames": float(effective_frames.mean().item()),
    }


# Compatibility helpers from the original repository.
def mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(prediction - target)).item())


def process_rgb(
    tensor: torch.Tensor,
    dataset_name: str = "new_multi",
    reflectance_scale: float = DEFAULT_REFLECTANCE_SCALE,
    reflectance_max: float = DEFAULT_REFLECTANCE_MAX,
) -> np.ndarray:
    if dataset_name == "new_multi":
        return tensor_to_original_tmfnet_rgb(tensor, reflectance_scale, reflectance_max)
    else:
        image = tensor_to_numpy01(tensor)
    return np.round(np.clip(image[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)


def calculate_sam_rgb(target: np.ndarray, prediction: np.ndarray) -> float:
    return calculate_sam(target.astype(np.float32), prediction.astype(np.float32))


def psnr_skimage(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, prediction, data_range=255))


def ssim_skimage(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(structural_similarity(target, prediction, channel_axis=-1, data_range=255))
