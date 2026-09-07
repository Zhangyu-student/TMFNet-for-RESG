from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def tensor_to_numpy01(tensor: torch.Tensor) -> np.ndarray:
    array = ((tensor.detach().float().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
    return array.permute(1, 2, 0).numpy()


def calculate_sam(target: np.ndarray, prediction: np.ndarray) -> float:
    dot = np.sum(target * prediction, axis=-1)
    denom = np.linalg.norm(target, axis=-1) * np.linalg.norm(prediction, axis=-1)
    cosine = np.clip(dot / np.maximum(denom, 1e-8), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).mean())


def image_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = tensor_to_numpy01(prediction)
    gt = tensor_to_numpy01(target)
    return {
        "psnr": float(peak_signal_noise_ratio(gt, pred, data_range=1.0)),
        "ssim": float(structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)),
        "sam": calculate_sam(gt, pred),
        "mae": float(np.mean(np.abs(gt - pred))),
    }


# Compatibility helpers from the original repository.
def mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(prediction - target)).item())


def process_rgb(tensor: torch.Tensor, *_: object) -> np.ndarray:
    return np.round(tensor_to_numpy01(tensor) * 255.0).astype(np.uint8)


def calculate_sam_rgb(target: np.ndarray, prediction: np.ndarray) -> float:
    return calculate_sam(target.astype(np.float32) / 255.0, prediction.astype(np.float32) / 255.0)


def psnr_skimage(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, prediction, data_range=255))


def ssim_skimage(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(structural_similarity(target, prediction, channel_axis=-1, data_range=255))
