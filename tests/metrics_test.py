from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import (
    calculate_sam,
    image_metrics,
    process_rgb,
    temporal_fusion_statistics,
    tensor_to_reflectance,
)


def main() -> None:
    # Model-domain values -1, -0.6, and 1 correspond to reflectance
    # 0, 2000, and 10000. Metrics intentionally clip the last value to 2000.
    values = torch.tensor([-1.0, -0.6, 1.0]).view(3, 1, 1)
    reflectance = tensor_to_reflectance(values)
    np.testing.assert_allclose(reflectance[0, 0], [0.0, 2000.0, 2000.0])

    image = torch.full((3, 16, 16), -0.8)  # reflectance = 1000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        perfect = image_metrics(
            image, image, dataset_type="new_multi", metric_mode="reflectance_2000"
        )
    assert math.isinf(perfect["psnr"])
    assert abs(perfect["ssim"] - 1.0) < 1e-6
    assert perfect["sam"] < 1e-3
    assert perfect["mae"] == 0.0

    dark = torch.full((3, 16, 16), -1.0)
    bright = torch.full((3, 16, 16), -0.6)  # reflectance = 2000
    contrast = image_metrics(
        dark, bright, dataset_type="new_multi", metric_mode="reflectance_2000"
    )
    assert abs(contrast["psnr"] - 0.0) < 1e-5
    assert abs(contrast["mae"] - 2000.0) < 1e-3

    display = process_rgb(values)
    # Original implementation casts (truncates) rather than rounds.
    np.testing.assert_array_equal(display[0, 0], [0, 254, 255])

    # The compatibility mode must match the original TMFNet sequence:
    # 0..2000 clipping -> independent per-image min-max -> uint8 metrics.
    target = torch.linspace(-1.0, -0.6, 16 * 16).reshape(1, 16, 16).repeat(3, 1, 1)
    prediction = torch.roll(target, shifts=1, dims=-1)
    target_rgb = process_rgb(target).astype(np.float32)
    prediction_rgb = process_rgb(prediction).astype(np.float32)
    original = image_metrics(
        prediction, target, dataset_type="new_multi", metric_mode="original_tmfnet"
    )
    assert abs(original["psnr"] - peak_signal_noise_ratio(
        target_rgb, prediction_rgb, data_range=255.0
    )) < 1e-6
    assert abs(original["ssim"] - structural_similarity(
        target_rgb, prediction_rgb, channel_axis=-1, data_range=255.0
    )) < 1e-6
    assert abs(original["sam"] - calculate_sam(target_rgb, prediction_rgb)) < 1e-6
    assert abs(original["mae"] - torch.mean(torch.abs(prediction - target)).item()) < 1e-8

    weights = torch.full((3, 1, 4, 4), 1.0 / 3.0)
    maps = torch.full_like(weights, 0.7)
    stats = temporal_fusion_statistics(weights, maps, maps)
    assert abs(stats["weight_entropy"] - 1.0) < 1e-6
    assert abs(stats["effective_frames"] - 3.0) < 1e-6
    assert abs(stats["mean_max_weight"] - 1.0 / 3.0) < 1e-6
    print("Original-compatible and reflectance-domain metric tests passed.")


if __name__ == "__main__":
    main()
