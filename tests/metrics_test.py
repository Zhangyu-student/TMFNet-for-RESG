from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import (
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
        perfect = image_metrics(image, image, dataset_type="new_multi")
    assert math.isinf(perfect["psnr"])
    assert abs(perfect["ssim"] - 1.0) < 1e-6
    assert perfect["sam"] < 1e-3
    assert perfect["mae"] == 0.0

    dark = torch.full((3, 16, 16), -1.0)
    bright = torch.full((3, 16, 16), -0.6)  # reflectance = 2000
    contrast = image_metrics(dark, bright, dataset_type="new_multi")
    assert abs(contrast["psnr"] - 0.0) < 1e-5
    assert abs(contrast["mae"] - 2000.0) < 1e-3

    display = process_rgb(values)
    np.testing.assert_array_equal(display[0, 0], [0, 255, 255])

    weights = torch.full((3, 1, 4, 4), 1.0 / 3.0)
    maps = torch.full_like(weights, 0.7)
    stats = temporal_fusion_statistics(weights, maps, maps)
    assert abs(stats["weight_entropy"] - 1.0) < 1e-6
    assert abs(stats["effective_frames"] - 3.0) < 1e-6
    assert abs(stats["mean_max_weight"] - 1.0 / 3.0) < 1e-6
    print("Reflectance-domain metric test passed.")


if __name__ == "__main__":
    main()
