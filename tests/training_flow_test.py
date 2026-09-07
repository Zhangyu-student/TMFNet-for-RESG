from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import train


def main() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        tile = "tile_001"
        cloud_dir = root / "Sen2_MTC" / tile / "cloud"
        clear_dir = root / "Sen2_MTC" / tile / "cloudless"
        cloud_dir.mkdir(parents=True)
        clear_dir.mkdir(parents=True)
        for split in ("train", "val"):
            (root / f"{split}.txt").write_text(f"{tile}\n", encoding="utf-8")
        for frame in range(2):
            image = np.full((32, 32, 3), 2500 + frame * 1000, dtype=np.uint16)
            tifffile.imwrite(cloud_dir / f"scene_{frame}.tif", image)
        tifffile.imwrite(
            clear_dir / "scene.tif", np.full((32, 32, 3), 5000, dtype=np.uint16)
        )

        config = {
            "dataset_type": "new_multi",
            "data_root": str(root),
            "input_channels": 3,
            "output_channels": 3,
            "min_temporal": 2,
            "max_temporal": 2,
            "random_temporal_subset": False,
            "random_reverse": False,
            "base_channels": 8,
            "temporal_depth": 1,
            "state_dim": 4,
            "temporal_expansion": 1,
            "batch_size": 1,
            "num_workers": 0,
            "num_epochs": 1,
            "device": "cpu",
            "amp": False,
            "tensorboard": False,
            "order_consistency_weight": 0.0,
            "subset_consistency_weight": 0.0,
            "save_dir": str(root / "checkpoints"),
            "log_file": str(root / "logs" / "training.csv"),
            "visualization_dir": str(root / "visualizations"),
            "visualization_interval": 1,
            "loss": {"sam_weight": 0.0},
        }
        train(config)
        latest = root / "checkpoints" / "latest.pth"
        best = root / "checkpoints" / "best_psnr.pth"
        assert latest.is_file() and best.is_file()
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        assert {"model", "optimizer", "scheduler", "epoch", "best_metric", "config"} <= payload.keys()
        assert payload["epoch"] == 1
        assert list((root / "visualizations").glob("epoch_*/comparison.png"))
        assert (root / "logs" / "training.csv").is_file()
    print("End-to-end training flow test passed.")


if __name__ == "__main__":
    main()
