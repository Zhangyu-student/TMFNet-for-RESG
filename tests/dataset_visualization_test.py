from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import Sen2MTCVariableDataset, _read_image, variable_temporal_collate
from visualize import save_training_visualization


def _write_scene(cloud_dir: Path, clear_dir: Path, name: str, frames: int) -> None:
    target = np.full((24, 20, 3), 5000, dtype=np.uint16)
    tifffile.imwrite(clear_dir / f"{name}.tif", target)
    for index in range(frames):
        observation = np.full((24, 20, 3), 1000 + index * 1000, dtype=np.uint16)
        tifffile.imwrite(cloud_dir / f"{name}_{index}.tif", observation)


def main() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        tile = "tile_001"
        cloud_dir = root / "Sen2_MTC" / tile / "cloud"
        clear_dir = root / "Sen2_MTC" / tile / "cloudless"
        cloud_dir.mkdir(parents=True)
        clear_dir.mkdir(parents=True)
        for split in ("train", "val", "test"):
            (root / f"{split}.txt").write_text(f"{tile}\n", encoding="utf-8")

        _write_scene(cloud_dir, clear_dir, "scene_a", frames=2)
        _write_scene(cloud_dir, clear_dir, "scene_b", frames=4)
        _write_scene(cloud_dir, clear_dir, "scene_too_short", frames=1)

        dark_tiff = root / "dark.tif"
        tifffile.imwrite(dark_tiff, np.full((4, 4, 3), 100, dtype=np.uint16))
        dark_tensor = _read_image(dark_tiff)
        assert torch.allclose(dark_tensor, torch.full_like(dark_tensor, -0.98))

        dataset = Sen2MTCVariableDataset(
            str(root), mode="val", min_temporal=2, max_temporal=6, augment=False
        )
        assert len(dataset) == 2
        samples = [dataset[index] for index in range(len(dataset))]
        batch = variable_temporal_collate(samples)
        assert batch["cond_image"].shape == (2, 4, 3, 24, 20)
        assert batch["valid_mask"].sum(dim=1).tolist() == [2, 4]

        comparison_path = root / "visualization" / "comparison.png"
        weights = torch.softmax(torch.randn(4, 1, 24, 20), dim=0)
        quality = torch.sigmoid(torch.randn(4, 1, 12, 10))
        gates = torch.sigmoid(torch.randn(4, 1, 12, 10))
        save_training_visualization(
            observations=batch["cond_image"][1],
            prediction=batch["gt_image"][1] * 0.9,
            target=batch["gt_image"][1],
            valid_mask=batch["valid_mask"][1],
            weights=weights,
            quality=quality,
            gates=gates,
            output_path=comparison_path,
        )
        assert comparison_path.is_file()
        assert (comparison_path.parent / "weights" / "weights.npy").is_file()
        assert (comparison_path.parent / "quality" / "quality.npy").is_file()
        assert (comparison_path.parent / "contribution_gates" / "gates.npy").is_file()
        with Image.open(comparison_path) as image:
            # Original-style main panel: T inputs + prediction + target, one row.
            assert image.size == (6 * 20, 24 + 24)
    print("Dataset and visualization smoke test passed.")


if __name__ == "__main__":
    main()
