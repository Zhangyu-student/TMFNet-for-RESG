from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

try:
    import tifffile as tiff
except ImportError:  # pragma: no cover
    tiff = None


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _natural_key(path: Path) -> List[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _read_image(path: Path, channels: int = 3) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        if tiff is None:
            raise ImportError("tifffile is required to read TIFF files.")
        array = np.asarray(tiff.imread(str(path)))
    else:
        array = np.asarray(Image.open(path).convert("RGB"))

    if array.ndim == 2:
        array = array[..., None]
    if array.shape[0] <= 16 and array.ndim == 3 and array.shape[-1] > 16:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3:
        raise ValueError(f"Unsupported image shape {array.shape} for {path}")
    if array.shape[-1] < channels:
        if array.shape[-1] == 1:
            array = np.repeat(array, channels, axis=-1)
        else:
            raise ValueError(
                f"Image {path} has {array.shape[-1]} channels, but {channels} are required."
            )
    array = array[..., :channels].astype(np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    max_value = float(np.nanmax(array)) if array.size else 1.0
    if max_value > 1.5:
        divisor = 10000.0 if max_value > 255.0 else 255.0
        array = array / divisor
    array = np.clip(array, 0.0, 1.0)
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    return tensor.mul(2.0).sub(1.0)


def _apply_joint_augmentation(images: List[torch.Tensor]) -> List[torch.Tensor]:
    if random.random() < 0.5:
        images = [torch.flip(image, dims=[2]) for image in images]
    if random.random() < 0.5:
        images = [torch.flip(image, dims=[1]) for image in images]
    rotations = random.randint(0, 3)
    if rotations:
        images = [torch.rot90(image, rotations, dims=[1, 2]) for image in images]
    return images


@dataclass
class SampleRecord:
    observations: List[Path]
    target: Path
    name: str


class Sen2MTCVariableDataset(data.Dataset):
    """Variable-length replacement for the original Sen2_MTC_New_Multi dataset.

    It remains compatible with the original layout:
      root/train.txt, val.txt, test.txt
      root/Sen2_MTC/<tile>/cloud/<sample>_0.tif, ...
      root/Sen2_MTC/<tile>/cloudless/<sample>.tif

    Unlike the original loader, all matching temporal files are discovered.
    """

    def __init__(
        self,
        data_root: str,
        mode: str = "train",
        input_channels: int = 3,
        min_temporal: int = 2,
        max_temporal: Optional[int] = 6,
        random_temporal_subset: bool = True,
        random_reverse: bool = True,
        augment: bool = True,
    ) -> None:
        self.root = Path(data_root)
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"mode must be train, val, or test; got {mode!r}.")
        if min_temporal < 1:
            raise ValueError("min_temporal must be at least 1.")
        if max_temporal is not None and max_temporal < min_temporal:
            raise ValueError("max_temporal must be greater than or equal to min_temporal.")
        self.mode = mode
        self.input_channels = input_channels
        self.min_temporal = min_temporal
        self.max_temporal = max_temporal
        self.random_temporal_subset = random_temporal_subset and mode == "train"
        self.random_reverse = random_reverse and mode == "train"
        self.augment = augment and mode == "train"
        self.records: List[SampleRecord] = []

        split_file = self.root / f"{mode}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        tiles = [line.strip() for line in split_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not tiles:
            raise RuntimeError(f"Split file is empty: {split_file}")
        for tile in tiles:
            tile_root = self.root / "Sen2_MTC" / str(tile)
            cloud_dir = tile_root / "cloud"
            clear_dir = tile_root / "cloudless"
            if not cloud_dir.exists() or not clear_dir.exists():
                continue
            for target in sorted(clear_dir.iterdir(), key=_natural_key):
                if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                stem = target.stem
                candidates = [
                    path
                    for path in cloud_dir.glob(f"{stem}_*")
                    if path.suffix.lower() in SUPPORTED_EXTENSIONS
                ]
                candidates = sorted(candidates, key=_natural_key)
                if not candidates:
                    # Also support the README example image_0.tif while target is image.tif.
                    candidates = sorted(
                        [p for p in cloud_dir.glob(f"{stem}*.*") if p.suffix.lower() in SUPPORTED_EXTENSIONS],
                        key=_natural_key,
                    )
                if len(candidates) < self.min_temporal:
                    continue
                self.records.append(
                    SampleRecord(candidates, target, f"{tile}_{stem}")
                )
        if not self.records:
            raise RuntimeError(
                f"No samples with at least {self.min_temporal} observations were found "
                f"under {self.root} for split {mode}. Expected "
                "Sen2_MTC/<tile>/cloud/<sample>_*.tif and cloudless/<sample>.tif."
            )

    def __len__(self) -> int:
        return len(self.records)

    def _select_indices(self, count: int) -> List[int]:
        upper = count if self.max_temporal is None else min(count, self.max_temporal)
        lower = min(self.min_temporal, upper)
        if self.random_temporal_subset and upper > lower:
            length = random.randint(lower, upper)
        else:
            length = upper
        if length < count:
            indices = sorted(random.sample(range(count), length)) if self.mode == "train" else list(range(length))
        else:
            indices = list(range(count))
        if self.random_reverse and random.random() < 0.5:
            indices.reverse()
        return indices

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        record = self.records[index]
        indices = self._select_indices(len(record.observations))
        selected_paths = [record.observations[i] for i in indices]
        observations = [_read_image(path, self.input_channels) for path in selected_paths]
        target = _read_image(record.target, self.input_channels)
        for path, observation in zip(selected_paths, observations):
            if observation.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for {record.name}: observation {path} has "
                    f"{tuple(observation.shape)}, target {record.target} has {tuple(target.shape)}."
                )
        if self.augment:
            augmented = _apply_joint_augmentation(observations + [target])
            observations, target = augmented[:-1], augmented[-1]

        return {
            "cond_image": torch.stack(observations, dim=0),
            "gt_image": target,
            "path": f"{record.name}.png",
        }


class LegacyMultipleDataset(data.Dataset):
    """Compatibility loader for the original cloudy/clear JPEG layout."""

    def __init__(self, data_root: str, mode: str = "train", input_channels: int = 3, **_: object) -> None:
        self.root = Path(data_root)
        self.mode = mode
        self.input_channels = input_channels
        cloudy_dir = self.root / "cloudy"
        clear_dir = self.root / "clear"
        self.targets = sorted([p for p in clear_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS], key=_natural_key)
        self.records: List[Tuple[List[Path], Path]] = []
        for target in self.targets:
            stem = target.stem
            observations = sorted(
                [p for p in cloudy_dir.glob(f"{stem}_*.*") if "_ir" not in p.stem],
                key=_natural_key,
            )
            if observations:
                self.records.append((observations, target))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        paths, target_path = self.records[index]
        observations = [_read_image(path, self.input_channels) for path in paths]
        target = _read_image(target_path, self.input_channels)
        if self.mode == "train":
            augmented = _apply_joint_augmentation(observations + [target])
            observations, target = augmented[:-1], augmented[-1]
        return {
            "cond_image": torch.stack(observations),
            "gt_image": target,
            "path": f"{target_path.stem}.png",
        }


def variable_temporal_collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    max_t = max(int(item["cond_image"].shape[0]) for item in batch)  # type: ignore[index]
    first = batch[0]["cond_image"]  # type: ignore[index]
    _, channels, height, width = first.shape
    observations = torch.zeros(len(batch), max_t, channels, height, width, dtype=first.dtype)
    valid_mask = torch.zeros(len(batch), max_t, dtype=torch.bool)
    targets = []
    paths = []

    for index, item in enumerate(batch):
        sequence = item["cond_image"]  # type: ignore[index]
        if tuple(sequence.shape[1:]) != (channels, height, width):
            raise ValueError(
                "All observations in one batch must share [C,H,W]; "
                f"got {tuple(sequence.shape[1:])} and {(channels, height, width)}."
            )
        length = sequence.shape[0]
        observations[index, :length] = sequence
        valid_mask[index, :length] = True
        targets.append(item["gt_image"])  # type: ignore[index]
        paths.append(item["path"])  # type: ignore[index]

    return {
        "cond_image": observations,
        "gt_image": torch.stack(targets),
        "valid_mask": valid_mask,
        "path": paths,
    }


# Compatibility aliases used by the original repository.
Sen2_MTC_New_Multi = Sen2MTCVariableDataset
MultipleDataset = LegacyMultipleDataset
