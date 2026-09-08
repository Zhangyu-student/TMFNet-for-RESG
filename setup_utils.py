from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import LegacyMultipleDataset, Sen2MTCVariableDataset, variable_temporal_collate
from models import TMFNetPlusPlus
from training_utils import seed_worker


def create_dataloader(config: Dict, mode: str) -> DataLoader:
    if mode not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {mode}")
    dataset_type = config.get("dataset_type", "new_multi")
    dataset_class = Sen2MTCVariableDataset if dataset_type == "new_multi" else LegacyMultipleDataset
    common = {
        "data_root": config["data_root"] if dataset_type == "new_multi" else os.path.join(config["data_root"], "multipleImage"),
        "input_channels": config.get("input_channels", 3),
        "min_temporal": config.get("min_temporal", 2),
        "max_temporal": config.get("max_temporal", 6),
        "random_temporal_subset": config.get("random_temporal_subset", True),
        "random_reverse": config.get("random_reverse", True),
        "reflectance_scale": config.get("reflectance_scale", 10000.0),
    }
    if dataset_type != "new_multi":
        common = {"data_root": common["data_root"], "input_channels": common["input_channels"]}

    dataset = dataset_class(mode=mode, **common)

    workers = int(config.get("num_workers", 4))
    seed = int(config.get("seed", 2026))
    loader_args = {
        "batch_size": int(config.get("batch_size", 4)),
        "num_workers": workers,
        "pin_memory": bool(config.get("pin_memory", True) and torch.cuda.is_available()),
        "collate_fn": variable_temporal_collate,
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker,
    }
    split_offset = {"train": 0, "val": 1, "test": 2}[mode]
    generator = torch.Generator().manual_seed(seed + split_offset)
    return DataLoader(
        dataset,
        shuffle=mode == "train",
        drop_last=False,
        generator=generator,
        **loader_args,
    )


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Compatibility helper for callers that explicitly need all three splits."""
    return tuple(create_dataloader(config, mode) for mode in ("train", "val", "test"))  # type: ignore[return-value]


def create_model(config: Dict, device: torch.device | str | None = None) -> TMFNetPlusPlus:
    device = torch.device(device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = TMFNetPlusPlus(
        input_channels=int(config.get("input_channels", 3)),
        output_channels=int(config.get("output_channels", config.get("input_channels", 3))),
        base_channels=int(config.get("base_channels", 48)),
        temporal_depth=int(config.get("temporal_depth", 2)),
        state_dim=int(config.get("state_dim", 8)),
        temporal_expansion=int(config.get("temporal_expansion", 2)),
        dropout=float(config.get("dropout", 0.0)),
        coarse_weight_blend=float(config.get("coarse_weight_blend", 0.25)),
        quality_temperature=float(config.get("quality_temperature", 1.5)),
        quality_learned_strength=float(config.get("quality_learned_strength", 0.25)),
        quality_floor=float(config.get("quality_floor", 0.02)),
    ).to(device)

    checkpoint_path = config.get("pretrained_path") or config.get("checkpoint")
    if checkpoint_path:
        checkpoint_path = os.path.expanduser(str(checkpoint_path))
        if os.path.isfile(checkpoint_path):
            payload = torch.load(checkpoint_path, map_location=device)
            state_dict = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint: {checkpoint_path}")
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
        else:
            print(f"Checkpoint not found, training from scratch: {checkpoint_path}")
    return model
