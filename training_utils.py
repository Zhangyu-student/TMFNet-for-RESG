from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def set_seed(seed: int = 2026, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(_: int) -> None:
    """Give NumPy/Python RNGs the deterministic seed assigned by DataLoader."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def move_batch(batch: Dict, device: torch.device) -> Dict:
    result = dict(batch)
    for key in ("cond_image", "gt_image", "valid_mask"):
        if key in result:
            result[key] = result[key].to(device, non_blocking=True)
    return result


def reverse_valid_sequences(
    observations: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    reversed_observations = torch.zeros_like(observations)
    for batch_index in range(observations.shape[0]):
        positions = valid_mask[batch_index].nonzero(as_tuple=False).squeeze(1)
        reversed_observations[batch_index, positions] = torch.flip(
            observations[batch_index, positions], dims=[0]
        )
    return reversed_observations


def random_subset_mask(valid_mask: torch.Tensor, minimum: int = 2) -> torch.Tensor:
    subset = valid_mask.clone()
    for batch_index in range(valid_mask.shape[0]):
        positions = valid_mask[batch_index].nonzero(as_tuple=False).squeeze(1).tolist()
        length = len(positions)
        keep = random.randint(min(minimum, length), length)
        selected = random.sample(positions, keep)
        subset[batch_index] = False
        subset[batch_index, selected] = True
    return subset


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    config: Dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


def append_csv(path: str | Path, row: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_json(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
