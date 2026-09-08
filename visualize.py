from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

def _display_rgb(tensor: torch.Tensor, stretch_percent: float = 0.0) -> np.ndarray:
    """Convert a [-1,1] tensor to display RGB; stretching is display-only."""
    array = ((tensor.detach().float().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
    array = array.permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 2:
        array = np.concatenate([array, array[..., -1:]], axis=-1)
    else:
        array = array[..., :3]

    if stretch_percent > 0:
        stretched = np.empty_like(array)
        for channel in range(3):
            low, high = np.percentile(
                array[..., channel], [stretch_percent, 100.0 - stretch_percent]
            )
            if high > low:
                stretched[..., channel] = (array[..., channel] - low) / (high - low)
            else:
                stretched[..., channel] = array[..., channel]
        array = stretched
    return np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _colorize_map(value: torch.Tensor | np.ndarray, normalize: bool = False) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().squeeze().numpy()
    else:
        array = np.asarray(value, dtype=np.float32).squeeze()
    if normalize:
        scale = float(np.percentile(array, 98.0))
        array = array / max(scale, 1e-6)
    array = np.clip(array, 0.0, 1.0)
    # Compact blue-cyan-yellow-red map without adding a matplotlib dependency.
    red = np.clip(1.5 * array - 0.25, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(3.0 * array - 1.5), 0.0, 1.0)
    blue = np.clip(1.25 - 1.5 * array, 0.0, 1.0)
    return np.round(np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def _labeled_tile(array: np.ndarray, label: str, size: tuple[int, int]) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image)
    image = Image.fromarray(array).resize(size, resample=resampling.BILINEAR)
    tile = Image.new("RGB", (size[0], size[1] + 24), color=(18, 18, 18))
    tile.paste(image, (0, 24))
    ImageDraw.Draw(tile).text((6, 5), label, fill=(255, 255, 255))
    return tile


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_display_rgb(tensor)).save(path)


def save_temporal_weights(weights: torch.Tensor, directory: str | Path) -> None:
    """Save [T,1,H,W] or [T,H,W] temporal weights as grayscale PNGs and NPY."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = weights.detach().float().cpu()
    if weights.ndim == 4:
        weights = weights[:, 0]
    np.save(directory / "weights.npy", weights.numpy())
    for index, weight in enumerate(weights):
        image = np.round(weight.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        Image.fromarray(image).save(directory / f"weight_t{index}.png")


def save_training_visualization(
    observations: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    output_path: str | Path,
    valid_mask: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    quality: Optional[torch.Tensor] = None,
    gates: Optional[torch.Tensor] = None,
    stretch_percent: float = 2.0,
) -> Path:
    """Save the original single-row comparison and separate diagnostics.

    Inputs are unbatched: observations [T,C,H,W], prediction/target [C,H,W].
    The 2% linear stretch follows the original repository and affects PNGs only.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observations = observations.detach().float().cpu()
    prediction = prediction.detach().float().cpu()
    target = target.detach().float().cpu()
    if valid_mask is None:
        valid_mask = torch.ones(observations.shape[0], dtype=torch.bool)
    valid_indices = valid_mask.detach().cpu().bool().nonzero(as_tuple=False).squeeze(1).tolist()
    if not valid_indices:
        raise ValueError("The visualization sample has no valid temporal observation.")

    height, width = prediction.shape[-2:]
    tile_size = (min(width, 256), min(height, 256))
    input_arrays = [_display_rgb(observations[i], stretch_percent) for i in valid_indices]
    pred_array = _display_rgb(prediction, stretch_percent)
    target_array = _display_rgb(target, stretch_percent)
    pred01 = ((prediction + 1.0) * 0.5).clamp(0.0, 1.0)
    target01 = ((target + 1.0) * 0.5).clamp(0.0, 1.0)
    error_array = _colorize_map((pred01 - target01).abs().mean(dim=0), normalize=True)

    comparison = [
        (array, f"Input T{position + 1}")
        for position, array in enumerate(input_arrays)
    ] + [(pred_array, "Prediction"), (target_array, "Target")]
    if weights is not None:
        weights = weights.detach().float().cpu()
    if quality is not None:
        quality = quality.detach().float().cpu()
    if gates is not None:
        gates = gates.detach().float().cpu()

    columns = len(comparison)
    tile_width, tile_height = tile_size[0], tile_size[1] + 24
    canvas = Image.new("RGB", (columns * tile_width, tile_height), (10, 10, 10))
    for column_index, (array, label) in enumerate(comparison):
        canvas.paste(
            _labeled_tile(array, label, tile_size),
            (column_index * tile_width, 0),
        )
    canvas.save(output_path)

    # Separate files reproduce the original visualize2.py workflow for paper figures.
    epoch_dir = output_path.parent
    for array, temporal_index in zip(input_arrays, valid_indices):
        image_path = epoch_dir / "inputs" / f"input_t{temporal_index}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(image_path)
    for name, array in (
        ("prediction.png", pred_array),
        ("ground_truth.png", target_array),
        ("absolute_error.png", error_array),
    ):
        Image.fromarray(array).save(epoch_dir / name)
    if weights is not None:
        save_temporal_weights(weights[valid_indices], epoch_dir / "weights")
    if quality is not None:
        quality_dir = epoch_dir / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        np.save(quality_dir / "quality.npy", quality[valid_indices].numpy())
        for temporal_index in valid_indices:
            Image.fromarray(_colorize_map(quality[temporal_index])).save(
                quality_dir / f"quality_t{temporal_index}.png"
            )
    if gates is not None:
        gate_dir = epoch_dir / "contribution_gates"
        gate_dir.mkdir(parents=True, exist_ok=True)
        np.save(gate_dir / "gates.npy", gates[valid_indices].numpy())
        for temporal_index in valid_indices:
            Image.fromarray(_colorize_map(gates[temporal_index])).save(
                gate_dir / f"gate_t{temporal_index}.png"
            )
    return output_path
