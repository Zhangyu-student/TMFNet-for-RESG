from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from metrics import image_metrics, process_rgb, temporal_fusion_statistics
from setup_utils import create_dataloader, create_model
from training_utils import load_json, move_batch


def save_image(
    tensor: torch.Tensor,
    path: Path,
    dataset_type: str,
    reflectance_scale: float,
    reflectance_max: float,
) -> None:
    array = process_rgb(tensor, dataset_type, reflectance_scale, reflectance_max)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def colorize_weight(weight: torch.Tensor) -> np.ndarray:
    value = weight.detach().float().cpu().numpy()
    value = np.clip(value, 0.0, 1.0)
    return np.round(value * 255.0).astype(np.uint8)


def evaluate(config: Dict) -> None:
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    test_loader = create_dataloader(config, "test")
    model = create_model(config, device)
    model.eval()

    output_dir = Path(config.get("test_output_dir", "inference_results/tmfnet_pp"))
    pred_dir = output_dir / "pred"
    gt_dir = output_dir / "gt"
    weight_dir = output_dir / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_type = str(config.get("dataset_type", "new_multi"))
    reflectance_scale = float(config.get("reflectance_scale", 10000.0))
    reflectance_max = float(config.get("metric_reflectance_max", 2000.0))

    rows: List[Dict[str, object]] = []
    totals = {
        "psnr": 0.0,
        "ssim": 0.0,
        "sam": 0.0,
        "mae": 0.0,
        "mean_quality": 0.0,
        "mean_gate": 0.0,
        "mean_max_weight": 0.0,
        "weight_entropy": 0.0,
        "effective_frames": 0.0,
    }
    count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = move_batch(batch, device)
            output, aux = model(
                batch["cond_image"], batch["valid_mask"], return_aux=True
            )
            for index, name in enumerate(batch["path"]):
                file_name = Path(str(name)).name
                save_image(
                    output[index], pred_dir / file_name, dataset_type,
                    reflectance_scale, reflectance_max,
                )
                save_image(
                    batch["gt_image"][index], gt_dir / file_name, dataset_type,
                    reflectance_scale, reflectance_max,
                )
                scores = image_metrics(
                    output[index], batch["gt_image"][index], dataset_type,
                    reflectance_scale, reflectance_max,
                )
                valid_t = int(batch["valid_mask"][index].sum())
                weights = aux["weights_full"][index, :valid_t]
                quality = aux["quality_full"][index, :valid_t]
                gates = aux["gates_full"][index, :valid_t]
                fusion_stats = temporal_fusion_statistics(weights, quality, gates)
                row: Dict[str, object] = {
                    "name": file_name,
                    "T": valid_t,
                    **scores,
                    **fusion_stats,
                }
                rows.append(row)
                for key in totals:
                    totals[key] += float(row[key])
                count += 1

                if config.get("save_temporal_weights", True):
                    sample_dir = weight_dir / Path(file_name).stem
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    weights = weights[:, 0]
                    quality = quality[:, 0]
                    gates = gates[:, 0]
                    np.save(sample_dir / "weights.npy", weights.detach().cpu().numpy())
                    np.save(sample_dir / "quality.npy", quality.detach().cpu().numpy())
                    np.save(sample_dir / "contribution_gates.npy", gates.detach().cpu().numpy())
                    for time_index in range(valid_t):
                        Image.fromarray(colorize_weight(weights[time_index])).save(
                            sample_dir / f"weight_t{time_index}.png"
                        )
                        Image.fromarray(colorize_weight(quality[time_index])).save(
                            sample_dir / f"quality_t{time_index}.png"
                        )
                        Image.fromarray(colorize_weight(gates[time_index])).save(
                            sample_dir / f"gate_t{time_index}.png"
                        )

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "name", "T", "psnr", "ssim", "sam", "mae",
            "mean_quality", "mean_gate", "mean_max_weight",
            "weight_entropy", "effective_frames",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {key: value / max(count, 1) for key, value in totals.items()}
    text = "\n".join([f"{key.upper()}: {value:.6f}" for key, value in summary.items()])
    (output_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test TMFNet++")
    parser.add_argument("--config", default="configs/tmfnet_pp.json")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    config = load_json(args.config)
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    evaluate(config)


if __name__ == "__main__":
    main()
