from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from metrics import calculate_sam


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired prediction and GT folders")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output", default="evaluation_results")
    parser.add_argument("--lpips", action="store_true")
    args = parser.parse_args()

    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    lpips_model = None
    if args.lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").eval()
        except ImportError:
            print("lpips is not installed; LPIPS will be skipped.")

    rows = []
    for pred_path in sorted(pred_dir.glob("*")):
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            continue
        pred, gt = read_rgb(pred_path), read_rgb(gt_path)
        row = {
            "name": pred_path.name,
            "psnr": peak_signal_noise_ratio(gt, pred, data_range=1.0),
            "ssim": structural_similarity(gt, pred, channel_axis=-1, data_range=1.0),
            "sam": calculate_sam(gt, pred),
            "mae": float(np.abs(gt - pred).mean()),
        }
        if lpips_model is not None:
            pred_t = torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0).mul(2).sub(1)
            gt_t = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).mul(2).sub(1)
            row["lpips"] = float(lpips_model(pred_t, gt_t).item())
        rows.append(row)

    if not rows:
        raise RuntimeError("No matching prediction/GT images were found.")
    fields = list(rows[0].keys())
    with (output / "detailed_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {field: float(np.mean([row[field] for row in rows])) for field in fields if field != "name"}
    (output / "summary_metrics.txt").write_text(
        "\n".join(f"{key.upper()}: {value:.6f}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
