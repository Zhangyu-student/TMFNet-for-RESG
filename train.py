from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from loss import ReconstructionLoss
from metrics import image_metrics
from setup_utils import create_dataloader, create_model
from training_utils import (
    append_csv,
    load_json,
    move_batch,
    random_subset_mask,
    reverse_valid_sequences,
    save_checkpoint,
    set_seed,
)
from visualize import save_training_visualization


def validate(model, loader, criterion, device: torch.device, config: Dict) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "sam": 0.0, "mae": 0.0}
    sample_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            batch = move_batch(batch, device)
            output = model(batch["cond_image"], batch["valid_mask"])
            loss, _ = criterion(output, batch["gt_image"])
            batch_size = output.shape[0]
            totals["loss"] += float(loss.item()) * batch_size
            for index in range(batch_size):
                scores = image_metrics(
                    output[index],
                    batch["gt_image"][index],
                    dataset_type=str(config.get("dataset_type", "new_multi")),
                    reflectance_scale=float(config.get("reflectance_scale", 10000.0)),
                    reflectance_max=float(config.get("metric_reflectance_max", 2000.0)),
                    metric_mode=str(config.get("metric_mode", "original_tmfnet")),
                )
                for key in ("psnr", "ssim", "sam", "mae"):
                    totals[key] += scores[key]
            sample_count += batch_size
    return {key: value / max(sample_count, 1) for key, value in totals.items()}


def _create_summary_writer(config: Dict):
    if not bool(config.get("tensorboard", True)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard is not installed; PNG and CSV logging will still be saved.")
        return None
    return SummaryWriter(log_dir=str(config.get("tensorboard_dir", "runs/TMFNet_pp")))


def _create_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # PyTorch 2.0 compatibility
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _validation_visualization_index(
    dataset_size: int,
    selection_step: int,
    seed: int,
) -> int:
    """Choose a reproducible random sample, covering each scene once per cycle."""
    if dataset_size < 1:
        raise ValueError("Cannot visualize an empty validation dataset.")
    cycle, position = divmod(max(selection_step, 1) - 1, dataset_size)
    generator = torch.Generator().manual_seed(seed + cycle)
    return int(torch.randperm(dataset_size, generator=generator)[position].item())


def _restore_training_state(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    metric_mode: str,
) -> tuple[int, float]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("A resume checkpoint must contain model/optimizer/scheduler training state.")
    model.load_state_dict(payload["model"], strict=True)
    if payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    completed_epoch = int(payload.get("epoch", 0))
    best_metric = float(payload.get("best_metric", -math.inf))
    saved_config = payload.get("config", {})
    saved_metric_mode = (
        str(saved_config.get("metric_mode", "reflectance_2000"))
        if isinstance(saved_config, dict)
        else "reflectance_2000"
    )
    if saved_metric_mode != metric_mode:
        print(
            f"Validation metric mode changed from {saved_metric_mode} to {metric_mode}; "
            "resetting best PSNR because the values are not comparable."
        )
        best_metric = -math.inf
    print(f"Resumed full training state from {path} at epoch {completed_epoch}.")
    return completed_epoch + 1, best_metric


def _visualize_validation_sample(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    epoch: int,
    output_root: Path,
    writer=None,
    seed: int = 2026,
    selection_step: int = 1,
) -> Path:
    model.eval()
    dataset_index = _validation_visualization_index(
        len(loader.dataset), selection_step, seed
    )
    sample = loader.dataset[dataset_index]
    batch = move_batch(loader.collate_fn([sample]), device)
    with torch.no_grad():
        prediction, aux = model(batch["cond_image"], batch["valid_mask"], return_aux=True)
    sample_index = 0
    sample_name = Path(str(batch["path"][sample_index])).stem
    output_path = output_root / f"epoch_{epoch:04d}_{sample_name}.png"
    output_path = save_training_visualization(
        observations=batch["cond_image"][sample_index],
        prediction=prediction[sample_index],
        target=batch["gt_image"][sample_index],
        valid_mask=batch["valid_mask"][sample_index],
        weights=aux["weights_full"][sample_index],
        quality=aux["quality_full"][sample_index],
        gates=aux["gates_full"][sample_index],
        output_path=output_path,
    )
    if writer is not None:
        writer.add_image(
            "validation/prediction",
            ((prediction[sample_index].detach().float().cpu() + 1.0) * 0.5).clamp(0, 1),
            epoch,
        )
        writer.add_image(
            "validation/ground_truth",
            ((batch["gt_image"][sample_index].detach().float().cpu() + 1.0) * 0.5).clamp(0, 1),
            epoch,
        )
        with Image.open(output_path) as preview:
            comparison = np.asarray(preview.convert("RGB")).copy()
        writer.add_image("validation/comparison", comparison, epoch, dataformats="HWC")
    return output_path


def train(config: Dict) -> None:
    set_seed(
        int(config.get("seed", 2026)),
        deterministic=bool(config.get("deterministic", True)),
    )
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        device = torch.device("cpu")
    train_loader = create_dataloader(config, "train")
    val_loader = create_dataloader(config, "val")
    model = create_model(config, device)
    criterion = ReconstructionLoss(**config.get("loss", {})).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        betas=(0.9, 0.999),
    )
    epochs = int(config.get("num_epochs", 300))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=float(config.get("min_learning_rate", 1e-6))
    )
    amp_enabled = bool(config.get("amp", True) and device.type == "cuda")
    scaler = _create_grad_scaler(amp_enabled)
    grad_clip = float(config.get("grad_clip", 1.0))
    order_weight = float(config.get("order_consistency_weight", 0.0))
    subset_weight = float(config.get("subset_consistency_weight", 0.0))
    metric_mode = str(config.get("metric_mode", "original_tmfnet"))
    print(f"Validation metric mode: {metric_mode}")

    save_dir = Path(config.get("save_dir", "checkpoints/tmfnet_pp"))
    log_file = Path(config.get("log_file", save_dir / "training_log.csv"))
    visualization_dir = Path(config.get("visualization_dir", "visualizations/TMFNet_pp"))
    validation_interval = max(int(config.get("validation_interval", 1)), 1)
    visualization_interval = max(int(config.get("visualization_interval", 5)), 1)
    scalar_log_interval = max(int(config.get("tensorboard_log_interval", 10)), 1)
    save_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    best_psnr = -math.inf
    start_epoch = 1
    visualization_step = 0
    resume_checkpoint = config.get("resume_checkpoint")
    if resume_checkpoint:
        start_epoch, best_psnr = _restore_training_state(
            resume_checkpoint, model, optimizer, scheduler, device, metric_mode
        )
    writer = _create_summary_writer(config)
    global_step = (start_epoch - 1) * len(train_loader)

    try:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            running = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            for batch in progress:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                amp_context = torch.amp.autocast("cuda") if amp_enabled else nullcontext()
                with amp_context:
                    output = model(batch["cond_image"], batch["valid_mask"])
                    loss, components = criterion(output, batch["gt_image"])

                    if order_weight > 0:
                        reversed_x = reverse_valid_sequences(batch["cond_image"], batch["valid_mask"])
                        reversed_output = model(reversed_x, batch["valid_mask"])
                        order_loss = torch.nn.functional.l1_loss(reversed_output, output.detach())
                        loss = loss + order_weight * order_loss
                        components["order"] = order_loss

                    if subset_weight > 0:
                        subset_mask = random_subset_mask(batch["valid_mask"], minimum=2)
                        subset_output = model(batch["cond_image"], subset_mask)
                        subset_loss = torch.nn.functional.l1_loss(subset_output, output.detach())
                        loss = loss + subset_weight * subset_loss
                        components["subset"] = subset_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.item())
                if writer is not None and global_step % scalar_log_interval == 0:
                    writer.add_scalar("train/total_loss", float(loss.item()), global_step)
                    for name, value in components.items():
                        writer.add_scalar(f"train/{name}", float(value.detach().item()), global_step)
                    writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1
                progress.set_postfix(
                    loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}"
                )

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            train_loss = running / max(len(train_loader), 1)
            should_validate = (
                epoch == start_epoch or epoch % validation_interval == 0 or epoch == epochs
            )
            if not should_validate:
                save_checkpoint(
                    save_dir / "latest.pth", model, optimizer, scheduler, epoch, best_psnr, config
                )
                continue

            validation = validate(model, val_loader, criterion, device, config)
            improved = validation["psnr"] > best_psnr
            if improved:
                best_psnr = validation["psnr"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation["loss"],
                "psnr": validation["psnr"],
                "ssim": validation["ssim"],
                "sam": validation["sam"],
                "mae": validation["mae"],
                "lr": current_lr,
            }
            append_csv(log_file, row)
            print(
                f"Epoch {epoch}: train={train_loss:.5f}, val={validation['loss']:.5f}, "
                f"PSNR={validation['psnr']:.3f}, SSIM={validation['ssim']:.4f}, "
                f"SAM={validation['sam']:.3f}"
            )
            if writer is not None:
                writer.add_scalar("epoch/train_loss", train_loss, epoch)
                for name, value in validation.items():
                    writer.add_scalar(f"validation/{name}", value, epoch)
            save_checkpoint(
                save_dir / "latest.pth", model, optimizer, scheduler, epoch, best_psnr, config
            )
            if improved:
                save_checkpoint(
                    save_dir / "best_psnr.pth", model, optimizer, scheduler, epoch, best_psnr, config
                )
            if epoch == start_epoch or epoch % visualization_interval == 0 or epoch == epochs:
                path = _visualize_validation_sample(
                    model,
                    val_loader,
                    device,
                    epoch,
                    visualization_dir,
                    writer,
                    seed=int(config.get("seed", 2026)),
                    selection_step=visualization_step + 1,
                )
                visualization_step += 1
                print(f"Saved validation visualization: {path}")
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TMFNet++")
    parser.add_argument("--config", default="configs/tmfnet_pp.json")
    args = parser.parse_args()
    train(load_json(args.config))


if __name__ == "__main__":
    main()
