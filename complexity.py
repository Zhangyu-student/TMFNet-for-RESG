from __future__ import annotations

import argparse
import time

import torch

from models import TMFNetPlusPlus
from training_utils import count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="TMFNet++ parameters and inference speed")
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--base", type=int, default=48)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = TMFNetPlusPlus(base_channels=args.base).to(device).eval()
    observations = torch.randn(1, args.T, 3, args.height, args.width, device=device)
    mask = torch.ones(1, args.T, device=device, dtype=torch.bool)
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,} ({total / 1e6:.3f} M)")
    print(f"Trainable parameters: {trainable:,} ({trainable / 1e6:.3f} M)")

    with torch.no_grad():
        for _ in range(5):
            model(observations, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.runs):
            model(observations, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / args.runs * 1000.0
    print(f"Average inference time: {elapsed:.3f} ms")

    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, (observations, mask)).total()
        print(f"FLOPs: {flops / 1e9:.3f} G")
    except Exception as error:
        print(f"FLOPs not reported ({error}). Install fvcore if needed.")


if __name__ == "__main__":
    main()
