from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loss import ReconstructionLoss
from models import TMFNetPlusPlus
from models.blocks import FeatureQualityEstimator


def run_case(temporal_length: int) -> None:
    torch.manual_seed(0)
    model = TMFNetPlusPlus(base_channels=12, temporal_depth=1, state_dim=4)
    observations = torch.randn(2, temporal_length, 3, 32, 32)
    valid_mask = torch.ones(2, temporal_length, dtype=torch.bool)
    if temporal_length > 2:
        valid_mask[1, -1] = False
        observations[1, -1] = 0
    output, aux = model(observations, valid_mask, return_aux=True)
    assert output.shape == (2, 3, 32, 32)
    assert output.abs().max() <= 1.0
    assert "base_image" not in aux
    assert "residual" not in aux
    assert aux["weights_full"].shape == (2, temporal_length, 1, 32, 32)
    assert aux["quality_full"].shape == (2, temporal_length, 1, 32, 32)
    assert aux["gates_full"].shape == (2, temporal_length, 1, 32, 32)
    assert torch.all((aux["quality_full"] >= 0) & (aux["quality_full"] <= 1))
    assert torch.all((aux["gates_full"] >= 0) & (aux["gates_full"] <= 1))
    sums = aux["weights_full"].sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    target = torch.randn_like(output).clamp(-1, 1)
    loss, parts = ReconstructionLoss()(output, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    print(f"T={temporal_length}: output={tuple(output.shape)}, loss={loss.item():.5f}")


def test_padding_invariance() -> None:
    torch.manual_seed(1)
    model = TMFNetPlusPlus(base_channels=12, temporal_depth=1, state_dim=4).eval()
    observations = torch.randn(1, 2, 3, 32, 32)
    output = model(observations, torch.ones(1, 2, dtype=torch.bool))

    padded = torch.cat([observations, torch.zeros(1, 1, 3, 32, 32)], dim=1)
    padded_mask = torch.tensor([[True, True, False]])
    padded_output = model(padded, padded_mask)
    assert torch.allclose(output, padded_output, atol=1e-5, rtol=1e-5)
    print("Padding invariance: passed")


def test_quality_semantics() -> None:
    estimator = FeatureQualityEstimator(
        channels=4,
        hidden_channels=8,
        temperature=1.5,
        learned_strength=0.25,
        quality_floor=0.02,
    ).eval()
    features = torch.zeros(1, 3, 4, 8, 8)
    features[:, 1] = 5.0  # T1 is the sole strong temporal outlier.
    valid_mask = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        quality = estimator(features, valid_mask)
    consistent_quality = torch.minimum(quality[:, 0], quality[:, 2])
    assert torch.all(quality[:, 1] < consistent_quality)
    assert float(quality[:, 1].max()) <= 0.021
    assert float(consistent_quality.min()) >= 0.75
    print("Quality semantics: temporal outlier receives lower quality")


if __name__ == "__main__":
    test_quality_semantics()
    for length in (2, 3, 5):
        run_case(length)
    test_padding_invariance()
    print("TMFNet++ smoke test passed.")
