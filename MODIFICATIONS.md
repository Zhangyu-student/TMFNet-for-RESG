# TMFNet → TMFNet++ modification map

## Correctness and consistency fixes

- Training and testing now instantiate the same model implementation.
- Reliability semantics are unified as `1 = reliable`, `0 = degraded`.
- All temporal blocks return complete state sequences rather than an incompatible
  final-state tensor between stacked blocks.
- Sentinel-2 validation/testing supports both the exact original TMFNet
  per-image min-max metric path and a fixed `[0,2000]` physical metric path.
  The original-compatible mode is the default for baseline comparison.

## Architecture changes

- Fixed three-frame input → variable-length input with padding mask.
- Unidirectional gated recurrence → bidirectional input-dependent diagonal SSM.
- Frame-only suppression head → multi-scale feature-consistency quality.
- Unconstrained learned quality → robust temporal-median consistency prior with
  a bounded learned suppressor, structurally preserving `high = reliable`.
- Last-token/mean aggregation → quality-gated bidirectional temporal fusion.
- Three independent quality heads → one quality map resized across all scales.
- Selection/contribution gates and local refinement scores → direct normalized
  quality weights, with no redundant intermediate gate.
- Independent skip averaging → quality-driven coarse-to-fine fusion with a
  configurable mild coarse-weight blend.
- The current implementation uses direct decoder prediction without a
  base-image residual path.

## Training changes

- Random temporal subset sampling.
- Optional temporal reversal augmentation.
- Optional order-consistency loss.
- Optional subset-consistency loss.
- AdamW + cosine schedule + AMP + gradient clipping.
- CSV and TensorBoard logging plus periodic validation visualization.
- Complete resumable checkpoints (model, optimizer, scheduler, epoch, best PSNR,
  and configuration).

## Dataset and visualization changes

- The original `Sen2_MTC/<tile>/cloud|cloudless` layout remains supported.
- Samples with fewer than `min_temporal` observations are rejected, and channel
  or spatial mismatches now produce explicit errors.
- Training visualization combines inputs, prediction, target, quality, and
  normalized fusion weights into one comparison PNG.
- Test CSV output includes weight entropy, effective frame count, and maximum
  temporal weight so winner-take-all behavior can be measured directly.

## Interface

```python
output = model(observations, valid_mask)
output, aux = model(observations, valid_mask, return_aux=True)
```

Shapes:

- `observations`: `[B,T,C,H,W]`
- `valid_mask`: `[B,T]`, Boolean
- `output`: `[B,C,H,W]`

The compatibility aliases `TemporalMambaFusionNet`, `models.TMFNet`, and
`models.mamba_test` are retained, but old TMFNet checkpoints are not strictly
compatible with the new architecture.
