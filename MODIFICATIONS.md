# TMFNet → TMFNet++ modification map

## Correctness and consistency fixes

- Training and testing now instantiate the same model implementation.
- Reliability semantics are unified as `1 = reliable`, `0 = degraded`.
- All temporal blocks return complete state sequences rather than an incompatible
  final-state tensor between stacked blocks.
- Sentinel-2 validation/testing maps `[-1,1]` back through the `10000` scale,
  clips to `[0,2000]`, and evaluates with `data_range=2000`; it does not apply
  per-image min-max stretching.

## Architecture changes

- Fixed three-frame input → variable-length input with padding mask.
- Unidirectional gated recurrence → bidirectional input-dependent diagonal SSM.
- Frame-only suppression head → multi-scale feature-consistency quality.
- Last-token/mean aggregation → quality-gated bidirectional temporal fusion.
- Competitive softmax and recursive log-weight sharpening → independent sigmoid
  quality/contribution gates, normalized only at each actual feature fusion.
- Independent skip averaging → hierarchical coarse-to-fine refinement with a
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
- Inputs, prediction, ground truth, absolute error, temporal weights, and
  independent quality maps, contribution gates, and normalized fusion weights
  are saved together and as separate paper-ready files.
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
