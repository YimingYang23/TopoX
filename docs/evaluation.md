# Training and evaluation

## Released model

The main checkpoint combines iterative positional embedding, lane denoising, temporal BEV fusion, propagated lane queries, and explicit directed adjacency memory. For each frame, the model stores the 66 selected queries and the adjacency submatrix in the same order. Incoming and outgoing messages update the propagated queries with confidence gating and a residual scale of 0.25.

Training uses the two-stage schedule in the README. The stage-2 configuration enables graph memory and aggregation with `stream_topology_loss=False`; topology is supervised through the current-frame relationship head. GeoDist is applied during output formatting through `output_geodist=True`.

## Frame-level metrics

`tools/test.py` evaluates complete validation sequences. Prediction formatting preserves ten-point XYZ centerlines and boundaries. AP_lane and AP_ped are the lane and pedestrian-crossing average precision, and mAP is their mean. TOP_lsls scores directed lane-segment connectivity using the LaneSegNet/OpenLane-V2 v2.1 evaluator.

GeoDist uses

```text
A_output[i,j] = A_pred[i,j] + exp(-d(i,j)^2 / 11.5275)
d(i,j) = ||end(centerline_i) - start(centerline_j)||_1
```

The geometric diagonal is zero. Scores are not clipped or normalized. The operation follows lane formatting and does not alter geometry, confidence, boundary types, or the adjacency saved in temporal memory.

## Temporal results

The README reports completed sequence measurements on the same 150 validation scenes. Metrics use adjacent frames, confidence >= 0.3, and an edge threshold > 0.5 where a binary graph is required.

- **Centerline jitter:** ego-motion-aligned geometry change for instances correctly matched in both frames, in metres.
- **Adapted C-mAP:** a temporal extension of lane/crosswalk AP with an external association procedure and look-back 1.
- **Adapted temporal TOP_lsls:** native ten-point XYZ lane-segment matching followed by the v2.1 directed vertex AP rules. Persistent GT identities and unchanged GT relations define cross-frame items; the edge score is the smaller score from the two frames. Missing endpoints receive the evaluator's `(1 - GT) * (0.5 + float32 epsilon)` penalty.
- **Persistent-edge recall / path preservation:** recall of GT connections / multi-hop paths across the frame pair, including node misses.

Adapted temporal TOP_lsls covers 4,656 adjacent frame pairs. Its scene-bootstrap 95% confidence intervals are 11.82–15.59 for LaneSegNet, 8.77–11.74 for SQD-MapNet, and 18.97–24.05 for TopoX + GeoDist (1,000 resamples, seed 42). The SQD-MapNet lane-segment comparison embeds its 2D geometry in XYZ with zero height.

The standard evaluation command produces the frame-level metrics. Temporal measurements in the README are the separately scored sequence results summarized in [results.json](results.json).

## Checkpoint

The downloadable file contains `state_dict` and compact model metadata. All 932 state tensors are bitwise identical to the final training checkpoint at iteration 33,516. Optimizer state is omitted from the inference release.

Use `--load-from` to initialize training from a checkpoint, and `--resume-from` with a full training checkpoint to restore optimizer and runner state.
