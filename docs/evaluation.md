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

The geometric diagonal is zero; off-diagonal scores use direct additive fusion. GeoDist refines the formatted output after temporal memory is saved.

## Checkpoint

The checkpoint contains the model `state_dict` and metadata for iteration 33,516. To verify the download, run `sha256sum -c checkpoints.sha256` from the repository root.

Use `--load-from` to initialize training from a checkpoint, and `--resume-from` with a full training checkpoint to restore optimizer and runner state.
