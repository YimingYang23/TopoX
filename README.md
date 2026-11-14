<div align="center">

# TopoX
### Spatio-Temporal Lane Graph Generation via Iterative Positional Embedding and Denoising

**KDD 2027**

Yiming Yang, Yueru Luo, Bingkun He, Hongbin Lin, Suzhong Fu, Chao Zheng,<br>
Zhipeng Cao, Erlong Li, Chao Yan, Shuguang Cui, Zhen Li

[Preprint](https://arxiv.org/abs/2507.00709) · [Model weights](https://huggingface.co/y352184741/TopoX) · [Installation](docs/installation.md) · [Data preparation](docs/data.md)

</div>

TopoX builds lane graphs from surround-view video, jointly predicting lane geometry, boundary semantics, pedestrian crossings, and directed connectivity. Iterative positional embedding refines lane queries through the decoder, denoising strengthens structured prediction, and temporal graph propagation carries both lane features and their adjacency into the next frame.

## News

- **2026-11:** TopoX has been **accepted to KDD 2027**!
- **2026-11:** We release the core training and inference code, our checkpoint with temporal adjacency propagation, and the latest OpenLane-V2 results.

## Highlights

- **Explicit temporal graph memory.** Top-K lane queries and their ordered adjacency submatrix are stored together. Confidence-gated predecessor and successor messages update the next frame's queries.
- **Geometry and semantics in one representation.** Iterative positional embedding and lane denoising learn centerlines, boundaries, boundary types, and connectivity together.
- **Strong lane-graph reconstruction.** The released model reaches **39.06 mAP** and **32.10 TOP_lsls** on OpenLane-V2 Subset A with GeoDist output refinement.

<div align="center">
<img src="motivation1.gif" width="85%" alt="Temporal lane graph reconstruction">
</div>

## Results

### OpenLane-V2 Subset A

Validation on **150 scenes / 4,806 frames**, using the LaneSegNet/OpenLane-V2 v2.1 lane-segment evaluator. All scores below are percentages; higher is better.

| Model | Backbone | mAP | AP_lane | AP_ped | TOP_lsls | Config |
|---|---|---:|---:|---:|---:|---|
| TopoX | ResNet-50 | 39.06 | 37.40 | 40.71 | 30.15 | [Temporal graph](configs/topox_stage2.py) |
| **TopoX + GeoDist** | **ResNet-50** | **39.06** | **37.40** | **40.71** | **32.10** | [Temporal graph + GeoDist](configs/topox_stage2_geodist.py) |

Both rows use the **same released checkpoint**, including temporal adjacency propagation. GeoDist adds endpoint-distance scores to the final predicted adjacency. The temporal memory stores the learned graph. Exact scores and checkpoint identifiers are provided in [results.json](docs/results.json).

### Temporal consistency

The same validation sequences also measure geometric stability and persistence of graph connections.

| Model | Centerline jitter ↓ (m) | Adapted C-mAP ↑ | Adapted temporal TOP_lsls ↑ | Persistent-edge recall ↑ | Path preservation ↑ |
|---|---:|---:|---:|---:|---:|
| LaneSegNet | 0.255 | 8.19 | 13.69 | 15.04 | 9.87 |
| SQD-MapNet | 0.188 | 8.58 | 10.25 | 12.24 | 7.33 |
| **TopoX + GeoDist** | **0.123** | **13.75** | **21.51** | **25.73** | **18.58** |

TopoX preserves lane geometry and connectivity across frames, with more than **50% lower jitter** than LaneSegNet. Temporal metrics use confidence 0.3 and adjacent frames; edge/path recall includes missed nodes. Adapted temporal TOP_lsls follows the v2.1 lane-segment matching and missing-endpoint rules. See [evaluation protocols](docs/evaluation.md) for definitions.

## Quick start

Follow [installation](docs/installation.md) and [data preparation](docs/data.md), then download the checkpoint:

```bash
mkdir -p checkpoints
curl -L --fail --retry 3 \
  https://huggingface.co/y352184741/TopoX/resolve/main/topox_r50_openlanev2_subset_a.pth \
  -o checkpoints/topox_r50_openlanev2_subset_a.pth
sha256sum -c checkpoints.sha256
```

Evaluate the main result:

```bash
bash tools/dist_test.sh 1 checkpoints/topox_r50_openlanev2_subset_a.pth
```

Evaluate the learned graph with the same weights:

```bash
CONFIG=configs/topox_stage2.py \
  bash tools/dist_test.sh 1 checkpoints/topox_r50_openlanev2_subset_a.pth
```

Use `--show` for per-frame visualization. Evaluation assigns complete scenes to each GPU and preserves frame order.

## Training

The released schedule uses **4 GPUs, 2 samples per GPU, and seed 42**. Stage 1 learns per-frame lane representations for 12 epochs; stage 2 trains temporal graph propagation for 33,516 iterations.

```bash
bash tools/dist_train.sh 4 1
bash tools/dist_train.sh 4 2
```

Stage 2 automatically loads `work_dirs/topox/stage1/latest.pth`. The final checkpoint is saved under `work_dirs/topox/stage2/`. Set `CUDA_VISIBLE_DEVICES` to choose GPUs, or `PORT` to choose the distributed port. See [training and evaluation](docs/evaluation.md) for configuration details.

## Repository

```text
configs/             Two-stage training and GeoDist evaluation configs
topox/models/        TopoX detector, iterative decoder, and graph propagation
topox/streaming/     Temporal memory, motion updates, and sequence samplers
topox/bev/           Multi-camera BEV construction
topox/datasets/      OpenLane-V2 inputs, transforms, and prediction formatting
topox/core/          Matching, evaluation hooks, and GeoDist
tools/               Data preparation, training, and evaluation entry points
tests/               Directed propagation and GeoDist checks
```

## Visualizations

<div align="center">
<img src="vis1.gif" width="80%" alt="TopoX lane graph sequence">
<img src="vis2.gif" width="80%" alt="Lane geometry and topology across frames">
<img src="vis5.gif" width="80%" alt="Surround-view lane graph reconstruction">
</div>

## Citation

```bibtex
@inproceedings{yang2027topox,
  title={TopoX: Spatio-Temporal Lane Graph Generation via Iterative Positional Embedding and Denoising},
  author={Yang, Yiming and Luo, Yueru and He, Bingkun and Lin, Hongbin and Fu, Suzhong and Zheng, Chao and Cao, Zhipeng and Li, Erlong and Yan, Chao and Cui, Shuguang and Li, Zhen},
  booktitle={Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2027}
}
```

## Acknowledgements and license

TopoX builds on [LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet), [SQD-MapNet](https://github.com/shuowang666/SQD-MapNet), [StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet), and [BEVFormer](https://github.com/fundamentalvision/BEVFormer). We use [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2) for data and evaluation, and [TopoLogic](https://github.com/Franpin/TopoLogic) for GeoDist. The temporal evaluation draws on [MapTracker](https://github.com/woodfrog/maptracker).

Code is distributed under [GPL-3.0](LICENSE), with upstream notices retained in [NOTICE](NOTICE) and [licenses/](licenses/).
