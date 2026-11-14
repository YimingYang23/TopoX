# Installation

Use Linux, an NVIDIA GPU, and Python 3.8. The reference environment uses PyTorch 1.9.0 with CUDA 11.1 and the OpenMMLab 1.x stack.

```bash
git clone https://github.com/YimingYang23/TopoX.git
cd TopoX
conda create -n topox python=3.8 -y
conda activate topox
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.26.0 mmsegmentation==0.29.1
pip install mmdet3d==1.0.0rc6
pip install -r requirements.txt

mkdir -p third_party
git clone --branch v2.1.0 --depth 1 \
  https://github.com/OpenDriveLab/OpenLane-V2.git third_party/OpenLane-V2
pip install -e third_party/OpenLane-V2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c "import topox; print('TopoX is ready')"
```

MMCV must match the PyTorch/CUDA versions above because deformable attention uses compiled CUDA operators. Launch scripts set the repository import path automatically.

Run the graph unit checks with:

```bash
python -m unittest discover -s tests -v
```
