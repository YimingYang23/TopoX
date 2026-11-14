#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
GPUS=${1:?Usage: bash tools/dist_test.sh GPUS CHECKPOINT [extra arguments]}
CHECKPOINT=${2:?Provide the checkpoint path}
shift 2
python -m torch.distributed.run --nproc_per_node="$GPUS" --master_port="${PORT:-29501}" \
  tools/test.py "${CONFIG:-configs/topox_stage2_geodist.py}" "$CHECKPOINT" \
  --launcher pytorch --out-dir work_dirs/topox/eval --eval openlane_v2 "$@"
