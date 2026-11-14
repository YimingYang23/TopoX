#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
GPUS=${1:?Usage: bash tools/dist_train.sh GPUS STAGE [extra arguments]}
STAGE=${2:?Choose stage 1 or 2}
shift 2
case "$STAGE" in
  1) ENTRY=tools/train.py ;;
  2) ENTRY=tools/streamtrain.py ;;
  *) echo 'STAGE must be 1 or 2' >&2; exit 2 ;;
esac
python -m torch.distributed.run --nproc_per_node="$GPUS" --master_port="${PORT:-29500}" \
  "$ENTRY" "configs/topox_stage${STAGE}.py" --launcher pytorch \
  --work-dir "work_dirs/topox/stage${STAGE}" --seed 42 "$@"
