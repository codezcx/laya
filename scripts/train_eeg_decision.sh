#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
PYTHON_BIN="${EEG_PYTHON:-$PWD/laya/.venv-eeg/bin/python}"
NPROC="${NPROC_PER_NODE:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
"$PYTHON_BIN" -c 'import sys,torch; n=int(sys.argv[1]); assert 0<n<=torch.cuda.device_count(), f"Requested {n} ranks, found {torch.cuda.device_count()} visible GPUs"' "$NPROC"
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m tusz_sft.train_decision "$@"
