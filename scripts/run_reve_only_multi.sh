#!/usr/bin/env bash
# Two independent REVE-only experiments, on separate available GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR="${1:?Usage: bash scripts/run_reve_only_multi.sh RUN_DIR --allow-local-reve-code}"
if [[ "${2:-}" != "--allow-local-reve-code" ]]; then
  echo "Explicit authorization to execute the local REVE snapshots is required." >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Use a new run directory: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
EEG_PYTHON="${EEG_PYTHON:-$PWD/laya/.venv-eeg/bin/python}"
CUDA_VISIBLE_DEVICES="${REVE_PROBE_GPU:-0}" "$EEG_PYTHON" -m eeg_align.train_reve_only \
  --mode linear_probe --output "$RUN_DIR/linear_probe" \
  --allow-local-reve-code --allow-unverified-provenance > "$RUN_DIR/linear_probe.log" 2>&1 &
probe_pid=$!
CUDA_VISIBLE_DEVICES="${REVE_FULL_GPU:-1}" "$EEG_PYTHON" -m eeg_align.train_reve_only \
  --mode full_sft --output "$RUN_DIR/full_sft" \
  --allow-local-reve-code --allow-unverified-provenance > "$RUN_DIR/full_sft.log" 2>&1 &
full_pid=$!
failed=0
wait "$probe_pid" || failed=1
wait "$full_pid" || failed=1
exit "$failed"
