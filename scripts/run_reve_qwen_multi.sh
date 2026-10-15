#!/usr/bin/env bash
# Subject-disjoint mixed MI training. Test sets never select checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR="${1:?Usage: bash scripts/run_reve_qwen_multi.sh RUN_DIR}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
mkdir -p "$RUN_DIR"
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "failed (exit $code)" > "$RUN_DIR/status.txt"; fi' EXIT
echo warmup > "$RUN_DIR/status.txt"
bash scripts/train_reve_qwen.sh --config "$RUN_DIR/config.json" \
  --stage warmup --epochs 1 --allow-unverified-provenance \
  --output "$RUN_DIR/warmup"
echo lora > "$RUN_DIR/status.txt"
bash scripts/train_reve_qwen.sh --config "$RUN_DIR/config.json" \
  --stage lora --epochs 3 --allow-unverified-provenance \
  --init-checkpoint "$RUN_DIR/warmup/best" --output "$RUN_DIR/lora"
for SPLIT in test external_2a external_2b; do
  echo "$SPLIT" > "$RUN_DIR/status.txt"
  bash scripts/train_reve_qwen.sh --config "$RUN_DIR/config.json" \
    --stage lora --evaluate "$SPLIT" --allow-unverified-provenance \
    --init-checkpoint "$RUN_DIR/lora/best" --output "$RUN_DIR/eval_$SPLIT"
done
echo complete > "$RUN_DIR/status.txt"
