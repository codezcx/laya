#!/usr/bin/env bash
# Four-GPU warmup -> LoRA -> held-out test. A failed stage stops the pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR="${1:?Usage: bash scripts/run_reve_qwen_pipeline.sh RUN_DIR}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
mkdir -p "$RUN_DIR"
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "failed (exit $code)" > "$RUN_DIR/status.txt"; fi' EXIT
echo warmup > "$RUN_DIR/status.txt"
bash scripts/train_reve_qwen.sh --config "$RUN_DIR/config.json" \
  --stage warmup --epochs 3 --allow-unverified-provenance \
  --output "$RUN_DIR/warmup"
echo lora > "$RUN_DIR/status.txt"
bash scripts/train_reve_qwen.sh --config "$RUN_DIR/config.json" \
  --stage lora --epochs 3 --allow-unverified-provenance \
  --init-checkpoint "$RUN_DIR/warmup/best" --output "$RUN_DIR/lora"
echo test > "$RUN_DIR/status.txt"
bash scripts/train_reve_qwen.sh --config "$RUN_DIR/config.json" \
  --stage lora --evaluate test --allow-unverified-provenance \
  --init-checkpoint "$RUN_DIR/lora/best" --output "$RUN_DIR/test_lora"
echo complete > "$RUN_DIR/status.txt"
