#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DIAG_ROOT="${1:?Provide a fresh diagnostic directory}"
[[ "${2:-}" == "--allow-local-reve-code" ]] || exit 2
[[ ! -e "$DIAG_ROOT" ]] || exit 2
mkdir -p "$DIAG_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
DIAG_PYTHON="$PWD/laya/.venv-eeg/bin/python"
extract_pids=()
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$rank" "$DIAG_PYTHON" -m eeg_align.extract_stage_features \
    --output "$DIAG_ROOT" --rank "$rank" --world 4 --batch-size 64 \
    --allow-local-reve-code > "$DIAG_ROOT/extract_$rank.log" 2>&1 &
  extract_pids+=("$!")
done
failed=0
for pid in "${extract_pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" == 0 ]] || exit 1
stages=(reve connector qwen_eeg qwen_answer)
probe_pids=()
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$rank" "$DIAG_PYTHON" -m eeg_align.fit_stage_probe \
    --root "$DIAG_ROOT" --stage "${stages[$rank]}" --world 4 \
    > "$DIAG_ROOT/probe_${stages[$rank]}.log" 2>&1 &
  probe_pids+=("$!")
done
for pid in "${probe_pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
