#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PYTHON="${EEG_PYTHON:-$PWD/laya/.venv-eeg/bin/python}"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" -m eeg_align.train "$@"
