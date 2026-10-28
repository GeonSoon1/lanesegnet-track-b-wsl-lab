#!/usr/bin/env bash
set -euo pipefail
mountpoint -q /mnt/d || { echo 'SSD not mounted; no fallback permitted' >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GG49_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/engine"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
STORE=/mnt/d/mms_experiments/gg49-lanesegnet-lab_sj
mkdir -p "$STORE"/{cache/tmp,cache/pip,cache/matplotlib,cache/torch,cache/huggingface,cache/cuda,logs,reports,runs,checkpoints}
export TMPDIR="$STORE/cache/tmp" TMP="$STORE/cache/tmp" TEMP="$STORE/cache/tmp"
export PIP_CACHE_DIR="$STORE/cache/pip" MPLCONFIGDIR="$STORE/cache/matplotlib"
export TORCH_HOME="$STORE/cache/torch" HF_HOME="$STORE/cache/huggingface" XDG_CACHE_HOME="$STORE/cache"
export CUDA_CACHE_PATH="$STORE/cache/cuda"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY="$ROOT/.venv-mms/bin/python"
ulimit -c 0
