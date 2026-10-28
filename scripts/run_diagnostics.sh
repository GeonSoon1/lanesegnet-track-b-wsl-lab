#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
mountpoint -q /mnt/d || { echo 'D: is not mounted'; exit 2; }
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/src:$root/engine"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/mnt/d/mms_experiments/gg49-lanesegnet-lab_sj/cache/tmp
export MPLCONFIGDIR=/mnt/d/mms_experiments/gg49-lanesegnet-lab_sj/cache/matplotlib
cd "$root"
exec /home/rjstns23/gg49-lanesegnet-lab_sj/.venv-mms/bin/python -u -m gg49_lab.diagnostics
