#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
mountpoint -q /mnt/d || { echo "D: is not mounted in WSL. Connect D: and mount it before retrying." >&2; exit 2; }
export PYTHONPATH="$root/src:$root/engine"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR=/mnt/d/mms_experiments/gg49-lanesegnet-lab_gs/cache/tmp
export MPLCONFIGDIR=/mnt/d/mms_experiments/gg49-lanesegnet-lab_gs/cache/matplotlib
py=/home/rjstns23/gg49-lanesegnet-lab_gs/.venv-mms/bin/python
"$py" - <<'PY'
from gg49_lab.dataset import load_frames
from gg49_lab.common import STORE
frames=load_frames()
for p in [STORE/'checkpoints/lanesegnet_r50_8x1_24e_olv2_subset_A.pth',STORE/'cache/rgb'/(frames[0]['key']+'.npz'),STORE/'cache/lidar'/(frames[0]['key']+'.npz')]:
    if not p.is_file():raise SystemExit('Missing required input: '+str(p))
print('Preflight OK: mounted D:, prepared frames:',len(frames))
PY
if [[ "${1:-}" == --check-only ]]; then exit 0; fi
if [[ $# -gt 0 ]]; then echo 'Usage: bash scripts/run_demo.sh [--check-only]' >&2; exit 2; fi
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
exec 9>"$TMPDIR/recording-demo.lock"
flock -n 9 || { echo 'Another recording demo is running.' >&2; exit 2; }
run="demo-camera-$(date +%Y%m%d-%H%M%S)-$$"
echo "Starting a NEW camera demonstration: 1 epoch; run=$run"
echo "Outputs: /mnt/d/mms_experiments/gg49-lanesegnet-lab_gs/runs/$run"
"$py" -u -m gg49_lab.train train --mode camera --resolution 384 --batch 2 --epochs 1 --run "$run"
