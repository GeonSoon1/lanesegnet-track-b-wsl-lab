#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
run="${1:-track-b-geometry-$(date +%Y%m%d-%H%M%S)}"
[[ "$run" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run name' >&2; exit 2; }
source "$root/scripts/local_env.sh"
work="$STORE/artifacts/$run"
[[ ! -e "$work" ]] || { echo 'Existing GT preserved; select a new run name' >&2; exit 2; }
"$PY" "$root/../gg49-lanesegnet-gt-builder/scripts/build_gt.py" \
  --stage geometry --tracks B --shp-encoding cp949 \
  --shp-root "$STORE/data/shp/통합SHP" --raw-root /mnt/d/2026-APR-01_GG803SEC012 \
  --work "$work"
