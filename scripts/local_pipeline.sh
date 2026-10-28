#!/usr/bin/env bash
source "$(dirname "$0")/local_env.sh"
exec 9>"$STORE/runs/training.lock"
flock -n 9 || { echo 'Training already holds the GPU lock'; exit 2; }
exec "$PY" -u -m gg49_lab.supervisor
