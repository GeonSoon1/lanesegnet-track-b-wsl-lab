#!/usr/bin/env bash
set -euo pipefail
# Dedicated environment is already created; no existing conda environments are changed.
exec bash "$(dirname "$0")/install_local_dependencies.sh"
