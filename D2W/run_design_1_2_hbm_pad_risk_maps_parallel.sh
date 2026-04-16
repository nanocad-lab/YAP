#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec "${SCRIPT_DIR}/run_all_pad_risk_maps_parallel.sh" "$@" design_1 design_2 HBM_A HBM_B
