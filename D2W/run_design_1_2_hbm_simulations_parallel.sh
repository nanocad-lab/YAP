#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec "${SCRIPT_DIR}/run_design_1_p5_design_2_p10_hbm_parallel.sh" "$@" design_1 design_2 HBM_A HBM_B
