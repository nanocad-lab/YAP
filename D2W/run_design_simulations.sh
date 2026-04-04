#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="d2w_simulation"
VERBOSE_FLAG=""

usage() {
  cat <<'EOF'
Usage:
  ./run_design_simulations.sh [--verbose] [DESIGN_ID_OR_NAME ...]
  ./run_design_simulations.sh [--verbose] [DESIGN_1,DESIGN_2,...]

Examples:
  ./run_design_simulations.sh
  ./run_design_simulations.sh --verbose
  ./run_design_simulations.sh 3
  ./run_design_simulations.sh design_17
  ./run_design_simulations.sh design_1 design_2 design_3
  ./run_design_simulations.sh 1,2,3
EOF
}

normalize_design_name() {
  local raw_name="$1"
  if [[ "$raw_name" == design_* ]]; then
    printf '%s\n' "$raw_name"
  else
    printf 'design_%s\n' "$raw_name"
  fi
}

resolve_config_path() {
  local design_name="$1"
  local nested_config="configs/${design_name}/${design_name}.yaml"
  local flat_config="configs/${design_name}.yaml"

  if [[ -f "$nested_config" ]]; then
    printf '%s\n' "$nested_config"
  elif [[ -f "$flat_config" ]]; then
    printf '%s\n' "$flat_config"
  else
    return 1
  fi
}

run_one_design() {
  local design_name="$1"
  local config design_root
  local -a variants=()
  local status=0

  if ! config="$(resolve_config_path "$design_name")"; then
    echo "Config file not found for ${design_name}." >&2
    return 1
  fi

  design_root="input/${design_name}"
  if [[ ! -d "$design_root" ]]; then
    echo "Design directory not found: ${design_root}" >&2
    return 1
  fi

  for variant in Center_IO Edge_IO Random_1 Random_2 Random_3; do
    if [[ -d "${design_root}/${variant}" ]]; then
      variants+=("$variant")
    fi
  done

  if [[ ${#variants[@]} -eq 0 ]]; then
    echo "No design variants found under ${design_root}" >&2
    return 1
  fi

  for variant in "${variants[@]}"; do
    echo "============================================================"
    echo "Running simulation for ${design_name}/${variant}"
    echo "============================================================"
    if ! "$PYTHON_BIN" simulator_main.py \
      --config "$config" \
      --mode "$MODE" \
      --ds_name "${design_name}/${variant}" \
      --ds_dir "${design_root}/${variant}" \
      ${VERBOSE_FLAG:+$VERBOSE_FLAG}; then
      echo "Failed: ${design_name}/${variant}" >&2
      status=1
    fi
  done

  return "$status"
}

declare -a raw_design_args=()
declare -a design_names=()
declare -A seen_designs=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --verbose|-v)
      VERBOSE_FLAG="--verbose"
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  raw_design_args=("design_1")
else
  raw_design_args=("$@")
fi

for raw_arg in "${raw_design_args[@]}"; do
  IFS=',' read -r -a split_designs <<< "$raw_arg"
  for split_design in "${split_designs[@]}"; do
    if [[ -z "$split_design" ]]; then
      continue
    fi
    design_name="$(normalize_design_name "$split_design")"
    if [[ -z "${seen_designs[$design_name]:-}" ]]; then
      design_names+=("$design_name")
      seen_designs["$design_name"]=1
    fi
  done
done

overall_status=0
for design_name in "${design_names[@]}"; do
  if ! run_one_design "$design_name"; then
    overall_status=1
  fi
done

exit "$overall_status"
