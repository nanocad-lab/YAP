#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="d2w_simulation"
VERBOSE_FLAG=""
RATIO_NAME=""

usage() {
  cat <<'EOF'
Usage:
  ./run_design_simulations.sh [--verbose] [--ratio RATIO_NAME] [DESIGN_ID_OR_NAME ...]
  ./run_design_simulations.sh [--verbose] [--ratio RATIO_NAME] [DESIGN_1,DESIGN_2,...]

Examples:
  ./run_design_simulations.sh --ratio c25_r0_pg50_dm25 design_1
  ./run_design_simulations.sh --ratio c20_r10_pg50_dm20 --verbose design_17
  ./run_design_simulations.sh --ratio c25_r0_pg50_dm25 3
  ./run_design_simulations.sh --ratio c25_r0_pg50_dm25 design_1 design_2 design_3
  ./run_design_simulations.sh --ratio c25_r0_pg50_dm25 --verbose 1,2,3
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

resolve_design_root() {
  local design_name="$1"
  local ratio_name="$2"
  local legacy_root="input/${design_name}"
  local candidate_root
  local -a ratio_dirs=()

  if [[ ! -d "$legacy_root" ]]; then
    return 1
  fi

  if [[ -n "$ratio_name" ]]; then
    candidate_root="${legacy_root}/${ratio_name}"
    if [[ -d "$candidate_root" ]]; then
      printf '%s\n%s\n' "$candidate_root" "${design_name}/${ratio_name}"
      return 0
    fi
    return 1
  fi

  if [[ -d "${legacy_root}/Center_IO" ]]; then
    printf '%s\n%s\n' "$legacy_root" "${design_name}"
    return 0
  fi

  while IFS= read -r ratio_dir; do
    ratio_dirs+=("$ratio_dir")
  done < <(find "$legacy_root" -mindepth 1 -maxdepth 1 -type d -name 'c*_r*_pg*_dm*' | sort)

  if [[ ${#ratio_dirs[@]} -eq 1 ]]; then
    candidate_root="${ratio_dirs[0]}"
    printf '%s\n%s\n' "$candidate_root" "${design_name}/$(basename "$candidate_root")"
    return 0
  fi

  return 1
}

run_one_design() {
  local design_name="$1"
  local config design_root ds_prefix resolved_root_output
  local -a variants=()
  local status=0

  if ! config="$(resolve_config_path "$design_name")"; then
    echo "Config file not found for ${design_name}." >&2
    return 1
  fi

  if ! resolved_root_output="$(resolve_design_root "$design_name" "$RATIO_NAME")"; then
    if [[ -n "$RATIO_NAME" ]]; then
      echo "Design ratio directory not found: input/${design_name}/${RATIO_NAME}" >&2
    else
      echo "Could not resolve design input root for ${design_name}. Use --ratio when multiple ratio folders exist." >&2
    fi
    return 1
  fi
  design_root="$(printf '%s\n' "$resolved_root_output" | sed -n '1p')"
  ds_prefix="$(printf '%s\n' "$resolved_root_output" | sed -n '2p')"

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
    echo "Running simulation for ${ds_prefix}/${variant}"
    echo "============================================================"
    if ! "$PYTHON_BIN" simulator_main.py \
      --config "$config" \
      --mode "$MODE" \
      --ds_name "${ds_prefix}/${variant}" \
      --ds_dir "${design_root}/${variant}" \
      ${VERBOSE_FLAG:+$VERBOSE_FLAG}; then
      echo "Failed: ${ds_prefix}/${variant}" >&2
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
    --ratio)
      if [[ $# -lt 2 ]]; then
        echo "--ratio requires a value." >&2
        exit 1
      fi
      RATIO_NAME="$2"
      shift 2
      ;;
    --ratio=*)
      RATIO_NAME="${1#*=}"
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
