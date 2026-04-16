#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
JOBS="${JOBS:-16}"
CRITICALITY_PROFILE="${CRITICALITY_PROFILE:-default}"
SKIP_EXISTING=0
DRY_RUN=0
INCLUDE_HBM_ORIGINAL=1

LOGICAL_CORES="$(nproc 2>/dev/null || echo 1)"
PHYSICAL_CORES="$(
  lscpu 2>/dev/null | awk -F: '
    /Core\(s\) per socket/ {gsub(/ /, "", $2); cores=$2}
    /Socket\(s\)/ {gsub(/ /, "", $2); sockets=$2}
    END {
      if (cores != "" && sockets != "") {
        print cores * sockets
      }
    }
  '
)"
if [[ -z "$PHYSICAL_CORES" ]]; then
  PHYSICAL_CORES="$LOGICAL_CORES"
fi

declare -a DEFAULT_DESIGNS=(design_1_p5 design_2_p10 HBM_A HBM_B)
declare -a CONFIG_SUFFIXES=("" "_overlay_pessimistic" "_particle_pessimistic" "_mechanical_pessimistic" "_ESD_pessimistic")

usage() {
  cat <<EOF
Usage:
  ./run_design_1_p5_6_hbm_parallel.sh [--jobs N] [--criticality-profile PROFILE] [--skip-existing] [--dry-run] [DESIGN ...]

Defaults:
  designs: ${DEFAULT_DESIGNS[*]}
  jobs: ${JOBS}
  criticality profile: ${CRITICALITY_PROFILE}

Examples:
  ./run_design_1_p5_6_hbm_parallel.sh
  ./run_design_1_p5_6_hbm_parallel.sh --jobs 32
  ./run_design_1_p5_6_hbm_parallel.sh --jobs 16 --skip-existing
  ./run_design_1_p5_6_hbm_parallel.sh design_1_p5 design_2_p10
  ./run_design_1_p5_6_hbm_parallel.sh HBM_A HBM_B

Notes:
  - This script runs simulation only.
  - Default HBM variants are: Original, Center_IO, Edge_IO, Random_IO.
  - The table's "Random_1" corresponds to the actual folder name "Random_IO".
  - Jobs sharing the same ds_dir run serially; only different ds_dir values run in parallel.
  - Each experiment is forced to single-threaded execution via OMP/BLAS/NUMBA env vars.
  - Output logs are written under output/<ds_name>/parallel_simulation__<config_stem>__<profile>.log
  - Detected logical cores: ${LOGICAL_CORES}
  - Detected physical cores: ${PHYSICAL_CORES}
EOF
}

normalize_design_name() {
  local raw_name="$1"
  if [[ "$raw_name" == design_* || "$raw_name" == HBM_* ]]; then
    printf '%s\n' "$raw_name"
  elif [[ "$raw_name" =~ ^[0-9]+$ ]]; then
    printf 'design_%s\n' "$raw_name"
  else
    printf '%s\n' "$raw_name"
  fi
}

config_paths_for_design() {
  local design_name="$1"
  local suffix candidate
  for suffix in "${CONFIG_SUFFIXES[@]}"; do
    candidate="configs/${design_name}/${design_name}${suffix}.yaml"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
    fi
  done
}

list_experiments_for_design() {
  local design_name="$1"
  local design_root="input/${design_name}"
  local ratio_dir ratio_name cfg variant
  local -a variants=()
  local -a configs=()

  if [[ ! -d "$design_root" ]]; then
    return 0
  fi

  mapfile -t configs < <(config_paths_for_design "$design_name")
  if [[ ${#configs[@]} -eq 0 ]]; then
    return 0
  fi

  if [[ "$design_name" == HBM_* ]]; then
    variants=(Center_IO Edge_IO Random_IO)
    if [[ $INCLUDE_HBM_ORIGINAL -eq 1 ]]; then
      variants=(Original "${variants[@]}")
    fi
    for cfg in "${configs[@]}"; do
      for variant in "${variants[@]}"; do
        if [[ -d "${design_root}/${variant}" ]]; then
          printf '%s\t%s\t%s\n' "$cfg" "${design_name}/${variant}" "${design_root}/${variant}"
        fi
      done
    done
    return 0
  fi

  while IFS= read -r ratio_dir; do
    ratio_name="$(basename "$ratio_dir")"
    for cfg in "${configs[@]}"; do
      for variant in Center_IO Edge_IO Random_IO; do
        if [[ -d "${ratio_dir}/${variant}" ]]; then
          printf '%s\t%s\t%s\n' "$cfg" "${design_name}/${ratio_name}/${variant}" "${ratio_dir}/${variant}"
        fi
      done
    done
  done < <(find "$design_root" -mindepth 1 -maxdepth 1 -type d -name 'c*_r*_pg*_dm*' | sort)
}

declare -a pids=()
declare -A pid_desc=()
declare -A pid_log=()
declare -A pid_group=()
declare -A active_group_pid=()
done_count=0
fail_count=0
skip_count=0

cleanup_children() {
  if [[ ${#pids[@]} -eq 0 ]]; then
    return 0
  fi
  kill "${pids[@]}" 2>/dev/null || true
  wait "${pids[@]}" 2>/dev/null || true
}

trap 'echo; echo "Interrupt received. Stopping launched simulations..."; cleanup_children; exit 130' INT TERM

reap_finished() {
  local pid status group_pid
  local -a alive=()
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
      continue
    fi

    if wait "$pid"; then
      status=0
    else
      status=$?
    fi

    if [[ $status -eq 0 ]]; then
      echo "[DONE] ${pid_desc[$pid]}"
      done_count=$((done_count + 1))
    else
      echo "[FAIL] ${pid_desc[$pid]}"
      echo "       log: ${pid_log[$pid]}"
      fail_count=$((fail_count + 1))
    fi

    unset 'pid_desc[$pid]'
    unset 'pid_log[$pid]'
    group_pid="${pid_group[$pid]:-}"
    if [[ -n "$group_pid" ]]; then
      unset 'pid_group[$pid]'
      unset 'active_group_pid[$group_pid]'
    fi
  done
  pids=("${alive[@]}")
}

launch_experiment() {
  local cfg="$1"
  local ds_name="$2"
  local ds_dir="$3"
  local config_stem output_dir summary_path log_path desc pid

  config_stem="$(basename "$cfg" .yaml)"
  output_dir="output/${ds_name}"
  summary_path="${output_dir}/assembly_yield_summary__${config_stem}__${CRITICALITY_PROFILE}.txt"
  log_path="${output_dir}/parallel_simulation__${config_stem}__${CRITICALITY_PROFILE}.log"
  desc="${ds_name} :: ${config_stem}"

  mkdir -p "$output_dir"

  if [[ $SKIP_EXISTING -eq 1 && -f "$summary_path" ]]; then
    echo "[SKIP] ${desc}"
    skip_count=$((skip_count + 1))
    return 0
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[PLAN] ${desc}"
    return 0
  fi

  (
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export BLIS_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1
    exec "$PYTHON_BIN" simulator_main.py \
      --config "$cfg" \
      --mode d2w_simulation \
      --ds_name "$ds_name" \
      --ds_dir "$ds_dir" \
      --criticality-profile "$CRITICALITY_PROFILE" \
      --verbose >"$log_path" 2>&1
  ) &

  pid=$!
  pids+=("$pid")
  pid_desc["$pid"]="$desc"
  pid_log["$pid"]="$log_path"
  pid_group["$pid"]="$ds_dir"
  active_group_pid["$ds_dir"]="$pid"
  echo "[START] ${desc} (pid=${pid})"
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
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --jobs=*)
      JOBS="${1#*=}"
      shift
      ;;
    --criticality-profile)
      CRITICALITY_PROFILE="$2"
      shift 2
      ;;
    --criticality-profile=*)
      CRITICALITY_PROFILE="${1#*=}"
      shift
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "--jobs must be a positive integer." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  raw_design_args=("${DEFAULT_DESIGNS[@]}")
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

echo "Detected logical cores: ${LOGICAL_CORES}"
echo "Detected physical cores: ${PHYSICAL_CORES}"
echo "Using parallel jobs: ${JOBS}"
echo "Criticality profile: ${CRITICALITY_PROFILE}"
echo "Python: ${PYTHON_BIN}"
echo "Designs: ${design_names[*]}"
if [[ $INCLUDE_HBM_ORIGINAL -eq 1 ]]; then
  echo "HBM variants: Original Center_IO Edge_IO Random_IO"
else
  echo "HBM variants: Center_IO Edge_IO Random_IO"
fi

declare -a experiments=()
while IFS= read -r line; do
  experiments+=("$line")
done < <(
  for design_name in "${design_names[@]}"; do
    list_experiments_for_design "$design_name"
  done
)

echo "Total experiments: ${#experiments[@]}"

if [[ $DRY_RUN -eq 1 ]]; then
  for experiment in "${experiments[@]}"; do
    IFS=$'\t' read -r cfg ds_name ds_dir <<< "$experiment"
    launch_experiment "$cfg" "$ds_name" "$ds_dir"
  done
else
  declare -a pending_experiments=("${experiments[@]}")
  declare -a remaining_experiments=()
  launched_any=0

  while [[ ${#pending_experiments[@]} -gt 0 ]]; do
    remaining_experiments=()
    launched_any=0

    for experiment in "${pending_experiments[@]}"; do
      IFS=$'\t' read -r cfg ds_name ds_dir <<< "$experiment"

      if [[ ${#pids[@]} -ge $JOBS || -n "${active_group_pid[$ds_dir]:-}" ]]; then
        remaining_experiments+=("$experiment")
        continue
      fi

      launch_experiment "$cfg" "$ds_name" "$ds_dir"
      launched_any=1
    done

    pending_experiments=("${remaining_experiments[@]}")

    if [[ ${#pending_experiments[@]} -gt 0 && $launched_any -eq 0 ]]; then
      sleep 1
      reap_finished
    fi

    while [[ ${#pids[@]} -ge $JOBS ]]; do
      sleep 1
      reap_finished
    done
  done
fi

while [[ ${#pids[@]} -gt 0 ]]; do
  sleep 1
  reap_finished
done

echo "============================================================"
echo "Parallel simulation sweep finished."
echo "Completed: ${done_count}"
echo "Skipped:   ${skip_count}"
echo "Failed:    ${fail_count}"
echo "============================================================"

if [[ $fail_count -gt 0 ]]; then
  exit 1
fi
