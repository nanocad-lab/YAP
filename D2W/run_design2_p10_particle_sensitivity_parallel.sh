#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
JOBS="${JOBS:-8}"
CRITICALITY_PROFILE="${CRITICALITY_PROFILE:-default}"
SKIP_EXISTING=0
DRY_RUN=0

RATIO="${RATIO:-c10_r0_pg60_dm30}"
D0_VALUE="${D0_VALUE:-2e-9}"
D1_LIST_CSV="${D1_LIST_CSV:-2e-9,2e-8,4e-8}"
WIDTH_LIST_CSV="${WIDTH_LIST_CSV:-100,300}"
VARIANTS_CSV="${VARIANTS_CSV:-Center_IO,Edge_IO,Random_IO}"
NUM_DIE_STACKS_VALUE="${NUM_DIE_STACKS_VALUE:-10000}"

BASE_CONFIG="configs/design_2_p10/design_2_p10.yaml"
OUT_TSV="${OUT_TSV:-output/particle_sensitivity_design_2_p10.tsv}"

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

usage() {
  cat <<EOF
Usage:
  ./run_design2_p10_particle_sensitivity_parallel.sh [--jobs N] [--criticality-profile PROFILE] [--skip-existing] [--dry-run]

Defaults:
  ratio: ${RATIO}
  D0: ${D0_VALUE}
  D1 list: ${D1_LIST_CSV}
  width list (um): ${WIDTH_LIST_CSV}
  variants: ${VARIANTS_CSV}
  jobs: ${JOBS}
  criticality profile: ${CRITICALITY_PROFILE}
  output TSV: ${OUT_TSV}

Examples:
  ./run_design2_p10_particle_sensitivity_parallel.sh
  ./run_design2_p10_particle_sensitivity_parallel.sh --jobs 16 --skip-existing
  VARIANTS_CSV=Center_IO,Edge_IO ./run_design2_p10_particle_sensitivity_parallel.sh
  RATIO=c5_r10_pg60_dm25 ./run_design2_p10_particle_sensitivity_parallel.sh

Notes:
  - Jobs sharing the same ds_dir run serially; only different ds_dir values run in parallel.
  - Each experiment is forced to single-threaded execution via OMP/BLAS/NUMBA env vars.
  - Per-case logs are written under output/<ds_name>/particle_sensitivity_parallel__<config_stem>__<profile>.log
EOF
}

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
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "--jobs must be a positive integer." >&2
  exit 1
fi

IFS=',' read -r -a D1_LIST <<< "$D1_LIST_CSV"
IFS=',' read -r -a WIDTH_LIST <<< "$WIDTH_LIST_CSV"
IFS=',' read -r -a VARIANTS <<< "$VARIANTS_CSV"

mkdir -p "$(dirname "$OUT_TSV")"

make_config() {
  local d0="$1"
  local d1="$2"
  local width="$3"
  local suffix="D0_${d0}_D1_${d1}_W${width}um"
  local cfg="configs/design_2_p10/design_2_p10__${suffix}.yaml"

  cp "$BASE_CONFIG" "$cfg"
  "$PYTHON_BIN" - <<PY
import re
from pathlib import Path

path = Path("$cfg")
text = path.read_text(encoding="utf-8")

replacements = {
    "D0": ("$d0", "# Particle density (1/\u00b5m\u00b2)"),
    "D1": ("$d1", "# Edge peak particle density (1/\u00b5m\u00b2)"),
    "EDGE_REGION_WIDTH_um": ("$width", "# Width of the edge region for elevated particle density (\u00b5m)"),
    "NUM_DIE_STACKS": ("$NUM_DIE_STACKS_VALUE", "# Number of die stacks used in simulation"),
}

for key, (value, comment) in replacements.items():
    pattern = re.compile(rf"^(\\s*{re.escape(key)}:\\s*).*$", re.M)
    text = pattern.sub(lambda m: f"{m.group(1)}{value}            {comment}", text)

path.write_text(text, encoding="utf-8")
PY

  printf '%s\n' "$cfg"
}

declare -a experiments=()
for d1 in "${D1_LIST[@]}"; do
  for width in "${WIDTH_LIST[@]}"; do
    cfg_path="$(make_config "$D0_VALUE" "$d1" "$width")"
    config_stem="$(basename "$cfg_path" .yaml)"

    for variant in "${VARIANTS[@]}"; do
      ds_name="design_2_p10/${RATIO}/${variant}"
      ds_dir="input/design_2_p10/${RATIO}/${variant}"
      summary_path="output/${ds_name}/assembly_yield_summary__${config_stem}__${CRITICALITY_PROFILE}.txt"
      log_path="output/${ds_name}/particle_sensitivity_parallel__${config_stem}__${CRITICALITY_PROFILE}.log"
      experiments+=("${cfg_path}"$'\t'"${config_stem}"$'\t'"${ds_name}"$'\t'"${ds_dir}"$'\t'"${d1}"$'\t'"${width}"$'\t'"${summary_path}"$'\t'"${log_path}")
    done
  done
done

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
  local config_stem="$2"
  local ds_name="$3"
  local ds_dir="$4"
  local summary_path="$5"
  local log_path="$6"
  local desc pid

  desc="${ds_name} :: ${config_stem}"
  mkdir -p "$(dirname "$summary_path")"

  if [[ "$SKIP_EXISTING" -eq 1 && -f "$summary_path" ]]; then
    echo "[SKIP] ${desc}"
    skip_count=$((skip_count + 1))
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
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

echo "Detected logical cores: ${LOGICAL_CORES}"
echo "Detected physical cores: ${PHYSICAL_CORES}"
echo "Using parallel jobs: ${JOBS}"
echo "Criticality profile: ${CRITICALITY_PROFILE}"
echo "Ratio: ${RATIO}"
echo "D0: ${D0_VALUE}"
echo "D1 list: ${D1_LIST_CSV}"
echo "Width list (um): ${WIDTH_LIST_CSV}"
echo "Variants: ${VARIANTS_CSV}"
echo "Output TSV: ${OUT_TSV}"
echo "Total experiments: ${#experiments[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  for experiment in "${experiments[@]}"; do
    IFS=$'\t' read -r cfg_path config_stem ds_name ds_dir d1 width summary_path log_path <<< "$experiment"
    launch_experiment "$cfg_path" "$config_stem" "$ds_name" "$ds_dir" "$summary_path" "$log_path"
  done
else
  declare -a pending_experiments=("${experiments[@]}")
  declare -a remaining_experiments=()
  launched_any=0

  while [[ ${#pending_experiments[@]} -gt 0 ]]; do
    remaining_experiments=()
    launched_any=0

    for experiment in "${pending_experiments[@]}"; do
      IFS=$'\t' read -r cfg_path config_stem ds_name ds_dir d1 width summary_path log_path <<< "$experiment"

      if [[ ${#pids[@]} -ge $JOBS || -n "${active_group_pid[$ds_dir]:-}" ]]; then
        remaining_experiments+=("$experiment")
        continue
      fi

      launch_experiment "$cfg_path" "$config_stem" "$ds_name" "$ds_dir" "$summary_path" "$log_path"
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

printf "design\tratio\tvariant\tD0\tD1\tedge_width_um\tstack_assembly_yield\n" >"$OUT_TSV"
for experiment in "${experiments[@]}"; do
  IFS=$'\t' read -r cfg_path config_stem ds_name ds_dir d1 width summary_path log_path <<< "$experiment"
  if [[ -f "$summary_path" ]]; then
    yield_value="$(awk -F': ' '/^stack_assembly_yield:/ {print $2}' "$summary_path" | head -n1)"
  else
    yield_value=""
  fi

  printf "design_2_p10\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$RATIO" "$(basename "$ds_dir")" "$D0_VALUE" "$d1" "$width" "${yield_value}" >>"$OUT_TSV"
done

echo "============================================================"
echo "Parallel particle sensitivity sweep finished."
echo "Completed: ${done_count}"
echo "Skipped:   ${skip_count}"
echo "Failed:    ${fail_count}"
echo "Wrote results to $OUT_TSV"
echo "============================================================"

if [[ $fail_count -gt 0 ]]; then
  exit 1
fi
