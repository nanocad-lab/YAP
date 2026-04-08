#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CRITICALITY_PROFILE="${CRITICALITY_PROFILE:-default}"

run_case() {
  local config_path="$1"
  local ds_name="$2"
  local ds_dir="$3"
  local config_stem expected_summary

  config_stem="$(basename "$config_path" .yaml)"
  expected_summary="output/${ds_name}/assembly_yield_summary__${config_stem}__${CRITICALITY_PROFILE}.txt"

  if [[ -f "$expected_summary" ]]; then
    echo "[SKIP] ${ds_name} :: ${config_stem}"
    return 0
  fi

  echo "[RUN ] ${ds_name} :: ${config_stem}"
  "$PYTHON_BIN" simulator_main.py \
    --config "$config_path" \
    --mode d2w_simulation \
    --ds_name "$ds_name" \
    --ds_dir "$ds_dir" \
    --criticality-profile "$CRITICALITY_PROFILE" \
    --verbose
}

failures=0

run_or_record_failure() {
  if ! run_case "$1" "$2" "$3"; then
    echo "[FAIL] $2 :: $(basename "$1" .yaml)" >&2
    failures=$((failures + 1))
  fi
}

echo "Starting full simulation sweep at $(date)"
echo "Using python: $PYTHON_BIN"
echo "Criticality profile: $CRITICALITY_PROFILE"

for design in design_1 design_2; do
  for config_path in \
    "configs/${design}/${design}.yaml" \
    "configs/${design}/${design}_overlay_pessimistic.yaml" \
    "configs/${design}/${design}_particle_pessimistic.yaml" \
    "configs/${design}/${design}_mechanical_pessimistic.yaml" \
    "configs/${design}/${design}_ESD_pessimistic.yaml"
  do
    for ratio_dir in input/"${design}"/c*_r*_pg*_dm*; do
      [[ -d "$ratio_dir" ]] || continue
      ratio_name="$(basename "$ratio_dir")"
      for variant in Center_IO Edge_IO Random_IO; do
        [[ -d "${ratio_dir}/${variant}" ]] || continue
        run_or_record_failure \
          "$config_path" \
          "${design}/${ratio_name}/${variant}" \
          "${ratio_dir}/${variant}"
      done
    done
  done
done

for design in HBM_A HBM_B; do
  for config_path in \
    "configs/${design}/${design}.yaml" \
    "configs/${design}/${design}_overlay_pessimistic.yaml" \
    "configs/${design}/${design}_particle_pessimistic.yaml" \
    "configs/${design}/${design}_mechanical_pessimistic.yaml" \
    "configs/${design}/${design}_ESD_pessimistic.yaml"
  do
    for variant in Original Center_IO Edge_IO Random_IO; do
      [[ -d "input/${design}/${variant}" ]] || continue
      run_or_record_failure \
        "$config_path" \
        "${design}/${variant}" \
        "input/${design}/${variant}"
    done
  done
done

echo "Finished full simulation sweep at $(date)"
echo "Failure count: $failures"
exit "$failures"
