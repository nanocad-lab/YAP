#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RATIO="${RATIO:-c10_r0_pg60_dm30}"
CRITICALITY_PROFILE="${CRITICALITY_PROFILE:-default}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NUM_DIE_STACKS_VALUE="${NUM_DIE_STACKS_VALUE:-10000}"

VOLTAGE_LIST=("5" "30")
TILT_STD_LIST=("1e-5" "1e-4")
VARIANTS=("Center_IO" "Edge_IO")

BASE_CONFIG="configs/design_1_p5/design_1_p5.yaml"
OUT_TSV="${OUT_TSV:-output/esd_sensitivity_check/esd_sensitivity_design_1_p5.tsv}"

mkdir -p "$(dirname "$OUT_TSV")"
printf "design\tratio\tvariant\tvmax_v\ttilt_std_deg\tstack_assembly_yield\tcompute_yield\tmemory_yield\n" >"$OUT_TSV"

make_config() {
  local vmax="$1"
  local tilt_std="$2"
  local suffix="ESD_V${vmax}V_TILT${tilt_std}"
  local cfg="configs/design_1_p5/design_1_p5__${suffix}.yaml"
  cp "$BASE_CONFIG" "$cfg"
  "$PYTHON_BIN" - <<PY
import re
from pathlib import Path

path = Path("$cfg")
text = path.read_text(encoding="utf-8")

def replace_line(src, key, value, comment):
    pattern = re.compile(rf"^(\\s*{re.escape(key)}:\\s*).*$", re.M)
    return pattern.sub(lambda m: f"{m.group(1)}{value}  {comment}", src)

text = replace_line(text, "D0", "1e-11", "# Particle density (1/µm²)")
text = replace_line(text, "D1", "1e-11", "# Edge peak particle density (1/µm²)")
text = replace_line(text, "V_MIN_V", "0.0", "# Minimum ESD voltage (V)")
text = replace_line(text, "V_MAX_V", "$vmax", "# Maximum ESD voltage (V)")
text = replace_line(text, "TILT_X_STD_DEG", "$tilt_std", "# X tilt std (deg)")
text = replace_line(text, "TILT_Y_STD_DEG", "$tilt_std", "# Y tilt std (deg)")
text = replace_line(text, "NUM_DIE_STACKS", "$NUM_DIE_STACKS_VALUE", "# Number of die stacks used in simulation")

path.write_text(text, encoding="utf-8")
PY
  printf "%s\n" "$cfg"
}

for vmax in "${VOLTAGE_LIST[@]}"; do
  for tilt_std in "${TILT_STD_LIST[@]}"; do
    cfg_path="$(make_config "$vmax" "$tilt_std")"
    config_stem="$(basename "$cfg_path" .yaml)"
    for variant in "${VARIANTS[@]}"; do
      ds_name="design_1_p5/${RATIO}/${variant}"
      ds_dir="input/design_1_p5/${RATIO}/${variant}"
      summary_path="output/${ds_name}/assembly_yield_summary__${config_stem}__${CRITICALITY_PROFILE}.txt"

      if [[ "$SKIP_EXISTING" -ne 1 || ! -f "$summary_path" ]]; then
        "$PYTHON_BIN" simulator_main.py \
          --config "$cfg_path" \
          --mode d2w_simulation \
          --ds_name "$ds_name" \
          --ds_dir "$ds_dir" \
          --criticality-profile "$CRITICALITY_PROFILE" \
          --verbose
      fi

      stack_yield="$(awk -F': ' '/^stack_assembly_yield:/ {print $2}' "$summary_path" | head -n1)"
      compute_yield="$(awk -F': ' '/^Compute_Small_From_Substrate_Silicon:/ {print $2}' "$summary_path" | head -n1)"
      memory_yield="$(awk -F': ' '/^Memory_DRAM_From_Substrate_Silicon:/ {print $2}' "$summary_path" | head -n1)"

      printf "design_1_p5\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$RATIO" "$variant" "$vmax" "$tilt_std" "${stack_yield:-}" "${compute_yield:-}" "${memory_yield:-}" >>"$OUT_TSV"
    done
  done
done

echo "Wrote results to $OUT_TSV"
