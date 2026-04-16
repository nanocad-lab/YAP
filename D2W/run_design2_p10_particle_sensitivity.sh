#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RATIO="${RATIO:-c10_r0_pg60_dm30}"
CRITICALITY_PROFILE="${CRITICALITY_PROFILE:-default}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NUM_DIE_STACKS_VALUE="${NUM_DIE_STACKS_VALUE:-10000}"

D0_VALUE="${D0_VALUE:-2e-9}"
D1_LIST_CSV="${D1_LIST_CSV:-2e-9,2e-8,4e-8}"
WIDTH_LIST_CSV="${WIDTH_LIST_CSV:-100,300}"
VARIANTS_CSV="${VARIANTS_CSV:-Center_IO,Edge_IO}"

IFS=',' read -r -a D1_LIST <<< "$D1_LIST_CSV"
IFS=',' read -r -a WIDTH_LIST <<< "$WIDTH_LIST_CSV"
IFS=',' read -r -a VARIANTS <<< "$VARIANTS_CSV"

BASE_CONFIG="configs/design_2_p10/design_2_p10.yaml"
OUT_TSV="${OUT_TSV:-output/particle_sensitivity_design_2_p10.tsv}"

mkdir -p "$(dirname "$OUT_TSV")"
printf "design\tratio\tvariant\tD0\tD1\tedge_width_um\tstack_assembly_yield\n" >"$OUT_TSV"

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
  printf "%s\n" "$cfg"
}

for d1 in "${D1_LIST[@]}"; do
  for width in "${WIDTH_LIST[@]}"; do
    cfg_path="$(make_config "$D0_VALUE" "$d1" "$width")"
    config_stem="$(basename "$cfg_path" .yaml)"

    for variant in "${VARIANTS[@]}"; do
      ds_name="design_2_p10/${RATIO}/${variant}"
      ds_dir="input/design_2_p10/${RATIO}/${variant}"
      summary_path="output/${ds_name}/assembly_yield_summary__${config_stem}__${CRITICALITY_PROFILE}.txt"

      if [[ "$SKIP_EXISTING" -eq 1 && -f "$summary_path" ]]; then
        yield_value="$(awk -F': ' '/^stack_assembly_yield:/ {print $2}' "$summary_path" | head -n1)"
      else
        "$PYTHON_BIN" simulator_main.py \
          --config "$cfg_path" \
          --mode d2w_simulation \
          --ds_name "$ds_name" \
          --ds_dir "$ds_dir" \
          --criticality-profile "$CRITICALITY_PROFILE" \
          --verbose
        yield_value="$(awk -F': ' '/^stack_assembly_yield:/ {print $2}' "$summary_path" | head -n1)"
      fi

      printf "design_2_p10\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$RATIO" "$variant" "$D0_VALUE" "$d1" "$width" "${yield_value:-}" >>"$OUT_TSV"
    done
  done
done

echo "Wrote results to $OUT_TSV"
