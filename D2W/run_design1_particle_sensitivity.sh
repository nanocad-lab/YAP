#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RATIO="${RATIO:-c25_r0_pg50_dm25}"
CRITICALITY_PROFILE="${CRITICALITY_PROFILE:-default}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NUM_DIE_STACKS_VALUE="${NUM_DIE_STACKS_VALUE:-100000}"

D0_LIST_CSV="${D0_LIST_CSV:-1e-9,1e-8}"
D0_LIST=()
IFS=',' read -r -a D0_LIST <<< "$D0_LIST_CSV"
D1_LIST=("1e-9" "2e-9" "1e-8")
WIDTH_LIST=("300" "100")
VARIANTS=("Center_IO" "Edge_IO")

BASE_CONFIG="configs/design_1/design_1.yaml"
OUT_TSV="${OUT_TSV:-output/particle_sensitivity_design_1.tsv}"

mkdir -p "$(dirname "$OUT_TSV")"
if [[ ! -f "$OUT_TSV" ]]; then
  printf "design\tratio\tvariant\tD0\tD1\tedge_width_um\tstack_assembly_yield\n" >"$OUT_TSV"
fi

make_config() {
  local d0="$1"
  local d1="$2"
  local width="$3"
  local suffix="D0_${d0}_D1_${d1}_W${width}um"
  local cfg="configs/design_1/design_1__${suffix}.yaml"
  cp "$BASE_CONFIG" "$cfg"
  python - <<PY
import re
from pathlib import Path

path = Path("$cfg")
text = path.read_text(encoding="utf-8")

def replace_line(key, value, comment):
    pattern = re.compile(rf"^(\\s*{re.escape(key)}:\\s*).*$", re.M)
    return pattern.sub(lambda m: f"{m.group(1)}{value}  {comment}", text)

text = replace_line("D0", "$d0", "# Particle density (1/µm²)")
text = replace_line("D1", "$d1", "# Edge peak particle density (1/µm²)")
text = replace_line("EDGE_REGION_WIDTH_um", f"{float($width):.1f}", "# Width of the edge region for elevated particle density (µm)")
text = replace_line("NUM_DIE_STACKS", "$NUM_DIE_STACKS_VALUE", "# Number of die stacks used in simulation")

path.write_text(text, encoding="utf-8")
PY
  printf "%s\n" "$cfg"
}

for d0 in "${D0_LIST[@]}"; do
  for d1 in "${D1_LIST[@]}"; do
    for width in "${WIDTH_LIST[@]}"; do
      cfg_path="$(make_config "$d0" "$d1" "$width")"
      config_stem="$(basename "$cfg_path" .yaml)"
      for variant in "${VARIANTS[@]}"; do
        ds_name="design_1/${RATIO}/${variant}"
        ds_dir="input/design_1/${RATIO}/${variant}"
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

        printf "design_1\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$RATIO" "$variant" "$d0" "$d1" "$width" "${yield_value:-}" >>"$OUT_TSV"
      done
    done
  done
done

echo "Wrote results to $OUT_TSV"
