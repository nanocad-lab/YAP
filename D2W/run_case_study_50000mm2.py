#!/usr/bin/env python3
"""Run the total-pad-ratio D2W ISO1/ISO3 case study at block resolution."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from Cu_expansion_yield_calculator import Cu_expansion_yield_calculator
from defect_yield_calculator import defect_yield_calculator
from overlay_yield_calculator import overlay_yield_calculator
from pad_bitmap_generation import assign_pad_blocks
from roughness_parameters import roughness_parameters
from utils.util import load_modeling_config, update_config_items


PACKAGE_AREAS_MM2 = (5_000.0, 50_000.0)
CHIPLET_AREAS_MM2 = (50.0, 500.0)
PITCHES_UM = (5.0, 1.0)
PAD_SCENARIOS = (
    {
        "name": "100% critical",
        "critical_pad_ratio": 1.0,
        "redundant_pad_ratio": 0.0,
        "redundant_logical_pad_ratio": 0.0,
    },
    {
        "name": "25% critical",
        "critical_pad_ratio": 0.25,
        "redundant_pad_ratio": 0.0,
        "redundant_logical_pad_ratio": 0.0,
    },
    {
        "name": "20% critical + 10% redundant",
        "critical_pad_ratio": 0.20,
        "redundant_pad_ratio": 0.10,
        # Half of the physical redundant pads are logical signals and half are copies.
        "redundant_logical_pad_ratio": 0.50,
    },
)
D0_CASES = (("ISO3", 1e-9), ("ISO2", 1e-10), ("ISO1", 1e-11))
DISPLAY_ISO_ORDER = ("ISO1", "ISO2", "ISO3")
PAD_LAYOUT_PATTERN = "center"
PAD_BLOCK_DIM_UM = 400.0
REDUNDANT_EMPTY_BLOCK_GAP = 1
REDUNDANT_LOGICAL_PAD_DISTANCE_UM = PAD_BLOCK_DIM_UM * (REDUNDANT_EMPTY_BLOCK_GAP + 1)
RANDOM_SEED = 20260721
SYSTEM_ROTATION_MEAN_RAD = 5e-6
CONTACT_AREA_CONSTRAINT = 0.75
CRITICAL_DIST_CONSTRAINT = 0.75
OUTPUT_FIELDS = (
    ("Config", "pad_scenario"),
    ("A_mm2", "chiplet_area_mm2"),
    ("p_um", "pitch_um"),
    ("ISO", "iso_class"),
    ("S_O", "single_overlay_yield"),
    ("S_P", "single_particle_yield"),
    ("S_C", "single_cu_recess_yield"),
    ("S_Y", "single_overall_yield"),
    ("5k_O", "package_5000_overlay_yield"),
    ("5k_P", "package_5000_particle_yield"),
    ("5k_C", "package_5000_cu_recess_yield"),
    ("5k_Y", "package_5000_overall_yield"),
    ("50k_O", "package_50000_overlay_yield"),
    ("50k_P", "package_50000_particle_yield"),
    ("50k_C", "package_50000_cu_recess_yield"),
    ("50k_Y", "package_50000_overall_yield"),
)
OUTPUT_HEADER_BY_KEY = {key: header for header, key in OUTPUT_FIELDS}


def output_value(row: dict, key: str):
    """Read a value from either an internal row or an abbreviated output row."""
    return row.get(key, row.get(OUTPUT_HEADER_BY_KEY.get(key, key)))


def block_corners(block_mask: np.ndarray, block_um: float, die_side_um: float) -> np.ndarray:
    """Return four centered bounding-box corners for a block mask."""
    rows, cols = np.where(block_mask)
    if len(rows) == 0:
        return np.zeros((4, 2), dtype=float)

    x0 = -die_side_um / 2 + cols.min() * block_um
    x1 = min(-die_side_um / 2 + (cols.max() + 1) * block_um, die_side_um / 2)
    y0 = -die_side_um / 2 + rows.min() * block_um
    y1 = min(-die_side_um / 2 + (rows.max() + 1) * block_um, die_side_um / 2)
    return np.array([[x0, y1], [x1, y1], [x0, y0], [x1, y0]], dtype=float)


def place_redundant_block_pairs(
    critical_ids: np.ndarray,
    block_rows: int,
    block_cols: int,
    target_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Place main/gap/copy triplets nearest the centered critical region."""
    if target_pairs == 0:
        empty = np.array([], dtype=int)
        return empty, empty

    critical_set = set(np.asarray(critical_ids, dtype=int))
    available = set(range(block_rows * block_cols)) - critical_set
    center_row = (block_rows - 1) / 2
    center_col = (block_cols - 1) / 2
    candidates = []

    for row in range(block_rows):
        for col in range(block_cols):
            for row_step, col_step in ((0, 2), (2, 0)):
                copy_row = row + row_step
                copy_col = col + col_step
                gap_row = row + row_step // 2
                gap_col = col + col_step // 2
                if copy_row >= block_rows or copy_col >= block_cols:
                    continue
                main_id = row * block_cols + col
                gap_id = gap_row * block_cols + gap_col
                copy_id = copy_row * block_cols + copy_col
                if {main_id, gap_id, copy_id}.issubset(available):
                    main_radius2 = (row - center_row) ** 2 + (col - center_col) ** 2
                    copy_radius2 = (copy_row - center_row) ** 2 + (copy_col - center_col) ** 2
                    if copy_radius2 < main_radius2:
                        main_id, copy_id = copy_id, main_id
                        main_radius2, copy_radius2 = copy_radius2, main_radius2
                    radius2 = (gap_row - center_row) ** 2 + (gap_col - center_col) ** 2
                    candidates.append((main_radius2, radius2, main_id, gap_id, copy_id))

    reserved = set()
    pairs = []
    for _, _, main_id, gap_id, copy_id in sorted(candidates):
        triplet = {main_id, gap_id, copy_id}
        if reserved.isdisjoint(triplet):
            reserved.update(triplet)
            pairs.append((main_id, copy_id))
            if len(pairs) == target_pairs:
                break

    if len(pairs) != target_pairs:
        raise ValueError("Not enough room for main/gap/copy redundant block triplets.")
    pair_array = np.asarray(pairs, dtype=int)
    return pair_array[:, 0], pair_array[:, 1]


def shared_25pct_layout(block_rows: int, block_cols: int):
    """Return the common 20% core and outer 5% main/copy layout."""
    num_blocks = block_rows * block_cols
    num_20pct = math.ceil(0.20 * num_blocks)
    num_25pct = math.ceil(0.25 * num_blocks)
    num_10pct_physical = min(
        num_blocks - num_20pct,
        math.ceil(0.10 * num_blocks),
    )
    core_ids, _, _ = assign_pad_blocks(
        PAD_LAYOUT_PATTERN,
        num_blocks,
        block_rows,
        block_cols,
        num_20pct,
        0,
        redundant_mesh_spacing=3,
    )
    target_pairs = min(num_10pct_physical // 2, num_25pct - num_20pct)
    main_ids, copy_ids = place_redundant_block_pairs(
        np.asarray(core_ids), block_rows, block_cols, target_pairs
    )
    return np.asarray(core_ids, dtype=int), main_ids, copy_ids, num_10pct_physical


def compact_bitmap_collection(cfg) -> dict:
    """Build the block-level fields consumed by YAP+'s die-level calculators."""
    block_size = cfg.pad_block_size
    block_rows = math.ceil(cfg.PAD_ARR_ROW / block_size)
    block_cols = math.ceil(cfg.PAD_ARR_COL / block_size)
    num_blocks = block_rows * block_cols
    is_25pct = math.isclose(cfg.critical_pad_ratio, 0.25) and math.isclose(
        cfg.redundant_pad_ratio, 0.0
    )
    is_redundant = math.isclose(cfg.critical_pad_ratio, 0.20) and math.isclose(
        cfg.redundant_pad_ratio, 0.10
    )
    if is_25pct or is_redundant:
        core_ids, shared_main_ids, shared_copy_ids, requested_redundant_blocks = (
            shared_25pct_layout(block_rows, block_cols)
        )
        critical_ids = (
            np.concatenate((core_ids, shared_main_ids)) if is_25pct else core_ids
        )
    else:
        num_critical = min(num_blocks, math.ceil(cfg.critical_pad_ratio * num_blocks))
        critical_ids, _, _ = assign_pad_blocks(
            PAD_LAYOUT_PATTERN,
            num_blocks,
            block_rows,
            block_cols,
            num_critical,
            0,
            redundant_mesh_spacing=3,
        )
        critical_ids = np.asarray(critical_ids, dtype=int)
        shared_main_ids = np.array([], dtype=int)
        shared_copy_ids = np.array([], dtype=int)
        requested_redundant_blocks = 0
    num_critical_blocks = len(critical_ids)

    critical_blocks = np.zeros((block_rows, block_cols), dtype=bool)
    critical_blocks[np.asarray(critical_ids) // block_cols, np.asarray(critical_ids) % block_cols] = True

    num_pair_blocks = len(shared_main_ids) if is_redundant else 0
    main_ids = shared_main_ids[:num_pair_blocks]
    copy_ids = shared_copy_ids[:num_pair_blocks]
    pair_distances_um = np.full(num_pair_blocks, REDUNDANT_LOGICAL_PAD_DISTANCE_UM)
    redundant_main = np.zeros((num_pair_blocks, block_rows, block_cols), dtype=bool)
    redundant_copy = np.zeros_like(redundant_main)
    if num_pair_blocks:
        pair_index = np.arange(num_pair_blocks)
        redundant_main[pair_index, main_ids // block_cols, main_ids % block_cols] = True
        redundant_copy[pair_index, copy_ids // block_cols, copy_ids % block_cols] = True

    # Keep electrical pad counts at the exact requested total-pad ratios. The
    # block masks above are only the spatial approximation used by the YAP+
    # particle critical-area model.
    total_pad_sites = cfg.PAD_ARR_ROW * cfg.PAD_ARR_COL
    num_critical_pads = int(cfg.critical_pad_ratio * total_pad_sites)
    num_redundant_pads = min(
        total_pad_sites - num_critical_pads,
        int(cfg.redundant_pad_ratio * total_pad_sites),
    )
    num_redundant_logical_pads = math.ceil(
        num_redundant_pads * cfg.redundant_logical_pad_ratio
    )

    return {
        "CRITICAL_PAD_BLOCK_BITMAP": critical_blocks,
        "REDUNDANT_MAIN_PAD_BLOCK_BITMAP": redundant_main,
        "REDUNDANT_COPY_PAD_BLOCK_BITMAP": redundant_copy,
        "is_redundant_copy_same_block": False,
        "pad_block_size": block_size,
        "num_critical_pads": num_critical_pads,
        "num_redundant_pads": num_redundant_pads,
        "num_redundant_logical_pads": num_redundant_logical_pads,
        "redundant_logical_pad_copy": cfg.redundant_logical_pad_copy,
        "critical_boundary_coords": block_corners(
            np.logical_or(critical_blocks, np.any(redundant_main, axis=0))
            if num_pair_blocks
            else critical_blocks,
            cfg.pad_block_dim,
            cfg.DIE_W_um,
        ),
        "num_blocks": num_blocks,
        "num_critical_blocks": num_critical_blocks,
        "num_redundant_blocks": min(requested_redundant_blocks, 2 * num_pair_blocks),
        "num_pair_blocks": num_pair_blocks,
        "pair_distance_min_um": float(pair_distances_um.min()) if num_pair_blocks else None,
        "pair_distance_max_um": float(pair_distances_um.max()) if num_pair_blocks else None,
    }


def make_die(cfg, bitmap: dict):
    half_w = cfg.PAD_ARR_W_um / 2
    half_l = cfg.PAD_ARR_L_um / 2
    return SimpleNamespace(
        DIE_W_um=cfg.DIE_W_um,
        DIE_L_um=cfg.DIE_L_um,
        ovl_critical_pad_boundary_coords=bitmap["critical_boundary_coords"],
        pad_array_box=np.array(
            [[-half_w, half_l], [half_w, half_l], [-half_w, -half_l], [half_w, -half_l]]
        ),
        pad_array=None,
        glb_pad_yield_min_max_dict={},
    )


def cu_per_pad_probability(cfg) -> float:
    zeta_0 = cfg.k_et * (cfg.T_anl - cfg.T_R) + cfg.k_eb * (cfg.T_anl - cfg.T_R)
    zeta_1 = max(
        roughness_parameters(
            Asperity_R_m=cfg.Asperity_R_m,
            Roughness_sigma_m=cfg.Roughness_sigma_m,
            eta_s=cfg.eta_s,
            Roughness_constant=cfg.Roughness_constant,
            Adhesion_energy=cfg.Adhesion_energy,
            Young_modulus_Pa=cfg.Young_modulus_Pa,
            Dielectric_thickness=cfg.Dielectric_thickness,
            PITCH_um=cfg.PITCH_um,
            PAD_BOT_R_um=cfg.PAD_BOT_R_um,
            DISH_0_m=cfg.DISH_0_m,
            k_peel=cfg.k_peel,
        ),
        0.0,
    )
    mean = cfg.TOP_DISH_MEAN_nm + cfg.BOT_DISH_MEAN_nm
    std = math.sqrt(cfg.TOP_DISH_STD_nm**2 + cfg.BOT_DISH_STD_nm**2)
    probability, _ = quad(lambda x: norm.pdf(x, loc=mean, scale=std), -zeta_0, -zeta_1)
    return probability


def log_cu_yield(per_pad: float, bitmap: dict) -> float:
    critical_term = bitmap["num_critical_pads"] * math.log(per_pad)
    redundant_success = 1.0 - (1.0 - per_pad) ** bitmap["redundant_logical_pad_copy"]
    redundant_term = bitmap["num_redundant_logical_pads"] * math.log(redundant_success)
    return critical_term + redundant_term


def safe_exp(log_value: float) -> float:
    return math.exp(log_value) if log_value > math.log(np.finfo(float).tiny) else 0.0


def scaled_yield(log_single_yield: float, count: int) -> float:
    return safe_exp(count * log_single_yield)


def run_case(cfg, chiplet_area_mm2: float, pitch_um: float, scenario: dict) -> list[dict]:
    die_side_um = math.sqrt(chiplet_area_mm2) * 1_000.0
    cfg.DIE_W_um = die_side_um
    cfg.DIE_L_um = die_side_um
    cfg.PITCH_um = pitch_um
    cfg.critical_pad_ratio = scenario["critical_pad_ratio"]
    cfg.redundant_pad_ratio = scenario["redundant_pad_ratio"]
    cfg.redundant_logical_pad_ratio = scenario["redundant_logical_pad_ratio"]
    cfg.redundant_logical_pad_copy = 2
    cfg.redundant_logical_pad_dist = REDUNDANT_LOGICAL_PAD_DISTANCE_UM / pitch_um
    cfg.pad_block_dim = PAD_BLOCK_DIM_UM
    cfg.pad_yield_flag = False
    cfg.SYSTEM_ROTATION_MEAN_rad = SYSTEM_ROTATION_MEAN_RAD
    cfg.CONTACT_AREA_CONSTRAINT = CONTACT_AREA_CONSTRAINT
    cfg.CRITICAL_DIST_CONSTRAINT = CRITICAL_DIST_CONSTRAINT
    update_config_items(cfg=cfg, mode="d2w_modeling")

    bitmap = compact_bitmap_collection(cfg)
    die = make_die(cfg, bitmap)

    np.random.seed(RANDOM_SEED)
    overlay_yield, _ = overlay_yield_calculator(
        PAD_TOP_R_um=cfg.PAD_TOP_R_um,
        PAD_BOT_R_um=cfg.PAD_BOT_R_um,
        PAD_ARR_ROW=cfg.PAD_ARR_ROW,
        PAD_ARR_COL=cfg.PAD_ARR_COL,
        PITCH_um=cfg.PITCH_um,
        num_samples=cfg.num_samples,
        CONTACT_AREA_CONSTRAINT=cfg.CONTACT_AREA_CONSTRAINT,
        CRITICAL_DIST_CONSTRAINT=cfg.CRITICAL_DIST_CONSTRAINT,
        SYSTEM_MAGNIFICATION_MEAN_ppm=cfg.SYSTEM_MAGNIFICATION_MEAN_ppm,
        SYSTEM_MAGNIFICATION_STD_ppm=cfg.SYSTEM_MAGNIFICATION_STD_ppm,
        SYSTEM_ROTATION_MEAN_rad=cfg.SYSTEM_ROTATION_MEAN_rad,
        SYSTEM_ROTATION_STD_rad=cfg.SYSTEM_ROTATION_STD_rad,
        SYSTEM_TRANSLATION_X_MEAN_um=cfg.SYSTEM_TRANSLATION_X_MEAN_um,
        SYSTEM_TRANSLATION_X_STD_um=cfg.SYSTEM_TRANSLATION_X_STD_um,
        SYSTEM_TRANSLATION_Y_MEAN_um=cfg.SYSTEM_TRANSLATION_Y_MEAN_um,
        SYSTEM_TRANSLATION_Y_STD_um=cfg.SYSTEM_TRANSLATION_Y_STD_um,
        RANDOM_MISALIGNMENT_MEAN_um=cfg.RANDOM_MISALIGNMENT_MEAN_um,
        RANDOM_MISALIGNMENT_STD_um=cfg.RANDOM_MISALIGNMENT_STD_um,
        die=die,
        redundant_flag=True,
        pad_yield_flag=False,
        pad_yield_map_sub_factor=cfg.pad_yield_map_sub_factor,
    )

    cu_yield, _ = Cu_expansion_yield_calculator(
        cfg=cfg,
        die=die,
        TOP_DISH_MEAN_nm=cfg.TOP_DISH_MEAN_nm,
        TOP_DISH_STD_nm=cfg.TOP_DISH_STD_nm,
        BOT_DISH_MEAN_nm=cfg.BOT_DISH_MEAN_nm,
        BOT_DISH_STD_nm=cfg.BOT_DISH_STD_nm,
        k_et=cfg.k_et,
        k_eb=cfg.k_eb,
        T_R=cfg.T_R,
        T_anl=cfg.T_anl,
        pad_bitmap_collection=bitmap,
        pad_yield_flag=False,
    )
    per_pad_cu = cu_per_pad_probability(cfg)
    log_cu = log_cu_yield(per_pad_cu, bitmap)

    rows = []
    for iso_class, d0 in D0_CASES:
        cfg.D0 = d0
        particle_yield, _ = defect_yield_calculator(
            cfg=cfg,
            eff_DIE_R=cfg.eff_DIE_R,
            D0=cfg.D0,
            t_0=cfg.t_0,
            z=cfg.z,
            k_r=cfg.k_r,
            k_r0=cfg.k_r0,
            k_n=cfg.k_n,
            k_S=cfg.k_S,
            k_L=cfg.k_L,
            PAD_TOP_R_um=cfg.PAD_TOP_R_um,
            PITCH_um=cfg.PITCH_um,
            PAD_ARR_ROW=cfg.PAD_ARR_ROW,
            PAD_ARR_COL=cfg.PAD_ARR_COL,
            PAD_ARR_W_um=cfg.PAD_ARR_W_um,
            PAD_ARR_L_um=cfg.PAD_ARR_L_um,
            VOID_SHAPE=cfg.VOID_SHAPE,
            die=die,
            pad_bitmap_collection=bitmap,
            pad_yield_flag=False,
            pad_yield_map_sub_factor=cfg.pad_yield_map_sub_factor,
        )
        log_overlay = math.log(overlay_yield)
        log_particle = math.log(particle_yield)
        log_overall = log_overlay + log_particle + log_cu
        row = {
            "pad_scenario": scenario["name"],
            "chiplet_area_mm2": chiplet_area_mm2,
            "pitch_um": pitch_um,
            "iso_class": iso_class,
            "single_overlay_yield": overlay_yield,
            "single_particle_yield": particle_yield,
            "single_cu_recess_yield": safe_exp(log_cu),
            "single_overall_yield": safe_exp(log_overall),
        }
        for package_area in PACKAGE_AREAS_MM2:
            label = str(int(package_area))
            chiplet_count = int(round(package_area / chiplet_area_mm2))
            row[f"package_{label}_chiplet_count"] = chiplet_count
            row[f"package_{label}_overlay_yield"] = scaled_yield(log_overlay, chiplet_count)
            row[f"package_{label}_particle_yield"] = scaled_yield(log_particle, chiplet_count)
            row[f"package_{label}_cu_recess_yield"] = scaled_yield(log_cu, chiplet_count)
            row[f"package_{label}_overall_yield"] = scaled_yield(log_overall, chiplet_count)
        rows.append(row)
    return rows


def write_results(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    json_rows = [
        {
            header: round(output_value(row, key), 6)
            if key.endswith("_yield")
            else output_value(row, key)
            for header, key in OUTPUT_FIELDS
        }
        for row in rows
    ]
    json_path.write_text(json.dumps(json_rows, indent=2) + "\n", encoding="ascii")

    def write_csv(path: Path, selected_rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="ascii") as stream:
            writer = csv.DictWriter(stream, fieldnames=[header for header, _ in OUTPUT_FIELDS])
            writer.writeheader()
            for row in selected_rows:
                writer.writerow(
                    {
                        header: f"{output_value(row, key):.6f}"
                        if key.endswith("_yield")
                        else output_value(row, key)
                        for header, key in OUTPUT_FIELDS
                    }
                )

    write_csv(csv_path, rows)
    for iso_class in DISPLAY_ISO_ORDER:
        write_csv(
            output_dir / f"results_{iso_class}.csv",
            [row for row in rows if output_value(row, "iso_class") == iso_class],
        )


def format_yield(value: float) -> str:
    return f"{value:.6f}"


def write_summary(rows: list[dict], output_dir: Path, num_samples: int) -> None:
    lines = [
        "# D2W total-pad ISO sweep",
        "",
        "All pad percentages are fractions of total physical pad sites. A/p are chiplet area (mm^2) and pitch (um); S/5k/50k are single-chiplet, 5,000 mm^2, and 50,000 mm^2 scales; O/P/C/Y are overlay, particle, Cu recess, and overall yield. Yields are shown as fractions with six decimal places.",
        "",
    ]
    d0_by_iso = dict(D0_CASES)
    for iso_class in DISPLAY_ISO_ORDER:
        d0 = d0_by_iso[iso_class]
        lines.extend(
            [
                f"## {iso_class}: D0 = {d0:g} /um^2",
                "",
                "| A | p | Config | S: O/P/C/Y | 5k: O/P/C/Y | 50k: O/P/C/Y |",
                "|---:|---:|---|---|---|---|",
            ]
        )
        for row in rows:
            if output_value(row, "iso_class") != iso_class:
                continue
            single = "/".join(
                format_yield(output_value(row, key))
                for key in (
                    "single_overlay_yield",
                    "single_particle_yield",
                    "single_cu_recess_yield",
                    "single_overall_yield",
                )
            )
            package_5000 = "/".join(
                format_yield(output_value(row, key))
                for key in (
                    "package_5000_overlay_yield",
                    "package_5000_particle_yield",
                    "package_5000_cu_recess_yield",
                    "package_5000_overall_yield",
                )
            )
            package_50000 = "/".join(
                format_yield(output_value(row, key))
                for key in (
                    "package_50000_overlay_yield",
                    "package_50000_particle_yield",
                    "package_50000_cu_recess_yield",
                    "package_50000_overall_yield",
                )
            )
            lines.append(
                f"| {output_value(row, 'chiplet_area_mm2'):g} mm^2 | {output_value(row, 'pitch_um'):g} um | "
                f"{output_value(row, 'pad_scenario')} | {single} | {package_5000} | {package_50000} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Assumptions",
            "",
            "- Chiplets are square and exactly tile each package area.",
            "- Package yields compound independent per-chiplet yields.",
            "- The 20% critical + 10% redundant case uses 10% of total physical pad sites for two-copy redundancy, producing 5% redundant logical signals and 25% total logical signals.",
            "- The 25% and 20% + 10% cases use the exact same logical main-block footprint: a centered 20% core plus the same outer 5% main blocks.",
            f"- All critical pad blocks are centered in the die; D2W uses {PAD_BLOCK_DIM_UM:g} um blocks.",
            f"- Every redundant main/copy pair has exactly {REDUNDANT_EMPTY_BLOCK_GAP} empty block between it, so its center distance is {REDUNDANT_LOGICAL_PAD_DISTANCE_UM:g} um.",
            f"- SYSTEM_ROTATION_MEAN_rad is {SYSTEM_ROTATION_MEAN_RAD:g}.",
            f"- CONTACT_AREA_CONSTRAINT and CRITICAL_DIST_CONSTRAINT are both {CONTACT_AREA_CONSTRAINT:g}.",
            f"- Every overlay case resets NumPy seed to {RANDOM_SEED} with num_samples = {num_samples}, so scenario comparisons use identical samples.",
            "- Other process inputs use the d2w_modeling defaults in configs/config.yaml.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    np.random.seed(RANDOM_SEED)
    script_dir = Path(__file__).resolve().parent
    cfg = load_modeling_config(script_dir / "configs/config.yaml", "d2w_modeling")
    rows = []
    with tempfile.TemporaryDirectory(prefix="yap_case_study_") as temp_dir:
        os.chdir(temp_dir)
        Path("pad_bitmap").mkdir()
        for chiplet_area in CHIPLET_AREAS_MM2:
            for pitch in PITCHES_UM:
                for scenario in PAD_SCENARIOS:
                    cache = Path("pad_bitmap/avg_num_defects_per_unit_area.npy")
                    cache.unlink(missing_ok=True)
                    print(
                        f"Running area={chiplet_area:g} mm^2, pitch={pitch:g} um, "
                        f"pads={scenario['name']}"
                    )
                    rows.extend(run_case(cfg, chiplet_area, pitch, scenario))

    output_dir = script_dir / "case_study_total_pad_iso_results"
    write_results(rows, output_dir)
    write_summary(rows, output_dir, cfg.num_samples)
    print(f"Wrote {output_dir / 'results.csv'}")
    print(f"Wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
