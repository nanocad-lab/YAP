#!/usr/bin/env python3
"""
Build original-size project benchmarks for design_1 at 5um / 10um pitch.

Outputs
-------
- D2W/input/design_1_p5_proj
- D2W/input/design_1_p10_proj
- D2W/configs/design_1_p5_proj
- D2W/configs/design_1_p10_proj

Rules
-----
- Start from the original-size legacy geometry under D2W/input/old_bmap/design_1.
- Keep original die sizes and stack placement unchanged.
- Only emit the requested project ratio / variant subset.
- Rebuild every .bmap from scratch at the requested pitch.
- Reapply design_1 naming/criticality using assign_design1_interchip_nets.py.
- Copy current process-parameter YAMLs from design_1_p5 as the config source.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
OLD_INPUT_ROOT = REPO_ROOT / "D2W/input/old_bmap/design_1"
SOURCE_CONFIG_ROOT = REPO_ROOT / "D2W/configs/design_1_p5"
DEFAULT_TEMPLATE_RATIO = "c25_r0_pg50_dm25"
DEFAULT_TARGET_RATIO = "c30_r0_pg50_dm20"
DEFAULT_VARIANTS = ("Edge_IO",)

CONFIG_SUFFIXES = (
    "",
    "_overlay_pessimistic",
    "_particle_pessimistic",
    "_mechanical_pessimistic",
    "_ESD_pessimistic",
)


@dataclass(frozen=True)
class BenchSpec:
    name: str
    pitch_um: float
    bump_size_um: float


SPECS = (
    BenchSpec(name="design_1_p5_proj", pitch_um=5.0, bump_size_um=2.5),
    BenchSpec(name="design_1_p10_proj", pitch_um=10.0, bump_size_um=5.0),
    BenchSpec(name="design_1_p20_proj", pitch_um=20.0, bump_size_um=10.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build project benchmark versions of design_1.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel workers for .bmap regeneration.",
    )
    parser.add_argument(
        "--template-ratio",
        default=DEFAULT_TEMPLATE_RATIO,
        help="Legacy ratio folder to use as the structural template.",
    )
    parser.add_argument(
        "--target-ratio",
        default=DEFAULT_TARGET_RATIO,
        help="Output ratio folder name to create.",
    )
    parser.add_argument(
        "--variants-csv",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated list of variant folders to keep, e.g. Edge_IO.",
    )
    parser.add_argument(
        "--designs-csv",
        default=",".join(spec.name for spec in SPECS),
        help="Comma-separated subset of benchmark names to build.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def replace_yaml_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)([^#\n]*)(.*)$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<1>{value}\g<3>", text)
    if count == 0:
        raise ValueError(f"Key '{key}' not found while updating text.")
    return updated


def clone_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def clone_minimal_tree(
    src_root: Path,
    dst_root: Path,
    *,
    template_ratio: str,
    target_ratio: str,
    variants: tuple[str, ...],
) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    template_ratio_path = src_root / template_ratio
    if not template_ratio_path.exists():
        raise FileNotFoundError(f"Template ratio folder not found: {template_ratio_path}")

    for item in sorted(src_root.iterdir()):
        if item.name == template_ratio:
            shutil.copytree(item, dst_root / target_ratio)
        elif item.is_file():
            shutil.copy2(item, dst_root / item.name)

    out_ratio_dir = dst_root / target_ratio
    for child in sorted(out_ratio_dir.iterdir()):
        if child.is_dir() and child.name not in variants:
            shutil.rmtree(child)


def update_3dbf_text(text: str, *, new_pitch_um: float, new_bump_size_um: float) -> str:
    updates = {
        "pitch": f"{new_pitch_um:.1f}",
        "bump_size": f"{new_bump_size_um:.1f}",
        "through_via_size": f"{new_bump_size_um:.1f}",
    }
    for key, value in updates.items():
        text = replace_yaml_scalar(text, key, value)
    return text


def owner_die_name_from_bmap(bmap_name: str) -> str:
    stem = Path(bmap_name).stem
    if "_From_" in stem:
        return stem.split("_From_", 1)[0]
    if "_To_" in stem:
        return stem.split("_To_", 1)[0]
    raise ValueError(f"Cannot infer owner die from {bmap_name}")


def load_entries(path: Path) -> list[list[str]]:
    entries: list[list[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if parts:
                entries.append(parts)
    if not entries:
        raise ValueError(f"No bump entries found in {path}")
    return entries


def parse_chiplet_areas(root_3dbv_path: Path) -> dict[str, tuple[float, float]]:
    data = yaml.safe_load(root_3dbv_path.read_text(encoding="utf-8"))
    defs = data["ChipletDef"]
    items = defs.values() if isinstance(defs, dict) else defs
    areas: dict[str, tuple[float, float]] = {}
    for item in items:
        name = item["name"] if "name" in item else None
        if name is None:
            # dict-valued 3dbv uses key as the name
            continue
    if isinstance(defs, dict):
        for name, item in defs.items():
            design_area = item.get("design_area")
            if design_area:
                areas[name] = (float(design_area[0]), float(design_area[1]))
    else:
        for item in defs:
            name = item.get("name")
            design_area = item.get("design_area")
            if name and design_area:
                areas[name] = (float(design_area[0]), float(design_area[1]))
    return areas


def infer_pitch_from_entries(entries: list[list[str]]) -> float:
    x_values = sorted({float(parts[2]) for parts in entries})
    y_values = sorted({float(parts[3]) for parts in entries})
    deltas = []
    if len(x_values) > 1:
        deltas.extend(x_values[i + 1] - x_values[i] for i in range(len(x_values) - 1))
    if len(y_values) > 1:
        deltas.extend(y_values[i + 1] - y_values[i] for i in range(len(y_values) - 1))
    deltas = [abs(v) for v in deltas if abs(v) > 1e-9]
    if not deltas:
        raise ValueError("Could not infer pitch from a single-point bump map.")
    return min(deltas)


def build_source_bmap_catalog(source_root: Path) -> dict[str, dict[str, float | str]]:
    chiplet_areas = parse_chiplet_areas(source_root / "generated_chiplet_definitions.3dbv")
    catalog: dict[str, dict[str, float | str]] = {}
    for bmap_path in sorted(source_root.rglob("*.bmap")):
        if bmap_path.name in catalog:
            continue
        entries = load_entries(bmap_path)
        x_values = sorted({float(parts[2]) for parts in entries})
        y_values = sorted({float(parts[3]) for parts in entries})
        old_pitch = infer_pitch_from_entries(entries)
        owner_die = owner_die_name_from_bmap(bmap_path.name)
        die_w_um, die_l_um = chiplet_areas[owner_die]
        bump_type = entries[0][1]
        catalog[bmap_path.name] = {
            "interface_name": bmap_path.stem,
            "bump_type": bump_type,
            "die_w_um": die_w_um,
            "die_l_um": die_l_um,
            "pad_arr_w_um": old_pitch * (len(x_values) - 1),
            "pad_arr_l_um": old_pitch * (len(y_values) - 1),
        }
    return catalog


def generate_bmap(
    out_path: Path,
    interface_name: str,
    bump_type: str,
    die_w_um: float,
    die_l_um: float,
    pad_arr_w_um: float,
    pad_arr_l_um: float,
    pitch_um: float,
) -> None:
    cols = int(round(pad_arr_w_um / pitch_um)) + 1
    rows = int(round(pad_arr_l_um / pitch_um)) + 1

    x0 = (die_w_um - pad_arr_w_um) / 2.0
    y_top = die_l_um - (die_l_um - pad_arr_l_um) / 2.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in range(rows):
            y = y_top - row * pitch_um
            for col in range(cols):
                x = x0 + col * pitch_um
                instance = f"{interface_name}_b_{row}_{col}"
                f.write(
                    f"{instance} {bump_type} {x:.4f} {y:.4f} placeholder placeholder\n"
                )
    tmp.replace(out_path)


def build_config_tree(spec: BenchSpec) -> None:
    target_root = REPO_ROOT / "D2W/configs" / spec.name
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    for suffix in CONFIG_SUFFIXES:
        source_name = f"design_1_p5{suffix}.yaml"
        target_name = f"{spec.name}{suffix}.yaml"
        source_path = SOURCE_CONFIG_ROOT / source_name
        target_path = target_root / target_name
        text = read_text(source_path).replace("design_1_p5", spec.name)
        write_text(target_path, text)


def build_input_tree(
    spec: BenchSpec,
    jobs: int,
    catalog: dict[str, dict[str, float | str]],
    *,
    template_ratio: str,
    target_ratio: str,
    variants: tuple[str, ...],
) -> None:
    target_root = REPO_ROOT / "D2W/input" / spec.name
    clone_minimal_tree(
        OLD_INPUT_ROOT,
        target_root,
        template_ratio=template_ratio,
        target_ratio=target_ratio,
        variants=variants,
    )

    for dbf_path in sorted(target_root.rglob("*.3dbf")):
        updated = update_3dbf_text(
            read_text(dbf_path),
            new_pitch_um=spec.pitch_um,
            new_bump_size_um=spec.bump_size_um,
        )
        write_text(dbf_path, updated)

    tasks = []
    for bmap_path in sorted(target_root.rglob("*.bmap")):
        meta = catalog[bmap_path.name]
        tasks.append(
            (
                bmap_path,
                meta["interface_name"],
                meta["bump_type"],
                float(meta["die_w_um"]),
                float(meta["die_l_um"]),
                float(meta["pad_arr_w_um"]),
                float(meta["pad_arr_l_um"]),
                spec.pitch_um,
            )
        )

    with cf.ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(generate_bmap, *task) for task in tasks]
        for future in cf.as_completed(futures):
            future.result()

    subprocess.run(
        ["python", str(REPO_ROOT / "D2W/utils/assign_design1_interchip_nets.py"), "--design-root", str(target_root)],
        check=True,
        cwd=REPO_ROOT,
    )


def main() -> None:
    args = parse_args()
    variants = tuple(v.strip() for v in args.variants_csv.split(",") if v.strip())
    if not variants:
        raise ValueError("At least one variant must be provided.")
    selected_names = {name.strip() for name in args.designs_csv.split(",") if name.strip()}
    specs = tuple(spec for spec in SPECS if spec.name in selected_names)
    missing = sorted(selected_names - {spec.name for spec in SPECS})
    if missing:
        raise ValueError(f"Unknown benchmark names requested: {', '.join(missing)}")
    if not specs:
        raise ValueError("No benchmark specs selected.")
    catalog = build_source_bmap_catalog(OLD_INPUT_ROOT)
    for spec in specs:
        print(
            f"=== Building {spec.name} at pitch {spec.pitch_um} um "
            f"for {args.target_ratio} ({', '.join(variants)}) ==="
        )
        build_config_tree(spec)
        build_input_tree(
            spec,
            jobs=args.jobs,
            catalog=catalog,
            template_ratio=args.template_ratio,
            target_ratio=args.target_ratio,
            variants=variants,
        )
        print(f"=== Finished {spec.name} ===")


if __name__ == "__main__":
    main()
