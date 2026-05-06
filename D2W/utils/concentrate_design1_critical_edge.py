#!/usr/bin/env python3
"""
Concentrate design_1 critical bumps toward the Compute/Memory adjacent edge.

Current use case:
- original-size project benchmarks such as design_1_p10_proj / design_1_p20_proj
- ratio c30_r0_pg50_dm20
- Edge_IO only

Behavior:
- Treat both shared critical (`*_link_*`) and external critical (`*_ext_crit_*`) bumps
  as "critical".
- For Compute_Small, pack all critical bumps starting from the left edge inward.
- For Memory_DRAM, pack all critical bumps starting from the right edge inward.
- Preserve the relative order of the critical names and the non-critical names.
- Rewrite the matching substrate-side bump maps and regenerate criticality files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import bmap_grid_sync as bgs
import generate_criticality as gc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concentrate design_1 critical bumps onto the adjacent die edge."
    )
    parser.add_argument(
        "--design-root",
        type=Path,
        required=True,
        help="Design input root, e.g. D2W/input/design_1_p10_proj",
    )
    parser.add_argument(
        "--ratio",
        default="c30_r0_pg50_dm20",
        help="Ratio directory to rewrite.",
    )
    parser.add_argument(
        "--variant",
        default="Edge_IO",
        help="Variant directory to rewrite.",
    )
    return parser.parse_args()


def read_entries(path: Path) -> list[list[str]]:
    entries: list[list[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if parts:
                entries.append(parts)
    if not entries:
        raise ValueError(f"No bump entries found in {path}")
    return entries


def write_entries(path: Path, entries: list[list[str]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for parts in entries:
            f.write(" ".join(parts) + "\n")
    tmp_path.replace(path)


def write_criticality(path: Path) -> None:
    net_counts: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if len(parts) == 6:
                net = parts[5]
                net_counts[net] = net_counts.get(net, 0) + 1

    for profile in gc.ALL_PROFILES:
        out_path = gc.get_output_filename(path, profile=profile)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for net in sorted(net_counts):
                group_size = net_counts[net]
                tolerated_esd, tolerated_mech = gc.tolerated_failures_for_group(
                    group_size=group_size,
                    profile=profile,
                    net=net,
                )
                f.write(f"{net} {group_size} {tolerated_esd} {tolerated_mech}\n")
        tmp_path.replace(out_path)


def is_critical_name(name: str) -> bool:
    return "_link_" in name or "_ext_crit_" in name


def parse_die_areas(
    design_root: Path,
    ratio: str,
    variant: str,
) -> dict[str, tuple[float, float]]:
    defs_path = design_root / ratio / variant / "generated_chiplet_definitions.3dbv"
    data = yaml.safe_load(defs_path.read_text(encoding="utf-8"))
    defs = data["ChipletDef"]
    areas: dict[str, tuple[float, float]] = {}
    for die_name, item in defs.items():
        design_area = item.get("design_area")
        if design_area:
            areas[die_name] = (float(design_area[0]), float(design_area[1]))
    return areas


def build_priority_order(entries: list[list[str]], owner_die: str) -> list[int]:
    x_values = sorted({float(parts[2]) for parts in entries})
    y_values = sorted({float(parts[3]) for parts in entries}, reverse=True)
    x_rank = {value: idx for idx, value in enumerate(x_values)}
    y_rank = {value: idx for idx, value in enumerate(y_values)}

    decorated: list[tuple[int, int, int]] = []
    for idx, parts in enumerate(entries):
        decorated.append((idx, x_rank[float(parts[2])], y_rank[float(parts[3])]))

    if owner_die == "Compute_Small":
        decorated.sort(key=lambda item: (item[1], item[2]))
    elif owner_die == "Memory_DRAM":
        decorated.sort(key=lambda item: (-item[1], item[2]))
    else:
        raise ValueError(f"Unexpected owner die: {owner_die}")

    return [idx for idx, _, _ in decorated]


def rewrite_entries(entries: list[list[str]], owner_die: str) -> list[list[str]]:
    critical_names = [(parts[4], parts[5]) for parts in entries if is_critical_name(parts[5])]
    noncritical_names = [(parts[4], parts[5]) for parts in entries if not is_critical_name(parts[5])]
    new_names = critical_names + noncritical_names
    if len(new_names) != len(entries):
        raise ValueError("Entry count mismatch while rewriting bump names.")

    order = build_priority_order(entries, owner_die)
    rewritten = [parts[:] for parts in entries]
    for entry_idx, (port_name, net_name) in zip(order, new_names):
        rewritten[entry_idx][4] = port_name
        rewritten[entry_idx][5] = net_name
    return rewritten


def summarize_max_distance(
    entries: list[list[str]],
    owner_die: str,
    die_w_um: float,
) -> tuple[float, int]:
    critical_points = [parts for parts in entries if is_critical_name(parts[5])]
    if not critical_points:
        return 0.0, 0

    if owner_die == "Compute_Small":
        distances = [float(parts[2]) for parts in critical_points]
        edge_columns = len({float(parts[2]) for parts in critical_points})
    else:
        distances = [die_w_um - float(parts[2]) for parts in critical_points]
        edge_columns = len({float(parts[2]) for parts in critical_points})
    return max(distances), edge_columns


def process_pair(design_root: Path, ratio: str, variant: str) -> None:
    areas = parse_die_areas(design_root, ratio, variant)
    variant_dir = design_root / ratio / variant
    ratio_dir = design_root / ratio

    file_specs = [
        ("Compute_Small", "Compute_Small_From_Substrate_Silicon.bmap", "Substrate_Silicon_To_Compute_Small.bmap"),
        ("Memory_DRAM", "Memory_DRAM_From_Substrate_Silicon.bmap", "Substrate_Silicon_To_Memory_DRAM.bmap"),
    ]

    summaries: list[str] = []
    for owner_die, die_bmap_name, substrate_bmap_name in file_specs:
        for base_dir in (variant_dir, ratio_dir):
            die_bmap_path = base_dir / die_bmap_name
            substrate_bmap_path = base_dir / substrate_bmap_name

            die_entries = read_entries(die_bmap_path)
            rewritten_die_entries = rewrite_entries(die_entries, owner_die)
            substrate_entries = read_entries(substrate_bmap_path)
            rewritten_substrate_entries = bgs.sync_names_by_normalized_grid(
                rewritten_die_entries,
                substrate_entries,
            )

            write_entries(die_bmap_path, rewritten_die_entries)
            write_entries(substrate_bmap_path, rewritten_substrate_entries)
            write_criticality(die_bmap_path)
            write_criticality(substrate_bmap_path)

            if base_dir == variant_dir:
                max_distance_um, edge_columns = summarize_max_distance(
                    rewritten_die_entries,
                    owner_die,
                    die_w_um=areas[owner_die][0],
                )
                summaries.append(
                    f"{owner_die}: max critical distance to adjacent edge = "
                    f"{max_distance_um:.3f} um across {edge_columns} columns"
                )

    for line in summaries:
        print(line)


def main() -> None:
    args = parse_args()
    process_pair(args.design_root, args.ratio, args.variant)


if __name__ == "__main__":
    main()
