#!/usr/bin/env python3
"""
Assign port/net names to pre-ordered bump maps in place.

The script preserves the existing bump order in each `.bmap` file and only
replaces the last two columns (`port`, `net`). Categories are assigned from top
to bottom in this order:

1. critical bumps
2. redundant bumps (always consecutive pairs)
3. power/ground bumps
4. dummy bumps

By default it processes:
  D2W/input/design_1
  D2W/input/design_2
  D2W/input/design_3
  D2W/input/design_4
  D2W/input/design_17
  D2W/input/design_18
  D2W/input/design_19

Examples
--------
Dry-run a single copied file:
  python D2W/utils/assign_bump_names.py --file /tmp/test.bmap --dry-run

Overwrite one file:
  python D2W/utils/assign_bump_names.py --file /tmp/test.bmap

Overwrite all default designs with custom ratios:
  python D2W/utils/assign_bump_names.py --critical-ratio 0.15 --redundant-ratio 0.10 --pg-ratio 0.75 --dummy-ratio 0.00
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterable


DEFAULT_DESIGNS = ("1", "2", "3", "4", "17", "18", "19")


@dataclass
class BmapEntry:
    instance: str
    bump_type: str
    x: str
    y: str
    port: str
    net: str
    raw_line: str


@dataclass(frozen=True)
class CategoryCounts:
    critical: int
    redundant: int
    pg: int
    dummy: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign critical/redundant/PG/dummy names to ordered bump maps."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("D2W/input"),
        help="Root directory that contains design_* folders.",
    )
    parser.add_argument(
        "--designs",
        type=str,
        default=",".join(DEFAULT_DESIGNS),
        help="Comma-separated design ids to process when --files is not used.",
    )
    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument(
        "--file",
        type=Path,
        help="Optional single .bmap file to process instead of discovering design folders.",
    )
    file_group.add_argument(
        "--files",
        nargs="*",
        type=Path,
        help="Optional explicit list of .bmap files to process instead of discovering design folders.",
    )
    parser.add_argument(
        "--critical-ratio",
        type=float,
        default=0.15,
        help="Physical bump ratio assigned to critical bumps.",
    )
    parser.add_argument(
        "--redundant-ratio",
        type=float,
        default=0.10,
        help="Physical bump ratio assigned to redundant bumps. Must resolve to an even bump count.",
    )
    parser.add_argument(
        "--pg-ratio",
        type=float,
        default=0.75,
        help="Physical bump ratio assigned to power/ground bumps.",
    )
    parser.add_argument(
        "--dummy-ratio",
        type=float,
        default=0.0,
        help="Physical bump ratio assigned to dummy bumps.",
    )
    parser.add_argument(
        "--critical-name-mode",
        choices=("suffix", "instance", "sequential"),
        default="suffix",
        help="How to name critical bumps. 'suffix' maps ..._b_0_1 -> b_0_1.",
    )
    parser.add_argument(
        "--critical-prefix",
        type=str,
        default="crit_",
        help="Prefix used only when --critical-name-mode=sequential.",
    )
    parser.add_argument(
        "--redundant-prefix",
        type=str,
        default="rd_",
        help="Prefix for redundant pair names.",
    )
    parser.add_argument(
        "--pg-pattern",
        type=str,
        default="VDD,VSS",
        help="Comma-separated PG net names assigned cyclically within the PG block.",
    )
    parser.add_argument(
        "--dummy-name",
        type=str,
        default="dummy",
        help="Port/net name used for dummy bumps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without overwriting files.",
    )
    return parser.parse_args()


def parse_design_ids(text: str) -> list[str]:
    design_ids = [item.strip() for item in text.split(",") if item.strip()]
    if not design_ids:
        raise ValueError("No design ids were provided.")
    return design_ids


def discover_bmap_files(input_root: Path, design_ids: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for design_id in design_ids:
        design_dir = input_root / f"design_{design_id}"
        if not design_dir.is_dir():
            raise FileNotFoundError(f"Design directory not found: {design_dir}")
        files.extend(sorted(design_dir.rglob("*.bmap")))
    return files


def read_bmap_entries(path: Path) -> list[BmapEntry]:
    entries: list[BmapEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 6:
                raise ValueError(
                    f"{path}:{line_num} has {len(parts)} columns; expected 6 columns."
                )
            entries.append(
                BmapEntry(
                    instance=parts[0],
                    bump_type=parts[1],
                    x=parts[2],
                    y=parts[3],
                    port=parts[4],
                    net=parts[5],
                    raw_line=stripped,
                )
            )
    if not entries:
        raise ValueError(f"No bump entries found in {path}.")
    return entries


def _validate_ratio(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")


def _largest_remainder_counts(total: int, desired: list[float]) -> list[int]:
    counts = [math.floor(value) for value in desired]
    remaining = total - sum(counts)
    order = sorted(
        range(len(desired)),
        key=lambda idx: (desired[idx] - counts[idx], desired[idx]),
        reverse=True,
    )
    for idx in order[:remaining]:
        counts[idx] += 1
    return counts


def _score(candidate: list[int], desired: list[float]) -> float:
    return sum((observed - target) ** 2 for observed, target in zip(candidate, desired))


def realize_category_counts(
    total_bumps: int,
    critical_ratio: float,
    redundant_ratio: float,
    pg_ratio: float,
    dummy_ratio: float,
) -> CategoryCounts:
    for name, value in (
        ("critical_ratio", critical_ratio),
        ("redundant_ratio", redundant_ratio),
        ("pg_ratio", pg_ratio),
        ("dummy_ratio", dummy_ratio),
    ):
        _validate_ratio(name, value)

    ratio_sum = critical_ratio + redundant_ratio + pg_ratio + dummy_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "critical_ratio + redundant_ratio + pg_ratio + dummy_ratio must sum to 1.0."
        )

    desired = [
        critical_ratio * total_bumps,
        redundant_ratio * total_bumps,
        pg_ratio * total_bumps,
        dummy_ratio * total_bumps,
    ]
    counts = _largest_remainder_counts(total_bumps, desired)

    if counts[1] % 2 == 1:
        best_candidate = None
        best_score = None
        for other_idx in (0, 2, 3):
            if counts[other_idx] > 0:
                candidate = counts.copy()
                candidate[1] += 1
                candidate[other_idx] -= 1
                score = _score(candidate, desired)
                if best_score is None or score < best_score:
                    best_candidate = candidate
                    best_score = score

            if counts[1] > 0:
                candidate = counts.copy()
                candidate[1] -= 1
                candidate[other_idx] += 1
                score = _score(candidate, desired)
                if best_score is None or score < best_score:
                    best_candidate = candidate
                    best_score = score

        if best_candidate is None or best_candidate[1] % 2 == 1:
            raise ValueError("Unable to resolve an even redundant bump count.")
        counts = best_candidate

    return CategoryCounts(
        critical=counts[0],
        redundant=counts[1],
        pg=counts[2],
        dummy=counts[3],
    )


def extract_suffix_name(instance: str) -> str:
    marker = "_b_"
    if marker in instance:
        return instance[instance.index(marker) + 1 :]
    return instance


def build_critical_name(
    entry: BmapEntry,
    index: int,
    mode: str,
    prefix: str,
) -> str:
    if mode == "instance":
        return entry.instance
    if mode == "sequential":
        return f"{prefix}{index}"
    return extract_suffix_name(entry.instance)


def assign_names(
    entries: list[BmapEntry],
    counts: CategoryCounts,
    critical_name_mode: str,
    critical_prefix: str,
    redundant_prefix: str,
    pg_pattern: list[str],
    dummy_name: str,
) -> list[str]:
    if not pg_pattern:
        raise ValueError("pg_pattern must contain at least one name.")

    rewritten_lines: list[str] = []
    idx = 0

    for critical_idx in range(1, counts.critical + 1):
        entry = entries[idx]
        name = build_critical_name(entry, critical_idx, critical_name_mode, critical_prefix)
        rewritten_lines.append(
            f"{entry.instance} {entry.bump_type} {entry.x} {entry.y} {name} {name}"
        )
        idx += 1

    for redundant_idx in range(1, counts.redundant // 2 + 1):
        pair_name = f"{redundant_prefix}{redundant_idx}"
        for _ in range(2):
            entry = entries[idx]
            rewritten_lines.append(
                f"{entry.instance} {entry.bump_type} {entry.x} {entry.y} {pair_name} {pair_name}"
            )
            idx += 1

    pg_name_iter = cycle(pg_pattern)
    for _ in range(counts.pg):
        entry = entries[idx]
        pg_name = next(pg_name_iter)
        rewritten_lines.append(
            f"{entry.instance} {entry.bump_type} {entry.x} {entry.y} {pg_name} {pg_name}"
        )
        idx += 1

    for _ in range(counts.dummy):
        entry = entries[idx]
        rewritten_lines.append(
            f"{entry.instance} {entry.bump_type} {entry.x} {entry.y} {dummy_name} {dummy_name}"
        )
        idx += 1

    if idx != len(entries):
        raise AssertionError("Internal error: not all bump entries were assigned.")

    return rewritten_lines


def write_lines(path: Path, lines: list[str]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    tmp_path.replace(path)


def summarize_counts(path: Path, counts: CategoryCounts, total_bumps: int) -> str:
    logical_signal_ratio = (counts.critical + counts.redundant / 2) / total_bumps
    return (
        f"{path}: total={total_bumps}, critical={counts.critical}, "
        f"redundant={counts.redundant} ({counts.redundant // 2} nets), "
        f"pg={counts.pg}, dummy={counts.dummy}, "
        f"logical_signal_ratio={logical_signal_ratio:.6f}"
    )


def main() -> None:
    args = parse_args()
    pg_pattern = [name.strip() for name in args.pg_pattern.split(",") if name.strip()]

    if args.file is not None:
        files = [args.file.resolve()]
    elif args.files:
        files = [path.resolve() for path in args.files]
    else:
        design_ids = parse_design_ids(args.designs)
        files = discover_bmap_files(args.input_root.resolve(), design_ids)

    if not files:
        raise FileNotFoundError("No .bmap files were found to process.")

    for path in files:
        entries = read_bmap_entries(path)
        counts = realize_category_counts(
            total_bumps=len(entries),
            critical_ratio=args.critical_ratio,
            redundant_ratio=args.redundant_ratio,
            pg_ratio=args.pg_ratio,
            dummy_ratio=args.dummy_ratio,
        )
        rewritten_lines = assign_names(
            entries=entries,
            counts=counts,
            critical_name_mode=args.critical_name_mode,
            critical_prefix=args.critical_prefix,
            redundant_prefix=args.redundant_prefix,
            pg_pattern=pg_pattern,
            dummy_name=args.dummy_name,
        )

        print(summarize_counts(path, counts, len(entries)))
        if not args.dry_run:
            write_lines(path, rewritten_lines)


if __name__ == "__main__":
    main()
