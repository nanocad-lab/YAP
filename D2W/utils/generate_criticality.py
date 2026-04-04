#!/usr/bin/env python3
"""
Generate criticality files from one or more bump maps.

For each net in the bump map, this script writes:
    <net> <group_size> <tolerated_esd_failures> <tolerated_mechanical_failures>

The default rule is:
    tolerated_esd_failures = group_size - 1
    tolerated_mechanical_failures = group_size - 1

That means:
- critical bumps with one copy become: 1 0 0
- redundant pairs with two copies become: 2 1 1

By default, the script processes all `.bmap` files under:
  D2W/input/design_1
  D2W/input/design_2
  D2W/input/design_3
  D2W/input/design_4
  D2W/input/design_17
  D2W/input/design_18
  D2W/input/design_19

Examples
--------
Single file:
  python D2W/utils/generate_criticality.py --file path/to/file.bmap

Single file, backward-compatible positional form:
  python D2W/utils/generate_criticality.py path/to/file.bmap

Dry-run one file:
  python D2W/utils/generate_criticality.py --file path/to/file.bmap --dry-run

Batch default designs:
  python D2W/utils/generate_criticality.py --force
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_DESIGNS = ("1", "2", "3", "4", "17", "18", "19")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate criticality files from one or more .bmap files."
    )
    parser.add_argument(
        "input_bmap_file",
        nargs="?",
        type=Path,
        help="Optional single .bmap file to process (backward-compatible form).",
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
        help="Comma-separated design ids to process when no explicit file list is given.",
    )

    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument(
        "--file",
        type=Path,
        help="Optional single .bmap file to process.",
    )
    file_group.add_argument(
        "--files",
        nargs="*",
        type=Path,
        help="Optional explicit list of .bmap files to process.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files.",
    )
    return parser


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


def read_bmap_nets(filename: Path) -> dict[str, int]:
    """
    Read a bump map file and count bumps per net.
    """
    net_counts: Counter[str] = Counter()

    with open(filename, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 6:
                print(
                    f"Warning: {filename}:{line_num} has {len(parts)} fields "
                    f"(expected 6), skipping: {line}"
                )
                continue

            net_counts[parts[5]] += 1

    if not net_counts:
        raise ValueError(f"No valid bump entries found in {filename}")

    return dict(net_counts)


def generate_criticality_lines(net_counts: dict[str, int]) -> list[str]:
    lines: list[str] = []
    for net in sorted(net_counts.keys()):
        group_size = net_counts[net]
        tolerated_esd = group_size - 1
        tolerated_mech = group_size - 1
        lines.append(f"{net} {group_size} {tolerated_esd} {tolerated_mech}")
    return lines


def get_output_filename(input_filename: Path) -> Path:
    return input_filename.with_name(f"{input_filename.stem}_criticality.txt")


def summarize_net_counts(net_counts: dict[str, int]) -> list[str]:
    criticality_values: defaultdict[float, list[str]] = defaultdict(list)
    for net, count in net_counts.items():
        criticality = 1.0 / count
        criticality_values[criticality].append(net)

    summary_lines: list[str] = []
    for criticality in sorted(criticality_values.keys(), reverse=True):
        nets = criticality_values[criticality]
        bump_count = int(round(1.0 / criticality))
        summary_lines.append(
            f"  Criticality {criticality:.4f} ({bump_count} bump"
            f"{'s' if bump_count > 1 else ''}): {len(nets)} net"
            f"{'s' if len(nets) > 1 else ''}"
        )
    return summary_lines


def write_criticality_file(output_filename: Path, lines: list[str], force: bool) -> None:
    if output_filename.exists() and not force:
        raise FileExistsError(
            f"{output_filename} already exists. Use --force to overwrite it."
        )

    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def resolve_input_files(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Path]:
    if args.input_bmap_file is not None and (args.file is not None or args.files):
        parser.error("Use only one of positional input_bmap_file, --file, or --files.")

    if args.file is not None:
        return [args.file.resolve()]

    if args.files:
        return [path.resolve() for path in args.files]

    if args.input_bmap_file is not None:
        return [args.input_bmap_file.resolve()]

    design_ids = parse_design_ids(args.designs)
    return discover_bmap_files(args.input_root.resolve(), design_ids)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    files = resolve_input_files(args, parser)

    if not files:
        raise FileNotFoundError("No .bmap files were found to process.")

    processed_count = 0
    for input_filename in files:
        if not input_filename.exists():
            raise FileNotFoundError(f"Input file '{input_filename}' not found")

        net_counts = read_bmap_nets(input_filename)
        output_filename = get_output_filename(input_filename)
        criticality_lines = generate_criticality_lines(net_counts)

        print(f"Reading bump map: {input_filename}")
        print(f"Output file: {output_filename}")
        print(f"Total nets: {len(net_counts)}")
        for line in summarize_net_counts(net_counts):
            print(line)

        if args.dry_run:
            print("Dry-run: not writing output file.")
        else:
            write_criticality_file(output_filename, criticality_lines, args.force)
            print(f"Successfully generated criticality file: {output_filename}")

        print()
        processed_count += 1

    print(f"Processed {processed_count} bump map file{'s' if processed_count != 1 else ''}.")


if __name__ == "__main__":
    main()
