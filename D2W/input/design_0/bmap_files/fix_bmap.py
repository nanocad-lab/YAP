#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fix .bmap lines by replacing leading placeholder "- -" (or "-- --")
and trailing placeholder "- -" (or "-- --") with the bump name in that line.

Example:
- - silicon_individual_bonding 5 3155 - -
=> b_0_315 silicon_individual_bonding 5 3155 b_0_315 b_0_315

Usage:
  python fix_bmap.py input.bmap output.bmap

If output path is omitted, it writes <input>_fixed.bmap
"""

import sys
from pathlib import Path

PLACEHOLDERS = {"-", "- -"}

def is_placeholder(tok: str) -> bool:
    return tok in PLACEHOLDERS

def fix_line(line: str) -> str:
    raw = line.rstrip("\n")
    if not raw.strip():
        return raw

    parts = raw.split()
    # Need at least 5 tokens to look like: <p0> <p1> <bump_name> ... <t-2> <t-1>
    if len(parts) < 5:
        return raw

    bump_name = parts[0]  # bump name is the 3rd token in your format

    # Replace leading two tokens if they are placeholders
    if len(parts) >= 2 and is_placeholder(parts[0]) and is_placeholder(parts[1]):
        parts[0] = bump_name
        parts[1] = bump_name

    # Replace trailing two tokens if they are placeholders
    if len(parts) >= 2 and is_placeholder(parts[-2]) and is_placeholder(parts[-1]):
        parts[-2] = bump_name
        parts[-1] = bump_name

    return " ".join(parts)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_bmap.py input.bmap [output.bmap]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}")
        sys.exit(2)

    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    else:
        out_path = in_path.with_name(in_path.stem + "_fixed" + in_path.suffix)

    lines = in_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    fixed = [fix_line(l) + ("" if l.endswith("\n") else "") for l in lines]

    # Preserve original newline behavior: write with '\n' join for consistency
    out_path.write_text("\n".join([s.rstrip("\n") for s in fixed]) + "\n", encoding="utf-8")
    print(f"OK: wrote {out_path}")

if __name__ == "__main__":
    main()
