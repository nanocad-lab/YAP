#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot critical bump distributions for design_1 project benchmarks."
    )
    parser.add_argument(
        "--output",
        default="D2W/output/design1_proj_critical_distribution.png",
        help="Output image path.",
    )
    return parser.parse_args()


def load_points(path: Path) -> tuple[list[float], list[float], list[float], list[float]]:
    crit_x: list[float] = []
    crit_y: list[float] = []
    other_x: list[float] = []
    other_y: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if len(parts) != 6:
                continue
            x = float(parts[2])
            y = float(parts[3])
            net = parts[5]
            if "_link_" in net or "_ext_crit_" in net:
                crit_x.append(x)
                crit_y.append(y)
            else:
                other_x.append(x)
                other_y.append(y)
    return crit_x, crit_y, other_x, other_y


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("x (um)", fontsize=15)
    ax.set_ylabel("y (um)", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


def plot_panel(ax: plt.Axes, path: Path, title: str, edge_side: str) -> None:
    crit_x, crit_y, other_x, other_y = load_points(path)
    ax.scatter(other_x, other_y, s=2, c="#c7c7c7", alpha=0.35, linewidths=0, label="non-critical")
    ax.scatter(crit_x, crit_y, s=5, c="#d62728", alpha=0.9, linewidths=0, label="critical")
    style_axis(ax, title)

    if edge_side == "left":
        edge_x = min(crit_x) if crit_x else min(other_x)
        ax.axvline(edge_x, color="#1f77b4", linestyle=":", linewidth=1.5)
        ax.text(edge_x, ax.get_ylim()[1], "adjacent edge", color="#1f77b4", fontsize=11, va="top", ha="left")
    else:
        edge_x = max(crit_x) if crit_x else max(other_x)
        ax.axvline(edge_x, color="#1f77b4", linestyle=":", linewidth=1.5)
        ax.text(edge_x, ax.get_ylim()[1], "adjacent edge", color="#1f77b4", fontsize=11, va="top", ha="right")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        (
            Path("D2W/input/design_1_p10_proj/c30_r0_pg50_dm20/Edge_IO/Compute_Small_From_Substrate_Silicon.bmap"),
            "design_1_p10_proj: Compute_Small",
            "left",
        ),
        (
            Path("D2W/input/design_1_p10_proj/c30_r0_pg50_dm20/Edge_IO/Memory_DRAM_From_Substrate_Silicon.bmap"),
            "design_1_p10_proj: Memory_DRAM",
            "right",
        ),
        (
            Path("D2W/input/design_1_p20_proj/c30_r0_pg50_dm20/Edge_IO/Compute_Small_From_Substrate_Silicon.bmap"),
            "design_1_p20_proj: Compute_Small",
            "left",
        ),
        (
            Path("D2W/input/design_1_p20_proj/c30_r0_pg50_dm20/Edge_IO/Memory_DRAM_From_Substrate_Silicon.bmap"),
            "design_1_p20_proj: Memory_DRAM",
            "right",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)
    for ax, (path, title, edge_side) in zip(axes.flat, panels):
        plot_panel(ax, path, title, edge_side)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=14, frameon=True)
    fig.suptitle("Critical Bump Distributions for design_1 Project Benchmarks", fontsize=22)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
