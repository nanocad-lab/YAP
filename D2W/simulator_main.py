#!/usr/bin/env python3

import argparse
import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"


def run(cfg):
    from assembly_yield_simulator import Assembly_Yield_Simulator
    from pad_bitmap_generation import pad_bitmap_generate_random
    from utils.util import update_config_items

    os.chdir(BASE_DIR)
    update_config_items(cfg=cfg, mode="d2w_simulation")

    overlay_mode = (
        "full pad-by-pad" if cfg.approximate_set == 1 else "boundary approximation"
    )
    print(
        f"Monte Carlo workload: {cfg.simulation_times} batch(es) x "
        f"{cfg.NUM_DIES} die(s); overlay mode: {overlay_mode}."
    )

    pad_bitmap_collection = pad_bitmap_generate_random(
        cfg=cfg,
        pad_layout_pattern=cfg.pad_layout_pattern,
    )

    start_time = time.time()
    result = Assembly_Yield_Simulator(
        cfg=cfg,
        pad_bitmap_collection=pad_bitmap_collection,
    )
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
    return result


def main(config_path=DEFAULT_CONFIG_PATH):
    from utils.util import load_modeling_config

    cfg = load_modeling_config(
        path=str(config_path),
        mode="d2w_simulation",
        debug=False,
    )
    return run(cfg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the D2W Monte Carlo simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a YAML file containing the d2w_simulation section.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().config.resolve())
