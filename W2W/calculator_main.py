#!/usr/bin/env python3

import argparse
import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
BITMAP_PATH = BASE_DIR / "pad_bitmap" / "bitmap_collection.npy"
DILATION_CACHE_PATH = BASE_DIR / "pad_bitmap" / "avg_num_defects_per_unit_area.npy"


def _load_cached_bitmap(cfg):
    import numpy as np

    if not BITMAP_PATH.exists():
        raise FileNotFoundError(
            f"PITCH_um={cfg.PITCH_um} requires an existing bitmap: {BITMAP_PATH}"
        )

    print("PITCH_um is below 1.0; loading the existing pad bitmap collection.")
    bitmap = np.load(BITMAP_PATH, allow_pickle=True).item()
    bitmap["pad_block_size"] = cfg.pad_block_size
    bitmap["num_critical_pads"] = (
        cfg.PAD_ARR_ROW * cfg.PAD_ARR_COL * cfg.critical_pad_ratio
    )
    bitmap["num_redundant_logical_pads"] = (
        cfg.PAD_ARR_ROW
        * cfg.PAD_ARR_COL
        * cfg.redundant_pad_ratio
        * cfg.redundant_logical_pad_ratio
    )
    bitmap["redundant_logical_pad_copy"] = cfg.redundant_logical_pad_copy
    return bitmap


def run(cfg):
    from assembly_yield_calculator import Assembly_Yield_Calculator
    from pad_bitmap_generation import pad_bitmap_generate
    from utils.util import update_config_items

    os.chdir(BASE_DIR)
    update_config_items(cfg=cfg, mode="w2w_modeling")

    if cfg.PITCH_um >= 1.0:
        pad_bitmap_collection = pad_bitmap_generate(
            cfg=cfg,
            pad_layout_pattern=cfg.pad_layout_pattern,
        )
    else:
        pad_bitmap_collection = _load_cached_bitmap(cfg)

    if not cfg.reuse_dilation and DILATION_CACHE_PATH.exists():
        DILATION_CACHE_PATH.unlink()

    start_time = time.time()
    yields = Assembly_Yield_Calculator(
        cfg=cfg,
        pad_bitmap_collection=pad_bitmap_collection,
    )
    elapsed = time.time() - start_time
    assembly_yield, overlay_yield, defect_yield, cu_expansion_yield = yields

    print(f"Y_ovl:  {overlay_yield:.4f}")
    print(f"Y_df:   {defect_yield:.4f}")
    print(f"Y_cr:   {cu_expansion_yield:.4f}")
    print(f"Y_asmb: {assembly_yield:.4f}")
    print(f"Total time taken: {elapsed:.2f} seconds")
    return yields


def main(config_path=DEFAULT_CONFIG_PATH):
    from utils.util import load_modeling_config

    cfg = load_modeling_config(
        path=str(config_path),
        mode="w2w_modeling",
        debug=False,
    )
    return run(cfg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the W2W analytical yield model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a YAML file containing the w2w_modeling section.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().config.resolve())
