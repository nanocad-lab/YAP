#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from utils.util import *
import time
import argparse
from assembly_yield_calculator import Pad_Yield_Map_Generator


def parse_args():
    p = argparse.ArgumentParser(description="Pad risk map / bitmap tools")
    p.add_argument("--config", "-c", required=True, help="Path to modeling config yaml")
    p.add_argument("--mode", "-m", default="w2w_modeling", help="Mode to load from config (default: w2w_modeling)")
    p.add_argument("--bmap", "-b", required=True, help="Path to .bmap file (overrides default input/<DESIGN>.bmap)")
    p.add_argument("--criticality", "-cr", required=True, help="Path to criticality file (overrides default input/<DESIGN>_criticality.txt)")
    p.add_argument("--plot", "-plot", default=False, action="store_true", help="Enable plotting of the pad risk map")
    p.add_argument("--debug", action="store_true", help="Enable debug output when loading config")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = load_modeling_config(path=args.config, mode=args.mode, debug=args.debug)

    # Plotting flag
    cfg.plot_flag = args.plot
    
    # Create output directory if it doesn't exist
    if not os.path.exists(cfg.OUTPUT_DIR + cfg.DESIGN):
        os.makedirs(cfg.OUTPUT_DIR + cfg.DESIGN)

    # Update cfg (same as before)
    update_config_items(cfg=cfg, mode=args.mode)

    # Determine .bmap path
    blox_bmap_path = args.bmap

    # Determine criticality path
    criticality_path = args.criticality

    # Step 1: convert .bmap -> pad bitmap collection
    pad_bitmap_collection = convert_3dblox_to_pad_bitmap(cfg=cfg,
                                                        blox_bmap_path=blox_bmap_path,
                                                        criticality_path=criticality_path,
                                                        pad_arrange_pattern=cfg.PAD_ARRANGE_PATTERN)

    # Step 2: generate pad-level yield map
    print("Calculating pad-level yield map...")
    start_time = time.time()
    Pad_Yield_Map_Generator(
        cfg=cfg,
        pad_bitmap_collection=pad_bitmap_collection,
    )
    print(f"Pad yield map generation finished in {time.time() - start_time:.2f} s")

if __name__ == "__main__":
    main()