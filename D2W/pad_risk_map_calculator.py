#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from utils.util import *
import time
import argparse
from assembly_yield_calculator import Pad_Yield_Map_Generator


def parse_args():
    p = argparse.ArgumentParser(description="Simulate assembly yield for D2W hybrid bonding")
    p.add_argument("--config", "-c", required=True, help="Path to modeling config yaml")
    p.add_argument("--mode", "-m", required=True, default="d2w_modeling", help="Mode to load from config (default: d2w_modeling)")
    p.add_argument("--ds_dir", required=True, help="Path to design directory (overrides default input/design_0/)")
    p.add_argument("--bmap", "-b", required=True, help="Path to .bmap file (overrides default input/<INTERFACE>.bmap)")
    p.add_argument("--criticality", "-cr", required=True, help="Path to criticality file (overrides default input/<INTERFACE>_criticality.txt)")
    p.add_argument("--plot", "-plot", default=False, action="store_true", help="Enable plotting of the pad risk map")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output during simulation")
    p.add_argument("--debug", action="store_true", help="Enable debug output when loading config")
    return p.parse_args()



def main():
    args = parse_args()

    # Extract the design input files directory if provided
    input_ds_dir = args.ds_dir
    # Determine .3dbv path
    blox_3dbv_path = input_ds_dir + "/generated_chiplet_definitions.3dbv"
    # Determine .bmap path
    blox_bmap_path = args.__dict__.get("bmap", None)
    # Determine criticality file path
    criticality_path = args.criticality

    # Load config and update with design and ADK parameters (from .3dbv and .bmap)
    cfg = load_base_config(base_config_path=args.config, 
                           input_ds_dir=input_ds_dir,
                           blox_3dbv_path=blox_3dbv_path,
                           blox_bmap_path=blox_bmap_path,
                           mode=args.mode, 
                           debug=args.debug)


    # Plotting flag
    cfg.plot_flag = args.plot
    
    # Create output directory if it doesn't exist
    if not os.path.exists(cfg.OUTPUT_DIR + cfg.INTERFACE):
        os.makedirs(cfg.OUTPUT_DIR + cfg.INTERFACE)


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