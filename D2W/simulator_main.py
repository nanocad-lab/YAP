#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from utils.util import *
import time
import argparse
from assembly_yield_simulator import Assembly_Yield_Simulator


def parse_args():
    p = argparse.ArgumentParser(description="Simulate assembly yield for D2W hybrid bonding")
    p.add_argument("--config", "-c", required=True, help="Path to modeling config yaml")
    p.add_argument("--mode", "-m", default="d2w_modeling", help="Mode to load from config (default: d2w_modeling)")
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



    # Determine .bmap path
    blox_bmap_path = args.bmap

    # Determine criticality path
    criticality_path = args.criticality

    # Step 1: convert .bmap -> pad bitmap collection
    pad_bitmap_collection = convert_3dblox_to_pad_bitmap(cfg=cfg,
                                                        blox_bmap_path=blox_bmap_path,
                                                        criticality_path=criticality_path,
                                                        pad_arrange_pattern=cfg.PAD_ARRANGE_PATTERN)
    
    # Update cfg (same as before)
    update_config_items(cfg=cfg, mode=args.mode)

    # Step 2: run assembly yield simulator
    print("Running assembly yield simulator over {} dies...".format(cfg.NUM_DIES))
    start_time = time.time()
    assembly_yield, _ = Assembly_Yield_Simulator(
        cfg=cfg,
        pad_bitmap_collection=pad_bitmap_collection,                                               
    )
   
    print(f"Assembly yield over {cfg.NUM_DIES} simulations: {assembly_yield*100:.2f}%")
    print("Total time taken: {:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()