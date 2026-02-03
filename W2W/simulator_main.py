#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from utils.util import *
import time
import argparse
from assembly_yield_simulator import Assembly_Yield_Simulator


def parse_args():
    p = argparse.ArgumentParser(description="Simulate assembly yield for W2W hybrid bonding")
    p.add_argument("--config", "-c", required=True, help="Path to skeleton config YAML file")
    p.add_argument("--mode", "-m", required=True, help="Mode to load from config (default: w2w_modeling)")
    p.add_argument("--ds_name", "-d", required=True, help="Name of design (used for output directory naming)")
    p.add_argument("--ds_dir", required=True, help="Path to design directory")
    p.add_argument("--plot", "-plot", default=False, action="store_true", help="Enable plotting of the pad risk map")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output during simulation")
    p.add_argument("--debug", action="store_true", help="Enable debug output when loading config")
    return p.parse_args()

def main():
    args = parse_args()

    # Extract the design input files directory if provided
    input_ds_dir = args.ds_dir
    # Determine .3dbv path (chiplet definitions)
    _3dbv_path = input_ds_dir + "/generated_chiplet_definitions.3dbv"
    # Determine .3dbx path (stack config)
    _3dbx_path = input_ds_dir + "/generated_stack_config.3dbx"
    # Read the config skeleton and update with design parameters
    cfg_skeleton = OmegaConf.load(args.config)[args.mode]

    # Load config and update with design and ADK parameters (from .3dbv and .bmap)
    cfg_dict = get_config_dict(cfg_skeleton=cfg_skeleton, 
                                input_ds_dir=input_ds_dir,
                                _3dbv_path=_3dbv_path,
                                _3dbx_path=_3dbx_path,
                                mode=args.mode, 
                                debug=args.debug)

    # Plotting flag
    for cfg in cfg_dict.values():
        cfg.plot_flag = args.plot
    
    # Create output directory if it doesn't exist
    for cfg in cfg_dict.values():
        output_path = os.path.join(cfg.OUTPUT_DIR, args.ds_name, cfg.INTERFACE)
        os.makedirs(output_path, exist_ok=True)

    # Verbose flag
    for cfg in cfg_dict.values():
        cfg.verbose = args.verbose

    # Run assembly yield simulation for each interface
    assembly_yield_dict = {}
    bmap_path_dict = {}
    criticality_path_dict = {}
    pad_bitmap_collection_dict = {}
    for interface, cfg in cfg_dict.items():
        # Step 1: convert .bmap -> pad bitmap collection
        bmap_path_dict[interface] = os.path.join(input_ds_dir, f"{cfg.INTERFACE}.bmap")
        criticality_path_dict[interface] = os.path.join(input_ds_dir, f"{cfg.INTERFACE}_criticality.txt")
        pad_bitmap_collection_dict[interface] = convert_3dblox_to_pad_bitmap(cfg=cfg,
                                                            _bmap_path=bmap_path_dict[interface],
                                                            criticality_path=criticality_path_dict[interface],
                                                            pad_arrange_pattern=cfg.PAD_ARRANGE_PATTERN)

    # Step 2: run assembly yield simulator
    print("Running assembly yield simulator over {} wafers...".format(cfg_skeleton.NUM_WAFER_STACKS))
    start_time = time.time()
    assembly_yield_dict, _ = Assembly_Yield_Simulator(
        input_args=vars(args),
        cfg_skeleton=cfg_skeleton,
        _3dbv_path=_3dbv_path,
        _3dbx_path=_3dbx_path,
        cfg_dict=cfg_dict,
        pad_bitmap_collection_dict=pad_bitmap_collection_dict,                                             
    )
    
    print(f"Yield simulation results for {args.ds_name}:")
    for interface, yield_rate in assembly_yield_dict.items():
        print("  Interface {}: {:.2f}%".format(interface, yield_rate * 100))
    print("Total time taken: {:.2f} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    main()