#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from utils.util import *
import time
import argparse
from assembly_yield_calculator import Pad_Yield_Map_Generator
from utils.interface_reuse import (
    copy_representative_bitmap_outputs,
    copy_representative_risk_outputs,
    format_group_summary,
    group_risk_equivalent_interfaces,
    group_raw_identical_interfaces,
    has_reused_interfaces,
    write_group_metadata,
)


def parse_args():
    p = argparse.ArgumentParser(description="Simulate assembly yield for D2W hybrid bonding")
    p.add_argument("--config", "-c", required=True, help="Path to modeling config yaml")
    p.add_argument("--mode", "-m", required=True, default="d2w_modeling", help="Mode to load from config (default: d2w_modeling)")
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
    assert os.path.exists(input_ds_dir), f"Design input directory not found at {input_ds_dir}"
    # Determine .3dbv path (chiplet definition file)
    _3dbv_path = input_ds_dir + "/generated_chiplet_definitions.3dbv"
    assert os.path.exists(_3dbv_path), f"3DBV file not found at {_3dbv_path}"
    # Determine .3dbx path (stack configuration file)
    _3dbx_path = input_ds_dir + "/generated_stack_config.3dbx"
    assert os.path.exists(_3dbx_path), f"3DBX file not found at {_3dbx_path}"
    # Read the config skeleton and update with design parameters
    cfg_skeleton = OmegaConf.load(args.config)[args.mode]

    print(">>>>>> Starting D2W pad-level risk map calculation for design: {}".format(args.ds_name))

    start_time = time.perf_counter()
    # Load config and update with design and ADK parameters (from .3dbv and .bmap)
    cfg_dict = get_config_dict(cfg_folder=args.config.rsplit('/', 1)[0],
                                cfg_skeleton=cfg_skeleton, 
                                ds_name=args.ds_name,
                                input_ds_dir=input_ds_dir,
                                _3dbv_path=_3dbv_path,
                                _3dbx_path=_3dbx_path,
                                mode=args.mode, 
                                debug=args.debug)
    cfg_loading_time = time.perf_counter() - start_time
    print(f"Config loading and processing finished in {cfg_loading_time:.2f} seconds.")

    # Plotting flag
    for cfg in cfg_dict.values():
        cfg.plot_flag = args.plot
    
    # Create output directory if it doesn't exist
    for cfg in cfg_dict.values():
        output_path = os.path.join(cfg.OUTPUT_DIR, args.ds_name, cfg.INTERFACE)
        os.makedirs(output_path, exist_ok=True)

    bmap_path_dict = {}
    criticality_path_dict = {}
    pad_bitmap_collection_dict = {}
    # Precompute the file paths once so we can collapse identical interfaces
    for interface, cfg in cfg_dict.items():
        bmap_path_dict[interface] = os.path.join(input_ds_dir, f"{cfg.INTERFACE}.bmap")
        criticality_path_dict[interface] = os.path.join(input_ds_dir, f"{cfg.INTERFACE}_criticality.txt")

    grouped_interfaces = group_raw_identical_interfaces(
        cfg_dict=cfg_dict,
        bmap_path_dict=bmap_path_dict,
        criticality_path_dict=criticality_path_dict,
    )
    output_root = os.path.join(next(iter(cfg_dict.values())).OUTPUT_DIR, args.ds_name)
    if has_reused_interfaces(grouped_interfaces):
        print("Reusing identical interfaces for bitmap generation and pad risk map calculation:")
        print(format_group_summary(grouped_interfaces))
        metadata_path = write_group_metadata(
            output_root,
            grouped_interfaces,
            filename="collapsed_bitmap_interface_groups.txt",
        )
        print(f"Collapsed interface groups saved to {metadata_path}.")

    # Step 1: convert .bmap -> pad bitmap collection
    if has_reused_interfaces(grouped_interfaces):
        for representative, members in grouped_interfaces.items():
            rep_cfg = cfg_dict[representative]
            rep_bitmap_collection = convert_3dblox_to_pad_bitmap(
                cfg=rep_cfg,
                _bmap_path=bmap_path_dict[representative],
                criticality_path=criticality_path_dict[representative],
                pad_arrange_pattern=rep_cfg.PAD_ARRANGE_PATTERN,
                input_args=vars(args),
            )
            for interface_name in members:
                pad_bitmap_collection_dict[interface_name] = rep_bitmap_collection
            for duplicate in members[1:]:
                copy_representative_bitmap_outputs(
                    output_root=output_root,
                    representative=representative,
                    duplicate=duplicate,
                )
    else:
        for interface, cfg in cfg_dict.items():
            pad_bitmap_collection_dict[interface] = convert_3dblox_to_pad_bitmap(
                cfg=cfg,
                _bmap_path=bmap_path_dict[interface],
                criticality_path=criticality_path_dict[interface],
                pad_arrange_pattern=cfg.PAD_ARRANGE_PATTERN,
                input_args=vars(args),
            )
    convert_time = time.perf_counter() - start_time - cfg_loading_time
    print("Pad bitmap collection generation finished in {:.2f} seconds.".format(convert_time))

    risk_equivalent_groups = group_risk_equivalent_interfaces(
        cfg_dict=cfg_dict,
        pad_bitmap_collection_dict=pad_bitmap_collection_dict,
    )
    if has_reused_interfaces(risk_equivalent_groups):
        print("Reusing geometry-equivalent interfaces for pad risk map calculation:")
        print(format_group_summary(risk_equivalent_groups))
        risk_metadata_path = write_group_metadata(
            output_root,
            risk_equivalent_groups,
            filename="collapsed_risk_interface_groups.txt",
        )
        print(f"Risk-map-equivalent interface groups saved to {risk_metadata_path}.")

    # Step 2: generate pad-level yield map
    print("Calculating pad-level yield map...\n")
    yield_map_generation_start_time = time.perf_counter()
    if has_reused_interfaces(risk_equivalent_groups):
        for representative, members in risk_equivalent_groups.items():
            print(
                f">>> Calculating pad-level yield maps for representative interface {representative} "
                f"(x{len(members)})"
            )
            Pad_Yield_Map_Generator(
                input_args=vars(args),
                cfg_dict={representative: cfg_dict[representative]},
                pad_bitmap_collection_dict={representative: pad_bitmap_collection_dict[representative]},
            )
            for duplicate in members[1:]:
                copy_representative_risk_outputs(
                    output_root=output_root,
                    representative=representative,
                    duplicate=duplicate,
                )
    else:
        Pad_Yield_Map_Generator(
            input_args=vars(args),
            cfg_dict=cfg_dict,
            pad_bitmap_collection_dict=pad_bitmap_collection_dict,
        )
    print(">>> D2W pad-level risk map calculation completed")
    print(f"Pad yield map generation finished in {time.perf_counter() - yield_map_generation_start_time:.2f} s\n")
    # Total running time
    print(f"Total D2W pad-level risk map calculation finished in {time.perf_counter() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
