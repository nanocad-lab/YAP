#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from utils.util import *
import time
import argparse
import secrets
from assembly_yield_simulator import Assembly_Yield_Simulator
from utils.interface_reuse import (
    copy_representative_bitmap_outputs,
    copy_representative_simulation_outputs,
    format_group_summary,
    group_raw_identical_interfaces,
    has_reused_interfaces,
    write_group_metadata,
    write_per_interface_yield_file,
)


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
    args.seed_run_base = secrets.randbits(63)

    # Extract the design input files directory if provided
    input_ds_dir = args.ds_dir
    # Determine .3dbv path (chiplet definitions)
    _3dbv_path = input_ds_dir + "/generated_chiplet_definitions.3dbv"
    # Determine .3dbx path (stack config)
    _3dbx_path = input_ds_dir + "/generated_stack_config.3dbx"
    # Read the config skeleton and update with design parameters
    cfg_skeleton = OmegaConf.load(args.config)[args.mode]

    print(">>>>>> Starting D2W yield simulation for design: {}".format(args.ds_name))
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

    # Verbose flag
    for cfg in cfg_dict.values():
        cfg.verbose = args.verbose

    # Run assembly yield simulation for each interface
    assembly_yield_dict = {}
    bmap_path_dict = {}
    criticality_path_dict = {}
    pad_bitmap_collection_dict = {}
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
        print("Reusing identical interfaces for bitmap generation and simulation:")
        print(format_group_summary(grouped_interfaces))
        metadata_path = write_group_metadata(output_root, grouped_interfaces)
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

    # Step 2: run assembly yield simulator
    print("Running assembly yield simulator over {} die stacks...".format(cfg_skeleton.NUM_DIE_STACKS))
    simulation_start_time = time.time()
    if has_reused_interfaces(grouped_interfaces):
        per_interface_yield_dict = {}
        stack_assembly_yield = 1.0
        for representative, members in grouped_interfaces.items():
            print(
                f">>> Simulating representative interface {representative} (x{len(members)})"
            )
            rep_args = dict(vars(args))
            rep_args["skip_verbose_root_artifacts"] = True
            _, _, rep_yield_dict = Assembly_Yield_Simulator(
                input_args=rep_args,
                cfg_skeleton=cfg_skeleton,
                cfg_dict={representative: cfg_dict[representative]},
                pad_bitmap_collection_dict={representative: pad_bitmap_collection_dict[representative]},
            )
            representative_yield = rep_yield_dict[representative]
            stack_assembly_yield *= representative_yield ** len(members)
            for interface_name in members:
                per_interface_yield_dict[interface_name] = representative_yield
            for duplicate in members[1:]:
                copy_representative_simulation_outputs(
                    output_root=output_root,
                    representative=representative,
                    duplicate=duplicate,
                )

        yield_path = write_per_interface_yield_file(output_root, per_interface_yield_dict)
        print(f"Per-interface simulation yield saved to {yield_path}.")
        if args.verbose:
            note_path = os.path.join(output_root, "collapsed_interface_simulation_note.txt")
            with open(note_path, "w") as f:
                f.write(
                    "Identical-interface reuse was active.\n"
                    "Per-interface yield and average failure-map PNGs were expanded from representative interfaces.\n"
                    "Root-level per-sample failure-vector NPZ artifacts were skipped because they cannot be "
                    "expanded to duplicates without inventing sample-wise correlations.\n"
                )
            print(f"Collapsed simulation note saved to {note_path}.")
    else:
        stack_assembly_yield, _, per_interface_yield_dict = Assembly_Yield_Simulator(
            input_args=vars(args),
            cfg_skeleton=cfg_skeleton,
            cfg_dict=cfg_dict,
            pad_bitmap_collection_dict=pad_bitmap_collection_dict,
        )

    print(f">>> Yield simulation results for {args.ds_name}: {stack_assembly_yield}")
    print("Per-interface simulation yield:")
    for interface_name, interface_yield in per_interface_yield_dict.items():
        print(f">>  {interface_name}: {interface_yield:.6f}")
    print("Simulation finished in {:.2f} seconds.".format(time.time() - simulation_start_time))
    # Total running time
    print(f"Total D2W assembly yield simulation finished in {time.perf_counter() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
