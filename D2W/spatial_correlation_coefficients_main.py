#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time

from omegaconf import OmegaConf

from spatial_correlation_coefficients_precalculate import (
    Spatial_Correlation_Coefficients_Precalculate,
)
from utils.generate_criticality import DEFAULT_PROFILE, resolve_criticality_path
from utils.util import convert_3dblox_to_pad_bitmap, get_config_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precalculate pad-failure spatial correlation coefficients for D2W hybrid bonding",
    )
    parser.add_argument("--config", "-c", required=True, help="Path to skeleton config YAML file")
    parser.add_argument("--mode", "-m", default="d2w_simulation", help="Mode to load from config (default: d2w_simulation)")
    parser.add_argument("--ds_name", "-d", required=True, help="Name of design (used for output directory naming)")
    parser.add_argument("--ds_dir", required=True, help="Path to design directory")
    parser.add_argument("--num-stack-samples", type=int, default=100, help="Total die-stack samples used to estimate correlation coefficients")
    parser.add_argument("--sim-batch-size", type=int, default=10, help="Batch size for die-stack generation during correlation precalculation")
    parser.add_argument("--distance-interval-um", type=float, default=5000.0, help="Distance span per KDTree processing round")
    parser.add_argument("--bin-width-um", type=float, default=40.0, help="Distance bin width used when calculating phi")
    parser.add_argument("--pair-query-chunk-size", type=int, default=256, help="Number of source pads per KDTree query chunk during pair precomputation")
    parser.add_argument("--max-correlation-distance-um", type=float, default=None, help="Optional maximum pad-to-pad distance to include in the correlation statistics")
    parser.add_argument("--plot", "-plot", action="store_true", help="Save phi-vs-distance plots")
    parser.add_argument("--debug", action="store_true", help="Enable debug output when loading config")
    parser.add_argument(
        "--criticality-profile",
        default=DEFAULT_PROFILE,
        choices=("default", "esd_strict"),
        help=(
            "Which criticality-file profile to use: "
            "'default' tolerates R-1 ESD + mechanical failures for replicated nets; "
            "'esd_strict' tolerates R-1 mechanical failures but 0 ESD failures."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if "simulation" not in args.mode:
        raise ValueError("Spatial correlation precalculation expects a simulation mode, such as d2w_simulation.")
    if args.num_stack_samples <= 0:
        raise ValueError("--num-stack-samples must be positive.")
    if args.sim_batch_size <= 0:
        raise ValueError("--sim-batch-size must be positive.")
    if args.pair_query_chunk_size <= 0:
        raise ValueError("--pair-query-chunk-size must be positive.")

    input_ds_dir = args.ds_dir
    _3dbv_path = os.path.join(input_ds_dir, "generated_chiplet_definitions.3dbv")
    _3dbx_path = os.path.join(input_ds_dir, "generated_stack_config.3dbx")
    cfg_skeleton = OmegaConf.load(args.config)[args.mode]
    cfg_skeleton.NUM_DIE_STACKS = int(args.num_stack_samples)
    cfg_skeleton.SIM_BATCH_SIZE = min(int(args.sim_batch_size), int(args.num_stack_samples))

    print(f">>>>>> Starting spatial correlation precalculation for design: {args.ds_name}")

    cfg_dict = get_config_dict(
        cfg_folder=args.config.rsplit("/", 1)[0],
        cfg_skeleton=cfg_skeleton,
        ds_name=args.ds_name,
        input_ds_dir=input_ds_dir,
        _3dbv_path=_3dbv_path,
        _3dbx_path=_3dbx_path,
        mode=args.mode,
        debug=args.debug,
    )

    for cfg in cfg_dict.values():
        cfg.plot_flag = args.plot
        output_path = os.path.join(cfg.OUTPUT_DIR, args.ds_name, cfg.INTERFACE)
        os.makedirs(output_path, exist_ok=True)

    bmap_path_dict = {}
    criticality_path_dict = {}
    pad_bitmap_collection_dict = {}
    for interface_name, cfg in cfg_dict.items():
        bmap_path_dict[interface_name] = os.path.join(input_ds_dir, f"{cfg.INTERFACE}.bmap")
        criticality_path_dict[interface_name] = str(
            resolve_criticality_path(
                input_dir=input_ds_dir,
                interface_name=cfg.INTERFACE,
                profile=args.criticality_profile,
            )
        )
        if not os.path.exists(criticality_path_dict[interface_name]):
            raise FileNotFoundError(
                f"Criticality file not found for profile '{args.criticality_profile}': "
                f"{criticality_path_dict[interface_name]}"
            )
        pad_bitmap_collection_dict[interface_name] = convert_3dblox_to_pad_bitmap(
            cfg=cfg,
            _bmap_path=bmap_path_dict[interface_name],
            criticality_path=criticality_path_dict[interface_name],
            pad_arrange_pattern=cfg.PAD_ARRANGE_PATTERN,
            input_args=vars(args),
        )

    start_time = time.time()
    Spatial_Correlation_Coefficients_Precalculate(
        input_args=vars(args),
        cfg_skeleton=cfg_skeleton,
        cfg_dict=cfg_dict,
        pad_bitmap_collection_dict=pad_bitmap_collection_dict,
    )
    print(
        "Total time taken for spatial correlation coefficient precalculation: "
        f"{time.time() - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
