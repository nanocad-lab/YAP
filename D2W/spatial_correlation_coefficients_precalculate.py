#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Spatial correlation coefficient precalculation for D2W hybrid bonding
#### Author: Zhichao Chen
#### Updated: Apr 6, 2026

import os
import time

from defect_yield_simulator import defect_yield_simulator
from overlay_yield_simulator import overlay_term_simulator
from spatial_correlation_coefficients import (
    finalize_spatial_correlation_coefficients,
    get_spatial_correlation_coefficients,
    initialize_spatial_correlation_state,
)
from wafer_die_stack_initialization import die_stack_list_initialize


def Spatial_Correlation_Coefficients_Precalculate(
    *,
    input_args: dict,
    cfg_skeleton: object,
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
):
    num_die_stacks = int(cfg_skeleton.NUM_DIE_STACKS)
    sim_batch_size = int(cfg_skeleton.SIM_BATCH_SIZE)
    if num_die_stacks <= 0:
        raise ValueError("NUM_DIE_STACKS must be positive for spatial correlation precalculation.")
    if sim_batch_size <= 0:
        raise ValueError("SIM_BATCH_SIZE must be positive for spatial correlation precalculation.")

    print("Initializing base pad coordinates for correlation precalculation...")
    temp_die_stack_list, base_pad_coords_dict = die_stack_list_initialize(
        cfg_dict=cfg_dict,
        pad_bitmap_collection_dict=pad_bitmap_collection_dict,
        num_stack_samples=1,
        base_pad_coords_flag=True,
        mode="simulation",
    )
    del temp_die_stack_list

    correlation_state_dict = initialize_spatial_correlation_state(
        input_args=input_args,
        cfg_dict=cfg_dict,
        pad_bitmap_collection_dict=pad_bitmap_collection_dict,
        base_pad_coords_dict=base_pad_coords_dict,
        distance_interval_um=float(input_args["distance_interval_um"]),
        bin_width_um=float(input_args["bin_width_um"]),
        pair_query_chunk_size=int(input_args["pair_query_chunk_size"]),
        max_correlation_distance_um=input_args.get("max_correlation_distance_um"),
    )

    processed_stacks = 0
    while processed_stacks < num_die_stacks:
        batch_size = min(sim_batch_size, num_die_stacks - processed_stacks)
        start_time = time.perf_counter()

        die_stack_list = die_stack_list_initialize(
            cfg_dict=cfg_dict,
            pad_bitmap_collection_dict=pad_bitmap_collection_dict,
            num_stack_samples=batch_size,
            mode="simulation",
        )

        overlay_term_simulator(
            cfg_dict=cfg_dict,
            die_stack_list=die_stack_list,
        )
        defect_yield_simulator(
            cfg_dict=cfg_dict,
            die_stack_list=die_stack_list,
        )

        get_spatial_correlation_coefficients(
            cfg_dict=cfg_dict,
            die_stack_list=die_stack_list,
            pad_bitmap_collection_dict=pad_bitmap_collection_dict,
            base_pad_coords_dict=base_pad_coords_dict,
            correlation_state_dict=correlation_state_dict,
            sample_index_offset=processed_stacks,
        )

        processed_stacks += batch_size
        print(
            f"Correlation precalculation progress: {processed_stacks} / {num_die_stacks} "
            f"die stacks processed. Time taken: {time.perf_counter() - start_time:.2f} seconds.",
            end="\r",
        )
        del die_stack_list

    print("\n>>> Spatial correlation precalculation completed. Saving results...")
    results_dict = finalize_spatial_correlation_coefficients(
        input_args=input_args,
        cfg_dict=cfg_dict,
        correlation_state_dict=correlation_state_dict,
    )

    temp_dirs = {
        os.path.join(cfg.OUTPUT_DIR, cfg.DESIGN, "temp")
        for cfg in cfg_dict.values()
    }
    for temp_dir in temp_dirs:
        if not os.path.isdir(temp_dir):
            continue
        for name in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    return results_dict
