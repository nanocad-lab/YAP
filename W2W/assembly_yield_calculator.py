#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Feb 3, 2026

import time
import pickle
import gzip
import numpy as np
from wafer_die_stack_initialization import WaferStack
from overlay_yield_calculator import stack_overlay_yield_calculator
from defect_yield_calculator import stack_defect_yield_calculator
from Cu_expansion_yield_calculator import stack_stress_yield_calculator
from utils.util import risk_map_generator
from esd_yield_calculator import stack_esd_yield_calculator



def Assembly_Yield_Calculator(
    input_args: dict,
    cfg_skeleton: object,
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
):
    start_time = time.time()

    # Initialize the wafer stack with dies and pads
    waf_stack = WaferStack(
        cfg_dict=cfg_dict,
        pad_bitmap_collection_dict=pad_bitmap_collection_dict,
    )
    waf_stack_init_time = time.time() - start_time
    print("Wafer stack initialization time: {} seconds.".format(waf_stack_init_time))
    
    valid_pad_mask_dict = {}
    for interface, pad_bitmap_collection in pad_bitmap_collection_dict.items():
        valid_pad_mask_dict[interface] = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)

    
    # Calculate the overlay yield
    stack_overlay_yield_calculator(
        cfg_dict                    =   cfg_dict,
        waf_stack                   =   waf_stack
    )
    overlay_yield_time = time.time() - start_time - waf_stack_init_time
    print("Overlay yield calculation time: {} seconds.".format(overlay_yield_time))

    # Calculate the defect distribution
    stack_defect_yield_calculator(
        cfg_dict                    =   cfg_dict,
        waf_stack                   =   waf_stack,
    )
    defect_yield_time = time.time() - start_time - waf_stack_init_time - overlay_yield_time
    print("Defect yield calculation time: {} seconds.".format(defect_yield_time))

    # Calculate the Cu expansion yield
    stack_stress_yield_calculator(
        cfg_dict                    =   cfg_dict,
        waf_stack                   =   waf_stack,
        pad_bitmap_collection_dict  =   pad_bitmap_collection_dict,
        valid_pad_mask_dict         =   valid_pad_mask_dict,
    )

    Cu_expansion_yield_time = time.time() - start_time - waf_stack_init_time - overlay_yield_time - defect_yield_time
    print("Cu expansion yield calculation time: {} seconds.".format(Cu_expansion_yield_time))

    # Calculate the ESD yield
    stack_esd_yield_calculator(
        cfg_dict                    =   cfg_dict,
        waf_stack                   =   waf_stack,
        pad_bitmap_collection_dict  =   pad_bitmap_collection_dict,
    )
            esd_valid_pad_yield_vec, _, _ = pad_esd_yield_map_generator(
                cfg                   = cfg,
                pad_coords_um         = valid_die_pad_coords,
                pad_size_um           = cfg.PAD_TOP_R_um * 2,
                pad_pitch_um          = cfg.PITCH_r_um,
                top_wafer_radius_um   = cfg.WAF_R_um,
                n_tilts               = cfg.n_tilts_samples,
                n_dishes              = cfg.n_dishes_samples,
                tilt_x_mean_deg       = cfg.TILT_X_MEAN_DEG,
                tilt_x_std_deg        = cfg.TILT_X_STD_DEG,
                tilt_y_mean_deg       = cfg.TILT_Y_MEAN_DEG,
                tilt_y_std_deg        = cfg.TILT_Y_STD_DEG,
                top_dish_mean_nm      = cfg.TOP_DISH_MEAN_nm,
                top_dish_std_nm       = cfg.TOP_DISH_STD_nm,
                bot_dish_mean_nm      = cfg.BOT_DISH_MEAN_nm,
                bot_dish_std_nm       = cfg.BOT_DISH_STD_nm,
            )
            esd_pad_yield_map[valid_pad_mask == 1] = esd_valid_pad_yield_vec
        else:
            # For dies not in the center, assign full yield (1.0)
            esd_pad_yield_map[valid_pad_mask == 1] = 1.0
        die.pad_yield_map['Y_esd'] = esd_pad_yield_map

    del waf_stack