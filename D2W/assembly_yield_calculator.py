#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Author: Zhichao Chen
#### Date: Feb 9, 2026

import numpy as np
import time
import matplotlib.pyplot as plt
from YAP.D2W.wafer_die_stack_initialization import die_initialize
from overlay_yield_calculator import stack_overlay_yield_calculator
from defect_yield_calculator import stack_defect_yield_calculator
from Cu_expansion_yield_calculator import stack_stress_yield_calculator
from utils.util import DieStack
from esd_hybrid import pad_esd_yield_map_generator




def Assembly_Yield_Calculator(
    input_args: dict,
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
):  
    '''
    This function calculates the die-stack-level yield
    '''
    start_time = time.time()

    # Initialize the die stack
    die_stack = DieStack(
        cfg_dict                    =   cfg_dict,
        pad_bitmap_collection_dict  =   pad_bitmap_collection_dict,
        mode                        =   input_args['mode'],
        base_pad_coords_flag        =   True,
    )
    
    valid_pad_mask_dict = {}
    for interface, pad_bitmap_collection in pad_bitmap_collection_dict.items():
        valid_pad_mask_dict[interface] = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)

    
    # Calculate the overlay yield
    stack_overlay_yield_calculator(
        cfg_dict            =       cfg_dict,
        die_stack           =       die_stack,
    )


    # Calculate the defect yield
    stack_defect_yield_calculator(
        cfg_dict            =       cfg_dict,
        die_stack           =       die_stack,
    )

    # Calculate the Cu expansion yield
    stack_stress_yield_calculator(
        cfg_dict                    =   cfg_dict,
        die_stack                   =   die_stack,
        pad_bitmap_collection_dict  =   pad_bitmap_collection_dict,
        valid_pad_mask_dict         =   valid_pad_mask_dict,
    )

    esd_start_time = time.time()
    # Calculate the ESD yield
    esd_valid_pad_yield_vec, _, _ = pad_esd_yield_map_generator(
        cfg                   = cfg,
        pad_coords_um         = valid_die_pad_coords,
        pad_size_um           = cfg.PAD_TOP_R_um * 2,
        pad_pitch_um          = cfg.PITCH_r_um,
        top_die_w_um          = cfg.DIE_W_um,
        top_die_h_um          = cfg.DIE_L_um,
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
    
    die_stack_yield = die_stack.get_die_stack_yield()
    print(f"Calculated die stack yield: {die_stack_yield:.6f}")

    del die_stack