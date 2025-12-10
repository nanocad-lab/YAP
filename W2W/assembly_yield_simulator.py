#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import time
import os

from wafer_die_initialization import wafer_initialize
from overlay_yield_simulator import overlay_term_simulator
from defect_yield_simulator import defect_yield_simulator
from roughness_parameters import roughness_parameters
from overall_yield_simulator import overall_yield_simulator


def Assembly_Yield_Simulator(
    cfg,
    pad_bitmap_collection,
):
    # Remove temporary files if any
    for name in os.listdir(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp'):
        file_path = os.path.join(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp', name)
        if os.path.isdir(file_path):
            os.remove(file_path)


    num_sim_epoch = cfg.NUM_WAFERS // cfg.SIM_BATCH_SIZE
    epoch_yield_list = []
    for epoch in range(num_sim_epoch):
        # Record the time
        start_time = time.time()
        # Initialize the wafer
        waf_list = wafer_initialize(
            NUM_WAFERS              = cfg.SIM_BATCH_SIZE,
            DIE_W_um                = cfg.DIE_W_um,
            DIE_L_um                = cfg.DIE_L_um,
            PAD_ARR_W_um            = cfg.PAD_ARR_W_um,
            PAD_ARR_L_um            = cfg.PAD_ARR_L_um,
            PAD_ARR_ROW             = cfg.PAD_ARR_ROW,
            PAD_ARR_COL             = cfg.PAD_ARR_COL,
            PITCH_r_um              = cfg.PITCH_r_um,
            PITCH_c_um              = cfg.PITCH_c_um,
            WAF_R_um                = cfg.WAF_R_um,
            PAD_TOP_R_um            = cfg.PAD_TOP_R_um,
            PAD_BOT_R_um            = cfg.PAD_BOT_R_um,
            dice_width              = cfg.dice_width,
            pad_bitmap_collection   = pad_bitmap_collection,
        )
        # wafer_sample = waf_list[0]
        # Draw the swhole wafer
        # wafer_sample.draw_wafer_die(fig_size=(15, 15))
        # raise NotImplementedError("Stop here for debug.")
        # Generate overlay terms
        system_translation_x_um, system_translation_y_um, system_rotation_rad, system_magnification_ppm, MAX_ALLOWED_MISALIGNMENT_um = overlay_term_simulator(
            cfg                             =       cfg,
            PAD_TOP_R_um                    =       cfg.PAD_TOP_R_um,
            PAD_BOT_R_um                    =       cfg.PAD_BOT_R_um,
            PITCH_r_um                      =       cfg.PITCH_r_um,
            PITCH_c_um                      =       cfg.PITCH_c_um,
            CONTACT_AREA_CONSTRAINT         =       cfg.CONTACT_AREA_CONSTRAINT,
            CRITICAL_DIST_CONSTRAINT        =       cfg.CRITICAL_DIST_CONSTRAINT,
            SYSTEM_ROTATION_MEAN_rad        =       cfg.SYSTEM_ROTATION_MEAN_rad,
            SYSTEM_ROTATION_STD_rad         =       cfg.SYSTEM_ROTATION_STD_rad,
            SYSTEM_TRANSLATION_X_MEAN_um    =       cfg.SYSTEM_TRANSLATION_X_MEAN_um,
            SYSTEM_TRANSLATION_X_STD_um     =       cfg.SYSTEM_TRANSLATION_X_STD_um,
            SYSTEM_TRANSLATION_Y_MEAN_um    =       cfg.SYSTEM_TRANSLATION_Y_MEAN_um,
            SYSTEM_TRANSLATION_Y_STD_um     =       cfg.SYSTEM_TRANSLATION_Y_STD_um,
            BOW_DIFFERENCE_MEAN_um          =       cfg.BOW_DIFFERENCE_MEAN_um,
            BOW_DIFFERENCE_STD_um           =       cfg.BOW_DIFFERENCE_STD_um,
            NUM_WAFER_SAMPLES               =       cfg.SIM_BATCH_SIZE,
            k_mag                           =       cfg.k_mag,
            M_0                             =       cfg.M_0,
        )
        
        # Generate void defects
        defect_yield_simulator(
            WAF_R_um            =       cfg.WAF_R_um,
            D0                  =       cfg.D0,
            t_0                 =       cfg.t_0,
            z                   =       cfg.z,
            k_r                 =       cfg.k_r,
            k_r0                =       cfg.k_r0,
            k_n                 =       cfg.k_n,
            k_L                 =       cfg.k_L,
            k_S                 =       cfg.k_S,
            VOID_SHAPE          =       cfg.VOID_SHAPE,
            NUM_WAFER_SAMPLES   =       cfg.SIM_BATCH_SIZE,
            waf_list            =       waf_list,
        )
        
        # Calculate the overall yield
        yield_list = overall_yield_simulator(
            cfg                             =       cfg,
            waf_list                        =       waf_list,
            WAF_R_um                        =       cfg.WAF_R_um,
            system_translation_x_um         =       system_translation_x_um,
            system_translation_y_um         =       system_translation_y_um,
            system_rotation_rad             =       system_rotation_rad,
            system_magnification_ppm        =       system_magnification_ppm,
            MAX_ALLOWED_MISALIGNMENT_um     =       MAX_ALLOWED_MISALIGNMENT_um,
            PAD_ARR_W_um                    =       cfg.PAD_ARR_W_um,
            PAD_ARR_L_um                    =       cfg.PAD_ARR_L_um,
            PAD_ARR_ROW                     =       cfg.PAD_ARR_ROW,
            PAD_ARR_COL                     =       cfg.PAD_ARR_COL,
            TOP_DISH_MEAN_nm                =       cfg.TOP_DISH_MEAN_nm,
            TOP_DISH_STD_nm                 =       cfg.TOP_DISH_STD_nm,
            BOT_DISH_MEAN_nm                =       cfg.BOT_DISH_MEAN_nm,
            BOT_DISH_STD_nm                 =       cfg.BOT_DISH_STD_nm,
            TILT_X_MEAN_DEG                 =       cfg.TILT_X_MEAN_DEG,
            TILT_X_STD_DEG                  =       cfg.TILT_X_STD_DEG,
            TILT_Y_MEAN_DEG                 =       cfg.TILT_Y_MEAN_DEG,
            TILT_Y_STD_DEG                  =       cfg.TILT_Y_STD_DEG,
            k_et                            =       cfg.k_et,
            k_eb                            =       cfg.k_eb,
            T_R                             =       cfg.T_R,
            T_anl                           =       cfg.T_anl,
            PITCH_r_um                      =       cfg.PITCH_r_um,
            PITCH_c_um                      =       cfg.PITCH_c_um,
            PAD_TOP_R_um                    =       cfg.PAD_TOP_R_um,
            RANDOM_MISALIGNMENT_MEAN_um     =       cfg.RANDOM_MISALIGNMENT_MEAN_um,
            RANDOM_MISALIGNMENT_STD_um      =       cfg.RANDOM_MISALIGNMENT_STD_um,
            approximate_set                 =       cfg.approximate_set,
            pad_bitmap_collection           =       pad_bitmap_collection,
        )
        epoch_yield_list.append(yield_list)
        print(f"Simulation progress: {epoch+1}/{num_sim_epoch} epochs completed.", end='\r')
        
        
        del waf_list

    assembly_yield = np.mean(epoch_yield_list)

    return assembly_yield, epoch_yield_list
