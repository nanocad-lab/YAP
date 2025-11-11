#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import os
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import time

from wafer_die_initialization import die_initialize
from overlay_yield_simulator import overlay_term_simulator
from defect_yield_simulator import defect_yield_simulator
from overall_yield_simulator import overall_yield_simulator
from spatial_correlation_coefficients import get_spatial_correlation_coefficients


def Assembly_Yield_Simulator(
    cfg,
    pad_bitmap_collection,
):   
    # Initialize the die list (Extract the base pad coordinates seperately for later use, so that a lot of memory can be saved)
    die_list, base_pad_coords = die_initialize(
        NUM_DIES                    =       cfg.NUM_DIES,
        DIE_W_um                    =       cfg.DIE_W_um,
        DIE_L_um                    =       cfg.DIE_L_um,
        PAD_ARR_W_um                =       cfg.PAD_ARR_W_um,
        PAD_ARR_L_um                =       cfg.PAD_ARR_L_um,
        PAD_ARR_ROW                 =       cfg.PAD_ARR_ROW,
        PAD_ARR_COL                 =       cfg.PAD_ARR_COL,
        PITCH_r_um                  =       cfg.PITCH_r_um,
        PITCH_c_um                  =       cfg.PITCH_c_um,
        PAD_TOP_R_um                =       cfg.PAD_TOP_R_um,
        PAD_BOT_R_um                =       cfg.PAD_BOT_R_um,
        pad_bitmap_collection       =       pad_bitmap_collection,
        pad_yield_flag              =       cfg.pad_yield_flag,
    )
    # die_sample = die_list[0]
    # die_sample.draw_die(fig_size=(6, 6))

    # Generate overlay terms
    system_translation_x_um, system_translation_y_um, system_rotation_rad, system_magnification_ppm, MAX_ALLOWED_MISALIGNMENT_um = overlay_term_simulator(
        cfg                            =       cfg,
        PAD_TOP_R_um                   =       cfg.PAD_TOP_R_um,
        PAD_BOT_R_um                   =       cfg.PAD_BOT_R_um,
        PITCH_r_um                     =       cfg.PITCH_r_um,
        PITCH_c_um                     =       cfg.PITCH_c_um,
        CONTACT_AREA_CONSTRAINT     =       cfg.CONTACT_AREA_CONSTRAINT,
        CRITICAL_DIST_CONSTRAINT    =       cfg.CRITICAL_DIST_CONSTRAINT,
        SYSTEM_ROTATION_MEAN_rad        =       cfg.SYSTEM_ROTATION_MEAN_rad,
        SYSTEM_ROTATION_STD_rad         =       cfg.SYSTEM_ROTATION_STD_rad,
        SYSTEM_TRANSLATION_X_MEAN_um   =       cfg.SYSTEM_TRANSLATION_X_MEAN_um,
        SYSTEM_TRANSLATION_X_STD_um    =       cfg.SYSTEM_TRANSLATION_X_STD_um,
        SYSTEM_TRANSLATION_Y_MEAN_um   =       cfg.SYSTEM_TRANSLATION_Y_MEAN_um,
        SYSTEM_TRANSLATION_Y_STD_um    =       cfg.SYSTEM_TRANSLATION_Y_STD_um,
        BOW_DIFFERENCE_MEAN_um         =       cfg.BOW_DIFFERENCE_MEAN_um,
        BOW_DIFFERENCE_STD_um          =       cfg.BOW_DIFFERENCE_STD_um,
        NUM_DIES                    =       cfg.NUM_DIES,
        k_mag                       =       cfg.k_mag,
        M_0                         =       cfg.M_0,
    )
    
    # Generate void defects
    defect_yield_simulator(
        cfg             =       cfg,
        D0              =       cfg.D0,  # Number of particles of all thicknesses per unit area (um^{-1}) on the die
        t_0             =       cfg.t_0,
        z               =       cfg.z,
        k_r             =       cfg.k_r,
        k_r0            =       cfg.k_r0,
        k_n             =       cfg.k_n,
        k_L             =       cfg.k_L,
        k_S             =       cfg.k_S,
        VOID_SHAPE      =       cfg.VOID_SHAPE,
        DIE_W_um        =       cfg.DIE_W_um,
        DIE_L_um        =       cfg.DIE_L_um,
        NUM_DIES        =       cfg.NUM_DIES,
        die_list        =       die_list,
    )

    
    
    # Calculate the overall yield
    yield_list = overall_yield_simulator(
        cfg                             =       cfg,
        die_list                        =       die_list,
        NUM_DIES                        =       cfg.NUM_DIES,
        base_pad_coords                 =       base_pad_coords,
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
        PITCH_r_um                      =       cfg.PITCH_r_um,
        PITCH_c_um                      =       cfg.PITCH_c_um,
        PAD_TOP_R_um                    =       cfg.PAD_TOP_R_um,
        RANDOM_MISALIGNMENT_MEAN_um     =       cfg.RANDOM_MISALIGNMENT_MEAN_um,
        RANDOM_MISALIGNMENT_STD_um      =       cfg.RANDOM_MISALIGNMENT_STD_um,
        approximate_set                 =       cfg.approximate_set,
        pad_bitmap_collection           =       pad_bitmap_collection,
    )
    assembly_yield = np.mean(yield_list)

    return assembly_yield, yield_list