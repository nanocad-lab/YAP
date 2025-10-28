#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Author: Zhichao Chen
#### Date: Oct 23, 2025

import numpy as np
import time
import matplotlib.pyplot as plt
from wafer_die_initialization import die_initialize
from overlay_yield_calculator import pad_overlay_yield_map_generator
from defect_yield_calculator import pad_defect_yield_map_generator
from Cu_expansion_yield_calculator import pad_Cu_expansion_yield_map_generator
from utils.util import risk_map_generator
from esd_hybrid import pad_esd_yield_map_generator




def Pad_Yield_Map_Generator(
    cfg,
    pad_bitmap_collection: dict,
):  
    '''
    This function calculates the pad-level yield map for a single die
    '''
    # Initialize the die list
    die_list, _ = die_initialize(
        NUM_DIES            =       1,
        DIE_W_um            =       cfg.DIE_W_um,
        DIE_L_um            =       cfg.DIE_L_um,
        PAD_ARR_W_um        =       cfg.PAD_ARR_W_um,
        PAD_ARR_L_um        =       cfg.PAD_ARR_L_um,
        PAD_ARR_ROW         =       cfg.PAD_ARR_ROW,
        PAD_ARR_COL         =       cfg.PAD_ARR_COL,
        PITCH_r_um          =       cfg.PITCH_r_um,
        PITCH_c_um          =       cfg.PITCH_c_um,
        PAD_TOP_R_um        =       cfg.PAD_TOP_R_um,
        PAD_BOT_R_um        =       cfg.PAD_BOT_R_um,
        pad_bitmap_collection = pad_bitmap_collection,  
        pad_yield_flag      =       cfg.pad_yield_flag,
    )
    die = die_list[0]
    # fig, ax = plt.subplots(figsize=(4, 6))
    # die.draw_die(ax)

    # Calculate the overlay yield
    overlay_pad_yield_map = pad_overlay_yield_map_generator(
        PAD_TOP_R_um                    =       cfg.PAD_TOP_R_um,
        PAD_BOT_R_um                    =       cfg.PAD_BOT_R_um,
        PAD_ARR_ROW                     =       cfg.PAD_ARR_ROW,
        PAD_ARR_COL                     =       cfg.PAD_ARR_COL,
        PITCH_r_um                      =       cfg.PITCH_r_um,
        PITCH_c_um                      =       cfg.PITCH_c_um,
        num_samples                     =       cfg.num_samples,
        CONTACT_AREA_CONSTRAINT         =       cfg.CONTACT_AREA_CONSTRAINT,
        CRITICAL_DIST_CONSTRAINT        =       cfg.CRITICAL_DIST_CONSTRAINT,
        SYSTEM_MAGNIFICATION_MEAN_ppm   =       cfg.SYSTEM_MAGNIFICATION_MEAN_ppm,
        SYSTEM_MAGNIFICATION_STD_ppm    =       cfg.SYSTEM_MAGNIFICATION_STD_ppm,
        SYSTEM_ROTATION_MEAN_rad        =       cfg.SYSTEM_ROTATION_MEAN_rad,
        SYSTEM_ROTATION_STD_rad         =       cfg.SYSTEM_ROTATION_STD_rad,
        SYSTEM_TRANSLATION_X_MEAN_um    =       cfg.SYSTEM_TRANSLATION_X_MEAN_um,
        SYSTEM_TRANSLATION_X_STD_um     =       cfg.SYSTEM_TRANSLATION_X_STD_um,
        SYSTEM_TRANSLATION_Y_MEAN_um    =       cfg.SYSTEM_TRANSLATION_Y_MEAN_um,
        SYSTEM_TRANSLATION_Y_STD_um     =       cfg.SYSTEM_TRANSLATION_Y_STD_um,
        RANDOM_MISALIGNMENT_MEAN_um     =       cfg.RANDOM_MISALIGNMENT_MEAN_um,
        RANDOM_MISALIGNMENT_STD_um      =       cfg.RANDOM_MISALIGNMENT_STD_um,
        die                             =       die,
        pad_yield_flag                  =       cfg.pad_yield_flag,
        pad_yield_map_sub_factor        =       cfg.pad_yield_map_sub_factor,
    )
    die.pad_yield_map['Y_ovl'] = overlay_pad_yield_map
    # raise Exception("Overlay yield calculation done. Stop execution here for debugging.")

    # Calculate the defect yield
    start_time = time.time()
    defect_pad_yield_map = pad_defect_yield_map_generator(
        cfg               =       cfg,
        D0                =       cfg.D0,
        t_0               =       cfg.t_0,
        z                 =       cfg.z,
        k_r               =       cfg.k_r,
        k_r0              =       cfg.k_r0,
        PAD_TOP_R_um      =       cfg.PAD_TOP_R_um,
        PAD_ARR_ROW       =       cfg.PAD_ARR_ROW,
        PAD_ARR_COL       =       cfg.PAD_ARR_COL,
        die               =       die,
        pad_yield_flag    =       cfg.pad_yield_flag,
        pad_yield_map_sub_factor = cfg.pad_yield_map_sub_factor,
    )
    die.pad_yield_map['Y_df'] = defect_pad_yield_map
    print(f"Defect yield calculation took {time.time() - start_time:.2f} seconds")

    # Calculate the Cu expansion yield
    Cu_expansion_pad_yield_map = pad_Cu_expansion_yield_map_generator(
        cfg                 =       cfg,
        die                 =       die,
        TOP_DISH_MEAN_nm    =       cfg.TOP_DISH_MEAN_nm,
        TOP_DISH_STD_nm     =       cfg.TOP_DISH_STD_nm,
        BOT_DISH_MEAN_nm    =       cfg.BOT_DISH_MEAN_nm,
        BOT_DISH_STD_nm     =       cfg.BOT_DISH_STD_nm,
        k_et                =       cfg.k_et,
        k_eb                =       cfg.k_eb,
        T_R                 =       cfg.T_R,
        T_anl               =       cfg.T_anl,
    )
    die.pad_yield_map['Y_ce'] = Cu_expansion_pad_yield_map

    # Calculate the ESD yield
    esd_pad_yield_map, _, _ = pad_esd_yield_map_generator(
        cfg                   = cfg,
        pad_coords_um         = die.pad_coords,
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
    die.pad_yield_map['Y_esd'] = esd_pad_yield_map

    die.pad_yield_map['Y_bond'] = die.pad_yield_map['Y_ovl'] * die.pad_yield_map['Y_df'] * die.pad_yield_map['Y_ce'] * die.pad_yield_map['Y_esd']
    risk_map_generator(cfg=cfg, 
                        die=die,
                    )
    del die
    return die.pad_yield_map['Y_bond'] 