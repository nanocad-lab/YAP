#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Feb 3, 2026

import time
import pickle
import gzip
import numpy as np
from wafer_die_stack_initialization import wafer_stack_list_initialize
from overlay_yield_calculator import pad_overlay_yield_map_generator
from defect_yield_calculator import pad_defect_yield_map_generator
from Cu_expansion_yield_calculator import pad_Cu_expansion_yield_map_generator
from utils.util import risk_map_generator
from esd_hybrid import pad_esd_yield_map_generator





def Pad_Yield_Map_Generator(
    cfg,
    pad_bitmap_collection,
):
    start_time = time.time()
    # Initialize the wafer
    single_waf_list = wafer_initialize(
        NUM_WAFER_SAMPLES           = 1,
        DIE_W_um                    = cfg.DIE_W_um,
        DIE_L_um                    = cfg.DIE_L_um,
        PAD_ARR_W_um                = cfg.PAD_ARR_W_um,
        PAD_ARR_L_um                = cfg.PAD_ARR_L_um,
        PAD_ARR_ROW                 = cfg.PAD_ARR_ROW,
        PAD_ARR_COL                 = cfg.PAD_ARR_COL,
        PITCH_r_um                  = cfg.PITCH_r_um,
        PITCH_c_um                  = cfg.PITCH_c_um,
        WAF_R_um                    = cfg.WAF_R_um,
        PAD_TOP_R_um                = cfg.PAD_TOP_R_um,
        PAD_BOT_R_um                = cfg.PAD_BOT_R_um,
        dice_width                  = cfg.dice_width,
        pad_bitmap_collection       = pad_bitmap_collection,
        pad_yield_flag              = cfg.pad_yield_flag,
    )
    
    wafer = single_waf_list[0]
    valid_pad_mask = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)

    # # Save wafer info
    # with gzip.open('wafer_info.pkl.gz', 'wb') as f:
    #     pickle.dump(wafer, f, protocol=pickle.HIGHEST_PROTOCOL)
    # raise Exception("Wafer info saved. Stop execution here for debugging.")
    print(len(wafer.die_list), "dies initialized on the wafer.")
    wafer_init_time = time.time() - start_time
    print("Wafer initialization time: {} seconds.".format(wafer_init_time))

    # Calculate the overlay yield
    pad_overlay_yield_map_generator(
        cfg                             = cfg,
        PAD_ARR_ROW                     = cfg.PAD_ARR_ROW,
        PAD_ARR_COL                     = cfg.PAD_ARR_COL,
        PAD_TOP_R_um                    = cfg.PAD_TOP_R_um,
        PAD_BOT_R_um                    = cfg.PAD_BOT_R_um,
        PITCH_r_um                      = cfg.PITCH_r_um,
        PITCH_c_um                      = cfg.PITCH_c_um,
        num_samples                     = cfg.num_samples,
        CONTACT_AREA_CONSTRAINT         = cfg.CONTACT_AREA_CONSTRAINT,
        CRITICAL_DIST_CONSTRAINT        = cfg.CRITICAL_DIST_CONSTRAINT,
        SYSTEM_MAGNIFICATION_MEAN_ppm   = cfg.SYSTEM_MAGNIFICATION_MEAN_ppm,
        SYSTEM_MAGNIFICATION_STD_ppm    = cfg.SYSTEM_MAGNIFICATION_STD_ppm,
        SYSTEM_ROTATION_MEAN_rad        = cfg.SYSTEM_ROTATION_MEAN_rad,
        SYSTEM_ROTATION_STD_rad         = cfg.SYSTEM_ROTATION_STD_rad,
        SYSTEM_TRANSLATION_X_MEAN_um    = cfg.SYSTEM_TRANSLATION_X_MEAN_um,
        SYSTEM_TRANSLATION_X_STD_um     = cfg.SYSTEM_TRANSLATION_X_STD_um,
        SYSTEM_TRANSLATION_Y_MEAN_um    = cfg.SYSTEM_TRANSLATION_Y_MEAN_um,
        SYSTEM_TRANSLATION_Y_STD_um     = cfg.SYSTEM_TRANSLATION_Y_STD_um,
        RANDOM_MISALIGNMENT_MEAN_um     = cfg.RANDOM_MISALIGNMENT_MEAN_um,
        RANDOM_MISALIGNMENT_STD_um      = cfg.RANDOM_MISALIGNMENT_STD_um,
        wafer                           = wafer,
        pad_bitmap_collection           = pad_bitmap_collection,
        pad_yield_flag                  = cfg.pad_yield_flag,
        pad_yield_map_sub_factor        = cfg.pad_yield_map_sub_factor,
    )
    overlay_yield_time = time.time() - start_time - wafer_init_time
    print("Overlay yield calculation time: {} seconds.".format(overlay_yield_time))
    # # Draw the die-level overlay yield map
    # wafer.draw_wafer_die(fig_size=(15, 15), draw_pad_yield_map_option='Y_ovl')
    # raise Exception("Debug stop after overlay yield calculation.")

    # Calculate the defect distribution
    pad_defect_yield_map_generator(
        cfg                         = cfg,
        wafer                       = wafer,
        D0                          = cfg.D0,
        t_0                         = cfg.t_0,
        z                           = cfg.z,
        k_r                         = cfg.k_r,
        k_r0                        = cfg.k_r0,
        k_n                         = cfg.k_n,
        k_S                         = cfg.k_S,
        PAD_TOP_R_um                = cfg.PAD_TOP_R_um,
        PAD_ARR_ROW                 = cfg.PAD_ARR_ROW,
        PAD_ARR_COL                 = cfg.PAD_ARR_COL,
        pad_yield_flag              = cfg.pad_yield_flag,
        pad_yield_map_sub_factor    = cfg.pad_yield_map_sub_factor,
    )
    # defect_yield_time = time.time() - start_time - wafer_init_time - overlay_yield_time
    # print("Defect yield calculation time: {} seconds.".format(defect_yield_time))
    # Draw the die-level defect yield map
    # wafer.draw_wafer_die(fig_size=(10, 10), draw_pad_yield_map_option='Y_df')
    # raise Exception("Debug stop after defect yield calculation.")


    # Calculate the Cu expansion yield
    pad_Cu_expansion_yield_map_generator(
        cfg                     = cfg,
        wafer                   = wafer,
        TOP_DISH_MEAN_nm        = cfg.TOP_DISH_MEAN_nm,
        TOP_DISH_STD_nm         = cfg.TOP_DISH_STD_nm,
        BOT_DISH_MEAN_nm        = cfg.BOT_DISH_MEAN_nm,
        BOT_DISH_STD_nm         = cfg.BOT_DISH_STD_nm,
        pad_bitmap_collection   = pad_bitmap_collection,
    )

    # Cu_expansion_yield_time = time.time() - start_time - wafer_init_time - overlay_yield_time - defect_yield_time
    # print("Cu expansion yield calculation time: {} seconds.".format(Cu_expansion_yield_time))
    # wafer.draw_wafer_die(fig_size=(10, 10), draw_pad_yield_map_option='Y_ce')

    # Calculate the ESD yield
    glb_esd_pad_yield_min = 1.0  # Initialize to a high value
    glb_esd_pad_yield_max = 0.0  # Initialize to a low value
    for die_ind, die in enumerate(wafer.die_list):
        die_center_x, die_center_y = die.die_center[0], die.die_center[1]
        esd_pad_yield_map = np.full((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), np.nan)
        if np.abs(die_center_x) < die.DIE_W_um / 2 and np.abs(die_center_y) < die.DIE_L_um / 2:
            # Assume dies in the center will be the first contact point and have higher ESD hazard
            die_pad_coords = wafer.base_pad_coords + die.die_center
            valid_die_pad_coords = die_pad_coords[valid_pad_mask.flatten() == 1]
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
        glb_esd_pad_yield_min = min(glb_esd_pad_yield_min, np.nanmin(die.pad_yield_map['Y_esd']))
        glb_esd_pad_yield_max = max(glb_esd_pad_yield_max, np.nanmax(die.pad_yield_map['Y_esd']))
    wafer.glb_pad_yield_min_max_dict['Y_esd'] = (glb_esd_pad_yield_min, glb_esd_pad_yield_max)
    # esd_yield_time = time.time() - start_time - wafer_init_time - overlay_yield_time - defect_yield_time - Cu_expansion_yield_time
    # print("ESD yield calculation time: {} seconds.".format(esd_yield_time))

    wafer_glb_pad_yield_min = 1.0
    wafer_glb_pad_yield_max = 0.0
    for die_id, die in enumerate(wafer.die_list):
        die.pad_yield_map['Y_bond'] = die.pad_yield_map['Y_ovl'] * die.pad_yield_map['Y_df'] * die.pad_yield_map['Y_ce'] * die.pad_yield_map['Y_esd']
        wafer_glb_pad_yield_min = min(wafer_glb_pad_yield_min, np.nanmin(die.pad_yield_map['Y_bond']))
        wafer_glb_pad_yield_max = max(wafer_glb_pad_yield_max, np.nanmax(die.pad_yield_map['Y_bond']))
    risk_map_generator(cfg=cfg, 
                          wafer=wafer, 
                    )
    wafer.glb_pad_yield_min_max_dict['Y_bond'] = (wafer_glb_pad_yield_min, wafer_glb_pad_yield_max)
    print("wafer pad yield min max for Y_bond:", wafer.glb_pad_yield_min_max_dict['Y_bond'])
    wafer.draw_wafer_die(fig_size=(20, 20), draw_pad_yield_map_option='Y_bond')
    del wafer