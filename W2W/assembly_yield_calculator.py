#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import time
import pickle
import gzip
from wafer_die_initialization import wafer_initialize
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
        NUM_WAFERS                  = 1,
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
    # Save wafer info
    with gzip.open('wafer_info.pkl.gz', 'wb') as f:
        pickle.dump(wafer, f, protocol=pickle.HIGHEST_PROTOCOL)
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
        redundant_flag                  = cfg.redundant_flag,
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
    defect_yield_time = time.time() - start_time - wafer_init_time - overlay_yield_time
    print("Defect yield calculation time: {} seconds.".format(defect_yield_time))
    # # Draw the die-level defect yield map
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
        k_et                    = cfg.k_et,
        k_eb                    = cfg.k_eb,
        T_R                     = cfg.T_R,
        T_anl                   = cfg.T_anl,
        pad_bitmap_collection   = pad_bitmap_collection,
        pad_yield_flag          = cfg.pad_yield_flag,
    )

    Cu_expansion_yield_time = time.time() - start_time - wafer_init_time - overlay_yield_time - defect_yield_time
    print("Cu expansion yield calculation time: {} seconds.".format(Cu_expansion_yield_time))

    # Calculate the ESD yield
    pad_esd_yield_map_generator(
        cfg                     = cfg,
        wafer                   = wafer,
        pad_bitmap_collection   = pad_bitmap_collection,
        pad_yield_flag          = cfg.pad_yield_flag,
    )
    esd_yield_time = time.time() - start_time - wafer_init_time - overlay_yield_time - defect_yield_time - Cu_expansion_yield_time
    print("ESD yield calculation time: {} seconds.".format(esd_yield_time))


    for die_id, die in enumerate(wafer.die_list):
        die.pad_yield_map['Y_bond'] = die.pad_yield_map['Y_ovl'] * die.pad_yield_map['Y_df'] * die.pad_yield_map['Y_ce'] * die.pad_yield_map['Y_esd']
        risk_map_generator(cfg=cfg, 
                            die_id=die_id,
                            die=die,
                        )
    del wafer
