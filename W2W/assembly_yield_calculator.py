#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import time
from wafer_die_initialization import wafer_initialize
from overlay_yield_calculator import overlay_yield_calculator
from defect_yield_calculator import defect_yield_calculator
from Cu_expansion_yield_calculator import Cu_expansion_yield_calculator
from roughness_parameters import roughness_parameters




def Assembly_Yield_Calculator(
    cfg,
    pad_bitmap_collection,
):
    zeta_1_ = roughness_parameters(
        Asperity_R_m          =   cfg.Asperity_R_m,
        Roughness_sigma_m     =   cfg.Roughness_sigma_m,
        eta_s               =   cfg.eta_s,
        Roughness_constant  =   cfg.Roughness_constant,
        Adhesion_energy     =   cfg.Adhesion_energy,
        Young_modulus_Pa       =   cfg.Young_modulus_Pa,
        Dielectric_thickness=   cfg.Dielectric_thickness,
        PITCH_um               =   cfg.PITCH_um,
        PAD_BOT_R_um           =   cfg.PAD_BOT_R_um,
        DISH_0_m              =   cfg.DISH_0_m,
        k_peel              =   cfg.k_peel,
    )
    zeta_1 = max(zeta_1_, 0)
    start_time = time.time()
    # Initialize the wafer
    waf_list = wafer_initialize(
        NUM_WAFERS              = 1,
        DIE_W_um                   = cfg.DIE_W_um,
        DIE_L_um                   = cfg.DIE_L_um,
        PAD_ARR_W_um               = cfg.PAD_ARR_W_um,
        PAD_ARR_L_um               = cfg.PAD_ARR_L_um,
        PAD_ARR_ROW             = cfg.PAD_ARR_ROW,
        PAD_ARR_COL             = cfg.PAD_ARR_COL,
        PITCH_um                   = cfg.PITCH_um,
        WAF_R_um                   = cfg.WAF_R_um,
        PAD_TOP_R_um               = cfg.PAD_TOP_R_um,
        PAD_BOT_R_um               = cfg.PAD_BOT_R_um,
        dice_width              = cfg.dice_width,
        pad_bitmap_collection   = pad_bitmap_collection,
    )
    
    wafer = waf_list[0]
    wafer_time = time.time() - start_time
    print("Wafer initialization time: {} seconds.".format(wafer_time))
    # Calculate the overlay yield
    overlay_yield = overlay_yield_calculator(
        PAD_TOP_R_um                  = cfg.PAD_TOP_R_um,
        PAD_BOT_R_um                  = cfg.PAD_BOT_R_um,
        PITCH_um                      = cfg.PITCH_um,
        num_samples                = cfg.num_samples,
        CONTACT_AREA_CONSTRAINT    = cfg.CONTACT_AREA_CONSTRAINT,
        CRITICAL_DIST_CONSTRAINT   = cfg.CRITICAL_DIST_CONSTRAINT,
        SYSTEM_MAGNIFICATION_MEAN_ppm  = cfg.SYSTEM_MAGNIFICATION_MEAN_ppm,
        SYSTEM_MAGNIFICATION_STD_ppm   = cfg.SYSTEM_MAGNIFICATION_STD_ppm,
        SYSTEM_ROTATION_MEAN_rad       = cfg.SYSTEM_ROTATION_MEAN_rad,
        SYSTEM_ROTATION_STD_rad        = cfg.SYSTEM_ROTATION_STD_rad,
        SYSTEM_TRANSLATION_X_MEAN_um  = cfg.SYSTEM_TRANSLATION_X_MEAN_um,
        SYSTEM_TRANSLATION_X_STD_um   = cfg.SYSTEM_TRANSLATION_X_STD_um,
        SYSTEM_TRANSLATION_Y_MEAN_um  = cfg.SYSTEM_TRANSLATION_Y_MEAN_um,
        SYSTEM_TRANSLATION_Y_STD_um   = cfg.SYSTEM_TRANSLATION_Y_STD_um,
        RANDOM_MISALIGNMENT_MEAN_um   = cfg.RANDOM_MISALIGNMENT_MEAN_um,
        RANDOM_MISALIGNMENT_STD_um    = cfg.RANDOM_MISALIGNMENT_STD_um,
        wafer                      = wafer,
        redundant_flag             = cfg.redundant_flag,
    )
    overlay_yield_time = time.time() - start_time - wafer_time
    print("Overlay yield calculation time: {} seconds.".format(overlay_yield_time))
    # Calculate the defect distribution
    defect_yield = defect_yield_calculator(
        cfg                    = cfg,
        WAF_R_um                  = cfg.WAF_R_um,
        D0                     = cfg.D0,
        t_0                    = cfg.t_0,
        z                      = cfg.z,
        k_r                    = cfg.k_r,
        k_r0                   = cfg.k_r0,
        k_n                    = cfg.k_n,
        k_S                    = cfg.k_S,
        k_L                    = cfg.k_L,
        PAD_TOP_R_um              = cfg.PAD_TOP_R_um,
        PITCH_um                  = cfg.PITCH_um,
        PAD_ARR_ROW            = cfg.PAD_ARR_ROW,
        PAD_ARR_COL            = cfg.PAD_ARR_COL,
        PAD_ARR_W_um              = cfg.PAD_ARR_W_um,
        PAD_ARR_L_um              = cfg.PAD_ARR_L_um,
        VOID_SHAPE             = cfg.VOID_SHAPE,
        num_die                = len(wafer.DIE_L_umist),
        dice_width             = cfg.dice_width,
        pad_bitmap_collection  = pad_bitmap_collection,
    )
    defect_yield_time = time.time() - start_time - wafer_time - overlay_yield_time
    print("Defect yield calculation time: {} seconds.".format(defect_yield_time))



    # Calculate the Cu expansion yield
    Cu_expansion_yield = Cu_expansion_yield_calculator(
        TOP_DISH_MEAN_nm          = cfg.TOP_DISH_MEAN_nm,
        TOP_DISH_STD_nm           = cfg.TOP_DISH_STD_nm,
        BOT_DISH_MEAN_nm          = cfg.BOT_DISH_MEAN_nm,
        BOT_DISH_STD_nm           = cfg.BOT_DISH_STD_nm,
        k_et                   = cfg.k_et,
        k_eb                   = cfg.k_eb,
        T_R                    = cfg.T_R,
        T_anl                  = cfg.T_anl,
        zeta_1                 = zeta_1, 
        num_critical_pads      = pad_bitmap_collection["num_critical_pads"],
        num_redundant_logical_pads = pad_bitmap_collection["num_redundant_logical_pads"],
        redundant_logical_pad_copy = pad_bitmap_collection["redundant_logical_pad_copy"],
    )

    Cu_expansion_yield_time = time.time() - start_time - wafer_time - overlay_yield_time - defect_yield_time
    print("Cu expansion yield calculation time: {} seconds.".format(Cu_expansion_yield_time))
    


    assembly_yield = overlay_yield * defect_yield * Cu_expansion_yield
    
    del wafer

    return assembly_yield, overlay_yield, defect_yield, Cu_expansion_yield