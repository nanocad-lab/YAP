#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import time
import matplotlib.pyplot as plt
from wafer_die_initialization import die_initialize
from overlay_yield_calculator import overlay_yield_calculator
from defect_yield_calculator import defect_yield_calculator
from Cu_expansion_yield_calculator import Cu_expansion_yield_calculator
from roughness_parameters import roughness_parameters




def Assembly_Yield_Calculator(
    cfg,
    pad_bitmap_collection,
):
    zeta_1_ = roughness_parameters(
        Asperity_R            =       cfg.Asperity_R,
        Roughness_sigma       =       cfg.Roughness_sigma,
        eta_s                 =       cfg.eta_s,
        Roughness_constant    =       cfg.Roughness_constant,
        Adhesion_energy       =       cfg.Adhesion_energy,
        Young_modulus         =       cfg.Young_modulus,
        Dielectric_thickness  =       cfg.Dielectric_thickness,
        PITCH                 =       cfg.PITCH,
        PAD_BOT_R             =       cfg.PAD_BOT_R,
        DISH_0                =       cfg.DISH_0,
        k_peel                =       cfg.k_peel,
    )
    zeta_1 = max(zeta_1_, 0)
    
    # Initialize the die list
    die_list, _ = die_initialize(
        NUM_DIES        =       1,
        DIE_W           =       cfg.DIE_W,
        DIE_L           =       cfg.DIE_L,
        PAD_ARR_W       =       cfg.PAD_ARR_W,
        PAD_ARR_L       =       cfg.PAD_ARR_L,
        PAD_ARR_ROW     =       cfg.PAD_ARR_ROW,
        PAD_ARR_COL     =       cfg.PAD_ARR_COL,
        PITCH           =       cfg.PITCH,
        pad_bitmap_collection = pad_bitmap_collection,  
    )
    die = die_list[0]
    # fig, ax = plt.subplots(figsize=(4, 6))
    # die.draw_die(ax)

    # Calculate the overlay yield
    # Calculate the overlay yield
    overlay_yield = overlay_yield_calculator(
        PAD_TOP_R                    =       cfg.PAD_TOP_R,
        PAD_BOT_R                    =       cfg.PAD_BOT_R,
        PITCH                        =       cfg.PITCH,
        num_samples                  =       cfg.num_samples,
        CONTACT_AREA_CONSTRAINT      =       cfg.CONTACT_AREA_CONSTRAINT,
        CRITICAL_DIST_CONSTRAINT     =       cfg.CRITICAL_DIST_CONSTRAINT,
        SYSTEM_MAGNIFICATION_MEAN    =       cfg.SYSTEM_MAGNIFICATION_MEAN,
        SYSTEM_MAGNIFICATION_STD     =       cfg.SYSTEM_MAGNIFICATION_STD,
        SYSTEM_ROTATION_MEAN         =       cfg.SYSTEM_ROTATION_MEAN,
        SYSTEM_ROTATION_STD          =       cfg.SYSTEM_ROTATION_STD,
        SYSTEM_TRANSLATION_X_MEAN    =       cfg.SYSTEM_TRANSLATION_X_MEAN,
        SYSTEM_TRANSLATION_X_STD     =       cfg.SYSTEM_TRANSLATION_X_STD,
        SYSTEM_TRANSLATION_Y_MEAN    =       cfg.SYSTEM_TRANSLATION_Y_MEAN,
        SYSTEM_TRANSLATION_Y_STD     =       cfg.SYSTEM_TRANSLATION_Y_STD,
        RANDOM_MISALIGNMENT_MEAN     =       cfg.RANDOM_MISALIGNMENT_MEAN,
        RANDOM_MISALIGNMENT_STD      =       cfg.RANDOM_MISALIGNMENT_STD,
        die                          =       die,
        redundant_flag               =       cfg.redundant_flag,
    )
    # Calculate the defect distribution
    start_time = time.time()
    defect_yield = defect_yield_calculator(
        cfg               =       cfg,
        eff_DIE_R         =       cfg.eff_DIE_R,
        D0                =       cfg.D0,
        t_0               =       cfg.t_0,
        z                 =       cfg.z,
        k_r               =       cfg.k_r,
        k_r0              =       cfg.k_r0,
        k_n               =       cfg.k_n,
        k_S               =       cfg.k_S,
        k_L               =       cfg.k_L,
        PAD_TOP_R         =       cfg.PAD_TOP_R,
        PITCH             =       cfg.PITCH,
        PAD_ARR_ROW       =       cfg.PAD_ARR_ROW,
        PAD_ARR_COL       =       cfg.PAD_ARR_COL,
        VOID_SHAPE        =       cfg.VOID_SHAPE,
        PAD_ARR_W         =       cfg.PAD_ARR_W,
        PAD_ARR_L         =       cfg.PAD_ARR_L,
        pad_bitmap_collection  = pad_bitmap_collection,
    )
    print(f"Defect yield calculation took {time.time() - start_time:.2f} seconds")
    # Calculate the Cu expansion yield
    Cu_expansion_yield = Cu_expansion_yield_calculator(
        top_dish_mean    =       cfg.TOP_DISH_MEAN,
        top_dish_std     =       cfg.TOP_DISH_STD,
        bot_dish_mean    =       cfg.BOT_DISH_MEAN,
        bot_dish_std     =       cfg.BOT_DISH_STD,
        k_et             =       cfg.k_et,
        k_eb             =       cfg.k_eb,
        T_R              =       cfg.T_R,
        T_anl            =       cfg.T_anl,
        zeta_1           =       zeta_1,
        pad_bitmap_collection  = pad_bitmap_collection,
    )
    assembly_yield = overlay_yield * defect_yield * Cu_expansion_yield
    
    del die
    return assembly_yield, overlay_yield, defect_yield, Cu_expansion_yield