#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import time

from wafer_die_initialization import wafer_initialize
from overlay_yield_simulator import overlay_term_simulator
from defect_yield_simulator import defect_yield_simulator
from roughness_parameters import roughness_parameters
from overall_yield_simulator import overall_yield_simulator


def Assembly_Yield_Simulator(
    cfg,
    pad_bitmap_collection,
):
    zeta_0 = cfg.k_et * (cfg.T_anl - cfg.T_R) + cfg.k_eb * (cfg.T_anl - cfg.T_R)    # The total expansion of the Cu pad after annealing (nm)
    zeta_1_ = roughness_parameters(
        Asperity_R           = cfg.Asperity_R,
        Roughness_sigma      = cfg.Roughness_sigma,
        eta_s                = cfg.eta_s,
        Roughness_constant   = cfg.Roughness_constant,
        Adhesion_energy      = cfg.Adhesion_energy,
        Young_modulus        = cfg.Young_modulus,
        Dielectric_thickness = cfg.Dielectric_thickness,
        PITCH                = cfg.PITCH,
        PAD_BOT_R            = cfg.PAD_BOT_R,
        DISH_0               = cfg.DISH_0,
        k_peel               = cfg.k_peel,
    )
    zeta_1 = max(zeta_1_, 0)

    single_config_yield_list = []

    for i in range(cfg.simulation_times):
        if i % 1 == 0 and cfg.simulation_times > 1:
            print("Processing batch {}/{}...".format(i + 1, cfg.simulation_times))
        # Record the time
        start_time = time.time()
        # Initialize the wafer
        waf_list = wafer_initialize(
            NUM_WAFERS           = cfg.NUM_WAFERS,
            DIE_W                = cfg.DIE_W,
            DIE_L                = cfg.DIE_L,
            PAD_ARR_W            = cfg.PAD_ARR_W,
            PAD_ARR_L            = cfg.PAD_ARR_L,
            PAD_ARR_ROW          = cfg.PAD_ARR_ROW,
            PAD_ARR_COL          = cfg.PAD_ARR_COL,
            PITCH                = cfg.PITCH,
            WAF_R                = cfg.WAF_R,
            PAD_TOP_R            = cfg.PAD_TOP_R,
            PAD_BOT_R            = cfg.PAD_BOT_R,
            dice_width           = cfg.dice_width,
            pad_bitmap_collection= pad_bitmap_collection,
        )

        # Record the time
        end_time = time.time()
        # print("Time taken to initialize the wafer: {:.2f} seconds".format(end_time - start_time))
        # Generate overlay terms
        system_translation_x, system_translation_y, system_rotation, system_magnification, MAX_ALLOWED_MISALIGNMENT = overlay_term_simulator(
            PAD_TOP_R                   =       cfg.PAD_TOP_R,
            PAD_BOT_R                   =       cfg.PAD_BOT_R,
            PITCH                       =       cfg.PITCH,
            CONTACT_AREA_CONSTRAINT     =       cfg.CONTACT_AREA_CONSTRAINT,
            CRITICAL_DIST_CONSTRAINT    =       cfg.CRITICAL_DIST_CONSTRAINT,
            SYSTEM_ROTATION_MEAN        =       cfg.SYSTEM_ROTATION_MEAN,
            SYSTEM_ROTATION_STD         =       cfg.SYSTEM_ROTATION_STD,
            SYSTEM_TRANSLATION_X_MEAN   =       cfg.SYSTEM_TRANSLATION_X_MEAN,
            SYSTEM_TRANSLATION_X_STD    =       cfg.SYSTEM_TRANSLATION_X_STD,
            SYSTEM_TRANSLATION_Y_MEAN   =       cfg.SYSTEM_TRANSLATION_Y_MEAN,
            SYSTEM_TRANSLATION_Y_STD    =       cfg.SYSTEM_TRANSLATION_Y_STD,
            BOW_DIFFERENCE_MEAN         =       cfg.BOW_DIFFERENCE_MEAN,
            BOW_DIFFERENCE_STD          =       cfg.BOW_DIFFERENCE_STD,
            NUM_WAFERS                  =       cfg.NUM_WAFERS,
            k_mag                       =       cfg.k_mag,
            M_0                         =       cfg.M_0,
        )
        
        # Generate void defects
        defect_yield_simulator(
            WAF_R           =       cfg.WAF_R,
            D0              =       cfg.D0,
            t_0             =       cfg.t_0,
            z               =       cfg.z,
            k_r             =       cfg.k_r,
            k_r0            =       cfg.k_r0,
            k_n             =       cfg.k_n,
            k_L             =       cfg.k_L,
            k_S             =       cfg.k_S,
            VOID_SHAPE      =       cfg.VOID_SHAPE,
            NUM_WAFERS      =       cfg.NUM_WAFERS,
            waf_list        =       waf_list,
        )
        # Calculate the overall yield
        yield_list = overall_yield_simulator(
            waf_list                    =       waf_list,
            WAF_R                       =       cfg.WAF_R,
            system_translation_x        =       system_translation_x,
            system_translation_y        =       system_translation_y,
            system_rotation             =       system_rotation,
            system_magnification        =       system_magnification,
            MAX_ALLOWED_MISALIGNMENT    =       MAX_ALLOWED_MISALIGNMENT,
            zeta_0                      =       zeta_0,
            zeta_1                      =       zeta_1,
            PAD_ARR_W                   =       cfg.PAD_ARR_W,
            PAD_ARR_L                   =       cfg.PAD_ARR_L,
            PAD_ARR_ROW                 =       cfg.PAD_ARR_ROW,
            PAD_ARR_COL                 =       cfg.PAD_ARR_COL,
            TOP_DISH_MEAN               =       cfg.TOP_DISH_MEAN,
            TOP_DISH_STD                =       cfg.TOP_DISH_STD,
            BOT_DISH_MEAN               =       cfg.BOT_DISH_MEAN,
            BOT_DISH_STD                =       cfg.BOT_DISH_STD,
            k_et                        =       cfg.k_et,
            k_eb                        =       cfg.k_eb,
            T_R                         =       cfg.T_R,
            T_anl                       =       cfg.T_anl,
            PITCH                       =       cfg.PITCH,
            PAD_TOP_R                   =       cfg.PAD_TOP_R,
            RANDOM_MISALIGNMENT_MEAN    =       cfg.RANDOM_MISALIGNMENT_MEAN,
            RANDOM_MISALIGNMENT_STD     =       cfg.RANDOM_MISALIGNMENT_STD,
            redundant_survival_ratio    =       cfg.redundant_survival_ratio,
            approximate_set             =       cfg.approximate_set,
            redundant_flag              =       cfg.redundant_flag,
            pad_bitmap_collection       =       pad_bitmap_collection,
        )
        single_config_yield_list.append(yield_list)
        
        del waf_list
    if cfg.simulation_times > 1:
        print("The batch yield list is: ", single_config_yield_list)
    assembly_yield = np.mean(single_config_yield_list)
    print("The assembly yield is {:.2f}%.".format(assembly_yield * 100))

    return assembly_yield, single_config_yield_list
