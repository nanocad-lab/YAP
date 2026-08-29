#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import time

from wafer_die_initialization import Die, Wafer, die_initialize
from overlay_yield_simulator import overlay_term_simulator, die_pad_misalignment
from Cu_gap_simulator import Cu_gap_simulator
from defect_yield_simulator import defect_yield_simulator
from roughness_parameters import roughness_parameters
from overall_yield_simulator import overall_yield_simulator
from utils.util import configure_random_seed, format_yield_array


def Assembly_Yield_Simulator(
    cfg,
    pad_bitmap_collection,
):
    configure_random_seed(cfg, announce=False)
    expected_bitmap_shape = (cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL)
    actual_bitmap_shape = pad_bitmap_collection["CRITICAL_PAD_BITMAP"].shape
    if actual_bitmap_shape != expected_bitmap_shape:
        raise ValueError(
            "Pad bitmap shape does not match the current simulation configuration: "
            f"bitmap={actual_bitmap_shape}, expected={expected_bitmap_shape}. "
            "Call update_config_items() before generating the bitmap."
        )
    zeta_0 = cfg.k_et * (cfg.T_anl - cfg.T_R) + cfg.k_eb * (cfg.T_anl - cfg.T_R)    # The total expansion of the Cu pad after annealing (nm)
    zeta_1_ = roughness_parameters(
        Asperity_R_m             =       cfg.Asperity_R_m,
        Roughness_sigma_m        =       cfg.Roughness_sigma_m,
        eta_s                  =       cfg.eta_s,
        Roughness_constant     =       cfg.Roughness_constant,
        Adhesion_energy        =       cfg.Adhesion_energy,
        Young_modulus_Pa          =       cfg.Young_modulus_Pa,
        Dielectric_thickness   =       cfg.Dielectric_thickness,
        PITCH_um                  =       cfg.PITCH_um,
        PAD_BOT_R_um              =       cfg.PAD_BOT_R_um,
        DISH_0_m                 =       cfg.DISH_0_m,
        k_peel                 =       cfg.k_peel,
    )
    zeta_1 = max(zeta_1_, 0) 
    
    single_config_yield_list = []

    for i in range(cfg.simulation_times):
        if i % 1 == 0 and cfg.simulation_times > 1:
            print("Processing batch {}/{}...".format(i + 1, cfg.simulation_times))

        # Initialize the die list
        die_list, base_pad_coords = die_initialize(
            NUM_DIES                =       cfg.NUM_DIES,
            DIE_W_um                   =       cfg.DIE_W_um,
            DIE_L_um                   =       cfg.DIE_L_um,
            PAD_ARR_W_um               =       cfg.PAD_ARR_W_um,
            PAD_ARR_L_um               =       cfg.PAD_ARR_L_um,
            PAD_ARR_ROW             =       cfg.PAD_ARR_ROW,
            PAD_ARR_COL             =       cfg.PAD_ARR_COL,
            PITCH_um                   =       cfg.PITCH_um,
            pad_bitmap_collection   =       pad_bitmap_collection,
            generate_pad_coords     =       (cfg.approximate_set == 1),
        )

        # Generate overlay terms
        system_translation_x_um, system_translation_y_um, system_rotation_rad, system_magnification_ppm, MAX_ALLOWED_MISALIGNMENT = overlay_term_simulator(
            PAD_TOP_R_um                   =       cfg.PAD_TOP_R_um,
            PAD_BOT_R_um                   =       cfg.PAD_BOT_R_um,
            PITCH_um                       =       cfg.PITCH_um,
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
            D0              =       cfg.D0,  # Number of particles of all thicknesses per unit area (um^{-1}) on the die
            t_0             =       cfg.t_0,
            z               =       cfg.z,
            k_r             =       cfg.k_r,
            k_r0            =       cfg.k_r0,
            k_n             =       cfg.k_n,
            k_L             =       cfg.k_L,
            k_S             =       cfg.k_S,
            VOID_SHAPE      =       cfg.VOID_SHAPE,
            DIE_W_um           =       cfg.DIE_W_um,
            DIE_L_um           =       cfg.DIE_L_um,
            NUM_DIES        =       cfg.NUM_DIES,
            die_list        =       die_list,
        )

       
        # Calculate the overall yield
        yield_list = overall_yield_simulator(
            cfg                         =       cfg,
            die_list                    =       die_list,
            NUM_DIES                    =       cfg.NUM_DIES,
            DIE_W_um                       =       cfg.DIE_W_um,
            DIE_L_um                       =       cfg.DIE_L_um,
            base_pad_coords             =       base_pad_coords,
            system_translation_x_um        =       system_translation_x_um,
            system_translation_y_um        =       system_translation_y_um,
            system_rotation_rad            =       system_rotation_rad,
            system_magnification_ppm       =       system_magnification_ppm,
            MAX_ALLOWED_MISALIGNMENT    =       MAX_ALLOWED_MISALIGNMENT,
            zeta_0                      =       zeta_0,
            zeta_1                      =       zeta_1,
            PAD_ARR_W_um                   =       cfg.PAD_ARR_W_um,
            PAD_ARR_L_um                   =       cfg.PAD_ARR_L_um,
            PAD_ARR_ROW                 =       cfg.PAD_ARR_ROW,
            PAD_ARR_COL                 =       cfg.PAD_ARR_COL,
            TOP_DISH_MEAN_nm               =       cfg.TOP_DISH_MEAN_nm,
            TOP_DISH_STD_nm                =       cfg.TOP_DISH_STD_nm,
            BOT_DISH_MEAN_nm               =       cfg.BOT_DISH_MEAN_nm,
            BOT_DISH_STD_nm                =       cfg.BOT_DISH_STD_nm,
            PITCH_um                       =       cfg.PITCH_um,
            PAD_TOP_R_um                   =       cfg.PAD_TOP_R_um,
            RANDOM_MISALIGNMENT_MEAN_um    =       cfg.RANDOM_MISALIGNMENT_MEAN_um,
            RANDOM_MISALIGNMENT_STD_um     =       cfg.RANDOM_MISALIGNMENT_STD_um,
            redundant_survival_ratio    =       cfg.redundant_survival_ratio,
            approximate_set             =       cfg.approximate_set,
            redundant_flag              =       cfg.redundant_flag,
            pad_bitmap_collection       =       pad_bitmap_collection,
        )
        single_config_yield_list.append(yield_list)
        
        del die_list
    if cfg.simulation_times > 1:
        print("The batch yield list is:", format_yield_array(single_config_yield_list))
    assembly_yield = np.mean(single_config_yield_list)
    print("The assembly yield is {:.4f}%.".format(assembly_yield * 100))

    return assembly_yield, single_config_yield_list
