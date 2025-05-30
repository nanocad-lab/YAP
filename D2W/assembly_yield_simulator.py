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


def Assembly_Yield_Simulator(
        NUM_DIES = 100, # number of wafers. Too high number will break the memory
        PITCH = 5,  # pitch (um)``
        DIE_W = 8e+3,  # die width (um)
        DIE_L = 12e+3,  # die length (um)
        WAF_R = 150e+3,  # wafer radius (um)
        PAD_TOP_R = 1.0,  # top Cu pad radius (um)
        PAD_BOT_R = 1.5,  # bottom Cu pad radius (um)
        PAD_ARR_ROW = 800,  # number of pads in a row of pad array
        PAD_ARR_COL = 400,  # number of pads in a column of pad array
        VOID_SHAPE = "circle",  # void shape: 'circle' or 'square'
        dice_width = 400,  # width of the dice (um)
        CONTACT_AREA_CONSTRAINT = 0.75,
        CRITICAL_DIST_CONSTRAINT = 0.75,
        RANDOM_MISALIGNMENT_MEAN = 0,  # random misalignment mean (um)
        RANDOM_MISALIGNMENT_STD = 0.005 * 1.5,  # random misalignment standard deviation (um)
        SYSTEM_TRANSLATION_X_MEAN = 0.0,  # systematic translation mean (um) - x direction
        SYSTEM_TRANSLATION_X_STD = 0.1,  # systematic translation standard deviation (um)  - x direction
        SYSTEM_TRANSLATION_Y_MEAN = 0.0,  # systematic translation mean (um) - y direction
        SYSTEM_TRANSLATION_Y_STD = 0.1,  # systematic translation standard deviation (um)  - y direction
        SYSTEM_ROTATION_MEAN = 0.0,  # systematic rotation mean (rad)
        SYSTEM_ROTATION_STD = 1e-6,  # systematic rotation standard deviation (rad)
        BOW_DIFFERENCE_MEAN = 0.0,  # bow difference mean (um)
        BOW_DIFFERENCE_STD = 100,  # bow difference standard deviation (um)
        k_mag = 0.03,
        M_0 = -1,
        D0 = 1e-9,  # Number of particles of all thicknesses per unit area (um^{-1})
        z = 3,  # Exponential factor of the particle thickness distribution
        t_0 = 1,  # The smallest particle thickness (um)
        k_r = 1.8e-4,
        k_r0 = 230,
        k_L = 6.2e-2,
        k_n = 9e-5,
        k_S = 2.7,
        TOP_DISH_MEAN = -3.0,        # Top Cu pad dish mean (nm), negative value means the dish is concave
        TOP_DISH_STD = 0.75,          # Top Cu pad dish standard deviation (nm)
        BOT_DISH_MEAN = -2.8,        # Bottom Cu pad dish mean (nm), negative value means the dish is concave
        BOT_DISH_STD = 0.75,          # Bottom Cu pad dish standard deviation (nm)
        k_et = 0.063,            # Top Cu pad expansion/temp coefficient (nm/K)
        k_eb = 0.065,            # Bottom Cu pad expansion/temp coefficient (nm/K)
        T_R = 25,                # Room temperature (°C)
        T_anl = 300,             # Annealing temperature (°C)
        simulation_times = 5,    # Should be set to 100
        approximate_set = 1,
        Asperity_R = 2e-6,  # Asperity curvature radius (m)
        Roughness_sigma = 1e-9,  # Surface roughness standard deviation (m)
        eta_s = 40e+12,  # Asperity density (m^{-2})
        Roughness_constant = 0.07,
        Adhesion_energy = 1.2,  # Adhesion energy (J/m^2)
        Young_modulus = 73e+9,  # Young's modulus of dielectric material (Pa)
        Dielectric_thickness = 1.5e-6,  # Dielectric thickness (m)
        DISH_0 = 75e-9,  # Reference dish (m)
        k_peel = 6.66e+15  # Peel force constant (N/m^4/K)
):
    PAD_ARR_L = (PAD_ARR_ROW - 1) * PITCH
    PAD_ARR_W = (PAD_ARR_COL - 1) * PITCH
    zeta_0 = k_et * (T_anl - T_R) + k_eb * (T_anl - T_R)    # The total expansion of the Cu pad after annealing (nm)
    zeta_1_ = roughness_parameters(
        Asperity_R=Asperity_R,
        Roughness_sigma=Roughness_sigma,
        eta_s=eta_s,
        Roughness_constant=Roughness_constant,
        Adhesion_energy=Adhesion_energy,
        Young_modulus=Young_modulus,
        Dielectric_thickness=Dielectric_thickness,
        PITCH=PITCH,
        PAD_BOT_R=PAD_BOT_R,
        DISH_0=DISH_0,
        k_peel=k_peel,
    )
    zeta_1 = max(zeta_1_, 0)
    # print("The roughness parameter zeta_0 is: ", zeta_0)
    # print("The roughness parameter zeta_1 is: ", zeta_1)    
    
    single_config_yield_list = []
    for i in range(simulation_times):
        if i % 1 == 0 and simulation_times > 1:
            print("Processing batch {}/{}...".format(i + 1, simulation_times))

        # Initialize the die list
        die_list, base_pad_coords = die_initialize(
            NUM_DIES=NUM_DIES,
            DIE_W=DIE_W,
            DIE_L=DIE_L,
            PAD_ARR_W=PAD_ARR_W,
            PAD_ARR_L=PAD_ARR_L,
            PAD_ARR_ROW=PAD_ARR_ROW,
            PAD_ARR_COL=PAD_ARR_COL,
            PITCH=PITCH,
        )

        # Generate overlay terms
        system_translation_x, system_translation_y, system_rotation, system_magnification, MAX_ALLOWED_MISALIGNMENT = overlay_term_simulator(
            PAD_TOP_R=PAD_TOP_R,
            PAD_BOT_R=PAD_BOT_R,
            PITCH=PITCH,
            CONTACT_AREA_CONSTRAINT=CONTACT_AREA_CONSTRAINT,
            CRITICAL_DIST_CONSTRAINT=CRITICAL_DIST_CONSTRAINT,
            SYSTEM_ROTATION_MEAN=SYSTEM_ROTATION_MEAN,
            SYSTEM_ROTATION_STD=SYSTEM_ROTATION_STD,
            SYSTEM_TRANSLATION_X_MEAN=SYSTEM_TRANSLATION_X_MEAN,
            SYSTEM_TRANSLATION_X_STD=SYSTEM_TRANSLATION_X_STD,
            SYSTEM_TRANSLATION_Y_MEAN=SYSTEM_TRANSLATION_Y_MEAN,
            SYSTEM_TRANSLATION_Y_STD=SYSTEM_TRANSLATION_Y_STD,
            BOW_DIFFERENCE_MEAN=BOW_DIFFERENCE_MEAN,
            BOW_DIFFERENCE_STD=BOW_DIFFERENCE_STD,
            NUM_DIES=NUM_DIES,
            k_mag=k_mag,
            M_0=M_0,
        )
        
        # Generate void defects
        defect_yield_simulator(
            D0=D0,  # Number of particles of all thicknesses per unit area (um^{-1}) on the die
            t_0=t_0,
            z=z,
            k_r=k_r,
            k_r0=k_r0,
            k_n=k_n,
            k_L=k_L,
            k_S=k_S,
            VOID_SHAPE=VOID_SHAPE,
            DIE_W=DIE_W,
            DIE_L=DIE_L,
            NUM_DIES=NUM_DIES,
            die_list=die_list,
        )

       
        # Calculate the overall yield
        yield_list = overall_yield_simulator(
            die_list=die_list,
            NUM_DIES=NUM_DIES,
            DIE_W=DIE_W,
            DIE_L=DIE_L,
            base_pad_coords=base_pad_coords,
            system_translation_x=system_translation_x,
            system_translation_y=system_translation_y,
            system_rotation=system_rotation,
            system_magnification=system_magnification,
            MAX_ALLOWED_MISALIGNMENT=MAX_ALLOWED_MISALIGNMENT,
            zeta_0=zeta_0,
            zeta_1=zeta_1,
            PAD_ARR_W=PAD_ARR_W,
            PAD_ARR_L=PAD_ARR_L,
            TOP_DISH_MEAN=TOP_DISH_MEAN,
            TOP_DISH_STD=TOP_DISH_STD,
            BOT_DISH_MEAN=BOT_DISH_MEAN,
            BOT_DISH_STD=BOT_DISH_STD,
            PITCH=PITCH,
            PAD_TOP_R=PAD_TOP_R,
            RANDOM_MISALIGNMENT_MEAN=RANDOM_MISALIGNMENT_MEAN,
            RANDOM_MISALIGNMENT_STD=RANDOM_MISALIGNMENT_STD,
            approximate_set=approximate_set,
        )
        single_config_yield_list.append(yield_list)
        
        del die_list
    if simulation_times > 1:
        print("The batch yield list is: ", single_config_yield_list)
    assembly_yield = np.mean(single_config_yield_list)
    print("The assembly yield is {:.2f}%.".format(assembly_yield * 100))

    return assembly_yield, single_config_yield_list