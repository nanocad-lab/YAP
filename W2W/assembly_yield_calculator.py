#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import time

from wafer_die_initialization import Die, Wafer, wafer_initialize
from overlay_yield_calculator import overlay_yield_calculator
from defect_yield_calculator import defect_yield_calculator
from Cu_expansion_yield_calculator import Cu_expansion_yield_calculator
from roughness_parameters import roughness_parameters




def Assembly_Yield_Calculator(
        PITCH = 5,  # pitch (um)
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
        RANDOM_MISALIGNMENT_STD = 1e-3,  # random misalignment standard deviation (um)
        SYSTEM_TRANSLATION_X_MEAN = 0.0,  # systematic translation mean (um) - x direction
        SYSTEM_TRANSLATION_X_STD = 1e-2,  # systematic translation standard deviation (um)  - x direction
        SYSTEM_TRANSLATION_Y_MEAN = 0.0,  # systematic translation mean (um) - y direction
        SYSTEM_TRANSLATION_Y_STD = 1e-2,  # systematic translation standard deviation (um)  - y direction
        SYSTEM_ROTATION_MEAN = 0.0,  # systematic rotation mean (rad)
        SYSTEM_ROTATION_STD = 1e-7,  # systematic rotation standard deviation (rad)
        BOW_DIFFERENCE_MEAN = 0.0,  # bow difference mean (um)
        BOW_DIFFERENCE_STD = 30,  # bow difference standard deviation (um)
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
        cluster_para = 0.1,
        TOP_DISH_MEAN = -10.0,        # Top Cu pad dish mean (nm), negative value means the dish is concave
        TOP_DISH_STD = 1.0,          # Top Cu pad dish standard deviation (nm)
        BOT_DISH_MEAN = -10.0,        # Bottom Cu pad dish mean (nm), negative value means the dish is concave
        BOT_DISH_STD = 1.0,          # Bottom Cu pad dish standard deviation (nm)
        k_et = 0.052,            # Top Cu pad expansion/temp coefficient (nm/K)
        k_eb = 0.052,            # Bottom Cu pad expansion/temp coefficient (nm/K)
        T_R = 25,                # Room temperature (°C)
        T_anl = 300,             # Annealing temperature (°C)
        Asperity_R = 2e-6,  # Asperity curvature radius (m)
        Roughness_sigma = 1.0e-9,  # Surface roughness standard deviation (m)
        eta_s = 40e+12,  # Asperity density (m^{-2})
        Roughness_constant = 0.07,
        Adhesion_energy = 1.2,  # Adhesion energy (J/m^2)
        Young_modulus = 73e+9,  # Young's modulus of dielectric material (Pa)
        Dielectric_thickness = 1.5e-6,  # Dielectric thickness (m)
        DISH_0 = 75e-9,  # Reference dish (m)
        k_peel = 6.66e+15  # Peel force constant (N/m^4/K)
):
    # # print every parameter:
    # print("PITCH: ", PITCH)
    # print("DIE_W: ", DIE_W)
    # print("DIE_L: ", DIE_L)
    # print("WAF_R: ", WAF_R)
    # print("PAD_TOP_R: ", PAD_TOP_R)
    # print("PAD_BOT_R: ", PAD_BOT_R)
    # print("PAD_ARR_ROW: ", PAD_ARR_ROW)
    # print("PAD_ARR_COL: ", PAD_ARR_COL)
    # print("VOID_SHAPE: ", VOID_SHAPE)
    # print("dice_width: ", dice_width)
    # print("RANDOM_MISALIGNMENT_MEAN: ", RANDOM_MISALIGNMENT_MEAN)
    # print("RANDOM_MISALIGNMENT_STD: ", RANDOM_MISALIGNMENT_STD)
    # print("SYSTEM_TRANSLATION_X_MEAN: ", SYSTEM_TRANSLATION_X_MEAN)
    # print("SYSTEM_TRANSLATION_X_STD: ", SYSTEM_TRANSLATION_X_STD)
    # print("SYSTEM_TRANSLATION_Y_MEAN: ", SYSTEM_TRANSLATION_Y_MEAN)
    # print("SYSTEM_TRANSLATION_Y_STD: ", SYSTEM_TRANSLATION_Y_STD)
    # print("SYSTEM_ROTATION_MEAN: ", SYSTEM_ROTATION_MEAN)
    # print("SYSTEM_ROTATION_STD: ", SYSTEM_ROTATION_STD)
    # print("BOW_DIFFERENCE_MEAN: ", BOW_DIFFERENCE_MEAN)
    # print("BOW_DIFFERENCE_STD: ", BOW_DIFFERENCE_STD)
    # print("k_mag: ", k_mag)
    # print("M_0: ", M_0)
    # print("D0: ", D0)
    # print("z: ", z)
    # print("t_0: ", t_0)
    # print("k_r: ", k_r)
    # print("k_r0: ", k_r0)
    # print("k_L: ", k_L)
    # print("k_n: ", k_n)
    # print("k_S: ", k_S)
    # print("TOP_DISH_MEAN: ", TOP_DISH_MEAN)
    # print("TOP_DISH_STD: ", TOP_DISH_STD)
    # print("BOT_DISH_MEAN: ", BOT_DISH_MEAN)
    # print("BOT_DISH_STD: ", BOT_DISH_STD)
    # print("k_et: ", k_et)
    # print("k_eb: ", k_eb)
    # print("T_R: ", T_R)
    # print("T_anl: ", T_anl)
    # print("Asperity_R: ", Asperity_R)
    # print("Roughness_sigma: ", Roughness_sigma)
    # print("eta_s: ", eta_s)
    # print("Roughness_constant: ", Roughness_constant)
    # print("Adhesion_energy: ", Adhesion_energy)
    # print("Young_modulus: ", Young_modulus)
    # print("Dielectric_thickness: ", Dielectric_thickness)
    # print("DISH_0: ", DISH_0)
    # print("k_peel: ", k_peel)

    PAD_ARR_L = (PAD_ARR_ROW - 1) * PITCH
    PAD_ARR_W = (PAD_ARR_COL - 1) * PITCH
    SYSTEM_MAGNIFICATION_MEAN = (k_mag * BOW_DIFFERENCE_MEAN + M_0) / 1e6  # systematic magnification mean (ppm)
    SYSTEM_MAGNIFICATION_STD = (k_mag * BOW_DIFFERENCE_STD) ** 2 / 1e6 # systematic magnification standard deviation (ppm)
    L_m = WAF_R
    
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

    start_time = time.time()
    # Initialize the wafer
    waf_list = wafer_initialize(
        NUM_WAFERS=1,
        DIE_W=DIE_W,
        DIE_L=DIE_L,
        PAD_ARR_W=PAD_ARR_W,
        PAD_ARR_L=PAD_ARR_L,
        PAD_ARR_ROW=PAD_ARR_ROW,
        PAD_ARR_COL=PAD_ARR_COL,
        PITCH=PITCH,
        WAF_R=WAF_R,
        PAD_TOP_R=PAD_TOP_R,
        PAD_BOT_R=PAD_BOT_R,
        dice_width=dice_width,
    )
    wafer = waf_list[0]
    # print("Wafer initialization time: {} seconds.".format(time.time() - start_time))
    # Calculate the overlay yield
    overlay_yield = overlay_yield_calculator(
        PAD_TOP_R=PAD_TOP_R,
        PAD_BOT_R=PAD_BOT_R,
        PITCH=PITCH,
        num_samples=10000,
        CONTACT_AREA_CONSTRAINT=CONTACT_AREA_CONSTRAINT,
        CRITICAL_DIST_CONSTRAINT=CRITICAL_DIST_CONSTRAINT,
        SYSTEM_MAGNIFICATION_MEAN=SYSTEM_MAGNIFICATION_MEAN,
        SYSTEM_MAGNIFICATION_STD=SYSTEM_MAGNIFICATION_STD,
        SYSTEM_ROTATION_MEAN=SYSTEM_ROTATION_MEAN,
        SYSTEM_ROTATION_STD=SYSTEM_ROTATION_STD,
        SYSTEM_TRANSLATION_X_MEAN=SYSTEM_TRANSLATION_X_MEAN,
        SYSTEM_TRANSLATION_X_STD=SYSTEM_TRANSLATION_X_STD,
        SYSTEM_TRANSLATION_Y_MEAN=SYSTEM_TRANSLATION_Y_MEAN,
        SYSTEM_TRANSLATION_Y_STD=SYSTEM_TRANSLATION_Y_STD,
        RANDOM_MISALIGNMENT_MEAN=RANDOM_MISALIGNMENT_MEAN,
        RANDOM_MISALIGNMENT_STD=RANDOM_MISALIGNMENT_STD,
        wafer=wafer,
    )
    # Calculate the defect distribution
    defect_yield = defect_yield_calculator(
        WAF_R=WAF_R,
        D0=D0,
        t_0=t_0,
        z=z,
        k_r=k_r,
        k_r0=k_r0,
        k_n=k_n,
        L_m=L_m,
        k_S=k_S,
        k_L=k_L,
        PAD_TOP_R=PAD_TOP_R,
        PITCH=PITCH,
        PAD_ARR_ROW=PAD_ARR_ROW,
        PAD_ARR_COL=PAD_ARR_COL,
        PAD_ARR_W=PAD_ARR_W,
        PAD_ARR_L=PAD_ARR_L,
        VOID_SHAPE=VOID_SHAPE,
        num_die=len(wafer.die_list),
        dice_width=dice_width,
    )
    # Calculate the Cu expansion yield
    Cu_expansion_yield = Cu_expansion_yield_calculator(
        top_dish_mean=TOP_DISH_MEAN,
        top_dish_std=TOP_DISH_STD,
        bot_dish_mean=BOT_DISH_MEAN,
        bot_dish_std=BOT_DISH_STD,
        k_et=k_et,
        k_eb=k_eb,
        T_R=T_R,
        T_anl=T_anl,
        wafer=wafer,
        zeta_1=zeta_1,
    )

    # print("time: ", time.time() - start_time)

    assembly_yield = overlay_yield * defect_yield * Cu_expansion_yield
    
    del wafer

    return assembly_yield, overlay_yield, defect_yield, Cu_expansion_yield