#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from roughness_parameters import roughness_parameters


def Cu_expansion_yield_calculator(*,
        cfg,
        wafer,
        TOP_DISH_MEAN_nm: float,
        TOP_DISH_STD_nm: float,
        BOT_DISH_MEAN_nm: float,
        BOT_DISH_STD_nm: float,
        k_et: float,
        k_eb: float,
        T_R: float,
        T_anl: float,
        pad_bitmap_collection: dict,
        pad_yield_flag: bool = False,
    ):
    zeta_0 = k_et * (T_anl - T_R) + k_eb * (T_anl - T_R)
    zeta_1_ = roughness_parameters(
        Asperity_R_m            =   cfg.Asperity_R_m,
        Roughness_sigma_m       =   cfg.Roughness_sigma_m,
        eta_s                   =   cfg.eta_s,
        Roughness_constant      =   cfg.Roughness_constant,
        Adhesion_energy         =   cfg.Adhesion_energy,
        Young_modulus_Pa        =   cfg.Young_modulus_Pa,
        Dielectric_thickness    =   cfg.Dielectric_thickness,
        PITCH_r_um              =   cfg.PITCH_r_um,
        PITCH_c_um              =   cfg.PITCH_c_um,
        PAD_BOT_R_um            =   cfg.PAD_BOT_R_um,
        DISH_0_m                =   cfg.DISH_0_m,
        k_peel                  =   cfg.k_peel,
    )
    zeta_1 = max(zeta_1_, 0)
    upper_limit = - zeta_1
    lower_limit = - zeta_0
    print("upper_limit: ", upper_limit)
    print("lower_limit: ", lower_limit)

    num_critical_pads = pad_bitmap_collection["num_critical_pads"]
    num_redundant_logical_pads = pad_bitmap_collection["num_redundant_logical_pads"]
    redundant_logical_pad_copy = pad_bitmap_collection["redundant_logical_pad_copy"]

    pos_pad, _ = quad(lambda x: norm.pdf(x, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)), lower_limit, upper_limit)
    
    # TODO: When calculate the pad-level yield map, we ignore the pad type difference
    Cu_expansion_pad_yield_map = None
    if pad_yield_flag:
        pass
        # for die, i in zip(wafer.die_list, range(wafer.NUM_DIES)):
            # TODO: upper_limit_map_die_i = Input from Cain
            # TODO: lower_limit_map_die_i = Input from Cain
            # pos_pad_map_i, _ = quad(lambda x: norm.pdf(x, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, 
            #                       scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)), 
            #                       lower_limit_map, upper_limit_map)
            # die.pad_yield_map['Y_Cr'] = pos_pad_map_i
            # Cu_expansion_die_yield_i = pos_pad_map_i multiply together
            # die.die_yield['Y_Cr'] = Cu_expansion_die_yield_i
        # Cu_expansion_die_yield = mean of all die.die_yield['Y_Cr']


    Cu_expansion_die_yield_critical = pos_pad ** num_critical_pads
    Cu_expansion_die_yield_redundant = (1 - (1 - pos_pad) ** redundant_logical_pad_copy) ** num_redundant_logical_pads
    
    Cu_expansion_die_yield = Cu_expansion_die_yield_critical * Cu_expansion_die_yield_redundant

    return min(Cu_expansion_die_yield, 1.0)