#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Author: Zhichao Chen
#### Date: Oct 3, 2025

'''
Cu expansion yield calculator for D2W hybrid bonding:
This module contains functions to calculate die-level and pad-level Cu expansion-induced yield 
based on Cu dish distribution and pad layout.
'''

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from roughness_parameters import roughness_parameters




def Cu_expansion_yield_calculator(*,
                                  cfg,
                                  die,
                                  PAD_ARR_ROW: int,
                                  PAD_ARR_COL: int,
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
        Asperity_R_m            =       cfg.Asperity_R_m,
        Roughness_sigma_m       =       cfg.Roughness_sigma_m,
        eta_s                   =       cfg.eta_s,
        Roughness_constant      =       cfg.Roughness_constant,
        Adhesion_energy         =       cfg.Adhesion_energy,
        Young_modulus_Pa        =       cfg.Young_modulus_Pa,
        Dielectric_thickness    =       cfg.Dielectric_thickness,
        PITCH_um                =       cfg.PITCH_um,
        PAD_BOT_R_um            =       cfg.PAD_BOT_R_um,
        DISH_0_m                =       cfg.DISH_0_m,
        k_peel                  =       cfg.k_peel,
    )
    zeta_1 = max(zeta_1_, 0)
    upper_limit = - zeta_1
    lower_limit = - zeta_0
    # print("upper_limit: ", upper_limit)
    # print("lower_limit: ", lower_limit)

    num_critical_pads = pad_bitmap_collection["num_critical_pads"]
    num_redundant_logical_pads = pad_bitmap_collection["num_redundant_logical_pads"]
    redundant_logical_pad_copy = pad_bitmap_collection["redundant_logical_pad_copy"]

    pos_pad, _ = quad(lambda x: norm.pdf(x, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, 
                                         scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)), 
                                         lower_limit, upper_limit
                    )

    Cu_expansion_die_yield_critical = pos_pad ** num_critical_pads
    Cu_expansion_die_yield_redundant = (1 - (1 - pos_pad) ** redundant_logical_pad_copy) ** num_redundant_logical_pads
    
    Cu_expansion_die_yield = Cu_expansion_die_yield_critical * Cu_expansion_die_yield_redundant

    return Cu_expansion_die_yield