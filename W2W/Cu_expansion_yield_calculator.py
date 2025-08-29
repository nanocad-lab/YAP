#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm



def Cu_expansion_yield_calculator(top_dish_mean, 
                                  top_dish_std, 
                                  bot_dish_mean, 
                                  bot_dish_std, 
                                  k_et, 
                                  k_eb, 
                                  T_R, 
                                  T_anl, 
                                  zeta_1,
                                  num_critical_pads,
                                  num_redundant_logical_pads,
                                  redundant_logical_pad_copy,
                                  ):
    zeta_0 = k_et * (T_anl - T_R) + k_eb * (T_anl - T_R)
    upper_limit = - zeta_1
    lower_limit = - zeta_0
    print("upper_limit: ", upper_limit)
    print("lower_limit: ", lower_limit)



    pos_pad, _ = quad(lambda x: norm.pdf(x, loc=top_dish_mean + bot_dish_mean, scale=np.sqrt(top_dish_std**2 + bot_dish_std**2)), lower_limit, upper_limit)

    Cu_expansion_die_yield_critical = pos_pad ** num_critical_pads
    Cu_expansion_die_yield_redundant = (1 - (1 - pos_pad) ** redundant_logical_pad_copy) ** num_redundant_logical_pads
    
    Cu_expansion_die_yield = Cu_expansion_die_yield_critical * Cu_expansion_die_yield_redundant

    return min(Cu_expansion_die_yield, 1.0)  # Ensure yield does not exceed 1.0