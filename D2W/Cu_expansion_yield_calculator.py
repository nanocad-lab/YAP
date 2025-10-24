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
import matplotlib.pyplot as plt
from debond import debond_dishing_bounds_calculator




def pad_Cu_expansion_yield_map_generator(*,
                                  cfg,
                                  die,
                                  TOP_DISH_MEAN_nm: float,
                                  TOP_DISH_STD_nm: float,
                                  BOT_DISH_MEAN_nm: float,
                                  BOT_DISH_STD_nm: float,
                                  pad_bitmap_collection: dict,
                                  pad_yield_flag: bool = False,
                                  ):
    dishing_bound_array = debond_dishing_bounds_calculator(cfg, die.pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
    upper_limits = - dishing_bound_array[:, 0]  # - upper Cu height limits
    lower_limits = - dishing_bound_array[:, 1]  # - lower Cu height limits
    pos_pads, _ = quad(lambda x: norm.pdf(x, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)), lower_limits, upper_limits)
    pad_yield_map = pos_pads.reshape(die.PAD_ARR_ROW, die.PAD_ARR_COL)
    glb_defect_pad_yield_min = min(glb_defect_pad_yield_min, np.nanmin(pad_yield_map))
    glb_defect_pad_yield_max = max(glb_defect_pad_yield_max, np.nanmax(pad_yield_map))
    die.pad_yield_map['Y_ce'] = pad_yield_map
    die.glb_pad_yield_min_max_dict['Y_ce'] = (glb_defect_pad_yield_min, glb_defect_pad_yield_max)
    print("Generated pad-level Cu expansion yield map for die {}.".format(i))

    # Draw the pad yield map
    plt.figure(figsize=(8, 6))
    plt.imshow(
        die.pad_yield_map['Y_ce'],
        cmap='viridis', 
        vmin=die.glb_pad_yield_min_max_dict['Y_ce'][0],
        vmax=die.glb_pad_yield_min_max_dict['Y_ce'][1],
        interpolation='nearest',
        )
    plt.colorbar(label='Pad Cu Expansion Yield')
    plt.xlabel('Pad Column Index')
    plt.ylabel('Pad Row Index')
    plt.show()