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
                                  ):
    glb_cu_expansion_pad_yield_min = 1.0  # Initialize to a high value
    glb_cu_expansion_pad_yield_max = 0.0  # Initialize to a low value
    dishing_bound_array = debond_dishing_bounds_calculator(cfg, die.pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)

    upper_limits = - dishing_bound_array[:, 0] * 2 # - upper Cu height limits
    lower_limits = - dishing_bound_array[:, 1] * 2 # - lower Cu height limits
    pos_pads = norm.cdf(upper_limits, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)) - \
               norm.cdf(lower_limits, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2))
    pad_yield_map = pos_pads.reshape(cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL)
    mask = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1)
    pad_yield_map[~mask] = np.nan

    # Draw dishing lower bound as heatmap (use mask to hide non-pad areas)
    dishing_bound_array_no_nan = dishing_bound_array.copy()
    dishing_bound_array_no_nan[~mask.flatten(), 0] = np.nan
    lower_bound_min = np.nanmin(dishing_bound_array_no_nan[:, 0])
    lower_bound_max = np.nanmax(dishing_bound_array_no_nan[:, 0])
    upper_bound_min = np.nanmin(dishing_bound_array_no_nan[:, 1])
    upper_bound_max = np.nanmax(dishing_bound_array_no_nan[:, 1])
    print("Dishing Lower Bound Min (nm):", lower_bound_min)
    print("Dishing Lower Bound Max (nm):", lower_bound_max)
    print("Dishing Upper Bound Min (nm):", upper_bound_min)
    print("Dishing Upper Bound Max (nm):", upper_bound_max)
    # Draw dishing lower bound as histogram (use mask to hide non-pad areas)
    plt.figure(figsize=(15, 6))
    plt.hist(
        dishing_bound_array_no_nan[:, 0].flatten(),
        bins=50,
        color='blue',
        alpha=0.7
        )
    plt.xlabel('Dishing Lower Bound (nm)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Dishing Lower Bound')
    plt.show()
    # Draw dishing upper bound as histogram (use mask to hide non-pad areas)
    plt.figure(figsize=(15, 6))
    plt.hist(
        dishing_bound_array_no_nan[:, 1].flatten(),
        bins=50,
        color='green',
        alpha=0.7
        )
    plt.xlabel('Dishing Upper Bound (nm)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Dishing Upper Bound')
    plt.show()


    glb_cu_expansion_pad_yield_min = min(glb_cu_expansion_pad_yield_min, np.nanmin(pad_yield_map))
    glb_cu_expansion_pad_yield_max = max(glb_cu_expansion_pad_yield_max, np.nanmax(pad_yield_map))
    die.glb_pad_yield_min_max_dict['Y_ce'] = (glb_cu_expansion_pad_yield_min, glb_cu_expansion_pad_yield_max)

    # Draw the pad yield map
    plt.figure(figsize=(8, 6))
    plt.imshow(
        pad_yield_map,
        cmap='viridis', 
        vmin=die.glb_pad_yield_min_max_dict['Y_ce'][0],
        vmax=die.glb_pad_yield_min_max_dict['Y_ce'][1],
        interpolation='nearest',
        )
    plt.colorbar(label='Pad Cu Expansion Yield')
    plt.xlabel('Pad Column Index')
    plt.ylabel('Pad Row Index')
    plt.show()

    return pad_yield_map