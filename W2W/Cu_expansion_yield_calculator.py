#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Oct 24, 2025

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from debond import debond_dishing_bounds_calculator
import matplotlib.pyplot as plt


def pad_Cu_expansion_yield_map_generator(*,
        cfg,
        wafer,
        TOP_DISH_MEAN_nm: float,
        TOP_DISH_STD_nm: float,
        BOT_DISH_MEAN_nm: float,
        BOT_DISH_STD_nm: float,
        pad_bitmap_collection: dict,
    ):
    glb_cu_expansion_pad_yield_min = 1.0  # Initialize to a high value
    glb_cu_expansion_pad_yield_max = 0.0  # Initialize to a low value
    for i, die in enumerate(wafer.die_list):
        die_pad_coords = wafer.base_pad_coords + die.die_center
        dishing_bound_array = debond_dishing_bounds_calculator(cfg, die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
        upper_limits = - dishing_bound_array[:, 0] * 2 # - upper limits of the sum of top and bottom Cu heights
        lower_limits = - dishing_bound_array[:, 1] * 2 # - lower limits of the sum of top and bottom Cu heights
        pos_pads = norm.cdf(upper_limits, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)) - \
                   norm.cdf(lower_limits, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2))
        pad_yield_map = pos_pads.reshape(cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL)
        mask = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1)
        pad_yield_map[~mask] = np.nan
        glb_cu_expansion_pad_yield_min = min(glb_cu_expansion_pad_yield_min, np.nanmin(pad_yield_map))
        glb_cu_expansion_pad_yield_max = max(glb_cu_expansion_pad_yield_max, np.nanmax(pad_yield_map))
        die.pad_yield_map['Y_ce'] = pad_yield_map
        print("Generated pad-level Cu expansion yield map for die {}.".format(i))
        
    wafer.glb_pad_yield_min_max_dict['Y_ce'] = (glb_cu_expansion_pad_yield_min, glb_cu_expansion_pad_yield_max)
    print("Global min of the pad-level Cu expansion yield: {}".format(glb_cu_expansion_pad_yield_min))
    print("Global max of the pad-level Cu expansion yield: {}".format(glb_cu_expansion_pad_yield_max))