#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from roughness_parameters import roughness_parameters
from debond import debond_dishing_bounds_calculator



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
    num_critical_pads = pad_bitmap_collection["num_critical_pads"]
    num_redundant_logical_pads = pad_bitmap_collection["num_redundant_logical_pads"]
    redundant_logical_pad_copy = pad_bitmap_collection["redundant_logical_pad_copy"]
    for die in wafer.die_list:
        dishing_bound_array = debond_dishing_bounds_calculator(cfg, die.pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
        upper_limit = - dishing_bound_array[:, 0]  # - upper Cu height limits
        lower_limit = - dishing_bound_array[:, 1]  # - lower Cu height limits

        pos_pad, _ = quad(lambda x: norm.pdf(x, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)), lower_limit, upper_limit)
        # TODO: Finish the rest