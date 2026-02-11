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
import time


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
    valid_pad_mask = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)
    for i, die in enumerate(wafer.die_list):
        die_pad_coords = wafer.base_pad_coords + die.die_center
        valid_die_pad_coords = die_pad_coords[valid_pad_mask.flatten() == 1]
        start_time = time.time()
        valid_dishing_bound_array = debond_dishing_bounds_calculator(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
        print("Dishing bound calculation time for die {}: {:.2f} seconds".format(i, time.time() - start_time))
        
        upper_limits_valid_pads = - valid_dishing_bound_array[:, 0] * 2 # - upper limits of the sum of top and bottom Cu heights
        lower_limits_valid_pads = - valid_dishing_bound_array[:, 1] * 2 # - lower limits of the sum of top and bottom Cu heights
        pos_valid_pads = norm.cdf(upper_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)) - \
                   norm.cdf(lower_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2))
        pad_yield_map = np.full((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), np.nan)
        pad_yield_map[valid_pad_mask == 1] = pos_valid_pads
        
        glb_cu_expansion_pad_yield_min = min(glb_cu_expansion_pad_yield_min, np.nanmin(pad_yield_map))
        glb_cu_expansion_pad_yield_max = max(glb_cu_expansion_pad_yield_max, np.nanmax(pad_yield_map))
        die.pad_yield_map['Y_ce'] = pad_yield_map
        print("Generated pad-level Cu expansion yield map for die {}.".format(i))
        
    wafer.glb_pad_yield_min_max_dict['Y_ce'] = (glb_cu_expansion_pad_yield_min, glb_cu_expansion_pad_yield_max)
    print("Global min of the pad-level Cu expansion yield: {}".format(glb_cu_expansion_pad_yield_min))
    print("Global max of the pad-level Cu expansion yield: {}".format(glb_cu_expansion_pad_yield_max))



def stack_stress_yield_calculator(
        cfg_dict: dict,
        waf_stack,
        pad_bitmap_collection_dict: dict,
        valid_pad_mask_dict: dict,
):
    for interface_name, cfg in cfg_dict.items():
        interface = waf_stack.interfaces.interface_dict[interface_name]
        pad_bitmap_collection = pad_bitmap_collection_dict[interface_name]
        valid_pad_mask = valid_pad_mask_dict[interface_name]

        # Extract the necessary parameters for Cu expansion yield calculation
        TOP_DISH_MEAN_nm, TOP_DISH_STD_nm = cfg.TOP_DISH_MEAN_nm, cfg.TOP_DISH_STD_nm
        BOT_DISH_MEAN_nm, BOT_DISH_STD_nm = cfg.BOT_DISH_MEAN_nm, cfg.BOT_DISH_STD_nm
        CRITICAL_PAD_MASK = pad_bitmap_collection['CRITICAL_PAD_BITMAP'].flatten()
        redundant_net_to_1d_physical_mask = pad_bitmap_collection['redundant_net_to_1d_physical_mask']


        stress_yield_list = []

        for die_ind, die in enumerate(interface.die_list):
            die_pad_coords = interface.base_pad_coords + die.die_center
            valid_die_pad_coords = die_pad_coords[valid_pad_mask.flatten() == 1]
            start_time = time.time()
            valid_dishing_bound_array = debond_dishing_bounds_calculator(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
            print("Dishing bound calculation time for die {}: {:.2f} seconds".format(die_ind, time.time() - start_time))
            upper_limits_valid_pads = - valid_dishing_bound_array[:, 0] * 2 # - upper limits of the sum of top and bottom Cu heights
            lower_limits_valid_pads = - valid_dishing_bound_array[:, 1] * 2 # - lower limits of the sum of top and bottom Cu heights
            pos_valid_pads = norm.cdf(upper_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)) - \
                    norm.cdf(lower_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2))
            # Critical yield is the pos of the critical pads multiplied together
            stress_yield_critical_pads = np.prod(pos_valid_pads[CRITICAL_PAD_MASK == 1])
            stress_yield_redundant_nets = 1.0
            for redundant_net, physical_pad_indices in redundant_net_to_1d_physical_mask.items():
                num_replicas = len(physical_pad_indices)
                stress_yield_redundant_nets *= 1 - (1 - np.prod(pos_valid_pads[physical_pad_indices])) ** num_replicas
            stress_yield = stress_yield_critical_pads * stress_yield_redundant_nets
            stress_yield_list.append(stress_yield)

            break
            
        # Update the die yield list for this interface in the wafer stack
        waf_stack.die_yield_list_per_interface_dict[interface_name]['mechanical'] = np.array(stress_yield_list)
