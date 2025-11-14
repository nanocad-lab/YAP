#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### Author: Zhichao Chen
#### Date: Oct 3, 2025

'''
Cu expansion yield calculator for D2W hybrid bonding:
This module contains functions to calculate die-level and pad-level Cu expansion-induced yield 
based on Cu dish distribution and pad layout.
'''

import os
import time
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
    valid_pad_mask = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)
    valid_die_pad_coords = die.pad_coords[valid_pad_mask.flatten() == 1]
    
    # if not os.path.exists(cfg.OUTPUT_DIR + cfg.DESIGN + '/' + cfg.DESIGN + "_dishing_bound_array.npy") or cfg.DEBUG:
    #     start_time = time.time()
    #     valid_pad_dishing_bound_array = debond_dishing_bounds_calculator(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
    #     print("Dishing bound calculation time: {:.2f} seconds".format(time.time() - start_time))
    #     np.save(cfg.OUTPUT_DIR + cfg.DESIGN + '/' + cfg.DESIGN + "_dishing_bound_array.npy", valid_pad_dishing_bound_array)
    # else:
    #     print("Loading dishing bound array from file {}".format(cfg.OUTPUT_DIR + cfg.DESIGN + '/' + cfg.DESIGN + "_dishing_bound_array.npy"))
    #     valid_pad_dishing_bound_array = np.load(cfg.OUTPUT_DIR + cfg.DESIGN + '/' + cfg.DESIGN + "_dishing_bound_array.npy")

    start_time = time.time()
    valid_pad_dishing_bound_array = debond_dishing_bounds_calculator(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
    print("Dishing bound calculation time: {:.2f} seconds".format(time.time() - start_time))

    upper_limits_valid_pads = - valid_pad_dishing_bound_array[:, 0] * 2 # - upper Cu height limits
    lower_limits_valid_pads = - valid_pad_dishing_bound_array[:, 1] * 2 # - lower Cu height limits
    print("Max upper limit (nm): {:.2f}, Min upper limit (nm): {:.2f}".format(np.max(upper_limits_valid_pads), np.min(upper_limits_valid_pads)))
    print("Max lower limit (nm): {:.2f}, Min lower limit (nm): {:.2f}".format(np.max(lower_limits_valid_pads), np.min(lower_limits_valid_pads)))
    pos_valid_pads = norm.cdf(upper_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)) - \
                     norm.cdf(lower_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2))
    pad_yield_map = np.full((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), np.nan)
    pad_yield_map[valid_pad_mask == 1] = pos_valid_pads

    # # Draw dishing lower bound as heatmap (use mask to hide non-pad areas)
    # valid_dishing_bound_array_no_nan = valid_pad_dishing_bound_array.copy()
    # lower_bound_min = np.nanmin(valid_dishing_bound_array_no_nan[:, 0])
    # lower_bound_max = np.nanmax(valid_dishing_bound_array_no_nan[:, 0])
    # upper_bound_min = np.nanmin(valid_dishing_bound_array_no_nan[:, 1])
    # upper_bound_max = np.nanmax(valid_dishing_bound_array_no_nan[:, 1])
    
    # print("Dishing Lower Bound Min (nm):", lower_bound_min)
    # print("Dishing Lower Bound Max (nm):", lower_bound_max)
    # print("Dishing Upper Bound Min (nm):", upper_bound_min)
    # print("Dishing Upper Bound Max (nm):", upper_bound_max)
    # # Draw dishing lower bound as histogram (use mask to hide non-pad areas)
    # plt.figure(figsize=(10, 6))
    # plt.hist(
    #     valid_dishing_bound_array_no_nan[:, 0].flatten(),
    #     bins=50,
    #     color='blue',
    #     alpha=0.7
    #     )
    # plt.xlabel('Dishing Lower Bound (nm)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Dishing Lower Bound')
    # plt.show()
    # # Draw dishing upper bound as histogram (use mask to hide non-pad areas)
    # plt.figure(figsize=(10, 6))
    # plt.hist(
    #     valid_dishing_bound_array_no_nan[:, 1].flatten(),
    #     bins=50,
    #     color='green',
    #     alpha=0.7
    #     )
    # plt.xlabel('Dishing Upper Bound (nm)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Dishing Upper Bound')
    # plt.show()


    glb_cu_expansion_pad_yield_min = min(glb_cu_expansion_pad_yield_min, np.nanmin(pad_yield_map))
    glb_cu_expansion_pad_yield_max = max(glb_cu_expansion_pad_yield_max, np.nanmax(pad_yield_map))
    die.glb_pad_yield_min_max_dict['Y_ce'] = (glb_cu_expansion_pad_yield_min, glb_cu_expansion_pad_yield_max)
    print("Cu Expansion Pad Yield Min: {:.6f}".format(glb_cu_expansion_pad_yield_min))
    print("Cu Expansion Pad Yield Max: {:.6f}".format(glb_cu_expansion_pad_yield_max))


    if cfg.plot_flag:
        # Draw pad yield v.s. pad distance to the die center
        pad_distances_um = np.linalg.norm(die.pad_coords, axis=1)  # (num_pads,)
        plt.figure(figsize=(10, 6))
        plt.scatter(
            pad_distances_um[valid_pad_mask.flatten() == 1],
            pad_yield_map[valid_pad_mask == 1],
            c='blue',
            s=8,
            alpha=0.6,
            )
        np.savez(cfg.OUTPUT_DIR + cfg.DESIGN + '/' + cfg.DESIGN + "_cu_expansion_yield_vs_distance_300warp_0d5dish.npz",
                 pad_distances_um=pad_distances_um[valid_pad_mask.flatten() == 1],
                    pad_yields=pad_yield_map[valid_pad_mask == 1],
                    )
        plt.xlabel('Pad Distance to Die Center (um)')
        plt.ylabel('Pad Cu Expansion Yield')
        plt.title('Pad Cu Expansion Yield vs. Pad Distance to Die Center')
        plt.grid(True)
        plt.show()
    

        # Draw the pad yield map
        plt.figure(figsize=(13.5, 6), dpi=300)
        plt.imshow(
            pad_yield_map,
            cmap='viridis', 
            vmin=die.glb_pad_yield_min_max_dict['Y_ce'][0],
            vmax=die.glb_pad_yield_min_max_dict['Y_ce'][1],
            interpolation='nearest',
            )
        cb = plt.colorbar(label='Pad Cu Expansion Yield')
        cb.ax.yaxis.label.set_size(16)
        plt.title('Pad Mechanical Stress Yield Map', fontsize=16)
        plt.xlabel('Pad Column Index', fontsize=16)
        plt.ylabel('Pad Row Index', fontsize=16)
        plt.show()
        raise NotImplementedError("Disabled detailed pad yield map plot to reduce runtime.")



    return pad_yield_map