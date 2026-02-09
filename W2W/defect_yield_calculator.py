#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Author: Zhichao Chen
#### Date: Oct 8, 2025

"""
Defect yield calculator for the W2W hybrid bonding process.
- Calculate the critical area of the voids and the defects regarding the die and the pad.
- Calculate the defect yield of on die-level and pad-level.
"""

import numpy as np
import sympy as sp
import math
import os


def get_bitmap_bounds(*,
        bitmap: np.ndarray,
        pad_block_size: int
    ):
    # Find the bounds of the non-zero pixels in the bitmap
    rows = np.any(bitmap, axis=1)
    cols = np.any(bitmap, axis=0)

    if not np.any(rows) or not np.any(cols):
        return 0, 0

    top, bottom = np.where(rows)[0][[0, -1]] * pad_block_size
    left, right = np.where(cols)[0][[0, -1]] * pad_block_size

    height = bottom - top + 1
    width = right - left + 1

    return width, height







def pad_defect_yield_map_generator(
    cfg,
    wafer,
    D0: float,
    t_0: float,
    z: float,
    k_r: float,
    k_r0: float,
    k_n: float,
    k_S: float,
    PAD_TOP_R_um: float,
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    pad_yield_flag: bool = False,
    pad_yield_map_sub_factor: int = 1,
):
    def avg_defects_fail_pad_map(cfg, wafer, die, D0, k_S, k_r, k_r0, t_0, z, PAD_TOP_R_um) -> np.ndarray:
        '''
        This function calculate the average number of fatal defects to the pad
        '''
        current_die_pad_array = wafer.base_pad_coords + die.die_center
        pad2waf_center_dist_um = np.sqrt(current_die_pad_array[:, 0]**2 + current_die_pad_array[:, 1]**2)
        

        term1 = 2 * D0 * (z - 1) * k_S * pad2waf_center_dist_um * t_0 ** 0.5 / (2 * z - 3)
        term = k_r * pad2waf_center_dist_um + k_r0
        part1 = PAD_TOP_R_um**2
        part2 = ((z - 1) / (z - 2)) * (term**2) * t_0
        part3 = (4 * (z - 1) / (2 * z - 3)) * term * PAD_TOP_R_um * t_0
        return term1 + np.pi * D0 * (part1 + part2 + part3)

    '''
    Calculate the pad-level defect yield
    '''
    if pad_yield_flag == True:
        glb_defect_pad_yield_min = 1.0  # Initialize to a high value
        glb_defect_pad_yield_max = 0.0  # Initialize to a low value
        # Subsampling the pad yield map to save memory and speed up the calculation
        nr = int(math.ceil(PAD_ARR_ROW / pad_yield_map_sub_factor))
        nc = int(math.ceil(PAD_ARR_COL / pad_yield_map_sub_factor))
        r_idx = np.round(np.linspace(0, PAD_ARR_ROW - 1, nr)).astype(int)
        c_idx = np.round(np.linspace(0, PAD_ARR_COL - 1, nc)).astype(int)
        RR, CC = np.meshgrid(r_idx, c_idx, indexing='ij')   # shape (nr, nc)
        I = RR * PAD_ARR_COL + CC  # linear indices. shape (nr, nc)
        for i, die in enumerate(wafer.die_list):
            avg_defects_fail_pad_map_i = avg_defects_fail_pad_map(cfg, wafer, die, D0, k_S, k_r, k_r0, t_0, z, PAD_TOP_R_um)
            pad_yield_map_i = np.exp(-avg_defects_fail_pad_map_i)
            pad_yield_map_i_sub = pad_yield_map_i.ravel()[I]
            glb_defect_pad_yield_min = min(glb_defect_pad_yield_min, np.nanmin(pad_yield_map_i))
            glb_defect_pad_yield_max = max(glb_defect_pad_yield_max, np.nanmax(pad_yield_map_i))
            die.pad_yield_map['Y_df'] = pad_yield_map_i_sub
            print("Generated pad-level defect yield map for die {}.".format(i))
        wafer.glb_pad_yield_min_max_dict['Y_df'] = (glb_defect_pad_yield_min, glb_defect_pad_yield_max)
        print("Global min of the pad-level defect yield: {}".format(glb_defect_pad_yield_min))
        print("Global max of the pad-level defect yield: {}".format(glb_defect_pad_yield_max))





def stack_defect_yield_calculator(
    cfg_dict: dict,
    waf_stack,
):
    def avg_defects_per_die(cfg, die_list):
        """
        This function calculate the average number of fatal defects to each die on the wafer interface
        return:
            avg_defects_per_die_array: shape (num_dies_per_wafer, )
        """
        DIE_W_um, DIE_L_um = cfg.DIE_W_um, cfg.DIE_L_um
        D0 = cfg.D0
        z = cfg.z
        k_l = cfg.k_L
        t_0 = cfg.t_0
        die_center_array = np.array([die.die_center for die in die_list])
        die2waf_center_dist_um = np.sqrt(die_center_array[:, 0]**2 + die_center_array[:, 1]**2)

        term1 = DIE_W_um * DIE_L_um
        term2 = 4 * (DIE_W_um + DIE_L_um) * (z - 1) /(np.pi * (2 * z - 3)) * k_l * die2waf_center_dist_um * t_0 ** 0.5
        avg_defects_per_die_array = D0 * (term1 + term2)

        return avg_defects_per_die_array

    def die_yield_given_defect_params(avg_defects_per_die_array):
        """
        This function calculate the die yield given the average number of defects per die
        return:
            die_yield_array: shape (num_dies_per_wafer, )
        """
        defect_yield_array = np.exp(-avg_defects_per_die_array)
        return defect_yield_array

    """
    Calculate the pad-level defect yield
    """
    for interface_name, cfg in cfg_dict.items():
        avg_defects_per_die_array = avg_defects_per_die(cfg, waf_stack.interfaces.interface_dict[interface_name].die_list)
        defect_yield_array = die_yield_given_defect_params(avg_defects_per_die_array)
        # Update the die yield in the die objects
        waf_stack.die_yield_list_per_interface_dict[interface_name] = defect_yield_array