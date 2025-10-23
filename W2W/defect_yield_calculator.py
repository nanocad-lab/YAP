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
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from pad_bitmap_generation import A_critical_l_across_theta


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

def defect_yield_calculator(
    cfg,
    wafer,
    WAF_R_um: float,
    D0: float,
    t_0: float,
    z: float,
    k_r: float,
    k_r0: float,
    k_n: float,
    k_S: float,
    k_L: float,
    PAD_TOP_R_um: float,
    PITCH_r_um: float,
    PITCH_c_um: float,
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    PAD_ARR_W_um: float,
    PAD_ARR_L_um: float,
    VOID_SHAPE: str,
    num_die: int,
    dice_width: float,
    pad_bitmap_collection: dict,
    pad_yield_flag: bool = False,
    pad_yield_map_sub_factor: int = 1,
):
    # Read the bitmap collection info
    r_mv = sp.symbols("r_mv")
    t_avg = (z - 1) * t_0 / (z - 2)
    pad_block_size = pad_bitmap_collection["pad_block_size"]
    critical_pad_ratio = pad_bitmap_collection["critical_pad_ratio"]
    CRITICAL_PAD_BLOCK_BITMAP = pad_bitmap_collection["CRITICAL_PAD_BLOCK_BITMAP"]
    REDUNDANT_MAIN_PAD_BLOCK_BITMAP = pad_bitmap_collection["REDUNDANT_MAIN_PAD_BLOCK_BITMAP"]
    is_redundant_copy_same_block = pad_bitmap_collection["is_redundant_copy_same_block"]
    CRITICAL_PAD_ARR_W_IND, CRITICAL_PAD_ARR_L_IND = get_bitmap_bounds(bitmap=CRITICAL_PAD_BLOCK_BITMAP, pad_block_size=pad_block_size)
    REDUNDANT_MAIN_PAD_ARR_W_IND, REDUNDANT_MAIN_PAD_ARR_L_IND = get_bitmap_bounds(bitmap=REDUNDANT_MAIN_PAD_BLOCK_BITMAP, pad_block_size=pad_block_size)
    REDUNDANT_MAIN_PAD_ARR_W_um = REDUNDANT_MAIN_PAD_ARR_W_IND * PITCH_c_um
    REDUNDANT_MAIN_PAD_ARR_L_um = REDUNDANT_MAIN_PAD_ARR_L_IND * PITCH_r_um
    CRITICAL_PAD_ARR_W_um = CRITICAL_PAD_ARR_W_IND * PITCH_c_um
    CRITICAL_PAD_ARR_L_um = CRITICAL_PAD_ARR_L_IND * PITCH_r_um
    if is_redundant_copy_same_block:
        EFF_PAD_ARR_W_um = max(CRITICAL_PAD_ARR_W_um, REDUNDANT_MAIN_PAD_ARR_W_um)
        EFF_PAD_ARR_L_um = max(CRITICAL_PAD_ARR_L_um, REDUNDANT_MAIN_PAD_ARR_L_um)
    else:
        EFF_PAD_ARR_W_um = CRITICAL_PAD_ARR_W_um
        EFF_PAD_ARR_L_um = CRITICAL_PAD_ARR_L_um
    # print("EFF_PAD_ARR_W_um: ", EFF_PAD_ARR_W_um, "EFF_PAD_ARR_L_um: ", EFF_PAD_ARR_L_um)


    # TODO: Modify the PITCH_um to PITCH_r_um and PITCH_c_um throughout the codebase and correct all formulas
    def void_critical_area_per_die(PAD_TOP_R_um, r_v, PITCH_c_um, PITCH_r_um, PAD_ARR_ROW, PAD_ARR_COL, PAD_ARR_W_um, PAD_ARR_L_um, VOID_SHAPE):
        N = PAD_ARR_ROW * PAD_ARR_COL
        r_p = PAD_TOP_R_um
        a = PAD_ARR_ROW
        b = PAD_ARR_COL
        if VOID_SHAPE == 'circle':
            if 2 * (r_v + r_p) <= PITCH_um:
                return N * np.pi * (r_v + r_p)**2
            elif 2 * (r_v + r_p) > PITCH_um and 2 * (r_v + r_p) <= np.sqrt(2) * PITCH_um:
                theta = np.arccos(PITCH_um / (2 * (r_v + r_p)))
                return N * np.pi * (r_v + r_p)**2 \
                    - 2 * ((a-1)*b+(b-1)*a) * (theta - 0.5*np.sin(2*theta)) * (r_v + r_p)**2
            elif 2 * (r_v + r_p) > np.sqrt(2) * PITCH_um:
                theta = np.arccos(PITCH_um / (2 * (r_v + r_p)))
                return (a-1)*(b-1) * PITCH_um**2 + 2 * ((a-1)+(b-1)) * 0.5 * 0.5*np.sin(2*theta) * (r_v + r_p)**2 \
                + ((3*np.pi-4*theta)+((a-2)+(b-2))*(np.pi-2*theta)) * (r_v + r_p)**2
                # return PAD_ARR_W_um * PAD_ARR_L_um
        if VOID_SHAPE == 'square':
            if 2 * (r_v + r_p) <= PITCH_um:
                return 4 * N * (r_v + r_p)**2
            elif 2 * (r_v + r_p) > PITCH_um:
                return ((a-1) * PITCH_um + 2 * (r_v + r_p)) * ((b-1) * PITCH_um + 2 * (r_v + r_p))
            
    def integral_main_voids(r_mv):
        Distr_r_mv = (4 * D0 * (z - 1) * t_0**(z - 1)) / (WAF_R_um**2 * r_mv**(2*z - 1) * k_r**2) \
            * (
                ((k_r * WAF_R_um + k_r0) ** (2*z) - k_r0**(2*z)) / (2*z)
                - k_r0 * ((k_r * WAF_R_um + k_r0) ** (2*z-1) - k_r0**(2*z-1))/(2*z - 1)
            )
        A_r_mv = void_critical_area_per_die(PAD_TOP_R_um, r_mv, PITCH_um, PAD_ARR_ROW, PAD_ARR_COL, PAD_ARR_W_um, PAD_ARR_L_um, VOID_SHAPE)
        return Distr_r_mv * A_r_mv
    
    def f_r_mv(r_mv, k_r, k_r0, WAF_R_um, t_0, z):
        # Define critical radius value
        r_critical = (k_r * WAF_R_um + k_r0) * np.sqrt(t_0)
        if r_mv < k_r0 * np.sqrt(t_0):
            return 0
        if r_mv < r_critical:
            # Calculate f_r_mv for r < r_critical
            term1 = ((z - 1) * t_0**(z - 1)) / (k_r**2 * WAF_R_um**2)
            inner_term1 = (2 * r_mv) / (z * t_0 ** z) + (2 * k_r0**(2 * z)) / (z * (2 * z - 1) * r_mv**(2 * z - 1))
            inner_term2 = (2 * k_r0) / ((z - 1 / 2) * t_0**(z - 1 / 2))
            f_r_mv_value = term1 * (inner_term1 - inner_term2)
        
        else:
            # Calculate f_r_mv for r >= r_critical
            term1 = (2 * (z - 1) * t_0**(z - 1) * (k_r * WAF_R_um + k_r0)**(2*z-2)) / (r_mv**(2 * z - 1))
            out_term2 = 2 * (z - 1)**2 * t_0**(z-1) / (k_r**2 * WAF_R_um**2 * r_mv**(2 * z - 1))
            bracket_term2 = ((k_r * WAF_R_um + k_r0)**(2*z) - k_r0**(2*z)) / z - (2*k_r0*(k_r * WAF_R_um + k_r0)**(2*z-1)-2*k_r0**(2*z)) / (z - 1/2) + (k_r0**2 * (k_r*WAF_R_um+k_r0)**(2*z-2)-k_r0**(2*z)) / (z-1)
            f_r_mv_value = term1 - out_term2 * bracket_term2

        return f_r_mv_value
    
    '''
    The distribution of the void tail length
    '''
    def f_l(l, D0, k_l, WAF_R_um, t_0, z):
        if l <= k_l * WAF_R_um * np.sqrt(t_0):
            f_l_value = 2 * D0 * (z - 1) / (z * k_l**2 * WAF_R_um**2 * t_0) * l
        else:
            f_l_value = 2 * D0 * (z - 1) * (k_l**2 * WAF_R_um**2 * t_0) ** (z - 1) / z / (l**(2 * z - 1))
        return f_l_value
    
    
    def integral_size_main_voids(r_mv):
        return f_r_mv(r_mv, k_r, k_r0, WAF_R_um, t_0, z) * r_mv
    
    
    # # Calculate the average size of the main voids
    def avg_size_void_defects(k_r0, t_0):
        avg_size_main_voids = quad(integral_size_main_voids, k_r0*t_0**0.5, np.inf)[0]
    
        return avg_size_main_voids

    r_avg_mv = avg_size_void_defects(k_r0, t_0)

    if cfg.DEBUG == True:
        L_avg = k_L * 2/3 * WAF_R_um * (z-1) / (z-1.5) * t_0**0.5
        print("Average void tail length: ", L_avg)
        print("The average main void size is {}.".format(r_avg_mv))
    
    
    def A_critical_l_formula(a, b, l):
        return a * b + 2 / np.pi * (a + b) * l

    def avg_defects_fail_die_critical_fine(cfg, D0, k_L, WAF_R_um, t_0, z):
        '''
        This function calculate the average number of defects that will fail the critical pad
        '''
        if not os.path.exists('pad_bitmap/avg_num_defects_per_unit_area.npy'):
            avg_num_defects1 = quad(
                lambda l: f_l(l, 1e-11, k_L, WAF_R_um, t_0, z) * \
                A_critical_l_across_theta(cfg, PITCH_um, l, r_avg_mv, angle_step=30, bitmap_collection=pad_bitmap_collection),
                0, 
                np.sqrt(PAD_ARR_W_um**2 + PAD_ARR_L_um**2),
                epsabs=1e-1, epsrel=1e-1)[0]
        
            avg_num_defects = (avg_num_defects1) / 1e-11 * D0    # In this way, the integration can be faster
            np.save('pad_bitmap/avg_num_defects_per_unit_area.npy', avg_num_defects / D0)
        else:
            avg_num_defects = np.load('./pad_bitmap/avg_num_defects_per_unit_area.npy') * D0
        return avg_num_defects
    
    def avg_defects_fail_die_critical_coarse(cfg, D0, k_L, WAF_R_um, t_0, z):
        '''
        This function calculate the average number of fatal defects to the die
        '''
        if not os.path.exists('pad_bitmap/avg_num_defects_per_unit_area.npy'):
            avg_num_defects1 = quad(
                lambda l: f_l(l, 1e-11, k_L, WAF_R_um, t_0, z) * \
                A_critical_l_across_theta(cfg, PITCH_um, l, r_avg_mv, angle_step=60, bitmap_collection=pad_bitmap_collection),
                0, 
                np.sqrt(PAD_ARR_W_um**2 + PAD_ARR_L_um**2),
                epsabs=1e-1, epsrel=1e-1)[0]

            avg_num_defects2 = quad(
                lambda l: f_l(l, 1e-11, k_L, WAF_R_um, t_0, z) * \
                A_critical_l_formula(EFF_PAD_ARR_W_um, EFF_PAD_ARR_L_um, l), 
                np.sqrt(PAD_ARR_W_um**2 + PAD_ARR_L_um**2),
                np.inf)[0]
            print("avg_num_defects1: ", avg_num_defects1, "avg_num_defects2: ", avg_num_defects2)
            avg_num_defects = (avg_num_defects1 + avg_num_defects2) / 1e-11 * D0    # In this way, the integration can be faster
            # Save the critical area
            np.save('pad_bitmap/avg_num_defects_per_unit_area.npy', avg_num_defects / D0)
        else:
            avg_num_defects = np.load('pad_bitmap/avg_num_defects_per_unit_area.npy') * D0
        return avg_num_defects
    
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
        

    # num_vtl_defects_critical = D0 * PAD_ARR_W_um * PAD_ARR_L_um \
    #     + (8*D0*(z-1)/(3*np.pi*(2*z-3))) * (PAD_ARR_W_um + PAD_ARR_L_um) * k_L * WAF_R_um * t_0**0.5
    '''
    Calculate the die-level defect yield
    '''
    scale_factor = num_die * PAD_ARR_W_um * PAD_ARR_L_um / (np.pi * WAF_R_um**2) \
        / ((PAD_ARR_W_um * PAD_ARR_L_um) / ((PAD_ARR_W_um + dice_width) * (PAD_ARR_L_um + dice_width)))
    if critical_pad_ratio < 0.1:
        num_vtl_defects = scale_factor * avg_defects_fail_die_critical_fine(cfg, D0, k_L, WAF_R_um, t_0, z)
    else:
        num_vtl_defects = scale_factor * avg_defects_fail_die_critical_coarse(cfg, D0, k_L, WAF_R_um, t_0, z)
    particle_defect_die_yield = np.exp(-num_vtl_defects)

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
            glb_defect_pad_yield_min = min(glb_defect_pad_yield_min, np.min(pad_yield_map_i))
            glb_defect_pad_yield_max = max(glb_defect_pad_yield_max, np.max(pad_yield_map_i))
            die.pad_yield_map['Y_df'] = pad_yield_map_i_sub
            print("Generated pad-level defect yield map for die {}.".format(i))
        wafer.glb_pad_yield_min_max_dict['Y_df'] = (glb_defect_pad_yield_min, glb_defect_pad_yield_max)
        print("Global min of the pad-level defect yield: {}".format(glb_defect_pad_yield_min))
        print("Global max of the pad-level defect yield: {}".format(glb_defect_pad_yield_max))
    return particle_defect_die_yield