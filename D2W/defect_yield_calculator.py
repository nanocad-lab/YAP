#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sympy as sp
import os
from scipy.integrate import quad
from pad_bitmap_generation import A_critical_r_mv

def get_bitmap_bounds(bitmap, pad_block_size):
    # Find the bounds of the non-zero pixels in the bitmap
    rows = np.any(bitmap, axis=1)
    cols = np.any(bitmap, axis=0)

    if not np.any(rows) or not np.any(cols):
        return 0, 0

    top, bottom = np.where(rows)[0][[0, -1]] * pad_block_size
    left, right = np.where(cols)[0][[0, -1]] 

    height = bottom - top + 1
    width = right - left + 1

    return width, height

def defect_yield_calculator(
    cfg,
    eff_DIE_R,
    D0,
    t_0,
    z,
    k_r,
    k_r0,
    k_n,
    k_S,
    k_L,
    PAD_TOP_R,
    PITCH,
    PAD_ARR_ROW,
    PAD_ARR_COL,
    PAD_ARR_W,
    PAD_ARR_L,
    VOID_SHAPE,
    pad_bitmap_collection,
):
    # r_mv = sp.symbols("r_mv")
    pad_block_size = pad_bitmap_collection["pad_block_size"]
    CRITICAL_PAD_BLOCK_BITMAP = pad_bitmap_collection["CRITICAL_PAD_BLOCK_BITMAP"]
    REDUNDANT_MAIN_PAD_BLOCK_BITMAP = pad_bitmap_collection["REDUNDANT_MAIN_PAD_BLOCK_BITMAP"]
    is_redundant_copy_same_block = pad_bitmap_collection["is_redundant_copy_same_block"]
    CRITICAL_PAD_ARR_W_IND, CRITICAL_PAD_ARR_L_IND = get_bitmap_bounds(CRITICAL_PAD_BLOCK_BITMAP, pad_block_size)
    REDUNDANT_MAIN_PAD_ARR_W_IND, REDUNDANT_MAIN_PAD_ARR_L_IND = get_bitmap_bounds(REDUNDANT_MAIN_PAD_BLOCK_BITMAP, pad_block_size)
    REDUNDANT_MAIN_PAD_ARR_W = REDUNDANT_MAIN_PAD_ARR_W_IND * PITCH
    REDUNDANT_MAIN_PAD_ARR_L = REDUNDANT_MAIN_PAD_ARR_L_IND * PITCH
    CRITICAL_PAD_ARR_W = CRITICAL_PAD_ARR_W_IND * PITCH
    CRITICAL_PAD_ARR_L = CRITICAL_PAD_ARR_L_IND * PITCH
    if is_redundant_copy_same_block:
        EFF_PAD_ARR_W = max(CRITICAL_PAD_ARR_W, REDUNDANT_MAIN_PAD_ARR_W)
        EFF_PAD_ARR_L = max(CRITICAL_PAD_ARR_L, REDUNDANT_MAIN_PAD_ARR_L)
    else:
        EFF_PAD_ARR_W = CRITICAL_PAD_ARR_W
        EFF_PAD_ARR_L = CRITICAL_PAD_ARR_L
    def f_r_mv(r_mv, D0, k_r, k_r0, WAF_R, t_0, z):
        # Define critical radius value
        r_critical = (k_r * WAF_R + k_r0) * np.sqrt(t_0)
        if r_mv < k_r0 * np.sqrt(t_0):
            return 0
        if r_mv < r_critical:
            # Calculate f_r_mv for r < r_critical
            term1 = (D0 * (z - 1) * t_0**(z - 1)) / (k_r**2 * WAF_R**2)
            inner_term1 = (2 * r_mv) / (z * t_0 ** z) + (2 * k_r0**(2 * z)) / (z * (2 * z - 1) * r_mv**(2 * z - 1))
            inner_term2 = (2 * k_r0) / ((z - 1 / 2) * t_0**(z - 1 / 2))
            f_r_mv_value = term1 * (inner_term1 - inner_term2)
        
        else:
            # Calculate f_r_mv for r >= r_critical
            term1 = (2 * D0 * (z - 1) * t_0**(z - 1) * (k_r * WAF_R + k_r0)**(2*z-2)) / (r_mv**(2 * z - 1))
            out_term2 = 2 * D0 * (z - 1)**2 * t_0**(z-1) / (k_r**2 * WAF_R**2 * r_mv**(2 * z - 1))
            bracket_term2 = ((k_r * WAF_R + k_r0)**(2*z) - k_r0**(2*z)) / z - (2*k_r0*(k_r * WAF_R + k_r0)**(2*z-1)-2*k_r0**(2*z)) / (z - 1/2) + (k_r0**2 * (k_r*WAF_R+k_r0)**(2*z-2)-k_r0**(2*z)) / (z-1)
            f_r_mv_value = term1 - out_term2 * bracket_term2

        return f_r_mv_value
    def void_critical_area_per_die(PAD_TOP_R, r_v, PITCH, PAD_ARR_ROW, PAD_ARR_COL, VOID_SHAPE):
        N = PAD_ARR_ROW * PAD_ARR_COL
        r_p = PAD_TOP_R
        a = PAD_ARR_ROW
        b = PAD_ARR_COL
        if VOID_SHAPE == 'circle':
            if 2 * (r_v + r_p) <= PITCH:
                return N * np.pi * (r_v + r_p)**2
            elif 2 * (r_v + r_p) > PITCH and 2 * (r_v + r_p) <= np.sqrt(2) * PITCH:
                theta = np.arccos(PITCH / (2 * (r_v + r_p)))
                return N * np.pi * (r_v + r_p)**2 \
                    - 2 * ((a-1)*b+(b-1)*a) * (theta - 0.5*np.sin(2*theta)) * (r_v + r_p)**2
            elif 2 * (r_v + r_p) > np.sqrt(2) * PITCH:
                theta = np.arccos(PITCH / (2 * (r_v + r_p)))
                return (a-1)*(b-1) * PITCH**2 + 2 * ((a-1)+(b-1)) * 0.5 * 0.5*np.sin(2*theta) * (r_v + r_p)**2 \
                + ((3*np.pi-4*theta)+((a-2)+(b-2))*(np.pi-2*theta)) * (r_v + r_p)**2
        if VOID_SHAPE == 'square':
            if 2 * (r_v + r_p) <= PITCH:
                return 4 * N * (r_v + r_p)**2
            elif 2 * (r_v + r_p) > PITCH:
                return ((a-1) * PITCH + 2 * (r_v + r_p)) * ((b-1) * PITCH + 2 * (r_v + r_p))
            
    def integral_main_voids(r_mv):
        Distr_r_mv = f_r_mv(r_mv, D0, k_r, k_r0, eff_DIE_R, t_0, z)
        A_r_mv = void_critical_area_per_die(PAD_TOP_R, r_mv, PITCH, PAD_ARR_ROW, PAD_ARR_COL, VOID_SHAPE)
        return Distr_r_mv * A_r_mv

    def avg_defects_fail_die_critical(cfg, D0, PITCH, bitmap_collection):
        '''
        This function calculate the average number of fatal main void defects to the die using dilation-based critical calculation method.
        '''
        if not os.path.exists('pad_bitmap/avg_num_defects_per_unit_area.npy'):
            avg_num_defects1 = quad(
                lambda r_mv: f_r_mv(r_mv, 1e-11, k_r, k_r0, eff_DIE_R, t_0, z) * A_critical_r_mv(cfg, PITCH, r_mv, bitmap_collection),
                k_r0 * t_0**0.5,
                np.sqrt(PAD_ARR_W**2 + PAD_ARR_L**2)/2,
                epsabs=1e-1, epsrel=1e-1
            )[0]
            avg_num_defects2 = quad(
                lambda r_mv: f_r_mv(r_mv, 1e-11, k_r, k_r0, eff_DIE_R, t_0, z) * void_critical_area_per_die(
                    PAD_TOP_R, r_mv, PITCH, int(EFF_PAD_ARR_L / PITCH), int(EFF_PAD_ARR_W / PITCH), VOID_SHAPE
                ),
                np.sqrt(PAD_ARR_W**2 + PAD_ARR_L**2)/2,
                np.inf,
            )[0]
            avg_num_defects = (avg_num_defects1 + avg_num_defects2) / 1e-11 * D0
            # Save the critical area
            np.save('pad_bitmap/avg_num_defects_per_unit_area.npy', avg_num_defects / D0)
        else:
            avg_num_defects = np.load('pad_bitmap/avg_num_defects_per_unit_area.npy') * D0
        return avg_num_defects


    # avg_main_voids = quad(integral_main_voids, k_r0*t_0**0.5, np.inf)[0]
    avg_main_voids = avg_defects_fail_die_critical(cfg, D0, PITCH, pad_bitmap_collection)    


    particle_defect_yield = np.exp(-avg_main_voids)

    return particle_defect_yield