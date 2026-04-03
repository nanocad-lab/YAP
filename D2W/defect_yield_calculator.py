#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Oct 2, 2025

'''
This module contains functions to calculate die-level and pad-level defect-induced yield based on void size distribution and pad layout.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sympy as sp
import os
import math
from scipy.integrate import quad
from pad_bitmap_generation import A_critical_r_mv

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
    left, right = np.where(cols)[0][[0, -1]] 

    height = bottom - top + 1
    width = right - left + 1

    return width, height

def defect_yield_calculator(
    cfg,
    eff_DIE_R: float,
    D0: float,
    t_0: float,
    z: float,
    k_r: float,
    k_r0: float,
    k_n: float,
    k_S: float,
    k_L: float,
    PAD_TOP_R_um: float,
    PITCH_um: float,
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    PAD_ARR_W_um: float,
    PAD_ARR_L_um: float,
    VOID_SHAPE: str,
    die,
    pad_bitmap_collection: dict,
    pad_yield_flag: bool = False,
    pad_yield_map_sub_factor: int = 1,
):
    # r_mv = sp.symbols("r_mv")
    pad_block_size = pad_bitmap_collection["pad_block_size"]
    CRITICAL_PAD_BLOCK_BITMAP = pad_bitmap_collection["CRITICAL_PAD_BLOCK_BITMAP"]
    REDUNDANT_MAIN_PAD_BLOCK_BITMAP = pad_bitmap_collection["REDUNDANT_MAIN_PAD_BLOCK_BITMAP"]
    is_redundant_copy_same_block = pad_bitmap_collection["is_redundant_copy_same_block"]
    CRITICAL_PAD_ARR_W_um_IND, CRITICAL_PAD_ARR_L_um_IND = get_bitmap_bounds(bitmap=CRITICAL_PAD_BLOCK_BITMAP, pad_block_size=pad_block_size)
    REDUNDANT_MAIN_PAD_ARR_W_um_IND, REDUNDANT_MAIN_PAD_ARR_L_um_IND = get_bitmap_bounds(bitmap=REDUNDANT_MAIN_PAD_BLOCK_BITMAP, pad_block_size=pad_block_size)
    REDUNDANT_MAIN_PAD_ARR_W_um = REDUNDANT_MAIN_PAD_ARR_W_um_IND * PITCH_um
    REDUNDANT_MAIN_PAD_ARR_L_um = REDUNDANT_MAIN_PAD_ARR_L_um_IND * PITCH_um
    CRITICAL_PAD_ARR_W_um = CRITICAL_PAD_ARR_W_um_IND * PITCH_um
    CRITICAL_PAD_ARR_L_um = CRITICAL_PAD_ARR_L_um_IND * PITCH_um
    if is_redundant_copy_same_block:
        EFF_PAD_ARR_W_um = max(CRITICAL_PAD_ARR_W_um, REDUNDANT_MAIN_PAD_ARR_W_um)
        EFF_PAD_ARR_L_um = max(CRITICAL_PAD_ARR_L_um, REDUNDANT_MAIN_PAD_ARR_L_um)
    else:
        EFF_PAD_ARR_W_um = CRITICAL_PAD_ARR_W_um
        EFF_PAD_ARR_L_um = CRITICAL_PAD_ARR_L_um
        
    def f_r_mv(r_mv, D0, k_r, k_r0, WAF_R_um, t_0, z):
        # Define critical radius value
        r_critical = (k_r * WAF_R_um + k_r0) * np.sqrt(t_0)
        if r_mv < k_r0 * np.sqrt(t_0):
            return 0
        if r_mv < r_critical:
            # Calculate f_r_mv for r < r_critical
            term1 = (D0 * (z - 1) * t_0**(z - 1)) / (k_r**2 * WAF_R_um**2)
            inner_term1 = (2 * r_mv) / (z * t_0 ** z) + (2 * k_r0**(2 * z)) / (z * (2 * z - 1) * r_mv**(2 * z - 1))
            inner_term2 = (2 * k_r0) / ((z - 1 / 2) * t_0**(z - 1 / 2))
            f_r_mv_value = term1 * (inner_term1 - inner_term2)
        else:
            # Calculate f_r_mv for r >= r_critical
            term1 = (2 * D0 * (z - 1) * t_0**(z - 1) * (k_r * WAF_R_um + k_r0)**(2*z-2)) / (r_mv**(2 * z - 1))
            out_term2 = 2 * D0 * (z - 1)**2 * t_0**(z-1) / (k_r**2 * WAF_R_um**2 * r_mv**(2 * z - 1))
            bracket_term2 = ((k_r * WAF_R_um + k_r0)**(2*z) - k_r0**(2*z)) / z - (2*k_r0*(k_r * WAF_R_um + k_r0)**(2*z-1)-2*k_r0**(2*z)) / (z - 1/2) + (k_r0**2 * (k_r*WAF_R_um+k_r0)**(2*z-2)-k_r0**(2*z)) / (z-1)
            f_r_mv_value = term1 - out_term2 * bracket_term2
        return f_r_mv_value
    
    def void_critical_area_per_die(PAD_TOP_R_um, r_v, PITCH_um, PAD_ARR_ROW, PAD_ARR_COL, VOID_SHAPE):
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
        if VOID_SHAPE == 'square':
            if 2 * (r_v + r_p) <= PITCH_um:
                return 4 * N * (r_v + r_p)**2
            elif 2 * (r_v + r_p) > PITCH_um:
                return ((a-1) * PITCH_um + 2 * (r_v + r_p)) * ((b-1) * PITCH_um + 2 * (r_v + r_p))
            
    def integral_main_voids(r_mv):
        Distr_r_mv = f_r_mv(r_mv, D0, k_r, k_r0, eff_DIE_R, t_0, z)
        A_r_mv = void_critical_area_per_die(PAD_TOP_R_um, r_mv, PITCH_um, PAD_ARR_ROW, PAD_ARR_COL, VOID_SHAPE)
        return Distr_r_mv * A_r_mv

    def avg_defects_fail_die_critical(*, cfg, D0: float, PITCH_um: float, pad_bitmap_collection: dict) -> float:
        '''
        This function calculate the average number of fatal main void defects to the die using dilation-based critical calculation method.
        '''
        if not os.path.exists('pad_bitmap/avg_num_defects_per_unit_area.npy'):
            avg_num_defects1 = quad(
                lambda r_mv: f_r_mv(r_mv, 1e-11, k_r, k_r0, eff_DIE_R, t_0, z) * A_critical_r_mv(cfg, PITCH_um, r_mv, pad_bitmap_collection),
                k_r0 * t_0**0.5,
                np.sqrt(PAD_ARR_W_um**2 + PAD_ARR_L_um**2)/2,
                epsabs=1e-1, epsrel=1e-1
            )[0]
            avg_num_defects2 = quad(
                lambda r_mv: f_r_mv(r_mv, 1e-11, k_r, k_r0, eff_DIE_R, t_0, z) * void_critical_area_per_die(
                    PAD_TOP_R_um, r_mv, PITCH_um, int(EFF_PAD_ARR_L_um / PITCH_um), int(EFF_PAD_ARR_W_um / PITCH_um), VOID_SHAPE
                ),
                np.sqrt(PAD_ARR_W_um**2 + PAD_ARR_L_um**2)/2,
                np.inf,
            )[0]
            avg_num_defects = (avg_num_defects1 + avg_num_defects2) / 1e-11 * D0
            # Save the critical area
            np.save('pad_bitmap/avg_num_defects_per_unit_area.npy', avg_num_defects / D0)
        else:
            avg_num_defects = np.load('pad_bitmap/avg_num_defects_per_unit_area.npy') * D0
        return avg_num_defects
    
    def avg_defects_fail_pad_critical(*, cfg, die, D0, PAD_TOP_R_um, k_r, k_r0, t_0, z) -> np.ndarray:
        '''
        This function calculate the average number of fatal main void defects to the pad
        To calculate the pad-level defect yield, we ignore whether the pad is redundant or not.
        '''
        pad2die_center_dist_um = np.sqrt(die.pad_array[:, 0]**2 + die.pad_array[:, 1]**2)

        # Use the formula to calculate the average number of fatal defects per pad
        term = k_r * pad2die_center_dist_um + k_r0
        part1 = PAD_TOP_R_um**2
        part2 = ((z - 1) / (z - 2)) * (term**2) * t_0
        part3 = (4 * (z - 1) / (2 * z - 3)) * term * PAD_TOP_R_um * t_0
        return np.pi * D0 * (part1 + part2 + part3)

    # avg_main_voids = quad(integral_main_voids, k_r0*t_0**0.5, np.inf)[0]
    avg_main_voids_per_die = avg_defects_fail_die_critical(cfg=cfg, D0=D0, PITCH_um=PITCH_um, pad_bitmap_collection=pad_bitmap_collection)    
    avg_main_voids_per_pad = avg_defects_fail_pad_critical(cfg=cfg, die=die, D0=D0, PAD_TOP_R_um=PAD_TOP_R_um, k_r=k_r, k_r0=k_r0, t_0=t_0, z=z) if pad_yield_flag else None

    particle_defect_die_yield = np.exp(-avg_main_voids_per_die)
    if pad_yield_flag == True:
        glb_defect_pad_yield_min = 1.0
        glb_defect_pad_yield_max = 0.0
        particle_defect_pad_yield_map = np.exp(-avg_main_voids_per_pad)
        glb_defect_pad_yield_min = min(glb_defect_pad_yield_min, particle_defect_pad_yield_map.min())
        glb_defect_pad_yield_max = max(glb_defect_pad_yield_max, particle_defect_pad_yield_map.max())
        die.glb_pad_yield_min_max_dict['Y_df'] = (glb_defect_pad_yield_min, glb_defect_pad_yield_max)
        # Subsampling the pad yield map to save memory and speed up the plotting
        nr = math.ceil(PAD_ARR_ROW / pad_yield_map_sub_factor)
        nc = math.ceil(PAD_ARR_COL / pad_yield_map_sub_factor)
        r_idx = np.round(np.linspace(0, PAD_ARR_ROW - 1, nr)).astype(int)
        c_idx = np.round(np.linspace(0, PAD_ARR_COL - 1, nc)).astype(int)
        RR, CC = np.meshgrid(r_idx, c_idx, indexing='ij')   # shape (nr, nc)
        I = RR * PAD_ARR_COL + CC  # linear indices. shape (nr, nc)
        particle_defect_pad_yield_map_sub = particle_defect_pad_yield_map[I]
    else:
        particle_defect_pad_yield_map = None
        particle_defect_pad_yield_map_sub = None

    if pad_yield_flag:
        # Draw heatmap of pad-level defect yield map
        plt.figure(figsize=(8, 6))
        plt.imshow(
            particle_defect_pad_yield_map_sub, 
            cmap='viridis', 
            vmin=die.glb_pad_yield_min_max_dict['Y_df'][0],
            vmax=die.glb_pad_yield_min_max_dict['Y_df'][1],
            interpolation='nearest'
            )
        plt.colorbar(label='Pad-level Defect Yield (Subsampled)')
        plt.title('Pad-level Defect Yield Map')
        plt.xlabel('Pad Column Index')
        plt.ylabel('Pad Row Index')
        plt.show()

    return particle_defect_die_yield, particle_defect_pad_yield_map_sub