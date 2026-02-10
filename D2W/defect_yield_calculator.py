#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Feb 10, 2026

'''
This module contains functions to calculate die-level and pad-level defect-induced yield based on void size distribution and pad layout.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.integrate import quad

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





def pad_defect_yield_map_generator(
    cfg,
    D0: float,
    t_0: float,
    z: float,
    k_r: float,
    k_r0: float,
    PAD_TOP_R_um: float,
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    die,
    pad_yield_flag: bool = False,
    pad_yield_map_sub_factor: int = 1,
):
    def avg_defects_fail_pad_critical(*, cfg, die, D0, PAD_TOP_R_um, k_r, k_r0, t_0, z) -> np.ndarray:
        '''
        This function calculate the average number of fatal main void defects to the pad
        To calculate the pad-level defect yield, we ignore whether the pad is redundant or not.
        '''
        if cfg.first_contact == 'center':
            L0 = np.sqrt(die.pad_coords[:, 0]**2 + die.pad_coords[:, 1]**2) # pad to die center distance
        elif cfg.first_contact == 'vertical-edge':
            L0 = np.abs(die.DIE_W_um / 2 + die.pad_coords[:, 0]) # pad to left die edge distance
        elif cfg.first_contact == 'horizontal-edge':
            L0 = np.abs(die.DIE_L_um / 2 + die.pad_coords[:, 1]) # pad to bottom die edge distance
        elif cfg.first_contact == 'corner':
            L0 = np.sqrt((die.DIE_W_um / 2 + die.pad_coords[:, 0])**2 + (die.DIE_L_um / 2 + die.pad_coords[:, 1])**2) # pad to left-bottom die corner distance
            
        # Use the formula to calculate the average number of fatal defects per pad
        term = k_r * L0 + k_r0
        part1 = PAD_TOP_R_um**2
        part2 = ((z - 1) / (z - 2)) * (term**2) * t_0
        part3 = (4 * (z - 1) / (2 * z - 3)) * term * PAD_TOP_R_um * t_0
        return np.pi * D0 * (part1 + part2 + part3)
  
    avg_main_voids_per_pad = avg_defects_fail_pad_critical(cfg=cfg, die=die, D0=D0, PAD_TOP_R_um=PAD_TOP_R_um, k_r=k_r, k_r0=k_r0, t_0=t_0, z=z) if pad_yield_flag else None

    if pad_yield_flag == True:
        glb_defect_pad_yield_min = 1.0
        glb_defect_pad_yield_max = 0.0
        particle_defect_pad_yield_map = np.exp(-avg_main_voids_per_pad)
        glb_defect_pad_yield_min = min(glb_defect_pad_yield_min, np.nanmin(particle_defect_pad_yield_map))
        glb_defect_pad_yield_max = max(glb_defect_pad_yield_max, np.nanmax(particle_defect_pad_yield_map))
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

    if cfg.plot_flag and pad_yield_flag:
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

    return particle_defect_pad_yield_map_sub



def stack_defect_yield_calculator(
    cfg_dict: dict,
    die_stack,
):
    def f_r_mv(r_mv, D0, k_r, k_r0, eff_DIE_R, t_0, z):
        # Define critical radius value
        r_critical = (k_r * eff_DIE_R + k_r0) * np.sqrt(t_0)
        if r_mv < k_r0 * np.sqrt(t_0):
            return 0
        if r_mv < r_critical:
            # Calculate f_r_mv for r < r_critical
            term1 = (D0 * (z - 1) * t_0**(z - 1)) / (k_r**2 * eff_DIE_R**2)
            inner_term1 = (2 * r_mv) / (z * t_0 ** z) + (2 * k_r0**(2 * z)) / (z * (2 * z - 1) * r_mv**(2 * z - 1))
            inner_term2 = (2 * k_r0) / ((z - 1 / 2) * t_0**(z - 1 / 2))
            f_r_mv_value = term1 * (inner_term1 - inner_term2)
        
        else:
            # Calculate f_r_mv for r >= r_critical
            term1 = (2 * D0 * (z - 1) * t_0**(z - 1) * (k_r * eff_DIE_R + k_r0)**(2*z-2)) / (r_mv**(2 * z - 1))
            out_term2 = 2 * D0 * (z - 1)**2 * t_0**(z-1) / (k_r**2 * eff_DIE_R**2 * r_mv**(2 * z - 1))
            bracket_term2 = ((k_r * eff_DIE_R + k_r0)**(2*z) - k_r0**(2*z)) / z - (2*k_r0*(k_r * eff_DIE_R + k_r0)**(2*z-1)-2*k_r0**(2*z)) / (z - 1/2) + (k_r0**2 * (k_r*eff_DIE_R+k_r0)**(2*z-2)-k_r0**(2*z)) / (z-1)
            f_r_mv_value = term1 - out_term2 * bracket_term2

        return f_r_mv_value



    

    def void_critical_area_per_die(PAD_TOP_R, r_v, PITCH_r, PITCH_c, PAD_ARR_ROW, PAD_ARR_COL, VOID_SHAPE):
        N = PAD_ARR_ROW * PAD_ARR_COL
        r_p = PAD_TOP_R
        a = PAD_ARR_ROW
        b = PAD_ARR_COL
        if VOID_SHAPE == 'circle':
            if 2 * (r_v + r_p) <= min(PITCH_r, PITCH_c):
                return N * np.pi * (r_v + r_p)**2
            elif 2 * (r_v + r_p) > min(PITCH_r, PITCH_c) and 2 * (r_v + r_p) <= np.sqrt(PITCH_r**2 + PITCH_c**2):
                if 2 * (r_v + r_p) > PITCH_r:
                    theta_r = np.arccos(PITCH_r / (2 * (r_v + r_p)))
                    return N * np.pi * (r_v + r_p)**2 - (a-1)*b*2*(theta_r - 0.5*np.sin(2*theta_r)) * (r_v + r_p)**2
                if 2 * (r_v + r_p) > PITCH_c:
                    theta_c = np.arccos(PITCH_c / (2 * (r_v + r_p)))
                    return N * np.pi * (r_v + r_p)**2 - (b-1)*a*2*(theta_c - 0.5*np.sin(2*theta_c)) * (r_v + r_p)**2
            elif 2 * (r_v + r_p) > np.sqrt(PITCH_r**2 + PITCH_c**2):
                theta_r = np.arccos(PITCH_r / (2 * (r_v + r_p)))
                theta_c = np.arccos(PITCH_c / (2 * (r_v + r_p)))
                return (a-1)*(b-1) * PITCH_r * PITCH_c + ((a-1)*np.sin(2*theta_c) + (b-1)*np.sin(2*theta_r)) * (r_v + r_p)**2 \
                + ((3*np.pi-2*theta_r-2*theta_c)+(a-2)*(np.pi-2*theta_r)+((b-2)*(np.pi-2*theta_c))) * (r_v + r_p)**2
        elif VOID_SHAPE == 'square':
            if 2 * (r_v + r_p) <= min(PITCH_c, PITCH_r):
                return 4 * N * (r_v + r_p)**2
            elif 2 * (r_v + r_p) > min(PITCH_c, PITCH_r) and 2 * (r_v + r_p) <= max(PITCH_c, PITCH_r):
                if PITCH_c < PITCH_r:
                    return 4 * N * (r_v + r_p)**2 - a * (b - 1) * 2 * (r_v + r_p) * (r_v + r_p - PITCH_c)
                else:
                    return 4 * N * (r_v + r_p)**2 - (a - 1) * b * 2 * (r_v + r_p) * (r_v + r_p - PITCH_r)
            elif 2 * (r_v + r_p) > max(PITCH_c, PITCH_r):
                return ((a-1) * PITCH_r + 2 * (r_v + r_p)) * ((b-1) * PITCH_c + 2 * (r_v + r_p))
        else:
            raise ValueError("Invalid VOID_SHAPE value. Please specify 'circle' or 'square'.")
            
    def integral_main_voids(r_mv, D0, k_r, k_r0, eff_DIE_R, t_0, z, PAD_TOP_R, PITCH_r, PITCH_c, PAD_ARR_ROW, PAD_ARR_COL, VOID_SHAPE):
        Distr_r_mv = f_r_mv(r_mv, D0, k_r, k_r0, eff_DIE_R, t_0, z)
        A_r_mv = void_critical_area_per_die(PAD_TOP_R, r_mv, PITCH_r, PITCH_c, PAD_ARR_ROW, PAD_ARR_COL, VOID_SHAPE)
        return Distr_r_mv * A_r_mv
    
    for interface_name, cfg in cfg_dict.items():
        D0 = cfg.D0
        t_0 = cfg.t_0
        z = cfg.z
        k_r = cfg.k_r
        k_r0 = cfg.k_r0
        PAD_TOP_R_um = cfg.PAD_TOP_R_um
        PAD_ARR_ROW, PAD_ARR_COL = cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL
        PITCH_r_um, PITCH_c_um = cfg.PITCH_r_um, cfg.PITCH_c_um
        VOID_SHAPE = cfg.VOID_SHAPE
        eff_DIE_R_um = cfg.eff_DIE_R_um
        avg_main_voids = quad(integral_main_voids, k_r0*t_0**0.5, np.inf, args=(D0, k_r, k_r0, eff_DIE_R_um, 
                                                                                t_0, z, PAD_TOP_R_um, PITCH_r_um, PITCH_c_um,
                                                                                PAD_ARR_ROW, PAD_ARR_COL, VOID_SHAPE))[0]
        particle_defect_yield = np.exp(-avg_main_voids)
        die_stack.die_yield_list_per_interface_dict[interface_name]['defect'] = particle_defect_yield

    return particle_defect_yield