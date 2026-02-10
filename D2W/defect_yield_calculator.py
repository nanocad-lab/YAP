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
import os
import math

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
    def avg_defects_per_die