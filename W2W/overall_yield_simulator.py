#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Jan 20, 2026

import os
import sys
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import time
import matplotlib.pyplot as plt

from overlay_yield_simulator import die_pad_misalignment
from Cu_gap_simulator import Cu_gap_simulator
from debond import debond_dishing_bounds_calculator
from esd_hybrid import esd_failure_simulator

def total_memory_mb(obj):
    total = sys.getsizeof(obj)
    if isinstance(obj, list):
        for item in obj:
            try:
                # numpy arrays
                total += item.nbytes
            except AttributeError:
                # fallback
                total += sys.getsizeof(item)
    return total / 1024 / 1024  # MB


def overall_yield_simulator(
    cfg,
    waf_stack_list: list,
    num_dies_per_wafer: int,
    WAF_R_um: float,
    system_translation_x_um: np.ndarray,
    system_translation_y_um: np.ndarray,
    system_rotation_rad: np.ndarray,
    system_magnification_ppm: np.ndarray,
    MAX_ALLOWED_MISALIGNMENT_um: float,
    PAD_ARR_W_um: float,
    PAD_ARR_L_um: float,
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    TOP_DISH_MEAN_nm: float,
    TOP_DISH_STD_nm: float,
    BOT_DISH_MEAN_nm: float,
    BOT_DISH_STD_nm: float,
    TILT_X_MEAN_DEG: float,
    TILT_X_STD_DEG: float,
    TILT_Y_MEAN_DEG: float,
    TILT_Y_STD_DEG: float,
    k_et: float,
    k_eb: float,
    T_R: float,
    T_anl: float,
    PITCH_r_um: float,
    PITCH_c_um: float,
    PAD_TOP_R_um: float,
    RANDOM_MISALIGNMENT_MEAN_um: float,
    RANDOM_MISALIGNMENT_STD_um: float,
    approximate_set: int,
    pad_bitmap_collection: dict,
):
    die_stack_yield_list = []
    # print("The memory size of the waf_list is {} MB.".format(total_memory_mb(waf_list)))

    # Read the parameters
    NUM_WAFERS_PER_STACK = waf_stack_list[0].num_layers
    NUM_STACKS = len(waf_stack_list)
    NUM_BONDING_INTERFACES = waf_stack_list[0].num_bonding_interfaces

    epoch_fail_map_dict = {}    # This dict stores the fail bump maps for all die samples in this epoch for each mechanism
    epoch_fail_vec_dict = {}    # This dict stores failure reason (each mechanism) for all die samples in this epoch
    if cfg.verbose:
        epoch_fail_map_dict['overlay']    = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))
        epoch_fail_map_dict['particle']   = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))
        epoch_fail_map_dict['mechanical'] = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))
        epoch_fail_map_dict['ESD']        = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))
        epoch_fail_map_dict['overall']    = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))

        epoch_fail_vec_dict['overlay']    = np.zeros((NUM_STACKS, NUM_WAFERS_PER_STACK, num_dies_per_wafer))
        epoch_fail_vec_dict['particle']   = np.zeros((NUM_STACKS, NUM_WAFERS_PER_STACK, num_dies_per_wafer))
        epoch_fail_vec_dict['mechanical'] = np.zeros((NUM_STACKS, NUM_WAFERS_PER_STACK, num_dies_per_wafer))
        epoch_fail_vec_dict['ESD']        = np.zeros((NUM_STACKS, NUM_WAFERS_PER_STACK, num_dies_per_wafer))
        epoch_fail_vec_dict['overall']    = np.zeros((NUM_STACKS, NUM_WAFERS_PER_STACK, num_dies_per_wafer))
    
    for stack_ind, waf_stack in enumerate(waf_stack_list):
        for interface_ind in range(NUM_BONDING_INTERFACES):
            # Record the time
            start_time = time.time()
            die_count = 0
            bot_wafer = waf_stack.layer_list[interface_ind]
            top_wafer = waf_stack.layer_list[interface_ind + 1]
            # Read the critical pad bitmap
            die_critical_pad_bitmap = pad_bitmap_collection["CRITICAL_PAD_BITMAP"]
            # Read the redundant critical pad bitmap
            die_redundant_pad_bitmap = pad_bitmap_collection["REDUNDANT_PAD_BITMAP"]
            # Read the ESD-critical pad bitmap
            die_esd_critical_pad_bitmap = pad_bitmap_collection["ESD_CRITICAL_PAD_BITMAP"]
            # Read the redundant net to bump ids mapping
            redundant_net_to_bumpids = pad_bitmap_collection["redundant_net_to_bumpids"]
            valid_pad_mask = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)
            # Read the mapping from physical pad location to bump id
            mapping_physical_to_bumpid = pad_bitmap_collection["mapping_physical_to_bumpid"]
            # Read the criticality info
            criticality_info = pad_bitmap_collection["criticality_info"]
            # Read the redundant net to 1D physical mask mapping
            redundant_net_to_1d_physical_mask = pad_bitmap_collection["redundant_net_to_1d_physical_mask"]

            for die_ind, die in enumerate(bot_wafer.die_list):
                die_pad_coords = bot_wafer.base_pad_coords + die.die_center
                valid_die_pad_coords = die_pad_coords[valid_pad_mask.flatten() == 1]
                die_count += 1
                if die_count % 10 == 0:
                    print("Processing die {}/{}...Time taken for every 10 dies: {:.2f} seconds".format(die_count, len(bot_wafer.die_list), (time.time() - start_time) / die_count * 10), end='\r')
                    # start_time = time.time()
                redundant_pad_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))
                
                '''
                Check the overlay errors
                '''
                # # Check the pad misalignment
                die.pad_misalignment = die_pad_misalignment(die=die, 
                                                            base_pad_coords=bot_wafer.base_pad_coords,
                                                            system_translation_x_um=system_translation_x_um[stack_ind, interface_ind],
                                                            system_translation_y_um=system_translation_y_um[stack_ind, interface_ind],
                                                            system_rotation_rad=system_rotation_rad[stack_ind, interface_ind],
                                                            system_magnification_ppm=system_magnification_ppm[stack_ind, interface_ind],
                                                            RANDOM_MISALIGNMENT_MEAN_um=RANDOM_MISALIGNMENT_MEAN_um,
                                                            RANDOM_MISALIGNMENT_STD_um=RANDOM_MISALIGNMENT_STD_um,
                                                            approximate_set=approximate_set,
                                                            )
                if approximate_set == 1:
                    # pad fail criteria: pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um
                    die.pad_misalignment = die.pad_misalignment.reshape(cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL)
                    if cfg.verbose:
                        epoch_fail_map_dict['overlay'] += (die.pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um).astype(int)

                    critical_pad_misalignment = die.pad_misalignment * die_critical_pad_bitmap      # Shape: (PAD_ARR_ROW, PAD_ARR_COL)
                    # Check if any critical pad misalignment is greater than the maximum allowed misalignment
                    if np.any(critical_pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um):
                        bot_wafer.die_list[die_ind].survival, top_wafer.die_list[die_ind].survival = False, False
                        waf_stack.die_stack_survival[die_ind] = False
                        if cfg.verbose:
                            epoch_fail_vec_dict['overlay'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                            epoch_fail_vec_dict['overall'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                        if not cfg.verbose:
                            continue
                    # Check if too many redundant pad misalignment is greater than the maximum allowed misalignment
                    redundant_pad_misalignment = die.pad_misalignment * die_redundant_pad_bitmap    # Shape: (PAD_ARR_ROW, PAD_ARR_COL)
                    redundant_pad_fail_map[redundant_pad_misalignment > MAX_ALLOWED_MISALIGNMENT_um] = 1 # 1: redundant pad fails, shape: (PAD_ARR_ROW, PAD_ARR_COL)
                    for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
                        tolerated_mechanical_failures = criticality_info[redundant_net]['tolerated_mechanical_failures']
                        num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
                        if num_fail_pad_in_net > tolerated_mechanical_failures:
                            bot_wafer.die_list[die_ind].survival, top_wafer.die_list[die_ind].survival = False, False
                            waf_stack.die_stack_survival[die_ind] = False
                            if cfg.verbose:
                                epoch_fail_vec_dict['overlay'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                                epoch_fail_vec_dict['overall'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                            break
                    # # Get the fail bump indices
                    # fail_bump_id = mapping_physical_to_bumpid[redundant_pad_fail_map == 1]
                    # # Switch to set for easier checking
                    # fail_bump_id_set = set(fail_bump_id.astype(int))
                # Delete the die.pad_misalignment to save memory
                del die.pad_misalignment
            
                
                # # Check every net connecting redundant pads, if all the redundant pad replicas fail, then the die fails
                # if any(redundant_bumpid_set.issubset(fail_bump_id_set) for net, redundant_bumpid_set in redundant_net_to_bumpids.items()):
                #     wafer.survival_die -= 1
                #     die.survival = False
                #     break
                if not waf_stack.die_stack_survival[die_ind] and not cfg.verbose:
                    continue
                
                '''
                Check the void defects
                '''
                # # Check the void overlap with the pad
                # Assuming wafer.voids is an array of shape (N, 3), where N is the number of voids. [x, y, r]
                # Critical pad bitmap is a 2D array of shape (PAD_ARR_ROW, PAD_ARR_COL) with 1s for critical pads and 0s for non-critical pads
                voids = np.array(waf_stack.interfaces.failure_params['voids'][interface_ind])  # shape (N, 3), N is the number of voids
                if voids.size > 0:
                    # Coordinates and dimensions of the die pad array box
                    pad_array_box_x = die.pad_array_box[2][0]
                    pad_array_box_y = die.pad_array_box[2][1]

                    # Calculate closest x and y distances for all voids simultaneously
                    closest_x = np.maximum(pad_array_box_x, np.minimum(voids[:, 0], pad_array_box_x + PAD_ARR_W_um))
                    closest_y = np.maximum(pad_array_box_y, np.minimum(voids[:, 1], pad_array_box_y + PAD_ARR_L_um))

                    # Calculate distance from each void to the closest point on the pad array box
                    distances = (closest_x - voids[:, 0]) ** 2 + (closest_y - voids[:, 1]) ** 2

                    # Create a mask for voids overlapping with the pad array box
                    overlap_void_die_mask = distances < voids[:, 2] ** 2  # shape (N,)

                    # Use critical pad bitmap and grid search to find if any void overlaps with the die
                    if np.any(overlap_void_die_mask):
                        # Calculate the pad range we need to consider (critical, near the void)
                        # The i, j here are the indices of the pad array bitmap. The origin is the bottom left corner of the pad array box. 
                        # It is noticed that the origin of the bitmap is the top left corner of the pad array box. Switching is needed.
                        in_die_voids = voids[overlap_void_die_mask]
                        i_coord_min = min(in_die_voids[:, 0] - in_die_voids[:, 2] - PAD_TOP_R_um - pad_array_box_x)
                        i_coord_max = max(in_die_voids[:, 0] + in_die_voids[:, 2] + PAD_TOP_R_um - pad_array_box_x)
                        j_coord_min = min(in_die_voids[:, 1] - in_die_voids[:, 2] - PAD_TOP_R_um - pad_array_box_y)
                        j_coord_max = max(in_die_voids[:, 1] + in_die_voids[:, 2] + PAD_TOP_R_um - pad_array_box_y)
                        i_min = max(0,              int(np.floor(i_coord_min / PITCH_c_um)))    # (col_start)
                        i_max = min(PAD_ARR_COL-1,  int(np.ceil (i_coord_max / PITCH_c_um)))    # H = i_max - i_min + 1 (col_end)
                        j_min = max(0,              int(np.floor(j_coord_min / PITCH_r_um)))    # (row_start)
                        j_max = min(PAD_ARR_ROW-1,  int(np.ceil (j_coord_max / PITCH_r_um)))    # W = j_max - j_min + 1 (row_end)

                        check_pad_x_coords = pad_array_box_x + np.arange(i_min, i_max+1) * PITCH_c_um
                        check_pad_y_coords = pad_array_box_y + np.arange(j_min, j_max+1) * PITCH_r_um
                        check_pad_x_mesh, check_pad_y_mesh = np.meshgrid(check_pad_x_coords, check_pad_y_coords, indexing='xy')

                        # Calculate the distance from each void to the closest point on the critical pads
                        voids_xy = in_die_voids[:, :2]   # shape (N, 2), N is the number of voids
                        voids_x = voids_xy[:, 0][:, np.newaxis, np.newaxis]  # shape (N, 1, 1)
                        voids_y = voids_xy[:, 1][:, np.newaxis, np.newaxis]  # shape (N, 1, 1)
                        voids_r = in_die_voids[:, 2][:, np.newaxis, np.newaxis]  # shape (N, 1, 1)
                        pad_x = check_pad_x_mesh[np.newaxis, :, :]  # shape (1, H, W)
                        # print(pad_x)
                        pad_y = check_pad_y_mesh[np.newaxis, :, :]  # shape (1, H, W)
                        # print(pad_y)
                        dist_sq = (pad_x - voids_x) ** 2 + (pad_y - voids_y) ** 2  # shape (N, H, W)
                        overlap_void_pad_mask = dist_sq < (voids_r + PAD_TOP_R_um) ** 2 # shape (N, H, W)
                        overlap_void_pad_mask = np.any(overlap_void_pad_mask, axis=0)  # shape (H, W)
                        if np.any(overlap_void_pad_mask):
                            bot_wafer.die_list[die_ind].voids_occur, top_wafer.die_list[die_ind].voids_occur = True, True
                            
                        # Get the critical pad bitmap for the pads we need to consider
                        check_critical_pad_bitmap = die_critical_pad_bitmap[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                        # Get the redundant critical pad bitmap for the pads we need to consider
                        check_redundant_pad_bitmap = die_redundant_pad_bitmap[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                        # Record the fail pads due to voids
                        if cfg.verbose:
                            epoch_fail_map_dict['particle'][PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1][overlap_void_pad_mask] += 1

                        # Check if any void overlaps with the critical pads
                        overlap_critical = overlap_void_pad_mask & check_critical_pad_bitmap.astype(bool)
                        if np.any(overlap_critical):
                            # print("Die fails due to critical pad void overlap.")
                            bot_wafer.die_list[die_ind].survival, top_wafer.die_list[die_ind].survival = False, False
                            waf_stack.die_stack_survival[die_ind] = False
                            if cfg.verbose:
                                epoch_fail_vec_dict['particle'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                                epoch_fail_vec_dict['overall'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                        else:   # Voids overlapping with the redundant pads.
                            # Check if any void overlaps with the redundant critical pads
                            overlap_redundant = overlap_void_pad_mask & check_redundant_pad_bitmap.astype(bool) # shape (H, W)
                            # if overlap #pads is greater than a percentage of the total pads, then the die fails
                            redundant_pad_fail_map[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1][overlap_redundant] = 1
                            for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
                                tolerated_mechanical_failures = criticality_info[redundant_net]['tolerated_mechanical_failures']
                                num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
                                if num_fail_pad_in_net > tolerated_mechanical_failures:
                                    bot_wafer.die_list[die_ind].survival, top_wafer.die_list[die_ind].survival = False, False
                                    waf_stack.die_stack_survival[die_ind] = False
                                    if cfg.verbose:
                                        epoch_fail_vec_dict['particle'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                                        epoch_fail_vec_dict['overall'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                                    if not cfg.verbose:
                                        break                        
                            # # Get the fail bump indices
                            # fail_bump_id = mapping_physical_to_bumpid[redundant_pad_fail_map == 1]
                            # # Switch to set for easier checking
                            # fail_bump_id_set = set(fail_bump_id.astype(int))
                            # # Check every net connecting redundant pads, if all the redundant pad replicas fail due to voids, then the die fails
                            # if any(redundant_bumpid_set.issubset(fail_bump_id_set) for net, redundant_bumpid_set in redundant_net_to_bumpids.items()):
                            #         print("Die fails due to redundant pad void overlap.")
                            #         waf_stack.die_stack_survival[die_ind] = False
                            #         bot_wafer.die_list[die_ind].survival, top_wafer.die_list[die_ind].survival = False, False
                            #         break

                # Proceed if die still survives
                if not waf_stack.die_stack_survival[die_ind] and not cfg.verbose:
                    continue
                
                '''
                Check the Cu gap, a true Monte Carlo simulator
                '''
                # Check the Cu expansion
                top_dish, bot_dish = Cu_gap_simulator(TOP_DISH_MEAN_nm, TOP_DISH_STD_nm, BOT_DISH_MEAN_nm, BOT_DISH_STD_nm, int(die.num_pads))
                Cu_gap_in_valid_pads = top_dish + bot_dish
                Cu_gap_map = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)
                Cu_gap_map[valid_pad_mask == 1] = Cu_gap_in_valid_pads

                # Calculate the safe range for single pad Cu recess
                if not os.path.exists(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp/' + cfg.DESIGN + "_dishing_bound_array_die_{}.npy".format(die_ind)) or cfg.DEBUG:
                    if not os.path.exists(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp/'):
                        os.makedirs(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp/')
                    # start_time = time.time()
                    valid_pad_dishing_bound_array = debond_dishing_bounds_calculator(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
                    # print("Dishing bound calculation time: {:.2f} seconds".format(time.time() - start_time))
                    np.save(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp/' + cfg.DESIGN + "_dishing_bound_array_die_{}.npy".format(die_ind), valid_pad_dishing_bound_array)
                else:
                    valid_pad_dishing_bound_array = np.load(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp/' + cfg.DESIGN + "_dishing_bound_array_die_{}.npy".format(die_ind))
                zeta_0 = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)
                zeta_1 = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)
                zeta_0[valid_pad_mask == 1] = - valid_pad_dishing_bound_array[:, 1] * 2 # lower limits of the sum of top and bottom Cu heights
                zeta_1[valid_pad_mask == 1] = - valid_pad_dishing_bound_array[:, 0] * 2 # upper limits of the sum of top and bottom Cu heights

                if cfg.verbose:
                    epoch_fail_map_dict['mechanical'] += ((Cu_gap_map > zeta_1) | (Cu_gap_map < zeta_0)).astype(int)

                # Check critical pad Cu gap
                critical_pad_Cu_gap = Cu_gap_map * die_critical_pad_bitmap      # Shape: (PAD_ARR_ROW, PAD_ARR_COL)
                if np.any(critical_pad_Cu_gap > zeta_1 * die_critical_pad_bitmap) or np.any(critical_pad_Cu_gap < zeta_0 * die_critical_pad_bitmap):
                    # print("Die fails due to critical pad Cu gap failure.")
                    bot_wafer.die_list[die_ind].survival, top_wafer.die_list[die_ind].survival = False, False
                    waf_stack.die_stack_survival[die_ind] = False
                    if cfg.verbose:
                        epoch_fail_vec_dict['mechanical'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                        epoch_fail_vec_dict['overall'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                    if not cfg.verbose:
                        continue
                
                # Check redundant pad Cu gap
                redundant_pad_Cu_gap = Cu_gap_map * die_redundant_pad_bitmap    # Shape: (PAD_ARR_ROW, PAD_ARR_COL)
                redundant_pad_fail_map[redundant_pad_Cu_gap > zeta_1 * die_redundant_pad_bitmap] = 1
                redundant_pad_fail_map[redundant_pad_Cu_gap < zeta_0 * die_redundant_pad_bitmap] = 1
                for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
                    tolerated_mechanical_failures = criticality_info[redundant_net]['tolerated_mechanical_failures']
                    num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
                    if num_fail_pad_in_net > tolerated_mechanical_failures:
                        bot_wafer.die_list[die_ind].survival, top_wafer.die_list[die_ind].survival = False, False
                        waf_stack.die_stack_survival[die_ind] = False
                        if cfg.verbose:
                            epoch_fail_vec_dict['mechanical'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                            epoch_fail_vec_dict['overall'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                        if not cfg.verbose:
                            break
                # # Get the fail bump indices
                # fail_bump_id = mapping_physical_to_bumpid[redundant_pad_fail_map == 1]
                # # Switch to set for easier checking
                # fail_bump_id_set = set(fail_bump_id.astype(int))

                # # Check every net connecting redundant pads, if all the redundant pad replicas fail due to voids, then the die fails
                # if any(redundant_bumpid_set.issubset(fail_bump_id_set) for net, redundant_bumpid_set in redundant_net_to_bumpids.items()):
                #     print("Die fails due to redundant pad Cu gap failure.")
                #     waf_stack.die_stack_survival[die_ind] = False
                #     bot_wafer.die_list[die_ind].survival, top_wafer.die_list[die_ind].survival = False, False
                #     break

                '''
                Check the ESD failure
                '''
                # TODO: ESD failure simulation to be implemented
                # Check if the die is in the wafer center
                die_center_x, die_center_y = die.die_center[0], die.die_center[1]
                if np.abs(die_center_x) < die.DIE_W_um / 2 and np.abs(die_center_y) < die.DIE_L_um / 2:
                    # Assume dies in the center will be the first contact point and have higher ESD hazard
                    # Check critical pads specifically for the ESD failure mechanisms (ESD-critical pads)
                    first_contact_pad_idx, survive_bool = esd_failure_simulator(
                                                    pad_coords_um=valid_die_pad_coords,
                                                    pad_size_um=PAD_TOP_R_um * 2,
                                                    top_wafer_radius_um=WAF_R_um,
                                                    top_dish_nm_ext=top_dish,
                                                    bot_dish_nm_ext=bot_dish,
                                                    tilt_x_mean_deg=TILT_X_MEAN_DEG,
                                                    tilt_x_std_deg=TILT_X_STD_DEG,
                                                    tilt_y_mean_deg=TILT_Y_MEAN_DEG,
                                                    tilt_y_std_deg=TILT_Y_STD_DEG,
                                                    )
                    if first_contact_pad_idx is not None and survive_bool == False:
                        r_idx, c_idx = first_contact_pad_idx // PAD_ARR_COL, first_contact_pad_idx % PAD_ARR_COL
                        if cfg.verbose:
                            epoch_fail_map_dict['ESD'][r_idx, c_idx] += 1
                        if die_esd_critical_pad_bitmap[r_idx, c_idx] == 1:
                            # print("Die fails due to ESD")
                            waf_stack.die_stack_survival[die_ind] = False
                            bot_wafer.die_list[die_ind].survival, top_wafer.die_list[die_ind].survival = False, False
                            if cfg.verbose:
                                epoch_fail_vec_dict['ESD'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                                epoch_fail_vec_dict['overall'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                            continue
                        for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
                            tolerated_esd_failures = criticality_info[redundant_net]['tolerated_esd_failures']
                            num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
                            if num_fail_pad_in_net > tolerated_esd_failures:
                                waf_stack.die_stack_survival[die_ind] = False
                                bot_wafer.die_list[die_ind].survival, top_wafer.die_list[die_ind].survival = False, False
                                if cfg.verbose:
                                    epoch_fail_vec_dict['ESD'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                                    epoch_fail_vec_dict['overall'][stack_ind, interface_ind: interface_ind+2, die_ind] = 1
                                break                    

            # Record the time
            # print("The time for checking wafer {} is {} seconds.".format(waf_ind, time.time() - start_time))
            # # print("The number of survival dies in the wafer is {}.".format(wafer.survival_die))
            # Draw the swhole wafer
            # wafer.draw_wafer_die(fig_size=(10, 10))
            # raise ValueError("Stop here")

        # One stack is done, calculate the die stack yield
        die_stack_yield = waf_stack.die_stack_survival.sum() / num_dies_per_wafer
        die_stack_yield_list.append(die_stack_yield)        # die stack yield for 
    

    return die_stack_yield_list, epoch_fail_map_dict, epoch_fail_vec_dict