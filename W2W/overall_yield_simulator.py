#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

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
    waf_list,
    WAF_R_um,
    system_translation_x_um,
    system_translation_y_um,
    system_rotation_rad,
    system_magnification_ppm,
    MAX_ALLOWED_MISALIGNMENT_um: float,
    PAD_ARR_W_um,
    PAD_ARR_L_um,
    PAD_ARR_ROW,
    PAD_ARR_COL,
    TOP_DISH_MEAN_nm,
    TOP_DISH_STD_nm,
    BOT_DISH_MEAN_nm,
    BOT_DISH_STD_nm,
    TILT_X_MEAN_DEG,
    TILT_X_STD_DEG,
    TILT_Y_MEAN_DEG,
    TILT_Y_STD_DEG,
    k_et,
    k_eb,
    T_R,
    T_anl,
    PITCH_r_um,
    PITCH_c_um,
    PAD_TOP_R_um,
    RANDOM_MISALIGNMENT_MEAN_um,
    RANDOM_MISALIGNMENT_STD_um,
    approximate_set,
    redundant_flag,
    pad_bitmap_collection,
):
    yield_list = []
    # print("The memory size of the waf_list is {} MB.".format(total_memory_mb(waf_list)))
    for waf_ind in range(len(waf_list)):
        # Record the time
        start_time = time.time()
        die_count = 0
        wafer = waf_list[waf_ind]
        # Read the critical pad bitmap
        die_critical_pad_bitmap = pad_bitmap_collection["CRITICAL_PAD_BITMAP"]
        # Read the redundant critical pad bitmap
        die_redundant_pad_bitmap = pad_bitmap_collection["REDUNDANT_PAD_BITMAP"]
        # Read the ESD-critical pad bitmap
        die_esd_critical_pad_bitmap = pad_bitmap_collection["ESD_CRITICAL_PAD_BITMAP"]
        # Read the redundant net to bump ids mapping
        redundant_net_to_bumpids = pad_bitmap_collection["redundant_net_to_bumpids"]

        critical_fail = 0
        redundant_fail = 0


        for die_ind, die in enumerate(wafer.die_list):
            die_count += 1
            if die_count % 10 == 0:
                print("Processing die {}/{}...".format(die_count, len(wafer.die_list)))
                print("Time taken for every 10 dies: {:.2f} seconds".format((time.time() - start_time) / die_count * 10))
                # start_time = time.time()
            redundant_pad_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))
            
            '''
            Check the overlay errors
            '''
            # # Check the pad misalignment
            die.pad_misalignment = die_pad_misalignment(die=die, 
                                                        base_pad_coords=wafer.base_pad_coords,
                                                        system_translation_x_um=system_translation_x_um[waf_ind],
                                                        system_translation_y_um=system_translation_y_um[waf_ind],
                                                        system_rotation_rad=system_rotation_rad[waf_ind],
                                                        system_magnification_ppm=system_magnification_ppm[waf_ind],
                                                        RANDOM_MISALIGNMENT_MEAN_um=RANDOM_MISALIGNMENT_MEAN_um,
                                                        RANDOM_MISALIGNMENT_STD_um=RANDOM_MISALIGNMENT_STD_um,
                                                        approximate_set=approximate_set,
                                                        redundant_flag=redundant_flag,
                                                        )
            if approximate_set == 1:
                # pad fail criteria: pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um
                die.pad_misalignment = die.pad_misalignment.reshape(die.PAD_ARR_ROW, die.PAD_ARR_COL)
                critical_pad_misalignment = die.pad_misalignment * die_critical_pad_bitmap      # Shape: (PAD_ARR_ROW, PAD_ARR_COL)
                # Check if any critical pad misalignment is greater than the maximum allowed misalignment
                if np.any(critical_pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um):
                    wafer.survival_die -= 1
                    die.survival = False
                    critical_fail += 1
                    # print("Fail due to critical pad misalignment.")
                    continue
                # Check if too many redundant pad misalignment is greater than the maximum allowed misalignment
                redundant_pad_misalignment = die.pad_misalignment * die_redundant_pad_bitmap    # Shape: (PAD_ARR_ROW, PAD_ARR_COL)
                redundant_pad_fail_map[redundant_pad_misalignment > MAX_ALLOWED_MISALIGNMENT_um] = 1 # 1: redundant pad fails, shape: (PAD_ARR_ROW, PAD_ARR_COL)
                # Get those failing pad indices
                failing_redundant_pad_ind = np.argwhere(redundant_pad_fail_map == 1)
                # Get the fail bump indices (specifically for UCIe mapping)
                fail_bump_id_set = set((failing_redundant_pad_ind[:, 0] * PAD_ARR_COL / 2 + failing_redundant_pad_ind[:, 1] // 2).astype(int))

            # Delete the die.pad_misalignment to save memory
            del die.pad_misalignment
        
            
            # Check every net connecting redundant pads, if all the redundant pad replicas fail, then the die fails
            for net, redundant_bumpid_set in redundant_net_to_bumpids.items():
                if redundant_bumpid_set.issubset(fail_bump_id_set):
                    wafer.survival_die -= 1
                    die.survival = False
                    redundant_fail += 1
                    break
            if not die.survival:
                continue
            
            '''
            Check the void defects
            '''
            # # Check the void overlap with the pad
            # Assuming wafer.voids is an array of shape (N, 3), where N is the number of voids. [x, y, r]
            # Critical pad bitmap is a 2D array of shape (PAD_ARR_ROW, PAD_ARR_COL) with 1s for critical pads and 0s for non-critical pads

            voids = np.array(wafer.voids)
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
                        die.voids_occur = True      # Will draw the die to green if it still survives
                        
                    # Get the critical pad bitmap for the pads we need to consider
                    check_critical_pad_bitmap = die_critical_pad_bitmap[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                    # Get the redundant critical pad bitmap for the pads we need to consider
                    check_redundant_pad_bitmap = die_redundant_pad_bitmap[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                    
                    # Check if any void overlaps with the critical pads
                    overlap_critical = overlap_void_pad_mask & check_critical_pad_bitmap.astype(bool)
                    if np.any(overlap_critical):
                        wafer.survival_die -= 1
                        die.survival = False
                    else:   # Voids overlapping with the redundant pads.
                        # Check if any void overlaps with the redundant critical pads
                        overlap_redundant = overlap_void_pad_mask & check_redundant_pad_bitmap.astype(bool) # shape (H, W)
                        # if overlap #pads is greater than a percentage of the total pads, then the die fails
                        redundant_pad_fail_map[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1][overlap_redundant] = 1
                        # Get those failing pad indices
                        failing_redundant_pad_ind = np.argwhere(redundant_pad_fail_map == 1)
                        # Get the fail bump indices (specifically for UCIe mapping)
                        fail_bump_id_set = set((failing_redundant_pad_ind[:, 0] * PAD_ARR_COL / 2 + failing_redundant_pad_ind[:, 1] // 2).astype(int))

                        # Check every net connecting redundant pads, if all the redundant pad replicas fail due to voids, then the die fails
                        for net, redundant_bumpid_set in redundant_net_to_bumpids.items():
                            if redundant_bumpid_set.issubset(fail_bump_id_set):
                                wafer.survival_die -= 1
                                die.survival = False
                                redundant_fail += 1
                                break

            # Proceed if die still survives
            if not die.survival:
                continue
            
            '''
            Check the Cu gap, a true Monte Carlo simulator
            '''
            # TODO: add the roughness parameters to Cain's model
            # Check the Cu expansion
            top_dish, bot_dish = Cu_gap_simulator(TOP_DISH_MEAN_nm, TOP_DISH_STD_nm, BOT_DISH_MEAN_nm, BOT_DISH_STD_nm, int(die.num_pads))
            Cu_gap = top_dish + bot_dish
            Cu_gap = Cu_gap.reshape(die.PAD_ARR_ROW, die.PAD_ARR_COL)
            # Calculate the safe range for Cu recess
            die_pad_coords = wafer.base_pad_coords + die.die_center
            dishing_bound_array = debond_dishing_bounds_calculator(cfg, die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
            zeta_1 = - dishing_bound_array[:, 0].reshape(die.PAD_ARR_ROW, die.PAD_ARR_COL)  # - upper Cu height limits
            zeta_0 = - dishing_bound_array[:, 1].reshape(die.PAD_ARR_ROW, die.PAD_ARR_COL)  # - lower Cu height limits

            # Check critical pad Cu gap
            critical_pad_Cu_gap = Cu_gap * die_critical_pad_bitmap      # Shape: (PAD_ARR_ROW, PAD_ARR_COL)
            if np.any(critical_pad_Cu_gap > zeta_1 * die_critical_pad_bitmap) or np.any(critical_pad_Cu_gap < zeta_0 * die_critical_pad_bitmap):
                wafer.survival_die -= 1
                die.survival = False
                continue
            
            # Check redundant pad Cu gap
            redundant_pad_Cu_gap = Cu_gap * die_redundant_pad_bitmap    # Shape: (PAD_ARR_ROW, PAD_ARR_COL)
            redundant_pad_fail_map[redundant_pad_Cu_gap > zeta_1 * die_redundant_pad_bitmap] = 1
            redundant_pad_fail_map[redundant_pad_Cu_gap < zeta_0 * die_redundant_pad_bitmap] = 1
            # Get those failing pad indices
            failing_redundant_pad_ind = np.argwhere(redundant_pad_fail_map == 1)
            # Get the fail bump indices (specifically for UCIe mapping)
            fail_bump_id_set = set((failing_redundant_pad_ind[:, 0] * PAD_ARR_COL / 2 + failing_redundant_pad_ind[:, 1] // 2).astype(int))

            # Check every net connecting redundant pads, if all the redundant pad replicas fail due to voids, then the die fails
            for net, redundant_bumpid_set in redundant_net_to_bumpids.items():
                if redundant_bumpid_set.issubset(fail_bump_id_set):
                    wafer.survival_die -= 1
                    die.survival = False
                    redundant_fail += 1
                    break

            '''
            Check the ESD failure
            '''
            # TODO: ESD failure simulation to be implemented
            # Check if the die is in the wafer center
            die_center_x, die_center_y = die.die_center[0], die.die_center[1]
            if np.abs(die_center_x) < die.DIE_W_um / 2 and np.abs(die_center_y) < die.DIE_L_um / 2:
                # Assume dies in the center will be the first contact point and have higher ESD hazard
                # Check critical pads specifically for the ESD failure mechanisms (ESD-critical pads)
                first_contact_pad_idx, survive_bool = esd_failure_simulator(pad_coords_um=die.pad_coords,
                                                pad_size_um=PAD_TOP_R_um * 2,
                                                top_die_w_um=die.DIE_W_um,
                                                top_die_h_um=die.DIE_L_um,
                                                top_dish_nm_ext=top_dish,
                                                bot_dish_nm_ext=bot_dish,
                                                tilt_x_mean_deg=TILT_X_MEAN_DEG,
                                                tilt_x_std_deg=TILT_X_STD_DEG,
                                                tilt_y_mean_deg=TILT_Y_MEAN_DEG,
                                                tilt_y_std_deg=TILT_Y_STD_DEG,
                                                )
                if first_contact_pad_idx is not None and survive_bool == False:
                    r_idx, c_idx = first_contact_pad_idx // PAD_ARR_COL, first_contact_pad_idx % PAD_ARR_COL
                    if die_esd_critical_pad_bitmap[r_idx, c_idx] == 1:
                        wafer.survival_die -= 1
                        die.survival = False
                        continue

            #check time for 10 dies
            if die_count % 10 == 9:
                print("The time for checking ten dies is {} seconds.".format(time.time() - start_time))

        # Record the time
        # print("The time for checking wafer {} is {} seconds.".format(waf_ind, time.time() - start_time))
        # # print("The number of survival dies in the wafer is {}.".format(wafer.survival_die))
        # Draw the swhole wafer
        # wafer.draw_wafer_die(fig_size=(30, 30))
        # raise ValueError("Stop here")
        # print("Critical pad fail: {}, Redundant pad fail: {}".format(critical_fail, redundant_fail))
        die_yield = wafer.survival_die / len(wafer.die_list)

        # print("The die yield of the wafer is {:.2f}%.".format(die_yield * 100))
        yield_list.append(die_yield)
        if (waf_ind + 1) % 1 == 0:
            print("Processing wafer {}/{}..., Current mean yield is {:.2f}%.".format(waf_ind + 1, len(waf_list), np.mean(yield_list) * 100))
            # Print the memory size of the waf_list
            # print("The memory size of the waf_list is {} MB.".format(total_memory_mb(waf_list)))
        # raise ValueError("Stop here")
    return yield_list       