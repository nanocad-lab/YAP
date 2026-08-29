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
from collections import defaultdict

from wafer_die_initialization import Die, Wafer, wafer_initialize
from overlay_yield_simulator import overlay_term_simulator, die_pad_misalignment
from Cu_gap_simulator import Cu_gap_simulator
from Cu_expansion_yield_calculator import Cu_expansion_yield_calculator


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


def clipped_pad_window(x_min, x_max, y_min, y_max, pitch_um, num_rows, num_cols):
    """Return a bitmap-aligned, half-open pad window clipped to the array."""
    col_start = int(np.clip(np.floor(x_min / pitch_um), 0, num_cols))
    col_stop = int(np.clip(np.ceil(x_max / pitch_um) + 1, 0, num_cols))
    bottom_row_start = int(np.clip(np.floor(y_min / pitch_um), 0, num_rows))
    bottom_row_stop = int(np.clip(np.ceil(y_max / pitch_um) + 1, 0, num_rows))
    if col_start >= col_stop or bottom_row_start >= bottom_row_stop:
        return None

    row_start = num_rows - bottom_row_stop
    row_stop = num_rows - bottom_row_start
    return row_start, row_stop, col_start, col_stop


def overall_yield_simulator(
    cfg,
    waf_list,
    WAF_R_um,
    system_translation_x_um,
    system_translation_y_um,
    system_rotation_rad,
    system_magnification_ppm,
    MAX_ALLOWED_MISALIGNMENT,
    zeta_0,
    zeta_1,
    PAD_ARR_W_um,
    PAD_ARR_L_um,
    PAD_ARR_ROW,
    PAD_ARR_COL,
    TOP_DISH_MEAN_nm,
    TOP_DISH_STD_nm,
    BOT_DISH_MEAN_nm,
    BOT_DISH_STD_nm,
    k_et,
    k_eb,
    T_R,
    T_anl,
    PITCH_um,
    PAD_TOP_R_um,
    RANDOM_MISALIGNMENT_MEAN_um,
    RANDOM_MISALIGNMENT_STD_um,
    redundant_survival_ratio,
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
        # The mapping of the redundant logical pads to the physical pads
        redundant_logical_to_physical_arr = pad_bitmap_collection["redundant_logical_to_physical_arr"]
        # The mapping of the redundant physical pads to the logical pads
        redundant_physical_to_logical_arr = pad_bitmap_collection["redundant_physical_to_logical_arr"]

        num_redundant_pads = pad_bitmap_collection["num_redundant_pads"]
        critical_fail = 0
        redundant_fail = 0


        for die, die_ind in zip(wafer.die_list, range(len(wafer.die_list))):
            die_count += 1
            if die_count % 10 == 0 or die_count == len(wafer.die_list):
                print(
                    "Processing die {}/{}..., Time taken for every 10 dies: {:.2f} seconds".format(
                        die_count,
                        len(wafer.die_list),
                        (time.time() - start_time) / die_count * 10,
                    ),
                    end="\r",
                    flush=True,
                )
                # start_time = time.time()
            redundant_pad_fail_map = np.zeros(
                (PAD_ARR_ROW, PAD_ARR_COL), dtype=bool
            ) if num_redundant_pads > 0 else np.empty((0, 0), dtype=bool)
            # Update the scoreboard for the redundant pads
            redundant_logical_scoreboard = np.ones(redundant_logical_to_physical_arr.shape[0], dtype=int) * redundant_logical_to_physical_arr.shape[1]
            
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
                # pad fail criteria: pad_misalignment >= MAX_ALLOWED_MISALIGNMENT
                die.pad_misalignment = die.pad_misalignment.reshape(die_critical_pad_bitmap.shape)
                critical_pad_misalignment = die.pad_misalignment * die_critical_pad_bitmap
                # Check if any critical pad misalignment is greater than the maximum allowed misalignment
                if np.any(critical_pad_misalignment >= MAX_ALLOWED_MISALIGNMENT):
                    wafer.survival_die -= 1
                    die.survival = False
                    critical_fail += 1
                    # print("Fail due to critical pad misalignment.")
                    continue
                # Check if too many redundant pad misalignment is greater than the maximum allowed misalignment
                if num_redundant_pads > 0:
                    redundant_pad_misalignment = die.pad_misalignment * die_redundant_pad_bitmap
                    redundant_pad_fail_map[redundant_pad_misalignment > MAX_ALLOWED_MISALIGNMENT] = 1
                    failing_pad_ind = np.argwhere(redundant_pad_misalignment > MAX_ALLOWED_MISALIGNMENT)
                    failing_physical_pad_inds = failing_pad_ind[:, 0] * PAD_ARR_COL + failing_pad_ind[:, 1]
                    failing_logical_pad_inds = redundant_physical_to_logical_arr[failing_physical_pad_inds]
                    failing_logical_pad_inds = failing_logical_pad_inds[failing_logical_pad_inds >= 0]
                    fail_counts = np.bincount(
                        failing_logical_pad_inds,
                        minlength=redundant_logical_scoreboard.shape[0],
                    )
                    redundant_logical_scoreboard -= fail_counts
            else:
                max_pad_misalignment = die.pad_misalignment
                # Check if any critical pad misalignment is greater than the maximum allowed misalignment
                if np.any(max_pad_misalignment >= MAX_ALLOWED_MISALIGNMENT):
                    wafer.survival_die -= 1
                    die.survival = False
                    continue
            
            

            # Delete the die.pad_misalignment to save memory
            del die.pad_misalignment
        
            
            # If all the redundant pad replicas fail, then the die fails
            if np.any(redundant_logical_scoreboard == 0):
                wafer.survival_die -= 1
                die.survival = False
                redundant_fail += 1
                # print("Fail due to all copies failing.")
                continue
            # Check if the number of redundant pads with misalignment is greater than the survival ratio
            if np.sum(redundant_pad_fail_map) > (1 - redundant_survival_ratio) * num_redundant_pads:
                wafer.survival_die -= 1
                die.survival = False
                redundant_fail += 1
                # print("Fail due to too many redundant pads with misalignment.")
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
                overlap_void_die_mask = distances < (voids[:, 2] + PAD_TOP_R_um) ** 2  # shape (N,)

                # Use critical pad bitmap and grid search to find if any void overlaps with the die
                if np.any(overlap_void_die_mask):
                    num_overlap_redundant_pads = 0
                    # Calculate the pad range we need to consider (critical, near the void)
                    # The i, j here are the indices of the pad array bitmap. The origin is the bottom left corner of the pad array box. 
                    # It is noticed that the origin of the bitmap is the top left corner of the pad array box. Switching is needed.
                    in_die_voids = voids[overlap_void_die_mask]
                    i_coord_min = min(in_die_voids[:, 0] - in_die_voids[:, 2] - PAD_TOP_R_um - pad_array_box_x)
                    i_coord_max = max(in_die_voids[:, 0] + in_die_voids[:, 2] + PAD_TOP_R_um - pad_array_box_x)
                    j_coord_min = min(in_die_voids[:, 1] - in_die_voids[:, 2] - PAD_TOP_R_um - pad_array_box_y)
                    j_coord_max = max(in_die_voids[:, 1] + in_die_voids[:, 2] + PAD_TOP_R_um - pad_array_box_y)
                    window = clipped_pad_window(
                        i_coord_min, i_coord_max, j_coord_min, j_coord_max,
                        PITCH_um, PAD_ARR_ROW, PAD_ARR_COL,
                    )
                    if window is None:
                        continue
                    row_start, row_stop, col_start, col_stop = window

                    check_pad_x_coords = pad_array_box_x + np.arange(col_start, col_stop) * PITCH_um
                    check_pad_y_coords = pad_array_box_y + (PAD_ARR_ROW - 1 - np.arange(row_start, row_stop)) * PITCH_um
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
                    # # Draw the overlap_void_pad_mask
                    # plt.imshow(overlap_void_pad_mask, cmap='gray', origin='lower')
                    # plt.title("Overlap void pad mask")
                    # plt.show()

                    # Get the critical pad bitmap for the pads we need to consider
                    check_critical_pad_bitmap = die_critical_pad_bitmap[row_start:row_stop, col_start:col_stop]
                    # # Draw the critical pad bitmap
                    # plt.imshow(check_critical_pad_bitmap, cmap='gray')
                    # plt.title("Check critical pad bitmap")
                    # plt.show()
                    # Get the redundant critical pad bitmap for the pads we need to consider
                    check_redundant_pad_bitmap = die_redundant_pad_bitmap[row_start:row_stop, col_start:col_stop]
                    # # Draw the redundant critical pad bitmap
                    # plt.imshow(check_redundant_pad_bitmap, cmap='gray')
                    # plt.title("Check redundant pad bitmap")
                    # plt.show()
                    
                    # Check if any void overlaps with the critical pads
                    overlap_critical = overlap_void_pad_mask & check_critical_pad_bitmap.astype(bool)
                    if np.any(overlap_critical):
                        # print("Overlapping with the critical pads.")
                        wafer.survival_die -= 1
                        die.survival = False
                    elif num_redundant_pads > 0:
                        # print("Overlapping with the redundant pads.")
                        # Check if any void overlaps with the redundant critical pads
                        overlap_redundant = overlap_void_pad_mask & check_redundant_pad_bitmap.astype(bool)
                        # if overlap #pads is greater than a percentage of the total pads, then the die fails
                        num_overlap_redundant_pads += np.sum(overlap_redundant)
                        redundant_pad_fail_map[row_start:row_stop, col_start:col_stop][overlap_redundant] = 1
                        # Get those failing pad indices
                        failing_pad_ind = np.argwhere(overlap_redundant)
                        # make the coords global
                        failing_pad_ind += np.array([row_start, col_start])
                        # Get the physical pad indices
                        failing_physical_pad_inds = failing_pad_ind[:, 0] * PAD_ARR_COL + failing_pad_ind[:, 1] 
                        # Use the physical -> logical mapping to get the reduce the logical score
                        failing_logical_pad_inds = redundant_physical_to_logical_arr[failing_physical_pad_inds]
                        # Extract those pads with a logical id that is not -1 (not PG pads)
                        failing_logical_pad_inds = failing_logical_pad_inds[failing_logical_pad_inds >= 0] # Those PG pads logical ids are -1
                        # Update the scoreboard
                        fail_counts = np.bincount(failing_logical_pad_inds, minlength=redundant_logical_scoreboard.shape[0])
                        redundant_logical_scoreboard -= fail_counts

                        # If all the copied redundant pads fail, then the die fails
                        if np.any(redundant_logical_scoreboard == 0):
                            # print("Here is the pad failing due to all copies failing.")
                            wafer.survival_die -= 1
                            die.survival = False
                        if np.sum(redundant_pad_fail_map) > (1 - redundant_survival_ratio) * num_redundant_pads:
                            # print("The number of redundant pads overlapping with the void is {}.".format(num_overlap_redundant_pads))
                            wafer.survival_die -= 1
                            die.survival = False

            # Proceed if die still survives
            if not die.survival:
                continue
            
            '''
            Check the Cu gap, a true Monte Carlo simulator
            '''
            # # Check the Cu expansion
            # Cu_gap = Cu_gap_simulator(TOP_DISH_MEAN_nm, TOP_DISH_STD_nm, BOT_DISH_MEAN_nm, BOT_DISH_STD_nm, int(die.num_pads))
            # Cu_gap = Cu_gap.reshape(die_critical_pad_bitmap.shape)
            # # Check critical pad Cu gap
            # critical_pad_Cu_gap = Cu_gap * die_critical_pad_bitmap
            # if critical_pad_Cu_gap.min() < -zeta_0 or critical_pad_Cu_gap.max() > -zeta_1:
            #     wafer.survival_die -= 1
            #     die.survival = False
            #     continue
            # # Check redundant pad Cu gap
            # redundant_pad_Cu_gap = Cu_gap * die_redundant_pad_bitmap
            # num_redundant_pad_over_Cu_gap = np.sum(redundant_pad_Cu_gap > -zeta_1) + np.sum(redundant_pad_Cu_gap < -zeta_0)
            # redundant_pad_fail_map[redundant_pad_Cu_gap > -zeta_1] = 1
            # redundant_pad_fail_map[redundant_pad_Cu_gap < -zeta_0] = 1
            # # Get those failing pad indices
            # failing_pad_ind = np.concatenate((np.argwhere(redundant_pad_Cu_gap > -zeta_1), np.argwhere(redundant_pad_Cu_gap < -zeta_0)), axis=0)
            # # Get the physical pad indices
            # failing_physical_pad_inds = failing_pad_ind[:, 0] * PAD_ARR_COL + failing_pad_ind[:, 1]
            # # Use the physical -> logical mapping to get the reduce the logical score
            # failing_logical_pad_inds = redundant_physical_to_logical_arr[failing_physical_pad_inds]
            # # Extract those pads with a logical id that is not -1 (not PG pads)
            # failing_logical_pad_inds = failing_logical_pad_inds[failing_logical_pad_inds >= 0] # Those PG pads logical ids are -1
            # # Update the scoreboard
            # fail_counts = np.bincount(failing_logical_pad_inds, minlength=redundant_logical_scoreboard.shape[0])
            # redundant_logical_scoreboard -= fail_counts

            # # If all the copied redundant pads fail, then the die fails
            # if np.any(redundant_logical_scoreboard == 0) or np.sum(redundant_pad_fail_map) > (1 - redundant_survival_ratio) * num_redundant_pads:
            #     wafer.survival_die -= 1
            #     die.survival = False
            #     continue

            # #check time for 10 dies
            # if die_count % 10 == 9:
            #     print("The time for checking ten dies is {} seconds.".format(time.time() - start_time))

        # Record the time
        # print("The time for checking wafer {} is {} seconds.".format(waf_ind, time.time() - start_time))
        # # print("The number of survival dies in the wafer is {}.".format(wafer.survival_die))
        # Draw the swhole wafer
        # wafer.draw_wafer_die(fig_size=(30, 30))
        # raise ValueError("Stop here")
        # print("Critical pad fail: {}, Redundant pad fail: {}".format(critical_fail, redundant_fail))
        if die_count:
            print()
        die_yield = wafer.survival_die / len(wafer.die_list)
        # Because there are too many pads! Cu pad recess height simulation will be very slow!
        # Hence, we will not consider the Cu pad recess height simulation here.
        # We use Cu yield model to calculate the yield.
        Cu_expansion_yield = Cu_expansion_yield_calculator(
            cfg=cfg,
            wafer=wafer,
            TOP_DISH_MEAN_nm=TOP_DISH_MEAN_nm,
            TOP_DISH_STD_nm=TOP_DISH_STD_nm,
            BOT_DISH_MEAN_nm=BOT_DISH_MEAN_nm,
            BOT_DISH_STD_nm=BOT_DISH_STD_nm,
            k_et=k_et,
            k_eb=k_eb,
            T_R=T_R,
            T_anl=T_anl,
            pad_bitmap_collection=pad_bitmap_collection,
            pad_yield_flag=False,
        )
        die_yield *= Cu_expansion_yield
        # print("The die yield of the wafer is {:.2f}%.".format(die_yield * 100))
        yield_list.append(die_yield)
        if (waf_ind + 1) % 1 == 0:
            print("Processing wafer {}/{}..., Current mean yield is {:.4f}%.".format(waf_ind + 1, len(waf_list), np.mean(yield_list) * 100))
            # Print the memory size of the waf_list
            # print("The memory size of the waf_list is {} MB.".format(total_memory_mb(waf_list)))
        # raise ValueError("Stop here")
    return yield_list       
