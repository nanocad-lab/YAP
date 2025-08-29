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


def overall_yield_simulator(
    waf_list,
    WAF_R,
    system_translation_x,
    system_translation_y,
    system_rotation,
    system_magnification,
    MAX_ALLOWED_MISALIGNMENT,
    zeta_0,
    zeta_1,
    PAD_ARR_W,
    PAD_ARR_L,
    PAD_ARR_ROW,
    PAD_ARR_COL,
    TOP_DISH_MEAN,
    TOP_DISH_STD,
    BOT_DISH_MEAN,
    BOT_DISH_STD,
    k_et,
    k_eb,
    T_R,
    T_anl,
    PITCH,
    PAD_TOP_R,
    RANDOM_MISALIGNMENT_MEAN,
    RANDOM_MISALIGNMENT_STD,
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

        num_redundant_pads = np.sum(die_redundant_pad_bitmap)
        critical_fail = 0
        redundant_fail = 0


        for die, die_ind in zip(wafer.die_list, range(len(wafer.die_list))):
            die_count += 1
            if die_count % 10 == 0:
                print("Processing die {}/{}...".format(die_count, len(wafer.die_list)))
                print("Time taken for every 10 dies: {:.2f} seconds".format((time.time() - start_time) / die_count * 10))
                # start_time = time.time()
            redundant_pad_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))
            # Update the scoreboard for the redundant pads
            redundant_logical_scoreboard = np.ones(redundant_logical_to_physical_arr.shape[0], dtype=int) * redundant_logical_to_physical_arr.shape[1]
            
            '''
            Check the overlay errors
            '''
            # # Check the pad misalignment
            die.pad_misalignment = die_pad_misalignment(die=die, 
                                                        base_pad_coords=wafer.base_pad_coords,
                                                        system_translation_x=system_translation_x[waf_ind],
                                                        system_translation_y=system_translation_y[waf_ind],
                                                        system_rotation=system_rotation[waf_ind],
                                                        system_magnification=system_magnification[waf_ind],
                                                        RANDOM_MISALIGNMENT_MEAN=RANDOM_MISALIGNMENT_MEAN,
                                                        RANDOM_MISALIGNMENT_STD=RANDOM_MISALIGNMENT_STD,
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
                redundant_pad_misalignment = die.pad_misalignment * die_redundant_pad_bitmap
                num_redundant_pad_over_misalignment = np.sum(redundant_pad_misalignment > MAX_ALLOWED_MISALIGNMENT)
                redundant_pad_fail_map[redundant_pad_misalignment > MAX_ALLOWED_MISALIGNMENT] = 1 # 1: redundant pad fails
                # Get those failing pad indices
                failing_pad_ind = np.argwhere(redundant_pad_misalignment > MAX_ALLOWED_MISALIGNMENT)
                # Get the physical pad indices
                failing_physical_pad_inds = failing_pad_ind[:, 0] * PAD_ARR_COL + failing_pad_ind[:, 1]
                # Use the physical -> logical mapping to get the reduce the logical score
                failing_logical_pad_inds = redundant_physical_to_logical_arr[failing_physical_pad_inds]
                # Extract those pads with a logical id that is not -1 (not PG pads)
                failing_logical_pad_inds = failing_logical_pad_inds[failing_logical_pad_inds >= 0] # Those PG pads logical ids are -1
                # Update the scoreboard
                fail_counts = np.bincount(failing_logical_pad_inds, minlength=redundant_logical_scoreboard.shape[0])
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
                closest_x = np.maximum(pad_array_box_x, np.minimum(voids[:, 0], pad_array_box_x + PAD_ARR_W))
                closest_y = np.maximum(pad_array_box_y, np.minimum(voids[:, 1], pad_array_box_y + PAD_ARR_L))

                # Calculate distance from each void to the closest point on the pad array box
                distances = (closest_x - voids[:, 0]) ** 2 + (closest_y - voids[:, 1]) ** 2

                # Create a mask for voids overlapping with the pad array box
                overlap_void_die_mask = distances < voids[:, 2] ** 2  # shape (N,)

                # Use critical pad bitmap and grid search to find if any void overlaps with the die
                if np.any(overlap_void_die_mask):
                    num_overlap_redundant_pads = 0
                    # Calculate the pad range we need to consider (critical, near the void)
                    # The i, j here are the indices of the pad array bitmap. The origin is the bottom left corner of the pad array box. 
                    # It is noticed that the origin of the bitmap is the top left corner of the pad array box. Switching is needed.
                    in_die_voids = voids[overlap_void_die_mask]
                    i_coord_min = min(in_die_voids[:, 0] - in_die_voids[:, 2] - PAD_TOP_R - pad_array_box_x)
                    i_coord_max = max(in_die_voids[:, 0] + in_die_voids[:, 2] + PAD_TOP_R - pad_array_box_x)
                    j_coord_min = min(in_die_voids[:, 1] - in_die_voids[:, 2] - PAD_TOP_R - pad_array_box_y)
                    j_coord_max = max(in_die_voids[:, 1] + in_die_voids[:, 2] + PAD_TOP_R - pad_array_box_y)
                    i_min = max(0, int(np.floor(i_coord_min / PITCH)))  # (col_start)
                    i_max = min(PAD_ARR_ROW-1, int(np.ceil(i_coord_max / PITCH)))   # H = i_max - i_min + 1 (col_end)
                    j_min = max(0, int(np.floor(j_coord_min / PITCH)))  # (row_start)
                    j_max = min(PAD_ARR_COL-1, int(np.ceil(j_coord_max / PITCH)))   # W = j_max - j_min + 1 (row_end)

                    check_pad_x_coords = pad_array_box_x + np.arange(i_min, i_max+1) * PITCH
                    check_pad_y_coords = pad_array_box_y + np.arange(j_min, j_max+1) * PITCH
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
                    overlap_void_pad_mask = dist_sq < (voids_r + PAD_TOP_R) ** 2 # shape (N, H, W)
                    overlap_void_pad_mask = np.any(overlap_void_pad_mask, axis=0)  # shape (H, W)
                    if np.any(overlap_void_pad_mask):
                        die.voids_occur = True      # Will draw the die to green if it still survives
                    # # Draw the overlap_void_pad_mask
                    # plt.imshow(overlap_void_pad_mask, cmap='gray', origin='lower')
                    # plt.title("Overlap void pad mask")
                    # plt.show()

                    # Get the critical pad bitmap for the pads we need to consider
                    check_critical_pad_bitmap = die_critical_pad_bitmap[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                    # # Draw the critical pad bitmap
                    # plt.imshow(check_critical_pad_bitmap, cmap='gray')
                    # plt.title("Check critical pad bitmap")
                    # plt.show()
                    # Get the redundant critical pad bitmap for the pads we need to consider
                    check_redundant_pad_bitmap = die_redundant_pad_bitmap[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
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
                    else:
                        # print("Overlapping with the redundant pads.")
                        # Check if any void overlaps with the redundant critical pads
                        overlap_redundant = overlap_void_pad_mask & check_redundant_pad_bitmap.astype(bool)
                        # if overlap #pads is greater than a percentage of the total pads, then the die fails
                        num_overlap_redundant_pads += np.sum(overlap_redundant)
                        redundant_pad_fail_map[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1][overlap_redundant] = 1
                        # Get those failing pad indices
                        failing_pad_ind = np.argwhere(overlap_redundant)
                        # make the coords global
                        failing_pad_ind += np.array([PAD_ARR_ROW-j_max-1, i_min])
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
            # Cu_gap = Cu_gap_simulator(TOP_DISH_MEAN, TOP_DISH_STD, BOT_DISH_MEAN, BOT_DISH_STD, int(die.num_pad))
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
        die_yield = wafer.survival_die / len(wafer.die_list)
        # Because there are too many pads! Cu pad recess height simulation will be very slow!
        # Hence, we will not consider the Cu pad recess height simulation here.
        # We use Cu yield model to calculate the yield.
        Cu_expansion_yield = Cu_expansion_yield_calculator(
            top_dish_mean=TOP_DISH_MEAN,
            top_dish_std=TOP_DISH_STD,
            bot_dish_mean=BOT_DISH_MEAN,
            bot_dish_std=BOT_DISH_STD,
            k_et=k_et,
            k_eb=k_eb,
            T_R=T_R,
            T_anl=T_anl,
            zeta_1=zeta_1,
            num_critical_pads      = pad_bitmap_collection["num_critical_pads"],
            num_redundant_logical_pads = pad_bitmap_collection["num_redundant_logical_pads"],
            redundant_logical_pad_copy = pad_bitmap_collection["redundant_logical_pad_copy"],
        )
        die_yield *= Cu_expansion_yield
        # print("The die yield of the wafer is {:.2f}%.".format(die_yield * 100))
        yield_list.append(die_yield)
        if (waf_ind + 1) % 1 == 0:
            print("Processing wafer {}/{}..., Current mean yield is {:.2f}%.".format(waf_ind + 1, len(waf_list), np.mean(yield_list) * 100))
            # Print the memory size of the waf_list
            # print("The memory size of the waf_list is {} MB.".format(total_memory_mb(waf_list)))
        # raise ValueError("Stop here")
    return yield_list       