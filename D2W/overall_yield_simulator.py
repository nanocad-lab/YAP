#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import matplotlib.pyplot as plt
import time

from wafer_die_initialization import Die, Wafer, die_initialize
from overlay_yield_simulator import overlay_term_simulator, die_pad_misalignment
from Cu_gap_simulator import Cu_gap_simulator
from Cu_expansion_yield_calculator import Cu_expansion_yield_calculator


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
    die_list,
    NUM_DIES,
    base_pad_coords,
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
    DIE_W_um,
    DIE_L_um,
    TOP_DISH_MEAN_nm,
    TOP_DISH_STD_nm,
    BOT_DISH_MEAN_nm,
    BOT_DISH_STD_nm,
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
    die_count = 0
    safe_die_count = 0
    die_critical_pad_bitmap = pad_bitmap_collection["CRITICAL_PAD_BITMAP"]
    die_redundant_pad_bitmap = pad_bitmap_collection["REDUNDANT_PAD_BITMAP"]
    redundant_logical_to_physical_arr = pad_bitmap_collection["redundant_logical_to_physical_arr"]
    redundant_physical_to_logical_arr = pad_bitmap_collection["redundant_physical_to_logical_arr"]
    num_redundant_pads = pad_bitmap_collection["num_redundant_pads"]

    for die_ind in range(NUM_DIES):
        # # check start time
        # start_time = time.time()
        die = die_list[die_ind]
        die_count += 1
        if die_count % 100 == 0 or die_count == len(die_list):
            print(
                "Processing die {}/{}...".format(die_count, len(die_list)),
                end="\r",
                flush=True,
            )
        critical_fail = 0
        redundant_fail = 0

        redundant_pad_fail_map = np.zeros(
            (PAD_ARR_ROW, PAD_ARR_COL), dtype=bool
        ) if num_redundant_pads > 0 else np.empty((0, 0), dtype=bool)
        # Make a scoreboard for the redundant pads (redundant_logical_to_physical_arr.shape[0]), initialized to redundant_logical_to_physical_arr.shape[1]
        redundant_logical_scoreboard = np.ones(redundant_logical_to_physical_arr.shape[0], dtype=int) * redundant_logical_to_physical_arr.shape[1]

        """
        Check the overlay errors
        """
        # Check the pad misalignment
        die.pad_misalignment = die_pad_misalignment(die=die, 
                                                    base_pad_coords=base_pad_coords,
                                                    system_translation_x_um=system_translation_x_um[die_ind],
                                                    system_translation_y_um=system_translation_y_um[die_ind],
                                                    system_rotation_rad=system_rotation_rad[die_ind],
                                                    system_magnification_ppm=system_magnification_ppm[die_ind],
                                                    RANDOM_MISALIGNMENT_MEAN_um=RANDOM_MISALIGNMENT_MEAN_um,
                                                    RANDOM_MISALIGNMENT_STD_um=RANDOM_MISALIGNMENT_STD_um,
                                                    approximate_set=approximate_set,
                                                    redundant_flag=redundant_flag,
                                                    )
        if approximate_set == 1:
            # die fail criteria: any pad_misalignment >= MAX_ALLOWED_MISALIGNMENT
            die.pad_misalignment = die.pad_misalignment.reshape(die_critical_pad_bitmap.shape)
            critical_pad_misalignment = die.pad_misalignment * die_critical_pad_bitmap
            # Check if any critical pad misalignment is greater than the maximum allowed misalignment
            if np.any(critical_pad_misalignment >= MAX_ALLOWED_MISALIGNMENT):
                die.survival = False
                continue
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
            max_pad_misalignment = die.pad_misalignment     # Misalignment at the edge of the die
            # Check if any critical pad misalignment is greater than the maximum allowed misalignment
            if np.any(max_pad_misalignment >= MAX_ALLOWED_MISALIGNMENT):
                die.survival = False
                continue            

        # Delete the die.pad_misalignment to save memory
        del die.pad_misalignment

        # If all the redundant pad replicas fail, then the die fails
        if np.any(redundant_logical_scoreboard == 0):
            die.survival = False
            redundant_fail += 1
            # print("Fail due to all copies failing.")
            continue
        # Check if the number of redundant pads with misalignment is greater than the survival ratio
        if np.sum(redundant_pad_fail_map) > (1 - redundant_survival_ratio) * num_redundant_pads:
            die.survival = False
            redundant_fail += 1
            # print("Fail due to too many redundant pads with misalignment.")
            continue

        """
        Check the void defects
        """
        ## Check the void overlap with the pad
        # Assuming wafer.voids is an array of shape (N, 3), where N is the number of voids. [x, y, r]
        # Critical pad bitmap is a 2D array of shape (PAD_ARR_ROW, PAD_ARR_COL) with 1s for critical pads and 0s for non-critical pads
        voids = np.array(die.voids)
        if voids.size > 0:
            # Coordinates and dimensions of the die pad array box
            pad_array_box_x = die.pad_array_box[2][0]
            pad_array_box_y = die.pad_array_box[2][1]

            # Calculate closest x and y distances for all voids simultaneously
            closest_x = np.maximum(pad_array_box_x, np.minimum(voids[:, 0], pad_array_box_x + PAD_ARR_W_um))
            closest_y = np.maximum(pad_array_box_y, np.minimum(voids[:, 1], pad_array_box_y + PAD_ARR_L_um))

            # Calculate distance from each void to the closest point on the pad array box
            distances = np.sqrt((closest_x - voids[:, 0]) ** 2 + (closest_y - voids[:, 1]) ** 2)

            # Create a mask for voids overlapping with the pad array box
            overlapping_mask = distances < voids[:, 2] + PAD_TOP_R_um

            # Use critical pad bitmap and grid search to find if any void overlaps with the critical pads
            if np.any(overlapping_mask):
                num_overlap_redundant_pads = 0
                
                # Calculate the pad range we need to consider (critical, near the void)
                for void_index, void in enumerate(voids[overlapping_mask]):
                    # Calculate the pad range we need to consider (critical, near the void)
                    # The i, j here are the indices of the pad array bitmap. The origin is the bottom left corner of the pad array box. 
                    # It is noticed that the origin of the bitmap is the top left corner of the pad array box. Switching is needed.
                    i_coords_min = void[0] - void[2] - PAD_TOP_R_um - pad_array_box_x
                    i_coords_max = void[0] + void[2] + PAD_TOP_R_um - pad_array_box_x
                    j_coords_min = void[1] - void[2] - PAD_TOP_R_um - pad_array_box_y
                    j_coords_max = void[1] + void[2] + PAD_TOP_R_um - pad_array_box_y
                    window = clipped_pad_window(
                        i_coords_min, i_coords_max, j_coords_min, j_coords_max,
                        PITCH_um, PAD_ARR_ROW, PAD_ARR_COL,
                    )
                    if window is None:
                        continue
                    row_start, row_stop, col_start, col_stop = window

                    check_pad_x_coords = pad_array_box_x + np.arange(col_start, col_stop) * PITCH_um
                    check_pad_y_coords = pad_array_box_y + (PAD_ARR_ROW - 1 - np.arange(row_start, row_stop)) * PITCH_um
                    check_pad_x_mesh, check_pad_y_mesh = np.meshgrid(check_pad_x_coords, check_pad_y_coords, indexing='xy')

                    # Calculate the distance from the void to the closest point on the critical pads
                    dist_sq = (check_pad_x_mesh - void[0]) ** 2 + (check_pad_y_mesh - void[1]) ** 2 # Shape (H, W)
                    overlap_void_pad_mask = (dist_sq < (void[2] + PAD_TOP_R_um) ** 2)      # shape (H, W)
                    if np.any(overlap_void_pad_mask):
                        die.voids_occur = True  # Will draw the die to green if it still survives

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
                        die.survival = False
                    elif num_redundant_pads > 0:
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
                            die.survival = False
                        if np.sum(redundant_pad_fail_map) > (1 - redundant_survival_ratio) * num_redundant_pads:
                            # print("The number of redundant pads overlapping with the void is {}.".format(num_overlap_redundant_pads))
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
        #     die.survival = False
        #     continue


        if die.survival:
            safe_die_count += 1

        # # Check the time taken for each die
        # end_time = time.time()
        # time_taken = end_time - start_time
        # print("Time taken for die {}: {:.2f} seconds".format(die_ind, time_taken))
    # raise ValueError("Test error")
    die_yield = safe_die_count / die_count
    Cu_expansion_yield, _ = Cu_expansion_yield_calculator(
        cfg=cfg,
        die=die_list[0],
        TOP_DISH_MEAN_nm=TOP_DISH_MEAN_nm,
        TOP_DISH_STD_nm=TOP_DISH_STD_nm,
        BOT_DISH_MEAN_nm=BOT_DISH_MEAN_nm,
        BOT_DISH_STD_nm=BOT_DISH_STD_nm,
        k_et=cfg.k_et,
        k_eb=cfg.k_eb,
        T_R=cfg.T_R,
        T_anl=cfg.T_anl,
        pad_bitmap_collection=pad_bitmap_collection,
        pad_yield_flag=False,
    )
    die_yield *= Cu_expansion_yield
    # print("The yield of dies is {:.2f}%.".format(die_yield * 100))
    yield_list.append(die_yield)

    if die_count:
        print()
    return yield_list
