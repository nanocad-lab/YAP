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
from roughness_parameters import roughness_parameters

def overall_yield_simulator(
    cfg,
    die_list: list,
    NUM_DIES: int,
    base_pad_coords: np.ndarray,
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
    PITCH_c_um: float,
    PITCH_r_um: float,
    PAD_TOP_R_um: float,
    RANDOM_MISALIGNMENT_MEAN_um: float,
    RANDOM_MISALIGNMENT_STD_um: float,
    approximate_set: int,
    redundant_flag: int,
    pad_bitmap_collection: dict,
):
    yield_list = []
    die_count = 0
    safe_die_count = 0
    # Read the critical pad bitmap
    for die_ind in range(NUM_DIES):
        # # check start time
        # start_time = time.time()
        if die_count % 100 == 0:
            print("Processing die {}/{}...".format(die_count, len(die_list)))
        die = die_list[die_ind]
        die_count += 1
        # Read the critical pad bitmap
        die_critical_pad_bitmap = pad_bitmap_collection["CRITICAL_PAD_BITMAP"]
        # Read the redundant critical pad bitmap
        die_redundant_pad_bitmap = pad_bitmap_collection["REDUNDANT_PAD_BITMAP"]
        # Read the redundant net to bump ids mapping
        redundant_net_to_bumpids = pad_bitmap_collection["redundant_net_to_bumpids"]

        critical_fail = 0
        redundant_fail = 0

        redundant_pad_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))

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
            # die fail criteria: any pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um
            die.pad_misalignment = die.pad_misalignment.reshape(die_critical_pad_bitmap.shape)
            critical_pad_misalignment = die.pad_misalignment * die_critical_pad_bitmap
            # Check if any critical pad misalignment is greater than the maximum allowed misalignment
            if np.any(critical_pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um):
                die.survival = False
                critical_fail += 1
                continue
            # Check if too many redundant pad misalignment is greater than the maximum allowed misalignment
            redundant_pad_misalignment = die.pad_misalignment * die_redundant_pad_bitmap
            num_redundant_pad_over_misalignment = np.sum(redundant_pad_misalignment > MAX_ALLOWED_MISALIGNMENT_um)
            redundant_pad_fail_map[redundant_pad_misalignment > MAX_ALLOWED_MISALIGNMENT_um] = 1   # 1: redundant pad fails
            # Get those failing pad indices
            failing_redundant_pad_ind = np.argwhere(redundant_pad_misalignment > MAX_ALLOWED_MISALIGNMENT_um)
            # Get the fail bump indices (specifically for UCIe mapping)
            fail_bump_id_set = set((failing_redundant_pad_ind[:, 0] * PAD_ARR_COL / 2 + failing_redundant_pad_ind[:, 1] // 2).astype(int))      

        # Delete the die.pad_misalignment to save memory
        del die.pad_misalignment

        # If all the redundant pad replicas fail, then the die fails
        for net, redundant_bump_ids in redundant_net_to_bumpids.items():
            if redundant_bump_ids.issubset(fail_bump_id_set):
                die.survival = False
                redundant_fail += 1
                break
        if not die.survival:
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
            overlapping_mask = distances < voids[:, 2]

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
                    i_min = max(0,              int(np.floor(i_coords_min / PITCH_c_um)))     # (col_start)
                    i_max = min(PAD_ARR_COL-1,  int(np.ceil (i_coords_max / PITCH_c_um))) # H = i_max - i_min + 1 (col_end)
                    j_min = max(0,              int(np.floor(j_coords_min / PITCH_r_um)))     # (row_start)
                    j_max = min(PAD_ARR_ROW-1,  int(np.ceil (j_coords_max / PITCH_r_um))) # W = j_max - j_min + 1 (row_end)

                    check_pad_x_coords = pad_array_box_x + np.arange(i_min, i_max+1) * PITCH_c_um
                    check_pad_y_coords = pad_array_box_y + np.arange(j_min, j_max+1) * PITCH_r_um
                    check_pad_x_mesh, check_pad_y_mesh = np.meshgrid(check_pad_x_coords, check_pad_y_coords, indexing='xy')

                    # Calculate the distance from the void to the closest point on the critical pads
                    dist_sq = (check_pad_x_mesh - void[0]) ** 2 + (check_pad_y_mesh - void[1]) ** 2 # Shape (H, W)
                    overlap_void_pad_mask = (dist_sq < (void[2] + PAD_TOP_R_um) ** 2)      # shape (H, W)
                    if np.any(overlap_void_pad_mask):
                        die.voids_occur = True  # Will draw the die to green if it still survives

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
                        die.survival = False
                    else:
                        # Check if any void overlaps with the redundant critical pads
                        overlap_redundant = overlap_void_pad_mask & check_redundant_pad_bitmap.astype(bool)
                        # if overlap #pads is greater than a percentage of the total pads, then the die fails
                        num_overlap_redundant_pads += np.sum(overlap_redundant)
                        redundant_pad_fail_map[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1][overlap_redundant] = 1
                        # Get those failing pad indices
                        failing_redundant_pad_ind = np.argwhere(overlap_redundant)
                        # make the coords global
                        failing_redundant_pad_ind += np.array([PAD_ARR_ROW-j_max-1, i_min])
                        # Get the fail bump indices (specifically for UCIe mapping)
                        fail_bump_id_set = set((failing_redundant_pad_ind[:, 0] * PAD_ARR_COL / 2 + failing_redundant_pad_ind[:, 1] // 2).astype(int))
                
                        # Check every net connecting redundant pads, if all the redundant pad replicas fail due to voids, then the die fails
                        for net, redundant_bump_ids in redundant_net_to_bumpids.items():
                            if redundant_bump_ids.issubset(fail_bump_id_set):
                                die.survival = False
                                break
                    if not die.survival:
                        break
                    

        # Proceed if die still survives
        if not die.survival:
            continue

        
        '''
        Check the Cu gap, a true Monte Carlo simulator
        '''
        # Check the Cu expansion
        Cu_gap = Cu_gap_simulator(TOP_DISH_MEAN_nm, TOP_DISH_STD_nm, BOT_DISH_MEAN_nm, BOT_DISH_STD_nm, int(die.num_pads))
        Cu_gap = Cu_gap.reshape(die_critical_pad_bitmap.shape)
        # Calculate the safe range for Cu recess
        zeta_0 = cfg.k_et * (cfg.T_anl - cfg.T_R) + cfg.k_eb * (cfg.T_anl - cfg.T_R)    # The total expansion of the Cu pad after annealing (nm)
        zeta_1_ = roughness_parameters(
            Asperity_R_m            = cfg.Asperity_R_m,
            Roughness_sigma_m       = cfg.Roughness_sigma_m,
            eta_s                   = cfg.eta_s,
            Roughness_constant      = cfg.Roughness_constant,
            Adhesion_energy         = cfg.Adhesion_energy,
            Young_modulus_Pa        = cfg.Young_modulus_Pa,
            Dielectric_thickness    = cfg.Dielectric_thickness,
            PITCH_r_um              = cfg.PITCH_r_um,
            PITCH_c_um              = cfg.PITCH_c_um,
            PAD_BOT_R_um            = cfg.PAD_BOT_R_um,
            DISH_0_m                = cfg.DISH_0_m,
            k_peel                  = cfg.k_peel,
        )
        zeta_1 = max(zeta_1_, 0)

        # Check critical pad Cu gap
        critical_pad_Cu_gap = Cu_gap * die_critical_pad_bitmap
        if critical_pad_Cu_gap.min() < -zeta_0 or critical_pad_Cu_gap.max() > -zeta_1:
            die.survival = False
            continue

        # Check redundant pad Cu gap
        redundant_pad_Cu_gap = Cu_gap * die_redundant_pad_bitmap
        num_redundant_pad_over_Cu_gap = np.sum(redundant_pad_Cu_gap > -zeta_1) + np.sum(redundant_pad_Cu_gap < -zeta_0)
        redundant_pad_fail_map[redundant_pad_Cu_gap > -zeta_1] = 1
        redundant_pad_fail_map[redundant_pad_Cu_gap < -zeta_0] = 1
        # Get those failing pad indices
        failing_redundant_pad_ind = np.concatenate((np.argwhere(redundant_pad_Cu_gap > -zeta_1), np.argwhere(redundant_pad_Cu_gap < -zeta_0)), axis=0)
        # Get the fail bump indices (specifically for UCIe mapping)
        fail_bump_id_set = set((failing_redundant_pad_ind[:, 0] * PAD_ARR_COL / 2 + failing_redundant_pad_ind[:, 1] // 2).astype(int))
        # Check every net connecting redundant pads, if all the redundant pad replicas fail due to voids, then the die fails
        for net, redundant_bumpid_set in redundant_net_to_bumpids.items():
            if redundant_bumpid_set.issubset(fail_bump_id_set):
                die.survival = False
                break

        if die.survival:
            safe_die_count += 1

        # # Check the time taken for each die
        # end_time = time.time()
        # time_taken = end_time - start_time
        # print("Time taken for die {}: {:.2f} seconds".format(die_ind, time_taken))
    # raise ValueError("Test error")
    die_yield = safe_die_count / die_count
    # print("The yield of dies is {:.2f}%.".format(die_yield * 100))
    yield_list.append(die_yield)

    return yield_list       