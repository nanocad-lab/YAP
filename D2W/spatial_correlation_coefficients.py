#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from overlay_yield_simulator import die_pad_misalignment
from Cu_gap_simulator import Cu_gap_simulator
from debond import debond_dishing_bounds_calculator
from esd_hybrid import esd_failure_simulator
from sklearn.neighbors import KDTree

def get_spatial_correlation_coefficients(
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
    TILT_X_MEAN_DEG: float,
    TILT_X_STD_DEG: float,
    TILT_Y_MEAN_DEG: float,
    TILT_Y_STD_DEG: float,
    PITCH_c_um: float,
    PITCH_r_um: float,
    PAD_TOP_R_um: float,
    RANDOM_MISALIGNMENT_MEAN_um: float,
    RANDOM_MISALIGNMENT_STD_um: float,
    approximate_set: int,
    pad_bitmap_collection: dict,
):
    yield_list = []
    die_count = 0
    safe_die_count = 0
    # Get the valid pad mask
    valid_pad_mask = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)
    valid_die_pad_coords = die_list[0].pad_coords[valid_pad_mask.flatten() == 1]
    start_time = time.time()
    valid_pad_dishing_bound_array = debond_dishing_bounds_calculator(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
    print("Dishing bound calculation time: {:.2f} seconds".format(time.time() - start_time))

    # Doing correlation experiments
    pad_xy = valid_die_pad_coords[:, :2]  # (num_valid_pads, 2)
    # Parameters for spatial correlation calculation
    dist_interval_um = 5000.0  # um
    dist_part = int(max(cfg.DIE_W_um, cfg.DIE_L_um) // dist_interval_um + 1)
    bin_width = 40.0 # um
    # All failure mechanisms that are considered for spatial correlation
    failure_mechanisms = ['overlay', 'particle', 'stress', 'esd']
    # A dictionary to store overall phi for each mechanism
    overall_phi_dict = {mech: np.array([]) for mech in failure_mechanisms}
    global_dist_list = np.array([])


    # In order to save memory, we process a distance bin at a time
    for part_ind in range(dist_part):
        print(f"Starting correlation calculation round {part_ind+1}/{dist_part}...")

        if cfg.PAD_ARRANGE_PATTERN == "checkerboard":
            Rmin = max(np.sqrt(PITCH_c_um**2 + PITCH_r_um**2), part_ind * dist_interval_um)  # um
        Rmax = min((part_ind + 1) * dist_interval_um, np.sqrt((PAD_ARR_W_um**2 + PAD_ARR_L_um**2)))  # um
        edges = np.arange(Rmin, Rmax + 1e-6, bin_width)
        bin_center = 0.5 * (edges[:-1] + edges[1:])
        num_bins = len(edges) - 1
        overall_counts_dict = {}
        for mech in failure_mechanisms:
            overall_counts_dict[mech] = np.zeros((num_bins, 4), dtype=np.float64)  # columns: sum_x, sum_y, sum_xx, count

        # KDTree for fast neighbor search
        tree = KDTree(pad_xy)
        neighbor_list, dist_list = tree.query_radius(pad_xy, r=Rmax, return_distance=True)

        pair_i, pair_j = [], []
        for i, (neigh, dist) in enumerate(zip(neighbor_list, dist_list)):
            dist_mask = (dist >= Rmin) & (dist <= Rmax)
            for j in neigh[dist_mask]:
                if j > i:
                    pair_i.append(i)
                    pair_j.append(j)
        pair_i = np.array(pair_i, dtype=np.int32)
        pair_j = np.array(pair_j, dtype=np.int32)
        print(f"Total {len(pair_i)} pad pairs found within distance range [{Rmin:.2f}, {Rmax:.2f}] um.")

        dist = np.linalg.norm(pad_xy[pair_i] - pad_xy[pair_j], axis=1)  # (num_pairs, )
        # print("dist", dist)
        global_dist_list = np.concatenate((global_dist_list, bin_center))
        # print("global_dist_list", global_dist_list)
        bin_id = np.digitize(dist, edges, right=False) - 1
        bin_id = np.clip(bin_id, 0, num_bins - 1)

        for die_ind in range(NUM_DIES):
            # # check start time
            start_time = time.time()
            if die_ind % 10 == 0:
                print("Processing die {}/{}...".format(die_ind, len(die_list)))
            die = die_list[die_ind]
            # Get the valid pad mask
            valid_pad_mask = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)
            overall_pad_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))
            overlay_pad_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))
            particle_pad_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))
            stress_pad_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))
            esd_pad_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))

            # """
            # Check the overlay errors
            # """
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
                                                        )
            # die fail criteria: any pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um
            die.pad_misalignment = die.pad_misalignment.reshape(cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL) 
            overall_pad_fail_map[die.pad_misalignment > MAX_ALLOWED_MISALIGNMENT_um] = 1
            overlay_pad_fail_map[die.pad_misalignment > MAX_ALLOWED_MISALIGNMENT_um] = 1

            # Delete the die.pad_misalignment to save memory
            del die.pad_misalignment


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

                        overall_pad_fail_map[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1][overlap_void_pad_mask] = 1
                        particle_pad_fail_map[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1][overlap_void_pad_mask] = 1


            
            '''
            Check the Cu gap, a true Monte Carlo simulator
            '''
            # Check the Cu expansion
            top_dish, bot_dish = Cu_gap_simulator(TOP_DISH_MEAN_nm, TOP_DISH_STD_nm, BOT_DISH_MEAN_nm, BOT_DISH_STD_nm, int(die.num_pads))
            Cu_gap_in_valid_pads = top_dish + bot_dish
            Cu_gap_map = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)
            Cu_gap_map[valid_pad_mask == 1] = Cu_gap_in_valid_pads
            

            # Calculate the safe range for single pad Cu recess
            zeta_0 = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)
            zeta_1 = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)

            zeta_0[valid_pad_mask == 1] = - valid_pad_dishing_bound_array[:, 1] * 2 # lower limits of the sum of top and bottom Cu heights
            zeta_1[valid_pad_mask == 1] = - valid_pad_dishing_bound_array[:, 0] * 2 # upper limits of the sum of top and bottom Cu heights

            overall_pad_fail_map[Cu_gap_map > zeta_1] = 1
            overall_pad_fail_map[Cu_gap_map < zeta_0] = 1
            stress_pad_fail_map[Cu_gap_map > zeta_1] = 1
            stress_pad_fail_map[Cu_gap_map < zeta_0] = 1

            '''
            Check the ESD failure
            '''
            # TODO: ESD failure simulation to be implemented
            first_contact_pad_idx, survive_bool = esd_failure_simulator(pad_coords_um=valid_die_pad_coords,
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
            if first_contact_pad_idx is not None and survive_bool == False:    # One pad will form the first contact and fail
                r_idx, c_idx = first_contact_pad_idx // PAD_ARR_COL, first_contact_pad_idx % PAD_ARR_COL
                overall_pad_fail_map[r_idx, c_idx] = 1
                esd_pad_fail_map[r_idx, c_idx] = 1
            
            if die.survival:
                safe_die_count += 1


            overall_pad_fail_map[~valid_pad_mask.astype(bool)] = np.nan  # Mark invalid pads as nan
            overlay_pad_fail_map[~valid_pad_mask.astype(bool)] = np.nan
            particle_pad_fail_map[~valid_pad_mask.astype(bool)] = np.nan
            stress_pad_fail_map[~valid_pad_mask.astype(bool)] = np.nan
            esd_pad_fail_map[~valid_pad_mask.astype(bool)] = np.nan
            
            
            for mech in failure_mechanisms:
                # The fail bool under the current mechanism
                fail_bool = np.asarray(eval(f"{mech}_pad_fail_map")[valid_pad_mask.astype(bool)] > 0, dtype=bool)
                counts = overall_counts_dict[mech]
                # Calculate overlay correlation counts
                a = fail_bool[pair_i]
                b = fail_bool[pair_j]
                n11 = (a & b).astype(np.int64)
                n10 = (a & ~b).astype(np.int64)
                n01 = (~a & b).astype(np.int64)
                n00 = (~a & ~b).astype(np.int64)

                counts[:, 0] += np.bincount(bin_id, weights=n11, minlength=num_bins)
                counts[:, 1] += np.bincount(bin_id, weights=n10, minlength=num_bins)
                counts[:, 2] += np.bincount(bin_id, weights=n01, minlength=num_bins)
                counts[:, 3] += np.bincount(bin_id, weights=n00, minlength=num_bins)

                overall_counts_dict[mech] = counts

            # End of die loop
        
        for mech in failure_mechanisms:
            counts = overall_counts_dict[mech]
            n11, n10, n01, n00 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
            n1dot = n11 + n10
            n0dot = n01 + n00
            ndot1 = n11 + n01
            ndot0 = n10 + n00
            phi = (n11 * n00 - n10 * n01) / np.sqrt(n1dot * n0dot * ndot1 * ndot0 + 1e-10)
            overall_phi_dict[mech] = np.append(overall_phi_dict[mech], phi)

    # End of distance part loop


    for mech in failure_mechanisms:
        with open(cfg.OUTPUT_DIR + cfg.DESIGN + "/{}_pad_fail_correlation_stats.txt".format(mech), 'w') as f:
            f.write("distance phi\n")
            for i in range(len(global_dist_list)):
                f.write(f"{global_dist_list[i]:.2f} {overall_phi_dict[mech][i]:.6f}\n")


        print("{} correlation statistics saved in ".format(mech) + cfg.OUTPUT_DIR + cfg.DESIGN + "/{}_pad_fail_correlation_stats.txt".format(mech))

        if cfg.DEBUG:
            plt.figure(figsize=(10,6))
            plt.plot(global_dist_list, overall_phi_dict[mech], marker='o', linestyle='-')
            plt.xlabel("Pad-to-Pad Distance (µm)")
            plt.ylabel("{} Pad Failure Correlation Coefficient (ϕ)".format(mech.capitalize()))
            plt.title("{} Pad Failure Correlation vs Distance".format(mech.capitalize()))
            plt.grid(True)
            plt.xticks(np.arange(Rmin, Rmax + 1, dist_interval_um))
            # plt.savefig(cfg.OUTPUT_DIR + cfg.DESIGN + "/{}_phi_vs_distance.png".format(mech), dpi=300)
            plt.show()

    