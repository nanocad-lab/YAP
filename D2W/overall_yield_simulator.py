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
from debond import debond_dishing_bounds_calculator, post_bond_warpage_calculator
from esd_hybrid import esd_failure_simulator

def overall_yield_simulator(
    cfg,
    die_list: list,
    NUM_DIE_SAMPLES: int,
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
    if not os.path.exists(cfg.OUTPUT_DIR + cfg.INTERFACE + '/temp/' + cfg.INTERFACE + "_dishing_bound_array.npy") or cfg.DEBUG:
        if not os.path.exists(cfg.OUTPUT_DIR + cfg.INTERFACE + '/temp/'):
            os.makedirs(cfg.OUTPUT_DIR + cfg.INTERFACE + '/temp/')
        # start_time = time.time()
        valid_pad_dishing_bound_array = debond_dishing_bounds_calculator(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
        # print("Dishing bound calculation time: {:.2f} seconds".format(time.time() - start_time))
        np.save(cfg.OUTPUT_DIR + cfg.INTERFACE + '/temp/' + cfg.INTERFACE + "_dishing_bound_array.npy", valid_pad_dishing_bound_array)
    else:
        valid_pad_dishing_bound_array = np.load(cfg.OUTPUT_DIR + cfg.INTERFACE + '/temp/' + cfg.INTERFACE + "_dishing_bound_array.npy")

    epoch_fail_map_dict = {}    # This dict stores the fail bump maps for all die samples in this epoch for each mechanism
    epoch_fail_vec_dict = {}    # This dict stores failure reason (each mechanism) for all die samples in this epoch
    
    failure_mechanism_list = ['overlay', 'particle', 'mechanical', 'ESD', 'overall']
    if cfg.verbose:    
        for mechanism in failure_mechanism_list:
            epoch_fail_map_dict[mechanism] = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))
            epoch_fail_vec_dict[mechanism] = np.zeros(NUM_DIE_SAMPLES) 


    for die_ind in range(NUM_DIE_SAMPLES):
        # # check start time
        start_time = time.time()
        # if die_count % 500 == 0:
        #     print("Processing die {}/{}...".format(die_count, len(die_list)))
        die = die_list[die_ind]
        die_count += 1
        # Read the critical pad bitmap
        die_critical_pad_bitmap = pad_bitmap_collection["CRITICAL_PAD_BITMAP"]
        # Read the redundant critical pad bitmap
        die_redundant_pad_bitmap = pad_bitmap_collection["REDUNDANT_PAD_BITMAP"]
        # Read the ESD critical pad bitmap
        die_esd_critical_pad_bitmap = pad_bitmap_collection["ESD_CRITICAL_PAD_BITMAP"]
        # Read the redundant net to bump ids mapping
        redundant_net_to_bumpids = pad_bitmap_collection["redundant_net_to_bumpids"]
        # Get the valid pad mask
        valid_pad_mask = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)
        # Read the mapping from physical to bump id
        mapping_physical_to_bumpid = pad_bitmap_collection["mapping_physical_to_bumpid"]
        # Read the criticality info
        criticality_info = pad_bitmap_collection["criticality_info"]
        # Read the redundant net to 1D physical mask mapping
        redundant_net_to_1d_physical_mask = pad_bitmap_collection["redundant_net_to_1d_physical_mask"]
        redundant_pad_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL))
        temp_overall_fail_map = np.zeros((PAD_ARR_ROW, PAD_ARR_COL), dtype=int)   # This temporary map is used to record the overall fail map for this die, which will be added to epoch_fail_map_dict['overall'] at the end of this die's checking. This is to avoid double counting when one pad fails due to multiple reasons.
        
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
                                                    )
        if approximate_set == 1:
            # die fail criteria: any pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um
            die.pad_misalignment = die.pad_misalignment.reshape(cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL)
            if cfg.verbose:
                epoch_fail_map_dict['overlay'] += (die.pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um).astype(int) 
                temp_overall_fail_map |= (die.pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um).astype(int)

            critical_pad_misalignment = die.pad_misalignment * die_critical_pad_bitmap
            # Check if any critical pad misalignment is greater than the maximum allowed misalignment
            if np.any(critical_pad_misalignment >= MAX_ALLOWED_MISALIGNMENT_um):
                die.survival = False
                if cfg.verbose:
                    epoch_fail_vec_dict['overlay'][die_ind] = 1
                    epoch_fail_vec_dict['overall'][die_ind] = 1
                if not cfg.verbose:
                    continue
            # Check if too many redundant pad misalignment is greater than the maximum allowed misalignment
            redundant_pad_misalignment = die.pad_misalignment * die_redundant_pad_bitmap    # shape (PAD_ARR_ROW, PAD_ARR_COL)
            redundant_pad_fail_map[redundant_pad_misalignment > MAX_ALLOWED_MISALIGNMENT_um] = 1   # 1: redundant pad fails
            for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
                tolerated_mechanical_failures = criticality_info[redundant_net]['tolerated_mechanical_failures']
                num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
                if num_fail_pad_in_net > tolerated_mechanical_failures:
                    die.survival = False
                    if cfg.verbose:
                        epoch_fail_vec_dict['overlay'][die_ind] = 1
                        epoch_fail_vec_dict['overall'][die_ind] = 1
                    break

            # # Get the fail bump indices
            # fail_bump_id = mapping_physical_to_bumpid[redundant_pad_fail_map == 1]
            # # Switch to set for easier checking
            # fail_bump_id_set = set(fail_bump_id.astype(int))   

        # Delete the die.pad_misalignment to save memory
        del die.pad_misalignment

        # # If all the redundant pad replicas fail, then the die fails
        # if any(redundant_bumpid_set.issubset(fail_bump_id_set) for net, redundant_bumpid_set in redundant_net_to_bumpids.items()):
        #         die.survival = False
        #         break
        if not die.survival and not cfg.verbose:
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
                    # Get the redundant critical pad bitmap for the pads we need to consider
                    check_redundant_pad_bitmap = die_redundant_pad_bitmap[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                    # Record the fail pads due to voids
                    if cfg.verbose:
                        sub_fail_map_particle = epoch_fail_map_dict['particle'][PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                        sub_fail_map_particle[overlap_void_pad_mask] += 1
                        sub_fail_map_overall = temp_overall_fail_map[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1]
                        sub_fail_map_overall[overlap_void_pad_mask] = 1

                    # Check if any void overlaps with the critical pads
                    overlap_critical = overlap_void_pad_mask & check_critical_pad_bitmap.astype(bool)
                    if np.any(overlap_critical):
                        die.survival = False
                        if cfg.verbose:
                            epoch_fail_vec_dict['particle'][die_ind] = 1
                            epoch_fail_vec_dict['overall'][die_ind] = 1
                        if not cfg.verbose:
                            break
                    else:
                        # Check if any void overlaps with the redundant critical pads
                        overlap_redundant = overlap_void_pad_mask & check_redundant_pad_bitmap.astype(bool)
                        redundant_pad_fail_map[PAD_ARR_ROW-j_max-1:PAD_ARR_ROW-j_min, i_min:i_max+1][overlap_redundant] = 1
                        for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
                            tolerated_mechanical_failures = criticality_info[redundant_net]['tolerated_mechanical_failures']
                            num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
                            if num_fail_pad_in_net > tolerated_mechanical_failures:
                                die.survival = False
                                if cfg.verbose:
                                    epoch_fail_vec_dict['particle'][die_ind] = 1
                                    epoch_fail_vec_dict['overall'][die_ind] = 1
                                if not cfg.verbose:
                                    break
                        # # Get the fail bump indices
                        # fail_bump_id = mapping_physical_to_bumpid[redundant_pad_fail_map == 1]
                        # # Switch to set for easier checking
                        # fail_bump_id_set = set(fail_bump_id.astype(int))
                
                        # # Check every net connecting redundant pads, if all the redundant pad replicas fail due to voids, then the die fails
                        # if any(redundant_bumpid_set.issubset(fail_bump_id_set) for net, redundant_bumpid_set in redundant_net_to_bumpids.items()):
                        #     die.survival = False
                        #     break
                    if not die.survival and not cfg.verbose:
                        break

        # Proceed if die still survives
        if not die.survival and not cfg.verbose:
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
        zeta_0 = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)    # lower limits to prevent Cu connection open
        zeta_1 = np.full((PAD_ARR_ROW, PAD_ARR_COL), np.nan)    # upper limits to prevent dielectric delamination

        zeta_0[valid_pad_mask == 1] = - valid_pad_dishing_bound_array[:, 1] * 2 # lower limits of the sum of top and bottom Cu heights
        zeta_1[valid_pad_mask == 1] = - valid_pad_dishing_bound_array[:, 0] * 2 # upper limits of the sum of top and bottom Cu heights

        # Commented on 03/18/2026
        if cfg.verbose:
            epoch_fail_map_dict['mechanical'] += ((Cu_gap_map > zeta_1) | (Cu_gap_map < zeta_0)).astype(int)
            temp_overall_fail_map |= ((Cu_gap_map > zeta_1) | (Cu_gap_map < zeta_0)).astype(int)

        # Check critical pad Cu gap
        critical_pad_Cu_gap = Cu_gap_map * die_critical_pad_bitmap  # shape: (PAD_ARR_ROW, PAD_ARR_COL)
        # if np.any(critical_pad_Cu_gap > zeta_1 * die_critical_pad_bitmap) or np.any(critical_pad_Cu_gap < zeta_0 * die_critical_pad_bitmap):
        if np.any(critical_pad_Cu_gap < zeta_0 * die_critical_pad_bitmap):
            die.survival = False
            if cfg.verbose:
                epoch_fail_vec_dict['mechanical'][die_ind] = 1
                epoch_fail_vec_dict['overall'][die_ind] = 1
            if not cfg.verbose:
                continue

        # Check redundant pad Cu gap
        redundant_pad_Cu_gap = Cu_gap_map * die_redundant_pad_bitmap
        # redundant_pad_fail_map[redundant_pad_Cu_gap > zeta_1 * die_redundant_pad_bitmap] = 1
        redundant_pad_fail_map[redundant_pad_Cu_gap < zeta_0 * die_redundant_pad_bitmap] = 1
        for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
            tolerated_mechanical_failures = criticality_info[redundant_net]['tolerated_mechanical_failures']
            num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
            if num_fail_pad_in_net > tolerated_mechanical_failures:
                die.survival = False
                if cfg.verbose:
                    epoch_fail_vec_dict['mechanical'][die_ind] = 1
                    epoch_fail_vec_dict['overall'][die_ind] = 1
                if not cfg.verbose:
                    break

        # Check whether there are too many pads with Cu gap out of the safe range, which will cause die failure
        num_cu_pad_fail_limit = cfg.CU_RECESS_PAD_FAIL_RATIO * np.sum(die_critical_pad_bitmap)
        post_bond_warpage = post_bond_warpage_calculator(cfg)
        if (np.sum(Cu_gap_map[valid_pad_mask == 1] > 0) > num_cu_pad_fail_limit) or (post_bond_warpage > cfg.WARPAGE_LIMIT_UM):
            die.survival = False
            if cfg.verbose:
                epoch_fail_vec_dict['mechanical'][die_ind] = 1
                epoch_fail_vec_dict['overall'][die_ind] = 1
            if not cfg.verbose:
                continue
            
        # # Get the fail bump indices
        # fail_bump_id = mapping_physical_to_bumpid[redundant_pad_fail_map == 1]
        # # Switch to set for easier checking
        # fail_bump_id_set = set(fail_bump_id.astype(int))
        # # Check every net connecting redundant pads, if all the redundant pad replicas fail due to voids, then the die fails
        # if any(redundant_bumpid_set.issubset(fail_bump_id_set) for net, redundant_bumpid_set in redundant_net_to_bumpids.items()):
        #     # print(f"Die {die_ind} fails due to redundant pad Cu gap.")  
        #     die.survival = False
        #     break

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
                                                dummy_pad_bitmap=pad_bitmap_collection['DUMMY_PAD_BITMAP'].flatten()[valid_pad_mask.flatten() == 1],
                                                )
        if first_contact_pad_idx is not None and survive_bool == False:    # One pad will form the first contact and fail
            r_idx, c_idx = first_contact_pad_idx // PAD_ARR_COL, first_contact_pad_idx % PAD_ARR_COL
            if cfg.verbose:
                epoch_fail_map_dict['ESD'][r_idx, c_idx] += 1
                temp_overall_fail_map[r_idx, c_idx] |= 1
            if die_esd_critical_pad_bitmap[r_idx, c_idx] == 1:  # If the failing pad is critical w.r.t. ESD
                print(f"Die {die_ind} fails due to ESD on critical pad.")
                die.survival = False
                if cfg.verbose:
                    epoch_fail_vec_dict['ESD'][die_ind] = 1
                    epoch_fail_vec_dict['overall'][die_ind] = 1
                continue
            for redundant_net, physical_mask in redundant_net_to_1d_physical_mask.items():
                tolerated_esd_failures = criticality_info[redundant_net]['tolerated_esd_failures']
                num_fail_pad_in_net = np.sum(redundant_pad_fail_map.flatten()[physical_mask])
                if num_fail_pad_in_net > tolerated_esd_failures:
                    die.survival = False
                    if cfg.verbose:
                        epoch_fail_vec_dict['ESD'][die_ind] = 1
                        epoch_fail_vec_dict['overall'][die_ind] = 1
                    break
        
        if die.survival:
            safe_die_count += 1

        if cfg.verbose == 0:
            epoch_fail_map_dict['overall'] += temp_overall_fail_map.astype(int)

        # # Check the time taken for each die
        # end_time = time.time()
        # time_taken = end_time - start_time
        # print("Time taken for die {}: {:.2f} seconds".format(die_ind, time_taken))
    # raise ValueError("Test error")
    die_yield = safe_die_count / die_count
    # print("The yield of dies is {:.2f}%.".format(die_yield * 100))
    yield_list.append(die_yield)

    return yield_list, epoch_fail_map_dict, epoch_fail_vec_dict