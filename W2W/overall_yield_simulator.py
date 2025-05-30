#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import time

from wafer_die_initialization import Die, Wafer, wafer_initialize
from overlay_yield_simulator import overlay_term_simulator, die_pad_misalignment
from Cu_gap_simulator import Cu_gap_simulator



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
    TOP_DISH_MEAN,
    TOP_DISH_STD,
    BOT_DISH_MEAN,
    BOT_DISH_STD,
    PITCH,
    PAD_TOP_R,
    RANDOM_MISALIGNMENT_MEAN,
    RANDOM_MISALIGNMENT_STD,
    approximate_set,
):
    yield_list = []
    for waf_ind in range(len(waf_list)):
        die_count = 0
        wafer = waf_list[waf_ind]
        for die, die_ind in zip(wafer.die_list, range(len(wafer.die_list))):
            die_count += 1
            # if die_count % 100 == 0:
            #     print("Processing die {}/{}...".format(die_count, len(wafer.die_list)))
                
            # # Check the pad misalignment
            # die.pad_misalignment = die_pad_misalignment(die=die, 
            #                                             base_pad_coords=wafer.base_pad_coords,
            #                                             system_translation_x=system_translation_x[waf_ind],
            #                                             system_translation_y=system_translation_y[waf_ind],
            #                                             system_rotation=system_rotation[waf_ind],
            #                                             system_magnification=system_magnification[waf_ind],
            #                                             RANDOM_MISALIGNMENT_MEAN=RANDOM_MISALIGNMENT_MEAN,
            #                                             RANDOM_MISALIGNMENT_STD=RANDOM_MISALIGNMENT_STD,
            #                                             approximate_set=approximate_set,
            #                                             )
            # # die fail criteria: any pad_misalignment >= MAX_ALLOWED_MISALIGNMENT
            # if die.pad_misalignment.max() >= MAX_ALLOWED_MISALIGNMENT:
            #     wafer.survival_die -= 1
            #     die.survival = False
            # if not die.survival:
            #     continue

            # # Check the void overlap with the pad
            # # calculate the pad array box
            # for index, void in enumerate(wafer.voids):
            #     # check if the void overlaps with the pad array box
            #     closest_x = max(die.pad_array_box[2][0], min(void[0], die.pad_array_box[2][0] + PAD_ARR_W))
            #     closest_y = max(die.pad_array_box[2][1], min(void[1], die.pad_array_box[2][1] + PAD_ARR_L))
            #     distance = np.sqrt((closest_x - void[0])**2 + (closest_y - void[1])**2)
            #     if distance <= void[2]:     # The void overlaps with the pad array box
            #         # wafer.survival_die -= 1
            #         # die.survival = False
            #         # wafer.safe_voids_mask[index] = 0
            #         # break
            #         if void[2] >= PITCH * np.sqrt(2) / 2:    # Generally it will be larger than the pitch
            #             wafer.survival_die -= 1
            #             die.survival = False
            #             wafer.safe_voids_mask[index] = 0
            #             break
            #         else:   # check if the void is in the top pad
            #             die_pad_coords = wafer.base_pad_coords + die.die_center
            #             if any(np.sqrt((die_pad_coords[:, 0] - void[0])**2 + (die_pad_coords[:, 1] - void[1])**2) <= void[2] + PAD_TOP_R):
            #                 wafer.survival_die -= 1
            #                 die.survival = False
            #                 wafer.safe_voids_mask[index] = 0
            #                 break
            #     else:
            #         continue
            # if not die.survival:
            #     continue

            # Assuming wafer.voids is an array of shape (N, 3), where N is the number of voids
            voids = np.array(wafer.voids)
            if voids.size > 0:
                # Coordinates and dimensions of the die pad array box
                pad_array_box_x = die.pad_array_box[2][0]
                pad_array_box_y = die.pad_array_box[2][1]

                # Calculate closest x and y distances for all voids simultaneously
                closest_x = np.maximum(pad_array_box_x, np.minimum(voids[:, 0], pad_array_box_x + PAD_ARR_W))
                closest_y = np.maximum(pad_array_box_y, np.minimum(voids[:, 1], pad_array_box_y + PAD_ARR_L))

                # Calculate distance from each void to the closest point on the pad array box
                distances = np.sqrt((closest_x - voids[:, 0]) ** 2 + (closest_y - voids[:, 1]) ** 2)

                # Create a mask for voids overlapping with the pad array box
                overlapping_mask = distances < voids[:, 2]

                # Handle voids with radius larger than PITCH * np.sqrt(2) / 2
                large_voids_mask = voids[:, 2] >= PITCH * np.sqrt(2) / 2

                # Mark voids that overlap and have radius larger than threshold
                wafer.safe_voids_mask[overlapping_mask & large_voids_mask] = 0
                die.survival = not np.any(overlapping_mask & large_voids_mask)
                wafer.survival_die -= 1 if not die.survival else 0

                if die.survival:
                    # Check the remaining voids (if not large_voids)
                    remaining_voids_mask = overlapping_mask & ~large_voids_mask

                    # Vectorized check for void overlap with the top pad
                    if np.any(remaining_voids_mask):
                        die_pad_coords = wafer.base_pad_coords + die.die_center
                        distances_to_pads = np.sqrt((die_pad_coords[:, 0][:, np.newaxis] - voids[remaining_voids_mask, 0]) ** 2 +
                                                    (die_pad_coords[:, 1][:, np.newaxis] - voids[remaining_voids_mask, 1]) ** 2)
                        
                        # Create a mask for voids overlapping with the top pad
                        top_pad_mask = np.any(distances_to_pads < (voids[remaining_voids_mask, 2] + PAD_TOP_R), axis=0)
                        
                        # Update survival and void mask based on overlap with top pad
                        wafer.safe_voids_mask[remaining_voids_mask] = np.where(top_pad_mask, 0, wafer.safe_voids_mask[remaining_voids_mask])
                        die.survival = not np.any(top_pad_mask)
                        wafer.survival_die -= 1 if not die.survival else 0

            # Proceed if die still survives
            if not die.survival:
                continue
            
            # # check time
            # start_time = time.time()
            # Check the Cu expansion
            # Cu_gap = Cu_gap_simulator(TOP_DISH_MEAN, TOP_DISH_STD, BOT_DISH_MEAN, BOT_DISH_STD, int(die.num_pad))
            # if Cu_gap.min() < -zeta_0 or Cu_gap.max() > -zeta_1:
            #     wafer.survival_die -= 1
            #     die.survival = False

            # if not die.survival:
            #     continue
            # #check time
            # print("The time for checking Cu gap is: ", time.time() - start_time)

            
            
        # # print("The number of survival dies in the wafer is {}.".format(wafer.survival_die))
        # Draw the whole wafer
        wafer.draw_wafer_die(fig_size=(15, 15))
        raise ValueError("Stop here")
        # wafer.draw_wafer_die(fig_size=(25, 25))

        die_yield = wafer.survival_die / len(wafer.die_list)
        # print("The die yield of the wafer is {:.2f}%.".format(die_yield * 100))
        yield_list.append(die_yield)
        if waf_ind % 100 == 0:
            print("Processing wafer {}/{}..., Current mean yield is {:.2f}%.".format(waf_ind, len(waf_list), np.mean(yield_list) * 100))
        # raise ValueError("Stop here")
    return yield_list       