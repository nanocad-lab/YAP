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



def overall_yield_simulator(
    die_list,
    NUM_DIES,
    base_pad_coords,
    system_translation_x,
    system_translation_y,
    system_rotation,
    system_magnification,
    MAX_ALLOWED_MISALIGNMENT,
    zeta_0,
    zeta_1,
    PAD_ARR_W,
    PAD_ARR_L,
    DIE_W,
    DIE_L,
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
    die_count = 0
    safe_die_count = 0
    for die_ind in range(NUM_DIES):
        die = die_list[die_ind]
        die_count += 1
        # if die_count % 10 == 0:
        #     print("Processing die {}/{}...".format(die_count, len(die_list)))

        # # Check the pad misalignment
        # die.pad_misalignment = die_pad_misalignment(die=die, 
        #                                             base_pad_coords=base_pad_coords,
        #                                             system_translation_x=system_translation_x[die_ind],
        #                                             system_translation_y=system_translation_y[die_ind],
        #                                             system_rotation=system_rotation[die_ind],
        #                                             system_magnification=system_magnification[die_ind],
        #                                             RANDOM_MISALIGNMENT_MEAN=RANDOM_MISALIGNMENT_MEAN,
        #                                             RANDOM_MISALIGNMENT_STD=RANDOM_MISALIGNMENT_STD,
        #                                             approximate_set=approximate_set,
        #                                             )
        # # die fail criteria: any pad_misalignment >= MAX_ALLOWED_MISALIGNMENT
        # if die.pad_misalignment.max() >= MAX_ALLOWED_MISALIGNMENT:
        #     die.survival = False
        # if not die.survival:
        #     continue

        # Check the void overlap with the pad
        # calculate the pad array box
        for index, void in enumerate(die.voids):
            # check if the void overlaps with the pad array box
            closest_x = max(die.pad_array_box[2][0], min(void[0], die.pad_array_box[2][0] + PAD_ARR_W))
            closest_y = max(die.pad_array_box[2][1], min(void[1], die.pad_array_box[2][1] + PAD_ARR_L))
            distance = np.sqrt((closest_x - void[0])**2 + (closest_y - void[1])**2)
            if distance <= void[2]:     # The void overlaps with the pad array box
                # if void[2] >= PITCH * np.sqrt(2) / 2:    # Generally it will be larger than the pitch
                #     die.survival = False
                #     die.safe_voids_mask[index] = 0
                #     # fig, ax = plt.subplots(figsize=(4, 4))
                #     # die.draw_die(ax)
                #     break
                # else:   # check if the void is in the top pad
                #     if any(np.sqrt((die.pad_coords[:, 0] - void[0])**2 + (die.pad_coords[:, 1] - void[1])**2) <= void[2] + PAD_TOP_R):
                #         die.survival = False
                #         die.safe_voids_mask[index] = 0
                #         break
                die.survival = False
                die.safe_voids_mask[index] = 0
                break
            else:
                continue
        if not die.survival:
            continue
        
        # # Check the Cu expansion
        # Cu_gap = Cu_gap_simulator(TOP_DISH_MEAN, TOP_DISH_STD, BOT_DISH_MEAN, BOT_DISH_STD, die.num_pad)
        # if Cu_gap.min() < -zeta_0 or Cu_gap.max() > -zeta_1:
        #     die.survival = False
        # if not die.survival:
        #     continue

        # # Check the roughness voids
        # roughness_voids_total_area = np.pi * (die.roughness_voids[:, 2] ** 2).sum() * D0_rough_scale_factor
        # particle_voids_total_area = np.pi * (die.voids[:, 2] ** 2 * die.safe_voids_mask).sum()
        # if roughness_voids_total_area + particle_voids_total_area >= (1 - die_contact_area_coefficient) * np.pi * DIE_W * DIE_L:
        #     die.survival = False

        if die.survival:
            safe_die_count += 1
    # raise ValueError("Test error")
    die_yield = safe_die_count / die_count
    # print("The yield of dies is {:.2f}%.".format(die_yield * 100))
    yield_list.append(die_yield)

    return yield_list       