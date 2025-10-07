#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Cu gap simulator for the yield model for D2W hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np



def Cu_gap_simulator(
        TOP_DISH_MEAN_nm, 
        TOP_DISH_STD_nm, 
        BOT_DISH_MEAN_nm, 
        BOT_DISH_STD_nm, 
        num_pads,
    ):
    top_dish = np.random.normal(TOP_DISH_MEAN_nm, TOP_DISH_STD_nm, num_pads).astype(np.float16)
    bot_dish = np.random.normal(BOT_DISH_MEAN_nm, BOT_DISH_STD_nm, num_pads).astype(np.float16)
    Cu_gap = top_dish + bot_dish
    return Cu_gap