#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np


    
def Cu_gap_simulator(
        TOP_DISH_MEAN_nm, 
        TOP_DISH_STD_nm, 
        BOT_DISH_MEAN_nm, 
        BOT_DISH_STD_nm, 
        num_pads,
    ) -> tuple[np.ndarray, np.ndarray]:
    top_dish = np.random.normal(TOP_DISH_MEAN_nm, TOP_DISH_STD_nm, num_pads)
    bot_dish = np.random.normal(BOT_DISH_MEAN_nm, BOT_DISH_STD_nm, num_pads)
    return top_dish, bot_dish