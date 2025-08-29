#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Cu gap simulator for the yield model for D2W hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np



def Cu_gap_simulator(
        top_dish_mean, 
        top_dish_std, 
        bot_dish_mean, 
        bot_dish_std, 
        num_pad,
    ):
    top_dish = np.random.normal(top_dish_mean, top_dish_std, num_pad).astype(np.float16)
    bot_dish = np.random.normal(bot_dish_mean, bot_dish_std, num_pad).astype(np.float16)
    Cu_gap = top_dish + bot_dish
    return Cu_gap