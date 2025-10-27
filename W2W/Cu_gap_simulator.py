#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np

# from multiprocessing import Pool

# # Function to generate samples for a subset of pads
# def generate_samples(num_pads, mean, std):
#     return np.random.normal(mean, std, num_pads)

# def Cu_gap_simulator(TOP_DISH_MEAN_nm, 
#         TOP_DISH_STD_nm, 
#         BOT_DISH_MEAN_nm, 
#         BOT_DISH_STD_nm, 
#         num_pads, num_processes=4):
#     mean = TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm
#     std = np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)
#     pool = Pool(processes=num_processes)
    
#     # Split num_padss into chunks for parallel processing
#     chunk_size = num_pads // num_processes
#     results = pool.starmap(generate_samples, [(chunk_size, mean, std)] * num_processes)
    
#     # Close the pool and wait for the processes to finish
#     pool.close()
#     pool.join()
    
#     return np.concatenate(results)

    
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