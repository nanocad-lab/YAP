#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np

# from multiprocessing import Pool

# # Function to generate samples for a subset of pads
# def generate_samples(num_pad, mean, std):
#     return np.random.normal(mean, std, num_pad)

# def Cu_gap_simulator(top_dish_mean, 
#         top_dish_std, 
#         bot_dish_mean, 
#         bot_dish_std, 
#         num_pad, num_processes=4):
#     mean = top_dish_mean + bot_dish_mean
#     std = np.sqrt(top_dish_std**2 + bot_dish_std**2)
#     pool = Pool(processes=num_processes)
    
#     # Split num_pads into chunks for parallel processing
#     chunk_size = num_pad // num_processes
#     results = pool.starmap(generate_samples, [(chunk_size, mean, std)] * num_processes)
    
#     # Close the pool and wait for the processes to finish
#     pool.close()
#     pool.join()
    
#     return np.concatenate(results)


def Cu_gap_simulator(
        top_dish_mean, 
        top_dish_std, 
        bot_dish_mean, 
        bot_dish_std, 
        num_pad,
    ):
    top_dish = np.random.normal(top_dish_mean, top_dish_std, num_pad)
    bot_dish = np.random.normal(bot_dish_mean, bot_dish_std, num_pad)
    Cu_gap = top_dish + bot_dish
    return Cu_gap