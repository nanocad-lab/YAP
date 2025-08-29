#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from assembly_yield_simulator import Assembly_Yield_Simulator
from utils.util import *
from pad_bitmap_generation import pad_bitmap_generate_random

# Load configuration
cfg = load_modeling_config(path='configs/config.yaml', 
                     mode='w2w_simulation',
                     debug=False)

# Generate pad bitmap collection
pad_bitmap_collection = pad_bitmap_generate_random(cfg=cfg)  

# Explore the impact of the pitch
particle_density_list = np.logspace(-10, -8.4, 100)

# Yield containers
assembly_yield_list = []
single_config_yield_list_array = np.zeros([len(particle_density_list), cfg.simulation_times * cfg.NUM_WAFERS])


for i, particle_density in enumerate(particle_density_list):
    cfg.D0 = float(particle_density_list[i])  # particle density (um^{-2})
    print("Processing particle density {}/{}, particle dednsity: {}".format(i + 1, len(particle_density_list), particle_density))
    assembly_yield, single_config_yield_list = Assembly_Yield_Simulator(
        cfg=cfg,
        pad_bitmap_collection=pad_bitmap_collection,                                    
    )
                                                                        
    assembly_yield_list.append(assembly_yield)
    single_config_yield_list_array[i] = np.array(single_config_yield_list).flatten()
    if i % 50 == 0:
        print("The running mean assembly yield is {:.2f}%.".format(np.mean(assembly_yield_list) * 100))

print("The assembly yield list is: ", assembly_yield_list)
# print("The assembly yield list is: ", assembly_yield_list)
# # Save the data
np.save("pd_-10_-8d8_20.npy", particle_density_list)
np.save("assembly_yield_list_pd_-10_-8d8_20_size_1e4.npy", assembly_yield_list)
np.save("single_config_yield_list_array_pd_-10_-8d8_20_size_1e4.npy", single_config_yield_list_array)