#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from assembly_yield_simulator import Assembly_Yield_Simulator
import sys


# WAFER PARAMETERS
NUM_WAFERS = 1000
# PER WAFER PARAMETERS
PITCH = 5  # pitch (um)
DIE_W = 1e+4  # die width (um)
DIE_L = 1e+4  # die length (um)
WAF_R = 150e+3  # wafer radius (um)
PAD_TOP_R = 1.0  # top Cu pad radius (um)
PAD_BOT_R = 1.5  # bottom Cu pad radius (um)
PAD_ARR_ROW = int(np.floor(float(DIE_L / PITCH)))  # number of pads in a row of pad array
PAD_ARR_COL = int(np.floor(float(DIE_W / PITCH)))  # number of pads in a column of pad array
VOID_SHAPE = "circle"  # void shape: 'circle' or 'square'
dice_width = 10000  # dice width (um)

# Overlay Model Parameters
RANDOM_MISALIGNMENT_MEAN = 0
RANDOM_MISALIGNMENT_STD = 1e-3
# Assume the systematic translation is normally distributed
SYSTEM_TRANSLATION_X_MEAN = 0.0  # systematic translation mean (um) - x direction
SYSTEM_TRANSLATION_X_STD = 1e-2  # systematic translation standard deviation (um)  - x direction
SYSTEM_TRANSLATION_Y_MEAN = 0.0  # systematic translation mean (um) - y direction
SYSTEM_TRANSLATION_Y_STD = 1e-2  # systematic translation standard deviation (um)  - y direction
# Assume the systematic rotation is normally distributed
SYSTEM_ROTATION_MEAN = 0.0  # systematic rotation mean (rad)
SYSTEM_ROTATION_STD = 1e-7  # systematic rotation standard deviation (rad)
# Assume the bow difference is normally distributed
BOW_DIFFERENCE_MEAN = 0.0  # bow difference mean (um)
BOW_DIFFERENCE_STD = 30  # bow difference standard deviation (um)
# systematic magnification model: e = k_mag * bow_diff
k_mag = 0.03
M_0 = -1
SYSTEM_MAGNIFICATION_MEAN = (k_mag * BOW_DIFFERENCE_MEAN + M_0) / 1e6  # systematic magnification mean (ppm)
SYSTEM_MAGNIFICATION_STD = (k_mag * BOW_DIFFERENCE_STD) ** 2 / 1e6 # systematic magnification standard deviation (ppm)

# Defect Model Parameters
D0 = 1e-9  # Number of particles of all thicknesses per unit area (um^{-1})
z = 3  # Exponential factor of the particle thickness distribution
t_0 = 1  # The smallest particle thickness (um)
L_m = WAF_R  # Beyond this distance, no voids in the void tail
k_r = 1.8e-4
k_r0 = 230
k_L = 6.2e-2
k_n = 9e-5
k_S = 2.7
cluster_para = 0.1

# Cu expansion model parameters
# Cu pad dish mean and standard deviation at room temperature
TOP_DISH_MEAN = -10.0        # Top Cu pad dish mean (nm), negative value means the dish is concave
TOP_DISH_STD = 0.7          # Top Cu pad dish standard deviation (nm)
BOT_DISH_MEAN = -10.0        # Bottom Cu pad dish mean (nm), negative value means the dish is concave
BOT_DISH_STD = 0.7          # Bottom Cu pad dish standard deviation (nm)
# Expansion coefficient of Cu pad,      ΔH = k_e * ΔT
k_et = 0.052            # Top Cu pad expansion/temp coefficient (nm/K)
k_eb = 0.052            # Bottom Cu pad expansion/temp coefficient (nm/K)
T_R = 25                # Room temperature (°C)
T_anl = 300             # Annealing temperature (°C)

# Roughness model parameters
Asperity_R = 2e-6  # Asperity curvature radius (m)
Roughness_sigma = 1e-9  # Surface roughness standard deviation (m)
eta_s = 40e+12  # Asperity density (m^{-2})
Roughness_constant = 0.07
Adhesion_energy = 1.2  # Adhesion energy (J/m^2)
Young_modulus = 73e+9  # Young's modulus of dielectric material (Pa)
Dielectric_thickness = 1.5e-6  # Dielectric thickness (m)
DISH_0 = 75e-9  # Reference dish (m)
k_peel = 6.66e+15  # Peel force fitting parameter

simulation_times = 1
# Explore the impact of the pitch
particle_density_list = np.logspace(-10, -8.4, 100)
assembly_yield_list = []
single_config_yield_list_array = np.zeros([len(particle_density_list), simulation_times * NUM_WAFERS])
for i, particle_density in enumerate(particle_density_list):
    print("Processing particle density {}/{}, particle dednsity: {}".format(i + 1, len(particle_density_list), particle_density))
    assembly_yield, single_config_yield_list = Assembly_Yield_Simulator(
        NUM_WAFERS=NUM_WAFERS,
        PITCH=PITCH,
        DIE_W=1e+4,
        DIE_L=1e+4,
        WAF_R=WAF_R,
        PAD_TOP_R=PAD_TOP_R,
        PAD_BOT_R=PAD_BOT_R,
        PAD_ARR_ROW=PAD_ARR_ROW,
        PAD_ARR_COL=PAD_ARR_COL,
        VOID_SHAPE=VOID_SHAPE,
        dice_width=1e+4,
        RANDOM_MISALIGNMENT_MEAN=RANDOM_MISALIGNMENT_MEAN,
        RANDOM_MISALIGNMENT_STD=RANDOM_MISALIGNMENT_STD,
        SYSTEM_TRANSLATION_X_MEAN=SYSTEM_TRANSLATION_X_MEAN,
        SYSTEM_TRANSLATION_X_STD=SYSTEM_TRANSLATION_X_STD,
        SYSTEM_TRANSLATION_Y_MEAN=SYSTEM_TRANSLATION_Y_MEAN,
        SYSTEM_TRANSLATION_Y_STD=SYSTEM_TRANSLATION_Y_STD,
        SYSTEM_ROTATION_MEAN=SYSTEM_ROTATION_MEAN,
        SYSTEM_ROTATION_STD=SYSTEM_ROTATION_STD,
        BOW_DIFFERENCE_MEAN=BOW_DIFFERENCE_MEAN,
        BOW_DIFFERENCE_STD=BOW_DIFFERENCE_STD,
        k_mag=k_mag,
        M_0=M_0,
        D0=particle_density,
        z=z,
        t_0=t_0,
        k_r=k_r,
        k_r0=k_r0,
        k_L=k_L,
        k_n=k_n,
        k_S=k_S,
        TOP_DISH_MEAN=TOP_DISH_MEAN,
        TOP_DISH_STD=TOP_DISH_STD,
        BOT_DISH_MEAN=BOT_DISH_MEAN,
        BOT_DISH_STD=BOT_DISH_STD,
        k_et=k_et,
        k_eb=k_eb,
        T_R=T_R,
        T_anl=T_anl,
        Asperity_R=Asperity_R,
        Roughness_sigma=Roughness_sigma,
        eta_s=eta_s,
        Roughness_constant=Roughness_constant,
        Adhesion_energy=Adhesion_energy,
        Young_modulus=Young_modulus,
        Dielectric_thickness = Dielectric_thickness,
        DISH_0 = DISH_0,
        k_peel = k_peel,
        simulation_times=simulation_times,      
        approximate_set=100                                                           
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