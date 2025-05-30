#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import fsolve
import sympy as sp
from scipy.integrate import quad
from scipy.stats import norm



# Calculate the misalignment of the pad based on the systematic translation, rotation, and magnification
def die_pad_misalignment(
    die,
    system_translation_x,
    system_translation_y,
    system_rotation,
    system_magnification,
):
    pad_misalignment = np.zeros(len(die.pad_array_box))
    dx = (system_translation_x - system_rotation * die.pad_array_box[:, 1] + system_magnification * die.pad_array_box[:, 0])
    dy = (system_translation_y + system_rotation * die.pad_array_box[:, 0] + system_magnification * die.pad_array_box[:, 1])
    pad_misalignment = np.sqrt(dx**2 + dy**2)
    return pad_misalignment

def overlay_yield_calculator(
    PAD_TOP_R,
    PAD_BOT_R,
    PITCH,
    CONTACT_AREA_CONSTRAINT,
    CRITICAL_DIST_CONSTRAINT,
    SYSTEM_MAGNIFICATION_MEAN,
    SYSTEM_MAGNIFICATION_STD,
    SYSTEM_ROTATION_MEAN,
    SYSTEM_ROTATION_STD,
    SYSTEM_TRANSLATION_X_MEAN,
    SYSTEM_TRANSLATION_X_STD,
    SYSTEM_TRANSLATION_Y_MEAN,
    SYSTEM_TRANSLATION_Y_STD,
    RANDOM_MISALIGNMENT_MEAN,
    RANDOM_MISALIGNMENT_STD,
    wafer,    
    num_samples=100000,
):
    def max_allowed_misalignment_calculator(
        PAD_TOP_R, PAD_BOT_R, PITCH, CONTACT_AREA_CONSTRAINT, CRITICAL_DIST_CONSTRAINT
    ):
        # Calculate the overlay misalignment that will fail the contact area constraint
        system_misalignment = sp.symbols("system_misalignment")
        theta1 = sp.acos((PAD_TOP_R**2 + system_misalignment**2 - PAD_BOT_R**2) / (2 * PAD_TOP_R * system_misalignment))
        theta2 = sp.acos((PAD_BOT_R**2 + system_misalignment**2 - PAD_TOP_R**2) / (2 * PAD_BOT_R * system_misalignment))
        contact_area = (PAD_TOP_R**2 * theta1 + PAD_BOT_R**2 * theta2 - system_misalignment * (PAD_TOP_R * sp.sin(theta1)))
        equation = sp.lambdify(system_misalignment, contact_area - CONTACT_AREA_CONSTRAINT * np.pi * PAD_TOP_R**2, "numpy")
        max_allowed_misalignment_for_ca = fsolve(equation, PAD_BOT_R)
        # print("The overlay misalignment that will fail the contact area constraint is {} um.".format(max_allowed_misalignment_for_ca[0]))
        # Calculate the overlay misalignment that will fail the contact area constraint
        system_misalignment = np.linspace(PAD_BOT_R - PAD_TOP_R, PAD_BOT_R + PAD_TOP_R, 1000)
        theta1 = np.arccos((PAD_TOP_R**2 + system_misalignment**2 - PAD_BOT_R**2) / (2 * PAD_TOP_R * system_misalignment))
        theta2 = np.arccos((PAD_BOT_R**2 + system_misalignment**2 - PAD_TOP_R**2) / (2 * PAD_BOT_R * system_misalignment))
        contact_area = (PAD_TOP_R**2 * theta1 + PAD_BOT_R**2 * theta2 - system_misalignment * (PAD_TOP_R * np.sin(theta1)))
        # plt.plot(system_misalignment, contact_area / (np.pi * PAD_TOP_R**2))
        # plt.axhline(y=CONTACT_AREA_CONSTRAINT, color="r", linestyle="--")
        # plt.axvline(x=max_allowed_misalignment_for_ca, color="g", linestyle="--")
        # plt.xlabel("System Misalignment (um)")
        # plt.ylabel("Contact Area Ratio")
        # plt.title("Contact Area Ratio vs. System Misalignment")
        # plt.show()

        # Calculate the overlay misalignment that will fail the critical distance constraint
        max_allowed_misalignment_for_cd = (1 - CRITICAL_DIST_CONSTRAINT) * PITCH - 0.5 * (2 * PAD_TOP_R) + (CRITICAL_DIST_CONSTRAINT - 0.5) * (2 * PAD_BOT_R)
        
        # print("The overlay misalignment that will fail the critical distance constraint is {} um.".format(max_allowed_misalignment_for_cd))

        MAX_ALLOWED_MISALIGNMENT = min(max_allowed_misalignment_for_ca[0], max_allowed_misalignment_for_cd)
        # print("The overlay misalignment that will fail the both constraints is {} um.".format(MAX_ALLOWED_MISALIGNMENT))

        return MAX_ALLOWED_MISALIGNMENT
    
    MAX_ALLOWED_MISALIGNMENT = max_allowed_misalignment_calculator(
        PAD_TOP_R,
        PAD_BOT_R,
        PITCH,
        CONTACT_AREA_CONSTRAINT,
        CRITICAL_DIST_CONSTRAINT,
    )
    # print("The maximum allowed misalignment is {} um.".format(MAX_ALLOWED_MISALIGNMENT))
    num_samples = num_samples
    system_translation_x_samples = np.random.normal(SYSTEM_TRANSLATION_X_MEAN, SYSTEM_TRANSLATION_X_STD, num_samples)
    system_translation_y_samples = np.random.normal(SYSTEM_TRANSLATION_Y_MEAN, SYSTEM_TRANSLATION_Y_STD, num_samples)
    system_rotation_samples = np.random.normal(SYSTEM_ROTATION_MEAN, SYSTEM_ROTATION_STD, num_samples)
    system_magnification_samples = np.random.normal(SYSTEM_MAGNIFICATION_MEAN, SYSTEM_MAGNIFICATION_STD, num_samples)
    overlay_die_yield_list = []
    # print("SYSTEM_TRANSLATION_X_MEAN: ", SYSTEM_TRANSLATION_X_MEAN)
    # print("SYSTEM_TRANSLATION_X_STD: ", SYSTEM_TRANSLATION_X_STD)
    # print("SYSTEM_TRANSLATION_Y_MEAN: ", SYSTEM_TRANSLATION_Y_MEAN)
    # print("SYSTEM_TRANSLATION_Y_STD: ", SYSTEM_TRANSLATION_Y_STD)
    # print("SYSTEM_ROTATION_MEAN: ", SYSTEM_ROTATION_MEAN)
    # print("SYSTEM_ROTATION_STD: ", SYSTEM_ROTATION_STD)
    # print("SYSTEM_MAGNIFICATION_MEAN: ", SYSTEM_MAGNIFICATION_MEAN)
    # print("SYSTEM_MAGNIFICATION_STD: ", SYSTEM_MAGNIFICATION_STD)
    # print("RANDOM_MISALIGNMENT_MEAN: ", RANDOM_MISALIGNMENT_MEAN)
    # print("RANDOM_MISALIGNMENT_STD: ", RANDOM_MISALIGNMENT_STD)
    # print("MAX_ALLOWED_MISALIGNMENT: ", MAX_ALLOWED_MISALIGNMENT)
    # print(system_translation_x_samples.mean()*1e3, " nm")
    # print(system_translation_y_samples.mean()*1e3, " nm")
    # print(system_rotation_samples.mean() * 150e+3 * 1e3, " nm")
    # print(system_magnification_samples.mean() * 150e+3 * 1e3, " nm")
    # far_pad_misalignment_samples_list = []
    # print("Total: ", np.sqrt((system_translation_x_samples.mean()*1e3 + system_magnification_samples.mean() * 150e+3 * 1e3)**2 + (system_translation_y_samples.mean()*1e3 + system_rotation_samples.mean() * 150e+3 * 1e3)**2), " nm")
    for die in wafer.die_list:
        # distances = np.sqrt(die.pad_array_box[:, 0]**2 + die.pad_array_box[:, 1]**2)
        # max_distance_index = np.argmax(distances)
        # # print("The maximum distance is {} um.".format(distances[max_distance_index]))
        # far_pad_x = die.pad_array_box[max_distance_index, 0]
        # far_pad_y = die.pad_array_box[max_distance_index, 1]
        # far_dx_samples = (system_translation_x_samples - system_rotation_samples * far_pad_y + system_magnification_samples * far_pad_x)
        # far_dy_samples = (system_translation_y_samples + system_rotation_samples * far_pad_x + system_magnification_samples * far_pad_y)
        # far_pad_misalignment_samples = np.sqrt(far_dx_samples**2 + far_dy_samples**2)
        far_dx_samples_0 = (system_translation_x_samples - system_rotation_samples * die.pad_array_box[0, 1] + system_magnification_samples * die.pad_array_box[0, 0])
        far_dy_samples_0 = (system_translation_y_samples + system_rotation_samples * die.pad_array_box[0, 0] + system_magnification_samples * die.pad_array_box[0, 1])
        far_dx_samples_1 = (system_translation_x_samples - system_rotation_samples * die.pad_array_box[1, 1] + system_magnification_samples * die.pad_array_box[1, 0])
        far_dy_samples_1 = (system_translation_y_samples + system_rotation_samples * die.pad_array_box[1, 0] + system_magnification_samples * die.pad_array_box[1, 1])
        far_dx_samples_2 = (system_translation_x_samples - system_rotation_samples * die.pad_array_box[2, 1] + system_magnification_samples * die.pad_array_box[2, 0])
        far_dy_samples_2 = (system_translation_y_samples + system_rotation_samples * die.pad_array_box[2, 0] + system_magnification_samples * die.pad_array_box[2, 1])
        far_dx_samples_3 = (system_translation_x_samples - system_rotation_samples * die.pad_array_box[3, 1] + system_magnification_samples * die.pad_array_box[3, 0])
        far_dy_samples_3 = (system_translation_y_samples + system_rotation_samples * die.pad_array_box[3, 0] + system_magnification_samples * die.pad_array_box[3, 1])
        far_pad_misalignment_samples_0 = np.sqrt(far_dx_samples_0**2 + far_dy_samples_0**2)
        far_pad_misalignment_samples_1 = np.sqrt(far_dx_samples_1**2 + far_dy_samples_1**2)
        far_pad_misalignment_samples_2 = np.sqrt(far_dx_samples_2**2 + far_dy_samples_2**2)
        far_pad_misalignment_samples_3 = np.sqrt(far_dx_samples_3**2 + far_dy_samples_3**2)
        far_pad_misalignment_samples = np.array([far_pad_misalignment_samples_0, far_pad_misalignment_samples_1, far_pad_misalignment_samples_2, far_pad_misalignment_samples_3]).max(axis=0)
        # far_pad_misalignment_samples_list.append(far_pad_misalignment_samples.mean())
        upper_limit = MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples
        lower_limit = -MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples
        current_die_yield = norm.cdf(upper_limit, loc=RANDOM_MISALIGNMENT_MEAN, scale=RANDOM_MISALIGNMENT_STD) \
                          - norm.cdf(lower_limit, loc=RANDOM_MISALIGNMENT_MEAN, scale=RANDOM_MISALIGNMENT_STD)
        current_die_yield = np.mean(current_die_yield)
        # if die.avg_misalignment_far + np.random.normal(RANDOM_MISALIGNMENT_MEAN, RANDOM_MISALIGNMENT_STD) > MAX_ALLOWED_MISALIGNMENT:
        #     current_die_yield = 0.0
        # else:
        #     current_die_yield = 1.0
        overlay_die_yield_list.append(current_die_yield)
        # print("The overlay die yield is {}.".format(current_die_yield))
    # print("The max far pad misalignment is {} nm.".format(max(far_pad_misalignment_samples_list)*1e3))
    overlay_die_yield = np.mean(overlay_die_yield_list)
    # print("The overlay die yield is {}.".format(overlay_die_yield))

    return overlay_die_yield
    