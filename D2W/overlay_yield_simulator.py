#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Overlay term simulator for the yield model for D2W hybrid bonding
#### Author: Zhichao Chen
#### Date: Oct 4, 2024

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
    base_pad_coords,
    system_translation_x,
    system_translation_y,
    system_rotation,
    system_magnification,
    RANDOM_MISALIGNMENT_MEAN,
    RANDOM_MISALIGNMENT_STD,
    approximate_set,
    redundant_flag,
):
    if approximate_set != 1:
        # Consider pad misalignment pads at the outer edge of the die
        if redundant_flag == True:
            pad_misalignment = np.zeros(len(die.ovl_critical_pad_boundary_coords))
            dx = (system_translation_x - system_rotation * die.ovl_critical_pad_boundary_coords[:, 1] + system_magnification * die.ovl_critical_pad_boundary_coords[:, 0])
            dy = (system_translation_y + system_rotation * die.ovl_critical_pad_boundary_coords[:, 0] + system_magnification * die.ovl_critical_pad_boundary_coords[:, 1])
            pad_misalignment = np.sqrt(dx**2 + dy**2) + np.random.normal(RANDOM_MISALIGNMENT_MEAN, RANDOM_MISALIGNMENT_STD, len(die.ovl_critical_pad_boundary_coords))
        else:
            pad_misalignment = np.zeros(len(die.pad_array_box))
            dx = (system_translation_x - system_rotation * die.pad_array_box[:, 1] + system_magnification * die.pad_array_box[:, 0])
            dy = (system_translation_y + system_rotation * die.pad_array_box[:, 0] + system_magnification * die.pad_array_box[:, 1])
            pad_misalignment = np.sqrt(dx**2 + dy**2) + np.random.normal(RANDOM_MISALIGNMENT_MEAN, RANDOM_MISALIGNMENT_STD, len(die.pad_array_box))
    else:
        die_pad_coords = base_pad_coords + die.die_center
        pad_misalignment = np.zeros(len(die_pad_coords))
        dx = (system_translation_x - system_rotation * die_pad_coords[:, 1] + system_magnification * die_pad_coords[:, 0])
        dy = (system_translation_y + system_rotation * die_pad_coords[:, 0] + system_magnification * die_pad_coords[:, 1])
        pad_misalignment = np.sqrt(dx**2 + dy**2) + np.random.normal(RANDOM_MISALIGNMENT_MEAN, RANDOM_MISALIGNMENT_STD, len(die_pad_coords))

    return pad_misalignment

def overlay_term_simulator(
    PAD_TOP_R,
    PAD_BOT_R,
    PITCH,
    CONTACT_AREA_CONSTRAINT,
    CRITICAL_DIST_CONSTRAINT,
    SYSTEM_ROTATION_MEAN,
    SYSTEM_ROTATION_STD,
    SYSTEM_TRANSLATION_X_MEAN,
    SYSTEM_TRANSLATION_X_STD,
    SYSTEM_TRANSLATION_Y_MEAN,
    SYSTEM_TRANSLATION_Y_STD,
    BOW_DIFFERENCE_MEAN,
    BOW_DIFFERENCE_STD,
    NUM_DIES,
    k_mag,
    M_0
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
    
    # Calculate the maximum allowed misalignment
    MAX_ALLOWED_MISALIGNMENT = max_allowed_misalignment_calculator(
        PAD_TOP_R, PAD_BOT_R, PITCH, CONTACT_AREA_CONSTRAINT, CRITICAL_DIST_CONSTRAINT
    )
    
    # Calculate the systematic translation, rotation, and magnification
    system_translation_x = (
        np.random.normal(SYSTEM_TRANSLATION_X_MEAN, SYSTEM_TRANSLATION_X_STD, NUM_DIES)
    )
    system_translation_y = (
        np.random.normal(SYSTEM_TRANSLATION_Y_MEAN, SYSTEM_TRANSLATION_Y_STD, NUM_DIES)
    )
    system_rotation = (
        np.random.normal(SYSTEM_ROTATION_MEAN, SYSTEM_ROTATION_STD, NUM_DIES)
    )
    bow_difference = np.random.normal(BOW_DIFFERENCE_MEAN, BOW_DIFFERENCE_STD, NUM_DIES)
    system_magnification = (
        (k_mag * bow_difference + M_0) / 1e6
    )  # systematic magnification unit (ppm)

    return system_translation_x, system_translation_y, system_rotation, system_magnification, MAX_ALLOWED_MISALIGNMENT