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
    system_translation_x_um,
    system_translation_y_um,
    system_rotation_rad,
    system_magnification_ppm,
    RANDOM_MISALIGNMENT_MEAN_um,
    RANDOM_MISALIGNMENT_STD_um,
    approximate_set,
    redundant_flag,
):
    if approximate_set != 1:
        # Consider pad misalignment pads at the outer edge of the die
        if redundant_flag == True:
            pad_misalignment = np.zeros(len(die.ovl_critical_pad_boundary_coords))
            dx = (system_translation_x_um - system_rotation_rad * die.ovl_critical_pad_boundary_coords[:, 1] + system_magnification_ppm * die.ovl_critical_pad_boundary_coords[:, 0])
            dy = (system_translation_y_um + system_rotation_rad * die.ovl_critical_pad_boundary_coords[:, 0] + system_magnification_ppm * die.ovl_critical_pad_boundary_coords[:, 1])
            pad_misalignment = np.sqrt(dx**2 + dy**2) + np.random.normal(RANDOM_MISALIGNMENT_MEAN_um, RANDOM_MISALIGNMENT_STD_um, len(die.ovl_critical_pad_boundary_coords))
        else:
            pad_misalignment = np.zeros(len(die.pad_array_box))
            dx = (system_translation_x_um - system_rotation_rad * die.pad_array_box[:, 1] + system_magnification_ppm * die.pad_array_box[:, 0])
            dy = (system_translation_y_um + system_rotation_rad * die.pad_array_box[:, 0] + system_magnification_ppm * die.pad_array_box[:, 1])
            pad_misalignment = np.sqrt(dx**2 + dy**2) + np.random.normal(RANDOM_MISALIGNMENT_MEAN_um, RANDOM_MISALIGNMENT_STD_um, len(die.pad_array_box))
    else:
        die_pad_coords = base_pad_coords + die.die_center
        pad_misalignment = np.zeros(len(die_pad_coords))
        dx = (system_translation_x_um - system_rotation_rad * die_pad_coords[:, 1] + system_magnification_ppm * die_pad_coords[:, 0])
        dy = (system_translation_y_um + system_rotation_rad * die_pad_coords[:, 0] + system_magnification_ppm * die_pad_coords[:, 1])
        pad_misalignment = np.sqrt(dx**2 + dy**2) + np.random.normal(RANDOM_MISALIGNMENT_MEAN_um, RANDOM_MISALIGNMENT_STD_um, len(die_pad_coords))

    return pad_misalignment

def overlay_term_simulator(
    PAD_TOP_R_um: float,
    PAD_BOT_R_um: float,
    PITCH_r_um: float,
    PITCH_c_um: float,
    CONTACT_AREA_CONSTRAINT: float,
    CRITICAL_DIST_CONSTRAINT: float,
    SYSTEM_ROTATION_MEAN_rad: float,
    SYSTEM_ROTATION_STD_rad: float,
    SYSTEM_TRANSLATION_X_MEAN_um: float,
    SYSTEM_TRANSLATION_X_STD_um: float,
    SYSTEM_TRANSLATION_Y_MEAN_um: float,
    SYSTEM_TRANSLATION_Y_STD_um: float,
    BOW_DIFFERENCE_MEAN_um: float,
    BOW_DIFFERENCE_STD_um: float,
    NUM_DIES: int,
    k_mag: float,
    M_0: float,
):
    def max_allowed_misalignment_calculator(
        PAD_TOP_R_um, PAD_BOT_R_um, PITCH_r_um, PITCH_c_um, CONTACT_AREA_CONSTRAINT, CRITICAL_DIST_CONSTRAINT
    ):
        # Calculate the overlay misalignment that will fail the contact area constraint
        system_misalignment = sp.symbols("system_misalignment")
        theta1 = sp.acos((PAD_TOP_R_um**2 + system_misalignment**2 - PAD_BOT_R_um**2) / (2 * PAD_TOP_R_um * system_misalignment))
        theta2 = sp.acos((PAD_BOT_R_um**2 + system_misalignment**2 - PAD_TOP_R_um**2) / (2 * PAD_BOT_R_um * system_misalignment))
        contact_area = (PAD_TOP_R_um**2 * theta1 + PAD_BOT_R_um**2 * theta2 - system_misalignment * (PAD_TOP_R_um * sp.sin(theta1)))
        equation = sp.lambdify(system_misalignment, contact_area - CONTACT_AREA_CONSTRAINT * np.pi * PAD_TOP_R_um**2, "numpy")
        max_allowed_misalignment_for_ca = fsolve(equation, PAD_BOT_R_um)
        # print("The overlay misalignment that will fail the contact area constraint is {} um.".format(max_allowed_misalignment_for_ca[0]))
        # Calculate the overlay misalignment that will fail the contact area constraint
        system_misalignment = np.linspace(PAD_BOT_R_um - PAD_TOP_R_um, PAD_BOT_R_um + PAD_TOP_R_um, 1000)
        theta1 = np.arccos((PAD_TOP_R_um**2 + system_misalignment**2 - PAD_BOT_R_um**2) / (2 * PAD_TOP_R_um * system_misalignment))
        theta2 = np.arccos((PAD_BOT_R_um**2 + system_misalignment**2 - PAD_TOP_R_um**2) / (2 * PAD_BOT_R_um * system_misalignment))
        contact_area = (PAD_TOP_R_um**2 * theta1 + PAD_BOT_R_um**2 * theta2 - system_misalignment * (PAD_TOP_R_um * np.sin(theta1)))
        # plt.plot(system_misalignment, contact_area / (np.pi * PAD_TOP_R_um**2))
        # plt.axhline(y=CONTACT_AREA_CONSTRAINT, color="r", linestyle="--")
        # plt.axvline(x=max_allowed_misalignment_for_ca, color="g", linestyle="--")
        # plt.xlabel("System Misalignment (um)")
        # plt.ylabel("Contact Area Ratio")
        # plt.title("Contact Area Ratio vs. System Misalignment")
        # plt.show()

        # Calculate the overlay misalignment that will fail the critical distance constraint
        max_allowed_misalignment_for_cd = (1 - CRITICAL_DIST_CONSTRAINT) * min(PITCH_r_um, PITCH_c_um) - 0.5 * (2 * PAD_TOP_R_um) + (CRITICAL_DIST_CONSTRAINT - 0.5) * (2 * PAD_BOT_R_um)
        # print("The overlay misalignment that will fail the critical distance constraint is {} um.".format(max_allowed_misalignment_for_cd))

        MAX_ALLOWED_MISALIGNMENT_um = min(max_allowed_misalignment_for_ca[0], max_allowed_misalignment_for_cd)
        # print("The overlay misalignment that will fail the both constraints is {} um.".format(MAX_ALLOWED_MISALIGNMENT))

        return MAX_ALLOWED_MISALIGNMENT_um
    
    # Calculate the maximum allowed misalignment
    MAX_ALLOWED_MISALIGNMENT_um = max_allowed_misalignment_calculator(
        PAD_TOP_R_um, PAD_BOT_R_um, PITCH_r_um, PITCH_c_um, CONTACT_AREA_CONSTRAINT, CRITICAL_DIST_CONSTRAINT
    )
    
    # Calculate the systematic translation, rotation, and magnification
    system_translation_x_um = (
        np.random.normal(SYSTEM_TRANSLATION_X_MEAN_um, SYSTEM_TRANSLATION_X_STD_um, NUM_DIES)
    )
    system_translation_y_um = (
        np.random.normal(SYSTEM_TRANSLATION_Y_MEAN_um, SYSTEM_TRANSLATION_Y_STD_um, NUM_DIES)
    )
    system_rotation_rad = (
        np.random.normal(SYSTEM_ROTATION_MEAN_rad, SYSTEM_ROTATION_STD_rad, NUM_DIES)
    )
    bow_difference = np.random.normal(BOW_DIFFERENCE_MEAN_um, BOW_DIFFERENCE_STD_um, NUM_DIES)
    system_magnification_ppm = (
        (k_mag * bow_difference + M_0) / 1e6
    )  # systematic magnification unit (ppm)

    return system_translation_x_um, system_translation_y_um, system_rotation_rad, system_magnification_ppm, MAX_ALLOWED_MISALIGNMENT_um