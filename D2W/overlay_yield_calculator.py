#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Author: Zhichao Chen
#### Date: Oct 1, 2025

'''
Overlay yield calculator for D2W hybrid bonding:
1. Calculate the maximum allowed misalignment
2. Calculate the systematic misalignment for every pad based on the systematic translation, rotation, and magnification
3. Calculate the overlay yield:
    i. If pad_yield_flag is True, calculate the overlay yield for each pad and return the pad yield map.
    ii. If pad_yield_flag is False, calculate the overlay yield for the die based on the worst-case pad misalignment.
4. Calculate the overall overlay yield for the die.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import fsolve
import math
import sympy as sp
from scipy.integrate import quad
from scipy.stats import norm

# Calculate the misalignment of the pad based on the systematic translation, rotation, and magnification
def die_pad_misalignment(
    die,
    system_translation_x_um: float,
    system_translation_y_um: float,
    system_rotation_um: float,
    system_magnification: float,
):
    pad_misalignment = np.zeros(len(die.pad_array_box))
    dx = (system_translation_x_um - system_rotation_um * die.pad_array_box[:, 1] + system_magnification * die.pad_array_box[:, 0])
    dy = (system_translation_y_um + system_rotation_um * die.pad_array_box[:, 0] + system_magnification * die.pad_array_box[:, 1])
    pad_misalignment = np.sqrt(dx**2 + dy**2)
    return pad_misalignment

def max_allowed_misalignment_calculator(*,
        PAD_TOP_R_um: float, 
        PAD_BOT_R_um: float, 
        PITCH_r_um: float,
        PITCH_c_um: float,
        CONTACT_AREA_CONSTRAINT, 
        CRITICAL_DIST_CONSTRAINT
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

        MAX_ALLOWED_MISALIGNMENT = min(max_allowed_misalignment_for_ca[0], max_allowed_misalignment_for_cd)
        # print("The overlay misalignment that will fail the both constraints is {} um.".format(MAX_ALLOWED_MISALIGNMENT))

        return MAX_ALLOWED_MISALIGNMENT

def overlay_yield_calculator(*,
    PAD_TOP_R_um: float,
    PAD_BOT_R_um: float,
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    PITCH_r_um: float,
    PITCH_c_um: float,
    num_samples: int,
    CONTACT_AREA_CONSTRAINT: float,
    CRITICAL_DIST_CONSTRAINT: float,
    SYSTEM_MAGNIFICATION_MEAN_ppm: float,
    SYSTEM_MAGNIFICATION_STD_ppm: float,
    SYSTEM_ROTATION_MEAN_rad: float,
    SYSTEM_ROTATION_STD_rad: float,
    SYSTEM_TRANSLATION_X_MEAN_um: float,
    SYSTEM_TRANSLATION_X_STD_um: float,
    SYSTEM_TRANSLATION_Y_MEAN_um: float,
    SYSTEM_TRANSLATION_Y_STD_um: float,
    RANDOM_MISALIGNMENT_MEAN_um: float,
    RANDOM_MISALIGNMENT_STD_um: float,
    die,
    redundant_flag: bool,
    pad_yield_flag: bool = False,
    pad_yield_map_sub_factor: int = 1,
):  
    MAX_ALLOWED_MISALIGNMENT = max_allowed_misalignment_calculator(
        PAD_TOP_R_um=PAD_TOP_R_um,
        PAD_BOT_R_um=PAD_BOT_R_um,
        PITCH_r_um=PITCH_r_um,
        PITCH_c_um=PITCH_c_um,
        CONTACT_AREA_CONSTRAINT=CONTACT_AREA_CONSTRAINT,
        CRITICAL_DIST_CONSTRAINT=CRITICAL_DIST_CONSTRAINT,
    )
    # print("PAD_TOP_R_um: ", PAD_TOP_R_um, "um")
    # print("PAD_BOT_R_um: ", PAD_BOT_R_um, "um")
    # print("PITCH_r_um: ", PITCH_r_um, "um")
    # print("PITCH_c_um: ", PITCH_c_um, "um")
    # print("CONTACT_AREA_CONSTRAINT: ", CONTACT_AREA_CONSTRAINT)
    # print("CRITICAL_DIST_CONSTRAINT: ", CRITICAL_DIST_CONSTRAINT)
    print("The maximum allowed misalignment is {} nm.".format(MAX_ALLOWED_MISALIGNMENT * 1e3))
    num_samples = num_samples
    system_translation_x_samples_um = np.random.normal(SYSTEM_TRANSLATION_X_MEAN_um, SYSTEM_TRANSLATION_X_STD_um, num_samples)
    system_translation_y_samples_um = np.random.normal(SYSTEM_TRANSLATION_Y_MEAN_um, SYSTEM_TRANSLATION_Y_STD_um, num_samples)
    system_rotation_samples_rad = np.random.normal(SYSTEM_ROTATION_MEAN_rad, SYSTEM_ROTATION_STD_rad, num_samples)
    system_magnification_samples = np.random.normal(SYSTEM_MAGNIFICATION_MEAN_ppm, SYSTEM_MAGNIFICATION_STD_ppm, num_samples)
    print("system_translation_x_samples_um contribution", system_translation_x_samples_um.mean()*1e3, " nm")
    print("system_translation_y_samples_um contribution", system_translation_y_samples_um.mean()*1e3, " nm")
    print("system_rotation_samples_rad contribution", system_rotation_samples_rad.mean() * np.sqrt(die.DIE_W_um**2 + die.DIE_L_um**2) * 1e3, " nm")
    print("system_magnification_samples contribution", system_magnification_samples.mean() * np.sqrt(die.DIE_W_um**2 + die.DIE_L_um**2) * 1e3, " nm")

    # Sample the systematic misalignment for corner pads based on the systematic translation, rotation, and magnification
    # Calculate the die yield based on the worst-case pad misalignment
    if redundant_flag == True:
        far_dx_samples_0 = (system_translation_x_samples_um - system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[0, 1] + system_magnification_samples * die.ovl_critical_pad_boundary_coords[0, 0])
        far_dy_samples_0 = (system_translation_y_samples_um + system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[0, 0] + system_magnification_samples * die.ovl_critical_pad_boundary_coords[0, 1])
        far_dx_samples_1 = (system_translation_x_samples_um - system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[1, 1] + system_magnification_samples * die.ovl_critical_pad_boundary_coords[1, 0])
        far_dy_samples_1 = (system_translation_y_samples_um + system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[1, 0] + system_magnification_samples * die.ovl_critical_pad_boundary_coords[1, 1])
        far_dx_samples_2 = (system_translation_x_samples_um - system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[2, 1] + system_magnification_samples * die.ovl_critical_pad_boundary_coords[2, 0])
        far_dy_samples_2 = (system_translation_y_samples_um + system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[2, 0] + system_magnification_samples * die.ovl_critical_pad_boundary_coords[2, 1])
        far_dx_samples_3 = (system_translation_x_samples_um - system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[3, 1] + system_magnification_samples * die.ovl_critical_pad_boundary_coords[3, 0])
        far_dy_samples_3 = (system_translation_y_samples_um + system_rotation_samples_rad * die.ovl_critical_pad_boundary_coords[3, 0] + system_magnification_samples * die.ovl_critical_pad_boundary_coords[3, 1])
    else:
        far_dx_samples_0 = (system_translation_x_samples_um - system_rotation_samples_rad * die.pad_array_box[0, 1] + system_magnification_samples * die.pad_array_box[0, 0])
        far_dy_samples_0 = (system_translation_y_samples_um + system_rotation_samples_rad * die.pad_array_box[0, 0] + system_magnification_samples * die.pad_array_box[0, 1])
        far_dx_samples_1 = (system_translation_x_samples_um - system_rotation_samples_rad * die.pad_array_box[1, 1] + system_magnification_samples * die.pad_array_box[1, 0])
        far_dy_samples_1 = (system_translation_y_samples_um + system_rotation_samples_rad * die.pad_array_box[1, 0] + system_magnification_samples * die.pad_array_box[1, 1])
        far_dx_samples_2 = (system_translation_x_samples_um - system_rotation_samples_rad * die.pad_array_box[2, 1] + system_magnification_samples * die.pad_array_box[2, 0])
        far_dy_samples_2 = (system_translation_y_samples_um + system_rotation_samples_rad * die.pad_array_box[2, 0] + system_magnification_samples * die.pad_array_box[2, 1])
        far_dx_samples_3 = (system_translation_x_samples_um - system_rotation_samples_rad * die.pad_array_box[3, 1] + system_magnification_samples * die.pad_array_box[3, 0])
        far_dy_samples_3 = (system_translation_y_samples_um + system_rotation_samples_rad * die.pad_array_box[3, 0] + system_magnification_samples * die.pad_array_box[3, 1])
    far_pad_misalignment_samples_0 = np.sqrt(far_dx_samples_0**2 + far_dy_samples_0**2)
    far_pad_misalignment_samples_1 = np.sqrt(far_dx_samples_1**2 + far_dy_samples_1**2)
    far_pad_misalignment_samples_2 = np.sqrt(far_dx_samples_2**2 + far_dy_samples_2**2)
    far_pad_misalignment_samples_3 = np.sqrt(far_dx_samples_3**2 + far_dy_samples_3**2)

    upper_limit_0 = MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_0
    lower_limit_0 = -MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_0
    upper_limit_1 = MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_1
    lower_limit_1 = -MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_1
    upper_limit_2 = MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_2
    lower_limit_2 = -MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_2
    upper_limit_3 = MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_3
    lower_limit_3 = -MAX_ALLOWED_MISALIGNMENT - far_pad_misalignment_samples_3

    overlay_die_yield_0 = np.mean(norm.cdf(upper_limit_0, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) \
                        - norm.cdf(lower_limit_0, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
    overlay_die_yield_1 = np.mean(norm.cdf(upper_limit_1, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) \
                        - norm.cdf(lower_limit_1, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
    overlay_die_yield_2 = np.mean(norm.cdf(upper_limit_2, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) \
                        - norm.cdf(lower_limit_2, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
    overlay_die_yield_3 = np.mean(norm.cdf(upper_limit_3, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um) \
                        - norm.cdf(lower_limit_3, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um))
    overlay_die_yield = min(overlay_die_yield_0, overlay_die_yield_1, overlay_die_yield_2, overlay_die_yield_3)
    
        
    return overlay_die_yield







def pad_overlay_yield_map_generator(*,
    PAD_TOP_R_um: float,
    PAD_BOT_R_um: float,
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    PITCH_r_um: float,
    PITCH_c_um: float,
    num_samples: int,
    CONTACT_AREA_CONSTRAINT: float,
    CRITICAL_DIST_CONSTRAINT: float,
    SYSTEM_MAGNIFICATION_MEAN_ppm: float,
    SYSTEM_MAGNIFICATION_STD_ppm: float,
    SYSTEM_ROTATION_MEAN_rad: float,
    SYSTEM_ROTATION_STD_rad: float,
    SYSTEM_TRANSLATION_X_MEAN_um: float,
    SYSTEM_TRANSLATION_X_STD_um: float,
    SYSTEM_TRANSLATION_Y_MEAN_um: float,
    SYSTEM_TRANSLATION_Y_STD_um: float,
    RANDOM_MISALIGNMENT_MEAN_um: float,
    RANDOM_MISALIGNMENT_STD_um: float,
    die,
    pad_yield_flag: bool = False,
    pad_yield_map_sub_factor: int = 1,
):  
    MAX_ALLOWED_MISALIGNMENT = max_allowed_misalignment_calculator(
        PAD_TOP_R_um=PAD_TOP_R_um,
        PAD_BOT_R_um=PAD_BOT_R_um,
        PITCH_r_um=PITCH_r_um,
        PITCH_c_um=PITCH_c_um,
        CONTACT_AREA_CONSTRAINT=CONTACT_AREA_CONSTRAINT,
        CRITICAL_DIST_CONSTRAINT=CRITICAL_DIST_CONSTRAINT,
    )
    print("The maximum allowed misalignment is {} nm.".format(MAX_ALLOWED_MISALIGNMENT * 1e3))
    num_samples = num_samples
    system_translation_x_samples_um = np.random.normal(SYSTEM_TRANSLATION_X_MEAN_um, SYSTEM_TRANSLATION_X_STD_um, num_samples)
    system_translation_y_samples_um = np.random.normal(SYSTEM_TRANSLATION_Y_MEAN_um, SYSTEM_TRANSLATION_Y_STD_um, num_samples)
    system_rotation_samples_rad = np.random.normal(SYSTEM_ROTATION_MEAN_rad, SYSTEM_ROTATION_STD_rad, num_samples)
    system_magnification_samples = np.random.normal(SYSTEM_MAGNIFICATION_MEAN_ppm, SYSTEM_MAGNIFICATION_STD_ppm, num_samples)
    print("system_translation_x_samples_um contribution", system_translation_x_samples_um.mean()*1e3, " nm")
    print("system_translation_y_samples_um contribution", system_translation_y_samples_um.mean()*1e3, " nm")
    print("system_rotation_samples_rad contribution", system_rotation_samples_rad.mean() * np.sqrt(die.DIE_W_um**2 + die.DIE_L_um**2) * 1e3, " nm")
    print("system_magnification_samples contribution", system_magnification_samples.mean() * np.sqrt(die.DIE_W_um**2 + die.DIE_L_um**2) * 1e3, " nm")

    if pad_yield_flag == True:
        glb_defect_pad_yield_min = 1.0
        glb_defect_pad_yield_max = 0.0
        # Sample the systematic misalignment for every pad based on the systematic translation, rotation, and magnification
        # Calculate the pad yield for each pad and return the pad yield map
        # When calculate the pad yield, we ignore the whether the pad is critical or not.
        nr = math.ceil(PAD_ARR_ROW / pad_yield_map_sub_factor)
        nc = math.ceil(PAD_ARR_COL / pad_yield_map_sub_factor)
        overlay_pad_yield_map_sub = np.zeros((nr, nc))
        for kr in range(nr):
            r = round(kr * (PAD_ARR_ROW - 1) / (nr - 1))
            for kc in range(nc):
                c = round(kc * (PAD_ARR_COL - 1) / (nc - 1))
                i = r * PAD_ARR_COL + c
                dx_array_samples_i = (system_translation_x_samples_um - system_rotation_samples_rad * die.pad_coords[i, 1] + system_magnification_samples * die.pad_coords[i, 0])
                dy_array_samples_i = (system_translation_y_samples_um + system_rotation_samples_rad * die.pad_coords[i, 0] + system_magnification_samples * die.pad_coords[i, 1])
                pad_misalignment_samples_i = np.sqrt(dx_array_samples_i**2 + dy_array_samples_i**2)
                upper_limit_i = MAX_ALLOWED_MISALIGNMENT - pad_misalignment_samples_i
                lower_limit_i = -MAX_ALLOWED_MISALIGNMENT - pad_misalignment_samples_i
                overlay_pad_yield_map_sub[kr, kc] = np.mean(
                                            norm.cdf(upper_limit_i, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um)  \
                                            - norm.cdf(lower_limit_i, loc=RANDOM_MISALIGNMENT_MEAN_um, scale=RANDOM_MISALIGNMENT_STD_um)
                                        )
        glb_defect_pad_yield_min = min(glb_defect_pad_yield_min, np.nanmin(overlay_pad_yield_map_sub))
        glb_defect_pad_yield_max = max(glb_defect_pad_yield_max, np.nanmax(overlay_pad_yield_map_sub))
        die.glb_pad_yield_min_max_dict['Y_ovl'] = (glb_defect_pad_yield_min, glb_defect_pad_yield_max)
        # Draw the pad yield map
        plt.figure(figsize=(8, 6))
        plt.imshow(
            overlay_pad_yield_map_sub, 
            cmap='viridis', 
            vmin=die.glb_pad_yield_min_max_dict['Y_ovl'][0],
            vmax=die.glb_pad_yield_min_max_dict['Y_ovl'][1],
            interpolation='nearest',
            )
        plt.colorbar(label='Pad Overlay Yield (Subsampled)')
        plt.xlabel('Pad Column Index')
        plt.ylabel('Pad Row Index')
        plt.show()
        
    return overlay_pad_yield_map_sub