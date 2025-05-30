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



def defect_yield_calculator(
    WAF_R,
    eff_DIE_R,
    D0,
    t_0,
    z,
    k_r,
    k_r0,
    k_n,
    k_S,
    k_L,
    PAD_TOP_R,
    PITCH,
    PAD_ARR_ROW,
    PAD_ARR_COL,
    PAD_ARR_W,
    PAD_ARR_L,
    VOID_SHAPE,
    dice_width,
    NUM_DIES_ON_WAF,
):
    L_m = 1.0
    # r_mv = sp.symbols("r_mv")
    def f_r_mv(r_mv, D0, k_r, k_r0, WAF_R, t_0, z):
        # Define critical radius value
        r_critical = (k_r * WAF_R + k_r0) * np.sqrt(t_0)
        if r_mv < k_r0 * np.sqrt(t_0):
            return 0
        if r_mv < r_critical:
            # Calculate f_r_mv for r < r_critical
            term1 = (D0 * (z - 1) * t_0**(z - 1)) / (k_r**2 * WAF_R**2)
            inner_term1 = (2 * r_mv) / (z * t_0 ** z) + (2 * k_r0**(2 * z)) / (z * (2 * z - 1) * r_mv**(2 * z - 1))
            inner_term2 = (2 * k_r0) / ((z - 1 / 2) * t_0**(z - 1 / 2))
            f_r_mv_value = term1 * (inner_term1 - inner_term2)
        
        else:
            # Calculate f_r_mv for r >= r_critical
            term1 = (2 * D0 * (z - 1) * t_0**(z - 1) * (k_r * WAF_R + k_r0)**(2*z-2)) / (r_mv**(2 * z - 1))
            out_term2 = 2 * D0 * (z - 1)**2 * t_0**(z-1) / (k_r**2 * WAF_R**2 * r_mv**(2 * z - 1))
            bracket_term2 = ((k_r * WAF_R + k_r0)**(2*z) - k_r0**(2*z)) / z - (2*k_r0*(k_r * WAF_R + k_r0)**(2*z-1)-2*k_r0**(2*z)) / (z - 1/2) + (k_r0**2 * (k_r*WAF_R+k_r0)**(2*z-2)-k_r0**(2*z)) / (z-1)
            f_r_mv_value = term1 - out_term2 * bracket_term2

        return f_r_mv_value
    # Distr_r_mv = f_r_mv(r_mv, D0, k_r, k_r0, WAF_R, t_0, z)
    D_r_vt = 2 * k_n * (z - 1) * np.sqrt(t_0) * L_m**3 * D0 / (3 * (z - 1.5) * eff_DIE_R**2)
    t_avg = (z - 1) * t_0 / (z - 2)
    n_m = np.round(k_n * L_m * t_avg**0.5)
    if n_m == 0:
        Distr_r_vt = 0
    elif n_m == 1:
        r_vt_min = np.sqrt(k_S / (2 * k_n *np.pi))
        r_vt_max = np.sqrt(k_S * L_m * t_avg**0.5 / np.pi)
        Distr_r_vt = D_r_vt / (r_vt_max - r_vt_min)     # r_vt_min <= r_vt <= r_vt_max
    elif n_m >= 2:
        if VOID_SHAPE == 'circle':
            r_vt_min = ((6 * k_S * L_m * t_avg**0.5) / (np.pi * (n_m + 2) * (n_m + 1) * n_m))**(1/2)
            r_vt_max = n_m * ((6 * k_S * L_m * t_avg**0.5) / (np.pi * (n_m + 2) * (n_m + 1) * n_m))**(1/2)
        elif VOID_SHAPE == 'square':
            r_vt_min = ((6 * k_S * L_m * t_avg**0.5) / (4 * (n_m + 2) * (n_m + 1) * n_m))**(1/2)
            r_vt_max = n_m * ((6 * k_S * L_m * t_avg**0.5) / (4 * (n_m + 2) * (n_m + 1) * n_m))**(1/2)
        Distr_r_vt = D_r_vt / (r_vt_max - r_vt_min)     # r_vt_min <= r_vt <= r_vt_max
        
    
    def void_critical_area_per_die(PAD_TOP_R, r_v, PITCH, PAD_ARR_ROW, PAD_ARR_COL, PAD_ARR_W, PAD_ARR_L, VOID_SHAPE):
        N = PAD_ARR_ROW * PAD_ARR_COL
        r_p = PAD_TOP_R
        a = PAD_ARR_ROW
        b = PAD_ARR_COL
        if VOID_SHAPE == 'circle':
            if 2 * (r_v + r_p) <= PITCH:
                return N * np.pi * (r_v + r_p)**2
            elif 2 * (r_v + r_p) > PITCH and 2 * (r_v + r_p) <= np.sqrt(2) * PITCH:
                theta = np.arccos(PITCH / (2 * (r_v + r_p)))
                return N * np.pi * (r_v + r_p)**2 \
                    - 2 * ((a-1)*b+(b-1)*a) * (theta - 0.5*np.sin(2*theta)) * (r_v + r_p)**2
            elif 2 * (r_v + r_p) > np.sqrt(2) * PITCH:
                theta = np.arccos(PITCH / (2 * (r_v + r_p)))
                return (a-1)*(b-1) * PITCH**2 + 2 * ((a-1)+(b-1)) * 0.5 * 0.5*np.sin(2*theta) * (r_v + r_p)**2 \
                + ((3*np.pi-4*theta)+((a-2)+(b-2))*(np.pi-2*theta)) * (r_v + r_p)**2
                # return PAD_ARR_W * PAD_ARR_L
        if VOID_SHAPE == 'square':
            if 2 * (r_v + r_p) <= PITCH:
                return 4 * N * (r_v + r_p)**2
            elif 2 * (r_v + r_p) > PITCH:
                return ((a-1) * PITCH + 2 * (r_v + r_p)) * ((b-1) * PITCH + 2 * (r_v + r_p))
            
    def integral_main_voids(r_mv):
        Distr_r_mv = f_r_mv(r_mv, D0, k_r, k_r0, eff_DIE_R, t_0, z)
        A_r_mv = void_critical_area_per_die(PAD_TOP_R, r_mv, PITCH, PAD_ARR_ROW, PAD_ARR_COL, PAD_ARR_W, PAD_ARR_L, VOID_SHAPE)
        return Distr_r_mv * A_r_mv

    def integral_tail_voids(r_vt):
        if n_m == 0:
            return 0
        else:
            Distr_r_vt = D_r_vt / (r_vt_max - r_vt_min)     # r_vt_min <= r_vt <= r_vt_max
        A_r_vt = void_critical_area_per_die(PAD_TOP_R, r_vt, PITCH, PAD_ARR_ROW, PAD_ARR_COL, PAD_ARR_W, PAD_ARR_L, VOID_SHAPE)
        return Distr_r_vt * A_r_vt
    
    def integral_size_main_voids(r_mv):
        Distr_r_mv = f_r_mv(r_mv, D0, k_r, k_r0, eff_DIE_R, t_0, z)

        return Distr_r_mv * r_mv


    avg_main_voids = quad(integral_main_voids, k_r0*t_0**0.5, np.inf)[0]
    if n_m == 0:
        avg_tail_voids = 0
    else:
        avg_tail_voids = quad(integral_tail_voids, r_vt_min, r_vt_max)[0]

    



    particle_defect_yield = np.exp(-avg_main_voids)


    return particle_defect_yield