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
from scipy.integrate import quad, dblquad
from scipy.stats import norm




def defect_yield_calculator(
    WAF_R,
    D0,
    t_0,
    z,
    k_r,
    k_r0,
    k_n,
    L_m,
    k_S,
    k_L,
    PAD_TOP_R,
    PITCH,
    PAD_ARR_ROW,
    PAD_ARR_COL,
    PAD_ARR_W,
    PAD_ARR_L,
    VOID_SHAPE,
    num_die,
    dice_width,
):
    r_mv = sp.symbols("r_mv")
    l = sp.symbols("l")
    Distr_r_mv = (4 * D0 * (z - 1) * t_0**(z - 1)) / (WAF_R**2 * r_mv**(2*z - 1) * k_r**2) \
        * (
            ((k_r * WAF_R + k_r0) ** (2*z) - k_r0**(2*z)) / (2*z)
            - k_r0 * ((k_r * WAF_R + k_r0) ** (2*z-1) - k_r0**(2*z-1))/(2*z - 1)
        )
    D_r_vt = 2 * k_n * (z - 1) * np.sqrt(t_0) * L_m**3 * D0 / (3 * (z - 1.5) * WAF_R**2)
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
        Distr_r_mv = (4 * D0 * (z - 1) * t_0**(z - 1)) / (WAF_R**2 * r_mv**(2*z - 1) * k_r**2) \
            * (
                ((k_r * WAF_R + k_r0) ** (2*z) - k_r0**(2*z)) / (2*z)
                - k_r0 * ((k_r * WAF_R + k_r0) ** (2*z-1) - k_r0**(2*z-1))/(2*z - 1)
            )
        A_r_mv = void_critical_area_per_die(PAD_TOP_R, r_mv, PITCH, PAD_ARR_ROW, PAD_ARR_COL, PAD_ARR_W, PAD_ARR_L, VOID_SHAPE)
        return Distr_r_mv * A_r_mv

    def integral_tail_voids(r_vt):
        Distr_r_vt = D_r_vt / (r_vt_max - r_vt_min)     # r_vt_min <= r_vt <= r_vt_max
        A_r_vt = void_critical_area_per_die(PAD_TOP_R, r_vt, PITCH, PAD_ARR_ROW, PAD_ARR_COL, PAD_ARR_W, PAD_ARR_L, VOID_SHAPE)
        return Distr_r_vt * A_r_vt
    
    '''
    The distribution of the distance of the particle to the wafer center
    '''
    def f_L(L, WAF_R):
        return 2 * L / WAF_R ** 2
    
    '''
    The distribution of the particle thickness
    '''
    def f_t(t, D0, z, t_0):
        if t < t_0:
            return 0
        else:
            return D0 * (z - 1) * t_0**(z - 1) / t**z
    
    '''
    The distribution of the void tail length
    '''
    def f_l(l, D0, k_l, WAF_R, t_0, z):
        if l <= k_l * WAF_R * np.sqrt(t_0):
            f_l_value = 2 * D0 * (z - 1) / (z * k_l**2 * WAF_R**2 * t_0) * l
        else:
            f_l_value = 2 * D0 * (z - 1) * (k_l**2 * WAF_R**2 * t_0) ** (z - 1) / z / (l**(2 * z - 1))
        return f_l_value
        
    '''
    The distribution of the main void size
    '''
    def f_r_mv(r_mv, D0, k_r, k_r0, WAF_R, t_0, z):
        # Define critical radius value
        r_critical = (k_r * WAF_R + k_r0) * np.sqrt(t_0)
        if r_mv < k_r0 * np.sqrt(t_0):
            return 0
        if r_mv < r_critical:
            # Calculate f_r_mv for r < r_critical
            term1 = ((z - 1) * t_0**(z - 1)) / (k_r**2 * WAF_R**2)
            inner_term1 = (2 * r_mv) / (z * t_0 ** z) + (2 * k_r0**(2 * z)) / (z * (2 * z - 1) * r_mv**(2 * z - 1))
            inner_term2 = (2 * k_r0) / ((z - 1 / 2) * t_0**(z - 1 / 2))
            f_r_mv_value = term1 * (inner_term1 - inner_term2) * D0
        
        else:
            # Calculate f_r_mv for r >= r_critical
            term1 = (2 * (z - 1) * t_0**(z - 1) * (k_r * WAF_R + k_r0)**(2*z-2)) / (r_mv**(2 * z - 1))
            out_term2 = 2 * (z - 1)**2 * t_0**(z-1) / (k_r**2 * WAF_R**2 * r_mv**(2 * z - 1))
            bracket_term2 = ((k_r * WAF_R + k_r0)**(2*z) - k_r0**(2*z)) / z - (2*k_r0*(k_r * WAF_R + k_r0)**(2*z-1)-2*k_r0**(2*z)) / (z - 1/2) + (k_r0**2 * (k_r*WAF_R+k_r0)**(2*z-2)-k_r0**(2*z)) / (z-1)
            f_r_mv_value = term1 - out_term2 * bracket_term2 * D0

        return f_r_mv_value
    
    '''
    The critical area of the 'rectangular' defects
    - Using l and r_mv to calculate the area of the defects
    '''   
    def A_l_r_mv(l, r_mv, PAD_ARR_W, PAD_ARR_L):
        a = PAD_ARR_W
        b = PAD_ARR_L
        return a*b + 2*l*r_mv + 2*(r_mv**2) + (2/np.pi)*(a + b)*(l + 3*r_mv)
    
    '''
    The critical area of the 'rectangular' defects
    - Using L and t to calculate the area of the defects
    '''
    def A_L_t(L, t, PAD_ARR_W, PAD_ARR_L, k_l, k_r, k_r0):
        a = PAD_ARR_W
        b = PAD_ARR_L
        return a*b + (2/np.pi)*(a + b)*(k_l*L*t**0.5)
                # 2*k_l*L*(k_r*L+k_r0)*t + \
                # 2*(k_r*L+k_r0)**2*t + \
                # (2/np.pi)*(a + b)*((k_l+3*k_r)*L*t**0.5 + 3*k_r0*t**0.5)

    '''
    A(L, t) * f_L(L) * f_t(t)
    '''
    def integral_A_L_t_f_L_f_t(L, t, D0, k_l, k_r, k_r0, WAF_R, t_0, z, PAD_ARR_W, PAD_ARR_L):
        return f_L(L, WAF_R) * f_t(t, D0, z, t_0) \
                * A_L_t(L, t, PAD_ARR_W, PAD_ARR_L, k_l, k_r, k_r0)
    
    def integral_size_main_voids(r_mv):
        return f_r_mv(r_mv, D0, k_r, k_r0, WAF_R, t_0, z) * r_mv

    # Calculate the average number of void defects per die
    def avg_num_void_defects_per_die(k_r0, t_0, r_vt_min, r_vt_max):
        num_main_voids = quad(integral_main_voids, k_r0*t_0**0.5, np.inf)[0]
        num_tail_voids = quad(integral_tail_voids, r_vt_min, r_vt_max)[0]

        return num_main_voids, num_tail_voids
        
    # # Calculate the average size of the main voids
    def avg_size_void_defects(k_r0, t_0):
        avg_size_main_voids = quad(integral_size_main_voids, k_r0*t_0**0.5, np.inf)[0]
    
        return avg_size_main_voids
    
    L_avg = k_L * 2/3 * WAF_R * (z-1) / (z-1.5) * t_0**0.5
    # print("Avergae void length: ", L_avg)
    
    avg_main_voids, avg_tail_voids = avg_num_void_defects_per_die(k_r0, t_0, r_vt_min, r_vt_max)
    r_avg_mv = avg_size_void_defects(k_r0, t_0)
    # print("The average main void size is {}.".format(r_avg_mv))
    
    def E_1_v(v, D0, k_r, k_r0, k_L, WAF_R, t_0, z):
        out_term = D0 * (z - 1) * t_0**(z - 1) / ((k_r + k_L)**2 * WAF_R**2)
        inner_term1 = (2 * v**3) / (3 * z * t_0**z)
        inner_term2 = (2 * k_r0**(2 * z)) / (z * (2 * z - 1) * (3 - 2 * z) * v**(2 * z - 3))
        inner_term3 = (k_r0 * v**2) / ((z - 0.5) * t_0**(z - 0.5))
        E_1_v_value = out_term * (inner_term1 + inner_term2 - inner_term3) 
        return E_1_v_value
    
    def E_2_v(v, D0, k_r, k_r0, k_L, WAF_R, t_0, z):
        out_term1 = (2 * D0 * (z - 1) * t_0**(z - 1) * ((k_r + k_L) * WAF_R + k_r0)**(2 * z - 2)) / ((2 * z - 3) * v**(2 * z - 3))
        out_term2 = (2 * D0 * (z - 1)**2 * t_0**(z - 1)) / ((k_r + k_L)**2 * WAF_R**2 * (2 * z - 3) * v**(2 * z - 3))
        
        bracket_term2 = (((k_r + k_L) * WAF_R + k_r0)**(2 * z) - k_r0**(2 * z)) / z \
                        - (2 * k_r0 * ((k_r + k_L) * WAF_R + k_r0)**(2 * z - 1) - 2 * k_r0**(2 * z)) / (z - 0.5) \
                        + (k_r0**2 * ((k_r + k_L) * WAF_R + k_r0)**(2 * z - 2) - k_r0**(2 * z)) / (z - 1)
        
        E_2_v_value = out_term1 - out_term2 * bracket_term2
        return E_2_v_value
        
    
    def Lambda_v(D0, k_r, k_r0, k_L, WAF_R, t_0, z):
        v1 = k_r0 * t_0**0.5
        v2 = ((k_r + k_L) * WAF_R + k_r0) * t_0**0.5
        E_1_v_res = E_1_v(v2, D0, k_r, k_r0, k_L, WAF_R, t_0, z) - E_1_v(v1, D0, k_r, k_r0, k_L, WAF_R, t_0, z)
        E_2_v_res = E_2_v(v2, D0, k_r, k_r0, k_L, WAF_R, t_0, z) - 0
        Lambda_v_value = D0 * PAD_ARR_W * PAD_ARR_L + 2 / np.pi * (PAD_ARR_W + PAD_ARR_L) * (E_1_v_res + E_2_v_res)
        return Lambda_v_value
    
    '''
    Calculate the average number of defects per die using 'rectangular' defects
    '''
    def avg_rectangular_defects(PAD_ARR_W, PAD_ARR_L, k_l, k_r, k_r0, WAF_R, t_0, z):
        a = PAD_ARR_W
        b = PAD_ARR_L
        # First term: ab
        term1 = D0 * a * b

        # Second term:
        t2_factor = D0 * (z - 1) * t_0 / (z - 2)
        inner2 = (
            k_l * (k_r * WAF_R**2 + (4/3) * k_r0 * WAF_R) +
            (k_r**2 * WAF_R**2 + (8/3) * k_r * k_r0 * WAF_R + 2 * k_r0**2)
        )
        term2 = t2_factor * inner2

        # Third term:
        t3_factor = 8 * D0 * (z - 1) * t_0**0.5 / (np.pi * (2*z - 3))
        inner3 = (1/3) * (k_l + 3 * k_r) * WAF_R + (3/2) * k_r0
        term3 = t3_factor * (a + b) * inner3

        return term1 + term2 + term3
    

    scale_factor = num_die * PAD_ARR_W * PAD_ARR_L / (np.pi * WAF_R**2) \
        / ((PAD_ARR_W * PAD_ARR_L) / ((PAD_ARR_W + dice_width) * (PAD_ARR_L + dice_width)))
    # print(PAD_ARR_W, PAD_ARR_L, scale_factor)
    num_vtl_defects = avg_rectangular_defects(PAD_ARR_W, PAD_ARR_L, k_L, k_r, k_r0, WAF_R, t_0, z)
    num_vtl_defects =  scale_factor * num_vtl_defects
    # num_vtl_defects =  scale_factor * Lambda_v(D0, 2*k_r, 2*k_r0, k_L, WAF_R, t_0, z)
    print("The average number of defects per die is {}.".format(num_vtl_defects))
    particle_defect_yield = np.exp(-num_vtl_defects)
    # print("The particle defect yield is {}.".format(particle_defect_yield))

    return particle_defect_yield