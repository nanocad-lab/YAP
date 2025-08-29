#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Roughness parameters preparation for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: NOv 5, 2024

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar


def theta_func(R, sigma, adhesion_w, young_modulus_E):
    # Only works when you assume two wafers have the same surface roughness profile
    theta = young_modulus_E / adhesion_w * np.sqrt(sigma ** 3 / R)  # dimensionless parameter
    return theta


def integrand_A(x, s_star):
    return (x - s_star) * np.exp(-x**2 / 2)

def A_star(s_star, constant):
    A_star_integral, _ = quad(integrand_A, s_star, np.inf, args=(s_star,))
    # A_value = np.pi * R * sigma * eta_s * A_star_integral / np.sqrt(2 * np.pi)
    A_value = np.pi * constant * A_star_integral / np.sqrt(2 * np.pi)
    return A_value


# 定义第二个积分的 integrand 函数
def integrand_P1(x, s_star):
    return (x - s_star)**(3/2) * np.exp(-x**2 / 2)

def integrand_P2(x):
    return np.exp(-x**2 / 2)

def P_star(s_star, constant, theta):
    P_star_integral1, _ = quad(integrand_P1, s_star, np.inf, args=(s_star,))
    P_star_integral2, _ = quad(integrand_P2, s_star, np.inf)
    # Calculate P* value
    # P_value = R * sigma * eta_s * ((4 * theta) / (3 * np.sqrt(2 * np.pi)) * P_star_integral1 - np.sqrt(2 * np.pi) * P_star_integral2)
    P_value = constant * ((4 * theta) / (3 * np.sqrt(2 * np.pi)) * P_star_integral1 - np.sqrt(2 * np.pi) * P_star_integral2)
    return P_value


def roughness_parameters(
    Asperity_R,
    Roughness_sigma,
    eta_s,
    Roughness_constant,
    Adhesion_energy,
    Young_modulus,
    Dielectric_thickness,
    PITCH,
    PAD_BOT_R,
    DISH_0,
    k_peel,
):
    Roughness_sigma_renorm = Roughness_sigma * np.sqrt(2)
    Young_modulus_renorm = Young_modulus * 0.5
    # Calculate theta
    theta = theta_func(R=Asperity_R,
                          sigma=Roughness_sigma_renorm,
                          adhesion_w=Adhesion_energy,
                          young_modulus_E=Young_modulus_renorm)
    constant = Roughness_constant
    # Calculate s_star_b: assume s_star_b is between -10 and 10
    s_star_b = root_scalar(lambda s_star: P_star(s_star, constant=constant, theta=theta), bracket=[-10, 10], method='brentq')
    # A_star_b: normalized effective contact area
    A_star_b = A_star(s_star_b.root, constant=constant)
    # print("The normalized effective contact area A_star_b is: ", A_star_b)

    # print("Young_modulus_renorm: ", Young_modulus_renorm)
    # print("Adhesion_energy: ", Adhesion_energy)
    # print("Dielectric_thickness: ", Dielectric_thickness)
    max_acceptable_stress = np.sqrt(2 * Young_modulus_renorm * Adhesion_energy / Dielectric_thickness)
    # print("The maximum acceptable stress is: ", max_acceptable_stress/1e6, "MPa")
    max_acceptable_stress = max_acceptable_stress * A_star_b
    # print("The effective maximum acceptable stress is: ", max_acceptable_stress/1e6, "MPa")

    # Calculate the Cu pattern density
    D_cu = np.pi * PAD_BOT_R ** 2 / PITCH ** 2
    # print("The Cu pattern density D_cu is: ", D_cu)

    # The equilibirum condition for dielectric layer delamination is max_acceptable_stress = peak_annealing_stress
    # peak_annealing_stress = k_peel * D_cu * (DISH_0 - zeta_1_)
    zeta_1_ = DISH_0 - max_acceptable_stress / (k_peel * D_cu)
    zeta_1_ = zeta_1_ * 1e9     # Convert to nm
    zeta_1_ = zeta_1_ * 2        # Convert to the sum of the top and bottom Cu pad expansion
    # print("The roughness parameter zeta_1_ is: ", zeta_1_)


    return zeta_1_