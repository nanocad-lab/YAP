#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Defect yield simulator for the yield model for D2W hybrid bonding
#### Author: Zhichao Chen
#### Date: Oct 4, 2024

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import fsolve
import sympy as sp
from scipy.integrate import quad
from scipy.stats import norm

class particle:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.dist_from_center = np.sqrt(x**2 + y**2)
        self.thickness = t


class single_void:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


class void_tail:
    def __init__(self, x, y, thickness, k_n, k_S, k_L, VOID_SHAPE, DIE_W, DIE_L):
        self.x = x
        self.y = y
        self.dist_from_center = np.sqrt(x**2 + y**2)
        self.L = k_L * self.dist_from_center * np.sqrt(thickness)  # void tail length
        self.n = np.round(
            k_n * self.dist_from_center * np.sqrt(thickness)
        )  # number of voids in the tail
        self.S = k_S * self.dist_from_center * np.sqrt(thickness)  # void tail area
        # self.S = k_S * self.dist_from_center * thickness  # void tail area
        self.voids = []
        if self.n > 0:
            x_incrt = self.L * x / self.dist_from_center / self.n
            y_incrt = self.L * y / self.dist_from_center / self.n
            if VOID_SHAPE == "circle":  # r_vt is the radius of the circular void
                r_vt1 = np.sqrt(
                    self.S / ((np.pi * (self.n + 2) * (self.n + 1) * self.n) / 6)
                )
            elif (VOID_SHAPE == "square"):  # r_vt is the half side length of the square void
                r_vt1 = np.sqrt(
                    self.S / ((4 * (self.n + 2) * (self.n + 1) * self.n) / 6)
                )
            # r_vt1 = np.sqrt(self.S / self.n)      # assume the voids are in square shape
            for i in range(int(self.n)):
                x_vt = x + x_incrt * (int(self.n) - i)
                y_vt = y + y_incrt * (int(self.n) - i)
                if np.abs(x_vt) < DIE_W / 2 and np.abs(y_vt) < DIE_L / 2:
                    self.voids.append(single_void(x_vt, y_vt, r_vt1 * (i + 1)))



def defect_yield_simulator(
    D0,
    t_0,
    z,
    k_r,
    k_r0,
    k_n,
    k_L,
    k_S,
    VOID_SHAPE,
    DIE_W,
    DIE_L,
    NUM_DIES,
    die_list,
):
    def cdf_particle_thickness(t):
        return 1 - (t_0 / t) ** (z - 1)

    def inverse_cdf_particle_thickness(u):
        return t_0 / (1 - u) ** (1 / (z - 1))

    def generate_particles(particle_thickness, DIE_W, DIE_L):
        particles = []
        for i in range(len(particle_thickness)):
            # x, y = np.random.uniform(-DIE_W / 2, DIE_W / 2), np.random.uniform(-DIE_L / 2, DIE_L / 2)
            x, y = np.random.uniform(-DIE_W / 2 * 2, DIE_W / 2 * 2), np.random.uniform(-DIE_L / 2 * 2, DIE_L / 2 * 2)
            particles.append(particle(x, y, particle_thickness[i]))
        return particles
    
    # Generate the main void and void tail based on the particles
    def generate_voids(particles, k_r, k_r0, k_n, k_S):
        voids = []
        main_voids = []
        tail_voids = []
        num_main_void = 0
        num_void_in_tail = 0
        for p in particles:
            # generate main void
            r_mv = (k_r * p.dist_from_center + k_r0) * np.sqrt(p.thickness)
            voids.append(single_void(p.x, p.y, r_mv))
            main_voids.append(single_void(p.x, p.y, r_mv))
            num_main_void += 1
            # generate void tail
            void_tail_obj = void_tail(p.x, p.y, p.thickness, k_n, k_S, k_L, VOID_SHAPE, DIE_W, DIE_L)
            voids += void_tail_obj.voids
            tail_voids += void_tail_obj.voids
            num_void_in_tail += void_tail_obj.n

        return voids, main_voids, tail_voids
    

    total_particles = (2 * DIE_W) * (2 * DIE_L) * D0 * NUM_DIES     # Put the particles on the 2*DIE_W * 2*DIE_L area
    particles_per_die = np.random.multinomial(
        total_particles, [1 / NUM_DIES] * NUM_DIES
    )
    for die_ind in range(NUM_DIES):
        num_particles = particles_per_die[die_ind]
        particle_thickness = np.zeros(num_particles)
        u = np.random.rand(num_particles)
        particle_thickness = inverse_cdf_particle_thickness(u)
        particles = generate_particles(particle_thickness, DIE_W, DIE_L)

        # Generate the main void and void tail based on the particles for each die
        voids, main_voids, tail_voids = generate_voids(particles, k_r, k_r0, k_n, k_S)
        # transform the voids struct to array
        voids_arr = np.zeros([len(voids), 3])
        for i, v in enumerate(voids):
            voids_arr[i] = [v.x, v.y, v.r]
        die_list[die_ind].voids = voids_arr
        die_list[die_ind].safe_voids_mask = np.ones(len(voids))