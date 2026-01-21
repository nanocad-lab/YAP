#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Overall yield simulator for hybrid bonding
#### Author: Zhichao Chen
#### Date: Sep 26, 2024

import numpy as np
import time
import os

from YAP.W2W.wafer_die_stack_initialization import wafer_stack_list_initialize
from overlay_yield_simulator import overlay_term_simulator
from defect_yield_simulator import defect_yield_simulator
from roughness_parameters import roughness_parameters
from overall_yield_simulator import overall_yield_simulator


def Assembly_Yield_Simulator(
    cfg,
    pad_bitmap_collection,
):
    num_sim_epoch = cfg.NUM_WAFER_STACKS // cfg.SIM_BATCH_SIZE
    epoch_yield_list = []

    if cfg.verbose:
        print("Verbose mode enabled: Tracking failure reasons for each die.")
        # Initialize a temporary wafer stack to get die count and initialize fail maps/vectors
        temp_waf_stack_list = wafer_stack_list_initialize(
            NUM_WAFER_LAYERS          = cfg.NUM_WAFER_LAYERS,
            num_stack_samples      = 1,
        )
        fail_map_dict = {}
        fail_map_dict['overlay']    = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))
        fail_map_dict['particle']   = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))
        fail_map_dict['mechanical'] = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))
        fail_map_dict['ESD']        = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))
        fail_map_dict['overall']    = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))

        fail_vec_dict = {}
        num_dies_per_wafer = temp_waf_stack_list[0].num_dies
        fail_vec_dict['overlay']    = np.zeros(cfg.NUM_WAFER_STACKS * num_dies_per_wafer)
        fail_vec_dict['particle']   = np.zeros(cfg.NUM_WAFER_STACKS * num_dies_per_wafer)
        fail_vec_dict['mechanical'] = np.zeros(cfg.NUM_WAFER_STACKS * num_dies_per_wafer)
        fail_vec_dict['ESD']        = np.zeros(cfg.NUM_WAFER_STACKS * num_dies_per_wafer)
        fail_vec_dict['overall']    = np.zeros(cfg.NUM_WAFER_STACKS * num_dies_per_wafer)
        del temp_waf_stack_list
    
    # Iterate over simulation epochs
    for epoch in range(num_sim_epoch):
        # Record the time
        start_time = time.time()
        # Initialize the wafer
        waf_stack_list = wafer_stack_list_initialize(
            NUM_WAFER_LAYERS          = cfg.NUM_WAFER_LAYERS,
            num_stack_samples      = cfg.SIM_BATCH_SIZE,
        )
        # Generate overlay terms， in the shape of (NUM_STACKS, NUM_BONDING_INTERFACES)
        system_translation_x_um, \
        system_translation_y_um, \
        system_rotation_rad, \
        system_magnification_ppm, \
        MAX_ALLOWED_MISALIGNMENT_um = overlay_term_simulator(
            cfg                             =       cfg,
            waf_stack_list                  =       waf_stack_list,
            PAD_TOP_R_um                    =       cfg.PAD_TOP_R_um,
            PAD_BOT_R_um                    =       cfg.PAD_BOT_R_um,
            PITCH_r_um                      =       cfg.PITCH_r_um,
            PITCH_c_um                      =       cfg.PITCH_c_um,
            CONTACT_AREA_CONSTRAINT         =       cfg.CONTACT_AREA_CONSTRAINT,
            CRITICAL_DIST_CONSTRAINT        =       cfg.CRITICAL_DIST_CONSTRAINT,
            SYSTEM_ROTATION_MEAN_rad        =       cfg.SYSTEM_ROTATION_MEAN_rad,
            SYSTEM_ROTATION_STD_rad         =       cfg.SYSTEM_ROTATION_STD_rad,
            SYSTEM_TRANSLATION_X_MEAN_um    =       cfg.SYSTEM_TRANSLATION_X_MEAN_um,
            SYSTEM_TRANSLATION_X_STD_um     =       cfg.SYSTEM_TRANSLATION_X_STD_um,
            SYSTEM_TRANSLATION_Y_MEAN_um    =       cfg.SYSTEM_TRANSLATION_Y_MEAN_um,
            SYSTEM_TRANSLATION_Y_STD_um     =       cfg.SYSTEM_TRANSLATION_Y_STD_um,
            BOW_DIFFERENCE_MEAN_um          =       cfg.BOW_DIFFERENCE_MEAN_um,
            BOW_DIFFERENCE_STD_um           =       cfg.BOW_DIFFERENCE_STD_um,
            k_mag                           =       cfg.k_mag,
            M_0                             =       cfg.M_0,
        )
        
        # Generate void defects
        defect_yield_simulator(
            WAF_R_um            =       cfg.WAF_R_um,
            D0                  =       cfg.D0,
            t_0                 =       cfg.t_0,
            z                   =       cfg.z,
            k_r                 =       cfg.k_r,
            k_r0                =       cfg.k_r0,
            k_n                 =       cfg.k_n,
            k_L                 =       cfg.k_L,
            k_S                 =       cfg.k_S,
            VOID_SHAPE          =       cfg.VOID_SHAPE,
            waf_stack_list      =       waf_stack_list,
        )
        
        # Calculate the overall yield
        yield_list, epoch_fail_map_dict, epoch_fail_vec_dict = overall_yield_simulator(
            cfg                             =       cfg,
            waf_stack_list                  =       waf_stack_list,
            num_dies_per_wafer              =       num_dies_per_wafer,
            WAF_R_um                        =       cfg.WAF_R_um,
            system_translation_x_um         =       system_translation_x_um,
            system_translation_y_um         =       system_translation_y_um,
            system_rotation_rad             =       system_rotation_rad,
            system_magnification_ppm        =       system_magnification_ppm,
            MAX_ALLOWED_MISALIGNMENT_um     =       MAX_ALLOWED_MISALIGNMENT_um,
            PAD_ARR_W_um                    =       cfg.PAD_ARR_W_um,
            PAD_ARR_L_um                    =       cfg.PAD_ARR_L_um,
            PAD_ARR_ROW                     =       cfg.PAD_ARR_ROW,
            PAD_ARR_COL                     =       cfg.PAD_ARR_COL,
            TOP_DISH_MEAN_nm                =       cfg.TOP_DISH_MEAN_nm,
            TOP_DISH_STD_nm                 =       cfg.TOP_DISH_STD_nm,
            BOT_DISH_MEAN_nm                =       cfg.BOT_DISH_MEAN_nm,
            BOT_DISH_STD_nm                 =       cfg.BOT_DISH_STD_nm,
            TILT_X_MEAN_DEG                 =       cfg.TILT_X_MEAN_DEG,
            TILT_X_STD_DEG                  =       cfg.TILT_X_STD_DEG,
            TILT_Y_MEAN_DEG                 =       cfg.TILT_Y_MEAN_DEG,
            TILT_Y_STD_DEG                  =       cfg.TILT_Y_STD_DEG,
            k_et                            =       cfg.k_et,
            k_eb                            =       cfg.k_eb,
            T_R                             =       cfg.T_R,
            T_anl                           =       cfg.T_anl,
            PITCH_r_um                      =       cfg.PITCH_r_um,
            PITCH_c_um                      =       cfg.PITCH_c_um,
            PAD_TOP_R_um                    =       cfg.PAD_TOP_R_um,
            RANDOM_MISALIGNMENT_MEAN_um     =       cfg.RANDOM_MISALIGNMENT_MEAN_um,
            RANDOM_MISALIGNMENT_STD_um      =       cfg.RANDOM_MISALIGNMENT_STD_um,
            approximate_set                 =       cfg.approximate_set,
            pad_bitmap_collection           =       pad_bitmap_collection,
        )
        epoch_yield_list.append(yield_list)
        if cfg.verbose:
            fail_map_dict['overlay']    += epoch_fail_map_dict['overlay']
            fail_map_dict['particle']   += epoch_fail_map_dict['particle']
            fail_map_dict['mechanical'] += epoch_fail_map_dict['mechanical']
            fail_map_dict['ESD']        += epoch_fail_map_dict['ESD']
            fail_map_dict['overall']    += epoch_fail_map_dict['overall']
            fail_vec_dict['overlay'][epoch*cfg.SIM_BATCH_SIZE:(epoch+1)*cfg.SIM_BATCH_SIZE * num_dies_per_wafer] = epoch_fail_vec_dict['overlay']
            fail_vec_dict['particle'][epoch*cfg.SIM_BATCH_SIZE:(epoch+1)*cfg.SIM_BATCH_SIZE * num_dies_per_wafer] = epoch_fail_vec_dict['particle']
            fail_vec_dict['mechanical'][epoch*cfg.SIM_BATCH_SIZE:(epoch+1)*cfg.SIM_BATCH_SIZE * num_dies_per_wafer] = epoch_fail_vec_dict['mechanical']
            fail_vec_dict['ESD'][epoch*cfg.SIM_BATCH_SIZE:(epoch+1)*cfg.SIM_BATCH_SIZE * num_dies_per_wafer] = epoch_fail_vec_dict['ESD']
            fail_vec_dict['overall'][epoch*cfg.SIM_BATCH_SIZE:(epoch+1)*cfg.SIM_BATCH_SIZE * num_dies_per_wafer] = epoch_fail_vec_dict['overall']

        print(f"Simulation progress: {(epoch+1)*cfg.SIM_BATCH_SIZE}/{cfg.NUM_WAFER_STACKS} wafer stacks simulated. Epoch yield: {np.mean(yield_list):.4f}. Time taken: {time.time() - start_time:.2f} seconds.")
        
        
        del waf_list

    print("Simulation for all epochs completed.")
    assembly_yield = np.mean(epoch_yield_list)
    # Remove temporary files if any
    for name in os.listdir(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp'):
        file_path = os.path.join(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp', name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    if cfg.verbose:
        fail_map_dict['overlay']    /= (num_sim_epoch * cfg.SIM_BATCH_SIZE * num_dies_per_wafer)
        fail_map_dict['particle']   /= (num_sim_epoch * cfg.SIM_BATCH_SIZE * num_dies_per_wafer)
        fail_map_dict['mechanical'] /= (num_sim_epoch * cfg.SIM_BATCH_SIZE * num_dies_per_wafer)
        fail_map_dict['ESD']        /= (num_sim_epoch * cfg.SIM_BATCH_SIZE * num_dies_per_wafer)
        fail_map_dict['overall']    /= (num_sim_epoch * cfg.SIM_BATCH_SIZE * num_dies_per_wafer)
        # Report the failure reasons statistics
        print("{} die failures due to overlay misalignment.".format(int(np.sum(fail_vec_dict['overlay']))))
        print("{} die failures due to particle defects.".format(int(np.sum(fail_vec_dict['particle']))))
        print("{} die failures due to mechanical issues.".format(int(np.sum(fail_vec_dict['mechanical']))))
        print("{} die failures due to ESD issues.".format(int(np.sum(fail_vec_dict['ESD']))))
        print("{} die failures in total.".format(int(np.sum(fail_vec_dict['overall']))))
        # Save fail map dict
        np.savez(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_map_dict.npz', **fail_map_dict)
        print("Failure heat maps saved to {}.".format(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_map_dict.npz'))
        # Save fail vec dict
        np.savez(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_vec_dict.npz', **fail_vec_dict)
        print("Failure vectors for all die samples saved to {}.".format(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_vec_dict.npz'))

    return assembly_yield, epoch_yield_list
