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
    input_args,
    cfg_skeleton,
    _3dbv_path: str,
    _3dbx_path: str,
    cfg_dict: dict,
    pad_bitmap_collection_dict,
): 
    NUM_WAFER_STACKS = cfg_skeleton.NUM_WAFER_STACKS
    SIM_BATCH_SIZE = cfg_skeleton.SIM_BATCH_SIZE
    num_sim_epoch = NUM_WAFER_STACKS // SIM_BATCH_SIZE
    
    epoch_yield_list = []

    if input_args['verbose']:
        print("Verbose mode enabled: Tracking failure reasons for each die.")
        # Initialize a temporary wafer stack to get die count and initialize fail maps/vectors
        temp_waf_stack_list = wafer_stack_list_initialize(
            cfg_dict                =       cfg_dict,
            _3dbv_path              =       _3dbv_path,
            _3dbx_path              =       _3dbx_path,
            num_stack_samples       =       1,
        )
        fail_map_per_interface_dict = {}
        fail_vec_per_interface_dict = {}
        for interface_name, waf_interface in temp_waf_stack_list[0].interfaces.interface_dict.items():
            cfg = cfg_dict[interface_name]
            fail_map_per_interface_dict[interface_name] = {}
            fail_map_per_interface_dict[interface_name]['overlay']    = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))
            fail_map_per_interface_dict[interface_name]['particle']   = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))
            fail_map_per_interface_dict[interface_name]['mechanical'] = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))
            fail_map_per_interface_dict[interface_name]['ESD']        = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))
            fail_map_per_interface_dict[interface_name]['overall']    = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL))

            fail_vec_per_interface_dict[interface_name] = {}
            num_dies_per_wafer = waf_interface.num_dies_per_wafer
            fail_vec_per_interface_dict[interface_name]['overlay']    = np.zeros((NUM_WAFER_STACKS, waf_interface.num_dies_per_wafer))
            fail_vec_per_interface_dict[interface_name]['particle']   = np.zeros((NUM_WAFER_STACKS, waf_interface.num_dies_per_wafer))
            fail_vec_per_interface_dict[interface_name]['mechanical'] = np.zeros((NUM_WAFER_STACKS, waf_interface.num_dies_per_wafer))
            fail_vec_per_interface_dict[interface_name]['ESD']        = np.zeros((NUM_WAFER_STACKS, waf_interface.num_dies_per_wafer))
            fail_vec_per_interface_dict[interface_name]['overall']    = np.zeros((NUM_WAFER_STACKS, waf_interface.num_dies_per_wafer))
            del temp_waf_stack_list
    
    # Iterate over simulation epochs
    for epoch in range(num_sim_epoch):
        # Record the time
        start_time = time.time()
        # Initialize the wafer stack
        waf_stack_list = wafer_stack_list_initialize(
            cfg_dict                =       cfg_dict,
            _3dbv_path              =       _3dbv_path,
            _3dbx_path              =       _3dbx_path,
            num_stack_samples       =       SIM_BATCH_SIZE,
        )
        num_dies_per_wafer = waf_stack_list[0].num_dies_per_wafer
        # Generate overlay terms， in the shape of (NUM_STACKS, )
        for interface, pad_bitmap_collection in pad_bitmap_collection_dict.items():
            cfg = cfg_dict[interface]
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
        
            # Generate void defects for each bonding interface
            defect_yield_simulator(
                cfg                 =       cfg,
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
        yield_list, \
        epoch_fail_map_per_interface_dict, \
        epoch_fail_vec_per_interface_dict = overall_yield_simulator(
            input_args                      =       input_args,
            cfg_dict                        =       cfg_dict,
            waf_stack_list                  =       waf_stack_list,
            num_dies_per_wafer              =       num_dies_per_wafer,
            pad_bitmap_collection_dict      =       pad_bitmap_collection_dict,
        )
        epoch_yield_list.append(yield_list)
        for interface_name, waf_interface in temp_waf_stack_list[0].interfaces.interface_dict.items():
            cfg = cfg_dict[interface_name]
            if input_args['verbose']:
                fail_map_per_interface_dict[interface_name]['overlay']    += epoch_fail_map_per_interface_dict[interface_name]['overlay']
                fail_map_per_interface_dict[interface_name]['particle']   += epoch_fail_map_per_interface_dict[interface_name]['particle']
                fail_map_per_interface_dict[interface_name]['mechanical'] += epoch_fail_map_per_interface_dict[interface_name]['mechanical']
                fail_map_per_interface_dict[interface_name]['ESD']        += epoch_fail_map_per_interface_dict[interface_name]['ESD']
                fail_map_per_interface_dict[interface_name]['overall']    += epoch_fail_map_per_interface_dict[interface_name]['overall']

                fail_vec_per_interface_dict[interface_name]['overlay'][epoch*SIM_BATCH_SIZE:(epoch+1)*SIM_BATCH_SIZE, :] = epoch_fail_vec_per_interface_dict[interface_name]['overlay']
                fail_vec_per_interface_dict[interface_name]['particle'][epoch*SIM_BATCH_SIZE:(epoch+1)*SIM_BATCH_SIZE, :] = epoch_fail_vec_per_interface_dict[interface_name]['particle']
                fail_vec_per_interface_dict[interface_name]['mechanical'][epoch*SIM_BATCH_SIZE:(epoch+1)*SIM_BATCH_SIZE, :] = epoch_fail_vec_per_interface_dict[interface_name]['mechanical']
                fail_vec_per_interface_dict[interface_name]['ESD'][epoch*SIM_BATCH_SIZE:(epoch+1)*SIM_BATCH_SIZE, :] = epoch_fail_vec_per_interface_dict[interface_name]['ESD']
                fail_vec_per_interface_dict[interface_name]['overall'][epoch*SIM_BATCH_SIZE:(epoch+1)*SIM_BATCH_SIZE, :] = epoch_fail_vec_per_interface_dict[interface_name]['overall']
        print(f"Simulation progress: {(epoch+1)*SIM_BATCH_SIZE}/{NUM_WAFER_STACKS} wafer stacks simulated. Epoch yield: {np.mean(yield_list):.4f}. Time taken: {time.time() - start_time:.2f} seconds.")
        
        
        del waf_list

    print("Simulation for all epochs completed.")
    assembly_yield = np.mean(epoch_yield_list)
    # Remove temporary files if any
    for name in os.listdir(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp'):
        file_path = os.path.join(cfg.OUTPUT_DIR + cfg.DESIGN + '/temp', name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    for interface_name, cfg in cfg_dict.items():
        if input_args['verbose']:
            fail_map_per_interface_dict[interface_name]['overlay']    /= (num_sim_epoch * SIM_BATCH_SIZE * num_dies_per_wafer)
            fail_map_per_interface_dict[interface_name]['particle']   /= (num_sim_epoch * SIM_BATCH_SIZE * num_dies_per_wafer)
            fail_map_per_interface_dict[interface_name]['mechanical'] /= (num_sim_epoch * SIM_BATCH_SIZE * num_dies_per_wafer)
            fail_map_per_interface_dict[interface_name]['ESD']        /= (num_sim_epoch * SIM_BATCH_SIZE * num_dies_per_wafer)
            fail_map_per_interface_dict[interface_name]['overall']    /= (num_sim_epoch * SIM_BATCH_SIZE * num_dies_per_wafer)
            # Report the failure reasons statistics
            print("{} die failures due to overlay misalignment.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['overlay']))))
            print("{} die failures due to particle defects.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['particle']))))
            print("{} die failures due to mechanical issues.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['mechanical']))))
            print("{} die failures due to ESD issues.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['ESD']))))
            print("{} die failures in total.".format(int(np.sum(fail_vec_per_interface_dict[interface_name]['overall']))))
            # Save fail map dict
            np.savez(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_map_dict.npz', **fail_map_per_interface_dict)
            print("Failure heat maps saved to {}.".format(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_map_dict.npz'))
            # Save fail vec dict
            np.savez(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_vec_dict.npz', **fail_vec_per_interface_dict)
            print("Failure vectors for all die samples saved to {}.".format(cfg.OUTPUT_DIR + cfg.DESIGN + '/assembly_fail_vec_dict.npz'))

    return assembly_yield, epoch_yield_list
