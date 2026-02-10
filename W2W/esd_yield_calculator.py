#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Author: Zhichao Chen
#### Date: Feb 7, 2026

"""
ESD yield calculator for the W2W hybrid bonding process.
"""

import numpy as np
from esd_hybrid import die_esd_yield_calculation







def stack_esd_yield_calculator(
    cfg_dict: dict,
    waf_stack,
    pad_bitmap_collection_dict: dict,
):
    for interface_name, cfg in cfg_dict.items():
        interface = waf_stack.interfaces.interface_dict[interface_name]
        pad_bitmap_collection = interface.pad_bitmap_collection
        interface = waf_stack.interfaces.interface_dict[interface_name]

        # Extract the parameters for the ESD yield calculation
        CRITICAL_PAD_MASK = pad_bitmap_collection['CRITICAL_PAD_BITMAP'].flatten()
        
        # TODO: ESD yield part needs to be updated after discussing with Cain   
        for die_ind, die in enumerate(interface.die_list):
            die_center_x, die_center_y = die.die_center[0], die.die_center[1]
            # Assume dies in the center will be the first contact point and have higher ESD hazard
            die_pad_coords = interface.base_pad_coords + die.die_center
            valid_die_pad_coords = die_pad_coords[CRITICAL_PAD_MASK == 1]
            esd_valid_pad_yield_vec, _, _ = die_esd_yield_calculation(  # TODO: To be implemented by Cain
                cfg                   = cfg,
                pad_coords_um         = valid_die_pad_coords,
            )
        # Update the die yield list for this interface in the wafer stack
        waf_stack.die_yield_list_per_interface_dict[interface_name]['ESD'] = XXX