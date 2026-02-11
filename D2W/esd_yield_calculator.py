#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Author: Zhichao Chen
#### Date: Feb 7, 2026

"""
ESD yield calculator for the D2W hybrid bonding process.
"""

import numpy as np
from esd_hybrid import die_esd_yield_calculation







def stack_esd_yield_calculator(
    cfg_dict: dict,
    die_stack,
    pad_bitmap_collection_dict: dict,
):
    for interface_name, cfg in cfg_dict.items():
        interface = die_stack.interfaces.interface_dict[interface_name]
        pad_bitmap_collection = pad_bitmap_collection_dict[interface_name]
        interface = die_stack.interfaces.interface_dict[interface_name]

        # Extract the parameters for the ESD yield calculation
        CRITICAL_PAD_MASK = pad_bitmap_collection['CRITICAL_PAD_BITMAP'].flatten()

        # The critical-pad ratio is the same for every die (independent of die_center),
        # so compute it once rather than allocating large arrays per die.
        num_total_pads = len(CRITICAL_PAD_MASK)
        num_critical_pads = int(np.count_nonzero(CRITICAL_PAD_MASK))

        p_lambda = 0.1 # FIXME: placeholder value for the lambda parameter in the ESD yield model, to be updated after discussion with Cain
        die_esd_yield = 1 - num_critical_pads / num_total_pads * p_lambda

        # TODO: ESD yield part needs to be updated after discussing with Cain
        die_esd_yield_list = [die_esd_yield] * len(interface.die_list)
        # Update the die yield list for this interface in the wafer stack
        die_stack.die_yield_list_per_interface_dict[interface_name]['ESD'] = die_esd_yield_list