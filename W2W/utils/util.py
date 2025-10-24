#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.io as sio


def load_modeling_config(path, mode, debug=False):
    full_cfg = OmegaConf.load(path)
    cfg = full_cfg[mode]

    if mode == "w2w_simulation" or mode == "w2w_modeling":
        cfg.PAD_ARR_L_um = (cfg.PAD_ARR_ROW - 1) * cfg.PITCH_r_um  # pad array length (um)
        cfg.PAD_ARR_W_um = (cfg.PAD_ARR_COL - 1) * cfg.PITCH_c_um  # pad array width (um)
        cfg.PAD_BOT_R_um = min(cfg.PITCH_r_um, cfg.PITCH_c_um) / 2 * cfg.PAD_BOT_R_um_ratio if cfg.PAD_BOT_R_um is None else cfg.PAD_BOT_R_um # bottom Cu pad radius (um)
        cfg.PAD_TOP_R_um = cfg.PAD_BOT_R_um * cfg.PAD_TOP_R_um_ratio if cfg.PAD_TOP_R_um is None else cfg.PAD_TOP_R_um  # top Cu pad radius (um)
        cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
        # TODO: You need to set pad_block_dim_x and pad_block_dim_y for pitch per row and pitch per column
        # Usually we set pad_block as a square block
        assert int(cfg.pad_block_dim_x / cfg.PITCH_c_um) == int(cfg.pad_block_dim_y / cfg.PITCH_r_um), \
                "Currently only square pad blocks are supported. Please set pad_block_dim_x/pitch_c_um equal to pad_block_dim_y/pitch_r_um."
        cfg.pad_block_size = int(cfg.pad_block_dim_x / cfg.PITCH_c_um)  # pad block size (#rows or #columns of the pad block)
    else:
        # TODO: Implement D2W modeling & simulation configuration
        raise NotImplementedError("D2W modeling is not implemented yet.")


    if debug:
        cfg.DEBUG = True
        print("Configuration loaded:")
        print(OmegaConf.to_yaml(cfg))

    return cfg


def add_config_items(cfg, keys, values):
    """
    Add items to the configuration dictionary.
    
    Args:
        cfg (dict): Configuration dictionary.
        keys (list): List of keys to add.
        values (list): List of values corresponding to the keys.
    """
    if len(keys) != len(values):
        raise ValueError("Keys and values must have the same length.")
    
    for key, value in zip(keys, values):
        cfg[key] = value

def update_config_items(cfg, mode):
    if mode == "w2w_simulation" or mode == "w2w_modeling":
        # cfg.PAD_ARR_ROW = int(np.floor(float(cfg.DIE_L_um / cfg.PITCH_r_um)))  # number of pads in a row of pad array
        # cfg.PAD_ARR_COL = int(np.floor(float(cfg.DIE_W_um / cfg.PITCH_c_um)))  # number of pads in a column of pad array
        cfg.PAD_ARR_L_um = (cfg.PAD_ARR_ROW - 1) * cfg.PITCH_r_um  # pad array length (um)
        cfg.PAD_ARR_W_um = (cfg.PAD_ARR_COL - 1) * cfg.PITCH_c_um  # pad array width (um)
        cfg.PAD_BOT_R_um = min(cfg.PITCH_r_um, cfg.PITCH_c_um) / 2 * cfg.PAD_BOT_R_um_ratio if cfg.PAD_BOT_R_um is None else cfg.PAD_BOT_R_um # bottom Cu pad radius (um)
        cfg.PAD_TOP_R_um = cfg.PAD_BOT_R_um * cfg.PAD_TOP_R_um_ratio if cfg.PAD_TOP_R_um is None else cfg.PAD_TOP_R_um  # top Cu pad radius (um)
        cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
        # TODO: You need to set pad_block_dim_x and pad_block_dim_y for pitch per row and pitch per column
        # Usually we set pad_block as a square block
        assert int(cfg.pad_block_dim_x / cfg.PITCH_c_um) == int(cfg.pad_block_dim_y / cfg.PITCH_r_um), \
                "Currently only square pad blocks are supported. Please set pad_block_dim_x/pitch_c_um equal to pad_block_dim_y/pitch_r_um."
        cfg.pad_block_size = int(cfg.pad_block_dim_x / cfg.PITCH_c_um)  # pad block size (#rows or #columns of the pad block)


def downsample_bitmap(bitmap, block_size):
    """
    Downsamples a binary bitmap by taking max value in each block.
    For binary image, this is equivalent to OR pooling.
    """
    if bitmap.ndim == 2:
        h, w = bitmap.shape
        h_new = h // block_size
        w_new = w // block_size

        # Trim to divisible shape
        bitmap = bitmap[:h_new * block_size, :w_new * block_size]

        # Reshape and apply max pooling
        bitmap_down = bitmap.reshape(h_new, block_size, w_new, block_size)
        bitmap_down = bitmap_down.max(axis=(1, 3))
    elif bitmap.ndim == 3:
        n, h, w = bitmap.shape
        h_new = h // block_size
        w_new = w // block_size

        # Trim to divisible shape
        bitmap = bitmap[:, :h_new * block_size, :w_new * block_size]

        # Reshape and apply max pooling
        bitmap_down = bitmap.reshape(n, h_new, block_size, w_new, block_size)
        bitmap_down = bitmap_down.max(axis=(2, 4))
    else:
        raise ValueError("Bitmap must be 2D or 3D.")

    return bitmap_down


def draw_pad_bitmap(cfg, bitmap_collection):
    # Draw the critical and redundant pad bitmaps in one figure (critical light red, redundant light blue, dummy light gray)
    CRITICAL_PAD_BITMAP = bitmap_collection["CRITICAL_PAD_BITMAP"]
    REDUNDANT_PAD_BITMAP = bitmap_collection["REDUNDANT_PAD_BITMAP"]
    DUMMY_PAD_BITMAP = bitmap_collection["DUMMY_PAD_BITMAP"]
    ## Use legend to show the color
    PAD_BITMAP = np.zeros_like(CRITICAL_PAD_BITMAP, dtype=int)

    PAD_BITMAP[CRITICAL_PAD_BITMAP == 1] = 1  # red
    PAD_BITMAP[REDUNDANT_PAD_BITMAP == 1] = 2  # blue
    PAD_BITMAP[DUMMY_PAD_BITMAP == 1] = 3  # gray

    plt.figure(figsize=(6, 6))
    cmap = ListedColormap([
        (1.0, 0.5, 0.5),    # 1 - critical (medium red)
        (0.4, 0.4, 0.9),    # 2 - redundant (medium blue)
        (0.8, 0.8, 0.8),    # 3 - dummy (light gray)
    ])
    red_patch = patches.Patch(color=(1.0, 0.7, 0.7), label='Critical Pads')
    blue_patch = patches.Patch(color=(0.7, 0.7, 1.0), label='Redundant Pads')
    gray_patch = patches.Patch(color=(0.8, 0.8, 0.8), label='Dummy Pads')
    plt.legend(
        handles=[red_patch, blue_patch, gray_patch],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.07),
        ncol=3,
        frameon=False
    )
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    plt.imshow(PAD_BITMAP, cmap=cmap, norm=norm)
    plt.title("Pad Block Bitmap")


    # Save the pad bitmaps
    plt.savefig(cfg.OUTPUT_DIR + "pad_bitmap.png")
    print("Pad bitmap collections info saved.")

    return


def criticality_generator(cfg, 
                          bump_data: list,
                          redundant_net_to_bumpids: dict,
                        ):
    '''
    Criticality file output format:
    <port> <esd_criticality> <mechanical_criticality>
    '''
    bump_criticality = list()
    for bump in bump_data:
        port = bump['port']
        num_copies = len(redundant_net_to_bumpids[bump['net']])
        mechanical_criticality = 1.0 / num_copies
        if num_copies == 1:
            esd_criticality = 1.0
        elif num_copies > 1 and ('vss' in port.lower() or 'vcc' in port.lower()):
            esd_criticality = 1.0 / num_copies
        elif num_copies > 1 and ('vss' not in port.lower() and 'vcc' not in port.lower()):
            esd_criticality = 1.0

        bump_criticality.append({
            "port": port,
            "esd_criticality": esd_criticality,
            "mechanical_criticality": mechanical_criticality
        })
    with open(cfg.OUTPUT_DIR + "UCIe_standard_criticality.txt", 'w') as f:
        for bump_crit in bump_criticality:
            f.write(f"{bump_crit['port']} {bump_crit['esd_criticality']} {bump_crit['mechanical_criticality']}\n")
    print("UCIe standard criticality file saved in ", cfg.OUTPUT_DIR + "UCIe_standard_criticality.txt")
    return


def risk_map_generator(cfg, 
                          die_id: int,
                          die: object,
                        ):
    '''
    Risk map output format:
    <pad_coords_x> <pad_coords_y> <esd_failure_probability> <overlay_failure_probability> <particle_failure_probability> <mechanical_failure_probability>
    '''
    risk_map = list()
    for pad_id in range(len(die.pad_coords)):
        pad_coords_x = die.pad_coords[pad_id, 0]
        pad_coords_y = die.pad_coords[pad_id, 1]
        if pad_coords_x == np.nan or pad_coords_y == np.nan:
            continue
        pad_ovl_yield = die.pad_yield_map['Y_ovl'].flatten()[pad_id]
        pad_df_yield = die.pad_yield_map['Y_df'].flatten()[pad_id]
        pad_ce_yield = die.pad_yield_map['Y_ce'].flatten()[pad_id]
        pad_esd_yield = die.pad_yield_map['Y_esd'].flatten()[pad_id]
        risk_map.append({
            "pad_coords_x": pad_coords_x,
            "pad_coords_y": pad_coords_y,
            "esd_failure_probability": 1 - pad_esd_yield,
            "overlay_failure_probability": 1 - pad_ovl_yield,
            "particle_failure_probability": 1 - pad_df_yield,
            "mechanical_failure_probability": 1 - pad_ce_yield,
        })
    with open(cfg.OUTPUT_DIR + "UCIe_standard_die_{}_risk_map.map".format(die_id), 'w') as f:
        for pad_risk in risk_map:
            f.write(f"{pad_risk['pad_coords_x']} {pad_risk['pad_coords_y']} {pad_risk['esd_failure_probability']} {pad_risk['overlay_failure_probability']} {pad_risk['particle_failure_probability']} {pad_risk['mechanical_failure_probability']}\n")
    print("UCIe standard die {} risk map file saved in ".format(die_id), cfg.OUTPUT_DIR + "UCIe_standard_die_{}_risk_map.map".format(die.die_id))
    return

def convert_3dblox_to_pad_bitmap(cfg, 
                                 blox_bmap_path: str, 
                                 pad_arrange_pattern='checkerboard'):
    '''
    This module converts the 3DBlox .bmap file to pad bitmap for YAP to process.
        - pad_arrange_pattern: 'checkerboard' for UCIe standard
    '''
    # Extract configuration parameters
    pad_block_size = cfg.pad_block_size     # In UCIe standard, pad block size is 1 (no downsampling)
    critical_pad_ratio = cfg.critical_pad_ratio
    redundant_pad_ratio = cfg.redundant_pad_ratio
    redundant_logical_pad_copy = cfg.redundant_logical_pad_copy
    redundant_logical_pad_dist = cfg.redundant_logical_pad_dist

    # Read the bump data from the .bmap file
    bump_data = []
    # Initialize the pad array boundaries
    [pad_array_left, pad_array_right, pad_array_top, pad_array_bottom] = [float('inf'), float('-inf'), float('-inf'), float('inf')]
    with open(blox_bmap_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                instance, bump_type, x, y, port, net = parts
                bumpid = int(instance.split("_")[1])
                bump_data.append({      # From the top-left corner to the bottom-right corner
                    "bumpid": bumpid,
                    "x": int(x),
                    "y": int(y),
                    "port": port,
                    "net": net
                })
                if float(x) < pad_array_left:
                    pad_array_left = float(x)
                if float(x) > pad_array_right:
                    pad_array_right = float(x)
                if float(y) < pad_array_bottom:
                    pad_array_bottom = float(y)
                if float(y) > pad_array_top:
                    pad_array_top = float(y)
    # Convert the bump data to pad bitmap
    redundant_net_to_bumpids = dict()
    for bump in bump_data:
        if bump['net'] not in redundant_net_to_bumpids:
            redundant_net_to_bumpids[bump['net']] = set()
        redundant_net_to_bumpids[bump['net']].add(bump['bumpid'])
    
    # Generate the criticality map
    criticality_generator(cfg, bump_data, redundant_net_to_bumpids)

    # Initialize the pad bitmap
    # TODO: You need to modify the simulator to support different pad arrangement patterns
    CRITICAL_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    CRITICAL_PAD_BLOCK_BITMAP = downsample_bitmap(CRITICAL_PAD_BITMAP, pad_block_size)
    REDUNDANT_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    REDUNDANT_PAD_BLOCK_BITMAP = downsample_bitmap(REDUNDANT_PAD_BITMAP, pad_block_size)
    pad_coords = np.full((cfg.PAD_ARR_ROW * cfg.PAD_ARR_COL, 2), np.nan, dtype=np.float32)  # x, y coordinates of each bump

    if pad_arrange_pattern == 'checkerboard': # This case is for UCIe standard
        for row in range(cfg.PAD_ARR_ROW):
            for col in range(cfg.PAD_ARR_COL):
                if (row + col) % 2 == 1:    # There is a bump (but not necessarily a critical pad)
                    bump_id = int(row * cfg.PAD_ARR_COL / 2 + col // 2)
                    current_bump_dict = bump_data[bump_id]
                    current_bump_net = current_bump_dict['net']
                    pad_coords[row * cfg.PAD_ARR_COL + col, 0] = current_bump_dict['x'] - (pad_array_left + pad_array_right) / 2
                    pad_coords[row * cfg.PAD_ARR_COL + col, 1] = current_bump_dict['y'] - (pad_array_top + pad_array_bottom) / 2
                    if len(redundant_net_to_bumpids[current_bump_net]) == 1:
                        CRITICAL_PAD_BITMAP[row, col] = 1
                    else:
                        REDUNDANT_PAD_BITMAP[row, col] = 1
                else:
                    continue
    else:
        raise NotImplementedError("Currently only support checkerboard pad arrangement pattern.")
    DUMMY_PAD_BITMAP = ~(CRITICAL_PAD_BITMAP | REDUNDANT_PAD_BITMAP)
    # Count the number of pads
    num_critical_pads = np.sum(CRITICAL_PAD_BITMAP)
    num_redundant_pads = np.sum(REDUNDANT_PAD_BITMAP)
    num_dummy_pads = 0 if DUMMY_PAD_BITMAP is None else np.sum(DUMMY_PAD_BITMAP)
    
    # Count the number of logical pads in redundant pads & Initialize the redundant net alive count dict
    print("redundant_net_to_bumpids:", redundant_net_to_bumpids)


    bitmap_collection = {}
    bitmap_collection["bump_data"] = bump_data
    bitmap_collection["CRITICAL_PAD_BITMAP"] = CRITICAL_PAD_BITMAP
    bitmap_collection["CRITICAL_PAD_BLOCK_BITMAP"] = CRITICAL_PAD_BLOCK_BITMAP
    bitmap_collection["REDUNDANT_PAD_BITMAP"] = REDUNDANT_PAD_BITMAP
    bitmap_collection["REDUNDANT_PAD_BLOCK_BITMAP"] = REDUNDANT_PAD_BLOCK_BITMAP
    bitmap_collection["DUMMY_PAD_BITMAP"] = DUMMY_PAD_BITMAP
    bitmap_collection["is_redundant_copy_same_block"] = False
    bitmap_collection["num_critical_pads"] = num_critical_pads
    bitmap_collection["num_redundant_pads"] = num_redundant_pads
    bitmap_collection["num_dummy_pads"] = num_dummy_pads

    bitmap_collection["critical_pad_ratio"] = critical_pad_ratio
    bitmap_collection["redundant_pad_ratio"] = redundant_pad_ratio
    bitmap_collection["redundant_logical_pad_copy"] = redundant_logical_pad_copy
    bitmap_collection["redundant_logical_pad_dist"] = redundant_logical_pad_dist
    bitmap_collection["pad_block_size"] = pad_block_size
    bitmap_collection["redundant_net_to_bumpids"] = redundant_net_to_bumpids
    bitmap_collection["pad_coords"] = pad_coords
    
    
    # Save the bitmap collection as npy file and mat file
    np.save(cfg.OUTPUT_DIR + "bitmap_collection.npy", bitmap_collection)
    # sio.savemat(cfg.OUTPUT_DIR + "bitmap_collection.mat", bitmap_collection)

    # # Draw the critical and redundant pad bitmaps in one figure (critical light red, redundant light blue, dummy light gray)
    draw_pad_bitmap(cfg, bitmap_collection)

    return bitmap_collection