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
        cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
    elif mode == "d2w_simulation" or mode == "d2w_modeling":
        cfg.PAD_ARR_L_um = (cfg.PAD_ARR_ROW - 1) * cfg.PITCH_r_um  # pad array length (um)
        cfg.PAD_ARR_W_um = (cfg.PAD_ARR_COL - 1) * cfg.PITCH_c_um  # pad array width (um)
        cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
        cfg.eff_DIE_R = float(np.sqrt((cfg.DIE_W_um / 2) ** 2 + (cfg.DIE_L_um / 2) ** 2))  # Effective die radius (um)
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes are 'w2w_simulation', 'w2w_modeling', 'd2w_simulation', and 'd2w_modeling'.")


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
    """
    Update configuration items.
    """
    if mode == "w2w_simulation" or mode == "w2w_modeling":
        # cfg.PAD_ARR_ROW = int(np.floor(float(cfg.DIE_L_um / cfg.PITCH_um)))  # number of pads in a row of pad array
        # cfg.PAD_ARR_COL = int(np.floor(float(cfg.DIE_W_um / cfg.PITCH_um)))  # number of pads in a column of pad array
        cfg.PAD_ARR_L_um = (cfg.PAD_ARR_ROW - 1) * cfg.PITCH_r_um  # pad array length (um)
        cfg.PAD_ARR_W_um = (cfg.PAD_ARR_COL - 1) * cfg.PITCH_c_um  # pad array width (um)
        cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
        cfg.S_INIT_A_M = 10e-6 * (cfg.WAF_R_um / 150000) ** 2
        cfg.S_INIT_B_M = 0.0
    elif mode == "d2w_simulation" or mode == "d2w_modeling":
        # cfg.PAD_ARR_ROW = int(np.floor(float(cfg.DIE_L_um / cfg.PITCH_um)))  # number of pads in a row of pad array
        # cfg.PAD_ARR_COL = int(np.floor(float(cfg.DIE_W_um / cfg.PITCH_um)))  # number of pads in a column of pad array
        cfg.PAD_ARR_L_um = (cfg.PAD_ARR_ROW - 1) * cfg.PITCH_r_um  # pad array length (um)
        cfg.PAD_ARR_W_um = (cfg.PAD_ARR_COL - 1) * cfg.PITCH_c_um  # pad array width (um)
        cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
        cfg.eff_DIE_R = float(np.sqrt((cfg.DIE_W_um / 2) ** 2 + (cfg.DIE_L_um / 2) ** 2))  # Effective die radius (um)
        cfg.S_INIT_A_M = 10e-6 * (cfg.eff_DIE_R / 150000) ** 2
        cfg.S_INIT_B_M = 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes are 'w2w_simulation', 'w2w_modeling', 'd2w_simulation', and 'd2w_modeling'.")



def draw_pad_bitmap(cfg, bitmap_collection):
    # Draw the critical and redundant pad bitmaps in one figure (critical light red, redundant light blue, dummy light gray)
    CRITICAL_PAD_BITMAP = bitmap_collection["CRITICAL_PAD_BITMAP"]
    REDUNDANT_PAD_BITMAP = bitmap_collection["REDUNDANT_PAD_BITMAP"]
    DUMMY_PAD_BITMAP = bitmap_collection["DUMMY_PAD_BITMAP"]
    ## Use legend to show the color
    PAD_BITMAP = np.zeros_like(CRITICAL_PAD_BITMAP, dtype=int)

    PAD_BITMAP[CRITICAL_PAD_BITMAP == 1] = 1  # red
    PAD_BITMAP[REDUNDANT_PAD_BITMAP == 1] = 2  # blue
    PAD_BITMAP[DUMMY_PAD_BITMAP == 1] = 3  # green
    # Remaining zeros are non-pad areas
    PAD_BITMAP[PAD_BITMAP == 0] = 4  # non-pad (light gray)

    plt.figure(figsize=(15, 15))
    cmap = ListedColormap([
        (1.0, 0.5, 0.5),    # 1 - critical (medium red)
        (0.4, 0.4, 0.9),    # 2 - redundant (medium blue)
        (0.0, 0.6, 0.0),    # 3 - dummy (medium green)
        (0.9, 0.9, 0.9),    # 4 - non-pad (light gray)
    ])
    red_patch = patches.Patch(color=(1.0, 0.5, 0.5), label='Critical Pads')
    blue_patch = patches.Patch(color=(0.4, 0.4, 0.9), label='Redundant Pads')
    green_patch = patches.Patch(color=(0.0, 0.6, 0.0), label='Dummy Pads')
    light_gray_patch = patches.Patch(color=(0.9, 0.9, 0.9), label='Non-Pad Areas')
    plt.legend(
        handles=[red_patch, blue_patch, green_patch, light_gray_patch],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.07),
        ncol=4,
        frameon=False
    )
    norm = BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5, 4.5], ncolors=cmap.N)
    plt.imshow(PAD_BITMAP, cmap=cmap, norm=norm)
    plt.title("Pad Block Bitmap")


    # Save the pad bitmaps
    plt.savefig(cfg.OUTPUT_DIR + cfg.DESIGN + "/" + cfg.DESIGN + "_pad_bitmap.png")
    # print("Pad bitmap collections info saved.")
    return



def sort_pads_bmap(input_path, output_path):
    """
    Read pad data from .bmap file, from top-left to right-bottom order 
    sorted by x ascending and y descending.
    - x is the 3rd column (index 2)
    - y is the 4th column (index 3)
    """

    pads = []
    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:  # 至少需要 x, y 两列
                continue
            try:
                x = float(parts[2])  # 第3列是x
                y = float(parts[3])  # 第4列是y
                pads.append((x, y, line.strip()))
            except ValueError:
                continue  # 跳过无法解析的行

    # Transform to numpy array for sorting
    data = np.array(pads, dtype=object)

    # x ascending, y descending
    idx = np.lexsort((data[:,0].astype(float), - data[:,1].astype(float)))
    sorted_data = data[idx]

    with open(output_path, 'w') as f:
        for _, _, line in sorted_data:
            f.write(line + '\n')

    # print(f"Sorted the order as from top-left to right-bottom and saved in {output_path}")


def criticality_generator(cfg, 
                          bump_data: list,
                          redundant_net_to_bumpids: dict,
                        ):
    '''
    Criticality file output format:
    <port> <esd_criticality> <mechanical_criticality>
    '''
    bump_criticality = list()
    bump_set = set()
    for bump in bump_data:
        port = bump['port']
        net = bump['net']
        if (bump['net'], port) in bump_set:
            continue
        if 'dummy' in net.lower():
            esd_criticality = 0.0
            mechanical_criticality = 0.0
        else:
            num_copies = len(redundant_net_to_bumpids[bump['net']])
            mechanical_criticality = 1.0 / num_copies
            esd_criticality = 1.0 / num_copies

        bump_criticality.append({
            "port": port,
            "esd_criticality": esd_criticality,
            "mechanical_criticality": mechanical_criticality
        })
        bump_set.add((bump['net'], port))
    with open(cfg.OUTPUT_DIR + cfg.DESIGN + "/" + cfg.DESIGN + "_criticality.txt", 'w') as f:
        for bump_crit in bump_criticality:
            f.write(f"{bump_crit['port']} {bump_crit['esd_criticality']:.6f} {bump_crit['mechanical_criticality']:.6f}\n")
    print("Criticality file saved in ", cfg.OUTPUT_DIR + cfg.DESIGN + "/" + cfg.DESIGN + "_criticality.txt")
    return


def risk_map_generator(cfg, 
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
        if np.isnan(pad_coords_x) or np.isnan(pad_coords_y):
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
    with open(cfg.OUTPUT_DIR + cfg.DESIGN + "/" + cfg.DESIGN + "_risk.map", 'w') as f:
        for pad_risk in risk_map:
            f.write(f"{pad_risk['pad_coords_x']} {pad_risk['pad_coords_y']} {pad_risk['esd_failure_probability']} {pad_risk['overlay_failure_probability']} {pad_risk['particle_failure_probability']} {pad_risk['mechanical_failure_probability']}\n")
    print("Risk map file saved in ", cfg.OUTPUT_DIR + cfg.DESIGN + "/" + cfg.DESIGN + "_risk.map")
    return




def convert_3dblox_to_pad_bitmap(cfg, 
                                 blox_bmap_path: str,
                                 pad_arrange_pattern: str):
    '''
    pad_arrange_pattern: 'checkerboard' for UCIe standard and HBM
    '''

    sort_pads_bmap(blox_bmap_path, blox_bmap_path)

    # Read the bump data from the .bmap file
    bump_data = []
    # Initialize the pad array boundaries
    [pad_array_left, pad_array_right, pad_array_top, pad_array_bottom] = [float('inf'), float('-inf'), float('-inf'), float('inf')]
    with open(blox_bmap_path, 'r') as f:
        bumpid = 0
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                instance, bump_type, x, y, port, net = parts
                bump_data.append({      # From the top-left corner to the bottom-right corner
                    "bumpid": bumpid,
                    "x": float(x),
                    "y": float(y),
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
                bumpid += 1
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
    REDUNDANT_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    DUMMY_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    ESD_CRITICAL_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    pad_coords = np.full((cfg.PAD_ARR_ROW * cfg.PAD_ARR_COL, 2), np.nan, dtype=np.float32)  # x, y coordinates of each bump
    # Build a mapping array from physical bump location (r, c) to bump id
    mapping_physical_to_bumpid = np.full((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), np.nan, dtype=np.float32) # Shape: (PAD_ARR_ROW, PAD_ARR_COL)

    if pad_arrange_pattern == 'checkerboard': # This case is for UCIe standard
        for bump in bump_data:
            x = bump['x']
            y = bump['y']
            row = int(round((pad_array_top - y ) / (cfg.PITCH_r_um)))   # Because in checkerboard pattern, the pitch per row is halved
            col = int(round((x - pad_array_left) / (cfg.PITCH_c_um)))   # Because in checkerboard pattern, the pitch per column is halved
            mapping_physical_to_bumpid[row, col] = bump['bumpid']
            pad_coords[row * cfg.PAD_ARR_COL + col, 0] = bump['x'] - (pad_array_left + pad_array_right) / 2
            pad_coords[row * cfg.PAD_ARR_COL + col, 1] = bump['y'] - (pad_array_top + pad_array_bottom) / 2
            current_bump_net = bump['net']
            current_bump_port = bump['port']
            num_copies = len(redundant_net_to_bumpids[current_bump_net])
            if 'dummy' in current_bump_net.lower():
                DUMMY_PAD_BITMAP[row, col] = 1
                continue
            if num_copies == 1:
                CRITICAL_PAD_BITMAP[row, col] = 1
                continue
            elif num_copies > 1 and ('vss' in current_bump_port.lower() or 'vcc' in current_bump_port.lower()): 
                REDUNDANT_PAD_BITMAP[row, col] = 1
                continue
            elif num_copies > 1 and ('vss' not in current_bump_port.lower() and 'vcc' not in current_bump_port.lower()):
                REDUNDANT_PAD_BITMAP[row, col] = 1
                ESD_CRITICAL_PAD_BITMAP[row, col] = 1       # TODO: Check this with Alex. If redundant pads are connected to the same transistor gate, then this is correct.
                continue
    else:
        raise NotImplementedError("Currently only support checkerboard pad arrangement pattern.")
    # Count the number of pads
    num_critical_pads = np.sum(CRITICAL_PAD_BITMAP)
    num_redundant_pads = np.sum(REDUNDANT_PAD_BITMAP)
    num_dummy_pads = 0 if DUMMY_PAD_BITMAP is None else np.sum(DUMMY_PAD_BITMAP)

    bitmap_collection = {}
    bitmap_collection["bump_data"] = bump_data
    bitmap_collection["CRITICAL_PAD_BITMAP"] = CRITICAL_PAD_BITMAP
    bitmap_collection["REDUNDANT_PAD_BITMAP"] = REDUNDANT_PAD_BITMAP
    bitmap_collection["DUMMY_PAD_BITMAP"] = DUMMY_PAD_BITMAP
    bitmap_collection["ESD_CRITICAL_PAD_BITMAP"] = ESD_CRITICAL_PAD_BITMAP
    bitmap_collection["is_redundant_copy_same_block"] = False
    bitmap_collection["num_critical_pads"] = num_critical_pads
    bitmap_collection["num_redundant_pads"] = num_redundant_pads
    bitmap_collection["num_dummy_pads"] = num_dummy_pads
    bitmap_collection["redundant_net_to_bumpids"] = redundant_net_to_bumpids
    bitmap_collection["pad_coords"] = pad_coords
    bitmap_collection["mapping_physical_to_bumpid"] = mapping_physical_to_bumpid
    
    # Save the bitmap collection as npy file and mat file
    np.save(cfg.OUTPUT_DIR + cfg.DESIGN + '/' + cfg.DESIGN + "_bitmap_collection.npy", bitmap_collection)
    # sio.savemat(cfg.OUTPUT_DIR + "bitmap_collection.mat", bitmap_collection)

    # # Draw the critical and redundant pad bitmaps in one figure (critical light red, redundant light blue, dummy light gray)
    # draw_pad_bitmap(cfg, bitmap_collection)

    return bitmap_collection




