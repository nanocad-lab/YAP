#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.io as sio
import os

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

def _upsample_pad_yield_map(pad_yield_map: np.ndarray,
                            pad_map_shape,
                            pad_yield_map_sub_factor: int) -> np.ndarray:
    """
    Upsample a subsampled pad yield map back to the full pad-array shape.

    The sampling grid follows the same endpoint-preserving indexing used in the
    overlay/defect calculators, so we reconstruct the dense map with 1D linear
    interpolation along columns and then rows.
    """
    if pad_yield_map.shape == pad_map_shape or pad_yield_map_sub_factor <= 1:
        return pad_yield_map

    target_rows, target_cols = pad_map_shape
    src_rows, src_cols = pad_yield_map.shape

    row_coords = np.round(np.linspace(0, target_rows - 1, src_rows)).astype(np.float64)
    col_coords = np.round(np.linspace(0, target_cols - 1, src_cols)).astype(np.float64)

    # Guard against duplicate coordinates in very small arrays.
    col_coords, unique_col_idx = np.unique(col_coords, return_index=True)
    pad_yield_map = pad_yield_map[:, unique_col_idx]
    row_coords, unique_row_idx = np.unique(row_coords, return_index=True)
    pad_yield_map = pad_yield_map[unique_row_idx, :]

    full_col_coords = np.arange(target_cols, dtype=np.float64)
    full_row_coords = np.arange(target_rows, dtype=np.float64)

    if pad_yield_map.shape[1] == 1:
        col_upsampled = np.repeat(pad_yield_map, target_cols, axis=1)
    else:
        col_upsampled = np.vstack([
            np.interp(full_col_coords, col_coords, row_vals)
            for row_vals in pad_yield_map
        ])

    if col_upsampled.shape[0] == 1:
        return np.repeat(col_upsampled, target_rows, axis=0)

    full_pad_yield_map = np.vstack([
        np.interp(full_row_coords, row_coords, col_upsampled[:, col_ind])
        for col_ind in range(target_cols)
    ]).T

    return full_pad_yield_map

def get_config_dict(cfg_folder: str,
                    cfg_skeleton: str,
                    ds_name: str,
                    input_ds_dir: str,
                    _3dbv_path: str,
                    _3dbx_path: str,
                    mode: str,
                    debug=False) -> dict:
    """
    Load base configuration from a YAML file and update with .3dbv and .bmap design parameters.
    args:
        cfg_folder: folder path of the config files
        cfg_skeleton: base config yaml file
        ds_name: design name
        input_ds_dir: input design directory
        _3dbv_path: path to .3dbv file
        mode: mode to load from config (w2w_simulation, w2w_modeling, d2w_simulation, d2w_modeling)
        debug: whether to enable debug output
    returns:
        cfg_dict: dictionary of configuration objects for each stack layer
    """
    cfg_dict = update_config_with_3dblox_params(cfg_skeleton=cfg_skeleton,
                                                input_ds_dir=input_ds_dir,
                                                _3dbv_path=_3dbv_path,
                                                _3dbx_path=_3dbx_path,)
    for interface_name, cfg in cfg_dict.items():
        cfg.DESIGN = ds_name
        # Derive additional parameters based on mode
        if mode == "w2w_simulation" or mode == "w2w_modeling":
            cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
            cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
            cfg.S_INIT_A_M = 10e-6 * (cfg.WAF_R_um / 150000) ** 2
            cfg.S_INIT_B_M = 0.0
        elif mode == "d2w_simulation" or mode == "d2w_modeling":
            cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
            cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
            cfg.eff_DIE_R = float(np.sqrt((cfg.DIE_W_um / 2) ** 2 + (cfg.DIE_L_um / 2) ** 2))  # Effective die radius (um)
            cfg.S_INIT_A_M = 10e-6 * (cfg.eff_DIE_R / 150000) ** 2
            cfg.S_INIT_B_M = 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes are 'w2w_simulation', 'w2w_modeling', 'd2w_simulation', and 'd2w_modeling'.")
        # Save updated config file for reference
        OmegaConf.save(cfg, cfg_folder + f"/{interface_name}.yaml")

    if debug:
        cfg.DEBUG = True
        print("Configuration loaded:")
        print(OmegaConf.to_yaml(cfg))

    
    
    return cfg_dict



def update_config_from_bmap(cfg, blox_bmap_path, y_tol=0.1, x_tol=0.1):
    """
    Extract pad array layout from .bmap file.

    args:
        cfg: configuration object
        blox_bmap_path: path to .bmap file
        y_tol: tolerance for clustering y coordinates (um), if the difference between two y coordinates is less than y_tol, they are considered in the same row
        x_tol: tolerance for clustering x coordinates (um), if the difference between two x coordinates is less than x_tol, they are considered in the same column
    """
    coords = []

    with open(blox_bmap_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                x, y = float(parts[2]), float(parts[3])
                coords.append((x, y))
            except ValueError:
                continue

    if not coords:
        print("No valid pad coordinates found in the .bmap file.") 
        return

    coords = np.array(coords)

    # Rank by y descending
    coords = coords[np.argsort(-coords[:, 1])]

    # Cluster to get unique rows
    y_vals = []
    for y in coords[:, 1]:
        if not y_vals or abs(y - y_vals[-1]) > y_tol:
            y_vals.append(y)
    num_rows = len(y_vals)

    # Cluster to get unique columns
    first_row_y = y_vals[0]
    first_row = coords[np.abs(coords[:, 1] - first_row_y) < y_tol]
    x_vals = []
    for x in sorted(first_row[:, 0]):
        if not x_vals or abs(x - x_vals[-1]) > x_tol:
            x_vals.append(x)
    num_cols = len(x_vals)
    
    add_config_items(cfg, keys=['PAD_ARR_ROW', 'PAD_ARR_COL'], values=[num_rows, num_cols])
    add_config_items(cfg, keys=['PAD_ARR_L_um', 'PAD_ARR_W_um'],
                        values=[(num_rows - 1) * cfg.PITCH_r_um,
                                (num_cols - 1) * cfg.PITCH_c_um])


def update_config_with_3dblox_params(cfg_skeleton: object, 
                                    input_ds_dir: str,
                                    _3dbv_path: str,
                                    _3dbx_path: str,):
    """
    Update configuration with design parameters from .3dbv and .bmap files.
    args:
        cfg_skeleton: configuration object skeleton
        input_ds_dir: path to design input files directory
        _3dbv_path: path to .3dbv file (chiplet definitions)
        _3dbx_path: path to .3dbx file (stack configuration)
        _bmap_path: path to .bmap file (bump map)
    file structure:
        input_ds_dir/
          |-  xx_chiplet_definitions.3dbv
          |-  xx_stack_config.3dbx
          |-  XX_From_XX.bmap
          |-  XX.3dbf
          |-  XX_From_XX_criticality.txt
    """
    ### Update cfg_list with design parameters from .3dbv and .bmap files
    cfg_dict = dict()
    stack_config_3dbx = OmegaConf.load(_3dbx_path)

    for _, connection in stack_config_3dbx.Connection.items():
        cfg = cfg_skeleton.copy()

        # Extract interface names
        cfg.INTERFACE_TOP = str(((connection.bot).split('.')[-1]).split('To_')[-1])
        cfg.INTERFACE_BOT = str(((connection.top).split('.')[-1]).split('From_')[-1])
        cfg.INTERFACE = f"{cfg.INTERFACE_TOP}_From_{cfg.INTERFACE_BOT}"

        ### Read .3dbv, .3dbx, and .bmap files
        ## Extract design parameters from .3dbv and .3dbf file
        _3dbv = OmegaConf.load(_3dbv_path)
        _bmap_path = os.path.join(input_ds_dir, f"{cfg.INTERFACE}.bmap")
        top_3dbf_path = os.path.join(input_ds_dir, f"{cfg.INTERFACE_TOP}.3dbf")
        bot_3dbf_path = os.path.join(input_ds_dir, f"{cfg.INTERFACE_BOT}.3dbf")
        top_3dbf = OmegaConf.load(top_3dbf_path)
        bot_3dbf = OmegaConf.load(bot_3dbf_path)

        # Check unit
        assert _3dbv.Header.unit == 'micron', "Only support .3dbv file with unit in microns."
        
        # Read die width and length
        add_config_items(cfg, keys=['DIE_W_um', 'DIE_L_um'], 
                        values=[float(_3dbv.ChipletDef[cfg.INTERFACE_TOP].design_area[0]),
                                float(_3dbv.ChipletDef[cfg.INTERFACE_TOP].design_area[1])])
        
        # Read bump size, size/2 = radius. Find matching bum type
        bump_type_list = list(top_3dbf.Bump_Types.keys())   # silicon_individual_bonding, organic_individual_bonding, ...
        selected_bump_type = None

        with open(_bmap_path, 'r') as f:
            first_line = f.readline()
            for bump_type in bump_type_list:
                if bump_type in first_line.split()[1]:
                    selected_bump_type = bump_type
                    break

        if selected_bump_type is None:
            raise ValueError(f"No matching bump type found in {_bmap_path} for top chiplet {cfg.INTERFACE_TOP}.")
        
        add_config_items(cfg, keys=['PAD_TOP_R_um', 'PAD_BOT_R_um'], 
                            values=[float(top_3dbf.Bump_Types[selected_bump_type].bump_size) / 2,
                                    float(bot_3dbf.Bump_Types[selected_bump_type].bump_size) / 2])
        
        # Read pad pitch (top chip pitch) NOTE: currently assume row and col pitch are the same
        add_config_items(cfg, keys=['PITCH_r_um', 'PITCH_c_um'], 
                            values=[float(top_3dbf.Chiplet_Grid.pitch), 
                                    float(top_3dbf.Chiplet_Grid.pitch)])
        

        ## Extract design parameters from .bmap file
        update_config_from_bmap(cfg, _bmap_path, 
                                y_tol=cfg.PITCH_r_um * 0.1, x_tol=cfg.PITCH_c_um * 0.1)

        # Store in config dictionary
        cfg_dict[cfg.INTERFACE] = cfg

    return cfg_dict






def draw_pad_bitmap(cfg, bitmap_collection, output_path):
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

    fig = plt.figure(figsize=(10, 10))
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
    # plt.legend(
    #     handles=[red_patch, blue_patch, green_patch, light_gray_patch],
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, -0.07),
    #     ncol=4,
    #     frameon=False
    # )
    # plt.legend().set_visible(False)
    norm = BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5, 4.5], ncolors=cmap.N)
    plt.axis('off')
    plt.imshow(PAD_BITMAP, cmap=cmap, norm=norm)
    # plt.title("Pad Block Bitmap")


    # Save the pad bitmaps
    plt.savefig(os.path.join(output_path, cfg.INTERFACE + "_pad_bitmap.png"))
    plt.close(fig)
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
    with open(cfg.OUTPUT_DIR + cfg.INTERFACE + "/" + cfg.INTERFACE + "_criticality.txt", 'w') as f:
        for bump_crit in bump_criticality:
            f.write(f"{bump_crit['port']} {bump_crit['esd_criticality']:.6f} {bump_crit['mechanical_criticality']:.6f}\n")
    print("Criticality file saved in ", cfg.OUTPUT_DIR + cfg.INTERFACE + "/" + cfg.INTERFACE + "_criticality.txt")
    return


def risk_map_generator(cfg, 
                    interface: object,
                    input_args
                    ):
    '''
    Risk map output format:
    <pad_coords_x> <pad_coords_y> <esd_failure_probability> <overlay_failure_probability> <particle_failure_probability> <mechanical_failure_probability>
    '''
    risk_map = list()
    output_dir = os.path.join(cfg.OUTPUT_DIR, input_args['ds_name'], cfg.INTERFACE)
    risk_map_path = os.path.join(output_dir, f"{cfg.INTERFACE}_risk.map")
    for pad_id in range(len(interface.pad_coords)):
        pad_coords_x = interface.pad_coords[pad_id, 0]
        pad_coords_y = interface.pad_coords[pad_id, 1]
        if np.isnan(pad_coords_x) or np.isnan(pad_coords_y):
            continue
        pad_ovl_yield = interface.pad_yield_map['Y_ovl'].flatten()[pad_id]
        pad_df_yield = interface.pad_yield_map['Y_df'].flatten()[pad_id]
        pad_ce_yield = interface.pad_yield_map['Y_ce'].flatten()[pad_id]
        pad_esd_yield = interface.pad_yield_map['Y_esd'].flatten()[pad_id]
        risk_map.append({
            "pad_coords_x": pad_coords_x,
            "pad_coords_y": pad_coords_y,
            "esd_failure_probability": 1 - pad_esd_yield,
            "overlay_failure_probability": 1 - pad_ovl_yield,
            "particle_failure_probability": 1 - pad_df_yield,
            "mechanical_failure_probability": 1 - pad_ce_yield,
        })
    with open(risk_map_path, 'w') as f:
        for pad_risk in risk_map:
            f.write(f"{pad_risk['pad_coords_x']} {pad_risk['pad_coords_y']} {pad_risk['esd_failure_probability']} {pad_risk['overlay_failure_probability']} {pad_risk['particle_failure_probability']} {pad_risk['mechanical_failure_probability']}\n")
    print("Risk map file saved in ", risk_map_path)

    mechanism_specs = {
        "esd": ("Y_esd", "ESD Failure Probability"),
        "overlay": ("Y_ovl", "Overlay Failure Probability"),
        "particle": ("Y_df", "Particle Failure Probability"),
        "mechanical": ("Y_ce", "Mechanical Failure Probability"),
        "overall": ("Y_bond", "Overall Failure Probability"),
    }
    for mechanism, (yield_key, colorbar_label) in mechanism_specs.items():
        failure_map = 1.0 - np.asarray(interface.pad_yield_map[yield_key], dtype=np.float64)
        masked_failure_map = np.ma.masked_invalid(failure_map)

        finite_vals = failure_map[np.isfinite(failure_map)]
        vmax = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
        if vmax <= 0.0:
            vmax = 1.0

        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(
            masked_failure_map,
            cmap='hot',
            interpolation='nearest',
            vmin=0.0,
            vmax=vmax,
        )
        fig.colorbar(image, ax=ax, label=colorbar_label)
        ax.set_title(f"{mechanism.title()} Risk Map")
        ax.set_xlabel('Pad Column Index')
        ax.set_ylabel('Pad Row Index')

        save_path = os.path.join(output_dir, f"{cfg.INTERFACE}_{mechanism}_risk_map.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print("Failure mechanism risk maps saved in ", output_dir)
    print()
    
    return




def convert_3dblox_to_pad_bitmap(cfg, 
                                 _bmap_path: str,
                                 criticality_path: str,
                                 pad_arrange_pattern: str,
                                 input_args):
    '''
    pad_arrange_pattern: 'checkerboard' for UCIe standard and HBM
    '''
    # Create output directory if not exist
    output_path = os.path.join(cfg.OUTPUT_DIR, input_args['ds_name'], cfg.INTERFACE)      
        
    sort_pads_bmap(_bmap_path, _bmap_path)

    # Read the bump data from the .bmap file
    bump_data = []
    # Initialize the pad array boundaries
    [pad_array_left, pad_array_right, pad_array_top, pad_array_bottom] = [float('inf'), float('-inf'), float('-inf'), float('inf')]
    with open(_bmap_path, 'r') as f:
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

    # Record the 1D physical locations of each pad in redundant nets
    '''Example: {NC: [0, 5, 10], VSS: [1, 6, 11], VDD: [2, 7, 12], ...}'''
    redundant_net_to_1d_physical_mask = dict()   
    # Record the bump ids of each pad in redundant nets, bump id is the index in bump_data list
    redundant_net_to_bumpids = dict()
    
    for bump in bump_data:
        if bump['net'] not in redundant_net_to_bumpids:
            redundant_net_to_bumpids[bump['net']] = set()
            redundant_net_to_1d_physical_mask[bump['net']] = np.array([], dtype=int)
        redundant_net_to_bumpids[bump['net']].add(bump['bumpid'])

    # Generate the criticality map
    '''
    Current Format: <net1> [net2] [net3] ... <group_size> <tolerated_esd_failures> <tolerated_mechanical_failures>
   
    Where:
    - group_size: Total number of pads/bumps in the redundancy group
    - tolerated_esd_failures: Number of ESD failures the group can tolerate before failing
    - tolerated_mechanical_failures: Number of mechanical failures the group can tolerate before failing
    
    Criticality values are calculated when reading the file:
    - esd_criticality = (group_size - tolerated_esd_failures) / group_size
    - mechanical_criticality = (group_size - tolerated_mechanical_failures) / group_size
    '''
    # criticality_generator(cfg, bump_data, redundant_net_to_bumpids)
    criticality_info = dict()
    with open(criticality_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            net, num_copy, tolerated_esd_failures, tolerated_mechanical_failures = parts
            criticality_info[net] = {
                "tolerated_esd_failures": int(tolerated_esd_failures),
                "tolerated_mechanical_failures": int(tolerated_mechanical_failures)
            }


    # Initialize the pad bitmap
    # TODO: You need to modify the simulator to support different pad arrangement patterns
    CRITICAL_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    REDUNDANT_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    DUMMY_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    ESD_CRITICAL_PAD_BITMAP = np.zeros((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), dtype=bool)
    pad_coords = np.full((cfg.PAD_ARR_ROW * cfg.PAD_ARR_COL, 2), np.nan, dtype=np.float32)  # x, y coordinates of each bump
    # Build a mapping array from physical bump location (r, c) to bump id
    mapping_physical_to_bumpid = np.full((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), np.nan, dtype=np.float32) # Shape: (PAD_ARR_ROW, PAD_ARR_COL)

    if pad_arrange_pattern in ('checkerboard', 'rectangular'):
        for bump in bump_data:
            x = bump['x']
            y = bump['y']
            row = int(round((pad_array_top - y ) / (cfg.PITCH_r_um)))   # Because in checkerboard pattern, the pitch per row is halved
            col = int(round((x - pad_array_left) / (cfg.PITCH_c_um)))   # Because in checkerboard pattern, the pitch per column is halved
            if not (0 <= row < cfg.PAD_ARR_ROW) or not (0 <= col < cfg.PAD_ARR_COL):
                raise IndexError(
                    f"Pad indexing out of bounds for interface {cfg.INTERFACE}: "
                    f"bumpid={bump['bumpid']} net={bump['net']} x={x} y={y} "
                    f"-> row={row}, col={col}, "
                    f"shape=({cfg.PAD_ARR_ROW}, {cfg.PAD_ARR_COL}), "
                    f"pitch=({cfg.PITCH_r_um}, {cfg.PITCH_c_um}), "
                    f"bbox(left={pad_array_left}, right={pad_array_right}, "
                    f"top={pad_array_top}, bottom={pad_array_bottom})"
                )
            mapping_physical_to_bumpid[row, col] = bump['bumpid']
            pad_coords[row * cfg.PAD_ARR_COL + col, 0] = bump['x'] - (pad_array_left + pad_array_right) / 2
            pad_coords[row * cfg.PAD_ARR_COL + col, 1] = bump['y'] - (pad_array_top + pad_array_bottom) / 2
            current_bump_net = bump['net']
            num_copies = len(redundant_net_to_bumpids[current_bump_net])
            if 'dummy' in current_bump_net.lower():
                DUMMY_PAD_BITMAP[row, col] = 1
                continue
            if num_copies == 1:
                CRITICAL_PAD_BITMAP[row, col] = 1
                ESD_CRITICAL_PAD_BITMAP[row, col] = 1
                redundant_net_to_bumpids.pop(current_bump_net, None)
                redundant_net_to_1d_physical_mask.pop(current_bump_net, None)
                continue
            elif num_copies > 1: 
                REDUNDANT_PAD_BITMAP[row, col] = 1
                ESD_CRITICAL_PAD_BITMAP[row, col] = 1 if criticality_info[current_bump_net]['tolerated_esd_failures'] == 0 else 0
                redundant_net_to_1d_physical_mask[bump['net']] = np.append(redundant_net_to_1d_physical_mask[bump['net']], row * cfg.PAD_ARR_COL + col)
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
    bitmap_collection["num_critical_pads"] = num_critical_pads
    bitmap_collection["num_redundant_pads"] = num_redundant_pads
    bitmap_collection["num_dummy_pads"] = num_dummy_pads
    bitmap_collection["redundant_net_to_bumpids"] = redundant_net_to_bumpids
    bitmap_collection["redundant_net_to_1d_physical_mask"] = redundant_net_to_1d_physical_mask
    bitmap_collection["pad_coords"] = pad_coords
    bitmap_collection["mapping_physical_to_bumpid"] = mapping_physical_to_bumpid
    bitmap_collection["criticality_info"] = criticality_info
    
    # Save the bitmap collection as npy file and mat file
    np.save(output_path + '/' + cfg.INTERFACE + "_bitmap_collection.npy", bitmap_collection)
    # sio.savemat(cfg.OUTPUT_DIR + "bitmap_collection.mat", bitmap_collection)

    # # Draw the critical and redundant pad bitmaps in one figure (critical light red, redundant light blue, dummy light gray)
    draw_pad_bitmap(cfg, bitmap_collection, output_path)
    # raise NotImplementedError("Stop here to avoid confusion.")

    return bitmap_collection



def result_wrapper(
        mode: str,
        output_dir: str,
        interface: str,
        fail_map_dict = None,
):
    """
    Wrap up the results, plot them and save the figures.
    """
    save_path = os.path.join(output_dir, interface)
    if mode in ["d2w_simulation", "w2w_simulation"]:
        for mechanism, fail_map in fail_map_dict.items():
            # Draw the failure map and save the figure to the output directory
            figure = plt.figure(figsize=(10, 10))
            plt.imshow(fail_map, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Failure Count')
            plt.title(f'Assembly Failure Map - {mechanism}')
            plt.savefig(save_path + f'/simulation_failure_map_{mechanism}.png')
            plt.close(figure)
            print(f"Failure map for {mechanism} saved to {save_path + f'/simulation_failure_map_{mechanism}.png'}")





