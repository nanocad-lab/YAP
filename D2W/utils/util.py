#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from omegaconf import OmegaConf
import numpy as np
import random


def configure_random_seed(cfg, announce=True):
    """Configure process-wide RNGs from an optional config seed."""
    use_random_seed = bool(getattr(cfg, "USE_RANDOM_SEED", False))
    if not use_random_seed:
        if announce:
            print("Random seed: disabled (non-reproducible run).")
        return None

    random_seed = int(getattr(cfg, "RANDOM_SEED", 0))
    np.random.seed(random_seed)
    random.seed(random_seed)
    if announce:
        print(f"Random seed: {random_seed} (fixed and reproducible).")
    return random_seed


def format_yield_array(values, decimals=4):
    """Format scalar or array-like yield output without rounding stored values."""
    return np.array2string(
        np.asarray(values, dtype=float), precision=decimals, floatmode="fixed"
    )

def load_modeling_config(path, mode, debug=False):
    full_cfg = OmegaConf.load(path)
    cfg = full_cfg[mode]

    if mode == "w2w_simulation" or mode == "w2w_modeling":
        cfg.PAD_ARR_ROW = int(np.floor(float(cfg.DIE_L_um / cfg.PITCH_um)))  # number of pads in a row of pad array
        cfg.PAD_ARR_COL = int(np.floor(float(cfg.DIE_W_um / cfg.PITCH_um)))  # number of pads in a column of pad array
        cfg.PAD_ARR_L_um = (cfg.PAD_ARR_ROW - 1) * cfg.PITCH_um  # pad array length (um)
        cfg.PAD_ARR_W_um = (cfg.PAD_ARR_COL - 1) * cfg.PITCH_um  # pad array width (um)
        cfg.PAD_BOT_R_um = cfg.PITCH_um / 2 * cfg.PAD_BOT_R_um_ratio  # bottom Cu pad radius (um)
        cfg.PAD_TOP_R_um = cfg.PAD_BOT_R_um * cfg.PAD_TOP_R_um_ratio  # top Cu pad radius (um)
        cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
        cfg.pad_block_size = int(cfg.pad_block_dim_um / cfg.PITCH_um)  # pad block size (#rows or #columns of the pad block)
    elif mode == "d2w_simulation" or mode == "d2w_modeling":
        cfg.PAD_ARR_ROW = int(np.floor(float(cfg.DIE_L_um / cfg.PITCH_um)))  # number of pads in a row of pad array
        cfg.PAD_ARR_COL = int(np.floor(float(cfg.DIE_W_um / cfg.PITCH_um)))  # number of pads in a column of pad array
        cfg.PAD_ARR_L_um = (cfg.PAD_ARR_ROW - 1) * cfg.PITCH_um  # pad array length (um)
        cfg.PAD_ARR_W_um = (cfg.PAD_ARR_COL - 1) * cfg.PITCH_um  # pad array width (um)
        cfg.PAD_BOT_R_um = cfg.PITCH_um / 2 * cfg.PAD_BOT_R_um_ratio  # bottom Cu pad radius (um)
        cfg.PAD_TOP_R_um = cfg.PAD_BOT_R_um * cfg.PAD_TOP_R_um_ratio  # top Cu pad radius (um)
        cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
        cfg.pad_block_size = int(cfg.pad_block_dim_um / cfg.PITCH_um)  # pad block size (#rows or #columns of the pad block)
        cfg.eff_DIE_R = float(np.sqrt(cfg.DIE_W_um * cfg.DIE_L_um / np.pi))  # Effective wafer radius (um)
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes are 'w2w_simulation', 'w2w_modeling', 'd2w_simulation', and 'd2w_modeling'.")


    configure_random_seed(cfg)

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
        cfg.PAD_ARR_ROW = int(np.floor(float(cfg.DIE_L_um / cfg.PITCH_um)))  # number of pads in a row of pad array
        cfg.PAD_ARR_COL = int(np.floor(float(cfg.DIE_W_um / cfg.PITCH_um)))  # number of pads in a column of pad array
        cfg.PAD_ARR_L_um = (cfg.PAD_ARR_ROW - 1) * cfg.PITCH_um  # pad array length (um)
        cfg.PAD_ARR_W_um = (cfg.PAD_ARR_COL - 1) * cfg.PITCH_um  # pad array width (um)
        cfg.PAD_BOT_R_um = cfg.PITCH_um / 2 * cfg.PAD_BOT_R_um_ratio  # bottom Cu pad radius (um)
        cfg.PAD_TOP_R_um = cfg.PAD_BOT_R_um * cfg.PAD_TOP_R_um_ratio  # top Cu pad radius (um)
        cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
        cfg.pad_block_size = int(cfg.pad_block_dim_um / cfg.PITCH_um)  # pad block size (#rows or #columns of the pad block)
    elif mode == "d2w_simulation" or mode == "d2w_modeling":
        cfg.PAD_ARR_ROW = int(np.floor(float(cfg.DIE_L_um / cfg.PITCH_um)))  # number of pads in a row of pad array
        cfg.PAD_ARR_COL = int(np.floor(float(cfg.DIE_W_um / cfg.PITCH_um)))  # number of pads in a column of pad array
        cfg.PAD_ARR_L_um = (cfg.PAD_ARR_ROW - 1) * cfg.PITCH_um  # pad array length (um)
        cfg.PAD_ARR_W_um = (cfg.PAD_ARR_COL - 1) * cfg.PITCH_um  # pad array width (um)
        cfg.PAD_BOT_R_um = cfg.PITCH_um / 2 * cfg.PAD_BOT_R_um_ratio  # bottom Cu pad radius (um)
        cfg.PAD_TOP_R_um = cfg.PAD_BOT_R_um * cfg.PAD_TOP_R_um_ratio  # top Cu pad radius (um)
        cfg.SYSTEM_MAGNIFICATION_MEAN_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN_um + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD_ppm = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD_um) ** 2 / 1e6
        cfg.pad_block_size = int(cfg.pad_block_dim_um / cfg.PITCH_um)  # pad block size (#rows or #columns of the pad block)
        cfg.eff_DIE_R = float(np.sqrt(cfg.DIE_W_um * cfg.DIE_L_um / np.pi))  # Effective wafer radius (um)
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes are 'w2w_simulation', 'w2w_modeling', 'd2w_simulation', and 'd2w_modeling'.")
