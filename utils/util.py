#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from omegaconf import OmegaConf
import numpy as np

def load_modeling_config(path, mode, debug=False):
    full_cfg = OmegaConf.load(path)
    cfg = full_cfg[mode]

    if mode == "w2w_simulation" or mode == "w2w_modeling":
        cfg.PAD_ARR_ROW = int(np.floor(float(cfg.DIE_L / cfg.PITCH)))  # number of pads in a row of pad array
        cfg.PAD_ARR_COL = int(np.floor(float(cfg.DIE_W / cfg.PITCH)))  # number of pads in a column of pad array
        cfg.PAD_ARR_L = (cfg.PAD_ARR_ROW - 1) * cfg.PITCH  # pad array length (um)
        cfg.PAD_ARR_W = (cfg.PAD_ARR_COL - 1) * cfg.PITCH  # pad array width (um)
        cfg.PAD_BOT_R = cfg.PITCH / 2 * cfg.PAD_BOT_R_ratio  # bottom Cu pad radius (um)
        cfg.PAD_TOP_R = cfg.PAD_BOT_R * cfg.PAD_TOP_R_ratio  # top Cu pad radius (um)
        cfg.SYSTEM_MAGNIFICATION_MEAN = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD) ** 2 / 1e6
        cfg.pad_block_size = int(cfg.pad_block_dim / cfg.PITCH)  # pad block size (#rows or #columns of the pad block)
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
        cfg.PAD_ARR_ROW = int(np.floor(float(cfg.DIE_L / cfg.PITCH)))  # number of pads in a row of pad array
        cfg.PAD_ARR_COL = int(np.floor(float(cfg.DIE_W / cfg.PITCH)))  # number of pads in a column of pad array
        cfg.PAD_ARR_L = (cfg.PAD_ARR_ROW - 1) * cfg.PITCH  # pad array length (um)
        cfg.PAD_ARR_W = (cfg.PAD_ARR_COL - 1) * cfg.PITCH  # pad array width (um)
        cfg.PAD_BOT_R = cfg.PITCH / 2 * cfg.PAD_BOT_R_ratio  # bottom Cu pad radius (um)
        cfg.PAD_TOP_R = cfg.PAD_BOT_R * cfg.PAD_TOP_R_ratio  # top Cu pad radius (um)
        cfg.SYSTEM_MAGNIFICATION_MEAN = (cfg.k_mag * cfg.BOW_DIFFERENCE_MEAN + cfg.M_0) / 1e6
        cfg.SYSTEM_MAGNIFICATION_STD = (cfg.k_mag * cfg.BOW_DIFFERENCE_STD) ** 2 / 1e6
        cfg.pad_block_size = int(cfg.pad_block_dim / cfg.PITCH)  # pad block size (#rows or #columns of the pad block)
