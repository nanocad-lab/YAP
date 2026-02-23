#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Feb 20, 2026

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.special import ndtr
from debond import debond_dishing_bounds_calculator
import matplotlib.pyplot as plt
import time


# =====================================================================
# Gauss-Hermite quadrature nodes/weights mapped to N(0,1)
# =====================================================================
def _gh_nodes_weights(n: int):
    """Return (z, w) for N(0,1) Gauss-Hermite quadrature with *n* points."""
    x, w = np.polynomial.hermite.hermgauss(n)
    z = np.sqrt(2.0) * x
    w_norm = w / np.sqrt(np.pi)
    return z, w_norm


# =====================================================================
# Semi-analytic Cu-recess die yield with spatial correlation
# (Gauss–Hermite, direct-domain — no log / exp tricks)
# =====================================================================
def cu_recess_die_yield_spatial(
    mu: float,
    a: np.ndarray,
    b: np.ndarray,
    sigma_T: float,
    sigma_eps: float,
    block_indices: np.ndarray,
    n_gh: int = 60,
) -> float:
    r"""
    Compute Cu-recess die yield under a 2-level spatial-correlation
    model using Gauss–Hermite quadrature — **all arithmetic in the
    direct probability domain** (no log / exp).

    Variance decomposition
    ----------------------
    Each pad's Cu height is modelled as

        h_i = mu  +  sigma_T * G_{T,b(i)}  +  sigma_eps * eps_i

    where
        G_{T,b}   ~ N(0,1)   per-block (tile-level), i.i.d. across blocks
        eps_i     ~ N(0,1)   per-pad  (residual),    i.i.d. across pads

    A pad *survives* when  a_i <= h_i <= b_i  (per-pad bounds).

    Parameters
    ----------
    mu : float
        Mean Cu height (nm).
    a : 1-D array, shape (N,)
        Lower survival bound for each pad (nm).  **Per-pad.**
    b : 1-D array, shape (N,)
        Upper survival bound for each pad (nm).  **Per-pad.**
    sigma_T : float
        Std-dev of the block-level (tile) random effect.
    sigma_eps : float
        Std-dev of the pad-level (residual) random effect.
    block_indices : 1-D int array, shape (N,)
        Block (tile) assignment for each pad.  Values in [0, B-1].
    n_gh : int
        Number of GH quadrature points for the G_T integral.

    Returns
    -------
    Y : float
        Estimated die yield in [0, 1].
    """
    mu = float(mu)
    a  = np.asarray(a,  dtype=np.float64)
    b  = np.asarray(b,  dtype=np.float64)
    block_indices = np.asarray(block_indices, dtype=np.intp)
    N = a.size
    assert b.shape == (N,) and block_indices.shape == (N,)

    # Unique blocks and per-block pad indices
    unique_blocks = np.unique(block_indices)
    B = unique_blocks.size
    block_pad_indices = [np.where(block_indices == blk)[0] for blk in unique_blocks]

    # GH nodes & weights for G_T
    J = n_gh
    z_T, w_T = _gh_nodes_weights(J)   # (J,)

    # ------------------------------------------------------------------
    # mean_vec[j] = mu + sigma_T * z_T[j]   (same for every pad)
    # Only (a_i, b_i) differ across pads.
    #
    # Compute the (N, J) survival-prob matrix for ALL pads at once
    # with just 2 norm.cdf calls, then gather per-block products.
    # ------------------------------------------------------------------
    mean_vec = mu + sigma_T * z_T   # shape (J,)

    p_all = norm.cdf((b[:, None] - mean_vec[None, :]) / sigma_eps) \
          - norm.cdf((a[:, None] - mean_vec[None, :]) / sigma_eps)
    np.clip(p_all, 0.0, 1.0, out=p_all)

    # For each block: prod over pads → E_{G_T}[block survives] → multiply
    Y = 1.0
    for pad_idx in block_pad_indices:
        # block_yield_per_t[j] = prod_{i in block} p_all[i, j]   shape (J,)
        block_yield_per_t = np.prod(p_all[pad_idx], axis=0)
        # E_block = sum_j  w_T[j] * block_yield_per_t[j]
        Y *= np.dot(w_T, block_yield_per_t)

    return float(np.clip(Y, 0.0, 1.0))


# =====================================================================
# Helper: assign pads to spatial blocks on a regular grid
# =====================================================================
def assign_pads_to_blocks(
    PAD_ARR_ROW: int,
    PAD_ARR_COL: int,
    block_size_r: int,
    block_size_c: int,
) -> np.ndarray:
    """
    Partition an (PAD_ARR_ROW x PAD_ARR_COL) pad array into rectangular
    tiles of *block_size_r x block_size_c* pads and return a 1-D block-index
    array aligned with the given (pad_rows, pad_cols) coordinates.

    Parameters
    ----------
    pad_rows, pad_cols : 1-D int arrays, shape (N_valid,)
        Row / column indices of the valid (non-dummy) pads.
    PAD_ARR_ROW, PAD_ARR_COL : int
        Full pad-array dimensions.
    block_size_r, block_size_c : int
        Tile side lengths **in pads** (e.g. 10 means 10×10 pad tiles).

    Returns
    -------
    block_indices : 1-D int array, shape (N_valid,)
    """
    pad_rows, pad_cols = np.meshgrid(np.arange(PAD_ARR_ROW), np.arange(PAD_ARR_COL), indexing='ij')
    pad_rows = pad_rows.ravel()
    pad_cols = pad_cols.ravel()
    block_r = pad_rows // block_size_r
    block_c = pad_cols // block_size_c
    return block_r * int(np.ceil(PAD_ARR_COL / block_size_c)) + block_c


def pad_Cu_expansion_yield_map_generator(*,
        cfg,
        wafer,
        TOP_DISH_MEAN_nm: float,
        TOP_DISH_STD_nm: float,
        BOT_DISH_MEAN_nm: float,
        BOT_DISH_STD_nm: float,
        pad_bitmap_collection: dict,
    ):
    glb_cu_expansion_pad_yield_min = 1.0  # Initialize to a high value
    glb_cu_expansion_pad_yield_max = 0.0  # Initialize to a low value
    valid_pad_mask = (pad_bitmap_collection['CRITICAL_PAD_BITMAP'] == 1) | (pad_bitmap_collection['REDUNDANT_PAD_BITMAP'] == 1) | (pad_bitmap_collection['DUMMY_PAD_BITMAP'] == 1)
    for i, die in enumerate(wafer.die_list):
        die_pad_coords = wafer.base_pad_coords + die.die_center
        valid_die_pad_coords = die_pad_coords[valid_pad_mask.flatten() == 1]
        start_time = time.time()
        valid_dishing_bound_array = debond_dishing_bounds_calculator(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
        print("Dishing bound calculation time for die {}: {:.2f} seconds".format(i, time.time() - start_time))
        
        upper_limits_valid_pads = - valid_dishing_bound_array[:, 0] * 2 # - upper limits of the sum of top and bottom Cu heights
        lower_limits_valid_pads = - valid_dishing_bound_array[:, 1] * 2 # - lower limits of the sum of top and bottom Cu heights
        pos_valid_pads = norm.cdf(upper_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)) - \
                   norm.cdf(lower_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2))
        pad_yield_map = np.full((cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL), np.nan)
        pad_yield_map[valid_pad_mask == 1] = pos_valid_pads
        
        glb_cu_expansion_pad_yield_min = min(glb_cu_expansion_pad_yield_min, np.nanmin(pad_yield_map))
        glb_cu_expansion_pad_yield_max = max(glb_cu_expansion_pad_yield_max, np.nanmax(pad_yield_map))
        die.pad_yield_map['Y_ce'] = pad_yield_map
        print("Generated pad-level Cu expansion yield map for die {}.".format(i))
        
    wafer.glb_pad_yield_min_max_dict['Y_ce'] = (glb_cu_expansion_pad_yield_min, glb_cu_expansion_pad_yield_max)
    print("Global min of the pad-level Cu expansion yield: {}".format(glb_cu_expansion_pad_yield_min))
    print("Global max of the pad-level Cu expansion yield: {}".format(glb_cu_expansion_pad_yield_max))



def stack_stress_yield_calculator_0(
        cfg_dict: dict,
        waf_stack,
        pad_bitmap_collection_dict: dict,
        valid_pad_mask_dict: dict,
):
    for interface_name, cfg in cfg_dict.items():
        interface = waf_stack.interfaces.interface_dict[interface_name]
        pad_bitmap_collection = pad_bitmap_collection_dict[interface_name]
        valid_pad_mask = valid_pad_mask_dict[interface_name]

        # Extract the necessary parameters for Cu expansion yield calculation
        TOP_DISH_MEAN_nm, TOP_DISH_STD_nm = cfg.TOP_DISH_MEAN_nm, cfg.TOP_DISH_STD_L_nm
        BOT_DISH_MEAN_nm, BOT_DISH_STD_nm = cfg.BOT_DISH_MEAN_nm, cfg.BOT_DISH_STD_L_nm
        CRITICAL_PAD_MASK = pad_bitmap_collection['CRITICAL_PAD_BITMAP'].flatten()
        redundant_net_to_1d_physical_mask = pad_bitmap_collection['redundant_net_to_1d_physical_mask']


        stress_yield_list = []

        for die_ind, die in enumerate(interface.die_list):
            die_pad_coords = interface.base_pad_coords + die.die_center
            valid_die_pad_coords = die_pad_coords[valid_pad_mask.flatten() == 1]
            start_time = time.time()
            valid_dishing_bound_array = debond_dishing_bounds_calculator(cfg, valid_die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
            # print("Dishing bound calculation time for die {}: {:.2f} seconds".format(die_ind, time.time() - start_time))
            upper_limits_valid_pads = - valid_dishing_bound_array[:, 0] * 2 # - upper limits of the sum of top and bottom Cu heights
            lower_limits_valid_pads = - valid_dishing_bound_array[:, 1] * 2 # - lower limits of the sum of top and bottom Cu heights
            pos_valid_pads = norm.cdf(upper_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2)) - \
                    norm.cdf(lower_limits_valid_pads, loc=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm, scale=np.sqrt(TOP_DISH_STD_nm**2 + BOT_DISH_STD_nm**2))
            # Critical yield is the pos of the critical pads multiplied together
            stress_yield_critical_pads = np.prod(pos_valid_pads[CRITICAL_PAD_MASK == 1])
            stress_yield_redundant_nets = 1.0
            for redundant_net, physical_pad_indices in redundant_net_to_1d_physical_mask.items():
                num_replicas = len(physical_pad_indices)
                stress_yield_redundant_nets *= 1 - (1 - np.prod(pos_valid_pads[physical_pad_indices])) ** num_replicas
            stress_yield = stress_yield_critical_pads * stress_yield_redundant_nets
            stress_yield_list.append(stress_yield)

            # break
            
        # Update the die yield list for this interface in the wafer stack
        waf_stack.die_yield_list_per_interface_dict[interface_name]['mechanical'] = np.array(stress_yield_list)




def stack_stress_yield_calculator(
        cfg_dict: dict,
        waf_stack,
):
    for interface_name, cfg in cfg_dict.items():
        interface = waf_stack.interfaces.interface_dict[interface_name]
        # Extract the necessary parameters for Cu expansion yield calculation
        PAD_ARR_ROW, PAD_ARR_COL = cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL
        TOP_DISH_MEAN_nm, TOP_DISH_STD_L_nm, TOP_DISH_STD_T_nm, TOP_DISH_STD_E_nm \
            = cfg.TOP_DISH_MEAN_nm, cfg.TOP_DISH_STD_L_nm, cfg.TOP_DISH_STD_T_nm, cfg.TOP_DISH_STD_E_nm
        BOT_DISH_MEAN_nm, BOT_DISH_STD_L_nm, BOT_DISH_STD_T_nm, BOT_DISH_STD_E_nm \
            = cfg.BOT_DISH_MEAN_nm, cfg.BOT_DISH_STD_L_nm, cfg.BOT_DISH_STD_T_nm, cfg.BOT_DISH_STD_E_nm
        block_size_r, block_size_c = cfg.TL_um // cfg.PITCH_r_um, cfg.TL_um // cfg.PITCH_c_um

        stress_yield_list = []

        for die_ind, die in enumerate(interface.die_list):
            die_pad_coords = interface.base_pad_coords + die.die_center
            start_time = time.time()
            pad_dishing_bound_array = debond_dishing_bounds_calculator(cfg, die_pad_coords) # (num_pads, 2) array: (dishing_low_nm, dishing_high_nm)
            print("Dishing bound calculation time for die {}: {:.2f} seconds".format(die_ind, time.time() - start_time))
            upper_limits_valid_pads = - pad_dishing_bound_array[:, 0] * 2 # - upper limits of the sum of top and bottom Cu heights
            lower_limits_valid_pads = - pad_dishing_bound_array[:, 1] * 2 # - lower limits of the sum of top and bottom Cu heights
            
            block_idx = assign_pads_to_blocks(PAD_ARR_ROW, PAD_ARR_COL, block_size_r, block_size_c)
            time_before_yield_calc = time.time()
            stress_die_yield = cu_recess_die_yield_spatial(
                    mu=TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm,
                    a=lower_limits_valid_pads,
                    b=upper_limits_valid_pads,
                    sigma_T=np.sqrt(TOP_DISH_STD_T_nm**2 + BOT_DISH_STD_T_nm**2),
                    sigma_eps=np.sqrt(TOP_DISH_STD_E_nm**2 + BOT_DISH_STD_E_nm**2),
                    block_indices=block_idx)
            print("Cu expansion yield calculation time for die {}: {:.2f} seconds".format(die_ind, time.time() - time_before_yield_calc))
            stress_yield_list.append(stress_die_yield)

        waf_stack.die_yield_list_per_interface_dict[interface_name]['mechanical'] = np.array(stress_yield_list)