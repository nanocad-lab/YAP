#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Wafers and Dies intialization for the yield model for hybrid bonding
#### Author: Zhichao Chen
#### Date: Feb 20, 2026

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.special import ndtr
from debond import debond_dishing_bounds_calculator, debond_dishing_bounds_calculator_coords
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
# Semi-analytic Cu-recess die yield with 3-level spatial correlation
# (Double Gauss–Hermite, log-domain for numerical stability)
#
#   Y = E_{G_L}[ Π_b  E_{G_T,b}[ Π_{i∈b} p_i(G_L, G_T,b) ] ]   (Eq. 13)
#
# Structure follows modeled_gh() in simulator_main.ipynb:
#   outer loop over G_L  →  inner loop over G_T  →  vectorised over pads
# Optimisations vs. naïve (N, nT) broadcast:
#   1. Precompute standardised z-scores  (avoid repeated (b-mu)/σ_ε)
#   2. Skip trivially-safe pads  (p ≈ 1 for all GH nodes → log p ≈ 0)
#   3. Inner loop over G_T nodes with (n_active,) arrays + reduceat
#      (avoids allocating the huge (N, nT) matrix)
# =====================================================================
def cu_recess_die_yield_spatial(
    mu: float,
    a: np.ndarray,
    b: np.ndarray,
    sigma_L: float,
    sigma_T: float,
    sigma_eps: float,
    block_indices: np.ndarray,
    n_gh_outer: int = 40,
    n_gh_inner: int = 40,
    g: float = None,
) -> float:
    r"""
    Compute Cu-recess die yield under a 3-level spatial-correlation
    model using Gauss–Hermite quadrature.

    When **g is None** (default):
        Full Eq. 13 — double GH over both G_L and G_T.

    When **g is given** (float):
        G_L = g is fixed; only single GH over G_T remains:

        Y(g) = Π_b  E_{G_{T,b}}[ Π_{i∈b} p_i(g, G_{T,b}) ]

        This is ~n_gh_outer× faster.

    Parameters
    ----------
    mu        : mean Cu height (nm)
    a, b      : per-pad lower/upper survival bounds, shape (N,)
    sigma_L   : std-dev of die-level (lot/wafer) random effect
    sigma_T   : std-dev of block-level (tile) random effect
    sigma_eps : std-dev of pad-level (residual) random effect
    block_indices : block assignment per pad, shape (N,), values in [0, B-1]
    n_gh_outer, n_gh_inner : GH quadrature points for G_L / G_T
    g         : if not None, fixed realisation of G_L (skip outer integral)

    Returns
    -------
    Y : float, estimated die yield in [0, 1]
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    block_indices = np.asarray(block_indices, dtype=np.intp)
    N = a.size

    # ── Sort all pads by block so reduceat works ──
    order    = np.argsort(block_indices, kind='mergesort')
    a_s      = a[order]
    b_s      = b[order]
    bi_s     = block_indices[order]
    _, blk_start = np.unique(bi_s, return_index=True)
    B = blk_start.size

    zT, wT = _gh_nodes_weights(n_gh_inner)
    inv_sE = 1.0 / sigma_eps if sigma_eps > 0 else 0.0

    # ── Pre-allocate work buffers ──
    _p = np.empty(N, dtype=np.float64)
    _q = np.empty(N, dtype=np.float64)
    log_blk = np.empty((B, n_gh_inner), dtype=np.float64)

    # ------------------------------------------------------------------
    # Build list of g-values to evaluate
    # ------------------------------------------------------------------
    if g is not None:
        g_vals = [float(g)]
        g_weights = None
    else:
        zG, wG = _gh_nodes_weights(n_gh_outer)
        g_vals = zG
        g_weights = wG

    log_outer = np.empty(len(g_vals), dtype=np.float64)

    for k, g_val in enumerate(g_vals):
        mu_g = mu + sigma_L * g_val

        for j in range(n_gh_inner):
            mean_j = mu_g + sigma_T * zT[j]

            if sigma_eps > 0:
                np.subtract(b_s, mean_j, out=_p)
                _p *= inv_sE
                ndtr(_p, out=_p)
                np.subtract(a_s, mean_j, out=_q)
                _q *= inv_sE
                ndtr(_q, out=_q)
                np.subtract(_p, _q, out=_p)
            else:
                _p[:] = np.where((mean_j >= a_s) & (mean_j <= b_s), 1.0, 0.0)

            np.clip(_p, 1e-300, 1.0, out=_p)
            np.log(_p, out=_p)
            log_blk[:, j] = np.add.reduceat(_p, blk_start)

        # Per-block logsumexp over T nodes
        mx = log_blk.max(axis=1, keepdims=True)
        log_E = mx[:, 0] + np.log(
            np.sum(wT[None, :] * np.exp(log_blk - mx), axis=1)
        )
        log_outer[k] = log_E.sum()

    # ------------------------------------------------------------------
    if g is not None:
        # Fast path: single g
        return float(np.clip(np.exp(log_outer[0]), 0.0, 1.0))

    # Full path: logsumexp over G_L
    mx = log_outer.max()
    Y = float(np.sum(g_weights * np.exp(log_outer - mx)) * np.exp(mx))
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




def _vertex_distance_key(die, radial_lut_array, decimals: int = 1):
    """
    Build a hashable cache key from the dishing-LUT values that a die's
    four vertices map to.

    The radial LUT columns are [r_um, D_sio2_nm, D_cu_nm].
    For each vertex we look up (D_sio2, D_cu) by interpolating into the
    LUT, round to *decimals* decimal places, and pack min/max into a tuple.
    Two dies that produce the same key have identical dishing bounds on
    all their pads (because dishing is radially monotone and the vertex
    radii bracket all pad radii inside the die).
    """
    lut_r    = radial_lut_array[:, 0]
    lut_dishing_ub = radial_lut_array[:, 1]
    lut_dishing_lb   = radial_lut_array[:, 2]

    verts = np.asarray(die.vertices_coords, dtype=np.float64)   # (4, 2)
    v_radii = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)          # (4,)

    d_ub = np.interp(v_radii, lut_r, lut_dishing_ub)  # (4,)
    d_lb   = np.interp(v_radii, lut_r, lut_dishing_lb)  # (4,)

    # Round and build immutable key from the range seen across vertices
    return (
        round(float(d_ub.min()), decimals),
        round(float(d_ub.max()), decimals),
        round(float(d_lb.min()),   decimals),
        round(float(d_lb.max()),   decimals),
    )


def stack_stress_yield_calculator(
        cfg_dict: dict,
        waf_stack,
):
    """
    Cu-expansion (mechanical) yield per die, using:
      1) Radial-layer grouping  — ~1200 dies  →  ~20 representative groups
      2) Vertex-distance cache  — skip GH when dishing bounds are identical
      3) debond_dishing_bounds_calculator()        — one-shot radial LUT
         debond_dishing_bounds_calculator_coords() — per-pad lookup on cache miss
    """
    for interface_name, cfg in cfg_dict.items():
        interface = waf_stack.interfaces.interface_dict[interface_name]

        # --- Config ---
        PAD_ARR_ROW, PAD_ARR_COL = cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL
        TOP_DISH_MEAN_nm  = cfg.TOP_DISH_MEAN_nm
        TOP_DISH_STD_L_nm = cfg.TOP_DISH_STD_L_nm
        TOP_DISH_STD_T_nm = cfg.TOP_DISH_STD_T_nm
        TOP_DISH_STD_E_nm = cfg.TOP_DISH_STD_E_nm
        BOT_DISH_MEAN_nm  = cfg.BOT_DISH_MEAN_nm
        BOT_DISH_STD_L_nm = cfg.BOT_DISH_STD_L_nm
        BOT_DISH_STD_T_nm = cfg.BOT_DISH_STD_T_nm
        BOT_DISH_STD_E_nm = cfg.BOT_DISH_STD_E_nm

        block_size_r = cfg.TL_um // cfg.PITCH_r_um
        block_size_c = cfg.TL_um // cfg.PITCH_c_um
        block_idx = assign_pads_to_blocks(PAD_ARR_ROW, PAD_ARR_COL, block_size_r, block_size_c)

        mu        = TOP_DISH_MEAN_nm + BOT_DISH_MEAN_nm
        sigma_L   = np.sqrt(TOP_DISH_STD_L_nm**2 + BOT_DISH_STD_L_nm**2)
        sigma_T   = np.sqrt(TOP_DISH_STD_T_nm**2 + BOT_DISH_STD_T_nm**2)
        sigma_eps = np.sqrt(TOP_DISH_STD_E_nm**2 + BOT_DISH_STD_E_nm**2)

        stress_yield_array = np.full(interface.num_dies, np.nan, dtype=np.float64)

        # ---- 1) Build radial dishing LUT once (no pad coords needed) ----
        radial_lut_array = debond_dishing_bounds_calculator(
            cfg, print_radial_lut=True, plot_radial_lut_flag=True,
        )  # columns: [r_um, D_sio2_nm, D_cu_nm]

        # ---- 2) Group dies by radial layer ----
        radial_bin_um = max(cfg.DIE_W_um, cfg.DIE_L_um)
        radial_info = interface.get_die_radial_layers(radius_bin_um=radial_bin_um)
        die_groups = radial_info['layers']
        # print(
        #     "[{}] Radial-layer acceleration: {} dies -> {} representative groups".format(
        #         interface_name, interface.num_dies, len(die_groups),
        #     )
        # )

        # ---- 3) Vertex-distance yield cache ----
        #   key  = (D_sio2_min, D_sio2_max, D_cu_min, D_cu_max) rounded to 0.1 nm
        #   value = computed die yield
        yield_cache = {}    # type: dict[tuple, float]
        cache_hits  = 0
        base_pad_xy = interface.base_pad_coords     # (N_pads, 2) um

        for grp in die_groups:
            die_ind = int(grp['representative_index'])
            die = interface.die_list[die_ind]

            # Build cache key from die vertex distances → LUT dishing values
            vkey = _vertex_distance_key(die, radial_lut_array, decimals=1)

            if vkey in yield_cache:
                stress_die_yield = yield_cache[vkey]
                cache_hits += 1
            else:
                # Cache miss: query per-pad dishing bounds via coords API
                die_pad_coords = base_pad_xy + die.die_center   # (N_pads, 2) um
                start_time = time.time()
                pad_dishing_bound_array = debond_dishing_bounds_calculator_coords(
                    cfg, die_pad_coords,
                )  # (N_pads, 2): each row sorted (D_low_nm, D_high_nm)
                # print(
                #     "Pad dishing lookup for die {} (layer {}, count {}): {:.2f}s".format(
                #         die_ind, grp['layer_id'], grp['count'],
                #         time.time() - start_time,
                #     )
                # )

                # Derive survival bounds (same convention as original code)
                upper_limits = -pad_dishing_bound_array[:, 0] * 2
                lower_limits = -pad_dishing_bound_array[:, 1] * 2

                time_before = time.time()
                stress_die_yield = cu_recess_die_yield_spatial(
                    mu=mu,
                    a=lower_limits,
                    b=upper_limits,
                    sigma_L=sigma_L,
                    sigma_T=sigma_T,
                    sigma_eps=sigma_eps,
                    block_indices=block_idx,
                    g=0.0,
                )
                # print(
                #     "GH yield for die {} (layer {}, count {}): {:.6f}  [{:.2f}s]".format(
                #         die_ind, grp['layer_id'], grp['count'],
                #         stress_die_yield, time.time() - time_before,
                #     )
                # )
                yield_cache[vkey] = stress_die_yield

            # Broadcast to all dies in this radial group
            stress_yield_array[grp['indices']] = stress_die_yield

        # print(
        #     "[{}] Vertex-distance cache: {} groups, {} hits, {} unique GH evaluations".format(
        #         interface_name, len(die_groups), cache_hits, len(yield_cache),
        #     )
        # )
        waf_stack.die_yield_list_per_interface_dict[interface_name]['mechanical'] = stress_yield_array