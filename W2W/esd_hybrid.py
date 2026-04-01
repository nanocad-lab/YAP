# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Global ESD model parameters.
V_MIN_V = 0.0
V_MAX_V = 5.0

WEIBULL_K = 4.44985
WEIBULL_LAMBDA = 0.0621816
CUTOFF_MIN_A = 0.0

# Unit conversion.
NM_TO_UM = 1e-3

# Default wafer and pad geometry.
TOP_WAFER_RADIUS_UM: float = 75_000.0

PAD_SIZE_UM: float = 50.0
PAD_PITCH_UM: float = 100.0
PAD_COORDS_UM: List[Tuple[float, float]] = []

# Default dishing distributions.
TOP_DISH_MEAN_NM: float = -4.0
TOP_DISH_STD_NM: float = 2.5
BOT_DISH_MEAN_NM: float = -4.0
BOT_DISH_STD_NM: float = 2.5

# Default tilt distributions.
TILT_X_MEAN_DEG = 0.000
TILT_X_STD_DEG = 0.01
TILT_Y_MEAN_DEG = 0.000
TILT_Y_STD_DEG = 0.01

# Default Monte Carlo settings.
N_TILTS = 5
N_DISHES = 5
BASE_SEED = int(np.random.default_rng().integers(0, 2**32 - 1, dtype=np.uint32))


def _z_linear_coeffs(ax_deg: float, ay_deg: float) -> Tuple[float, float, float]:
    """Return the plane coefficients for R = Ry(ay) @ Rx(ax)."""
    ax = np.deg2rad(float(ax_deg))
    ay = np.deg2rad(float(ay_deg))
    ca, sa = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    a = -sy
    b = cy * sa
    c = cy * ca
    return float(a), float(b), float(c)


def _ipeak_from_die_voltage(area_mm2: float, v_chg: float) -> float:
    """Empirical peak-current model."""
    return 0.0045 * (float(area_mm2) ** 0.35) * math.sqrt(float(v_chg))


def _weibull_cdf(current_a: float, k: float, lam: float) -> float:
    """Weibull cumulative distribution function."""
    current_a = max(current_a, 1e-12)
    return max(0.0, min(1.0, 1.0 - math.exp(-((current_a / lam) ** k))))


def _fail_prob_single(current_a: float, k: float, lam: float, cutoff_a: float) -> float:
    """Return the single-event failure probability."""
    if current_a < cutoff_a:
        return 0.0
    return _weibull_cdf(current_a, k, lam)


def _compute_p_fail_for_die(top_wafer_radius_um: float, v_chg: float) -> float:
    """Return the wafer-level failure probability for a sampled charging voltage."""
    area_mm2 = (float(top_wafer_radius_um) * 1e-3) ** 2 * math.pi
    i_peak = _ipeak_from_die_voltage(area_mm2, float(v_chg))
    return _fail_prob_single(i_peak, float(WEIBULL_K), float(WEIBULL_LAMBDA), float(CUTOFF_MIN_A))


def _arc_distance_um_from_voltage(v_chg: float) -> float:
    """
    Return the maximum air-gap distance [um] that can discharge at voltage v_chg [V].

    Modified Paschen curve:
      V = 97 d                       for d < 3.5 um
      V = 337                        for 3.5 um < d < 7 um
      V = 170 + 2.48 d + 58 sqrt(d) for d > 7 um
    """
    v_chg = max(0.0, float(v_chg))
    if v_chg <= 0.0:
        return 0.0

    plateau_v = 337.0
    small_gap_slope = 97.0
    plateau_upper_gap_um = 7.0

    if v_chg < plateau_v:
        return v_chg / small_gap_slope

    a = 2.48
    b = 58.0
    c = 170.0 - v_chg
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return plateau_upper_gap_um

    root = (-b + math.sqrt(disc)) / (2.0 * a)
    if root <= 0.0:
        return plateau_upper_gap_um
    return max(plateau_upper_gap_um, root * root)


def _prepare_pad_geometry_cache(
    pad_coords_um: np.ndarray,
    pad_size_um: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return reusable pad geometry arrays for repeated gap evaluations."""
    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    return pad_coords_um[:, 0], pad_coords_um[:, 1], 0.5 * float(pad_size_um)


def _prepare_die_geometry_cache(
    top_die_w_um: float,
    top_die_h_um: float,
) -> Tuple[float, float]:
    """Return reusable half-size values for die corner comparisons."""
    return 0.5 * float(top_die_w_um), 0.5 * float(top_die_h_um)


def _active_pad_ids_from_bitmap(
    pad_coords_um: np.ndarray,
    dummy_pad_bitmap: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the original pad array plus the ids of pads that are not dummy pads."""
    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    if pad_coords_um.ndim != 2 or pad_coords_um.shape[1] != 2:
        raise ValueError("pad_coords_um must have shape (n_pads, 2).")

    dummy_pad_bitmap = np.asarray(dummy_pad_bitmap, dtype=bool).reshape(-1)
    if pad_coords_um.shape[0] != dummy_pad_bitmap.shape[0]:
        raise ValueError("pad_coords_um and dummy_pad_bitmap must have the same length.")

    active_ids = np.flatnonzero(~dummy_pad_bitmap)
    if active_ids.size <= 0:
        raise ValueError("dummy_pad_bitmap masks out all pads.")
    return pad_coords_um, active_ids


def _candidate_pad_ids(
    top_dish_um_raw: np.ndarray,
    bot_dish_um: np.ndarray,
    arc_distance_um: float,
) -> np.ndarray:
    """Return pad ids that are close enough to enter first-touch competition."""
    arc_margin_um = max(0.0, float(arc_distance_um))
    return np.where((top_dish_um_raw + bot_dish_um) >= (-arc_margin_um))[0]


def _square_pad_min_gap_vec(
    *,
    cx_um: np.ndarray,
    cy_um: np.ndarray,
    half_pad_um: float,
    top_dish_um_raw: np.ndarray,
    bot_dish_um: np.ndarray,
    z_top_um: float,
    a: float,
    b: float,
    c: float,
    arc_distance_um: float,
) -> np.ndarray:
    """
    Return the exact minimum gap for each axis-aligned square pad.

    With the current model, z is linear in x/y, so the minimum over the four
    pad corners can be written analytically instead of expanding all corners.
    """
    corner_drop_um = float(half_pad_um) * (abs(float(a)) + abs(float(b)))
    return (
        float(z_top_um)
        + float(a) * np.asarray(cx_um, dtype=np.float64)
        + float(b) * np.asarray(cy_um, dtype=np.float64)
        + float(c) * (-np.asarray(top_dish_um_raw, dtype=np.float64))
        - np.asarray(bot_dish_um, dtype=np.float64)
        - max(0.0, float(arc_distance_um))
        - corner_drop_um
    )


def _rotate_and_min_choice(
    *,
    pad_coords_um: np.ndarray,
    pad_ids: np.ndarray,
    pad_size_um: float,
    top_die_w_um: float,
    top_die_h_um: float,
    top_dish_um_raw: np.ndarray,
    bot_dish_um: np.ndarray,
    tilt_x_deg: float,
    tilt_y_deg: float,
    z_top_um: float,
    rng_pick: np.random.Generator,
    pad_x_um: Optional[np.ndarray] = None,
    pad_y_um: Optional[np.ndarray] = None,
    half_pad_um: Optional[float] = None,
    half_die_w_um: Optional[float] = None,
    half_die_h_um: Optional[float] = None,
    arc_distance_um: float = 0.0,
    atol: float = 1e-12,
) -> Tuple[Optional[int], bool, float]:
    """
    Apply tilt to the die and candidate pads and return the minimum-gap winner.

    The current geometry model is affine in x/y, so each square pad can use an
    exact analytical minimum over its four corners without explicitly storing
    those corner coordinates.
    """
    if pad_x_um is None or pad_y_um is None or half_pad_um is None:
        pad_x_um, pad_y_um, half_pad_um = _prepare_pad_geometry_cache(pad_coords_um, pad_size_um)
    if half_die_w_um is None or half_die_h_um is None:
        half_die_w_um, half_die_h_um = _prepare_die_geometry_cache(top_die_w_um, top_die_h_um)

    a, b, c = _z_linear_coeffs(tilt_x_deg, tilt_y_deg)
    die_gap = (
        float(z_top_um)
        - float(half_die_w_um) * abs(float(a))
        - float(half_die_h_um) * abs(float(b))
    )

    if pad_ids.size <= 0:
        return None, True, float(die_gap)

    pad_gaps = _square_pad_min_gap_vec(
        cx_um=pad_x_um[pad_ids],
        cy_um=pad_y_um[pad_ids],
        half_pad_um=float(half_pad_um),
        top_dish_um_raw=top_dish_um_raw[pad_ids],
        bot_dish_um=bot_dish_um[pad_ids],
        z_top_um=z_top_um,
        a=a,
        b=b,
        c=c,
        arc_distance_um=arc_distance_um,
    )

    min_gap = min(float(die_gap), float(np.min(pad_gaps)))
    is_best = np.isclose(pad_gaps, min_gap, rtol=0.0, atol=atol)
    if np.any(is_best):
        candidate_pad_ids = pad_ids[is_best]
        pick = int(rng_pick.integers(0, candidate_pad_ids.size))
        return int(candidate_pad_ids[pick]), False, float(min_gap)
    return None, True, float(min_gap)


def _best_pad_among_all_pads(
    *,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    top_dish_um_raw: np.ndarray,
    bot_dish_um: np.ndarray,
    tilt_x_deg: float,
    tilt_y_deg: float,
    z_top_um: float,
    rng_pick: np.random.Generator,
    pad_x_um: Optional[np.ndarray] = None,
    pad_y_um: Optional[np.ndarray] = None,
    half_pad_um: Optional[float] = None,
    arc_distance_um: float = 0.0,
    atol_gap: float = 1e-12,
) -> Tuple[int, float]:
    """Choose the minimum-gap pad across all pads, without any candidate mask."""
    pad_count = pad_coords_um.shape[0]
    if pad_count <= 0:
        raise ValueError("pad_coords_um is empty; cannot choose a fallback pad.")

    if pad_x_um is None or pad_y_um is None or half_pad_um is None:
        pad_x_um, pad_y_um, half_pad_um = _prepare_pad_geometry_cache(pad_coords_um, pad_size_um)

    a, b, c = _z_linear_coeffs(tilt_x_deg, tilt_y_deg)
    pad_min_gaps = _square_pad_min_gap_vec(
        cx_um=pad_x_um,
        cy_um=pad_y_um,
        half_pad_um=float(half_pad_um),
        top_dish_um_raw=top_dish_um_raw,
        bot_dish_um=bot_dish_um,
        z_top_um=z_top_um,
        a=a,
        b=b,
        c=c,
        arc_distance_um=arc_distance_um,
    )
    best_gap = float(np.min(pad_min_gaps))
    is_best = np.isclose(pad_min_gaps, best_gap, rtol=0.0, atol=atol_gap)
    best_ids = np.where(is_best)[0]

    if best_ids.size == 1:
        return int(best_ids[0]), best_gap
    pick = int(rng_pick.integers(0, best_ids.size))
    return int(best_ids[pick]), best_gap


def _binary_halving_until_pad(
    *,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    top_die_w_um: float,
    top_die_h_um: float,
    z_top_um: float,
    tilt_x_init_deg: float,
    tilt_y_init_deg: float,
    top_dish_um_raw: np.ndarray,
    bot_dish_um: np.ndarray,
    rng_pick: np.random.Generator,
    arc_distance_um: float = 0.0,
    atol_gap: float = 1e-12,
    atol_tilt_deg: float = 1e-12,
    max_iter_guard: int = 10000,
) -> Tuple[int, float, float, float]:
    """
    Find the first-touch pad.

    The normal path compares die corners and candidate pad corners under the
    current tilt. If the result is still die-only, the tilt is halved until a
    pad appears or the stopping guard is hit. The final fallback always chooses
    the all-pad minimum-gap winner under zero tilt.
    """
    pad_x_um, pad_y_um, half_pad_um = _prepare_pad_geometry_cache(pad_coords_um, pad_size_um)
    half_die_w_um, half_die_h_um = _prepare_die_geometry_cache(top_die_w_um, top_die_h_um)
    candidate_pad_ids = _candidate_pad_ids(top_dish_um_raw, bot_dish_um, arc_distance_um)

    tilt_x = float(tilt_x_init_deg)
    tilt_y = float(tilt_y_init_deg)

    if candidate_pad_ids.size <= 0:
        best_pad, best_gap = _best_pad_among_all_pads(
            pad_coords_um=pad_coords_um,
            pad_size_um=pad_size_um,
            top_dish_um_raw=top_dish_um_raw,
            bot_dish_um=bot_dish_um,
            tilt_x_deg=0.0,
            tilt_y_deg=0.0,
            z_top_um=z_top_um,
            rng_pick=rng_pick,
            pad_x_um=pad_x_um,
            pad_y_um=pad_y_um,
            half_pad_um=half_pad_um,
            arc_distance_um=arc_distance_um,
            atol_gap=atol_gap,
        )
        return best_pad, tilt_x, tilt_y, float(best_gap)

    pad_choice, die_only, min_gap = _rotate_and_min_choice(
        pad_coords_um=pad_coords_um,
        pad_ids=candidate_pad_ids,
        pad_size_um=pad_size_um,
        top_die_w_um=top_die_w_um,
        top_die_h_um=top_die_h_um,
        top_dish_um_raw=top_dish_um_raw,
        bot_dish_um=bot_dish_um,
        tilt_x_deg=tilt_x,
        tilt_y_deg=tilt_y,
        z_top_um=z_top_um,
        rng_pick=rng_pick,
        pad_x_um=pad_x_um,
        pad_y_um=pad_y_um,
        half_pad_um=half_pad_um,
        half_die_w_um=half_die_w_um,
        half_die_h_um=half_die_h_um,
        arc_distance_um=arc_distance_um,
        atol=atol_gap,
    )
    if not die_only:
        return int(pad_choice), tilt_x, tilt_y, float(min_gap)

    iterations = 0
    while die_only:
        tilt_x *= 0.5
        tilt_y *= 0.5
        pad_choice, die_only, min_gap = _rotate_and_min_choice(
            pad_coords_um=pad_coords_um,
            pad_ids=candidate_pad_ids,
            pad_size_um=pad_size_um,
            top_die_w_um=top_die_w_um,
            top_die_h_um=top_die_h_um,
            top_dish_um_raw=top_dish_um_raw,
            bot_dish_um=bot_dish_um,
            tilt_x_deg=tilt_x,
            tilt_y_deg=tilt_y,
            z_top_um=z_top_um,
            rng_pick=rng_pick,
            pad_x_um=pad_x_um,
            pad_y_um=pad_y_um,
            half_pad_um=half_pad_um,
            half_die_w_um=half_die_w_um,
            half_die_h_um=half_die_h_um,
            arc_distance_um=arc_distance_um,
            atol=atol_gap,
        )
        iterations += 1

        if not die_only:
            return int(pad_choice), tilt_x, tilt_y, float(min_gap)

        if (abs(tilt_x) <= atol_tilt_deg and abs(tilt_y) <= atol_tilt_deg) or (iterations >= max_iter_guard):
            best_pad, best_gap = _best_pad_among_all_pads(
                pad_coords_um=pad_coords_um,
                pad_size_um=pad_size_um,
                top_dish_um_raw=top_dish_um_raw,
                bot_dish_um=bot_dish_um,
                tilt_x_deg=0.0,
                tilt_y_deg=0.0,
                z_top_um=z_top_um,
                rng_pick=rng_pick,
                pad_x_um=pad_x_um,
                pad_y_um=pad_y_um,
                half_pad_um=half_pad_um,
                arc_distance_um=arc_distance_um,
                atol_gap=atol_gap,
            )
            return best_pad, tilt_x, tilt_y, float(best_gap)

    best_pad, best_gap = _best_pad_among_all_pads(
        pad_coords_um=pad_coords_um,
        pad_size_um=pad_size_um,
        top_dish_um_raw=top_dish_um_raw,
        bot_dish_um=bot_dish_um,
        tilt_x_deg=0.0,
        tilt_y_deg=0.0,
        z_top_um=z_top_um,
        rng_pick=rng_pick,
        pad_x_um=pad_x_um,
        pad_y_um=pad_y_um,
        half_pad_um=half_pad_um,
        arc_distance_um=arc_distance_um,
        atol_gap=atol_gap,
    )
    return best_pad, tilt_x, tilt_y, float(best_gap)


def pad_esd_yield_map_generator(
    *,
    cfg,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    pad_pitch_um: float,
    top_wafer_radius_um: float,
    n_tilts: int,
    n_dishes: int,
    tilt_x_mean_deg: float,
    tilt_x_std_deg: float,
    tilt_y_mean_deg: float,
    tilt_y_std_deg: float,
    top_dish_mean_nm: float,
    top_dish_std_nm: float,
    bot_dish_mean_nm: float,
    bot_dish_std_nm: float,
    dummy_pad_bitmap: np.ndarray,
    base_seed: int = 20251006,
    z_top_um: float = 100.0,
) -> Tuple[np.ndarray, Optional[plt.Figure], float]:
    """Run Monte Carlo sampling and accumulate a per-pad ESD risk map."""
    _ = cfg

    pad_coords_um, active_ids = _active_pad_ids_from_bitmap(pad_coords_um, dummy_pad_bitmap)
    active_pad_coords_um = pad_coords_um[active_ids]
    pad_count = pad_coords_um.shape[0]
    active_pad_count = active_ids.size
    total_runs = int(n_tilts) * int(n_dishes)
    if total_runs <= 0:
        raise ValueError("n_tilts * n_dishes must be positive.")

    counts_vec = np.zeros((pad_count,), dtype=np.int64)
    risk_accum_vec = np.zeros((pad_count,), dtype=np.float64)
    p_fail_sum = 0.0

    rng_tilt = np.random.default_rng(base_seed ^ 0xC001FEED)
    progress_counter = 0
    wafer_span_um = 2.0 * float(top_wafer_radius_um)

    for tilt_index in range(n_tilts):
        tilt_x0 = float(rng_tilt.normal(tilt_x_mean_deg, tilt_x_std_deg))
        tilt_y0 = float(rng_tilt.normal(tilt_y_mean_deg, tilt_y_std_deg))

        for dish_index in range(n_dishes):
            progress_counter += 1
            if (progress_counter % 1000) == 0 or (progress_counter == total_runs):
                print(
                    f"[ESD Sim] Progress: {progress_counter} / {total_runs} runs completed.",
                    end="\r",
                    flush=True,
                )

            seed = base_seed + (tilt_index * n_dishes + dish_index)
            rng_top = np.random.default_rng(seed ^ 0x9E3779B1)
            rng_bot = np.random.default_rng(seed ^ 0x85EBCA77)
            rng_pick = np.random.default_rng(seed ^ 0xDEADBEEF)
            rng_v = np.random.default_rng(seed ^ 0xC0FFEE11)

            top_dish_um_raw = rng_top.normal(
                loc=float(top_dish_mean_nm) * NM_TO_UM,
                scale=max(float(top_dish_std_nm), 0.0) * NM_TO_UM,
                size=(active_pad_count,),
            ).astype(np.float64)
            bot_dish_um = rng_bot.normal(
                loc=float(bot_dish_mean_nm) * NM_TO_UM,
                scale=max(float(bot_dish_std_nm), 0.0) * NM_TO_UM,
                size=(active_pad_count,),
            ).astype(np.float64)

            v_chg = float(rng_v.uniform(V_MIN_V, V_MAX_V))
            arc_distance_um = _arc_distance_um_from_voltage(v_chg)

            pad_choice_active, _, _, _ = _binary_halving_until_pad(
                pad_coords_um=active_pad_coords_um,
                pad_size_um=pad_size_um,
                top_die_w_um=wafer_span_um,
                top_die_h_um=wafer_span_um,
                z_top_um=z_top_um,
                tilt_x_init_deg=tilt_x0,
                tilt_y_init_deg=tilt_y0,
                top_dish_um_raw=top_dish_um_raw,
                bot_dish_um=bot_dish_um,
                rng_pick=rng_pick,
                arc_distance_um=arc_distance_um,
            )

            p_fail_run = _compute_p_fail_for_die(top_wafer_radius_um, v_chg)
            p_fail_sum += p_fail_run

            if pad_choice_active is not None:
                pad_choice = int(active_ids[int(pad_choice_active)])
                counts_vec[pad_choice] += 1
                risk_accum_vec[pad_choice] += float(p_fail_run)

    print()
    valid_pad_risk_map_vec = risk_accum_vec / float(total_runs)
    p_fail_avg = p_fail_sum / float(total_runs)

    fig = plot_probability_over_pads_with_pitch(
        pad_coords_um=pad_coords_um,
        prob_vec=valid_pad_risk_map_vec,
        pitch_um=pad_pitch_um,
        title="Risk Pad Map = E[1(first-touch pad) * p_fail(V)], V~U[0,5]",
    )
    valid_pad_yield_map_vec = 1.0 - valid_pad_risk_map_vec
    return valid_pad_yield_map_vec, fig, float(p_fail_avg)


def esd_failure_simulator(
    *,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    top_wafer_radius_um: float,
    top_dish_nm_ext: np.ndarray,
    bot_dish_nm_ext: np.ndarray,
    tilt_x_mean_deg: float,
    tilt_x_std_deg: float,
    tilt_y_mean_deg: float,
    tilt_y_std_deg: float,
    dummy_pad_bitmap: np.ndarray,
    base_seed: int = 20251006,
    z_top_um: float = 100.0,
) -> Tuple[Optional[int], bool]:
    """Run a single stochastic experiment and return (first_touch_pad, survive_bool)."""
    pad_coords_um, active_ids = _active_pad_ids_from_bitmap(pad_coords_um, dummy_pad_bitmap)
    active_pad_coords_um = pad_coords_um[active_ids]
    top_dish_nm_ext = np.asarray(top_dish_nm_ext, dtype=np.float64).reshape(-1)
    bot_dish_nm_ext = np.asarray(bot_dish_nm_ext, dtype=np.float64).reshape(-1)

    if not (
        pad_coords_um.shape[0] == top_dish_nm_ext.shape[0] == bot_dish_nm_ext.shape[0]
    ):
        raise ValueError("pad_coords_um, top_dish_nm_ext, and bot_dish_nm_ext must have the same length.")

    rng = np.random.default_rng(base_seed ^ 0xA5A5A5A5)
    rng_pick = np.random.default_rng((base_seed ^ 0xA5A5A5A5) ^ 0xDEADBEEF)

    tilt_x = float(rng.normal(tilt_x_mean_deg, tilt_x_std_deg))
    tilt_y = float(rng.normal(tilt_y_mean_deg, tilt_y_std_deg))
    v_chg = float(rng.uniform(V_MIN_V, V_MAX_V))
    arc_distance_um = _arc_distance_um_from_voltage(v_chg)

    top_dish_um_raw = top_dish_nm_ext[active_ids] * NM_TO_UM
    bot_dish_um = bot_dish_nm_ext[active_ids] * NM_TO_UM
    wafer_span_um = 2.0 * float(top_wafer_radius_um)

    pad_choice_active, _, _, _ = _binary_halving_until_pad(
        pad_coords_um=active_pad_coords_um,
        pad_size_um=pad_size_um,
        top_die_w_um=wafer_span_um,
        top_die_h_um=wafer_span_um,
        z_top_um=z_top_um,
        tilt_x_init_deg=tilt_x,
        tilt_y_init_deg=tilt_y,
        top_dish_um_raw=top_dish_um_raw,
        bot_dish_um=bot_dish_um,
        rng_pick=rng_pick,
        arc_distance_um=arc_distance_um,
    )

    pad_choice = int(active_ids[int(pad_choice_active)]) if pad_choice_active is not None else None
    p_fail_single = _compute_p_fail_for_die(top_wafer_radius_um, v_chg)
    survive_bool = not ((pad_choice is not None) and (float(rng.uniform(0.0, 1.0)) < p_fail_single))
    return pad_choice, survive_bool


def plot_probability_over_pads_with_pitch(
    pad_coords_um: np.ndarray,
    prob_vec: np.ndarray,
    *,
    pitch_um: float,
    title: str = "Pad Selection Probability (squares at pitch)",
) -> plt.Figure:
    """Plot one display square per pad, using pitch as the display square size."""
    fig, ax = plt.subplots()
    try:
        fig.canvas.toolbar_visible = True
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
    except Exception:
        pass

    vmax = float(prob_vec.max()) if prob_vec.size > 0 else 0.0
    norm_max = vmax if vmax > 0.0 else 1.0
    half_pix = 0.5 * float(pitch_um)

    for (x, y), prob in zip(pad_coords_um, prob_vec):
        if prob <= 0.0:
            continue
        rect = Rectangle((x - half_pix, y - half_pix), 2 * half_pix, 2 * half_pix, linewidth=0.0)
        rect.set_facecolor(plt.cm.viridis(prob / norm_max))
        rect.set_edgecolor("none")
        ax.add_patch(rect)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-TOP_WAFER_RADIUS_UM, TOP_WAFER_RADIUS_UM)
    ax.set_ylim(-TOP_WAFER_RADIUS_UM, TOP_WAFER_RADIUS_UM)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("x (um), center at 0")
    ax.set_ylabel("y (um), top is smaller")

    sm = mpl.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(vmin=0.0, vmax=norm_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Risk = P_first-touch * p_fail_single")
    return fig


if __name__ == "__main__":
    if not PAD_COORDS_UM:
        xs = np.arange(-TOP_WAFER_RADIUS_UM + PAD_PITCH_UM * 0.5, TOP_WAFER_RADIUS_UM, PAD_PITCH_UM)
        ys = np.arange(TOP_WAFER_RADIUS_UM - PAD_PITCH_UM * 0.5, -TOP_WAFER_RADIUS_UM, -PAD_PITCH_UM)
        grid_x, grid_y = np.meshgrid(xs, ys)
        PAD_COORDS_UM = list(zip(grid_x.ravel().tolist(), grid_y.ravel().tolist()))

    pad_coords = np.asarray(PAD_COORDS_UM, dtype=np.float64).reshape(-1, 2)
    rng = np.random.default_rng(BASE_SEED ^ 0x13579BDF)
    pad_count = pad_coords.shape[0]
    dummy_pad_bitmap = np.zeros((pad_count,), dtype=bool)
    top_ext_nm = rng.normal(TOP_DISH_MEAN_NM, TOP_DISH_STD_NM, size=pad_count).astype(np.float64)
    bot_ext_nm = rng.normal(BOT_DISH_MEAN_NM, BOT_DISH_STD_NM, size=pad_count).astype(np.float64)

    pad_idx, survive = esd_failure_simulator(
        pad_coords_um=pad_coords,
        pad_size_um=PAD_SIZE_UM,
        top_wafer_radius_um=TOP_WAFER_RADIUS_UM,
        top_dish_nm_ext=top_ext_nm,
        bot_dish_nm_ext=bot_ext_nm,
        tilt_x_mean_deg=TILT_X_MEAN_DEG,
        tilt_x_std_deg=TILT_X_STD_DEG,
        tilt_y_mean_deg=TILT_Y_MEAN_DEG,
        tilt_y_std_deg=TILT_Y_STD_DEG,
        dummy_pad_bitmap=dummy_pad_bitmap,
        base_seed=BASE_SEED,
        z_top_um=100.0,
    )

    print("\nSingle-run demo")
    print(f"first-touch pad index: {pad_idx}")
    print(f"survive? {survive}")
