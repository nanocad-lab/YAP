# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import log_ndtr


def _require_cfg_attr(cfg, name: str):
    """Return cfg.name or raise a clear error if the attribute is missing."""
    if cfg is None or not hasattr(cfg, name):
        raise AttributeError(f"cfg must provide '{name}' for ESD calculation.")
    return getattr(cfg, name)


def _resolve_float_param(value: Optional[float], *, cfg, attr_name: str) -> float:
    """Prefer an explicit argument; otherwise read the value from cfg."""
    if value is not None:
        return float(value)
    return float(_require_cfg_attr(cfg, attr_name))


def _resolve_int_param(value: Optional[int], *, cfg, attr_name: str) -> int:
    """Prefer an explicit argument; otherwise read the value from cfg."""
    if value is not None:
        return int(value)
    return int(_require_cfg_attr(cfg, attr_name))


def _resolve_pad_size_um(pad_size_um: Optional[float], *, cfg) -> float:
    """Resolve pad size from an explicit argument or cfg.PAD_TOP_R_um."""
    if pad_size_um is not None:
        return float(pad_size_um)
    return 2.0 * float(_require_cfg_attr(cfg, "PAD_TOP_R_um"))


def _resolve_pad_pitch_um(pad_pitch_um: Optional[float], *, cfg) -> float:
    """Resolve pad pitch from an explicit argument or cfg.PITCH_r_um."""
    if pad_pitch_um is not None:
        return float(pad_pitch_um)
    return float(_require_cfg_attr(cfg, "PITCH_r_um"))


def _resolve_z_top_um(z_top_um: Optional[float], *, cfg) -> float:
    """Resolve the initial wafer-to-wafer separation used by the gap model."""
    if z_top_um is not None:
        return float(z_top_um)
    for attr_name in ("ESD_Z_TOP_UM", "Z_TOP_UM", "z_top_um"):
        if cfg is not None and hasattr(cfg, attr_name):
            return float(getattr(cfg, attr_name))
    return 100.0


def _resolve_voltage_range(cfg) -> Tuple[float, float]:
    """Return the charging-voltage range [V] from cfg."""
    return (
        float(_require_cfg_attr(cfg, "V_MIN_V")),
        float(_require_cfg_attr(cfg, "V_MAX_V")),
    )


def _resolve_failure_model_params(cfg) -> Tuple[float, float, float]:
    """Return the Weibull failure-model parameters from cfg."""
    return (
        float(_require_cfg_attr(cfg, "WEIBULL_K")),
        float(_require_cfg_attr(cfg, "WEIBULL_LAMBDA")),
        float(_require_cfg_attr(cfg, "CUTOFF_MIN_A")),
    )


def _resolve_calc_param(cfg, attr_name: str, default):
    """Return an optional numerical-control parameter from cfg or a default."""
    return getattr(cfg, attr_name, default)


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


def _compute_p_fail_for_die(
    top_wafer_radius_um: float,
    v_chg: float,
    *,
    weibull_k: float,
    weibull_lambda: float,
    cutoff_min_a: float,
) -> float:
    """Return the wafer-level failure probability for a sampled charging voltage."""
    area_mm2 = (float(top_wafer_radius_um) * 1e-3) ** 2 * math.pi
    i_peak = _ipeak_from_die_voltage(area_mm2, float(v_chg))
    return _fail_prob_single(i_peak, float(weibull_k), float(weibull_lambda), float(cutoff_min_a))


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

    plateau_v = 337.0                          # Paschen plateau voltage [V]
    small_gap_slope = 97.0                  # Slope of the small-gap Paschen curve [V/um]
    plateau_upper_gap_um = 7.0              # Upper gap limit of the Paschen plateau [um]

    if v_chg < plateau_v:
        return v_chg / small_gap_slope

    a = 2.48                                  # Coefficient of the linear term in the large-gap Paschen curve [V/um]
    b = 58.0                                # Coefficient of the sqrt term in the large-gap Paschen curve [V/sqrt(um)]
    c = 170.0 - v_chg                           # Constant term in the large-gap Paschen curve [V]
    disc = b * b - 4.0 * a * c                     # Discriminant of the quadratic equation for the large-gap Paschen curve
    if disc <= 0.0:
        return plateau_upper_gap_um

    root = (-b + math.sqrt(disc)) / (2.0 * a)
    if root <= 0.0:
        return plateau_upper_gap_um
    return max(plateau_upper_gap_um, root * root)


def _legendre_quadrature_interval(q: int, low: float, high: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes and weights over [low, high]."""
    x, w = leggauss(int(q))
    g = 0.5 * (x + 1.0) * (high - low) + low
    wg = 0.5 * (high - low) * w
    return g.astype(np.float64), wg.astype(np.float64)


def _deterministic_contact_limit_um(
    *,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    tilt_x_deg: float,
    tilt_y_deg: float,
    z_top_um: float,
) -> np.ndarray:
    """
    Return the deterministic contact limit C_i [um] for each pad.

    The exact simulator compares the minimum gap at the lowest pad corner. For the
    analytical calculator we keep that same deterministic corner term, but approximate
    the random top+bottom dishing contribution with a shared Gaussian H_i = T_i + B_i.
    """
    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    a, b, _ = _z_linear_coeffs(tilt_x_deg, tilt_y_deg)
    half_pad_um = 0.5 * float(pad_size_um)
    corner_drop_um = half_pad_um * (abs(float(a)) + abs(float(b)))
    return (
        float(z_top_um)
        + float(a) * pad_coords_um[:, 0]
        + float(b) * pad_coords_um[:, 1]
        - float(corner_drop_um)
    ).astype(np.float64)


def _fixed_tilt_probability_map_with_arcing(
    *,
    contact_limit_um: np.ndarray,
    mu_h_um: float,
    sigma_h_um: float,
    arc_distance_um: float,
    quadrature_points: int,
    tail_sigma: float,
    chunk_size: int,
    fill_residual_uniformly: bool,
) -> np.ndarray:
    """
    Return the per-pad first-touch probability map for fixed tilt and fixed voltage.

    This is the analytical counterpart to the Monte Carlo path. It follows the demo's
    exact fixed-tilt minimum-gap integral, while treating arcing as a pad-eligibility
    threshold on H_i = T_i + B_i.
    """
    contact_limit_um = np.asarray(contact_limit_um, dtype=np.float64).reshape(-1)
    pad_count = contact_limit_um.size
    if pad_count <= 0:
        return np.zeros((0,), dtype=np.float64)
    if sigma_h_um <= 0.0:
        raise ValueError("Combined dishing sigma must be positive for analytical ESD yield calculation.")

    mean_gap_um = contact_limit_um - float(arc_distance_um) - float(mu_h_um)
    low = float(np.min(mean_gap_um) - float(tail_sigma) * float(sigma_h_um))
    high = float(np.max(contact_limit_um))
    if high <= low:
        high = float(np.max(mean_gap_um) + float(tail_sigma) * float(sigma_h_um))

    g_nodes, g_weights = _legendre_quadrature_interval(int(quadrature_points), low, high)
    prob = np.zeros((pad_count,), dtype=np.float64)

    inactive_log_prob = float(log_ndtr((float(-arc_distance_um) - float(mu_h_um)) / float(sigma_h_um)))
    log_norm = -math.log(float(sigma_h_um)) - 0.5 * math.log(2.0 * math.pi)

    for g, w in zip(g_nodes, g_weights):
        valid_mask = g <= contact_limit_um
        if not np.any(valid_mask):
            continue

        log_survival = np.where(
            valid_mask,
            log_ndtr((mean_gap_um - float(g)) / float(sigma_h_um)),
            inactive_log_prob,
        )
        total_log_survival = float(np.sum(log_survival))
        logw = math.log(float(w))

        for start in range(0, pad_count, int(chunk_size)):
            end = min(start + int(chunk_size), pad_count)
            local_valid = valid_mask[start:end]
            if not np.any(local_valid):
                continue

            local_mean = mean_gap_um[start:end]
            t = (float(g) - local_mean) / float(sigma_h_um)
            logf = -0.5 * t * t + log_norm
            log_integrand = logw + logf + total_log_survival - log_survival[start:end]
            prob[start:end][local_valid] += np.exp(log_integrand[local_valid])

    prob_sum = float(np.sum(prob))
    if prob_sum <= 0.0:
        prob.fill(1.0 / float(pad_count))
        return prob

    if prob_sum < 1.0 and fill_residual_uniformly:
        prob += (1.0 - prob_sum) / float(pad_count)
        return prob

    prob /= prob_sum
    return prob


def _plot_probability_over_pads_with_pitch(
    pad_coords_um: np.ndarray,
    prob_vec: np.ndarray,
    *,
    pitch_um: float,
    wafer_radius_um: float,
    title: str,
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
        rect = Rectangle((x - half_pix, y - half_pix), 2.0 * half_pix, 2.0 * half_pix, linewidth=0.0)
        rect.set_facecolor(plt.cm.viridis(prob / norm_max))
        rect.set_edgecolor("none")
        ax.add_patch(rect)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-float(wafer_radius_um), float(wafer_radius_um))
    ax.set_ylim(-float(wafer_radius_um), float(wafer_radius_um))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("x (um), center at 0")
    ax.set_ylabel("y (um), top is smaller")

    sm = mpl.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(vmin=0.0, vmax=norm_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Risk = P_first-touch * p_fail_single")
    return fig


def pad_esd_yield_map_generator(
    *,
    cfg,
    pad_coords_um: np.ndarray,
    pad_size_um: Optional[float] = None,
    pad_pitch_um: Optional[float] = None,
    top_wafer_radius_um: Optional[float] = None,
    n_tilts: Optional[int] = None,
    n_dishes: Optional[int] = None,
    tilt_x_mean_deg: Optional[float] = None,
    tilt_x_std_deg: Optional[float] = None,
    tilt_y_mean_deg: Optional[float] = None,
    tilt_y_std_deg: Optional[float] = None,
    top_dish_mean_nm: Optional[float] = None,
    top_dish_std_nm: Optional[float] = None,
    bot_dish_mean_nm: Optional[float] = None,
    bot_dish_std_nm: Optional[float] = None,
    z_top_um: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[plt.Figure], float]:
    """
    Return the per-pad ESD yield map using the analytical minimum-gap method.

    Output matches the old Monte Carlo generator:
      (valid_pad_yield_map_vec, fig, p_fail_avg)
    """
    pad_size_um = _resolve_pad_size_um(pad_size_um, cfg=cfg)
    pad_pitch_um = _resolve_pad_pitch_um(pad_pitch_um, cfg=cfg)
    top_wafer_radius_um = _resolve_float_param(top_wafer_radius_um, cfg=cfg, attr_name="WAF_R_um")
    n_tilts = _resolve_int_param(n_tilts, cfg=cfg, attr_name="n_tilts_samples")
    n_dishes = _resolve_int_param(n_dishes, cfg=cfg, attr_name="n_dishes_samples")
    tilt_x_mean_deg = _resolve_float_param(tilt_x_mean_deg, cfg=cfg, attr_name="TILT_X_MEAN_DEG")
    tilt_x_std_deg = _resolve_float_param(tilt_x_std_deg, cfg=cfg, attr_name="TILT_X_STD_DEG")
    tilt_y_mean_deg = _resolve_float_param(tilt_y_mean_deg, cfg=cfg, attr_name="TILT_Y_MEAN_DEG")
    tilt_y_std_deg = _resolve_float_param(tilt_y_std_deg, cfg=cfg, attr_name="TILT_Y_STD_DEG")
    top_dish_mean_nm = _resolve_float_param(top_dish_mean_nm, cfg=cfg, attr_name="TOP_DISH_MEAN_nm")
    top_dish_std_nm = _resolve_float_param(top_dish_std_nm, cfg=cfg, attr_name="TOP_DISH_STD_nm")
    bot_dish_mean_nm = _resolve_float_param(bot_dish_mean_nm, cfg=cfg, attr_name="BOT_DISH_MEAN_nm")
    bot_dish_std_nm = _resolve_float_param(bot_dish_std_nm, cfg=cfg, attr_name="BOT_DISH_STD_nm")
    z_top_um = _resolve_z_top_um(z_top_um, cfg=cfg)
    v_min_v, v_max_v = _resolve_voltage_range(cfg)
    weibull_k, weibull_lambda, cutoff_min_a = _resolve_failure_model_params(cfg)

    quadrature_points = int(_resolve_calc_param(cfg, "ESD_ANALYTICAL_INNER_Q", 48))
    outer_qx = int(_resolve_calc_param(cfg, "ESD_ANALYTICAL_OUTER_QX", 5))
    outer_qy = int(_resolve_calc_param(cfg, "ESD_ANALYTICAL_OUTER_QY", 5))
    voltage_q = int(_resolve_calc_param(cfg, "ESD_ANALYTICAL_VOLTAGE_Q", 5))
    tail_sigma = float(_resolve_calc_param(cfg, "ESD_ANALYTICAL_TAIL_SIGMA", 8.0))
    chunk_size = int(_resolve_calc_param(cfg, "ESD_ANALYTICAL_CHUNK_SIZE", 100000))
    fill_residual_uniformly = bool(_resolve_calc_param(cfg, "ESD_ANALYTICAL_FILL_RESIDUAL_UNIFORMLY", True))
    verbose = bool(_resolve_calc_param(cfg, "verbose", False))

    pad_coords_um = np.asarray(pad_coords_um, dtype=np.float64)
    if pad_coords_um.ndim != 2 or pad_coords_um.shape[1] != 2:
        raise ValueError("pad_coords_um must have shape (n_pads, 2).")
    pad_count = pad_coords_um.shape[0]
    active_pad_count = pad_count

    if active_pad_count <= 0:
        raise ValueError("pad_coords_um is empty; analytical ESD yield calculation needs at least one pad.")
    if n_tilts <= 0 or n_dishes <= 0:
        raise ValueError("n_tilts_samples and n_dishes_samples must be positive.")

    mu_h_um = (float(top_dish_mean_nm) + float(bot_dish_mean_nm)) * 1e-3
    sigma_h_um = math.sqrt(max(float(top_dish_std_nm), 0.0) ** 2 + max(float(bot_dish_std_nm), 0.0) ** 2) * 1e-3
    if sigma_h_um <= 0.0:
        raise ValueError("Combined dishing sigma is zero. Analytical ESD yield requires positive variation.")

    x_nodes, x_weights = hermgauss(outer_qx)
    y_nodes, y_weights = hermgauss(outer_qy)
    v_nodes, v_weights = _legendre_quadrature_interval(voltage_q, float(v_min_v), float(v_max_v))
    voltage_norm = float(v_max_v) - float(v_min_v)
    if voltage_norm <= 0.0:
        raise ValueError("cfg.V_MAX_V must be greater than cfg.V_MIN_V.")

    total_cases = int(outer_qx) * int(outer_qy) * int(voltage_q)
    case_id = 0

    risk_active = np.zeros((active_pad_count,), dtype=np.float64)
    p_fail_avg = 0.0

    for v_chg, v_weight in zip(v_nodes, v_weights):
        arc_distance_um = _arc_distance_um_from_voltage(float(v_chg))
        p_fail_v = _compute_p_fail_for_die(
            top_wafer_radius_um,
            float(v_chg),
            weibull_k=weibull_k,
            weibull_lambda=weibull_lambda,
            cutoff_min_a=cutoff_min_a,
        )
        p_fail_avg += (float(v_weight) / voltage_norm) * float(p_fail_v)

        prob_v = np.zeros((active_pad_count,), dtype=np.float64)
        total_outer_weight = 0.0

        for xa, wa in zip(x_nodes, x_weights):
            theta_x_deg = float(tilt_x_mean_deg) + math.sqrt(2.0) * float(tilt_x_std_deg) * float(xa)

            for yb, wb in zip(y_nodes, y_weights):
                theta_y_deg = float(tilt_y_mean_deg) + math.sqrt(2.0) * float(tilt_y_std_deg) * float(yb)
                outer_coeff = float(wa * wb / math.pi)

                contact_limit_um = _deterministic_contact_limit_um(
                    pad_coords_um=pad_coords_um,
                    pad_size_um=pad_size_um,
                    tilt_x_deg=theta_x_deg,
                    tilt_y_deg=theta_y_deg,
                    z_top_um=z_top_um,
                )
                prob_case = _fixed_tilt_probability_map_with_arcing(
                    contact_limit_um=contact_limit_um,
                    mu_h_um=mu_h_um,
                    sigma_h_um=sigma_h_um,
                    arc_distance_um=arc_distance_um,
                    quadrature_points=quadrature_points,
                    tail_sigma=tail_sigma,
                    chunk_size=chunk_size,
                    fill_residual_uniformly=fill_residual_uniformly,
                )
                prob_v += outer_coeff * prob_case
                total_outer_weight += outer_coeff
                case_id += 1

                if verbose:
                    print(
                        f"[ESD analytical] {case_id}/{total_cases} | "
                        f"V={float(v_chg):.4f} V | "
                        f"theta_x={theta_x_deg:.3e} deg | "
                        f"theta_y={theta_y_deg:.3e} deg",
                        end="\r",
                        flush=True,
                    )

        if total_outer_weight > 0.0:
            prob_v /= total_outer_weight
        s = float(np.sum(prob_v))
        if s > 0.0:
            prob_v /= s
        else:
            prob_v.fill(1.0 / float(active_pad_count))

        risk_active += (float(v_weight) / voltage_norm) * float(p_fail_v) * prob_v

    if verbose:
        print()

    risk_vec = risk_active.copy()
    valid_pad_yield_map_vec = 1.0 - risk_vec

    if bool(getattr(cfg, "plot_flag", False)):
        fig = _plot_probability_over_pads_with_pitch(
            pad_coords_um=pad_coords_um,
            prob_vec=risk_vec,
            pitch_um=pad_pitch_um,
            wafer_radius_um=top_wafer_radius_um,
            title="Risk Pad Map = E[1(first-touch pad) * p_fail(V)], analytical",
        )
    else:
        fig = None

    return valid_pad_yield_map_vec, fig, float(p_fail_avg)
