#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debond.py  (produce dishing intervals from manual coord list)


This version:
- Removes ALL external wafer_info reading logic.
- Caller provides pad global coordinates (in µm) via debond_dishing_bounds_calculator(cfg, coords_um).
- Output: numpy.ndarray (N,2), each row = sorted (D_Cu_nm, D_SiO2_nm).

UPDATED (paper Eq.(27)–(36) pad-scale core):
  Eq.(27)–(36) replace the previous 9-parameter pad-scale model.

INVERSION (FIXED WINDOW, consistent with Eq.(30) singularity):
  - SiO2: search D in [0, 10] nm; if no root -> return 0
  - Cu  : search D in [0, D_contact_max] where D_contact_max = delta_heat/2;
          if no root -> return D_contact_max
  - To avoid phi->0 divergence at the boundary D = D_contact_max, evaluate f(hi)
    at hi_eval = nextafter(D_contact_max, 0).

FAST VERSION (LUT lookup, same window rules):
  - Build LUT once per __init_params(cfg): D_grid -> sigma(D)
      * SiO2 uses Eq.(32) over D in [0, 10] nm
      * Cu   uses Eq.(36) over D in [0, hi_eval] where hi_eval = nextafter(D_contact_max, 0)
  - Enforce monotonicity on LUT curves (numerical robustness)
  - Invert sigma_eff -> D by interpolation (vectorized for all points)

NEW FASTER VERSION (radial dishing LUT):
  - For each fixed wafer/die state (fixed peel_dict + R_m), build one LUT:
      r -> p_global(r) -> sigma_eff(r) -> dishing(r)
  - Per-pad path becomes:
      coords -> r -> np.interp(r, r_lut, D_lut)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np
import matplotlib.pyplot as plt

from roughness_coefficients import get_eff_contact_area_ratio


# =============================================================================
# =============================== PARAMETERS ==================================
# =============================================================================

# ---------- (F) Wafer-layer materials ----------
@dataclass(frozen=True)
class Material:
    name: str
    E_Pa: float
    alpha_perC: float
    nu: float

# ---------- (G) Wafer configs ----------
@dataclass
class LayerMix3:
    mat1: Material; V1: float
    mat2: Material; V2: float
    mat3: Material; V3: float
    t_m: float

@dataclass(frozen=True)
class EqLayer:
    E_Pa: float
    alpha_perC: float
    nu: float
    t_m: float

@dataclass
class WaferConfig:
    top: LayerMix3
    bottom: LayerMix3
    L_m: float
    T_C: float
    T0_C: float


# =============================================================================
# ============================== LUT CACHES ===================================
# =============================================================================
_LUT_READY: bool = False
_LUT_SIO2: dict | None = None
_LUT_CU: dict | None = None

# NEW: radial dishing LUT cache (one per fixed wafer/die state)
_RDISH_LUT: dict | None = None


def __init_params(cfg):
    """
    IMPORTANT:
    - This file keeps the original style: all parameters come from cfg via __init_params(cfg).
    - No hard-coded paper constants inside core; paper coefficients are read from cfg:
        C_HEAT_E, C_HEAT_P, C_COOL_E, C_COOL_P,
        EXP_PHI, BAUSCHINGER, EXP_INVPHI, EXP_AREA,
        KN_DEN_M.
    """
    global PITCH_UM, DIAM_UM, T_ANNEAL_C, T_REF_C, \
           CU_E_GPA, CU_NU, CU_ALPHA_PPM, OX_E_GPA, OX_NU, OX_ALPHA_PPM, \
           SIGMA_Y_MPA, \
           C_HEAT_E, C_HEAT_P, C_COOL_E, C_COOL_P, \
           EXP_PHI, BAUSCHINGER, EXP_INVPHI, EXP_AREA, \
           KN_DEN_M, \
           CRIT_aY2_UM, GC_SIO2_JPM2, GC_CU_JPM2, Effective_Contact_Area, \
           MAT_CU, MAT_SiO2, MAT_Si, \
           WAFER_A, WAFER_B, \
           S_INIT_A_M, S_INIT_B_M, \
           USE_PLOT, \
           _LUT_READY, _LUT_SIO2, _LUT_CU, _RDISH_LUT

    # Reset LUT caches on every init (cfg may change)
    _LUT_READY = False
    _LUT_SIO2 = None
    _LUT_CU = None
    _RDISH_LUT = None

    # ---------- (A) Pad-scale: Geometry & Temps ----------
    if cfg.PAD_ARRANGE_PATTERN == 'checkerboard':
        PITCH_UM = min(
            np.sqrt(cfg.PITCH_r_um ** 2 + cfg.PITCH_c_um ** 2),
            2 * cfg.PITCH_r_um,
            2 * cfg.PITCH_c_um
        )
    else:
        PITCH_UM = cfg.PITCH_r_um

    DIAM_UM    = cfg.PAD_TOP_R_um * 2.0   # pad diameter d [µm]
    T_ANNEAL_C = cfg.T_anl                # anneal temperature [°C]
    T_REF_C    = cfg.T_R                  # reference temperature [°C]

    # ---------- (B) Pad-scale: Material constants ----------
    CU_E_GPA     = cfg.CU_E_GPA
    CU_NU        = cfg.CU_NU
    CU_ALPHA_PPM = cfg.CU_ALPHA_PPM

    OX_E_GPA      = cfg.OX_E_GPA
    OX_NU         = cfg.OX_NU
    OX_ALPHA_PPM  = cfg.OX_ALPHA_PPM

    # ---------- (C) Pad-scale: Yield stress ----------
    SIGMA_Y_MPA = cfg.SIGMA_Y_MPA

    # ---------- (D) Paper Eq.(27)–(36) coefficients (from cfg) ----------
    # Eq.(29)
    C_HEAT_E = float(cfg.C_HEAT_E)
    C_HEAT_P = float(cfg.C_HEAT_P)
    # Eq.(35)
    C_COOL_E = float(cfg.C_COOL_E)
    C_COOL_P = float(cfg.C_COOL_P)
    # Eq.(30)
    EXP_PHI = float(cfg.EXP_PHI)
    # Eq.(33)
    BAUSCHINGER = float(cfg.BAUSCHINGER)
    # Eq.(36)
    EXP_INVPHI = float(cfg.EXP_INVPHI)
    EXP_AREA   = float(cfg.EXP_AREA)
    # Eq.(31) denominator length [m]
    KN_DEN_M = float(cfg.KN_DEN_M)

    # ---------- (E) Critical peeling stress ----------
    CRIT_aY2_UM   = cfg.CRIT_aY2_UM
    GC_SIO2_JPM2  = cfg.GC_SIO2_JPM2
    GC_CU_JPM2    = cfg.GC_CU_JPM2
    Effective_Contact_Area = get_eff_contact_area_ratio(
        Asperity_R_m                = cfg.Asperity_R_m,
        Roughness_sigma_m           = cfg.Roughness_sigma_m,
        eta_s                       = cfg.eta_s,
        Roughness_constant          = cfg.Roughness_constant,
        Adhesion_energy             = cfg.Adhesion_energy,
        Dielectric_Young_modulus_Pa = cfg.Dielectric_Young_modulus_Pa,
    )
    assert 0.0 < Effective_Contact_Area <= 1.0, \
        f"Effective_Contact_Area must be in (0,1], got {Effective_Contact_Area}"

    # ---------- (F) Wafer-layer materials ----------
    MAT_CU   = Material("Cu",   E_Pa=cfg.CU_E_GPA*1e9, alpha_perC=cfg.CU_ALPHA_PPM*1e-6, nu=cfg.CU_NU)
    MAT_SiO2 = Material("SiO2", E_Pa=cfg.OX_E_GPA*1e9, alpha_perC=cfg.OX_ALPHA_PPM*1e-6, nu=cfg.OX_NU)
    MAT_Si   = Material("Si",   E_Pa=cfg.SI_E_GPA*1e9, alpha_perC=cfg.SI_ALPHA_PPM*1e-6, nu=cfg.SI_NU)

    # ---------- (G) Wafer configs ----------
    # Keep original style: defined from cfg; no external I/O here.
    WAFER_A = WaferConfig(
        top=LayerMix3(MAT_CU,  cfg.B_Chip_Cu_V,   MAT_SiO2, cfg.B_Chip_Sio2_V, MAT_Si, cfg.B_Chip_Si_V, cfg.B_Chip_T),
        bottom=LayerMix3(MAT_Si, cfg.B_Sub_Si_V,  MAT_SiO2, cfg.B_Sub_Sio2_V,  MAT_CU, cfg.B_Sub_Cu_V, cfg.B_Sub_T),
        L_m=cfg.WAF_R_um*1e-6, T_C=cfg.T_anl, T0_C=cfg.T_R
    )
    WAFER_B = WaferConfig(
        top=LayerMix3(MAT_CU,  cfg.T_Chip_Cu_V,   MAT_SiO2, cfg.T_Chip_Sio2_V, MAT_Si, cfg.T_Chip_Si_V, cfg.T_Chip_T),
        bottom=LayerMix3(MAT_Si, cfg.T_Sub_Si_V,  MAT_SiO2, cfg.T_Sub_Sio2_V,  MAT_CU, cfg.T_Sub_Cu_V, cfg.T_Sub_T),
        L_m=cfg.WAF_R_um*1e-6, T_C=cfg.T_anl, T0_C=cfg.T_R
    )

    # ---------- (H) Pre-anneal warpages ----------
    S_INIT_A_M = cfg.S_INIT_A_M
    S_INIT_B_M = cfg.S_INIT_B_M

    # ---------- (J) Optional plotting ----------
    USE_PLOT = False


# =============================================================================
# ============================== PAD-SCALE CORE ===============================
# =============================================================================

def _geom_areas(p_um: float, d_um: float) -> Tuple[float, float, float]:
    """Return (A_cell, A_cu, A_ox) in m^2."""
    p = float(p_um) * 1e-6
    d = float(d_um) * 1e-6
    A_cell = p**2
    A_cu   = math.pi * d**2 / 4.0
    A_ox   = A_cell - A_cu
    if A_ox <= 0.0:
        raise ValueError("A_ox<=0, check PITCH_UM and DIAM_UM values.")
    return A_cell, A_cu, A_ox

# -------- Paper Eq.(27)–(36) --------

def _sigma_t_thermal_Pa() -> float:
    """Eq.(27): sigma_t in Pa."""
    dT = float(T_ANNEAL_C - T_REF_C)
    E  = float(CU_E_GPA) * 1e9
    nu = float(CU_NU)
    dalpha = (float(CU_ALPHA_PPM) - float(OX_ALPHA_PPM)) * 1e-6
    return (E / (1.0 - nu)) * dalpha * dT

def _split_sigma_ep_heat_paper(sigma_t_Pa: float) -> Tuple[float, float]:
    """Eq.(28) for heat: sigma_e=min(sigma_t, sigma_y), sigma_p=max(sigma_t-sigma_y,0)."""
    sigma_y_Pa = float(SIGMA_Y_MPA) * 1e6
    sigma_e = min(float(sigma_t_Pa), sigma_y_Pa)
    sigma_p = max(float(sigma_t_Pa) - sigma_y_Pa, 0.0)
    return sigma_e, sigma_p

def _delta_heat_m(sigma_e_Pa: float, sigma_p_Pa: float) -> float:
    """Eq.(29): delta_heat in meters."""
    E  = float(CU_E_GPA) * 1e9
    nu = float(CU_NU)
    return (4.0 * nu / E) * (float(C_HEAT_E) * float(sigma_e_Pa) + float(C_HEAT_P) * float(sigma_p_Pa))

def _phi_contact(delta_heat_m_val: float, D_nm: float) -> float:
    """
    Eq.(30): phi = clip(((delta_heat-2D)/(2D))^EXP_PHI, 0, 1)
    with safe guards for D<=0 and numer<=0.
    """
    D_m = float(D_nm) * 1e-9
    if D_m <= 0.0:
        return 1.0
    numer = float(delta_heat_m_val) - 2.0 * D_m
    if numer <= 0.0:
        return 0.0
    x = numer / (2.0 * D_m)
    val = x ** float(EXP_PHI)
    return float(max(0.0, min(1.0, val)))

def _k_n_Pa_per_m() -> float:
    """Eq.(31): k_n in Pa/m."""
    E  = float(CU_E_GPA) * 1e9
    nu = float(CU_NU)
    return (2.0 * E) / (float(KN_DEN_M) * (1.0 - nu))

def _sigma_peel_sio2_paper_MPa(D_nm: float) -> dict:
    """Eq.(32): SiO2 peeling stress during heat-dwell."""
    _, A_cu, A_ox = _geom_areas(PITCH_UM, DIAM_UM)

    sigma_t = _sigma_t_thermal_Pa()
    sigma_e, sigma_p = _split_sigma_ep_heat_paper(sigma_t)
    d_heat = _delta_heat_m(sigma_e, sigma_p)

    D_m = float(D_nm) * 1e-9
    opening = d_heat - 2.0 * D_m
    if opening <= 0.0:
        return dict(sigma_peel_MPa=0.0, phi=0.0, delta_heat_nm=float(d_heat/1e-9), reason="no_opening_or_contact")

    phi = _phi_contact(d_heat, D_nm)
    if phi <= 0.0:
        return dict(sigma_peel_MPa=0.0, phi=0.0, delta_heat_nm=float(d_heat/1e-9), reason="phi_zero")

    kn = _k_n_Pa_per_m()
    sigma_Pa = kn * opening * (phi * A_cu) / A_ox
    return dict(
        sigma_peel_MPa=float(sigma_Pa/1e6),
        phi=float(phi),
        delta_heat_nm=float(d_heat/1e-9),
        reason="ok"
    )

def _sigma_y_cool_Pa() -> float:
    """Eq.(33): sigma_y,cool = (1-BAUSCHINGER)*sigma_y."""
    sigma_y_Pa = float(SIGMA_Y_MPA) * 1e6
    return (1.0 - float(BAUSCHINGER)) * sigma_y_Pa

def _split_sigma_ep_cool_paper(sigma_t_Pa: float) -> Tuple[float, float]:
    """Eq.(34): split using sigma_y,cool."""
    syc = _sigma_y_cool_Pa()
    sigma_e = min(float(sigma_t_Pa), syc)
    sigma_p = max(float(sigma_t_Pa) - syc, 0.0)
    return sigma_e, sigma_p

def _delta_cool_m(sigma_e_Pa: float, sigma_p_Pa: float) -> float:
    """Eq.(35): delta_cool in meters."""
    E  = float(CU_E_GPA) * 1e9
    nu = float(CU_NU)
    return (4.0 * nu / E) * (float(C_COOL_E) * float(sigma_e_Pa) + float(C_COOL_P) * float(sigma_p_Pa))

def _sigma_peel_cu_paper_MPa(D_nm: float) -> dict:
    """Eq.(36): Cu peeling stress during cool-down."""
    A_cell, A_cu, _ = _geom_areas(PITCH_UM, DIAM_UM)

    sigma_t = _sigma_t_thermal_Pa()
    sigma_e_h, sigma_p_h = _split_sigma_ep_heat_paper(sigma_t)
    d_heat = _delta_heat_m(sigma_e_h, sigma_p_h)

    phi = _phi_contact(d_heat, D_nm)
    if phi <= 0.0:
        return dict(sigma_cu_peel_MPa=0.0, phi=0.0, delta_heat_nm=float(d_heat/1e-9), reason="no_contact_in_heat")

    sigma_e_c, sigma_p_c = _split_sigma_ep_cool_paper(sigma_t)
    d_cool = _delta_cool_m(sigma_e_c, sigma_p_c)

    D_m = float(D_nm) * 1e-9
    opening = d_cool - d_heat + 2.0 * D_m
    if opening <= 0.0:
        return dict(
            sigma_cu_peel_MPa=0.0,
            phi=float(phi),
            delta_heat_nm=float(d_heat/1e-9),
            delta_cool_nm=float(d_cool/1e-9),
            reason="no_opening_in_cool"
        )

    kn = _k_n_Pa_per_m()
    factor_phi  = (1.0 / max(phi, 1e-12)) ** float(EXP_INVPHI)
    factor_area = (A_cell / A_cu) ** float(EXP_AREA)

    sigma_Pa = kn * opening * factor_phi * factor_area
    return dict(
        sigma_cu_peel_MPa=float(sigma_Pa/1e6),
        phi=float(phi),
        delta_heat_nm=float(d_heat/1e-9),
        delta_cool_nm=float(d_cool/1e-9),
        reason="ok"
    )

# --- Public API used by wafer-level -> inversion pipeline ---
def compute_sigma_peel_MPa_at(D_nm: float) -> dict:
    """Heat-dwell SiO2 peeling stress at given D (paper Eq.32)."""
    out = _sigma_peel_sio2_paper_MPa(D_nm)
    return dict(
        sigma_peel_MPa=float(out["sigma_peel_MPa"]),
        phi_cu=float(out["phi"]),
        delta_eq_nm=float(out["delta_heat_nm"]),   # legacy key (kept)
        delta_heat_nm=float(out["delta_heat_nm"]),
        reason=str(out.get("reason", "ok")),
    )

def compute_cu_peel_cool_MPa_at(D_nm: float) -> dict:
    """Cool-down Cu peeling stress at given D (paper Eq.36)."""
    out = _sigma_peel_cu_paper_MPa(D_nm)
    return dict(
        sigma_cu_peel_MPa=float(out["sigma_cu_peel_MPa"]),
        phi_cu=float(out.get("phi", 0.0)),
        delta_heat_nm=float(out.get("delta_heat_nm", 0.0)),
        delta_cool_nm=float(out.get("delta_cool_nm", 0.0)),
        sigma_y_cool_MPa=float(_sigma_y_cool_Pa()/1e6),
        reason=str(out.get("reason", "ok")),
    )


# =============================================================================
# ============================ LUT (PAD-SCALE) ================================
# =============================================================================

def _padscale_precompute_constants() -> dict:
    """
    Precompute D-independent constants for Eq.(32)/(36).
    """
    A_cell, A_cu, A_ox = _geom_areas(PITCH_UM, DIAM_UM)

    sigma_t = _sigma_t_thermal_Pa()
    sigma_e_h, sigma_p_h = _split_sigma_ep_heat_paper(sigma_t)
    d_heat = _delta_heat_m(sigma_e_h, sigma_p_h)

    sigma_e_c, sigma_p_c = _split_sigma_ep_cool_paper(sigma_t)
    d_cool = _delta_cool_m(sigma_e_c, sigma_p_c)

    kn = _k_n_Pa_per_m()
    area_factor = (A_cell / A_cu) ** float(EXP_AREA)

    return dict(
        A_cell=float(A_cell),
        A_cu=float(A_cu),
        A_ox=float(A_ox),
        sigma_t=float(sigma_t),
        d_heat=float(d_heat),
        d_cool=float(d_cool),
        kn=float(kn),
        area_factor=float(area_factor),
    )

def _phi_contact_vec(delta_heat_m_val: float, D_nm_vec: np.ndarray) -> np.ndarray:
    """
    Vectorized Eq.(30):
      phi = clip(((delta_heat-2D)/(2D))^EXP_PHI, 0, 1)
    with special-case D=0 -> 1.
    """
    D_nm_vec = np.asarray(D_nm_vec, dtype=np.float64)
    D_m = D_nm_vec * 1e-9

    phi = np.ones_like(D_m)  # D=0 -> 1
    mask = D_m > 0.0
    if not np.any(mask):
        return phi

    numer = float(delta_heat_m_val) - 2.0 * D_m[mask]
    pos = numer > 0.0

    phi_masked = np.zeros_like(D_m[mask])
    if np.any(pos):
        x = numer[pos] / (2.0 * D_m[mask][pos])
        val = x ** float(EXP_PHI)
        phi_masked[pos] = np.clip(val, 0.0, 1.0)

    phi[mask] = phi_masked
    return phi

def _sigma_sio2_vec_MPa(D_nm_vec: np.ndarray, const: dict) -> np.ndarray:
    """
    Vectorized Eq.(32) -> MPa:
      sigma = kn*(d_heat-2D)*(phi*A_cu)/A_ox, invalid -> 0
    """
    D_nm_vec = np.asarray(D_nm_vec, dtype=np.float64)
    D_m = D_nm_vec * 1e-9

    d_heat = float(const["d_heat"])
    kn = float(const["kn"])
    A_cu = float(const["A_cu"])
    A_ox = float(const["A_ox"])

    opening = d_heat - 2.0 * D_m
    phi = _phi_contact_vec(d_heat, D_nm_vec)

    sigma_Pa = kn * opening * (phi * A_cu) / A_ox
    sigma_Pa = np.where((opening > 0.0) & (phi > 0.0), sigma_Pa, 0.0)
    return sigma_Pa / 1e6

def _sigma_cu_vec_MPa(D_nm_vec: np.ndarray, const: dict) -> np.ndarray:
    """
    Vectorized Eq.(36) -> MPa:
      sigma = kn*(d_cool-d_heat+2D) * (1/phi)^EXP_INVPHI * (A_cell/A_cu)^EXP_AREA
      invalid -> 0
    """
    D_nm_vec = np.asarray(D_nm_vec, dtype=np.float64)
    D_m = D_nm_vec * 1e-9

    d_heat = float(const["d_heat"])
    d_cool = float(const["d_cool"])
    kn = float(const["kn"])
    area_factor = float(const["area_factor"])

    phi = _phi_contact_vec(d_heat, D_nm_vec)
    opening = d_cool - d_heat + 2.0 * D_m

    phi_safe = np.maximum(phi, 1e-12)
    factor_phi = (1.0 / phi_safe) ** float(EXP_INVPHI)

    sigma_Pa = kn * opening * factor_phi * area_factor
    sigma_Pa = np.where((opening > 0.0) & (phi > 0.0), sigma_Pa, 0.0)
    return sigma_Pa / 1e6

def _ensure_luts_ready(sio2_n: int = 2001, cu_n: int = 4001):
    """
    Build LUTs once per __init_params(cfg).

    SiO2 window: D in [0,10] nm
    Cu window:   D in [0,hi_eval], hi_eval = nextafter(D_contact_max, 0)
    """
    global _LUT_READY, _LUT_SIO2, _LUT_CU
    if _LUT_READY:
        return

    const = _padscale_precompute_constants()

    # ---------- SiO2 LUT ----------
    D_sio2 = np.linspace(0.0, 10.0, int(sio2_n), dtype=np.float64)
    sig_sio2 = _sigma_sio2_vec_MPa(D_sio2, const)

    # enforce monotone non-increasing (numerical robustness)
    sig_sio2_mono = np.maximum.accumulate(sig_sio2[::-1])[::-1]

    _LUT_SIO2 = dict(
        D_nm=D_sio2,
        sigma_MPa=sig_sio2_mono,
        lo=0.0,
        hi=10.0,
        f_lo=float(sig_sio2_mono[0]),
        f_hi=float(sig_sio2_mono[-1]),
        n=int(sio2_n),
    )

    # ---------- Cu LUT ----------
    delta_heat_nm = float(const["d_heat"] / 1e-9)
    D_contact_max = max(0.0, 0.5 * delta_heat_nm)

    if D_contact_max <= 0.0:
        # no contact domain
        sig0 = float(_sigma_cu_vec_MPa(np.array([0.0], dtype=np.float64), const)[0])
        _LUT_CU = dict(
            D_nm=np.array([0.0], dtype=np.float64),
            sigma_MPa=np.array([sig0], dtype=np.float64),
            D_contact_max=0.0,
            hi_eval=0.0,
            lo=0.0,
            hi=0.0,
            f_lo=sig0,
            f_hi=sig0,
            n=1,
            mode="no_contact_domain",
        )
        _LUT_READY = True
        return

    hi_eval = float(np.nextafter(D_contact_max, 0.0))
    if hi_eval <= 0.0:
        sig0 = float(_sigma_cu_vec_MPa(np.array([0.0], dtype=np.float64), const)[0])
        _LUT_CU = dict(
            D_nm=np.array([0.0], dtype=np.float64),
            sigma_MPa=np.array([sig0], dtype=np.float64),
            D_contact_max=float(D_contact_max),
            hi_eval=float(hi_eval),
            lo=0.0,
            hi=float(D_contact_max),
            f_lo=sig0,
            f_hi=sig0,
            n=1,
            mode="tiny_contact_domain",
        )
        _LUT_READY = True
        return

    D_cu = np.linspace(0.0, hi_eval, int(cu_n), dtype=np.float64)
    sig_cu = _sigma_cu_vec_MPa(D_cu, const)

    # enforce monotone non-decreasing (numerical robustness)
    sig_cu_mono = np.maximum.accumulate(sig_cu)

    _LUT_CU = dict(
        D_nm=D_cu,
        sigma_MPa=sig_cu_mono,
        D_contact_max=float(D_contact_max),
        hi_eval=float(hi_eval),
        lo=0.0,
        hi=float(D_contact_max),
        f_lo=float(sig_cu_mono[0]),
        f_hi=float(sig_cu_mono[-1]),
        n=int(cu_n),
        mode="ok",
    )

    _LUT_READY = True

def _invert_sio2_from_lut(sigma_eff_MPa: np.ndarray) -> np.ndarray:
    """
    Vectorized inversion for SiO2 using LUT.
    Rule: window [0,10] nm; if no root -> return 0.
    """
    _ensure_luts_ready()
    lut = _LUT_SIO2
    D = lut["D_nm"]
    f = lut["sigma_MPa"]  # decreasing (monotone-enforced)

    f_lo = float(lut["f_lo"])
    f_hi = float(lut["f_hi"])

    t = np.asarray(sigma_eff_MPa, dtype=np.float64)
    out = np.zeros_like(t, dtype=np.float64)

    # Valid range for decreasing curve: t in [f_hi, f_lo]
    mask = (t >= f_hi) & (t <= f_lo)
    if np.any(mask):
        # np.interp needs increasing x; reverse (f, D)
        f_inc = f[::-1]
        D_inc = D[::-1]
        out[mask] = np.interp(t[mask], f_inc, D_inc)

    return out

def _invert_cu_from_lut(sigma_eff_MPa: np.ndarray) -> np.ndarray:
    """
    Vectorized inversion for Cu using LUT.
    Rule: window [0,D_contact_max]; if no root -> return D_contact_max.
    """
    _ensure_luts_ready()
    lut = _LUT_CU
    D = lut["D_nm"]
    f = lut["sigma_MPa"]  # increasing (monotone-enforced)

    D_contact_max = float(lut["D_contact_max"])

    t = np.asarray(sigma_eff_MPa, dtype=np.float64)
    out = np.full_like(t, fill_value=D_contact_max, dtype=np.float64)

    # if no usable domain, just return D_contact_max (already filled)
    if float(lut.get("hi_eval", 0.0)) <= 0.0 or D_contact_max <= 0.0:
        return out

    f_lo = float(lut["f_lo"])
    f_hi = float(lut["f_hi"])

    # Valid range: t in [f_lo, f_hi] (increasing curve)
    mask = (t >= f_lo) & (t <= f_hi)
    if np.any(mask):
        out[mask] = np.interp(t[mask], f, D)

    return out


# =============================================================================
# ============================ CRITICAL / INVERT ==============================
# =============================================================================

def sigma_critical_MPa(Gc_Jpm2: float, E_GPa: float, nu: float,
                       aY2_um: float,
                       Effective_Contact_Area: float) -> float:
    E_Pa  = float(E_GPa) * 1e9
    aY2_m = float(aY2_um) * 1e-6
    sigma_Pa = float(Effective_Contact_Area) * math.sqrt((float(Gc_Jpm2) * E_Pa) / (aY2_m * (1.0 - float(nu)**2)))
    return float(sigma_Pa * 1e-6)

def compute_critical_peeling_all():
    return {
        "sigma_crit_MPa": {
            "SiO2": sigma_critical_MPa(GC_SIO2_JPM2, OX_E_GPA, OX_NU, CRIT_aY2_UM, Effective_Contact_Area),
            "Cu":   sigma_critical_MPa(GC_CU_JPM2,   CU_E_GPA, CU_NU, CRIT_aY2_UM, 1.0),
        }
    }

def invert_dishing_sio2_given_sigma_eff(sigma_eff_MPa: float) -> Tuple[float, dict]:
    """
    Fixed window inversion for SiO2 (LUT):
      - window D in [0,10] nm
      - if no root -> return 0
    """
    _ensure_luts_ready()
    t = float(sigma_eff_MPa)
    D_val = float(_invert_sio2_from_lut(np.array([t], dtype=np.float64))[0])

    lut = _LUT_SIO2
    f_lo = float(lut["f_lo"])
    f_hi = float(lut["f_hi"])
    if not (f_hi <= t <= f_lo):
        return 0.0, dict(
            mode="no_root_in_window",
            lo=lut["lo"], hi=lut["hi"],
            target=t,
            f_lo=f_lo, f_hi=f_hi,
            n=lut["n"],
        )
    return D_val, dict(mode="ok", lo=lut["lo"], hi=lut["hi"], target=t, n=lut["n"])

def invert_dishing_cu_given_sigma_eff(sigma_eff_MPa: float) -> Tuple[float, dict]:
    """
    Fixed window inversion for Cu (LUT):
      - window D in [0, D_contact_max], D_contact_max = delta_heat/2
      - if no root -> return D_contact_max
    """
    _ensure_luts_ready()
    t = float(sigma_eff_MPa)
    D_val = float(_invert_cu_from_lut(np.array([t], dtype=np.float64))[0])

    lut = _LUT_CU
    mode = str(lut.get("mode", "ok"))
    D_contact_max = float(lut.get("D_contact_max", 0.0))
    if mode != "ok":
        return D_val, dict(mode=mode, target=t, D_contact_max=D_contact_max)

    f_lo = float(lut["f_lo"])
    f_hi = float(lut["f_hi"])
    if not (f_lo <= t <= f_hi):
        return D_contact_max, dict(
            mode="no_root_in_window",
            lo=lut["lo"], hi=lut["hi"],
            target=t,
            D_contact_max=D_contact_max,
            hi_eval=float(lut["hi_eval"]),
            f_lo=f_lo, f_hi=f_hi,
            n=lut["n"],
        )

    return D_val, dict(
        mode="ok",
        lo=lut["lo"], hi=lut["hi"],
        target=t,
        D_contact_max=D_contact_max,
        hi_eval=float(lut["hi_eval"]),
        n=lut["n"],
    )


# =============================================================================
# ============================ WAFER-LEVEL STACK ==============================
# =============================================================================

def equiv_from_three(mix: LayerMix3) -> EqLayer:
    V1, V2, V3 = float(mix.V1), float(mix.V2), float(mix.V3)
    totalV = V1 + V2 + V3
    if totalV == 0.0:
        raise ValueError("Sum of volumes must be >0.")
    aeq = (mix.mat1.alpha_perC*V1 + mix.mat2.alpha_perC*V2 + mix.mat3.alpha_perC*V3)/totalV
    Eeq = (mix.mat1.E_Pa*V1      + mix.mat2.E_Pa*V2      + mix.mat3.E_Pa*V3)/totalV
    nueq= (mix.mat1.nu*V1        + mix.mat2.nu*V2        + mix.mat3.nu*V3)/totalV
    return EqLayer(E_Pa=float(Eeq), alpha_perC=float(aeq), nu=float(nueq), t_m=float(mix.t_m))

def warpage_D_two_layer_exact(L_m, t_c_m, t_s_m, E_c, E_s, alpha_c, alpha_s, T_C, T0_C):
    ratio = float(t_c_m) / float(t_s_m)
    dT = float(T_C) - float(T0_C)
    num_pref = (3.0 * (float(L_m) ** 2)) / (4.0 * (float(t_c_m) + float(t_s_m)))
    numerator = num_pref * ((1.0 + ratio) ** 2) * (float(alpha_s) - float(alpha_c)) * dT
    denom_left  = 3.0 * (1.0 + float(t_c_m) / float(t_s_m)) ** 2
    denom_right = (1.0 + (float(t_c_m) * float(E_c)) / (float(t_s_m) * float(E_s))) * (
        (float(t_c_m) ** 2) / (float(t_s_m) ** 2) + (float(t_s_m) * float(E_s)) / (float(t_c_m) * float(E_c))
    )
    denominator = denom_left + denom_right
    if denominator == 0.0:
        raise ZeroDivisionError("Denominator zero.")
    return float(numerator / denominator)

def combine_two_layers_to_one(top_eq: EqLayer, bot_eq: EqLayer) -> EqLayer:
    Vt, Vs = float(top_eq.t_m), float(bot_eq.t_m)
    total = Vt + Vs
    if total == 0.0:
        raise ValueError("Total thickness is zero.")
    aeq = (top_eq.alpha_perC*Vt + bot_eq.alpha_perC*Vs)/total
    Eeq = (top_eq.E_Pa*Vt      + bot_eq.E_Pa*Vs)/total
    nueq= (top_eq.nu*Vt        + bot_eq.nu*Vs)/total
    return EqLayer(E_Pa=float(Eeq), alpha_perC=float(aeq), nu=float(nueq), t_m=float(total))

@dataclass(frozen=True)
class WaferResult:
    D_m: float
    final_eq: EqLayer

def process_wafer(cfg: WaferConfig) -> WaferResult:
    top_eq = equiv_from_three(cfg.top)
    bot_eq = equiv_from_three(cfg.bottom)
    D = warpage_D_two_layer_exact(
        cfg.L_m, top_eq.t_m, bot_eq.t_m,
        top_eq.E_Pa, bot_eq.E_Pa,
        top_eq.alpha_perC, bot_eq.alpha_perC,
        cfg.T_C, cfg.T0_C
    )
    final_eq = combine_two_layers_to_one(top_eq, bot_eq)
    return WaferResult(D_m=float(D), final_eq=final_eq)

def plate_bending_stiffness(E: float, nu: float, h: float) -> float:
    E = float(E); nu = float(nu); h = float(h)
    return E * h**3 / (12.0 * (1.0 - nu**2))

def foundation_stiffness_K_effective(E1, nu1, h1, E2, nu2, h2):
    E1=float(E1); nu1=float(nu1); h1=float(h1)
    E2=float(E2); nu2=float(nu2); h2=float(h2)
    return 1.0 / ((1.0 - nu1)*h1/(3.0*E1) + (1.0 - nu2)*h2/(3.0*E2))

def suhir_peeling_two_wafers_bottomA_topB(waferA_eq: EqLayer, waferB_eq: EqLayer, R_m: float,
                                         sag_total_A_m: float, sag_total_B_m: float,
                                         sample_points: int = 500):
    """
    Return peel kernel parameters dict: p_max_Pa, beta, decay_length_m.
    """
    D1 = plate_bending_stiffness(waferA_eq.E_Pa, waferA_eq.nu, waferA_eq.t_m)
    D2 = plate_bending_stiffness(waferB_eq.E_Pa, waferB_eq.nu, waferB_eq.t_m)
    K  = foundation_stiffness_K_effective(
        waferA_eq.E_Pa, waferA_eq.nu, waferA_eq.t_m,
        waferB_eq.E_Pa, waferB_eq.nu, waferB_eq.t_m
    )
    R_m = float(R_m)
    kappa1 = 2.0 * float(sag_total_A_m) / (R_m**2)
    kappa2 = 2.0 * float(sag_total_B_m) / (R_m**2)
    M = (D1 * D2) / (D1 + D2) * (kappa1 - kappa2)
    beta = ((K * (D1 + D2)) / (4.0 * D1 * D2)) ** 0.25
    p_max = K * M / (2.0 * beta * D1)  # [Pa]
    decay_len = 1.0 / beta
    return {"p_max_Pa": float(p_max), "beta": float(beta), "decay_length_m": float(decay_len)}

def peeling_stress_at_points_vec_MPa(peel_dict: dict, coords_mm_np: np.ndarray, R_m: float) -> np.ndarray:
    """
    coords_mm_np: (N,2) in mm, center-origin; returns (N,) peeling stress in MPa.
    Raises if any point is outside the wafer (r > R).

    Retained for compatibility / debugging; main fast path now uses radial dishing LUT.
    """
    if coords_mm_np.ndim != 2 or coords_mm_np.shape[1] != 2:
        raise ValueError("coords_mm_np must be shape (N,2).")
    xy_m = coords_mm_np.astype(np.float64, copy=False) * 1e-3
    r_m  = np.sqrt(xy_m[:,0]**2 + xy_m[:,1]**2)
    R_m = float(R_m)
    if np.any(r_m > R_m + 1e-15):
        idx = np.where(r_m > R_m + 1e-15)[0][:5]
        raise ValueError(f"{idx.size} points lie outside wafer radius R={R_m} m, e.g. indices {idx.tolist()}")
    s = R_m - r_m
    p_max = float(peel_dict["p_max_Pa"])
    beta  = float(peel_dict["beta"])
    p_pa  = p_max * np.exp(-beta*s) * (np.cos(beta*s) - np.sin(beta*s))

    if USE_PLOT:
        plt.figure()
        plt.scatter(r_m*1e3, p_pa/1e6, s=8)
        plt.xlabel("Radius r (mm)")
        plt.ylabel("Peeling Stress p (MPa)")
        plt.title("Peeling Stress vs Radius")
        plt.grid(True)
        plt.show()

    return p_pa / 1e6  # MPa


# =============================================================================
# ========================= RADIAL DISHING LUT (NEW) ==========================
# =============================================================================

def build_current_radial_dishing_lut_from_cfg(cfg, n_r: int = 4096) -> dict:
    """
    Convenience helper:
    Build (or refresh) the current radial dishing LUT directly from cfg.

    This runs the same wafer-level pipeline as debond_dishing_bounds_calculator(),
    but stops at LUT construction.

    Returns:
      radial LUT dict (_RDISH_LUT), with fields including:
        r_grid_m, D_sio2_nm, D_cu_nm, p_global_MPa, ...
    """
    __init_params(cfg)

    # 1) Wafer-level stack to get peeling kernel
    resA = process_wafer(WAFER_A)  # bottom
    resB = process_wafer(WAFER_B)  # top

    # Keep same sign convention as main path
    s_total_A_m = float(S_INIT_A_M) - float(resA.D_m)
    s_total_B_m = float(S_INIT_B_M) - float(resB.D_m)
    R_stack = float(min(WAFER_A.L_m, WAFER_B.L_m))

    peel = suhir_peeling_two_wafers_bottomA_topB(
        waferA_eq=resA.final_eq,
        waferB_eq=resB.final_eq,
        R_m=R_stack,
        sag_total_A_m=s_total_A_m,
        sag_total_B_m=s_total_B_m,
        sample_points=500
    )

    return _ensure_radial_dishing_lut(peel_dict=peel, R_m=R_stack, n_r=n_r)


def get_radial_dishing_lut_array(r_unit: str = "um",
                                 include_p_global: bool = True,
                                 include_sigma_eff: bool = False,
                                 require_prebuilt: bool = True) -> np.ndarray:
    """
    Export the already-built radial dishing LUT table directly from _RDISH_LUT.

    This function does NOT use pad coordinates.

    Returns columns (minimum):
      [r, D_sio2_nm, D_cu_nm]

    Optional columns:
      + p_global_MPa
      + sigma_eff_Cu_MPa, sigma_eff_SiO2_MPa
    """
    global _RDISH_LUT

    if _RDISH_LUT is None:
        if require_prebuilt:
            raise RuntimeError(
                "Radial dishing LUT is not built yet. "
                "Call _ensure_radial_dishing_lut(...) first."
            )
        else:
            raise RuntimeError(
                "No prebuilt LUT found. In this simplified mode, build LUT in main first."
            )

    lut = _RDISH_LUT

    r_grid_m  = np.asarray(lut["r_grid_m"], dtype=np.float64)
    D_sio2_nm = np.asarray(lut["D_sio2_nm"], dtype=np.float64)
    D_cu_nm   = np.asarray(lut["D_cu_nm"], dtype=np.float64)

    r_unit_l = str(r_unit).lower()
    if r_unit_l == "m":
        r_out = r_grid_m
    elif r_unit_l == "mm":
        r_out = r_grid_m * 1e3
    elif r_unit_l == "um":
        r_out = r_grid_m * 1e6
    else:
        raise ValueError(f"Unsupported r_unit={r_unit!r}. Use 'm', 'mm', or 'um'.")

    cols = [r_out, D_sio2_nm, D_cu_nm]

    if include_p_global:
        cols.append(np.asarray(lut["p_global_MPa"], dtype=np.float64))

    if include_sigma_eff:
        cols.append(np.asarray(lut["sigma_eff_Cu_MPa"], dtype=np.float64))
        cols.append(np.asarray(lut["sigma_eff_SiO2_MPa"], dtype=np.float64))

    return np.column_stack(cols)

def plot_radial_dishing_lut(r_unit: str = "mm",
                            show: bool = True,
                            require_prebuilt: bool = True,
                            print_table_preview: bool = False,
                            preview_rows: int = 10):
    """
    Plot the already-built radial dishing LUT directly from _RDISH_LUT.

    This function does NOT use pad coordinates.

    Returns:
      (fig, ax, arr)
      arr columns = [r, D_sio2_nm, D_cu_nm, p_global_MPa]
    """
    arr = get_radial_dishing_lut_array(
        r_unit=r_unit,
        include_p_global=True,
        include_sigma_eff=False,
        require_prebuilt=require_prebuilt
    )

    # columns = [r, D_sio2_nm, D_cu_nm, p_global_MPa]
    r = arr[:, 0]
    D_sio2 = arr[:, 1]
    D_cu   = arr[:, 2]

    if print_table_preview:
        n_show = int(max(1, min(preview_rows, arr.shape[0])))
        print(f"[Radial LUT preview] showing first {n_show} / {arr.shape[0]} rows")
        print("Columns = [r, D_sio2_nm, D_cu_nm, p_global_MPa]")
        print(arr[:n_show])

    r_unit_l = str(r_unit).lower()
    if r_unit_l == "m":
        xlab = "r (m)"
    elif r_unit_l == "mm":
        xlab = "r (mm)"
    elif r_unit_l == "um":
        xlab = "r (µm)"
    else:
        xlab = f"r ({r_unit})"

    fig, ax = plt.subplots()
    ax.plot(r, D_sio2, label="SiO2 dishing LUT")
    ax.plot(r, D_cu,   label="Cu dishing LUT")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Dishing (nm)")
    ax.set_title("Radial Dishing LUT (direct table plot)")
    ax.grid(True)
    ax.legend()

    if show:
        plt.show()

    return fig, ax, arr


def _build_radial_dishing_lut(peel_dict: dict, R_m: float, n_r: int = 4096) -> dict:
    """
    Build LUT for a fixed wafer/die state:

      r [m]
        -> p_global(r) [MPa]
        -> sigma_eff_Cu / sigma_eff_SiO2 [MPa]
        -> D_cu / D_sio2 [nm]

    This reuses existing pad-scale sigma->D LUTs.
    """
    global _RDISH_LUT

    R_m = float(R_m)
    if R_m <= 0.0:
        raise ValueError(f"R_m must be > 0, got {R_m}")

    # make sure pad-scale sigma->D LUTs exist
    _ensure_luts_ready()

    n_r = max(2, int(n_r))
    r_grid = np.linspace(0.0, R_m, n_r, dtype=np.float64)

    # p_global(r)
    p_max = float(peel_dict["p_max_Pa"])
    beta  = float(peel_dict["beta"])
    s = R_m - r_grid
    p_mpa = (p_max * np.exp(-beta * s) * (np.cos(beta * s) - np.sin(beta * s))) / 1e6

    # sigma_eff(r)
    crits = compute_critical_peeling_all()
    sigma_crit_SiO2 = float(crits["sigma_crit_MPa"]["SiO2"])
    sigma_crit_Cu   = float(crits["sigma_crit_MPa"]["Cu"])
    sigma_eff_SiO2 = sigma_crit_SiO2 - p_mpa
    sigma_eff_Cu   = sigma_crit_Cu   - p_mpa

    # D(r) via existing sigma->D LUTs
    D_sio2_nm = _invert_sio2_from_lut(sigma_eff_SiO2)
    D_cu_nm   = _invert_cu_from_lut(sigma_eff_Cu)

    _RDISH_LUT = dict(
        R_m=R_m,
        p_max_Pa=p_max,
        beta=beta,
        n_r=n_r,

        r_grid_m=r_grid,
        p_global_MPa=p_mpa,

        sigma_eff_Cu_MPa=sigma_eff_Cu,
        sigma_eff_SiO2_MPa=sigma_eff_SiO2,

        D_cu_nm=D_cu_nm,
        D_sio2_nm=D_sio2_nm,
    )
    return _RDISH_LUT

def _ensure_radial_dishing_lut(peel_dict: dict, R_m: float, n_r: int = 4096) -> dict:
    """
    Rebuild radial LUT only if state changed:
      (R_m, p_max_Pa, beta, n_r)
    """
    global _RDISH_LUT

    R_m = float(R_m)
    p_max = float(peel_dict["p_max_Pa"])
    beta  = float(peel_dict["beta"])
    n_r = max(2, int(n_r))

    if _RDISH_LUT is None:
        return _build_radial_dishing_lut(peel_dict, R_m, n_r=n_r)

    same = (
        float(_RDISH_LUT.get("R_m", np.nan)) == R_m and
        float(_RDISH_LUT.get("p_max_Pa", np.nan)) == p_max and
        float(_RDISH_LUT.get("beta", np.nan)) == beta and
        int(_RDISH_LUT.get("n_r", -1)) == n_r
    )
    if not same:
        return _build_radial_dishing_lut(peel_dict, R_m, n_r=n_r)

    return _RDISH_LUT

def _query_radial_dishing_lut_from_r(r_m: np.ndarray, peel_dict: dict, R_m: float, n_r: int = 4096):
    """
    Query radial LUT by radius array r [m].

    Returns:
      D_cu_nm, D_sio2_nm, p_global_MPa
    """
    lut = _ensure_radial_dishing_lut(peel_dict, R_m, n_r=n_r)

    r = np.asarray(r_m, dtype=np.float64)
    r_clip = np.clip(r, 0.0, float(lut["R_m"]))

    D_cu_nm   = np.interp(r_clip, lut["r_grid_m"], lut["D_cu_nm"])
    D_sio2_nm = np.interp(r_clip, lut["r_grid_m"], lut["D_sio2_nm"])
    p_mpa     = np.interp(r_clip, lut["r_grid_m"], lut["p_global_MPa"])

    return D_cu_nm, D_sio2_nm, p_mpa


# =============================================================================
# ===================== EFFICIENT CRITICAL & DISHING ARRAYS ===================
# =============================================================================

def build_effcrit_and_dishing_arrays(peel_dict: dict, coords_mm_np: np.ndarray, R_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      effcrit_array: (N,2) columns = (σ_eff_crit_Cu, σ_eff_crit_SiO2) [MPa]
      dishing_array: (N,2) columns = (D*_Cu_nm,      D*_SiO2_nm)      [nm]

    Fast path (new):
      coords -> r -> radial dishing LUT -> D
    """
    if coords_mm_np.ndim != 2 or coords_mm_np.shape[1] != 2:
        raise ValueError("coords_mm_np must be shape (N,2).")

    xy_m = coords_mm_np.astype(np.float64, copy=False) * 1e-3
    r_m  = np.sqrt(xy_m[:,0]**2 + xy_m[:,1]**2)

    R_m = float(R_m)
    if np.any(r_m > R_m + 1e-15):
        idx = np.where(r_m > R_m + 1e-15)[0][:5]
        raise ValueError(f"{idx.size} points lie outside wafer radius R={R_m} m, e.g. indices {idx.tolist()}")

    # Direct query: r -> dishing (and p_global for effcrit reconstruction)
    D_cu_nm, D_sio2_nm, p_MPa = _query_radial_dishing_lut_from_r(
        r_m=r_m, peel_dict=peel_dict, R_m=R_m, n_r=4096
    )

    # Reconstruct effcrit output (to keep same API)
    crits = compute_critical_peeling_all()
    sigma_crit_SiO2 = float(crits["sigma_crit_MPa"]["SiO2"])
    sigma_crit_Cu   = float(crits["sigma_crit_MPa"]["Cu"])

    sigma_eff_SiO2 = sigma_crit_SiO2 - p_MPa
    sigma_eff_Cu   = sigma_crit_Cu   - p_MPa

    effcrit = np.column_stack([sigma_eff_Cu, sigma_eff_SiO2])
    dishing = np.column_stack([D_cu_nm, D_sio2_nm])
    return effcrit, dishing

# =============================================================================
# ================================= MAIN ======================================
# =============================================================================

# Public API: simplified main function for LUT-only mode, no pad lookup, direct array output, optional preview/plot.

def debond_dishing_bounds_calculator(cfg,
                                     n_r: int = 4096,
                                     radial_lut_r_unit: str = "um",
                                     print_radial_lut: bool = True,
                                     radial_lut_preview_rows: int = 10,
                                     plot_radial_lut_flag: bool = True,
                                     include_p_global: bool = False,
                                     include_sigma_eff: bool = False):
    """
    Simplified public API (LUT-only mode):
      - Build radial dishing LUT from current wafer/die state
      - Directly return the LUT table array
      - Optionally print a preview and/or plot the LUT

    This function DOES NOT accept pad coordinates and DOES NOT perform pad lookup.

    Returns:
      radial_lut_array: ndarray
        minimum columns:
          [r, D_sio2_nm, D_cu_nm]
        optional appended columns:
          + p_global_MPa
          + sigma_eff_Cu_MPa, sigma_eff_SiO2_MPa
    """
    __init_params(cfg)

    # 1) Wafer-level stack to get peeling kernel
    resA = process_wafer(WAFER_A)  # bottom
    resB = process_wafer(WAFER_B)  # top

    # NOTE: sign convention kept as previous code (s_total = s_init - D)
    s_total_A_m = float(S_INIT_A_M) - float(resA.D_m)
    s_total_B_m = float(S_INIT_B_M) - float(resB.D_m)
    R_stack = float(min(WAFER_A.L_m, WAFER_B.L_m))

    peel = suhir_peeling_two_wafers_bottomA_topB(
        waferA_eq=resA.final_eq,
        waferB_eq=resB.final_eq,
        R_m=R_stack,
        sag_total_A_m=s_total_A_m,
        sag_total_B_m=s_total_B_m,
        sample_points=500
    )

    # 2) Build / ensure the radial LUT table (this is the target table)
    _ensure_radial_dishing_lut(peel_dict=peel, R_m=R_stack, n_r=int(n_r))

    # 3) Directly export the built LUT table
    radial_lut_array = get_radial_dishing_lut_array(
        r_unit=radial_lut_r_unit,
        include_p_global=include_p_global,
        include_sigma_eff=include_sigma_eff,
        require_prebuilt=True
    )

    # 4) Optional print preview (table preview only)
    if print_radial_lut:
        n_show = int(max(1, min(radial_lut_preview_rows, radial_lut_array.shape[0])))
        col_names = [f"r ({radial_lut_r_unit})", "D_sio2_nm", "D_cu_nm"]
        if include_p_global:
            col_names.append("p_global_MPa")
        if include_sigma_eff:
            col_names.extend(["sigma_eff_Cu_MPa", "sigma_eff_SiO2_MPa"])

        print(f"[Radial dishing LUT] total rows = {radial_lut_array.shape[0]}")
        print("Columns =", col_names)
        print(radial_lut_array[-n_show:])

    # 5) Optional plot (direct LUT plot only)
    if plot_radial_lut_flag:
        plot_radial_dishing_lut(
            r_unit=radial_lut_r_unit,
            show=True,
            require_prebuilt=True,
            print_table_preview=False
        )

    return radial_lut_array


# New main API: accepts pad coordinates, returns per-pad dishing intervals, optional effcrit and debug outputs.


def debond_dishing_bounds_calculator_coords(cfg,
                                            coords_um: np.ndarray,
                                            *,
                                            n_r: int = 4096,
                                            return_effcrit: bool = False,
                                            return_debug: bool = False):
    """
    New main API (coords -> per-pad dishing intervals):

      cfg + coords_um
        -> wafer-level peel_dict + R_stack
        -> ensure radial dishing LUT (r -> D)
        -> coords -> r -> interp (D_cu_nm, D_sio2_nm)
        -> output per pad: sorted(D_cu_nm, D_sio2_nm)

    Parameters
    ----------
    cfg : object
        Same cfg used in __init_params(cfg).
    coords_um : np.ndarray, shape (N,2)
        Pad global coordinates [um], columns = [x_um, y_um], center-origin.
    n_r : int
        Radial LUT resolution.
    return_effcrit : bool
        If True, also return effcrit array (N,2) with columns:
          [sigma_eff_Cu_MPa, sigma_eff_SiO2_MPa]
    return_debug : bool
        If True, also return debug dict including peel_dict, R_stack, and LUT cache key.

    Returns
    -------
    dishing_intervals : np.ndarray, shape (N,2)
        Each row = sorted(D_Cu_nm, D_SiO2_nm).

    If return_effcrit=True:
        (dishing_intervals, effcrit)

    If return_debug=True:
        (dishing_intervals, debug)  or  (dishing_intervals, effcrit, debug)
    """
    # -----------------------
    # 0) Validate coords
    # -----------------------
    coords_um = np.asarray(coords_um, dtype=np.float64)
    if coords_um.ndim != 2 or coords_um.shape[1] != 2:
        raise ValueError("coords_um must be shape (N,2) with columns [x_um, y_um].")

    # -----------------------
    # 1) Same wafer-level pipeline as radial LUT main
    # -----------------------
    __init_params(cfg)

    resA = process_wafer(WAFER_A)  # bottom
    resB = process_wafer(WAFER_B)  # top

    # Keep the same sign convention: s_total = s_init - D
    s_total_A_m = float(S_INIT_A_M) - float(resA.D_m)
    s_total_B_m = float(S_INIT_B_M) - float(resB.D_m)
    R_stack = float(min(WAFER_A.L_m, WAFER_B.L_m))

    peel = suhir_peeling_two_wafers_bottomA_topB(
        waferA_eq=resA.final_eq,
        waferB_eq=resB.final_eq,
        R_m=R_stack,
        sag_total_A_m=s_total_A_m,
        sag_total_B_m=s_total_B_m,
        sample_points=500
    )

    # -----------------------
    # 2) Ensure radial LUT exists for this (R_stack, p_max, beta, n_r)
    # -----------------------
    _ensure_radial_dishing_lut(peel_dict=peel, R_m=R_stack, n_r=int(n_r))

    # -----------------------
    # 3) coords_um -> r_m
    # -----------------------
    xy_m = coords_um * 1e-6
    r_m = np.sqrt(xy_m[:, 0]**2 + xy_m[:, 1]**2)

    if np.any(r_m > R_stack + 1e-15):
        idx = np.where(r_m > R_stack + 1e-15)[0][:5]
        raise ValueError(
            f"{idx.size} points lie outside wafer radius R={R_stack} m, "
            f"e.g. indices {idx.tolist()} (r_max={float(np.max(r_m))} m)"
        )

    # -----------------------
    # 4) Query radial LUT: r -> (D_cu_nm, D_sio2_nm, p_global_MPa)
    # -----------------------
    D_cu_nm, D_sio2_nm, p_MPa = _query_radial_dishing_lut_from_r(
        r_m=r_m, peel_dict=peel, R_m=R_stack, n_r=int(n_r)
    )

    # Output format required by your docstring: sorted(D_Cu_nm, D_SiO2_nm)
    dishing_intervals = np.column_stack([D_sio2_nm, D_cu_nm])

    # -----------------------
    # 5) Optional outputs
    # -----------------------
    effcrit = None
    if return_effcrit or return_debug:
        crits = compute_critical_peeling_all()
        sigma_crit_SiO2 = float(crits["sigma_crit_MPa"]["SiO2"])
        sigma_crit_Cu   = float(crits["sigma_crit_MPa"]["Cu"])
        sigma_eff_SiO2 = sigma_crit_SiO2 - p_MPa
        sigma_eff_Cu   = sigma_crit_Cu   - p_MPa
        effcrit = np.column_stack([sigma_eff_Cu, sigma_eff_SiO2])

    debug = None
    if return_debug:
        debug = {
            "R_stack_m": R_stack,
            "peel_dict": dict(peel),
            "n_r": int(n_r),
            "RDISH_cache_key": None if _RDISH_LUT is None else {
                "R_m": float(_RDISH_LUT.get("R_m", np.nan)),
                "p_max_Pa": float(_RDISH_LUT.get("p_max_Pa", np.nan)),
                "beta": float(_RDISH_LUT.get("beta", np.nan)),
                "n_r": int(_RDISH_LUT.get("n_r", -1)),
            },
        }

    if return_effcrit and return_debug:
        return dishing_intervals, effcrit, debug
    if return_effcrit:
        return dishing_intervals, effcrit
    if return_debug:
        return dishing_intervals, debug
    return dishing_intervals