#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debond.py  (produce dishing intervals from manual coord list)

- Remove ALL external wafer_info reading.
- User defines pad global coordinates (in µm) at the caller.
- Output: numpy.ndarray (N,2), each row = sorted (D_Cu_nm, D_SiO2_nm).

UPDATED (paper Eq.(27)–(36) version):
1) PAD-SCALE CORE replaced by paper Eq.(27)–(36).
2) All paper coefficients are loaded from cfg in __init_params(cfg).
3) Inversion (FIXED WINDOW):
   - SiO2: search D in [0, 10] nm only; if no root -> return 0
   - Cu  : search D in [0, D_contact_max] only, D_contact_max = delta_heat/2
           if no root -> return D_contact_max
   Notes:
     - To avoid phi->0 divergence at the boundary D = D_contact_max, the bisection
       evaluation upper bound uses hi_eval = nextafter(D_contact_max, 0).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math
import matplotlib.pyplot as plt
import numpy as np
from roughness_coefficients import get_eff_contact_area_ratio
from matplotlib.ticker import MultipleLocator

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

def __init_params(cfg):
    global PITCH_UM, DIAM_UM, T_ANNEAL_C, T_REF_C, \
           CU_E_GPA, CU_NU, CU_ALPHA_PPM, OX_E_GPA, OX_NU, OX_ALPHA_PPM, \
           SIGMA_Y_MPA, \
           CRIT_aY2_UM, GC_SIO2_JPM2, GC_CU_JPM2, Effective_Contact_Area, \
           MAT_CU, MAT_SiO2, MAT_Si, \
           WAFER_A, WAFER_B, \
           S_INIT_A_M, S_INIT_B_M, \
           USE_PLOT, \
           C_HEAT_E, C_HEAT_P, C_COOL_E, C_COOL_P, \
           EXP_PHI, BAUSCHINGER, EXP_INVPHI, EXP_AREA, \
           KN_DEN_M

    # ---------- (A) Pad-scale: Geometry & Temps ----------
    if cfg.PAD_ARRANGE_PATTERN == 'checkerboard':
        PITCH_UM = min(np.sqrt(cfg.PITCH_r_um ** 2 + cfg.PITCH_c_um ** 2), 2 * cfg.PITCH_r_um, 2 * cfg.PITCH_c_um)
    else:
        PITCH_UM = cfg.PITCH_r_um

    DIAM_UM       = cfg.PAD_TOP_R_um * 2      # pad diameter d [µm]
    T_ANNEAL_C    = cfg.T_anl                 # anneal temperature [°C]
    T_REF_C       = cfg.T_R                   # reference temperature [°C]

    # ---------- (B) Pad-scale: Material constants ----------
    CU_E_GPA      = cfg.CU_E_GPA
    CU_NU         = cfg.CU_NU
    CU_ALPHA_PPM  = cfg.CU_ALPHA_PPM
    OX_E_GPA      = cfg.OX_E_GPA
    OX_NU         = cfg.OX_NU
    OX_ALPHA_PPM  = cfg.OX_ALPHA_PPM

    # ---------- (C) Pad-scale: Yield stress ----------
    SIGMA_Y_MPA   = cfg.SIGMA_Y_MPA

    # ---------- (D) Paper Eq.(27)–(36) coefficients (from cfg) ----------
    C_HEAT_E     = float(cfg.C_HEAT_E)        # Eq.(29) coefficient for sigma_e
    C_HEAT_P     = float(cfg.C_HEAT_P)        # Eq.(29) coefficient for sigma_p
    C_COOL_E     = float(cfg.C_COOL_E)        # Eq.(35) coefficient for sigma_e,cool
    C_COOL_P     = float(cfg.C_COOL_P)        # Eq.(35) coefficient for sigma_p,cool
    EXP_PHI      = float(cfg.EXP_PHI)         # Eq.(30)
    BAUSCHINGER  = float(cfg.BAUSCHINGER)     # Eq.(33)
    EXP_INVPHI   = float(cfg.EXP_INVPHI)      # Eq.(36)
    EXP_AREA     = float(cfg.EXP_AREA)        # Eq.(36)
    KN_DEN_M     = float(cfg.KN_DEN_M)        # Eq.(31) denominator length in meters

    # ---------- (E) Critical peeling stress ----------
    CRIT_aY2_UM = cfg.CRIT_aY2_UM
    GC_SIO2_JPM2 = cfg.GC_SIO2_JPM2
    GC_CU_JPM2 = cfg.GC_CU_JPM2
    Effective_Contact_Area = get_eff_contact_area_ratio(
        Asperity_R_m = cfg.Asperity_R_m,
        Roughness_sigma_m = cfg.Roughness_sigma_m,
        eta_s = cfg.eta_s,
        Roughness_constant = cfg.Roughness_constant,
        Adhesion_energy = cfg.Adhesion_energy,
        Dielectric_Young_modulus_Pa = cfg.Dielectric_Young_modulus_Pa,
    )
    assert 0.0 < Effective_Contact_Area <= 1.0, f"Effective_Contact_Area must be in (0,1], got {Effective_Contact_Area}"

    # ---------- (F) Wafer-layer materials ----------
    MAT_CU   = Material("Cu",   E_Pa=cfg.CU_E_GPA*1e9,  alpha_perC=cfg.CU_ALPHA_PPM*1e-6, nu=cfg.CU_NU)
    MAT_SiO2 = Material("SiO2", E_Pa=cfg.OX_E_GPA*1e9,  alpha_perC=cfg.OX_ALPHA_PPM*1e-6, nu=cfg.OX_NU)
    MAT_Si   = Material("Si",   E_Pa=cfg.SI_E_GPA*1e9,  alpha_perC=cfg.SI_ALPHA_PPM*1e-6,  nu=cfg.SI_NU)

    # ---------- (G) Wafer configs ----------
    WAFER_A = WaferConfig(
        top=LayerMix3(MAT_CU,cfg.B_Chip_Cu_V,MAT_SiO2,cfg.B_Chip_Sio2_V,MAT_Si,cfg.B_Chip_Si_V,cfg.B_Chip_T),
        bottom=LayerMix3(MAT_Si,cfg.B_Sub_Si_V,MAT_SiO2,cfg.B_Sub_Sio2_V,MAT_CU,cfg.B_Sub_Cu_V,cfg.B_Sub_T),
        L_m= cfg.eff_DIE_R*1e-6, T_C= cfg.T_anl, T0_C= cfg.T_R
    )
    WAFER_B = WaferConfig(
        top=LayerMix3(MAT_CU,cfg.T_Chip_Cu_V,MAT_SiO2,cfg.T_Chip_Sio2_V,MAT_Si,cfg.T_Chip_Si_V,cfg.T_Chip_T),
        bottom=LayerMix3(MAT_Si,cfg.T_Sub_Si_V,MAT_SiO2,cfg.T_Sub_Sio2_V,MAT_CU,cfg.T_Sub_Cu_V,cfg.T_Sub_T),
        L_m= cfg.eff_DIE_R*1e-6, T_C= cfg.T_anl, T0_C= cfg.T_R
    )

    # ---------- (H) Pre-anneal warpages ----------
    S_INIT_A_M = cfg.S_INIT_A_M
    S_INIT_B_M = cfg.S_INIT_B_M

    # ---------- (J) Optional plotting ----------
    USE_PLOT = False


# =============================================================================
# ============================== PAD-SCALE CORE ===============================
# =============================================================================

def _units():
    return dict(um=1e-6, nm=1e-9, GPa=1e9, MPa=1e6)

def _geom_areas(p_um, d_um):
    U=_units(); p=p_um*U['um']; d=d_um*U['um']
    A_cell=p**2
    A_cu=math.pi*(d**2)/4.0
    A_ox=A_cell-A_cu
    if A_ox<=0:
        raise ValueError("A_ox<=0, check PITCH_UM and DIAM_UM values.")
    return A_cell, A_cu, A_ox

# -------- Paper Eq.(27)–(36) --------

def _sigma_t_thermal_Pa() -> float:
    """Eq.(27) in Pa"""
    dT = (T_ANNEAL_C - T_REF_C)
    E = CU_E_GPA * 1e9
    nu = CU_NU
    dalpha = (CU_ALPHA_PPM - OX_ALPHA_PPM) * 1e-6
    return (E / (1.0 - nu)) * dalpha * dT

def _split_sigma_ep_heat_paper(sigma_t_Pa: float) -> Tuple[float, float]:
    """Eq.(28): sigma_e=min(sigma_t,sigma_y), sigma_p=max(sigma_t-sigma_y,0)"""
    sigma_y_Pa = SIGMA_Y_MPA * 1e6
    sigma_e = min(sigma_t_Pa, sigma_y_Pa)
    sigma_p = max(sigma_t_Pa - sigma_y_Pa, 0.0)
    return sigma_e, sigma_p

def _delta_heat_m(sigma_e_Pa: float, sigma_p_Pa: float) -> float:
    """Eq.(29) -> meters"""
    E = CU_E_GPA * 1e9
    nu = CU_NU
    return (4.0 * nu / E) * (C_HEAT_E * sigma_e_Pa + C_HEAT_P * sigma_p_Pa)

def _phi_contact(delta_heat_m_val: float, D_nm: float) -> float:
    """Eq.(30): clip(((delta_heat-2D)/(2D))^EXP_PHI, 0, 1) with safe limits."""
    D_m = float(D_nm) * 1e-9
    if D_m <= 0.0:
        return 1.0
    numer = delta_heat_m_val - 2.0 * D_m
    if numer <= 0.0:
        return 0.0
    x = numer / (2.0 * D_m)
    val = x ** EXP_PHI
    return float(max(0.0, min(1.0, val)))

def _k_n_Pa_per_m() -> float:
    """Eq.(31) -> Pa/m"""
    E = CU_E_GPA * 1e9
    nu = CU_NU
    return (2.0 * E) / (KN_DEN_M * (1.0 - nu))

def _sigma_y_cool_Pa() -> float:
    """Eq.(33): sigma_y,cool = (1-BAUSCHINGER)*sigma_y"""
    sigma_y_Pa = SIGMA_Y_MPA * 1e6
    return (1.0 - BAUSCHINGER) * sigma_y_Pa

def _split_sigma_ep_cool_paper(sigma_t_Pa: float) -> Tuple[float, float]:
    """Eq.(34)"""
    syc = _sigma_y_cool_Pa()
    sigma_e = min(sigma_t_Pa, syc)
    sigma_p = max(sigma_t_Pa - syc, 0.0)
    return sigma_e, sigma_p

def _delta_cool_m(sigma_e_Pa: float, sigma_p_Pa: float) -> float:
    """Eq.(35) -> meters"""
    E = CU_E_GPA * 1e9
    nu = CU_NU
    return (4.0 * nu / E) * (C_COOL_E * sigma_e_Pa + C_COOL_P * sigma_p_Pa)

def _sigma_peel_sio2_paper_MPa(D_nm: float) -> dict:
    """Eq.(32)"""
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

def _sigma_peel_cu_paper_MPa(D_nm: float) -> dict:
    """Eq.(36)"""
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
        return dict(sigma_cu_peel_MPa=0.0, phi=float(phi), delta_heat_nm=float(d_heat/1e-9),
                    delta_cool_nm=float(d_cool/1e-9), reason="no_opening_in_cool")

    kn = _k_n_Pa_per_m()
    factor_phi  = (1.0 / max(phi, 1e-12)) ** EXP_INVPHI
    factor_area = (A_cell / A_cu) ** EXP_AREA

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
        delta_eq_nm=float(out["delta_heat_nm"]),   # legacy key
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
# ============================ CRITICAL / INVERT ==============================
# =============================================================================

def sigma_critical_MPa(Gc_Jpm2: float, E_GPa: float, nu: float,
                       aY2_um: float,
                       Effective_Contact_Area: float) -> float:
    E_Pa  = E_GPa * 1e9
    aY2_m = aY2_um * 1e-6
    sigma_Pa = Effective_Contact_Area * math.sqrt((Gc_Jpm2 * E_Pa) / (aY2_m * (1.0 - nu**2)))
    return float(sigma_Pa * 1e-6)

def compute_critical_peeling_all():
    return {
        "sigma_crit_MPa": {
            "SiO2": sigma_critical_MPa(GC_SIO2_JPM2, OX_E_GPA, OX_NU, CRIT_aY2_UM, Effective_Contact_Area),
            "Cu":   sigma_critical_MPa(GC_CU_JPM2,   CU_E_GPA, CU_NU, CRIT_aY2_UM, 1.0),
        }
    }

def _bisect_mono(f, target, lo, hi, is_increasing, tol=1e-6, maxit=80):
    """
    Monotone bisection that returns None if target is NOT within f(lo)..f(hi)
    under the monotonic assumption.
    """
    lo = float(lo); hi = float(hi)
    f_lo, f_hi = f(lo), f(hi)

    if is_increasing:
        if not (f_lo <= target <= f_hi):
            return None
    else:
        if not (f_hi <= target <= f_lo):
            return None

    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm - target) <= tol:
            return mid
        if is_increasing:
            if fm < target:
                lo = mid
            else:
                hi = mid
        else:
            if fm > target:
                lo = mid
            else:
                hi = mid

    return 0.5 * (lo + hi)

def _sio2_stress_at(D_nm: float) -> float:
    return compute_sigma_peel_MPa_at(D_nm)['sigma_peel_MPa']

def _cu_stress_at(D_nm: float) -> float:
    return compute_cu_peel_cool_MPa_at(D_nm)['sigma_cu_peel_MPa']

def invert_dishing_sio2_given_sigma_eff(sigma_eff_MPa: float) -> Tuple[float, dict]:
    """
    Fixed window inversion for SiO2:
      - search D in [0, 10] nm only
      - if no root in window -> return 0
    Assumption: sigma_sio2(D) monotone DECREASING in D over [0,10].
    """
    target = float(sigma_eff_MPa)
    lo, hi = 0.0, 10.0

    D_raw = _bisect_mono(_sio2_stress_at, target, lo, hi, is_increasing=False)
    if D_raw is None:
        return 0.0, dict(
            mode="no_root_in_window",
            lo=lo, hi=hi, target=target,
            f_lo=_sio2_stress_at(lo),
            f_hi=_sio2_stress_at(hi)
        )
    return float(D_raw), dict(mode="ok", lo=lo, hi=hi, target=target)

def invert_dishing_cu_given_sigma_eff(sigma_eff_MPa: float) -> Tuple[float, dict]:
    """
    Fixed window inversion for Cu:
      - search D in [0, D_contact_max] only, where D_contact_max = delta_heat/2
      - if no root in window -> return D_contact_max
    Assumption: sigma_cu(D) monotone INCREASING in D over [0, D_contact_max).
    """
    target = float(sigma_eff_MPa)

    # delta_heat from paper heat stage (independent of D), computed via D=0 call
    hot0 = compute_sigma_peel_MPa_at(0.0)
    delta_heat_nm = float(hot0.get("delta_heat_nm", hot0.get("delta_eq_nm", 0.0)))
    D_contact_max = max(0.0, 0.5 * delta_heat_nm)

    if D_contact_max <= 0.0:
        return 0.0, dict(mode="no_contact_domain", D_contact_max=D_contact_max, target=target)

    # Avoid evaluating exactly at D_contact_max where phi->0
    hi_eval = float(np.nextafter(D_contact_max, 0.0))
    if hi_eval <= 0.0:
        # contact domain exists but too tiny numerically
        return float(D_contact_max), dict(mode="tiny_contact_domain", D_contact_max=D_contact_max, target=target)

    lo, hi = 0.0, hi_eval

    def f_clip(D_nm: float) -> float:
        # extra safety clip
        return _cu_stress_at(min(float(D_nm), hi_eval))

    D_raw = _bisect_mono(f_clip, target, lo, hi, is_increasing=True)
    if D_raw is None:
        return float(D_contact_max), dict(
            mode="no_root_in_window",
            lo=lo, hi=D_contact_max, hi_eval=hi_eval,
            target=target, D_contact_max=D_contact_max,
            f_lo=f_clip(lo), f_hi=f_clip(hi)
        )
    return float(D_raw), dict(mode="ok", lo=lo, hi=D_contact_max, hi_eval=hi_eval, target=target, D_contact_max=D_contact_max)


# =============================================================================
# ============================ WAFER-LEVEL STACK ==============================
# =============================================================================

def equiv_from_three(mix: LayerMix3) -> EqLayer:
    V1,V2,V3 = mix.V1,mix.V2,mix.V3
    totalV = V1+V2+V3
    if totalV==0.0:
        raise ValueError("Sum of volumes must be >0.")
    aeq = (mix.mat1.alpha_perC*V1 + mix.mat2.alpha_perC*V2 + mix.mat3.alpha_perC*V3)/totalV
    Eeq = (mix.mat1.E_Pa      *V1 + mix.mat2.E_Pa      *V2 + mix.mat3.E_Pa      *V3)/totalV
    nueq= (mix.mat1.nu        *V1 + mix.mat2.nu        *V2 + mix.mat3.nu        *V3)/totalV
    return EqLayer(E_Pa=Eeq, alpha_perC=aeq, nu=nueq, t_m=mix.t_m)

def warpage_D_two_layer_exact(L_m,t_c_m,t_s_m,E_c,E_s,alpha_c,alpha_s,T_C,T0_C):
    ratio = t_c_m / t_s_m
    dT = (T_C - T0_C)
    num_pref = (3.0 * (L_m ** 2)) / (4.0 * (t_c_m + t_s_m))
    numerator = num_pref * ((1.0 + ratio) ** 2) * (alpha_s - alpha_c) * dT
    denom_left  = 3.0 * (1.0 + t_c_m / t_s_m) ** 2
    denom_right = (1.0 + (t_c_m * E_c) / (t_s_m * E_s)) * ((t_c_m ** 2) / (t_s_m ** 2) + (t_s_m * E_s) / (t_c_m * E_c))
    denominator = denom_left + denom_right
    if denominator == 0.0:
        raise ZeroDivisionError("Denominator zero.")
    return numerator / denominator

def combine_two_layers_to_one(top_eq: EqLayer, bot_eq: EqLayer) -> EqLayer:
    Vt,Vs = top_eq.t_m, bot_eq.t_m
    total=Vt+Vs
    if total==0.0:
        raise ValueError("Total thickness is zero.")
    aeq = (top_eq.alpha_perC*Vt + bot_eq.alpha_perC*Vs)/total
    Eeq = (top_eq.E_Pa      *Vt + bot_eq.E_Pa      *Vs)/total
    nueq= (top_eq.nu        *Vt + bot_eq.nu        *Vs)/total
    return EqLayer(E_Pa=Eeq, alpha_perC=aeq, nu=nueq, t_m=total)

@dataclass(frozen=True)
class WaferResult:
    D_m: float
    final_eq: EqLayer

def process_wafer(cfg: WaferConfig) -> WaferResult:
    top_eq = equiv_from_three(cfg.top)
    bot_eq = equiv_from_three(cfg.bottom)

    D = warpage_D_two_layer_exact(cfg.L_m, top_eq.t_m, bot_eq.t_m,
                                  top_eq.E_Pa, bot_eq.E_Pa,
                                  top_eq.alpha_perC, bot_eq.alpha_perC,
                                  cfg.T_C, cfg.T0_C)
    final_eq = combine_two_layers_to_one(top_eq, bot_eq)
    return WaferResult(D_m=D, final_eq=final_eq)

def plate_bending_stiffness(E: float, nu: float, h: float) -> float:
    return E * h**3 / (12.0 * (1.0 - nu**2))

def foundation_stiffness_K_effective(E1,nu1,h1,E2,nu2,h2):
    return 1.0 / ((1.0-nu1)*h1/(3.0*E1) + (1.0-nu2)*h2/(3.0*E2))

def suhir_peeling_two_wafers_bottomA_topB(waferA_eq: EqLayer, waferB_eq: EqLayer, R_m: float,
                                         sag_total_A_m: float, sag_total_B_m: float,
                                         sample_points: int = 500):
    D1 = plate_bending_stiffness(waferA_eq.E_Pa, waferA_eq.nu, waferA_eq.t_m)
    D2 = plate_bending_stiffness(waferB_eq.E_Pa, waferB_eq.nu, waferB_eq.t_m)
    K  = foundation_stiffness_K_effective(waferA_eq.E_Pa, waferA_eq.nu, waferA_eq.t_m,
                                          waferB_eq.E_Pa, waferB_eq.nu, waferB_eq.t_m)
    kappa1 = 2.0 * sag_total_A_m / (R_m**2)
    kappa2 = 2.0 * sag_total_B_m / (R_m**2)
    M = (D1 * D2) / (D1 + D2) * (kappa1 - kappa2)
    beta = ((K * (D1 + D2)) / (4.0 * D1 * D2)) ** 0.25
    p_max = K * M / (2.0 * beta * D1)  # [Pa]
    decay_len = 1.0 / beta
    return {"p_max_Pa": p_max, "beta": beta, "decay_length_m": decay_len}

def peeling_stress_at_points_vec_MPa(peel_dict: dict, coords_mm_np: np.ndarray, R_m: float) -> np.ndarray:
    if coords_mm_np.ndim != 2 or coords_mm_np.shape[1] != 2:
        raise ValueError("coords_mm_np must be shape (N,2).")
    xy_m = coords_mm_np.astype(np.float64, copy=False) * 1e-3
    r_m  = np.sqrt(xy_m[:,0]**2 + xy_m[:,1]**2)
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
# ===================== EFFICIENT CRITICAL & DISHING ARRAYS ===================
# =============================================================================

def build_effcrit_and_dishing_arrays(peel_dict: dict, coords_mm_np: np.ndarray, R_m: float)\
        -> Tuple[np.ndarray, np.ndarray]:
    p_MPa = peeling_stress_at_points_vec_MPa(peel_dict, coords_mm_np, R_m)   # (N,)
    crits = compute_critical_peeling_all()
    sigma_crit_SiO2 = float(crits["sigma_crit_MPa"]["SiO2"])
    sigma_crit_Cu   = float(crits["sigma_crit_MPa"]["Cu"])
    sigma_eff_SiO2 = sigma_crit_SiO2 - p_MPa
    sigma_eff_Cu   = sigma_crit_Cu   - p_MPa

    N = p_MPa.shape[0]
    D_sio2_nm = np.empty(N, dtype=np.float64)
    D_cu_nm   = np.empty(N, dtype=np.float64)

    for i in range(N):
        D_sio2_nm[i], _ = invert_dishing_sio2_given_sigma_eff(float(sigma_eff_SiO2[i]))
        D_cu_nm[i],   _ = invert_dishing_cu_given_sigma_eff(float(sigma_eff_Cu[i]))

    effcrit = np.column_stack([sigma_eff_Cu, sigma_eff_SiO2])
    dishing = np.column_stack([D_cu_nm, D_sio2_nm])
    return effcrit, dishing

# =============================================================================
# ================================= MAIN ======================================
# =============================================================================

def debond_dishing_bounds_calculator(cfg, coords_um):
    __init_params(cfg)

    # 1) Wafer-level stack to get peeling kernel
    resA = process_wafer(WAFER_A)  # bottom
    resB = process_wafer(WAFER_B)  # top

    # NOTE: sign convention kept as previous code (s_total = s_init - D)
    s_total_A_m = S_INIT_A_M - resA.D_m
    s_total_B_m = S_INIT_B_M - resB.D_m
    R_stack = min(WAFER_A.L_m, WAFER_B.L_m)

    peel = suhir_peeling_two_wafers_bottomA_topB(
        waferA_eq=resA.final_eq,
        waferB_eq=resB.final_eq,
        R_m=R_stack,
        sag_total_A_m=s_total_A_m,
        sag_total_B_m=s_total_B_m,
        sample_points=500
    )

    # 2) Use manual coords (µm) and convert to mm
    coords_um = np.asarray(coords_um, dtype=np.float64).reshape(-1, 2)
    if coords_um.size == 0:
        raise ValueError("Pad coords used for debond dishing bounds calculation is empty!")
    coords_mm = coords_um * 1e-3

    # 3) Build arrays and invert to dishing
    _, dishing_array = build_effcrit_and_dishing_arrays(
        peel_dict=peel,
        coords_mm_np=coords_mm.astype(np.float64, copy=False),
        R_m=R_stack,
    )

    # 4) Sort each row ascending (small first, large second) and return
    dishing_sorted = np.sort(dishing_array, axis=1)
    return dishing_sorted
