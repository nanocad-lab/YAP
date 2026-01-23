#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debond.py  (produce dishing intervals from manual coord list)

- Remove ALL external wafer_info reading.
- User defines pad global coordinates (in µm) at the caller.
- Output: numpy.ndarray (N,2), each row = sorted (D_Cu_nm, D_SiO2_nm).

2025-11-02 update (per user requests):
1) Remove DISHING_NM default (cfg.DISH_0_m * 1e9). No global dishing is kept.
2) Inversion only searches within [-10, 50] nm.
   - SiO2: if root exists → return max(0, root); else → 0
   - Cu  : if root exists → return min(root, H_single); else → H_single
   where H_single = 0.5 * Δ_eq_nm at heat dwell and is independent of dishing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
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
           SIGMA_Y_MPA, R_P, R_C, CREEP_FACTOR, \
           TCU_BASE_UM, TCU_K_PER_C, PHI_CU0, ETA_GROWTH, T_INT_NM, \
           COOL_BAUSINGER_REDUCTION, COOL_A_HARD_MPA, COOL_K_HARD_PER_MPA, KN_COOL_GAIN, \
           VISC_LAMBDA, VISC_EXP, TCU_CONST_UM_COOL, KDISH_TCU_GAIN_PER_NM, \
           CRIT_aY2_UM, GC_SIO2_JPM2, GC_CU_JPM2, Effective_Contact_Area, \
           MAT_CU, MAT_SiO2, MAT_Si, \
           WAFER_A, WAFER_B, \
           S_INIT_A_M, S_INIT_B_M, \
           USE_PLOT

    # ---------- (A) Pad-scale: Geometry & Temps ----------
    if cfg.PAD_ARRANGE_PATTERN == 'checkerboard':
        PITCH_UM = min(np.sqrt(cfg.PITCH_r_um ** 2 + cfg.PITCH_c_um ** 2), 2 * cfg.PITCH_r_um, 2 * cfg.PITCH_c_um)
    else:
        PITCH_UM = cfg.PITCH_r_um  

    DIAM_UM       = cfg.PAD_TOP_R_um * 2      # pad diameter d [µm]
    T_ANNEAL_C    = cfg.T_anl                 # anneal temperature [°C]
    T_REF_C       = cfg.T_R                   # reference temperature [°C]
    # >>> REMOVED (per request): no DISHING_NM from cfg.DISH_0_m

    # ---------- (B) Pad-scale: Material constants ----------
    CU_E_GPA      = cfg.CU_E_GPA
    CU_NU         = cfg.CU_NU
    CU_ALPHA_PPM  = cfg.CU_ALPHA_PPM
    OX_E_GPA      = cfg.OX_E_GPA
    OX_NU         = cfg.OX_NU
    OX_ALPHA_PPM  = cfg.OX_ALPHA_PPM

    # ---------- (C) Pad-scale: Modeling constants (9) ----------
    SIGMA_Y_MPA   = cfg.SIGMA_Y_MPA
    R_P           = cfg.R_P
    R_C           = cfg.R_C
    CREEP_FACTOR  = cfg.CREEP_FACTOR

    TCU_BASE_UM   = cfg.TCU_BASE_UM
    TCU_K_PER_C   = cfg.TCU_K_PER_C
    PHI_CU0       = cfg.PHI_CU0
    ETA_GROWTH    = cfg.ETA_GROWTH
    T_INT_NM      = cfg.T_INT_NM

    # ---------- (D) Pad-scale: Cool-down (Cu–Cu) ----------
    COOL_BAUSINGER_REDUCTION = cfg.COOL_BAUSINGER_REDUCTION
    COOL_A_HARD_MPA          = cfg.COOL_A_HARD_MPA
    COOL_K_HARD_PER_MPA      = cfg.COOL_K_HARD_PER_MPA
    KN_COOL_GAIN             = cfg.KN_COOL_GAIN
    VISC_LAMBDA = cfg.VISC_LAMBDA
    VISC_EXP    = cfg.VISC_EXP
    TCU_CONST_UM_COOL       = cfg.TCU_CONST_UM_COOL
    KDISH_TCU_GAIN_PER_NM   = cfg.KDISH_TCU_GAIN_PER_NM

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
    MAT_SiO2 = Material("SiO2", E_Pa=cfg.OX_E_GPA*1e9,  alpha_perC=cfg.OX_ALPHA_PPM*1e-6,  nu=cfg.OX_NU)
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
    A_cell=p**2; A_cu=math.pi*(d**2)/4.0; A_ox=A_cell-A_cu
    if A_ox<=0: raise ValueError("A_ox<=0, check PITCH_UM and DIAM_UM values.")
    return A_cell, A_cu, A_ox

def _fill_fraction(p_um,d_um):
    A_cell, A_cu, _ = _geom_areas(p_um,d_um)
    f=A_cu/A_cell
    return max(1e-12,min(0.999999999,f))

def _thermal_mismatch_sigma(CU_E_GPA, CU_NU, CU_ALPHA_PPM, OX_ALPHA_PPM, T_C, T_ref):
    U=_units(); dT=T_C-T_ref
    M_cu=CU_E_GPA*U['GPa']/(1.0-CU_NU)
    delta_alpha=(CU_ALPHA_PPM-OX_ALPHA_PPM)*1e-6
    return M_cu*delta_alpha*dT

def _split_sigma(sigma_m_Pa, SIGMA_Y_MPA, CREEP_FACTOR):
    U=_units(); sigma_y=SIGMA_Y_MPA*U['MPa']
    sigma_e=max(-sigma_y,min(sigma_m_Pa,sigma_y))
    sigma_p=sigma_m_Pa-sigma_e
    sigma_c=CREEP_FACTOR*sigma_m_Pa
    return sigma_e,sigma_p,sigma_c

def _k_n_bar(CU_E_GPA, CU_NU, T_INT_NM):
    U=_units(); M_cu=(CU_E_GPA*U['GPa'])/(1.0-CU_NU); t_int=T_INT_NM*U['nm']
    return M_cu/t_int

def _tcu_of_T_exp(T_C, T_ref_C, TCU_BASE_UM, TCU_K_PER_C):
    return TCU_BASE_UM*(math.e**(TCU_K_PER_C*(T_C-T_ref_C)))

def _apply_dish_gain_to_tcu_linear(tcu_um, D_nm):
    # Still allow linear gain vs dishing for cool-down stage model
    return tcu_um*(1.0+KDISH_TCU_GAIN_PER_NM*max(0.0,float(D_nm)))

def _delta_eq_two_pads(sigma_e,sigma_p,sigma_c,CU_E_GPA,CU_NU,T_CU_UM_eff,R_P,R_C):
    U=_units(); t_cu=T_CU_UM_eff*U['um']
    k_e_eff=(2.0*CU_NU/(CU_E_GPA*U['GPa']))*t_cu
    S=sigma_e+R_P*sigma_p+R_C*sigma_c
    return 2.0*(k_e_eff*S)

def _phi_cu(delta_eq_m, dishing_nm, ETA_GROWTH, PHI_CU0):
    # Heat-dwell Cu contact ratio; depends on D for SIO2 path and Cu gating in cool model.
    U=_units(); delta_sat=2*dishing_nm*U['nm']
    if delta_eq_m<=delta_sat: return 0.0
    x=(delta_eq_m-delta_sat)/max(delta_sat,1e-12)
    return PHI_CU0+(1.0-PHI_CU0)*(x**ETA_GROWTH)

# ---------- New: purely D-independent heat-dwell delta_eq (for H_single) ----------
def compute_delta_eq_nm_only():
    U=_units()
    sigma_m=_thermal_mismatch_sigma(CU_E_GPA,CU_NU,CU_ALPHA_PPM,OX_ALPHA_PPM,T_ANNEAL_C,T_REF_C)
    sigma_e,sigma_p,sigma_c=_split_sigma(sigma_m,SIGMA_Y_MPA,CREEP_FACTOR)
    T_CU_UM_eff=_tcu_of_T_exp(T_ANNEAL_C,T_REF_C,TCU_BASE_UM,TCU_K_PER_C)
    delta_eq=_delta_eq_two_pads(sigma_e,sigma_p,sigma_c,CU_E_GPA,CU_NU,T_CU_UM_eff,R_P,R_C)
    return dict(
        delta_eq_nm = delta_eq/U['nm'],
        sigma_e_MPa = sigma_e/U['MPa'],
        sigma_p_MPa = sigma_p/U['MPa'],
        sigma_c_MPa = sigma_c/U['MPa'],
        tcu_eff_um  = T_CU_UM_eff
    )

# ---------- Heat-dwell SiO2 peeling stress at given D ----------
def compute_sigma_peel_MPa_at(D_nm: float):
    U=_units()
    _,A_cu,A_ox=_geom_areas(PITCH_UM,DIAM_UM)
    # reuse the same sigma_e/p/c and TCU from D-independent path:
    base = compute_delta_eq_nm_only()
    delta_eq = base['delta_eq_nm']*U['nm']
    phi_cu = _phi_cu(delta_eq, D_nm, ETA_GROWTH, PHI_CU0)
    k_n=_k_n_bar(CU_E_GPA,CU_NU,T_INT_NM)
    N_cu=k_n*delta_eq*(A_cu*phi_cu)
    sigma_peel=N_cu/A_ox
    return dict(
        sigma_peel_MPa=sigma_peel/U['MPa'],
        phi_cu=phi_cu,
        delta_eq_nm=base['delta_eq_nm'],
        tcu_eff_um=base['tcu_eff_um'],
        sigma_e_MPa=base['sigma_e_MPa'],
        sigma_p_MPa=base['sigma_p_MPa'],
        sigma_c_MPa=base['sigma_c_MPa'],
    )

def _sigma_y_cool_MPa(sigma_p_heat_MPa, sigma_y_heat_MPa):
    base=(1.0-COOL_BAUSINGER_REDUCTION)*float(sigma_y_heat_MPa)
    hard=COOL_A_HARD_MPA*(1.0-math.exp(-COOL_K_HARD_PER_MPA*max(0.0,float(sigma_p_heat_MPa))))
    return min(sigma_y_heat_MPa, base+hard)

# ---------- Cool-down Cu peeling stress at given D ----------
def compute_cu_peel_cool_MPa_at(D_nm: float):
    U=_units()
    hot = compute_sigma_peel_MPa_at(D_nm)  # note: phi_cu now from D
    if hot['phi_cu']<=0.0:
        return dict(sigma_cu_peel_MPa=0.0, reason="no_contact_in_heat_dwell",
                    sigma_y_cool_MPa=_sigma_y_cool_MPa(0.0, SIGMA_Y_MPA), delta_eff_nm=0.0,
                    phi_cu=hot['phi_cu'], delta_eq_nm=hot['delta_eq_nm'])

    sigma_m=_thermal_mismatch_sigma(CU_E_GPA,CU_NU,CU_ALPHA_PPM,OX_ALPHA_PPM,T_ANNEAL_C,T_REF_C)
    sigma_y_cool=_sigma_y_cool_MPa(hot['sigma_p_MPa'],SIGMA_Y_MPA); sigma_y_Pa=sigma_y_cool*1e6
    sigma_e_cool=max(-sigma_y_Pa,min(sigma_m,sigma_y_Pa))
    sigma_p_cool=sigma_m-sigma_e_cool
    # S_heat uses D-independent sigma_e/p/c values we cached in hot:
    S_heat=(hot['sigma_e_MPa']*1e6)+R_P*(hot['sigma_p_MPa']*1e6)+R_C*(hot['sigma_c_MPa']*1e6)
    S_cool=sigma_e_cool+R_P*sigma_p_cool

    T_CU_UM_eff_cool=_apply_dish_gain_to_tcu_linear(TCU_CONST_UM_COOL,D_nm)
    t_cu=T_CU_UM_eff_cool*1e-6
    k_e_eff_cool=(2.0*CU_NU/(CU_E_GPA*1e9))*t_cu
    delta_eq_cool=2.0*(k_e_eff_cool*(S_cool-S_heat))
    delta_eff=max(0.0,delta_eq_cool)
    k_n_cool=_k_n_bar(CU_E_GPA,CU_NU,T_INT_NM)*KN_COOL_GAIN

    # Use phi from heat at this D as the effective contact fraction (bounded)
    phi_eff=max(1e-3,min(1.0,float(hot['phi_cu'])))
    k_n_cool/=phi_eff

    f=_fill_fraction(PITCH_UM,DIAM_UM)
    visc_scale=1.0 - VISC_LAMBDA*(f**VISC_EXP)
    sigma_cu_peel=(k_n_cool*delta_eff*visc_scale)/1e6
    return dict(sigma_cu_peel_MPa=sigma_cu_peel,
                sigma_y_cool_MPa=sigma_y_cool,
                delta_eff_nm=delta_eff/1e-9,
                tcu_eff_um=TCU_CONST_UM_COOL,
                phi_cu=hot['phi_cu'],
                delta_eq_nm=hot['delta_eq_nm'])

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

# --- Helpers for monotone bisection within FIXED interval [-10, 50] only ---
def _bisect_mono(f, target, lo, hi, is_increasing, tol=1e-1, maxit=80):
    f_lo, f_hi = f(lo), f(hi)
    # Check if target lies within [f(lo), f(hi)] under monotonic assumption
    if is_increasing:
        if not (f_lo <= target <= f_hi): return None
    else:
        if not (f_hi <= target <= f_lo): return None
    lo0, hi0 = lo, hi
    for _ in range(maxit):
        mid = 0.5*(lo+hi); fm = f(mid)
        if abs(fm - target) <= tol: return mid
        if is_increasing:
            (lo, hi) = (mid, hi) if fm < target else (lo, mid)
        else:
            (lo, hi) = (mid, hi) if fm > target else (lo, mid)
    return 0.5*(lo+hi)

def _sio2_stress_at(D_nm: float) -> float:
    return compute_sigma_peel_MPa_at(D_nm)['sigma_peel_MPa']

def _cu_stress_at(D_nm: float) -> float:
    return compute_cu_peel_cool_MPa_at(D_nm)['sigma_cu_peel_MPa']

# --- New: heat single-side growth height H_single (independent of dishing) ---
def _single_side_growth_height_nm() -> float:
    base = compute_delta_eq_nm_only()
    return max(0.0, 0.5*float(base['delta_eq_nm']))

def invert_dishing_sio2_given_sigma_eff(sigma_eff_MPa: float) -> Tuple[float, dict]:
    target = float(sigma_eff_MPa)
    lo, hi = -10.0, 50.0  # fixed search window per request (dishing in nm)
    # σ_SiO2(D) is monotonically DECREASING vs D in this model
    D_raw = _bisect_mono(_sio2_stress_at, target, lo, hi, is_increasing=False)
    if D_raw is None:
        # no solution in [-10,50] → return 0 (per request)
        return 0.0, dict(mode="no_root_in_range", lo=lo, hi=hi, target=target)
    return max(0.0, D_raw), dict(mode="ok", lo=lo, hi=hi, target=target)

def invert_dishing_cu_given_sigma_eff(sigma_eff_MPa: float) -> Tuple[float, dict]:
    target = float(sigma_eff_MPa)
    lo, hi = -10.0, 50.0  # fixed search window per request (dishing in nm)
    H_single = _single_side_growth_height_nm()  # independent of D
    # σ_Cu(D) is monotonically INCREASING vs D in this model (within range)
    D_raw = _bisect_mono(_cu_stress_at, target, lo, hi, is_increasing=True)
    if D_raw is None:
        # no solution → return H_single (per request)
        return H_single, dict(mode="no_root_in_range", lo=lo, hi=hi, target=target, H_single=H_single)
    return min(D_raw, H_single), dict(mode="ok", lo=lo, hi=hi, target=target, H_single=H_single)

# =============================================================================
# ============================ WAFER-LEVEL STACK ==============================
# =============================================================================

def equiv_from_three(mix: LayerMix3) -> EqLayer:
    V1,V2,V3 = mix.V1,mix.V2,mix.V3
    totalV = V1+V2+V3
    if totalV==0.0: raise ValueError("Sum of volumes must be >0.")
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
    if denominator == 0.0: raise ZeroDivisionError("Denominator zero.")
    return numerator / denominator

def combine_two_layers_to_one(top_eq: EqLayer, bot_eq: EqLayer) -> EqLayer:
    Vt,Vs = top_eq.t_m, bot_eq.t_m
    total=Vt+Vs
    if total==0.0: raise ValueError("Total thickness is zero.")
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

# -------- Vectorized peeling at arbitrary internal points (in mm) --------
def peeling_stress_at_points_vec_MPa(peel_dict: dict, coords_mm_np: np.ndarray, R_m: float) -> np.ndarray:
    """
    coords_mm_np: (N,2) in mm, center-origin; returns (N,) peeling stress in MPa.
    Raises if any point is outside the wafer (r > R).
    """
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
        # np.savez("peeling_stress_100warp.npz", r_m=r_m, p_pa=p_pa)
        # raise RuntimeError("Plotting peeling stress and saving data; interrupting for review.")
    return p_pa / 1e6  # MPa

# =============================================================================
# ===================== EFFICIENT CRITICAL & DISHING ARRAYS ===================
# =============================================================================

def build_effcrit_and_dishing_arrays(peel_dict: dict, coords_mm_np: np.ndarray, R_m: float)\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    effcrit_array:  (N,2) columns = (σ_eff_crit_Cu, σ_eff_crit_SiO2) [MPa]
    dishing_array:  (N,2) columns = (D*_Cu_nm,      D*_SiO2_nm)      [nm]
    """
    p_MPa = peeling_stress_at_points_vec_MPa(peel_dict, coords_mm_np, R_m)   # (N,)
    crits = compute_critical_peeling_all()
    sigma_crit_SiO2 = float(crits["sigma_crit_MPa"]["SiO2"])
    sigma_crit_Cu   = float(crits["sigma_crit_MPa"]["Cu"])
    sigma_eff_SiO2 = sigma_crit_SiO2 - p_MPa
    sigma_eff_Cu   = sigma_crit_Cu   - p_MPa

    # Precompute H_single (same for all points; heat-stage single-side growth)
    H_single = _single_side_growth_height_nm()

    N = p_MPa.shape[0]
    D_sio2_nm = np.empty(N, dtype=np.float64)
    D_cu_nm   = np.empty(N, dtype=np.float64)

    for i in range(N):
        # SiO2 inversion in [-10, 50] → max(0, root) or 0 if no root
        D_SiO2, _meta_s = invert_dishing_sio2_given_sigma_eff(float(sigma_eff_SiO2[i]))
        D_sio2_nm[i] = D_SiO2

        # Cu inversion in [-10, 50] → min(root, H_single) or H_single if no root
        D_Cu, _meta_c = invert_dishing_cu_given_sigma_eff(float(sigma_eff_Cu[i]))
        # _meta_c already applies min(root, H_single) or H_single; keep as-is
        D_cu_nm[i] = D_Cu

    if USE_PLOT:
        # Optional visualization of peeling stress field
        coords_um_np = coords_mm_np * 1000
        plt.figure(figsize=(13.5, 9), dpi=300)
        plt.scatter(coords_um_np[:, 0], coords_um_np[:, 1], c=p_MPa, cmap='viridis', s=8)
        cb = plt.colorbar(label='Peeling Stress p (MPa)')
        cb.ax.yaxis.label.set_size(16)
        plt.xlabel('X (μm)', fontsize=16)
        plt.ylabel('Y (μm)', fontsize=16)
        plt.title('Peeling Stress Distribution', fontsize=16)
        plt.axis('equal')
        plt.minorticks_on()

        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.yaxis.set_major_locator(MultipleLocator(1000))
        ax.xaxis.set_minor_locator(MultipleLocator(500))  # 次刻度间隔 0.25 mm
        ax.yaxis.set_minor_locator(MultipleLocator(500))
        ax.set_xticks(np.arange(-5500, 6000, 1000))
        ax.set_yticks(np.arange(-4500, 4600, 1000))
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.tick_params(which='major', length=6, labelsize=12)
        ax.tick_params(which='minor', length=3)
        ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.6)
        ax.grid(which='minor', linestyle='--', linewidth=0.4, alpha=0.4)
        plt.show()

        # coords_um_np = coords_mm_np * 1000

        # x = np.unique(coords_um_np[:, 0])
        # y = np.unique(coords_um_np[:, 1])
        # X, Y = np.meshgrid(x, y)

        # # 将 p_MPa 重塑为与坐标匹配的网格形状
        # Z = p_MPa.reshape(len(y), len(x))

        # plt.figure(figsize=(8, 6), dpi=100)
        # mesh = plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
        # plt.colorbar(mesh, label='Peeling Stress p (MPa)')
        # plt.xlabel('X (μm)')
        # plt.ylabel('Y (μm)')
        # plt.title('Peeling Stress Distribution')
        # plt.axis('equal')

        # ax = plt.gca()
        # ax.xaxis.set_major_locator(MultipleLocator(1000))
        # ax.yaxis.set_major_locator(MultipleLocator(1000))
        # ax.xaxis.set_minor_locator(MultipleLocator(500))
        # ax.yaxis.set_minor_locator(MultipleLocator(500))
        # ax.tick_params(which='both', direction='in', top=True, right=True)
        # ax.tick_params(which='major', length=6, labelsize=12)
        # ax.tick_params(which='minor', length=3)
        # ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.6)
        # ax.grid(which='minor', linestyle='--', linewidth=0.4, alpha=0.4)

        # plt.show()

    effcrit = np.column_stack([sigma_eff_Cu, sigma_eff_SiO2])      # (N,2)
    dishing = np.column_stack([D_cu_nm, D_sio2_nm])                # (N,2)
    return effcrit, dishing

# =============================================================================
# ================================= MAIN =====================================
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
