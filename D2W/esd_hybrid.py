# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================
# Weibull & simulate (globals)
# =========================
V_CHARGING_V     = 3.0
WEIBULL_K        = 3.981285
WEIBULL_LAMBDA   = 0.224454
CUTOFF_MIN_A     = 0.0

# =========================
# Units
# =========================
NM_TO_UM = 1e-3  # 1 nm = 1e-3 µm

# =========================
# 默认 Top/Bottom die 与 pad 输入（可在 main 中覆盖）
# =========================
TOP_DIE_W_UM: float = 10_000.0
TOP_DIE_H_UM: float = 10_000.0

PAD_SIZE_UM:  float = 50.0     # pad 真正边长
PAD_PITCH_UM: float = 100.0    # 可视化像素边长

# 如你有自己的 PAD_COORDS_UM，可以直接赋值覆盖
PAD_COORDS_UM: List[Tuple[float, float]] = []

# =========================
# Dishing 分布（demo1 用；demo2 外部传入）
# =========================
TOP_DISH_MEAN_NM: float = -4.0
TOP_DISH_STD_NM:  float =  2.5
BOT_DISH_MEAN_NM: float = -4.0
BOT_DISH_STD_NM:  float =  2.5

# 倾角（度）
TILT_X_MEAN_DEG = 0.000
TILT_X_STD_DEG  = 0.01
TILT_Y_MEAN_DEG = 0.000
TILT_Y_STD_DEG  = 0.01

# 采样设置（demo1 用；demo2 只做一次）
N_TILTS         = 5
N_DISHES        = 5
BASE_SEED       = 20251006


# =========================
# Helpers
# =========================
def _z_linear_coeffs(ax_deg: float, ay_deg: float):
    """R = Ry(ay) @ Rx(ax): z' = z0 + A*x + B*y + C*h"""
    ax = np.deg2rad(float(ax_deg)); ay = np.deg2rad(float(ay_deg))
    ca, sa = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    A = -sy
    B =  cy * sa
    C =  cy * ca
    return float(A), float(B), float(C)

def _ipeak_from_die_voltage(area_mm2: float, v_chg: float) -> float:
    # 你的缩放版公式
    return 0.0045 * (float(area_mm2) ** 0.35) * math.sqrt(float(v_chg))

def _weibull_cdf(I: float, k: float, lam: float) -> float:
    I = max(I, 1e-12)
    return max(0.0, min(1.0, 1.0 - math.exp(- (I/lam)**k)))

def _fail_prob_single(I: float, k: float, lam: float, cutoff: float) -> float:
    if I < cutoff:
        return 0.0
    return _weibull_cdf(I, k, lam)

def _four_corners(cx: np.ndarray, cy: np.ndarray, half: float):
    """返回四角：LL, LR, UR, UL  → 形状 (N,4)"""
    x4 = np.stack([cx - half, cx + half, cx + half, cx - half], axis=1)
    y4 = np.stack([cy - half, cy - half, cy + half, cy + half], axis=1)
    return x4, y4

def _compute_p_fail_for_die(top_die_w_um: float, top_die_h_um: float) -> float:
    area_mm2 = (float(top_die_w_um) * 1e-3) * (float(top_die_h_um) * 1e-3)
    I_peak   = _ipeak_from_die_voltage(area_mm2, float(V_CHARGING_V))
    return _fail_prob_single(I_peak, float(WEIBULL_K), float(WEIBULL_LAMBDA), float(CUTOFF_MIN_A))

def _choose_one_pad(pids: np.ndarray, rng: np.random.Generator) -> Optional[int]:
    """在多个 pad index 中随机选一个，若空返回 None。"""
    if pids is None or pids.size == 0:
        return None
    idx = rng.integers(0, pids.size)
    return int(pids[idx])


# =========================
# 几何核心（仅 pad 四角；不再考虑 die 四角；不再做角度细化）
# =========================
def find_min_gap_pads_only(
    *,
    pad_coords_um: np.ndarray,   # (Npad,2)
    pad_size_um: float,          # 正方形边长
    z_top_um: float,             # 旋转前 top面基准高度（足够大）
    tilt_x_deg: float,
    tilt_y_deg: float,
    top_dish_um_raw: np.ndarray, # (Npad,) —— 原始值（未取反）
    bot_dish_um: np.ndarray,     # (Npad,)
    verbose: bool = False,
) -> dict:
    """
    仅计算 pad 四角的最小间隙。
    入围条件：raw 值 (top_dish_raw + bot_dish) >= 0。
    旋转高度里 top dishing 取反：top_eff = -top_dish_raw。
    若入围 pad 为空，立即返回空结果。
    """
    z0 = float(z_top_um)
    A, B, C = _z_linear_coeffs(tilt_x_deg, tilt_y_deg)

    Npad = pad_coords_um.shape[0]
    cx = pad_coords_um[:,0].astype(np.float32)
    cy = pad_coords_um[:,1].astype(np.float32)

    # 入围：raw 值
    mask = (top_dish_um_raw + bot_dish_um) >= 0.0
    if not np.any(mask):
        if verbose:
            print("[MinGap] no eligible pads; skip this round")
        return {
            "has_pad": False,
            "min_value_um": np.nan,
            "pad_ids_min_equal": np.zeros((0,), dtype=int),
            "counts": {"pad_corner": 0},
        }

    sel = np.where(mask)[0]
    half = 0.5*float(pad_size_um)
    x4, y4 = _four_corners(cx[sel], cy[sel], half)

    # 旋转高度里 top dishing 取反
    top_eff = (-top_dish_um_raw[sel])[:, None]
    bot_loc = (bot_dish_um[sel])[:, None]

    z_top = z0 + A*x4 + B*y4 + C*top_eff
    gaps  = z_top - bot_loc

    vmin_pad = float(np.nanmin(gaps))
    hit_m, hit_k = np.where(gaps == vmin_pad)
    pad_ids = np.unique(sel[hit_m].astype(int))

    if verbose:
        print(f"[MinGap] min={vmin_pad:.6e} µm; pad_eq={pad_ids.size}")

    return {
        "has_pad": True,
        "min_value_um": float(vmin_pad),
        "pad_ids_min_equal": pad_ids,
        "counts": {"pad_corner": pad_ids.size},
    }


# =========================
# demo1: risk pad map 生成器（使用单次原始角度；每组 top/bot dishing 只评一次）
# =========================
def pad_esd_yield_map_generator(
    *,
    cfg,  # 占位，为兼容旧调用
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    pad_pitch_um: float,
    top_die_w_um: float,
    top_die_h_um: float,
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
    base_seed: int = 20251006,
    z_top_um: float = 100.0,
) -> Tuple[np.ndarray, Optional[plt.Figure], float]:
    """
    输出：
      - risk_map_vec: 每个 pad 的 [首触概率 × p_fail_single] 值（长度 = Npad）
      - fig: 可视化（pitch 方块）（此处返回 None，占位）
      - p_fail_single: 该 die 尺寸下的失效率（便于复用）

    变化点：
      - 仅按原始 tilt 计算一次，不做等分细化；
      - 仅 pad 四角参与比较；若入围为空则直接跳过本轮；
      - 多个 pad 并列时随机挑 1 个。
    """
    Npad = pad_coords_um.shape[0]
    counts_vec = np.zeros((Npad,), dtype=np.int64)

    rng_tilt = np.random.default_rng(base_seed ^ 0xC001FEED)
    total_runs = int(n_tilts) * int(n_dishes)

    for t in range(n_tilts):
        tx = float(rng_tilt.normal(tilt_x_mean_deg, tilt_x_std_deg))
        ty = float(rng_tilt.normal(tilt_y_mean_deg, tilt_y_std_deg))

        for d in range(n_dishes):
            seed = base_seed + (t * n_dishes + d)
            rng_top  = np.random.default_rng(seed ^ 0x9E3779B1)
            rng_bot  = np.random.default_rng(seed ^ 0x85EBCA77)
            rng_pick = np.random.default_rng(seed ^ 0xDEADBEEF)

            top_dish_um_raw = rng_top.normal(
                loc=float(top_dish_mean_nm)*NM_TO_UM,
                scale=max(float(top_dish_std_nm),0.0)*NM_TO_UM,
                size=(Npad,)
            ).astype(np.float32)
            bot_dish_um     = rng_bot.normal(
                loc=float(bot_dish_mean_nm)*NM_TO_UM,
                scale=max(float(bot_dish_std_nm),0.0)*NM_TO_UM,
                size=(Npad,)
            ).astype(np.float32)

            out = find_min_gap_pads_only(
                pad_coords_um=pad_coords_um,
                pad_size_um=pad_size_um,
                z_top_um=z_top_um,
                tilt_x_deg=tx,
                tilt_y_deg=ty,
                top_dish_um_raw=top_dish_um_raw,
                bot_dish_um=bot_dish_um,
                verbose=False
            )

            if not out.get("has_pad", False):
                # 入围为空，本轮直接跳过
                continue

            pids = out.get("pad_ids_min_equal", np.zeros((0,), dtype=int))
            pick = _choose_one_pad(pids, rng_pick)
            if pick is not None:
                counts_vec[pick] += 1

    # 首触概率
    prob_vec = counts_vec.astype(np.float32) / float(total_runs) if total_runs>0 else np.zeros_like(counts_vec, dtype=np.float32)
    print("Prob_vec min/max: {:.6f} / {:.6f}".format(float(prob_vec.min()), float(prob_vec.max())))

    # 该尺寸失效率
    p_fail_single = _compute_p_fail_for_die(top_die_w_um, top_die_h_um)

    # risk pad map
    valid_pad_risk_map_vec = prob_vec * float(p_fail_single)
    print("Risk map min/max: {:.6e} / {:.6e}".format(float(valid_pad_risk_map_vec.min()), float(valid_pad_risk_map_vec.max())))

    fig = None  # 如需可视化可用 plot_probability_over_pads_with_pitch
    valid_pad_yield_map_vec = 1.0 - valid_pad_risk_map_vec
    return valid_pad_yield_map_vec, fig, float(p_fail_single)


# =========================
# demo2: 单次外部数组模拟（仅原始角度）
# =========================
def esd_failure_simulator(
    *,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    top_die_w_um: float,
    top_die_h_um: float,
    top_dish_nm_ext: np.ndarray,  # 外部给定，上层每个 pad 的 dishing（nm）
    bot_dish_nm_ext: np.ndarray,  # 外部给定，下层每个 pad 的 dishing（nm）
    tilt_x_mean_deg: float,
    tilt_x_std_deg: float,
    tilt_y_mean_deg: float,
    tilt_y_std_deg: float,
    base_seed: int = 20251006,
    z_top_um: float = 100.0,
) -> Tuple[Optional[int], bool]:
    """
    仅随机选取一组 tilt 角度，做一次实验。
    返回：
      - pad_index 或 None
      - 当前尺寸的生存结果（True=生存，False=失效）
    变化点：
      - 仅按原始 tilt 计算一次；
      - 仅 pad 四角；若入围为空则直接“无首触 pad”，按概率判定生存。
    """
    assert pad_coords_um.shape[0] == top_dish_nm_ext.shape[0] == bot_dish_nm_ext.shape[0], \
        "The length of pad_coords_um, top_dish_nm_ext and bot_dish_nm_ext must be equal."

    rng      = np.random.default_rng(base_seed ^ 0xA5A5A5A5)
    rng_pick = np.random.default_rng((base_seed ^ 0xA5A5A5A5) ^ 0xDEADBEEF)

    tilt_x = float(rng.normal(tilt_x_mean_deg, tilt_x_std_deg))
    tilt_y = float(rng.normal(tilt_y_mean_deg, tilt_y_std_deg))

    top_dish_um_raw = (top_dish_nm_ext.astype(np.float32)) * NM_TO_UM
    bot_dish_um     = (bot_dish_nm_ext.astype(np.float32)) * NM_TO_UM

    out = find_min_gap_pads_only(
        pad_coords_um=pad_coords_um,
        pad_size_um=pad_size_um,
        z_top_um=z_top_um,
        tilt_x_deg=tilt_x,
        tilt_y_deg=tilt_y,
        top_dish_um_raw=top_dish_um_raw,
        bot_dish_um=bot_dish_um,
        verbose=False
    )

    pad_index: Optional[int] = None
    if out.get("has_pad", False):
        pids = out.get("pad_ids_min_equal", np.zeros((0,), dtype=int))
        pad_index = _choose_one_pad(pids, rng_pick)

    p_fail_single = _compute_p_fail_for_die(top_die_w_um, top_die_h_um)
    random_float = np.random.uniform(0.0, 1.0)  # decide if failure occurs

    if (pad_index is not None) and (random_float < p_fail_single):
        survive_bool = False
    else:
        survive_bool = True

    return pad_index, survive_bool


# =========================
# 可视化：按 pitch 为边长的小方块
# =========================
def plot_probability_over_pads_with_pitch(
    pad_coords_um: np.ndarray,
    prob_vec: np.ndarray,
    *,
    pitch_um: float,
    title: str = "Pad Selection Probability (squares @ pitch)"
) -> plt.Figure:
    """按“pitch”为边长的小方块标记（与真实 pad 几何解耦）。"""
    fig, ax = plt.subplots()
    try:
        fig.canvas.toolbar_visible = True
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
    except Exception:
        pass

    vmax = float(prob_vec.max()) if prob_vec.size>0 else 1.0
    half_pix = 0.5*float(pitch_um)

    for (x, y), p in zip(pad_coords_um, prob_vec):
        if p <= 0:
            continue
        rect = Rectangle((x - half_pix, y - half_pix),
                         2*half_pix, 2*half_pix, linewidth=0.0)
        ax.add_patch(rect)
        rect.set_facecolor(plt.cm.viridis(p / vmax))
        rect.set_edgecolor('none')

    ax.set_aspect('equal', adjustable='box')
    halfW, halfH = TOP_DIE_W_UM/2, TOP_DIE_H_UM/2
    ax.set_xlim(-halfW, halfW)
    ax.set_ylim(-halfH, halfH)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("x (µm) — center at 0")
    ax.set_ylabel("y (µm) — top is smaller")

    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Risk = P_first_touch × p_fail_single")
    return fig


# =========================
# Main（演示）
# =========================
if __name__ == "__main__":
    # 若未提供坐标，生成一个 pitch 网格做 demo
    if not PAD_COORDS_UM:
        halfW, halfH = TOP_DIE_W_UM/2, TOP_DIE_H_UM/2
        xs = np.arange(-halfW + PAD_PITCH_UM*0.5, halfW, PAD_PITCH_UM)
        ys = np.arange( halfH - PAD_PITCH_UM*0.5, -halfH, -PAD_PITCH_UM)
        X, Y = np.meshgrid(xs, ys)
        PAD_COORDS_UM = list(zip(X.ravel().tolist(), Y.ravel().tolist()))
    pad_coords = np.asarray(PAD_COORDS_UM, dtype=np.float32).reshape(-1, 2)

    print("\n=== DEMO2: single-run with external dishing arrays ===")
    rng = np.random.default_rng(BASE_SEED ^ 0x13579BDF)
    Npad = pad_coords.shape[0]
    top_ext_nm = rng.normal(TOP_DISH_MEAN_NM, TOP_DISH_STD_NM, size=Npad).astype(np.float32)
    bot_ext_nm = rng.normal(BOT_DISH_MEAN_NM, BOT_DISH_STD_NM, size=Npad).astype(np.float32)

    pad_idx, survive = esd_failure_simulator(
        pad_coords_um=pad_coords,
        pad_size_um=PAD_SIZE_UM,
        top_die_w_um=TOP_DIE_W_UM,
        top_die_h_um=TOP_DIE_H_UM,
        top_dish_nm_ext=top_ext_nm,
        bot_dish_nm_ext=bot_ext_nm,
        tilt_x_mean_deg=TILT_X_MEAN_DEG,
        tilt_x_std_deg=TILT_X_STD_DEG,
        tilt_y_mean_deg=TILT_Y_MEAN_DEG,
        tilt_y_std_deg=TILT_Y_STD_DEG,
        base_seed=BASE_SEED,
        z_top_um=100.0,
    )
    print(f"first-touch pad index: {pad_idx}")
    print(f"survive? {survive}")
