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

# Parameters for test
# =========================
# 默认 Top/Bottom die 与 pad 输入（可在 main 中覆盖）
# =========================
TOP_DIE_W_UM: float = 10_000.0
TOP_DIE_H_UM: float = 10_000.0

PAD_SIZE_UM:  float = 50.0     # pad 真正边长
PAD_PITCH_UM: float = 100.0    # 可视化像素边长

# 如你有自己的 PAD_COORDS_UM，可以直接赋值覆盖
PAD_COORDS_UM: List[Tuple[float, float]] = []

# PAD_COORDS_UM = [
#     ( -250.0,  300.0),  # index 0
#     (    0.0,    0.0),  # index 1
#     (  420.0, -150.0),  # index 2
# ]



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
N_STEPS_REFINE  = 10
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
    
def _choose_one_pad(pids: np.ndarray, rng: np.random.Generator) -> Optional[int]:
    """在多个 pad index 中随机选一个，若空返回 None。"""
    if pids is None or pids.size == 0:
        return None
    idx = rng.integers(0, pids.size)
    return int(pids[idx])


# =========================
# 几何核心：按坐标+边长，筛选 & 求最小间隙
# =========================
def find_min_gap_points(
    *,
    pad_coords_um: np.ndarray,   # (Npad,2)
    pad_size_um: float,          # 正方形边长
    top_die_w_um: float,
    top_die_h_um: float,
    z_top_um: float,             # 旋转前 top面基准高度（足够大）
    tilt_x_deg: float,
    tilt_y_deg: float,
    top_dish_um_raw: np.ndarray, # (Npad,) —— 原始值（未取反）
    bot_dish_um: np.ndarray,     # (Npad,)
    verbose: bool = False,
) -> dict:
    """
    确认点：
    (1) 入围筛选使用 raw：top_dish_raw + bot_dish >= 0。
    (2) 旋转高度里 top dishing 取反：top_dish_eff = -top_dish_raw。
    (3) 旋转绕 top die 面心（die中心），top基准高度 z0=100µm，bottom面=0µm。
    """
    z0 = float(z_top_um)
    A, B, C = _z_linear_coeffs(tilt_x_deg, tilt_y_deg)

    # die 四角（bottom 面高度=0）
    hw, hh = 0.5*float(top_die_w_um), 0.5*float(top_die_h_um)
    die_x = np.array([-hw,  hw,  hw, -hw], dtype=np.float32)
    die_y = np.array([-hh, -hh,  hh,  hh], dtype=np.float32)
    z_top_die = z0 + A*die_x + B*die_y
    gaps_die  = z_top_die  # - 0
    min_val = float(np.nanmin(gaps_die))
    recs = [{"kind":"die_corner","corner_index":int(i),
             "x_um":float(die_x[i]),"y_um":float(die_y[i]),
             "gap_um":float(gaps_die[i])}
            for i in np.where(gaps_die == min_val)[0].tolist()]

    # pad 四角
    Npad = pad_coords_um.shape[0]
    cx = pad_coords_um[:,0].astype(np.float32)
    cy = pad_coords_um[:,1].astype(np.float32)

    # (1) 入围：raw 值
    mask = (top_dish_um_raw + bot_dish_um) >= 0.0
    if np.any(mask):
        sel = np.where(mask)[0]
        half = 0.5*float(pad_size_um)
        x4, y4 = _four_corners(cx[sel], cy[sel], half)

        # (2) 旋转高度里 top dishing 取反
        top_eff = (-top_dish_um_raw[sel])[:,None]
        bot_loc =  (bot_dish_um[sel])[:,None]

        z_top = z0 + A*x4 + B*y4 + C*top_eff
        gaps  = z_top - bot_loc

        vmin_pad = float(np.nanmin(gaps))
        if vmin_pad < min_val:
            min_val = vmin_pad
            recs = []
        if vmin_pad <= min_val:
            hit_m, hit_k = np.where(gaps == vmin_pad)
            for m,k in zip(hit_m.tolist(), hit_k.tolist()):
                recs.append({
                    "kind":"pad_corner",
                    "pad_id": int(sel[m]),
                    "corner_index": int(k),
                    "x_um": float(x4[m,k]),
                    "y_um": float(y4[m,k]),
                    "gap_um": float(vmin_pad),
                })

    # 汇总
    die_eq = [r for r in recs if r["kind"]=="die_corner" and r["gap_um"]==min_val]
    pad_eq = [r for r in recs if r["kind"]=="pad_corner" and r["gap_um"]==min_val]
    pad_ids = np.unique(np.array([r["pad_id"] for r in pad_eq], dtype=int)) if pad_eq else np.zeros((0,),dtype=int)

    if verbose:
        print(f"[MinGap] min={min_val:.6e} µm; die_eq={len(die_eq)}; pad_eq={len(pad_eq)}")

    return {
        "min_value_um": float(min_val),
        "pad_ids_min_equal": pad_ids,
        "counts": {"die_corner": len(die_eq), "pad_corner": len(pad_eq)},
    }



# =========================
# 赢家类型与“等分减小角度”细化器
# =========================
def _winner_kind(out: dict) -> str:
    """返回 'pad' 或 'die'。"""
    pad_eq = out.get("counts", {}).get("pad_corner", 0)
    return "pad" if pad_eq > 0 else "die"

def _refine_by_equal_division(
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
    n_steps_refine: int,
) -> Tuple[float, float, dict]:
    """
    等分减小角度：每一步把角度减少 tilt_init / N_STEPS_REFINE，向 0° 收敛。
    一旦某个 pad 成为最小点即返回；否则到 0° 还无 pad，就返回最后状态（die）。
    """
    tx0 = float(tilt_x_init_deg)
    ty0 = float(tilt_y_init_deg)
    N = max(int(n_steps_refine), 0)

    # 初次评估
    out0 = find_min_gap_points(
        pad_coords_um=pad_coords_um, pad_size_um=pad_size_um,
        top_die_w_um=top_die_w_um, top_die_h_um=top_die_h_um,
        z_top_um=z_top_um,
        tilt_x_deg=tx0, tilt_y_deg=ty0,
        top_dish_um_raw=top_dish_um_raw, bot_dish_um=bot_dish_um,
        verbose=False
    )
    if _winner_kind(out0) == "pad" or N == 0:
        return tx0, ty0, out0

    # 等分减小角度，向 0° 逼近
    last_tx, last_ty, last_out = tx0, ty0, out0
    for i in range(1, N + 1):
        factor = 1.0 - (i / float(N))   # i=N 时正好到 0
        tx = tx0 * factor
        ty = ty0 * factor

        out = find_min_gap_points(
            pad_coords_um=pad_coords_um, pad_size_um=pad_size_um,
            top_die_w_um=top_die_w_um, top_die_h_um=top_die_h_um,
            z_top_um=z_top_um,
            tilt_x_deg=tx, tilt_y_deg=ty,
            top_dish_um_raw=top_dish_um_raw, bot_dish_um=bot_dish_um,
            verbose=False
        )
        if _winner_kind(out) == "pad":
            return tx, ty, out

        last_tx, last_ty, last_out = tx, ty, out

    return last_tx, last_ty, last_out




# =========================
# demo1: risk pad map 生成器
# =========================
def pad_esd_yield_map_generator(
    *,
    cfg,
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
    n_steps_refine: int = 10,
    base_seed: int = 20251006,
    z_top_um: float = 100.0,
) -> Tuple[np.ndarray, plt.Figure, float]:
    """
    输出：
      - risk_map_vec: 每个 pad 的 [首触概率 × p_fail_single] 值（长度 = Npad）
      - fig: 可视化（pitch 方块）
      - p_fail_single: 该 die 尺寸下的失效率（便于复用）
    """
    Npad = pad_coords_um.shape[0]
    counts_vec = np.zeros((Npad,), dtype=np.int64)

    rng_tilt = np.random.default_rng(base_seed ^ 0xC001FEED)
    total_runs = int(n_tilts) * int(n_dishes)

    for t in range(n_tilts):
        tx0 = float(rng_tilt.normal(tilt_x_mean_deg, tilt_x_std_deg))
        ty0 = float(rng_tilt.normal(tilt_y_mean_deg, tilt_y_std_deg))

        for d in range(n_dishes):
            seed = base_seed + (t * n_dishes + d)
            rng_top = np.random.default_rng(seed ^ 0x9E3779B1)
            rng_bot = np.random.default_rng(seed ^ 0x85EBCA77)
             # 选 pad 的随机器（确定性）
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

           # 等分细化
            _, _, out = _refine_by_equal_division(
                pad_coords_um=pad_coords_um,
                pad_size_um=pad_size_um,
                top_die_w_um=top_die_w_um,
                top_die_h_um=top_die_h_um,
                z_top_um=z_top_um,
                tilt_x_init_deg=tx0,
                tilt_y_init_deg=ty0,
                top_dish_um_raw=top_dish_um_raw,
                bot_dish_um=bot_dish_um,
                n_steps_refine=int(n_steps_refine),
            )

            # 只统计 pad 赢；并在可能的多个 pad 中随机挑选一个
            if _winner_kind(out) == "pad":
                pids = out.get("pad_ids_min_equal", np.zeros((0,), dtype=int))
                pick = _choose_one_pad(pids, rng_pick)
                if pick is not None:
                    counts_vec[pick] += 1

    # 首触概率
    prob_vec = counts_vec.astype(np.float32) / float(total_runs)

    print("Prob_vec min/max: {:.6f} / {:.6f}".format(float(prob_vec.min()), float(prob_vec.max())))
    
    # 该尺寸失效率
    p_fail_single = _compute_p_fail_for_die(top_die_w_um, top_die_h_um)
    
    # risk pad map
    valid_pad_risk_map_vec = prob_vec * float(p_fail_single)

    print("Risk map min/max: {:.6e} / {:.6e}".format(float(valid_pad_risk_map_vec.min()), float(valid_pad_risk_map_vec.max())))

    # # 画图（按 pitch 为边长）
    # fig = plot_probability_over_pads_with_pitch(
    #     pad_coords_um=pad_coords_um,
    #     prob_vec=valid_pad_risk_map_vec,
    #     pitch_um=pad_pitch_um,
    #     title="Risk Pad Map = P(first-touch) × p_fail (square side = PITCH)"
    # )
    fig = None

    valid_pad_yield_map_vec = 1.0 - valid_pad_risk_map_vec
    return valid_pad_yield_map_vec, fig, float(p_fail_single)


# =========================
# demo2: 单次外部数组模拟
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
    n_steps_refine: int = 10,
) -> Tuple[Optional[int], bool]:
    """
    仅随机选取一组 tilt 角度，做一次实验。
    返回：
      - pad_idx_list 或 None
      - 当前尺寸的 ESD failure rate（p_fail_single）
    说明：
      - 输入的 dishing 数组单位为 nm，这里统一转换为 µm 使用；
      - 入围筛选使用 raw 值；旋转时 top dishing 取反。
    """
    assert pad_coords_um.shape[0] == top_dish_nm_ext.shape[0] == bot_dish_nm_ext.shape[0], \
        "The length of pad_coords_um, top_dish_nm_ext and bot_dish_nm_ext must be equal."

    rng = np.random.default_rng(base_seed ^ 0xA5A5A5A5)
    rng_pick = np.random.default_rng((base_seed ^ 0xA5A5A5A5) ^ 0xDEADBEEF)

    tilt_x = float(rng.normal(tilt_x_mean_deg, tilt_x_std_deg))
    tilt_y = float(rng.normal(tilt_y_mean_deg, tilt_y_std_deg))

    top_dish_um_raw = (top_dish_nm_ext.astype(np.float32)) * NM_TO_UM
    bot_dish_um     = (bot_dish_nm_ext.astype(np.float32)) * NM_TO_UM

    _, _, out = _refine_by_equal_division(
        pad_coords_um=pad_coords_um,
        pad_size_um=pad_size_um,
        top_die_w_um=top_die_w_um,
        top_die_h_um=top_die_h_um,
        z_top_um=z_top_um,
        tilt_x_init_deg=tilt_x,
        tilt_y_init_deg=tilt_y,
        top_dish_um_raw=top_dish_um_raw,
        bot_dish_um=bot_dish_um,
        n_steps_refine=int(n_steps_refine),
    )

    pad_index: Optional[int] = None
    if _winner_kind(out) == "pad":
        pids = out.get("pad_ids_min_equal", np.zeros((0,), dtype=int))
        pad_index = _choose_one_pad(pids, rng_pick)

    p_fail_single = _compute_p_fail_for_die(top_die_w_um, top_die_h_um)
    random_float = np.random.uniform(0.0, 1.0)      # Used to decide if failure occurs
    if pad_index is not None and random_float < p_fail_single:
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
        if p <= 0:    # 只画非零
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

    # print("=== DEMO1: risk pad map ===")
    # risk_vec, fig, p_fail_single = pad_esd_yield_map_generator(
    #     pad_coords_um=pad_coords,
    #     pad_size_um=PAD_SIZE_UM,
    #     pad_pitch_um=PAD_PITCH_UM,
    #     top_die_w_um=TOP_DIE_W_UM,
    #     top_die_h_um=TOP_DIE_H_UM,
    #     n_tilts=N_TILTS,
    #     n_dishes=N_DISHES,
    #     tilt_x_mean_deg=TILT_X_MEAN_DEG,
    #     tilt_x_std_deg=TILT_X_STD_DEG,
    #     tilt_y_mean_deg=TILT_Y_MEAN_DEG,
    #     tilt_y_std_deg=TILT_Y_STD_DEG,
    #     top_dish_mean_nm=TOP_DISH_MEAN_NM,
    #     top_dish_std_nm=TOP_DISH_STD_NM,
    #     bot_dish_mean_nm=BOT_DISH_MEAN_NM,
    #     bot_dish_std_nm=BOT_DISH_STD_NM,
    #     n_steps_refine=N_STEPS_REFINE,
    #     base_seed=BASE_SEED,
    #     z_top_um=100.0,
    # )
    # print(f"p_fail_single (this die size) = {p_fail_single:.6f}")
    # print(f"risk map: nonzero pads = {(risk_vec>0).sum()} / {risk_vec.size}")
    # plt.show()

    print("\n=== DEMO2: single-run with external dishing arrays ===")
    # 构造一组“外部”dishing（你在真实调用时直接传入自己的数组即可，单位 nm）
    rng = np.random.default_rng(BASE_SEED ^ 0x13579BDF)
    Npad = pad_coords.shape[0]
    top_ext_nm = rng.normal(TOP_DISH_MEAN_NM, TOP_DISH_STD_NM, size=Npad).astype(np.float32)
    bot_ext_nm = rng.normal(BOT_DISH_MEAN_NM, BOT_DISH_STD_NM, size=Npad).astype(np.float32)


    # 外部 dishing（单位 nm），长度要等于 Npad=3
    # top_ext_nm = np.array([-5.2, -3.8, 1.0], dtype=np.float32)   # top raw
    # bot_ext_nm = np.array([-4.0, -4.0, -4.0], dtype=np.float32)  # bottom raw


    pad_idx_list, p_fail_single_2 = esd_failure_simulator(
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
    print(f"first-touch pad index list: {pad_idx_list}")
    print(f"p_fail_single (this die size) = {p_fail_single_2:.6f}")
