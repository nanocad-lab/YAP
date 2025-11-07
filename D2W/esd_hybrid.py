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
TOP_WAFER_RADIUS_UM: float = 75_000.0  # 300mm wafer 半径

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

# 采样设置
N_TILTS         = 5    # demo1: 倾角样本数
N_DISHES        = 5    # demo1: dishing 样本数
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
    return 0.0045 * (float(area_mm2) ** 0.35) * math.sqrt(float(v_chg))

def _weibull_cdf(I: float, k: float, lam: float) -> float:
    I = max(I, 1e-12)
    return max(0.0, min(1.0, 1.0 - math.exp(- (I/lam)**k)))

def _fail_prob_single(I: float, k: float, lam: float, cutoff: float) -> float:
    if I < cutoff:
        return 0.0
    return _weibull_cdf(I, k, lam)

def _compute_p_fail_for_die(top_die_w_um: float, top_die_h_um: float) -> float:
    area_mm2 = (float(top_die_w_um) * 1e-3) * (float(top_die_h_um) * 1e-3)
    I_peak   = _ipeak_from_die_voltage(area_mm2, float(V_CHARGING_V))
    return _fail_prob_single(I_peak, float(WEIBULL_K), float(WEIBULL_LAMBDA), float(CUTOFF_MIN_A))

def _four_corners(cx: np.ndarray, cy: np.ndarray, half: float):
    """返回四角：LL, LR, UR, UL  → 形状 (N,4)"""
    x4 = np.stack([cx - half, cx + half, cx + half, cx - half], axis=1)
    y4 = np.stack([cy - half, cy - half, cy + half, cy + half], axis=1)
    return x4, y4

# =========================
# 一次性收集 die 四角 + 入围 pad 四角 → 一起旋转并找最小
# =========================
def _collect_corners_and_baselines(
    *,
    pad_coords_um: np.ndarray,          # (Npad,2)
    pad_size_um: float,                 # pad 正方形边长
    top_die_w_um: float,
    top_die_h_um: float,
    top_dish_um_raw: np.ndarray,        # (Npad,)  原始 top dishing (µm)
    bot_dish_um: np.ndarray,            # (Npad,)  bottom dishing (µm)
    z_top_um: float
):
    """
    返回：
      x_all, y_all: (M,)  所有候选角坐标
      top_eff_all: (M,)  对应的 top 有效高度项（die 角为 0，pad 角取 -top_raw）
      bot_loc_all: (M,)  对应的底部局部高度（die 角为 0，pad 角取 bot_dish）
      is_pad_all:  (M,)  是否来自 pad 角
      pad_id_all:  (M,)  pad 索引（die 角记为 -1）
    其中 M = 4（die四角） + 4 * N_sel（入围 pad 数 * 4）
    """
    # die 四角
    hw, hh = 0.5*float(top_die_w_um), 0.5*float(top_die_h_um)
    die_x = np.array([-hw,  hw,  hw, -hw], dtype=np.float64)
    die_y = np.array([-hh, -hh,  hh,  hh], dtype=np.float64)
    M_die = die_x.shape[0]

    # 入围 pad
    mask = (top_dish_um_raw + bot_dish_um) >= 0.0
    sel  = np.where(mask)[0]
    N_sel = sel.size

    if N_sel > 0:
        cx = pad_coords_um[sel, 0].astype(np.float64)
        cy = pad_coords_um[sel, 1].astype(np.float64)
        half = 0.5*float(pad_size_um)
        x4, y4 = _four_corners(cx, cy, half)    # (N_sel,4)
        x_pad = x4.reshape(-1)
        y_pad = y4.reshape(-1)

        # top 有效高度（取反号）；bottom 局部高度
        top_eff_pad = (-top_dish_um_raw[sel])[:, None].repeat(4, axis=1).reshape(-1).astype(np.float64)
        bot_loc_pad = ( bot_dish_um[sel])[:, None].repeat(4, axis=1).reshape(-1).astype(np.float64)

        is_pad_pad = np.ones_like(x_pad, dtype=bool)
        pad_id_pad = np.repeat(sel.astype(np.int64), 4)
    else:
        x_pad = y_pad = top_eff_pad = bot_loc_pad = np.zeros((0,), dtype=np.float64)
        is_pad_pad = np.zeros((0,), dtype=bool)
        pad_id_pad = np.zeros((0,), dtype=np.int64)

    # 汇总
    x_all = np.concatenate([die_x, x_pad])
    y_all = np.concatenate([die_y, y_pad])

    top_eff_die = np.zeros((M_die,), dtype=np.float64)
    bot_loc_die = np.zeros((M_die,), dtype=np.float64)

    top_eff_all = np.concatenate([top_eff_die, top_eff_pad])
    bot_loc_all = np.concatenate([bot_loc_die, bot_loc_pad])

    is_pad_die = np.zeros((M_die,), dtype=bool)
    is_pad_all = np.concatenate([is_pad_die, is_pad_pad])

    pad_id_die = -np.ones((M_die,), dtype=np.int64)
    pad_id_all = np.concatenate([pad_id_die, pad_id_pad])

    return x_all, y_all, top_eff_all, bot_loc_all, is_pad_all, pad_id_all

def _rotate_and_min_choice(
    *,
    x_all: np.ndarray,
    y_all: np.ndarray,
    top_eff_all: np.ndarray,
    bot_loc_all: np.ndarray,
    tilt_x_deg: float,
    tilt_y_deg: float,
    z_top_um: float,
    is_pad_all: np.ndarray,
    pad_id_all: np.ndarray,
    rng_pick: np.random.Generator,
    atol: float = 1e-12
) -> Tuple[Optional[int], bool, float]:
    """
    旋转 + 统一求最小间隙。
    返回：
      (chosen_pad_index_or_None, is_die_only_min, min_gap_value)
    选择规则：
      - 找到全局最小 gap；
      - 若最小集合里包含任意 pad 角（无论是否混有 die 角），从该集合里的 **pad_id** 去重后随机挑一个返回；
      - 若最小集合只有 die 角 → 返回 (None, True, min).
    """
    A, B, C = _z_linear_coeffs(tilt_x_deg, tilt_y_deg)
    z_top = float(z_top_um) + A*x_all + B*y_all + C*top_eff_all
    gaps  = z_top - bot_loc_all

    min_val = float(np.min(gaps))
    is_min  = np.isclose(gaps, min_val, rtol=0.0, atol=atol)
    # 在最小集合里，挑 pad 的候选
    cand_mask = is_min & is_pad_all
    if np.any(cand_mask):
        # 候选 pad 的唯一 id 集合
        cand_pad_ids = np.unique(pad_id_all[cand_mask])
        idx = rng_pick.integers(0, cand_pad_ids.size)
        return int(cand_pad_ids[idx]), False, min_val
    else:
        # 全是 die 角
        return None, True, min_val

def _binary_halving_until_pad(
    *,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    top_die_w_um: float,
    top_die_h_um: float,
    z_top_um: float,
    tilt_x_init_deg: float,
    tilt_y_init_deg: float,
    top_dish_um_raw: np.ndarray,   # (Npad,) µm
    bot_dish_um: np.ndarray,       # (Npad,) µm
    rng_pick: np.random.Generator,
    atol_gap: float = 1e-12,
    atol_tilt_deg: float = 1e-12,
    max_iter_guard: int = 10000,
) -> Tuple[Optional[int], float, float, float]:
    """
    把 die 四角 + 入围 pad 四角放在一起，一次性旋转并找最小。
    若最小集合包含 pad，则在其中等概率随机选一个 pad_id 返回；
    否则（二者都是 die 角）对 (tilt_x, tilt_y) 进行二分（每次 /2），
    反复计算，直到出现 pad 为止。

    返回：(pad_index or None, final_tilt_x, final_tilt_y, final_min_gap_um)
    - 若倾角已到极小（或超过保护迭代次数）仍全是 die 角，返回 None。
    """
    # 先收集所有角
    x_all, y_all, top_eff_all, bot_loc_all, is_pad_all, pad_id_all = _collect_corners_and_baselines(
        pad_coords_um=pad_coords_um, pad_size_um=pad_size_um,
        top_die_w_um=top_die_w_um, top_die_h_um=top_die_h_um,
        top_dish_um_raw=top_dish_um_raw, bot_dish_um=bot_dish_um,
        z_top_um=z_top_um
    )

    if not np.any(is_pad_all):
        A, B, C = _z_linear_coeffs(tilt_x_init_deg, tilt_y_init_deg)
        z_top = float(z_top_um) + A*x_all + B*y_all + C*top_eff_all
        min_gap = float(np.min(z_top - bot_loc_all))
        return None, float(tilt_x_init_deg), float(tilt_y_init_deg), min_gap

    tx, ty = float(tilt_x_init_deg), float(tilt_y_init_deg)

    # 第一次判定
    pad_choice, die_only, min_gap = _rotate_and_min_choice(
        x_all=x_all, y_all=y_all, top_eff_all=top_eff_all, bot_loc_all=bot_loc_all,
        tilt_x_deg=tx, tilt_y_deg=ty, z_top_um=z_top_um,
        is_pad_all=is_pad_all, pad_id_all=pad_id_all,
        rng_pick=rng_pick, atol=atol_gap
    )
    if not die_only:
        return pad_choice, tx, ty, min_gap

    # 二分化：每次 /2，直到出现 pad
    it = 0
    while die_only:
        tx *= 0.5
        ty *= 0.5
        pad_choice, die_only, min_gap = _rotate_and_min_choice(
            x_all=x_all, y_all=y_all, top_eff_all=top_eff_all, bot_loc_all=bot_loc_all,
            tilt_x_deg=tx, tilt_y_deg=ty, z_top_um=z_top_um,
            is_pad_all=is_pad_all, pad_id_all=pad_id_all,
            rng_pick=rng_pick, atol=atol_gap
        )
        it += 1
        # 安全保护
        if (abs(tx) <= atol_tilt_deg and abs(ty) <= atol_tilt_deg) or (it >= max_iter_guard):
            return None, tx, ty, min_gap

    return pad_choice, tx, ty, min_gap

# =========================
# demo1: risk pad map 生成器（统一列表 + 二分直到 pad）
# =========================
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
    base_seed: int = 20251006,
    z_top_um: float = 100.0,
) -> Tuple[np.ndarray, Optional[plt.Figure], float]:
    Npad = pad_coords_um.shape[0]
    counts_vec = np.zeros((Npad,), dtype=np.int64)

    rng_tilt = np.random.default_rng(base_seed ^ 0xC001FEED)
    total_runs = int(n_tilts) * int(n_dishes)

    progress_counter = 0
    for t in range(n_tilts):
        tx0 = float(rng_tilt.normal(tilt_x_mean_deg, tilt_x_std_deg))
        ty0 = float(rng_tilt.normal(tilt_y_mean_deg, tilt_y_std_deg))

        for d in range(n_dishes):
            progress_counter += 1
            if (progress_counter % 1000) == 0 or (progress_counter == total_runs):
                print(f"[ESD Sim] Progress: {progress_counter} / {total_runs} runs completed.")
            seed = base_seed + (t * n_dishes + d)
            rng_top  = np.random.default_rng(seed ^ 0x9E3779B1)
            rng_bot  = np.random.default_rng(seed ^ 0x85EBCA77)
            rng_pick = np.random.default_rng(seed ^ 0xDEADBEEF)

            top_dish_um_raw = rng_top.normal(
                loc=float(top_dish_mean_nm)*NM_TO_UM,
                scale=max(float(top_dish_std_nm),0.0)*NM_TO_UM,
                size=(Npad,)
            ).astype(np.float64)
            bot_dish_um     = rng_bot.normal(
                loc=float(bot_dish_mean_nm)*NM_TO_UM,
                scale=max(float(bot_dish_std_nm),0.0)*NM_TO_UM,
                size=(Npad,)
            ).astype(np.float64)

            pad_choice, _, _, _ = _binary_halving_until_pad(
                pad_coords_um=pad_coords_um,
                pad_size_um=pad_size_um,
                top_die_w_um=top_wafer_radius_um,
                top_die_h_um=top_wafer_radius_um,
                z_top_um=z_top_um,
                tilt_x_init_deg=tx0,
                tilt_y_init_deg=ty0,
                top_dish_um_raw=top_dish_um_raw,
                bot_dish_um=bot_dish_um,
                rng_pick=rng_pick
            )

            if pad_choice is not None:
                counts_vec[pad_choice] += 1

    prob_vec = counts_vec.astype(np.float64) / float(total_runs)
    print("Prob_vec min/max: {:.6f} / {:.6f}".format(float(prob_vec.min()), float(prob_vec.max())))

    p_fail_single = _compute_p_fail_for_die(top_wafer_radius_um, top_wafer_radius_um)
    valid_pad_risk_map_vec = prob_vec * float(p_fail_single)
    print("Risk map min/max: {:.6e} / {:.6e}".format(float(valid_pad_risk_map_vec.min()),
                                                     float(valid_pad_risk_map_vec.max())))

    fig = plot_probability_over_pads_with_pitch(
        pad_coords_um=pad_coords_um,
        prob_vec=valid_pad_risk_map_vec,
        pitch_um=pad_pitch_um,
        title="Risk Pad Map = P(first-touch) × p_fail (square side = PITCH)"
    )

    valid_pad_yield_map_vec = 1.0 - valid_pad_risk_map_vec
    return valid_pad_yield_map_vec, fig, float(p_fail_single)

# =========================
# demo2: 单次外部数组模拟（统一列表 + 二分直到 pad）
# =========================
def esd_failure_simulator(
    *,
    pad_coords_um: np.ndarray,
    pad_size_um: float,
    top_wafer_radius_um: float,
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
      - pad_idx 或 None
      - survive_bool：用 die 级 p_fail_single 做一次伯努利
    """
    assert pad_coords_um.shape[0] == top_dish_nm_ext.shape[0] == bot_dish_nm_ext.shape[0], \
        "The length of pad_coords_um, top_dish_nm_ext and bot_dish_nm_ext must be equal."

    rng = np.random.default_rng(base_seed ^ 0xA5A5A5A5)
    rng_pick = np.random.default_rng((base_seed ^ 0xA5A5A5A5) ^ 0xDEADBEEF)

    tilt_x = float(rng.normal(tilt_x_mean_deg, tilt_x_std_deg))
    tilt_y = float(rng.normal(tilt_y_mean_deg, tilt_y_std_deg))

    top_dish_um_raw = (top_dish_nm_ext.astype(np.float64)) * NM_TO_UM
    bot_dish_um     = (bot_dish_nm_ext.astype(np.float64)) * NM_TO_UM

    pad_choice, _, _, _ = _binary_halving_until_pad(
        pad_coords_um=pad_coords_um,
        pad_size_um=pad_size_um,
        top_die_w_um=top_wafer_radius_um,
        top_die_h_um=top_wafer_radius_um,
        z_top_um=z_top_um,
        tilt_x_init_deg=tilt_x,
        tilt_y_init_deg=tilt_y,
        top_dish_um_raw=top_dish_um_raw,
        bot_dish_um=bot_dish_um,
        rng_pick=rng_pick
    )

    p_fail_single = _compute_p_fail_for_die(top_wafer_radius_um, top_wafer_radius_um)
    random_float = np.random.uniform(0.0, 1.0)
    if (pad_choice is not None) and (random_float < p_fail_single):
        survive_bool = False
    else:
        survive_bool = True
    return pad_choice, survive_bool

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
    halfW, halfH = TOP_WAFER_RADIUS_UM, TOP_WAFER_RADIUS_UM
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
    cbar.set_label("Risk = P_first-touch × p_fail_single")
    return fig

# =========================
# Main（演示）
# =========================
if __name__ == "__main__":
    # 若未提供坐标，生成一个 pitch 网格做 demo
    if not PAD_COORDS_UM:
        halfW, halfH = TOP_WAFER_RADIUS_UM, TOP_WAFER_RADIUS_UM
        xs = np.arange(-halfW + PAD_PITCH_UM*0.5, halfW, PAD_PITCH_UM)
        ys = np.arange( halfH - PAD_PITCH_UM*0.5, -halfH, -PAD_PITCH_UM)
        X, Y = np.meshgrid(xs, ys)
        PAD_COORDS_UM = list(zip(X.ravel().tolist(), Y.ravel().tolist()))
    pad_coords = np.asarray(PAD_COORDS_UM, dtype=np.float64).reshape(-1, 2)

    # # ===== DEMO1：风险/良率图 =====
    # print("=== DEMO1: risk pad map (with unified-corner + binary-halving search) ===")
    # yield_vec, fig, p_fail_single = pad_esd_yield_map_generator(
    #     pad_coords_um=pad_coords,
    #     pad_size_um=PAD_SIZE_UM,
    #     pad_pitch_um=PAD_PITCH_UM,
    #     top_wafer_radius_um=TOP_WAFER_RADIUS_UM,
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
    #     base_seed=BASE_SEED,
    #     z_top_um=100.0,
    # )
    # print(f"p_fail_single (this die size) = {p_fail_single:.6f}")
    # print(f"yield map: min/max = {float(yield_vec.min()):.6f} / {float(yield_vec.max()):.6f}")

    # # 展示图
    # plt.show()

    # ===== DEMO2：单次外部数组判定 =====
    print("\n=== DEMO2: single-run with external dishing arrays (unified-corner + binary-halving) ===")
    rng = np.random.default_rng(BASE_SEED ^ 0x13579BDF)
    Npad = pad_coords.shape[0]
    top_ext_nm = rng.normal(TOP_DISH_MEAN_NM, TOP_DISH_STD_NM, size=Npad).astype(np.float64)
    bot_ext_nm = rng.normal(BOT_DISH_MEAN_NM, BOT_DISH_STD_NM, size=Npad).astype(np.float64)

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
        base_seed=BASE_SEED,
        z_top_um=100.0,
    )
    print(f"first-touch pad index: {pad_idx}")
    print(f"survive? {survive}")
