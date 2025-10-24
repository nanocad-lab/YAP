from __future__ import annotations
from IPython.display import display
from typing import Optional, Dict, Sequence, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


# ==== Weibull & simulate 配置（新增） ====
V_CHARGING_V     = 3.0       # 充电电压 [V]
WEIBULL_K        = 3.981285  # 拟合得到或手工回填
WEIBULL_LAMBDA   = 2.176527  # ditto
CUTOFF_MIN_A     = 0.0       # 电流硬截断，小于该值 → 失效概率=0




# ---- Jupyter widget backend helper ----
def _enable_widget_backend():
    """Switch to %matplotlib widget inside IPython if ipympl is available."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return
        import importlib
        importlib.import_module("ipympl")  # ensure installed
        ip.run_line_magic("matplotlib", "widget")
    except Exception:
        pass

# =========================
# Units
# =========================
NM_TO_UM = 1e-3  # 1 nm = 1e-3 µm

# =========================
# Low-level helpers
# =========================
def _tile_seed(base_seed: int, i0: int, j0: int) -> int:
    """
    Stable 64-bit seed from (base_seed, i0, j0) using hashlib (no numpy bit ops).
    Cross-platform reproducible; returns in [1, 2**63-1].
    """
    import hashlib
    s = f"{int(base_seed)}:{int(i0)}:{int(j0)}".encode("utf-8")
    h = hashlib.blake2b(s, digest_size=8).digest()  # 64-bit
    val = int.from_bytes(h, byteorder="little", signed=False) % (2**63 - 1)
    return 1 if val == 0 else val

def _z_linear_coeffs(ax_deg: float, ay_deg: float) -> Tuple[float, float, float]:
    """
    For R = Ry(ay) @ Rx(ax), with dish entering as scalar height at the pad,
    z' = z0 + A*x + B*y + C*dish, where:
      A = -sin(ay)
      B = cos(ay)*sin(ax)
      C = cos(ay)*cos(ax)
    """
    ax = np.deg2rad(float(ax_deg)); ay = np.deg2rad(float(ay_deg))
    ca, sa = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    A = -sy
    B = cy * sa
    C = cy * ca
    return float(A), float(B), float(C)

def _eligible_pair_possible(
    TOP_DISH_MEAN_nm: float, TOP_DISH_STD_nm: float,
    BOT_DISH_MEAN_nm: float, BOT_DISH_STD_nm: float
) -> bool:
    """
    判断是否“存在概率”使 top_raw + bot >= 0。
    对独立正态分布，Sum ~ N(mu_t+mu_b, sqrt(sd_t^2+sd_b^2)).
    只有在 sd_t==0 且 sd_b==0 且 (mu_t+mu_b)<0 时，概率严格为 0。
    其它情况（任一方 std>0）概率总是 >0。
    """
    mu_t_um = float(TOP_DISH_MEAN_nm) * NM_TO_UM
    mu_b_um = float(BOT_DISH_MEAN_nm) * NM_TO_UM
    sd_t_um = max(float(TOP_DISH_STD_nm), 0.0) * NM_TO_UM
    sd_b_um = max(float(BOT_DISH_STD_nm), 0.0) * NM_TO_UM
    if sd_t_um == 0.0 and sd_b_um == 0.0:
        return (mu_t_um + mu_b_um) >= 0.0
    return True



import math

def _ipeak_from_die_voltage(area_mm2: float, v_chg: float) -> float:
    """I_PEAK = 0.015 * sqrt(area_mm2 * v_chg)"""
    return 0.0045 * (float(area_mm2) ** 0.35) * math.sqrt(float(v_chg))


def _weibull_cdf(I: float, k: float, lam: float) -> float:
    I = max(I, 1e-12)
    return max(0.0, min(1.0, 1.0 - math.exp(- (I/lam)**k)))

def _fail_prob_single(I: float, k: float, lam: float, cutoff: float) -> float:
    if I < cutoff:
        return 0.0
    return _weibull_cdf(I, k, lam)



# =========================
# Streaming candidate tiles (full die, no cropping)
# =========================
def iter_convex_candidate_tiles(
    *,
    DIE_W_um: float,
    PITCH_um: float,
    TOP_DISH_MEAN_nm: float, TOP_DISH_STD_nm: float,
    BOT_DISH_MEAN_nm: float, BOT_DISH_STD_nm: float,
    seed_top: int, seed_bot: int,
    tile_nx: int = 2048, tile_ny: int = 2048,
):
    """
    Full-die streaming. Screening rule:
        (top_dish_raw + bot_dish) >= 0   # raw values; DO NOT negate top here
    top die is negated only when computing gap: top_dish_eff = - top_dish_raw
    """
    Nx = int(np.floor(DIE_W_um / PITCH_um))
    Ny = Nx
    half = 0.5 * DIE_W_um

    cx_all = (-half + (np.arange(Nx, dtype=np.float32) + 0.5) * PITCH_um).astype(np.float32)
    cy_all = ( half - (np.arange(Ny, dtype=np.float32) + 0.5) * PITCH_um).astype(np.float32)

    mu_t = float(TOP_DISH_MEAN_nm) * NM_TO_UM
    sd_t = max(float(TOP_DISH_STD_nm), 0.0) * NM_TO_UM
    mu_b = float(BOT_DISH_MEAN_nm) * NM_TO_UM
    sd_b = max(float(BOT_DISH_STD_nm), 0.0) * NM_TO_UM

    for j0 in range(0, Ny, tile_ny):
        j1 = min(j0 + tile_ny, Ny)
        sub_ny = j1 - j0

        for i0 in range(0, Nx, tile_nx):
            i1 = min(i0 + tile_nx, Nx)
            sub_nx = i1 - i0

            rng_top = np.random.default_rng(_tile_seed(seed_top, i0, j0))
            rng_bot = np.random.default_rng(_tile_seed(seed_bot, i0, j0))

            top_dish_raw = rng_top.normal(loc=mu_t, scale=sd_t, size=(sub_ny, sub_nx)).astype(np.float32)
            bot_dish     = rng_bot.normal(loc=mu_b, scale=sd_b, size=(sub_ny, sub_nx)).astype(np.float32)

            # Screening
            sum_raw = top_dish_raw + bot_dish
            mask = (sum_raw >= 0.0)
            if not np.any(mask):
                continue

            jj, ii = np.nonzero(mask)
            i_idx = (i0 + ii).astype(np.int32)
            j_idx = (j0 + jj).astype(np.int32)
            pad_ids = (j_idx * Nx + i_idx).astype(np.int64)

            top_dish_eff = -top_dish_raw  # used later in gap computation

            yield {
                "i_idx": i_idx,
                "j_idx": j_idx,
                "pad_ids": pad_ids,
                "cx": cx_all[i_idx],
                "cy": cy_all[j_idx],
                "top_dish_raw": top_dish_raw[jj, ii],
                "top_dish_eff": top_dish_eff[jj, ii],
                "bot_dish": bot_dish[jj, ii],
                "Nx": Nx, "Ny": Ny,
            }

# =========================
# Min-gap search (streaming)
# =========================
def find_min_gap_points_streaming(
    *,
    DIE_W_um: float,
    PITCH_um: float,
    z_top_um: float,
    tilt_x_deg: float,
    tilt_y_deg: float,
    half_top_um: float,
    TOP_DISH_MEAN_nm: float, TOP_DISH_STD_nm: float,
    BOT_DISH_MEAN_nm: float, BOT_DISH_STD_nm: float,
    seed_top: int, seed_bot: int,
    tile_nx: int = 2048, tile_ny: int = 2048,
    verbose: bool = False, run_id: Optional[int] = None,
):
    Nx = int(np.floor(DIE_W_um / PITCH_um))
    Ny = Nx
    z0 = float(z_top_um)
    A, B, C = _z_linear_coeffs(tilt_x_deg, tilt_y_deg)

    # Early exit: impossible to have eligible pad pair
    if not _eligible_pair_possible(
        TOP_DISH_MEAN_nm=TOP_DISH_MEAN_nm, TOP_DISH_STD_nm=TOP_DISH_STD_nm,
        BOT_DISH_MEAN_nm=BOT_DISH_MEAN_nm, BOT_DISH_STD_nm=BOT_DISH_STD_nm
    ):
        if verbose:
            prefix = f"[Run {run_id:02d}] " if run_id is not None else ""
            print(f"{prefix}[Early Exit] No probability for (top_raw + bot) >= 0; skip.", flush=True)
        return {
            "angles_deg": (float(tilt_x_deg), float(tilt_y_deg)),
            "min_value_um": np.inf,
            "points": [],
            "counts": {"die_corner": 0, "pad_corner": 0},
        }

    min_val = np.inf
    records: List[dict] = []

    # die corners (no dishing)
    die_names = ["die_LL", "die_LR", "die_UR", "die_UL"]
    hdie = 0.5 * float(DIE_W_um)
    die_x = np.array([-hdie,  hdie,  hdie, -hdie], dtype=np.float32)
    die_y = np.array([-hdie, -hdie,  hdie,  hdie], dtype=np.float32)
    z_top_corners = z0 + A * die_x + B * die_y
    die_gaps = z_top_corners
    vmin_die = float(np.min(die_gaps))
    min_val = vmin_die
    for k in np.where(die_gaps == vmin_die)[0].tolist():
        records.append({
            "kind": "die_corner",
            "corner_name": die_names[k],
            "corner_index": int(k),
            "x_um": float(die_x[k]),
            "y_um": float(die_y[k]),
            "z_top_um": float(z_top_corners[k]),
            "bottom_ref_um": 0.0,
            "gap_um": float(die_gaps[k]),
        })

    h = float(half_top_um)

    for blk in iter_convex_candidate_tiles(
        DIE_W_um=DIE_W_um, PITCH_um=PITCH_um,
        TOP_DISH_MEAN_nm=TOP_DISH_MEAN_nm, TOP_DISH_STD_nm=TOP_DISH_STD_nm,
        BOT_DISH_MEAN_nm=BOT_DISH_MEAN_nm, BOT_DISH_STD_nm=BOT_DISH_STD_nm,
        seed_top=seed_top, seed_bot=seed_bot,
        tile_nx=tile_nx, tile_ny=tile_ny,
    ):
        cx = blk["cx"]; cy = blk["cy"]
        x4 = np.stack([cx - h, cx + h, cx + h, cx - h], axis=1)
        y4 = np.stack([cy - h, cy - h, cy + h, cy + h], axis=1)

        # use top_dish_eff (flipped sign) in z_top
        top_eff = blk["top_dish_eff"][:, None]
        bot_loc = blk["bot_dish"][:, None]

        z_top = z0 + A * x4 + B * y4 + C * top_eff
        gaps  = z_top - bot_loc

        vmin_blk = float(np.min(gaps))
        if vmin_blk < min_val:
            min_val = vmin_blk
            records = []
        if vmin_blk <= min_val:
            hit_m, hit_k = np.where(gaps == vmin_blk)
            for m, k in zip(hit_m.tolist(), hit_k.tolist()):
                records.append({
                    "kind": "pad_corner",
                    "pad_id_top_left": int(blk["pad_ids"][m]),
                    "pad_ij": (int(blk["i_idx"][m]), int(blk["j_idx"][m])),
                    "center_xy_um": (float(cx[m]), float(cy[m])),
                    "corner_index": int(k),  # 0:LL,1:LR,2:UR,3:UL
                    "x_um": float(x4[m, k]),
                    "y_um": float(y4[m, k]),
                    "z_top_um": float(z_top[m, k]),
                    "top_dish_raw_um": float(blk["top_dish_raw"][m]),
                    "top_dish_eff_um": float(blk["top_dish_eff"][m]),
                    "bottom_dish_um": float(blk["bot_dish"][m]),
                    "gap_um": float(vmin_blk),
                })

    die_list = [r for r in records if r["kind"] == "die_corner" and r["gap_um"] == min_val]
    pad_list = [r for r in records if r["kind"] == "pad_corner" and r["gap_um"] == min_val]
    counts = {"die_corner": len(die_list), "pad_corner": len(pad_list)}

    if verbose:
        prefix = f"[Run {run_id:02d}] " if run_id is not None else ""
        print(f"{prefix}[Min Gap] value = {min_val} um")
        print(f"{prefix}[Min Gap] total points = {len(records)}")
        print(f"{prefix}[Min Gap] die-corner count = {counts['die_corner']}")
        print(f"{prefix}[Min Gap] pad-corner count = {counts['pad_corner']}")

    return {
        "angles_deg": (float(tilt_x_deg), float(tilt_y_deg)),
        "min_value_um": float(min_val),
        "points": records,
        "counts": counts,
    }

def refine_min_to_pad_indices_streaming(
    *,
    DIE_W_um: float,
    PITCH_um: float,
    z_top_um: float,
    half_top_um: float,
    tilt_x_deg: float, tilt_y_deg: float,
    TOP_DISH_MEAN_nm: float, TOP_DISH_STD_nm: float,
    BOT_DISH_MEAN_nm: float, BOT_DISH_STD_nm: float,
    seed_top: int, seed_bot: int,
    n_steps: int = 10,
    tile_nx: int = 2048, tile_ny: int = 2048,
    verbose: bool = False,
):
    def _pad_ids_from(points):
        if not points:
            return np.zeros((0,), dtype=int)
        gmin = min(r["gap_um"] for r in points)
        ids = [r["pad_id_top_left"] for r in points if (r["kind"] == "pad_corner" and r["gap_um"] == gmin)]
        return np.unique(np.asarray(ids, dtype=int)) if ids else np.zeros((0,), dtype=int)

    # Early exit guard
    if not _eligible_pair_possible(
        TOP_DISH_MEAN_nm=TOP_DISH_MEAN_nm, TOP_DISH_STD_nm=TOP_DISH_STD_nm,
        BOT_DISH_MEAN_nm=BOT_DISH_MEAN_nm, BOT_DISH_STD_nm=BOT_DISH_STD_nm
    ):
        if verbose:
            print("[refine] Early Exit: no probability for (top_raw + bot) >= 0; return empty.")
        return {
            "initial": {"min_value_um": np.inf},
            "mode": "none" if (abs(float(tilt_x_deg))==0 and abs(float(tilt_y_deg))==0) else
                    ("one_axis" if (abs(float(tilt_x_deg))>0) ^ (abs(float(tilt_y_deg))>0) else "two_axis"),
            "found": False,
            "found_at_step": None,
            "angles_deg_at_found": None,
            "pad_ids_min_equal": np.zeros((0,), dtype=int),
        }

    mode = ("two_axis" if (abs(float(tilt_x_deg)) > 0 and abs(float(tilt_y_deg)) > 0)
            else "one_axis" if (abs(float(tilt_x_deg)) > 0 or abs(float(tilt_y_deg)) > 0)
            else "none")

    initial = find_min_gap_points_streaming(
        DIE_W_um=DIE_W_um, PITCH_um=PITCH_um, z_top_um=z_top_um,
        tilt_x_deg=tilt_x_deg, tilt_y_deg=tilt_y_deg, half_top_um=half_top_um,
        TOP_DISH_MEAN_nm=TOP_DISH_MEAN_nm, TOP_DISH_STD_nm=TOP_DISH_STD_nm,
        BOT_DISH_MEAN_nm=BOT_DISH_MEAN_nm, BOT_DISH_STD_nm=BOT_DISH_STD_nm,
        seed_top=seed_top, seed_bot=seed_bot, tile_nx=tile_nx, tile_ny=tile_ny,
        verbose=False
    )
    pad_ids0 = _pad_ids_from(initial["points"])
    if pad_ids0.size > 0:
        return {
            "initial": {"min_value_um": initial["min_value_um"]},
            "mode": mode,
            "found": True,
            "found_at_step": 0,
            "angles_deg_at_found": (float(tilt_x_deg), float(tilt_y_deg)),
            "pad_ids_min_equal": pad_ids0,
        }

    if mode == "none":
        return {
            "initial": {"min_value_um": initial["min_value_um"]},
            "mode": mode,
            "found": False,
            "found_at_step": None,
            "angles_deg_at_found": None,
            "pad_ids_min_equal": np.zeros((0,), dtype=int),
        }

    ax0, ay0 = float(tilt_x_deg), float(tilt_y_deg)
    nonzero_x = (abs(ax0) > 0.0); nonzero_y = (abs(ay0) > 0.0)

    for step in range(1, int(n_steps) + 1):
        fac = (n_steps - step) / float(n_steps)
        ax = fac * ax0 if nonzero_x else 0.0
        ay = fac * ay0 if nonzero_y else 0.0

        cur = find_min_gap_points_streaming(
            DIE_W_um=DIE_W_um, PITCH_um=PITCH_um, z_top_um=z_top_um,
            tilt_x_deg=ax, tilt_y_deg=ay, half_top_um=half_top_um,
            TOP_DISH_MEAN_nm=TOP_DISH_MEAN_nm, TOP_DISH_STD_nm=TOP_DISH_STD_nm,
            BOT_DISH_MEAN_nm=BOT_DISH_MEAN_nm, BOT_DISH_STD_nm=BOT_DISH_STD_nm,
            seed_top=seed_top, seed_bot=seed_bot, tile_nx=tile_nx, tile_ny=tile_ny,
            verbose=False
        )
        pids = _pad_ids_from(cur["points"])
        if verbose:
            print(f"[refine/stream] step={step}/{n_steps}  ax={ax:.6f}  ay={ay:.6f}  pad_min_count={pids.size}")
        if pids.size > 0:
            return {
                "initial": {"min_value_um": initial["min_value_um"]},
                "mode": mode,
                "found": True,
                "found_at_step": step,
                "angles_deg_at_found": (ax, ay),
                "pad_ids_min_equal": pids,
            }

    return {
        "initial": {"min_value_um": initial["min_value_um"]},
        "mode": mode,
        "found": False,
        "found_at_step": None,
        "angles_deg_at_found": None,
        "pad_ids_min_equal": np.zeros((0,), dtype=int),
    }

# =========================
# ESD prior + yield helpers
# =========================


# =========================
# Sparse counting utilities
# =========================
def increment_counts_by_ids_sparse(counter: Dict[int, int], pad_ids: np.ndarray) -> None:
    if pad_ids is None or pad_ids.size == 0:
        return
    for pid in pad_ids.tolist():
        counter[int(pid)] = counter.get(int(pid), 0) + 1

def sparse_to_dense_counts(counter: Dict[int, int], Nx: int, Ny: int) -> np.ndarray:
    arr = np.zeros((Ny, Nx), dtype=np.int32)
    if not counter:
        return arr
    ids = np.fromiter(counter.keys(), dtype=np.int64)
    vals = np.fromiter(counter.values(), dtype=np.int64)
    i = (ids % Nx).astype(np.int32)
    j = (ids // Nx).astype(np.int32)
    np.add.at(arr, (j, i), vals.astype(np.int32))
    return arr

def make_probability_map(counts_ji: np.ndarray, total_runs: int) -> np.ndarray:
    if total_runs <= 0:
        raise ValueError("total_runs must be > 0")
    return counts_ji.astype(np.float32) / float(total_runs)

# =========================
# Plotting (no flip; origin='upper')
# =========================
def plot_probability_map(prob_ji: np.ndarray,
                         *,
                         title: str = "Pad Selection Probability (white=0)") -> plt.Figure:
    data = np.ma.masked_where(prob_ji <= 0.0, prob_ji)
    vmax = float(np.max(prob_ji)) if np.any(prob_ji > 0) else 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    fig = plt.figure()
    try:
        fig.canvas.toolbar_visible = True
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
    except Exception:
        pass
    im = plt.imshow(data, cmap=cmap, vmin=0.0, vmax=vmax,
                    interpolation="nearest", origin="upper")
    cbar = plt.colorbar(im); cbar.set_label("Selection Probability")
    plt.title(title); plt.xlabel("i (column)"); plt.ylabel("j (row, top→bottom)")
    return fig



# ================================
# Unified experiment: N tilts × N dishing = N^2
# ================================
def _one_experiment_given_tilt(
    *,
    DIE_W: float, PITCH: float,
    PAD_BOT_R_ratio: float, PAD_TOP_R_ratio: float,
    TOP_DISH_MEAN: float, TOP_DISH_STD: float,
    BOT_DISH_MEAN: float, BOT_DISH_STD: float,
    tilt_x_deg: float, tilt_y_deg: float,
    N_STEPS: int,
    TILE_NX: int, TILE_NY: int,
    seed_top: int, seed_bot: int,
):
    SIDE_TOP = PAD_TOP_R_ratio * PAD_BOT_R_ratio * PITCH
    half_top_um = 0.5 * SIDE_TOP
    out = refine_min_to_pad_indices_streaming(
        DIE_W_um=DIE_W, PITCH_um=PITCH, z_top_um=100.0,
        half_top_um=half_top_um,
        tilt_x_deg=tilt_x_deg, tilt_y_deg=tilt_y_deg,
        TOP_DISH_MEAN_nm=TOP_DISH_MEAN, TOP_DISH_STD_nm=TOP_DISH_STD,
        BOT_DISH_MEAN_nm=BOT_DISH_MEAN, BOT_DISH_STD_nm=BOT_DISH_STD,
        seed_top=seed_top, seed_bot=seed_bot,
        n_steps=N_STEPS, tile_nx=TILE_NX, tile_ny=TILE_NY,
        verbose=False
    )
    return out

# ================================
# Main
# ================================
if __name__ == "__main__":
    # ---- Config ----
    PITCH = 10.0
    DIE_W = 10_000.0
    PAD_BOT_R_ratio = 0.5
    PAD_TOP_R_ratio = 0.667

    # dishing（原始约定：凹<0、凸>0；top 只在 gap 计算时取负）
    TOP_DISH_MEAN = -4   # nm
    TOP_DISH_STD  = 2.5
    BOT_DISH_MEAN = -4
    BOT_DISH_STD  = 2.5

    # 倾角分布（度）
    TILT_X_MEAN_DEG = 0.000
    TILT_X_STD_DEG  = 0.01
    TILT_Y_MEAN_DEG = 0.000
    TILT_Y_STD_DEG  = 0.01

    # 采样次数 N（倾角 N 次 × dishing N 次 → 总 N^2）
    N = 5

    N_STEPS = 10
    base_seed = 20251006
    TILE_NX, TILE_NY = 4096, 4096

    # ← 已删除：NPZ_BASEMAP_PATH、ESD_TIER、EQUIPOTENTIALIZED


    print("========================================================", flush=True)
    print(f"[CONFIG] PITCH={PITCH} µm  DIE_W={DIE_W} µm", flush=True)
    print(f"[CONFIG] DISH top={TOP_DISH_MEAN}±{TOP_DISH_STD} nm  bot={BOT_DISH_MEAN}±{BOT_DISH_STD} nm", flush=True)
    print(f"[CONFIG] TILT x~N({TILT_X_MEAN_DEG},{TILT_X_STD_DEG}), y~N({TILT_Y_MEAN_DEG},{TILT_Y_STD_DEG}) [deg]", flush=True)
    print(f"[CONFIG] unified mode: N tilt samples × N dishing samples = {N*N} experiments", flush=True)

    Nx = Ny = int(np.floor(DIE_W / PITCH))
    counts_dict_final: Dict[int, int] = {}

    rng_tilt = np.random.default_rng(base_seed^2)

    for t in range(N):
        tilt_x = float(rng_tilt.normal(TILT_X_MEAN_DEG, TILT_X_STD_DEG))
        tilt_y = float(rng_tilt.normal(TILT_Y_MEAN_DEG, TILT_Y_STD_DEG))

        for d in range(N):
            seed = base_seed + (t * N + d)
            seed_top = seed ^ 0x9E3779B1
            seed_bot = seed ^ 0x85EBCA77

            out = _one_experiment_given_tilt(
                DIE_W=DIE_W, PITCH=PITCH,
                PAD_BOT_R_ratio=PAD_BOT_R_ratio, PAD_TOP_R_ratio=PAD_TOP_R_ratio,
                TOP_DISH_MEAN=TOP_DISH_MEAN, TOP_DISH_STD=TOP_DISH_STD,
                BOT_DISH_MEAN=BOT_DISH_MEAN, BOT_DISH_STD=BOT_DISH_STD,
                tilt_x_deg=tilt_x, tilt_y_deg=tilt_y,
                N_STEPS=N_STEPS,
                TILE_NX=TILE_NX, TILE_NY=TILE_NY,
                seed_top=seed_top, seed_bot=seed_bot,
            )

            pad_ids = out.get("pad_ids_min_equal", np.zeros((0,), dtype=int))
            if out.get("found", False) and isinstance(pad_ids, np.ndarray) and pad_ids.size > 0:
                increment_counts_by_ids_sparse(counts_dict_final, pad_ids)

    total_experiments = N * N
    counts_ji = sparse_to_dense_counts(counts_dict_final, Nx, Ny)
    prob_ji   = make_probability_map(counts_ji, total_experiments)

    print("\n========================================================", flush=True)
    print(f"[SUMMARY] Total experiments accumulated = {total_experiments}", flush=True)

    # 仅保留：可视化 1
    print("[SUMMARY] Showing probability heatmap (no flip, origin='upper')...", flush=True)
    fig1 = plot_probability_map(prob_ji, title="Pad Selection Probability (white=0)")
    import matplotlib.pyplot as plt
    plt.show()



     # --- 新增：simulate（电压-面积 → 峰值电流 → 失效概率 → 良率） ---
    # 面积从 DIE_W(µm) 换算到 mm²
    area_mm2 = (float(DIE_W) * 1e-3)**2
    I_peak   = _ipeak_from_die_voltage(area_mm2, float(V_CHARGING_V))
    k        = float(WEIBULL_K)
    lam      = float(WEIBULL_LAMBDA)
    cutoff   = float(CUTOFF_MIN_A)

    p_fail_single = _fail_prob_single(I_peak, k, lam, cutoff)

    # 注意：prob_ji.sum() 等于“首要接触发生在 pad（而不是 die 角）”的总概率
    risk_sum = float(prob_ji.sum()) * p_fail_single
    yield_est = max(0.0, 1.0 - min(risk_sum, 1.0))

    print("\n===================== SIMULATE (yield estimate) =====================", flush=True)
    print(f"Die area       : {area_mm2:.6f} mm^2   (from DIE_W={DIE_W:.3f} µm)", flush=True)
    print(f"Voltage        : {V_CHARGING_V:.6f} V", flush=True)
    print(f"I_PEAK         : {I_peak:.6f} A   (0.015 * sqrt(A*V))", flush=True)
    print(f"Weibull (k,λ)  : ({k:.6f}, {lam:.6f})   cutoff={cutoff:.6f} A", flush=True)
    print(f"p_fail(single) : {p_fail_single:.6f}", flush=True)
    print(f"Σ pad prob     : {float(prob_ji.sum()):.6f}   (probability min occurs at pads)", flush=True)
    print(f"Risk sum       : {risk_sum:.6f}   (= p_fail * Σ pad prob)", flush=True)
    print(f"Yield estimate : {yield_est:.6f}", flush=True)
    print("=====================================================================", flush=True)