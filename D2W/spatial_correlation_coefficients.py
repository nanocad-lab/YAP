#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Spatial correlation coefficient utilities for D2W hybrid bonding
#### Author: Zhichao Chen
#### Updated: Apr 6, 2026

import os
from typing import Dict, List

import numpy as np
from sklearn.neighbors import KDTree

from Cu_gap_simulator import Cu_gap_simulator
from debond import debond_dishing_intervals_from_coords
from esd_yield_simulator import esd_failure_simulator
from overlay_yield_simulator import die_pad_misalignment
from utils.util import atomic_save_npy, get_dishing_bound_cache_path


FAILURE_MECHANISMS = ("overlay", "particle", "mechanical", "esd", "overall")


def _get_valid_pad_mask(pad_bitmap_collection: dict) -> np.ndarray:
    return (
        (pad_bitmap_collection["CRITICAL_PAD_BITMAP"] == 1)
        | (pad_bitmap_collection["REDUNDANT_PAD_BITMAP"] == 1)
        | (pad_bitmap_collection["DUMMY_PAD_BITMAP"] == 1)
    )


def _get_min_pair_distance_um(cfg) -> float:
    if cfg.PAD_ARRANGE_PATTERN == "checkerboard":
        return min(
            np.sqrt(cfg.PITCH_c_um ** 2 + cfg.PITCH_r_um ** 2),
            2 * cfg.PITCH_r_um,
            2 * cfg.PITCH_c_um,
        )
    return min(cfg.PITCH_c_um, cfg.PITCH_r_um)


def _get_valid_pad_dishing_bound_array(cfg, valid_die_pad_coords: np.ndarray, input_args: dict | None) -> np.ndarray:
    save_path = get_dishing_bound_cache_path(cfg, input_args)
    if not os.path.exists(save_path) or cfg.DEBUG:
        valid_pad_dishing_bound_array = debond_dishing_intervals_from_coords(
            cfg,
            valid_die_pad_coords,
        )
        atomic_save_npy(save_path, valid_pad_dishing_bound_array)
    else:
        valid_pad_dishing_bound_array = np.load(save_path)
        if valid_pad_dishing_bound_array.shape[0] != valid_die_pad_coords.shape[0]:
            valid_pad_dishing_bound_array = debond_dishing_intervals_from_coords(
                cfg,
                valid_die_pad_coords,
            )
            atomic_save_npy(save_path, valid_pad_dishing_bound_array)
    return valid_pad_dishing_bound_array


def _get_compact_uint_dtype(max_value: int):
    if max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    if max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def _build_pair_chunk_arrays(
    *,
    query_start: int,
    neighbor_list,
    dist_list,
    r_min: float,
    r_max: float,
    edges: np.ndarray,
    num_bins: int,
    index_dtype,
    bin_dtype,
):
    pair_i_list = []
    pair_j_list = []
    bin_id_list = []

    for local_idx, (neighbors, neighbor_distances) in enumerate(zip(neighbor_list, dist_list)):
        if neighbors.size == 0:
            continue

        pad_idx = query_start + local_idx
        dist_mask = (neighbor_distances >= r_min) & (neighbor_distances <= r_max)
        if not np.any(dist_mask):
            continue

        valid_neighbors = neighbors[dist_mask]
        valid_distances = neighbor_distances[dist_mask]
        upper_mask = valid_neighbors > pad_idx
        if not np.any(upper_mask):
            continue

        selected_neighbors = valid_neighbors[upper_mask]
        selected_distances = valid_distances[upper_mask]
        selected_bin_id = np.digitize(selected_distances, edges, right=False) - 1
        selected_bin_id = np.clip(selected_bin_id, 0, num_bins - 1)

        pair_i_list.append(np.full(selected_neighbors.size, pad_idx, dtype=index_dtype))
        pair_j_list.append(selected_neighbors.astype(index_dtype, copy=False))
        bin_id_list.append(selected_bin_id.astype(bin_dtype, copy=False))

    if not pair_i_list:
        return None

    return (
        np.concatenate(pair_i_list),
        np.concatenate(pair_j_list),
        np.concatenate(bin_id_list),
    )


def _build_interface_pair_parts(
    cfg,
    valid_pad_coords: np.ndarray,
    *,
    distance_interval_um: float,
    bin_width_um: float,
    pair_query_chunk_size: int,
    max_correlation_distance_um: float = None,
) -> List[dict]:
    pad_xy = np.asarray(valid_pad_coords[:, :2], dtype=np.float64)
    if pad_xy.shape[0] < 2:
        return []

    min_pair_distance_um = _get_min_pair_distance_um(cfg)
    max_distance_um = float(np.hypot(cfg.PAD_ARR_W_um, cfg.PAD_ARR_L_um))
    if max_correlation_distance_um is not None:
        max_distance_um = min(max_distance_um, float(max_correlation_distance_um))
    if max_distance_um <= min_pair_distance_um:
        return []

    num_parts = int(np.floor(max_distance_um / distance_interval_um)) + 1
    tree = KDTree(pad_xy)
    temp_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DESIGN, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    run_token = f"{cfg.INTERFACE}_pid{os.getpid()}"
    index_dtype = _get_compact_uint_dtype(pad_xy.shape[0] - 1)

    pair_parts = []
    for part_ind in range(num_parts):
        r_min = max(min_pair_distance_um, part_ind * distance_interval_um)
        r_max = min((part_ind + 1) * distance_interval_um, max_distance_um)
        if r_max <= r_min:
            continue

        edges = np.arange(r_min, r_max + bin_width_um, bin_width_um, dtype=np.float64)
        if edges.size < 2:
            edges = np.array([r_min, r_max], dtype=np.float64)
        elif edges[-1] < r_max:
            edges = np.append(edges, r_max)

        bin_center = 0.5 * (edges[:-1] + edges[1:])
        num_bins = len(bin_center)
        bin_dtype = _get_compact_uint_dtype(max(num_bins - 1, 0))
        counts = {
            mechanism: np.zeros((num_bins, 4), dtype=np.float64)
            for mechanism in FAILURE_MECHANISMS
        }
        pair_chunk_paths = []
        chunk_size = max(1, int(pair_query_chunk_size))
        chunk_counter = 0
        for query_start in range(0, pad_xy.shape[0], chunk_size):
            query_end = min(query_start + chunk_size, pad_xy.shape[0])
            neighbor_list, dist_list = tree.query_radius(
                pad_xy[query_start:query_end],
                r=r_max,
                return_distance=True,
            )
            pair_chunk = _build_pair_chunk_arrays(
                query_start=query_start,
                neighbor_list=neighbor_list,
                dist_list=dist_list,
                r_min=r_min,
                r_max=r_max,
                edges=edges,
                num_bins=num_bins,
                index_dtype=index_dtype,
                bin_dtype=bin_dtype,
            )
            if pair_chunk is None:
                continue

            pair_i, pair_j, bin_id = pair_chunk
            save_path = os.path.join(
                temp_dir,
                f"{run_token}_pair_part_{part_ind:03d}_chunk_{chunk_counter:05d}.npz",
            )
            np.savez(save_path, pair_i=pair_i, pair_j=pair_j, bin_id=bin_id)
            pair_chunk_paths.append(save_path)
            chunk_counter += 1

        pair_parts.append(
            {
                "part_index": part_ind,
                "r_min_um": r_min,
                "r_max_um": r_max,
                "bin_center_um": bin_center,
                "num_bins": num_bins,
                "pair_chunk_paths": pair_chunk_paths,
                "counts": counts,
            }
        )

    return pair_parts


def initialize_spatial_correlation_state(
    *,
    input_args: dict | None,
    cfg_dict: dict,
    pad_bitmap_collection_dict: dict,
    base_pad_coords_dict: dict,
    distance_interval_um: float = 5000.0,
    bin_width_um: float = 40.0,
    pair_query_chunk_size: int = 256,
    max_correlation_distance_um: float = None,
) -> dict:
    state_dict = {}
    for interface_name, cfg in cfg_dict.items():
        pad_bitmap_collection = pad_bitmap_collection_dict[interface_name]
        valid_pad_mask = _get_valid_pad_mask(pad_bitmap_collection)
        valid_flat_indices = np.flatnonzero(valid_pad_mask.flatten())
        base_pad_coords = base_pad_coords_dict[interface_name]
        valid_pad_coords = np.asarray(
            base_pad_coords[valid_pad_mask.flatten() == 1],
            dtype=np.float64,
        )

        valid_pad_dishing_bound_array = _get_valid_pad_dishing_bound_array(
            cfg,
            valid_pad_coords,
            input_args,
        )
        pair_parts = _build_interface_pair_parts(
            cfg,
            valid_pad_coords,
            distance_interval_um=distance_interval_um,
            bin_width_um=bin_width_um,
            pair_query_chunk_size=pair_query_chunk_size,
            max_correlation_distance_um=max_correlation_distance_um,
        )

        state_dict[interface_name] = {
            "valid_pad_mask": valid_pad_mask,
            "valid_flat_indices": valid_flat_indices,
            "valid_pad_dishing_bound_array": valid_pad_dishing_bound_array,
            "pair_parts": pair_parts,
        }
    return state_dict


def _compute_pad_fail_maps_for_stack(
    *,
    cfg,
    die_stack,
    interface_name: str,
    interface_index: int,
    stack_index: int,
    global_stack_index: int,
    pad_bitmap_collection: dict,
    base_pad_coords: np.ndarray,
    valid_pad_dishing_bound_array: np.ndarray,
    valid_pad_mask: np.ndarray,
    valid_flat_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    die_interface = die_stack.interfaces.interface_dict[interface_name]
    failure_params = die_stack.interfaces.failure_params_dict[interface_name]

    pad_arr_row, pad_arr_col = cfg.PAD_ARR_ROW, cfg.PAD_ARR_COL
    pad_arr_w_um, pad_arr_l_um = cfg.PAD_ARR_W_um, cfg.PAD_ARR_L_um
    pitch_c_um, pitch_r_um = cfg.PITCH_c_um, cfg.PITCH_r_um
    pad_top_r_um = cfg.PAD_TOP_R_um

    overlay_pad_fail_map = np.zeros((pad_arr_row, pad_arr_col), dtype=bool)
    particle_pad_fail_map = np.zeros((pad_arr_row, pad_arr_col), dtype=bool)
    mechanical_pad_fail_map = np.zeros((pad_arr_row, pad_arr_col), dtype=bool)
    esd_pad_fail_map = np.zeros((pad_arr_row, pad_arr_col), dtype=bool)

    system_translation_x_um = failure_params["system_translation_x_um"]
    system_translation_y_um = failure_params["system_translation_y_um"]
    system_rotation_rad = failure_params["system_rotation_rad"]
    system_magnification_ppm = failure_params["system_magnification_ppm"]
    max_allowed_misalignment_um = failure_params["MAX_ALLOWED_MISALIGNMENT_um"]
    approximate_set = cfg.approximate_set

    pad_misalignment = die_pad_misalignment(
        die_interface=die_interface,
        base_pad_coords=base_pad_coords,
        system_translation_x_um=system_translation_x_um,
        system_translation_y_um=system_translation_y_um,
        system_rotation_rad=system_rotation_rad,
        system_magnification_ppm=system_magnification_ppm,
        RANDOM_MISALIGNMENT_MEAN_um=cfg.RANDOM_MISALIGNMENT_MEAN_um,
        RANDOM_MISALIGNMENT_STD_um=cfg.RANDOM_MISALIGNMENT_STD_um,
        approximate_set=approximate_set,
    ).reshape(pad_arr_row, pad_arr_col)
    overlay_pad_fail_map = (pad_misalignment >= max_allowed_misalignment_um) & valid_pad_mask

    voids = np.asarray(failure_params["voids"], dtype=np.float64)
    if voids.size > 0:
        pad_array_box_x = die_interface.pad_array_box[2][0]
        pad_array_box_y = die_interface.pad_array_box[2][1]

        closest_x = np.maximum(pad_array_box_x, np.minimum(voids[:, 0], pad_array_box_x + pad_arr_w_um))
        closest_y = np.maximum(pad_array_box_y, np.minimum(voids[:, 1], pad_array_box_y + pad_arr_l_um))
        distances = np.sqrt((closest_x - voids[:, 0]) ** 2 + (closest_y - voids[:, 1]) ** 2)
        overlapping_mask = distances < voids[:, 2]

        if np.any(overlapping_mask):
            for void in voids[overlapping_mask]:
                i_coords_min = void[0] - void[2] - pad_top_r_um - pad_array_box_x
                i_coords_max = void[0] + void[2] + pad_top_r_um - pad_array_box_x
                j_coords_min = void[1] - void[2] - pad_top_r_um - pad_array_box_y
                j_coords_max = void[1] + void[2] + pad_top_r_um - pad_array_box_y
                i_min = max(0, int(np.floor(i_coords_min / pitch_c_um)))
                i_max = min(pad_arr_col - 1, int(np.ceil(i_coords_max / pitch_c_um)))
                j_min = max(0, int(np.floor(j_coords_min / pitch_r_um)))
                j_max = min(pad_arr_row - 1, int(np.ceil(j_coords_max / pitch_r_um)))

                check_pad_x_coords = pad_array_box_x + np.arange(i_min, i_max + 1) * pitch_c_um
                check_pad_y_coords = pad_array_box_y + np.arange(j_min, j_max + 1) * pitch_r_um
                check_pad_x_mesh, check_pad_y_mesh = np.meshgrid(
                    check_pad_x_coords,
                    check_pad_y_coords,
                    indexing="xy",
                )
                dist_sq = (check_pad_x_mesh - void[0]) ** 2 + (check_pad_y_mesh - void[1]) ** 2
                overlap_void_pad_mask = dist_sq < (void[2] + pad_top_r_um) ** 2
                sub_valid_pad_mask = valid_pad_mask[
                    pad_arr_row - j_max - 1: pad_arr_row - j_min,
                    i_min: i_max + 1,
                ]
                particle_pad_fail_map[
                    pad_arr_row - j_max - 1: pad_arr_row - j_min,
                    i_min: i_max + 1,
                ] |= overlap_void_pad_mask & sub_valid_pad_mask

    top_dish, bot_dish = Cu_gap_simulator(
        cfg.TOP_DISH_MEAN_nm,
        cfg.TOP_DISH_STD_nm,
        cfg.BOT_DISH_MEAN_nm,
        cfg.BOT_DISH_STD_nm,
        int(die_interface.num_pads),
    )
    cu_gap_map = np.full((pad_arr_row, pad_arr_col), np.nan)
    cu_gap_map[valid_pad_mask == 1] = top_dish + bot_dish

    zeta_0 = np.full((pad_arr_row, pad_arr_col), np.nan)
    zeta_1 = np.full((pad_arr_row, pad_arr_col), np.nan)
    zeta_0[valid_pad_mask == 1] = -valid_pad_dishing_bound_array[:, 1] * 2
    zeta_1[valid_pad_mask == 1] = -valid_pad_dishing_bound_array[:, 0] * 2
    mechanical_pad_fail_map = (
        ((cu_gap_map > zeta_1) | (cu_gap_map < zeta_0))
        & valid_pad_mask
    )

    esd_pad_idx, survive_bool = esd_failure_simulator(
        cfg=cfg,
        pad_coords_um=np.asarray(die_interface.pad_coords[valid_pad_mask.flatten() == 1], dtype=np.float64),
        pad_size_um=pad_top_r_um * 2,
        top_die_w_um=die_interface.DIE_W_um,
        top_die_h_um=die_interface.DIE_L_um,
        top_dish_nm_ext=top_dish,
        bot_dish_nm_ext=bot_dish,
        tilt_x_mean_deg=cfg.TILT_X_MEAN_DEG,
        tilt_x_std_deg=cfg.TILT_X_STD_DEG,
        tilt_y_mean_deg=cfg.TILT_Y_MEAN_DEG,
        tilt_y_std_deg=cfg.TILT_Y_STD_DEG,
        base_seed=interface_index * 1_000_003 + global_stack_index,
        dummy_pad_bitmap=pad_bitmap_collection["DUMMY_PAD_BITMAP"].flatten()[valid_pad_mask.flatten() == 1],
    )
    if esd_pad_idx is not None and survive_bool is False:
        flat_idx = int(valid_flat_indices[int(esd_pad_idx)])
        r_idx, c_idx = divmod(flat_idx, pad_arr_col)
        esd_pad_fail_map[r_idx, c_idx] = True

    overall_pad_fail_map = (
        overlay_pad_fail_map
        | particle_pad_fail_map
        | mechanical_pad_fail_map
        | esd_pad_fail_map
    )

    return {
        "overlay": overlay_pad_fail_map,
        "particle": particle_pad_fail_map,
        "mechanical": mechanical_pad_fail_map,
        "esd": esd_pad_fail_map,
        "overall": overall_pad_fail_map,
    }


def accumulate_spatial_correlation_counts(
    *,
    cfg_dict: dict,
    die_stack_list: list,
    pad_bitmap_collection_dict: dict,
    base_pad_coords_dict: dict,
    correlation_state_dict: dict,
    sample_index_offset: int = 0,
) -> dict:
    for stack_index, die_stack in enumerate(die_stack_list):
        for interface_index, (interface_name, cfg) in enumerate(cfg_dict.items()):
            state = correlation_state_dict[interface_name]
            valid_pad_mask = state["valid_pad_mask"]
            valid_flat_indices = state["valid_flat_indices"]
            pair_parts = state["pair_parts"]
            if not pair_parts:
                continue

            fail_map_dict = _compute_pad_fail_maps_for_stack(
                cfg=cfg,
                die_stack=die_stack,
                interface_name=interface_name,
                interface_index=interface_index,
                stack_index=stack_index,
                global_stack_index=sample_index_offset + stack_index,
                pad_bitmap_collection=pad_bitmap_collection_dict[interface_name],
                base_pad_coords=base_pad_coords_dict[interface_name],
                valid_pad_dishing_bound_array=state["valid_pad_dishing_bound_array"],
                valid_pad_mask=valid_pad_mask,
                valid_flat_indices=valid_flat_indices,
            )
            valid_fail_vector_dict = {
                mechanism: np.asarray(
                    fail_map_dict[mechanism][valid_pad_mask.astype(bool)],
                    dtype=bool,
                ).reshape(-1)
                for mechanism in FAILURE_MECHANISMS
            }

            for part in pair_parts:
                if not part["pair_chunk_paths"]:
                    continue
                num_bins = part["num_bins"]
                for chunk_path in part["pair_chunk_paths"]:
                    with np.load(chunk_path) as pair_chunk:
                        pair_i = pair_chunk["pair_i"]
                        pair_j = pair_chunk["pair_j"]
                        bin_id = pair_chunk["bin_id"]

                        for mechanism in FAILURE_MECHANISMS:
                            fail_bool = valid_fail_vector_dict[mechanism]
                            a = fail_bool[pair_i]
                            b = fail_bool[pair_j]
                            counts = part["counts"][mechanism]
                            counts[:, 0] += np.bincount(bin_id, weights=(a & b).astype(np.int64), minlength=num_bins)
                            counts[:, 1] += np.bincount(bin_id, weights=(a & ~b).astype(np.int64), minlength=num_bins)
                            counts[:, 2] += np.bincount(bin_id, weights=(~a & b).astype(np.int64), minlength=num_bins)
                            counts[:, 3] += np.bincount(bin_id, weights=(~a & ~b).astype(np.int64), minlength=num_bins)

    return correlation_state_dict


def get_spatial_correlation_coefficients(
    *,
    cfg_dict: dict,
    die_stack_list: list,
    pad_bitmap_collection_dict: dict,
    base_pad_coords_dict: dict,
    correlation_state_dict: dict,
    sample_index_offset: int = 0,
) -> dict:
    """
    Backward-compatible wrapper for the updated batch-based correlation flow.
    """
    return accumulate_spatial_correlation_counts(
        cfg_dict=cfg_dict,
        die_stack_list=die_stack_list,
        pad_bitmap_collection_dict=pad_bitmap_collection_dict,
        base_pad_coords_dict=base_pad_coords_dict,
        correlation_state_dict=correlation_state_dict,
        sample_index_offset=sample_index_offset,
    )


def finalize_spatial_correlation_coefficients(
    *,
    input_args: dict,
    cfg_dict: dict,
    correlation_state_dict: dict,
) -> dict:
    results_dict = {}
    plot_flag = bool(input_args.get("plot", False))
    plt = None
    if plot_flag:
        import matplotlib.pyplot as plt  # Lazy import so non-plot runs don't depend on matplotlib.

    for interface_name, cfg in cfg_dict.items():
        interface_output_dir = os.path.join(cfg.OUTPUT_DIR, input_args["ds_name"], interface_name)
        os.makedirs(interface_output_dir, exist_ok=True)

        pair_parts = correlation_state_dict[interface_name]["pair_parts"]
        distance_list = []
        phi_dict = {mechanism: [] for mechanism in FAILURE_MECHANISMS}

        for part in pair_parts:
            distance_list.append(part["bin_center_um"])
            for mechanism in FAILURE_MECHANISMS:
                counts = part["counts"][mechanism]
                n11, n10, n01, n00 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
                n1dot = n11 + n10
                n0dot = n01 + n00
                ndot1 = n11 + n01
                ndot0 = n10 + n00
                denominator = np.sqrt(n1dot * n0dot * ndot1 * ndot0)
                phi = np.divide(
                    n11 * n00 - n10 * n01,
                    denominator,
                    out=np.zeros_like(denominator),
                    where=denominator > 0,
                )
                phi_dict[mechanism].append(phi)

        distance_um = np.concatenate(distance_list) if distance_list else np.array([], dtype=np.float64)
        results_dict[interface_name] = {}

        for mechanism in FAILURE_MECHANISMS:
            phi = np.concatenate(phi_dict[mechanism]) if phi_dict[mechanism] else np.array([], dtype=np.float64)
            results_dict[interface_name][mechanism] = {
                "distance_um": distance_um,
                "phi": phi,
            }

            output_name_list = [mechanism]
            if mechanism == "mechanical":
                output_name_list.append("stress")

            for output_name in output_name_list:
                save_path = os.path.join(
                    interface_output_dir,
                    f"{output_name}_pad_fail_correlation_stats.txt",
                )
                with open(save_path, "w") as f:
                    f.write("distance phi\n")
                    for dist_val, phi_val in zip(distance_um, phi):
                        f.write(f"{dist_val:.2f} {phi_val:.6f}\n")
                print(f"{output_name} correlation statistics saved to {save_path}")

            if plot_flag and phi.size > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(distance_um, phi, marker="o", linestyle="-")
                ax.set_xlabel("Pad-to-Pad Distance (um)")
                ax.set_ylabel(f"{mechanism.capitalize()} Pad Failure Correlation Coefficient (phi)")
                ax.set_title(f"{interface_name}: {mechanism.capitalize()} Correlation vs Distance")
                ax.grid(True)
                plot_path = os.path.join(interface_output_dir, f"{mechanism}_phi_vs_distance.png")
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

    return results_dict
