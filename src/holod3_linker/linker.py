#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from holod3_linker.core import UnionFind, estimate_global_flow, l2_normalize, track_gate
from holod3_linker.outputs import write_track_outputs

LINKING_MODE_PRESETS: dict[str, dict[str, bool | float | int]] = {
    "fixed-lag": {
        "history_predictor": True,
        "online_max_gap": 12,
        "gap_gate": 1.2,
    },
    "legacy": {
        "history_predictor": False,
        "online_max_gap": 5,
        "gap_gate": 3.2,
    },
}


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


@dataclass
class TrackState3D:
    track_id: int
    last_idx: int
    last_frame: int
    pos: np.ndarray
    velocity: np.ndarray
    size_log: float
    gate_size_log: float
    diameter_size_log: float
    emb: np.ndarray
    hits: int = 1
    age: int = 1
    history_frames: list[int] = field(default_factory=list)
    history_pos: list[np.ndarray] = field(default_factory=list)


def robust_linear_history_prediction(
    state: TrackState3D,
    target_frame: int,
    window: int,
    min_points: int = 4,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Predict with a component-wise Theil--Sen line over recent detections.

    The wide-v2 motion is locally constant-velocity, while reconstructed depth
    occasionally contains large outliers.  Pairwise median slopes preserve the
    useful four-frame predictor without letting one bad depth dominate it.
    """
    required = max(2, int(min_points))
    take = min(max(required, int(window)), len(state.history_frames))
    if take < required:
        return None
    frames = np.asarray(state.history_frames[-take:], dtype=np.float64)
    pos = np.asarray(state.history_pos[-take:], dtype=np.float64)
    relative = frames - frames[-1]
    slopes: list[np.ndarray] = []
    for i in range(take - 1):
        dt = relative[i + 1 :] - relative[i]
        valid = dt > 0
        if np.any(valid):
            slopes.extend((pos[i + 1 :][valid] - pos[i]) / dt[valid, None])
    if not slopes:
        return None
    slope = np.median(np.asarray(slopes), axis=0)
    intercept = np.median(pos - relative[:, None] * slope[None, :], axis=0)
    horizon = float(target_frame - frames[-1])
    predicted = intercept + horizon * slope
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(slope)):
        return None
    return predicted.astype(np.float32), slope.astype(np.float32)


def finite_series(df: pd.DataFrame, primary: str, fallback: str) -> np.ndarray:
    if primary in df.columns:
        out = df[primary].to_numpy(np.float32)
        if fallback in df.columns:
            fb = df[fallback].to_numpy(np.float32)
            out = np.where(np.isfinite(out), out, fb)
        return out
    if fallback not in df.columns:
        raise ValueError(f"missing both {primary!r} and fallback {fallback!r}")
    return df[fallback].to_numpy(np.float32)


def prepare_link_columns(df: pd.DataFrame, xy_source: str, z_lap_scale: float) -> pd.DataFrame:
    required = {"frame", "file", "conf", "xc", "yc", "w", "h", "slice", "diameter_px"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")

    out = df.copy()
    if xy_source == "seg":
        x = finite_series(out, "seg_xc", "xc")
        y = finite_series(out, "seg_yc", "yc")
    elif xy_source == "bbox":
        x = out["xc"].to_numpy(np.float32)
        y = out["yc"].to_numpy(np.float32)
    elif xy_source == "x_lap":
        x = finite_series(out, "x_lap", "xc")
        y = finite_series(out, "y_lap", "yc")
    else:
        raise ValueError(f"unknown xy_source: {xy_source}")

    if "z_lap" in out.columns:
        z = out["z_lap"].to_numpy(np.float32)
    else:
        z = out["slice"].to_numpy(np.float32) * float(z_lap_scale)

    out["x_lap"] = x
    out["y_lap"] = y
    out["z_lap"] = z
    return out


def size_log_values(df: pd.DataFrame, size_source: str) -> np.ndarray:
    if size_source == "bbox_area":
        area = np.maximum(1.0, df["w"].to_numpy(np.float32) * df["h"].to_numpy(np.float32))
        return np.log(area).astype(np.float32)
    if size_source == "diameter":
        fallback = np.sqrt(np.maximum(1.0, df["w"].to_numpy(np.float32) * df["h"].to_numpy(np.float32)))
        diam = df["diameter_px"].to_numpy(np.float32)
        diam = np.where(np.isfinite(diam) & (diam > 0.0), diam, fallback)
        # Store diameter as a pseudo sqrt-area so existing gate logic keeps the same units.
        area = np.maximum(1.0, diam * diam)
        return np.log(area).astype(np.float32)
    raise ValueError(f"unknown size_source: {size_source}")


def load_embeddings(path: Path | None, df: pd.DataFrame) -> np.ndarray | None:
    if path is None:
        return None
    raw = np.load(path).astype(np.float32)
    if "row_id" in df.columns and np.issubdtype(df["row_id"].dtype, np.number):
        row_id = df["row_id"].to_numpy(np.int64)
        if row_id.min(initial=0) >= 0 and row_id.max(initial=-1) < raw.shape[0]:
            emb = raw[row_id]
        elif raw.shape[0] == len(df):
            emb = raw
        else:
            raise RuntimeError(f"cannot align embeddings {raw.shape} to rows {len(df)} with row_id range")
    elif raw.shape[0] == len(df):
        emb = raw
    else:
        raise RuntimeError(f"embedding row count {raw.shape[0]} does not match input rows {len(df)}")
    return l2_normalize(emb.astype(np.float32))


def effective_distance(a: np.ndarray, b: np.ndarray, z_weight: float) -> np.ndarray:
    diff = a - b
    if diff.shape[-1] != 3:
        raise ValueError("effective_distance expects 3D positions")
    return np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2 + (float(z_weight) * diff[..., 2]) ** 2)


def xy_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[..., :2] - b[..., :2]
    return np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)


def compute_z_tiebreak_penalty(
    base_cost: np.ndarray,
    finite: np.ndarray,
    dz: np.ndarray,
    gates: np.ndarray,
    *,
    z_weight: float,
    margin: float,
    penalty_weight: float,
    cap: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if z_weight <= 0.0 or margin <= 0.0 or penalty_weight <= 0.0 or not np.any(finite):
        return np.zeros_like(base_cost, dtype=np.float32), {
            "z_tiebreak_candidates": 0,
            "z_tiebreak_rows": 0,
            "z_tiebreak_cols": 0,
            "z_tiebreak_penalty_sum": 0.0,
        }

    masked = np.where(finite, base_cost, np.inf)
    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)
    row_near = finite & (base_cost <= row_best[:, None] + float(margin))
    col_near = finite & (base_cost <= col_best[None, :] + float(margin))
    row_amb = row_near.sum(axis=1) > 1
    col_amb = col_near.sum(axis=0) > 1
    ambiguous = (row_near & row_amb[:, None]) | (col_near & col_amb[None, :])
    if not np.any(ambiguous):
        return np.zeros_like(base_cost, dtype=np.float32), {
            "z_tiebreak_candidates": 0,
            "z_tiebreak_rows": 0,
            "z_tiebreak_cols": 0,
            "z_tiebreak_penalty_sum": 0.0,
        }

    z_norm = np.abs(dz) * float(z_weight) / np.maximum(gates[:, None], 1e-6)
    raw = np.minimum(float(cap), z_norm * z_norm) * float(penalty_weight)
    penalty = np.where(ambiguous, raw, 0.0).astype(np.float32)
    return penalty, {
        "z_tiebreak_candidates": int(np.count_nonzero(ambiguous)),
        "z_tiebreak_rows": int(np.count_nonzero(row_amb)),
        "z_tiebreak_cols": int(np.count_nonzero(col_amb)),
        "z_tiebreak_penalty_sum": float(penalty.sum()),
    }


def compute_scalar_tiebreak_penalty(
    base_cost: np.ndarray,
    finite: np.ndarray,
    delta: np.ndarray,
    *,
    margin: float,
    penalty_weight: float,
    cap: float,
    scale: float,
    prefix: str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    empty = {
        f"{prefix}_tiebreak_candidates": 0,
        f"{prefix}_tiebreak_rows": 0,
        f"{prefix}_tiebreak_cols": 0,
        f"{prefix}_tiebreak_penalty_sum": 0.0,
    }
    if margin <= 0.0 or penalty_weight <= 0.0 or not np.any(finite):
        return np.zeros_like(base_cost, dtype=np.float32), empty

    masked = np.where(finite, base_cost, np.inf)
    row_best = np.min(masked, axis=1)
    col_best = np.min(masked, axis=0)
    row_near = finite & (base_cost <= row_best[:, None] + float(margin))
    col_near = finite & (base_cost <= col_best[None, :] + float(margin))
    row_amb = row_near.sum(axis=1) > 1
    col_amb = col_near.sum(axis=0) > 1
    ambiguous = (row_near & row_amb[:, None]) | (col_near & col_amb[None, :])
    if not np.any(ambiguous):
        return np.zeros_like(base_cost, dtype=np.float32), empty

    norm = np.abs(delta) / max(float(scale), 1e-6)
    raw = np.minimum(float(cap), norm * norm) * float(penalty_weight)
    penalty = np.where(ambiguous, raw, 0.0).astype(np.float32)
    return penalty, {
        f"{prefix}_tiebreak_candidates": int(np.count_nonzero(ambiguous)),
        f"{prefix}_tiebreak_rows": int(np.count_nonzero(row_amb)),
        f"{prefix}_tiebreak_cols": int(np.count_nonzero(col_amb)),
        f"{prefix}_tiebreak_penalty_sum": float(penalty.sum()),
    }


def online_link_3d(
    df: pd.DataFrame,
    emb: np.ndarray | None,
    global_flow_xy: tuple[float, float],
    *,
    z_mode: str,
    z_weight: float,
    z_tiebreak_margin: float,
    z_tiebreak_penalty: float,
    z_tiebreak_cap: float,
    max_gap: int,
    base_gate: float,
    gap_gate: float,
    startup_gate: float,
    max_cost: float,
    gap_max_cost: float,
    gap_min_hits: int,
    gap_assignment_margin: float,
    gap_diameter_size_weight: float,
    motion_weight: float,
    feature_weight: float,
    size_weight: float,
    size_source: str,
    gate_size_source: str,
    size_cost_scale: float,
    diameter_size_weight: float,
    diameter_size_cost_scale: float,
    diameter_tiebreak_margin: float,
    diameter_tiebreak_penalty: float,
    diameter_tiebreak_cap: float,
    age_bonus: float,
    z_velocity_alpha: float,
    z_start_velocity_alpha: float,
    history_predictor: bool,
    history_window: int,
    history_blend: float,
    history_z_blend: float,
    gap_history_min_points: int,
    gap_history_window: int,
    gap_history_blend: float,
    gap_history_z_blend: float,
    gap_z_tiebreak_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(df)
    assigned = np.zeros(n, dtype=np.int32)
    tracks: dict[int, TrackState3D] = {}
    active: set[int] = set()
    next_id = 1
    frames = sorted(df["frame"].unique())
    pos_all = df[["x_lap", "y_lap", "z_lap"]].to_numpy(np.float32)
    size_log_all = size_log_values(df, size_source)
    gate_size_log_all = size_log_values(df, gate_size_source)
    diameter_size_log_all = size_log_values(df, "diameter")
    frame_indices = {frame: grp.index.to_numpy(np.int64) for frame, grp in df.groupby("frame", sort=True)}
    gf = np.asarray([global_flow_xy[0], global_flow_xy[1], 0.0], dtype=np.float32)

    counters = defaultdict(int)
    desc = f"Online {z_mode} z={z_weight:g}"
    for frame in tqdm(frames, desc=desc, unit="frame"):
        det_idx = frame_indices[frame]
        det_pos = pos_all[det_idx]
        det_size_log = size_log_all[det_idx]

        live_ids = sorted(tid for tid in active if 0 < frame - tracks[tid].last_frame <= max_gap)
        matched_dets: set[int] = set()

        if live_ids and len(det_idx):
            pred_pos = np.empty((len(live_ids), 3), dtype=np.float32)
            gates = np.empty(len(live_ids), dtype=np.float32)
            gap_rows = np.zeros(len(live_ids), dtype=bool)
            for i, tid in enumerate(live_ids):
                st = tracks[tid]
                dt = frame - st.last_frame
                gap_rows[i] = dt > 1
                if st.hits < 2:
                    v = gf
                else:
                    v = st.velocity.copy()
                    v[:2] = 0.88 * st.velocity[:2] + 0.12 * gf[:2]
                ema_pred = st.pos + v * dt
                pred_pos[i] = ema_pred
                if history_predictor:
                    is_gap = dt > 1
                    prediction_min_points = int(gap_history_min_points) if is_gap else 4
                    prediction_window = (
                        int(gap_history_window) if is_gap and gap_history_window > 0 else int(history_window)
                    )
                    prediction_blend = (
                        float(gap_history_blend) if is_gap and gap_history_blend >= 0.0 else float(history_blend)
                    )
                    prediction_z_blend = (
                        float(gap_history_z_blend)
                        if is_gap and gap_history_z_blend >= 0.0
                        else float(history_z_blend)
                    )
                    history_prediction = robust_linear_history_prediction(
                        st,
                        int(frame),
                        prediction_window,
                        min_points=prediction_min_points,
                    )
                    if history_prediction is not None:
                        robust_pred, robust_v = history_prediction
                        pred_pos[i, :2] = (
                            (1.0 - prediction_blend) * ema_pred[:2]
                            + prediction_blend * robust_pred[:2]
                        )
                        pred_pos[i, 2] = (
                            (1.0 - prediction_z_blend) * ema_pred[2]
                            + prediction_z_blend * robust_pred[2]
                        )
                        v = robust_v
                        counters["history_predictions"] += 1
                        if is_gap:
                            counters["gap_history_predictions"] += 1
                            if prediction_min_points <= 3 and len(st.history_frames) == 3:
                                counters["three_point_gap_history_predictions"] += 1
                speed_xy = float(np.linalg.norm(v[:2]))
                gates[i] = track_gate(dt, math.exp(0.5 * st.gate_size_log), speed_xy, base_gate, gap_gate)
                if st.hits < 2 and dt == 1:
                    gates[i] = max(gates[i], startup_gate)

            diff = pred_pos[:, None, :] - det_pos[None, :, :]
            if z_mode == "distance":
                dist = effective_distance(pred_pos[:, None, :], det_pos[None, :, :], z_weight)
            elif z_mode == "tiebreak":
                dist = xy_distance(pred_pos[:, None, :], det_pos[None, :, :])
            else:
                raise ValueError(f"unknown z_mode: {z_mode}")
            finite = dist <= gates[:, None]
            if gap_min_hits > 1 and np.any(gap_rows):
                track_hits = np.asarray([tracks[tid].hits for tid in live_ids], dtype=np.int32)
                insufficient = gap_rows & (track_hits < int(gap_min_hits))
                counters["gap_rows_rejected_min_hits"] += int(np.count_nonzero(insufficient))
                finite[insufficient, :] = False
            cost = np.full(dist.shape, 1.0e6, dtype=np.float32)
            if np.any(finite):
                motion_cost = (dist / np.maximum(gates[:, None], 1e-6)) ** 2
                size_cost = np.abs(
                    np.asarray([tracks[tid].size_log for tid in live_ids], dtype=np.float32)[:, None]
                    - det_size_log[None, :]
                )
                size_cost = np.clip(size_cost / max(float(size_cost_scale), 1e-6), 0.0, 2.0)
                cost_part = motion_weight * motion_cost + size_weight * size_cost
                if diameter_size_weight > 0.0:
                    diam_cost = np.abs(
                        np.asarray([tracks[tid].diameter_size_log for tid in live_ids], dtype=np.float32)[:, None]
                        - diameter_size_log_all[det_idx][None, :]
                    )
                    diam_cost = np.clip(diam_cost / max(float(diameter_size_cost_scale), 1e-6), 0.0, 2.0)
                    cost_part += float(diameter_size_weight) * diam_cost
                if gap_diameter_size_weight > 0.0 and np.any(gap_rows):
                    gap_diam_cost = np.abs(
                        np.asarray([tracks[tid].diameter_size_log for tid in live_ids], dtype=np.float32)[:, None]
                        - diameter_size_log_all[det_idx][None, :]
                    )
                    gap_diam_cost = np.clip(
                        gap_diam_cost / max(float(diameter_size_cost_scale), 1e-6), 0.0, 2.0
                    )
                    cost_part += (
                        float(gap_diameter_size_weight) * gap_rows[:, None].astype(np.float32) * gap_diam_cost
                    )
                if age_bonus > 0:
                    age = np.asarray([tracks[tid].hits for tid in live_ids], dtype=np.float32)
                    bonus = age_bonus * np.minimum(1.0, np.log1p(age) / math.log(40.0))
                    cost_part -= bonus[:, None]
                if emb is not None:
                    track_emb = np.stack([tracks[tid].emb for tid in live_ids], axis=0)
                    feat_dist = 1.0 - np.clip(track_emb @ emb[det_idx].T, -1.0, 1.0)
                    cost_part += feature_weight * np.clip(feat_dist / 0.42, 0.0, 2.5)
                if diameter_tiebreak_penalty > 0.0:
                    diam_delta = (
                        np.asarray([tracks[tid].diameter_size_log for tid in live_ids], dtype=np.float32)[:, None]
                        - diameter_size_log_all[det_idx][None, :]
                    )
                    penalty, ddiag = compute_scalar_tiebreak_penalty(
                        cost_part,
                        finite,
                        diam_delta,
                        margin=diameter_tiebreak_margin,
                        penalty_weight=diameter_tiebreak_penalty,
                        cap=diameter_tiebreak_cap,
                        scale=diameter_size_cost_scale,
                        prefix="diameter",
                    )
                    cost_part += penalty
                    for key, value in ddiag.items():
                        counters[key] += value
                if z_mode == "tiebreak":
                    penalty, zdiag = compute_z_tiebreak_penalty(
                        cost_part,
                        finite,
                        diff[:, :, 2],
                        gates,
                        z_weight=z_weight,
                        margin=z_tiebreak_margin,
                        penalty_weight=z_tiebreak_penalty,
                        cap=z_tiebreak_cap,
                    )
                    if gap_z_tiebreak_scale >= 0.0 and np.any(gap_rows):
                        counters["gap_z_tiebreak_penalty_unscaled_sum"] += float(penalty[gap_rows, :].sum())
                        penalty[gap_rows, :] *= float(gap_z_tiebreak_scale)
                        counters["gap_z_tiebreak_scaled_rows"] += int(np.count_nonzero(gap_rows))
                        zdiag["z_tiebreak_penalty_sum"] = float(penalty.sum())
                    cost_part += penalty
                    for key, value in zdiag.items():
                        counters[key] += value
                cost[finite] = cost_part[finite]
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols, strict=False):
                    is_gap = bool(gap_rows[int(r)])
                    accepted_max_cost = float(gap_max_cost) if is_gap and gap_max_cost >= 0.0 else float(max_cost)
                    if cost[r, c] > accepted_max_cost or not finite[r, c]:
                        if is_gap:
                            counters["gap_matches_rejected_max_cost"] += 1
                        continue
                    if is_gap and gap_assignment_margin > 0.0:
                        row_alternatives = cost[int(r), :].copy()
                        row_alternatives[int(c)] = np.inf
                        col_alternatives = cost[:, int(c)].copy()
                        col_alternatives[int(r)] = np.inf
                        row_alt = float(np.min(row_alternatives))
                        col_alt = float(np.min(col_alternatives))
                        assignment_margin = min(row_alt - float(cost[r, c]), col_alt - float(cost[r, c]))
                        if assignment_margin < float(gap_assignment_margin):
                            counters["gap_matches_rejected_assignment_margin"] += 1
                            continue
                        counters["accepted_gap_assignment_margin_sum"] += assignment_margin
                        counters["accepted_gap_assignment_margin_n"] += 1
                    if is_gap:
                        counters["accepted_gap_cost_sum"] += float(cost[r, c])
                        counters["accepted_gap_cost_n"] += 1
                    tid = live_ids[int(r)]
                    j = int(det_idx[int(c)])
                    st = tracks[tid]
                    dt = frame - st.last_frame
                    measured_v = (pos_all[j] - st.pos) / max(1, dt)
                    new_v = np.empty(3, dtype=np.float32)
                    if st.hits < 2:
                        new_v[:2] = 0.15 * gf[:2] + 0.85 * measured_v[:2]
                        new_v[2] = float(z_start_velocity_alpha) * measured_v[2]
                    else:
                        new_v[:2] = 0.72 * st.velocity[:2] + 0.28 * measured_v[:2]
                        new_v[2] = (1.0 - float(z_velocity_alpha)) * st.velocity[2] + float(z_velocity_alpha) * measured_v[2]
                    st.velocity = new_v
                    st.pos = pos_all[j].astype(np.float32)
                    st.last_idx = j
                    st.last_frame = int(frame)
                    st.size_log = 0.86 * st.size_log + 0.14 * float(size_log_all[j])
                    st.gate_size_log = 0.86 * st.gate_size_log + 0.14 * float(gate_size_log_all[j])
                    st.diameter_size_log = 0.86 * st.diameter_size_log + 0.14 * float(diameter_size_log_all[j])
                    if emb is not None:
                        st.emb = l2_normalize((0.88 * st.emb + 0.12 * emb[j])[None, :])[0].astype(np.float32)
                    st.hits += 1
                    st.age += dt
                    st.history_frames.append(int(frame))
                    st.history_pos.append(pos_all[j].astype(np.float32))
                    keep = max(4, int(history_window))
                    if len(st.history_frames) > keep:
                        del st.history_frames[:-keep]
                        del st.history_pos[:-keep]
                    assigned[j] = tid
                    matched_dets.add(j)
                    counters["online_matches"] += 1
                    if dt > 1:
                        counters["online_gap_matches"] += 1
                    counters["accepted_abs_dz_sum"] += float(abs(diff[r, c, 2]))
                    counters["accepted_abs_dz_n"] += 1

        for j in det_idx:
            if int(j) in matched_dets:
                continue
            tid = next_id
            next_id += 1
            e = emb[j].astype(np.float32) if emb is not None else np.zeros(1, dtype=np.float32)
            tracks[tid] = TrackState3D(
                track_id=tid,
                last_idx=int(j),
                last_frame=int(frame),
                pos=pos_all[j].astype(np.float32),
                velocity=gf.copy(),
                size_log=float(size_log_all[j]),
                gate_size_log=float(gate_size_log_all[j]),
                diameter_size_log=float(diameter_size_log_all[j]),
                emb=e,
                history_frames=[int(frame)],
                history_pos=[pos_all[j].astype(np.float32)],
            )
            assigned[j] = tid
            active.add(tid)
            counters["new_tracks"] += 1

        stale = [tid for tid in active if frame - tracks[tid].last_frame > max_gap]
        for tid in stale:
            active.remove(tid)

    n_dz = int(counters["accepted_abs_dz_n"])
    n_gap_cost = int(counters["accepted_gap_cost_n"])
    n_gap_margin = int(counters["accepted_gap_assignment_margin_n"])
    diag = {
        "initial_track_count": int(next_id - 1),
        "online_matches": int(counters["online_matches"]),
        "online_gap_matches": int(counters["online_gap_matches"]),
        "gap_rows_rejected_min_hits": int(counters["gap_rows_rejected_min_hits"]),
        "gap_matches_rejected_max_cost": int(counters["gap_matches_rejected_max_cost"]),
        "gap_matches_rejected_assignment_margin": int(counters["gap_matches_rejected_assignment_margin"]),
        "accepted_gap_cost_mean": (
            float(counters["accepted_gap_cost_sum"] / n_gap_cost) if n_gap_cost else None
        ),
        "accepted_gap_assignment_margin_mean": (
            float(counters["accepted_gap_assignment_margin_sum"] / n_gap_margin) if n_gap_margin else None
        ),
        "new_tracks": int(counters["new_tracks"]),
        "accepted_abs_dz_lap_mean": float(counters["accepted_abs_dz_sum"] / n_dz) if n_dz else None,
        "z_tiebreak_candidates": int(counters["z_tiebreak_candidates"]),
        "z_tiebreak_rows": int(counters["z_tiebreak_rows"]),
        "z_tiebreak_cols": int(counters["z_tiebreak_cols"]),
        "z_tiebreak_penalty_sum": float(counters["z_tiebreak_penalty_sum"]),
        "diameter_tiebreak_candidates": int(counters["diameter_tiebreak_candidates"]),
        "diameter_tiebreak_rows": int(counters["diameter_tiebreak_rows"]),
        "diameter_tiebreak_cols": int(counters["diameter_tiebreak_cols"]),
        "diameter_tiebreak_penalty_sum": float(counters["diameter_tiebreak_penalty_sum"]),
        "history_predictions": int(counters["history_predictions"]),
        "gap_history_predictions": int(counters["gap_history_predictions"]),
        "three_point_gap_history_predictions": int(counters["three_point_gap_history_predictions"]),
        "gap_z_tiebreak_scaled_rows": int(counters["gap_z_tiebreak_scaled_rows"]),
        "gap_z_tiebreak_penalty_unscaled_sum": float(counters["gap_z_tiebreak_penalty_unscaled_sum"]),
    }
    return assigned, diag


def tracklet_summary_3d(
    df: pd.DataFrame,
    track_ids: np.ndarray,
    emb: np.ndarray | None,
    *,
    size_source: str,
    gate_size_source: str,
) -> dict[int, dict[str, Any]]:
    size_log_all = size_log_values(df, size_source)
    gate_size_log_all = size_log_values(df, gate_size_source)
    diameter_size_log_all = size_log_values(df, "diameter")
    tmp = pd.DataFrame(
        {
            "track_id": track_ids,
            "frame": df["frame"].to_numpy(np.int32),
            "x": df["x_lap"].to_numpy(np.float32),
            "y": df["y_lap"].to_numpy(np.float32),
            "z": df["z_lap"].to_numpy(np.float32),
            "size_log": size_log_all,
            "gate_size_log": gate_size_log_all,
            "diameter_size_log": diameter_size_log_all,
            "idx": np.arange(len(df), dtype=np.int32),
        }
    )
    out: dict[int, dict[str, Any]] = {}
    for tid, g in tmp.groupby("track_id", sort=False):
        g = g.sort_values("frame")
        idx = g["idx"].to_numpy(np.int32)
        frames = g["frame"].to_numpy(np.int32)
        pos = g[["x", "y", "z"]].to_numpy(np.float32)
        area_log = g["size_log"].to_numpy(np.float32)
        gate_area_log = g["gate_size_log"].to_numpy(np.float32)
        diameter_area_log = g["diameter_size_log"].to_numpy(np.float32)

        def endpoint_velocity(tail: bool) -> np.ndarray:
            take = min(6, len(g))
            pp = pos[-take:] if tail else pos[:take]
            ff = frames[-take:] if tail else frames[:take]
            if len(pp) >= 2 and ff[-1] != ff[0]:
                dt = (ff - ff.mean()).astype(np.float32)
                denom = float(np.sum(dt * dt))
                if denom > 0:
                    centered = pp - pp.mean(axis=0, keepdims=True)
                    return (dt[:, None] * centered).sum(axis=0).astype(np.float32) / denom
            return np.zeros(3, dtype=np.float32)

        if emb is not None:
            head = l2_normalize(emb[idx[: min(5, len(idx))]].mean(axis=0, keepdims=True))[0]
            tail = l2_normalize(emb[idx[-min(5, len(idx)) :]].mean(axis=0, keepdims=True))[0]
        else:
            head = tail = np.zeros(1, dtype=np.float32)

        out[int(tid)] = {
            "id": int(tid),
            "len": int(len(g)),
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "start_pos": pos[0],
            "end_pos": pos[-1],
            "start_v": endpoint_velocity(tail=False),
            "end_v": endpoint_velocity(tail=True),
            "start_size": float(np.median(area_log[: min(5, len(area_log))])),
            "end_size": float(np.median(area_log[-min(5, len(area_log)) :])),
            "start_gate_size": float(np.median(gate_area_log[: min(5, len(gate_area_log))])),
            "end_gate_size": float(np.median(gate_area_log[-min(5, len(gate_area_log)) :])),
            "start_diameter_size": float(np.median(diameter_area_log[: min(5, len(diameter_area_log))])),
            "end_diameter_size": float(np.median(diameter_area_log[-min(5, len(diameter_area_log)) :])),
            "head_emb": head.astype(np.float32),
            "tail_emb": tail.astype(np.float32),
        }
    return out


def stitch_tracklets_3d(
    df: pd.DataFrame,
    track_ids: np.ndarray,
    emb: np.ndarray | None,
    global_flow_xy: tuple[float, float],
    *,
    z_mode: str,
    z_weight: float,
    z_tiebreak_margin: float,
    z_tiebreak_penalty: float,
    z_tiebreak_cap: float,
    max_gap: int,
    max_cost: float,
    base_gate: float,
    gap_gate: float,
    startup_gate: float,
    min_len_for_feature: int,
    large_sqrt_area: float,
    large_gate_scale: float,
    large_max_cost: float,
    large_feature_dist: float,
    large_size_log_max: float,
    large_vel_cost_max: float,
    size_source: str,
    gate_size_source: str,
    size_cost_scale: float,
    diameter_size_weight: float,
    diameter_size_cost_scale: float,
    diameter_tiebreak_margin: float,
    diameter_tiebreak_penalty: float,
    diameter_tiebreak_cap: float,
    max_examples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if max_gap <= 0:
        return track_ids, {"stitch_merges": 0}

    unique_ids = np.unique(track_ids)
    uf = UnionFind(unique_ids)
    summaries = tracklet_summary_3d(df, track_ids, emb, size_source=size_source, gate_size_source=gate_size_source)
    gf = np.asarray([global_flow_xy[0], global_flow_xy[1], 0.0], dtype=np.float32)
    merges: list[dict[str, Any]] = []
    counters = defaultdict(int)

    for _pass in range(2):
        roots = {tid: uf.find(tid) for tid in summaries}
        root_members: dict[int, list[int]] = defaultdict(list)
        for tid, root in roots.items():
            root_members[root].append(tid)

        root_summary: dict[int, dict[str, Any]] = {}
        for root, members in root_members.items():
            starts = [summaries[m] for m in members]
            first = min(starts, key=lambda s: s["start_frame"])
            last = max(starts, key=lambda s: s["end_frame"])
            root_summary[root] = {
                "id": root,
                "len": int(sum(s["len"] for s in starts)),
                "start_frame": first["start_frame"],
                "end_frame": last["end_frame"],
                "start_pos": first["start_pos"],
                "end_pos": last["end_pos"],
                "start_v": first["start_v"],
                "end_v": last["end_v"],
                "start_size": first["start_size"],
                "end_size": last["end_size"],
                "start_gate_size": first["start_gate_size"],
                "end_gate_size": last["end_gate_size"],
                "start_diameter_size": first["start_diameter_size"],
                "end_diameter_size": last["end_diameter_size"],
                "head_emb": first["head_emb"],
                "tail_emb": last["tail_emb"],
            }

        by_start: dict[int, list[int]] = defaultdict(list)
        for tid, s in root_summary.items():
            by_start[int(s["start_frame"])].append(tid)

        pass_merges = 0
        used_starts: set[int] = set()
        used_ends: set[int] = set()
        for gap in range(1, max_gap + 1):
            ends = [tid for tid in root_summary if tid not in used_ends]
            starts_by_frame = {f: [tid for tid in ids if tid not in used_starts] for f, ids in by_start.items()}
            pairs: list[tuple[Any, ...]] = []
            for a in ends:
                sa = root_summary[a]
                target_frame = int(sa["end_frame"]) + gap
                starts = starts_by_frame.get(target_frame)
                if not starts:
                    continue
                v = sa["end_v"].copy()
                if np.linalg.norm(v[:2]) < 0.5:
                    v[:2] = gf[:2]
                    v[2] = 0.0
                pred = sa["end_pos"] + v * gap
                b_pos = np.stack([root_summary[b]["start_pos"] for b in starts], axis=0)
                if z_mode == "distance":
                    dist = effective_distance(b_pos, pred[None, :], z_weight)
                elif z_mode == "tiebreak":
                    dist = xy_distance(b_pos, pred[None, :])
                else:
                    raise ValueError(f"unknown z_mode: {z_mode}")
                speed_xy = float(np.linalg.norm(v[:2]))
                gate = track_gate(gap, math.exp(0.5 * sa["end_gate_size"]), speed_xy, base_gate, gap_gate)
                if sa["len"] < 2 and gap == 1:
                    gate = max(gate, startup_gate)
                for j, b in enumerate(starts):
                    sb = root_summary[b]
                    end_sqrt_area = math.exp(0.5 * sa["end_gate_size"])
                    start_sqrt_area = math.exp(0.5 * sb["start_gate_size"])
                    large_pair = min(end_sqrt_area, start_sqrt_area) >= large_sqrt_area
                    effective_gate = gate * large_gate_scale if large_pair else gate
                    if dist[j] > effective_gate:
                        continue

                    if large_pair:
                        counters["large_candidates"] += 1

                    size_log_delta = abs(sa["end_size"] - sb["start_size"])
                    size_cost = min(2.0, size_log_delta / max(float(size_cost_scale), 1e-6))
                    diameter_size_log_delta = abs(sa["end_diameter_size"] - sb["start_diameter_size"])
                    diameter_size_cost = min(2.0, diameter_size_log_delta / max(float(diameter_size_cost_scale), 1e-6))
                    vel_b = sb["start_v"].copy()
                    if np.linalg.norm(vel_b[:2]) < 0.5:
                        vel_b[:2] = gf[:2]
                        vel_b[2] = 0.0
                    dv = v - vel_b
                    if z_mode == "distance":
                        vel_dist = math.sqrt(float(dv[0] ** 2 + dv[1] ** 2 + (float(z_weight) * dv[2]) ** 2))
                    else:
                        vel_dist = math.sqrt(float(dv[0] ** 2 + dv[1] ** 2))
                    vel_cost = min(2.0, vel_dist / 6.0)
                    feat_cost = 0.0
                    feature_dist = None
                    use_feature = emb is not None and (
                        sa["len"] >= min_len_for_feature or sb["len"] >= min_len_for_feature or large_pair
                    )
                    if use_feature:
                        feat_dist = 1.0 - float(np.clip(np.dot(sa["tail_emb"], sb["head_emb"]), -1.0, 1.0))
                        feature_dist = feat_dist
                        feat_cost = min(2.5, feat_dist / 0.42)

                    normal_cost = (
                        0.55 * (float(dist[j]) / max(gate, 1e-6)) ** 2
                        + 0.15 * vel_cost
                        + 0.12 * size_cost
                        + float(diameter_size_weight) * diameter_size_cost
                        + 0.18 * feat_cost
                    )
                    cost = normal_cost
                    effective_max_cost = max_cost
                    accepted_by = "normal"
                    large_feature_ok = (
                        large_pair
                        and feature_dist is not None
                        and feature_dist <= large_feature_dist
                        and size_log_delta <= large_size_log_max
                        and vel_cost <= large_vel_cost_max
                    )
                    if large_feature_ok:
                        counters["large_feature_ok_candidates"] += 1
                        large_cost = (
                            0.48 * (float(dist[j]) / max(effective_gate, 1e-6)) ** 2
                            + 0.12 * vel_cost
                            + 0.08 * size_cost
                            + float(diameter_size_weight) * diameter_size_cost
                            + 0.12 * feat_cost
                        )
                        if large_cost < cost:
                            cost = large_cost
                        effective_max_cost = max(max_cost, large_max_cost)
                        if normal_cost > max_cost:
                            accepted_by = "large_feature"

                    if cost <= effective_max_cost:
                        pairs.append(
                            (
                                cost,
                                a,
                                b,
                                float(dist[j]),
                                float(effective_gate),
                                float(abs(b_pos[j, 2] - pred[2])),
                                float(feat_cost),
                                float(feature_dist) if feature_dist is not None else None,
                                bool(large_feature_ok),
                                accepted_by,
                                float(normal_cost),
                                float(effective_max_cost),
                                float(diameter_size_log_delta),
                            )
                        )
            if diameter_tiebreak_penalty > 0.0 and diameter_tiebreak_margin > 0.0:
                best_end: dict[int, float] = {}
                best_start: dict[int, float] = {}
                count_end: dict[int, int] = defaultdict(int)
                count_start: dict[int, int] = defaultdict(int)
                for pair in pairs:
                    base_cost, a, b = float(pair[0]), int(pair[1]), int(pair[2])
                    best_end[a] = min(best_end.get(a, math.inf), base_cost)
                    best_start[b] = min(best_start.get(b, math.inf), base_cost)
                for pair in pairs:
                    base_cost, a, b = float(pair[0]), int(pair[1]), int(pair[2])
                    if base_cost <= best_end[a] + diameter_tiebreak_margin:
                        count_end[a] += 1
                    if base_cost <= best_start[b] + diameter_tiebreak_margin:
                        count_start[b] += 1

                adjusted_pairs: list[tuple[Any, ...]] = []
                for pair in pairs:
                    base_cost, a, b = float(pair[0]), int(pair[1]), int(pair[2])
                    ambiguous = (
                        count_end[a] > 1
                        and base_cost <= best_end[a] + diameter_tiebreak_margin
                    ) or (
                        count_start[b] > 1
                        and base_cost <= best_start[b] + diameter_tiebreak_margin
                    )
                    if ambiguous:
                        diameter_norm = float(pair[12]) / max(float(diameter_size_cost_scale), 1e-6)
                        penalty = float(diameter_tiebreak_penalty) * min(
                            float(diameter_tiebreak_cap), diameter_norm * diameter_norm
                        )
                        counters["diameter_tiebreak_candidates"] += 1
                        counters["diameter_tiebreak_penalty_sum"] += penalty
                        pair = (base_cost + penalty, *pair[1:])
                    adjusted_pairs.append(pair)
                pairs = adjusted_pairs

            if z_mode == "tiebreak" and z_weight > 0.0 and z_tiebreak_margin > 0.0 and z_tiebreak_penalty > 0.0:
                best_end: dict[int, float] = {}
                best_start: dict[int, float] = {}
                count_end: dict[int, int] = defaultdict(int)
                count_start: dict[int, int] = defaultdict(int)
                for pair in pairs:
                    base_cost, a, b = float(pair[0]), int(pair[1]), int(pair[2])
                    best_end[a] = min(best_end.get(a, math.inf), base_cost)
                    best_start[b] = min(best_start.get(b, math.inf), base_cost)
                for pair in pairs:
                    base_cost, a, b = float(pair[0]), int(pair[1]), int(pair[2])
                    if base_cost <= best_end[a] + z_tiebreak_margin:
                        count_end[a] += 1
                    if base_cost <= best_start[b] + z_tiebreak_margin:
                        count_start[b] += 1

                adjusted_pairs: list[tuple[Any, ...]] = []
                for pair in pairs:
                    base_cost, a, b, _dist, gate, abs_dz = (
                        float(pair[0]),
                        int(pair[1]),
                        int(pair[2]),
                        float(pair[3]),
                        float(pair[4]),
                        float(pair[5]),
                    )
                    ambiguous = (
                        count_end[a] > 1
                        and base_cost <= best_end[a] + z_tiebreak_margin
                    ) or (
                        count_start[b] > 1
                        and base_cost <= best_start[b] + z_tiebreak_margin
                    )
                    if ambiguous:
                        z_norm = abs_dz * float(z_weight) / max(gate, 1e-6)
                        penalty = float(z_tiebreak_penalty) * min(float(z_tiebreak_cap), z_norm * z_norm)
                        counters["z_tiebreak_candidates"] += 1
                        counters["z_tiebreak_penalty_sum"] += penalty
                        pair = (base_cost + penalty, *pair[1:])
                    adjusted_pairs.append(pair)
                pairs = adjusted_pairs

            for (
                cost,
                a,
                b,
                dist,
                gate,
                abs_dz,
                feat_cost,
                feature_dist,
                large_feature_ok,
                accepted_by,
                normal_cost,
                effective_max_cost,
                diameter_size_log_delta,
            ) in sorted(pairs, key=lambda x: x[0]):
                if cost > effective_max_cost:
                    continue
                if a in used_ends or b in used_starts:
                    continue
                if uf.find(a) == uf.find(b):
                    continue
                if uf.union(a, b):
                    used_ends.add(a)
                    used_starts.add(b)
                    pass_merges += 1
                    if len(merges) < max_examples:
                        merges.append(
                            {
                                "pass": _pass,
                                "gap": gap,
                                "from": int(a),
                                "to": int(b),
                                "cost": float(cost),
                                "dist": dist,
                                "gate": gate,
                                "abs_dz_lap": abs_dz,
                                "feature_cost": feat_cost,
                                "feature_dist": feature_dist,
                                "large_feature": large_feature_ok,
                                "accepted_by": accepted_by,
                                "normal_cost": normal_cost,
                                "diameter_size_log_delta": diameter_size_log_delta,
                            }
                        )
                    counters["stitch_merges"] += 1
                    counters["accepted_abs_dz_sum"] += abs_dz
                    counters["accepted_abs_dz_n"] += 1
                    if large_feature_ok:
                        counters["large_feature_merges"] += 1
                        if accepted_by == "large_feature":
                            counters["large_relaxed_merges"] += 1
                        else:
                            counters["large_normal_merges"] += 1
        if pass_merges == 0:
            break

    root_ids = np.asarray([uf.find(int(t)) for t in track_ids], dtype=np.int32)
    roots_sorted = sorted(np.unique(root_ids), key=lambda t: (df.loc[np.where(root_ids == t)[0][0], "frame"], t))
    remap = {old: i + 1 for i, old in enumerate(roots_sorted)}
    stitched = np.asarray([remap[int(t)] for t in root_ids], dtype=np.int32)
    n_dz = int(counters["accepted_abs_dz_n"])
    return stitched, {
        "stitch_merges": int(counters["stitch_merges"]),
        "large_candidates": int(counters["large_candidates"]),
        "large_feature_ok_candidates": int(counters["large_feature_ok_candidates"]),
        "large_feature_merges": int(counters["large_feature_merges"]),
        "large_relaxed_merges": int(counters["large_relaxed_merges"]),
        "large_normal_merges": int(counters["large_normal_merges"]),
        "accepted_abs_dz_lap_mean": float(counters["accepted_abs_dz_sum"] / n_dz) if n_dz else None,
        "z_tiebreak_candidates": int(counters["z_tiebreak_candidates"]),
        "z_tiebreak_penalty_sum": float(counters["z_tiebreak_penalty_sum"]),
        "diameter_tiebreak_candidates": int(counters["diameter_tiebreak_candidates"]),
        "diameter_tiebreak_penalty_sum": float(counters["diameter_tiebreak_penalty_sum"]),
        "stitch_examples": merges,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Depth-aware 3D variant of the hologram online linker.")
    p.add_argument("--input", required=True, help="tracks_with_diam_depth.csv or observations.csv with per-detection depth.")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--linking-mode",
        choices=tuple(LINKING_MODE_PRESETS),
        default="fixed-lag",
        help="Association preset. fixed-lag is the default; legacy reproduces the previous online linker defaults.",
    )
    p.add_argument("--embeddings", default="", help="Optional row-aligned ROIFeat embedding .npy.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--start-frame", type=int, default=None)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--xy-source", choices=("seg", "bbox", "x_lap"), default="seg")
    p.add_argument("--z-lap-scale", type=float, default=10.0, help="slice-to-XY-unit scale when z_lap is absent.")
    p.add_argument(
        "--z-mode",
        choices=("distance", "tiebreak"),
        default="distance",
        help="distance always includes z in motion distance; tiebreak uses z only for near-tied XY candidates.",
    )
    p.add_argument("--z-weight", type=float, default=0.02, help="Extra reliability weight applied to z_lap in motion costs.")
    p.add_argument("--z-tiebreak-margin", type=float, default=0.03, help="Base-cost margin defining near-tied XY candidates.")
    p.add_argument("--z-tiebreak-penalty", type=float, default=0.05, help="Maximum scale of the z tie-break cost.")
    p.add_argument("--z-tiebreak-cap", type=float, default=1.0, help="Cap applied before z tie-break penalty scaling.")
    p.add_argument("--z-velocity-alpha", type=float, default=0.08)
    p.add_argument("--z-start-velocity-alpha", type=float, default=0.20)
    p.add_argument(
        "--history-predictor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the mode preset for robust recent-history constant-velocity prediction.",
    )
    p.add_argument("--history-window", type=int, default=6)
    p.add_argument("--history-blend", type=float, default=0.85)
    p.add_argument("--history-z-blend", type=float, default=0.70)
    p.add_argument(
        "--gap-history-min-points",
        type=int,
        default=4,
        help="Minimum history points for the robust predictor on dt>1 recovery only.",
    )
    p.add_argument(
        "--gap-history-window",
        type=int,
        default=0,
        help="Robust-history window on dt>1 recovery only; non-positive preserves --history-window.",
    )
    p.add_argument(
        "--gap-history-blend",
        type=float,
        default=-1.0,
        help="XY robust-prediction blend on dt>1 only; negative preserves --history-blend.",
    )
    p.add_argument(
        "--gap-history-z-blend",
        type=float,
        default=-1.0,
        help="Z robust-prediction blend on dt>1 only; negative preserves --history-z-blend.",
    )
    p.add_argument(
        "--gap-z-tiebreak-scale",
        type=float,
        default=-1.0,
        help="Scale the Z tie-break penalty on dt>1 rows only; negative preserves the global penalty.",
    )
    p.add_argument("--online-max-gap", type=int, default=None, help="Override the mode preset maximum online gap.")
    p.add_argument("--stitch-max-gap", type=int, default=12)
    p.add_argument("--base-gate", type=float, default=8.0)
    p.add_argument("--gap-gate", type=float, default=None, help="Override the mode preset gap gate growth.")
    p.add_argument("--startup-gate", type=float, default=24.0)
    p.add_argument("--online-max-cost", type=float, default=0.90)
    p.add_argument(
        "--gap-max-cost",
        type=float,
        default=-1.0,
        help="Max cost only for online dt>1 recovery; negative preserves --online-max-cost.",
    )
    p.add_argument("--gap-min-hits", type=int, default=1, help="Minimum prior detections required for dt>1 recovery.")
    p.add_argument(
        "--gap-assignment-margin",
        type=float,
        default=0.0,
        help="For dt>1 only, require this cost margin over every row/column alternative.",
    )
    p.add_argument(
        "--gap-diameter-size-weight",
        type=float,
        default=0.0,
        help="Additional diameter consistency cost applied only to dt>1 recovery.",
    )
    p.add_argument("--stitch-max-cost", type=float, default=0.62)
    p.add_argument("--motion-weight", type=float, default=0.95)
    p.add_argument("--feature-weight", type=float, default=0.24)
    p.add_argument("--size-weight", type=float, default=0.18)
    p.add_argument(
        "--size-source",
        choices=("bbox_area", "diameter"),
        default="bbox_area",
        help="Size signal used for online/stitch association. bbox_area preserves the previous behavior.",
    )
    p.add_argument(
        "--gate-size-source",
        choices=("", "bbox_area", "diameter"),
        default="",
        help="Size signal used only for motion gates. Empty means use --size-source.",
    )
    p.add_argument("--size-cost-scale", type=float, default=1.25, help="Log-size delta divisor for size association cost.")
    p.add_argument(
        "--diameter-size-weight",
        type=float,
        default=0.0,
        help="Additional diameter-based log-size cost added on top of --size-source.",
    )
    p.add_argument(
        "--diameter-size-cost-scale",
        type=float,
        default=1.25,
        help="Log-diameter pseudo-area delta divisor for --diameter-size-weight.",
    )
    p.add_argument("--diameter-tiebreak-margin", type=float, default=0.0)
    p.add_argument("--diameter-tiebreak-penalty", type=float, default=0.0)
    p.add_argument("--diameter-tiebreak-cap", type=float, default=1.0)
    p.add_argument("--age-bonus", type=float, default=0.05)
    p.add_argument("--large-stitch-sqrt-area", type=float, default=9.0)
    p.add_argument("--large-stitch-gate-scale", type=float, default=1.65)
    p.add_argument("--large-stitch-max-cost", type=float, default=1.02)
    p.add_argument("--large-stitch-feature-dist", type=float, default=0.58)
    p.add_argument("--large-stitch-size-log-max", type=float, default=0.85)
    p.add_argument("--large-stitch-vel-cost-max", type=float, default=1.35)
    p.add_argument("--max-merge-examples", type=int, default=50)
    args = p.parse_args()
    preset = LINKING_MODE_PRESETS[args.linking_mode]
    for name, value in preset.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    input_path = resolve(args.input)
    out_dir = resolve(args.out_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if args.start_frame is not None:
        df = df[df["frame"] >= args.start_frame].copy()
    if args.end_frame is not None:
        df = df[df["frame"] <= args.end_frame].copy()
    if "row_id" in df.columns:
        order_col = "row_id"
    elif "obs_id" in df.columns:
        order_col = "obs_id"
    else:
        df["_input_order"] = np.arange(len(df), dtype=np.int64)
        order_col = "_input_order"
    df = df.sort_values(["frame", order_col]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No detections left after filtering.")
    df = prepare_link_columns(df, args.xy_source, args.z_lap_scale)
    gate_size_source = args.gate_size_source or args.size_source
    if "obs_id" not in df.columns:
        df.insert(0, "obs_id", np.arange(len(df), dtype=np.int64))

    emb_path = resolve(args.embeddings) if args.embeddings else None
    emb = load_embeddings(emb_path, df)

    dx, dy, flow_diag = estimate_global_flow(df)
    initial_ids, online_diag = online_link_3d(
        df,
        emb,
        (dx, dy),
        z_mode=args.z_mode,
        z_weight=args.z_weight,
        z_tiebreak_margin=args.z_tiebreak_margin,
        z_tiebreak_penalty=args.z_tiebreak_penalty,
        z_tiebreak_cap=args.z_tiebreak_cap,
        max_gap=args.online_max_gap,
        base_gate=args.base_gate,
        gap_gate=args.gap_gate,
        startup_gate=args.startup_gate,
        max_cost=args.online_max_cost,
        gap_max_cost=args.gap_max_cost,
        gap_min_hits=args.gap_min_hits,
        gap_assignment_margin=args.gap_assignment_margin,
        gap_diameter_size_weight=args.gap_diameter_size_weight,
        motion_weight=args.motion_weight,
        feature_weight=0.0 if emb is None else args.feature_weight,
        size_weight=args.size_weight,
        size_source=args.size_source,
        gate_size_source=gate_size_source,
        size_cost_scale=args.size_cost_scale,
        diameter_size_weight=args.diameter_size_weight,
        diameter_size_cost_scale=args.diameter_size_cost_scale,
        diameter_tiebreak_margin=args.diameter_tiebreak_margin,
        diameter_tiebreak_penalty=args.diameter_tiebreak_penalty,
        diameter_tiebreak_cap=args.diameter_tiebreak_cap,
        age_bonus=args.age_bonus,
        z_velocity_alpha=args.z_velocity_alpha,
        z_start_velocity_alpha=args.z_start_velocity_alpha,
        history_predictor=args.history_predictor,
        history_window=args.history_window,
        history_blend=args.history_blend,
        history_z_blend=args.history_z_blend,
        gap_history_min_points=args.gap_history_min_points,
        gap_history_window=args.gap_history_window,
        gap_history_blend=args.gap_history_blend,
        gap_history_z_blend=args.gap_history_z_blend,
        gap_z_tiebreak_scale=args.gap_z_tiebreak_scale,
    )
    stitched_ids, stitch_diag = stitch_tracklets_3d(
        df,
        initial_ids,
        emb,
        (dx, dy),
        z_mode=args.z_mode,
        z_weight=args.z_weight,
        z_tiebreak_margin=args.z_tiebreak_margin,
        z_tiebreak_penalty=args.z_tiebreak_penalty,
        z_tiebreak_cap=args.z_tiebreak_cap,
        max_gap=args.stitch_max_gap,
        max_cost=args.stitch_max_cost,
        base_gate=args.base_gate,
        gap_gate=args.gap_gate,
        startup_gate=args.startup_gate,
        min_len_for_feature=2,
        large_sqrt_area=args.large_stitch_sqrt_area,
        large_gate_scale=args.large_stitch_gate_scale,
        large_max_cost=args.large_stitch_max_cost,
        large_feature_dist=args.large_stitch_feature_dist,
        large_size_log_max=args.large_stitch_size_log_max,
        large_vel_cost_max=args.large_stitch_vel_cost_max,
        size_source=args.size_source,
        gate_size_source=gate_size_source,
        size_cost_scale=args.size_cost_scale,
        diameter_size_weight=args.diameter_size_weight,
        diameter_size_cost_scale=args.diameter_size_cost_scale,
        diameter_tiebreak_margin=args.diameter_tiebreak_margin,
        diameter_tiebreak_penalty=args.diameter_tiebreak_penalty,
        diameter_tiebreak_cap=args.diameter_tiebreak_cap,
        max_examples=args.max_merge_examples,
    )
    output_diag = write_track_outputs(df, stitched_ids, out_dir, args)

    diag = {
        "input": str(input_path),
        "out_dir": str(out_dir),
        "linking_mode": args.linking_mode,
        "rows": int(len(df)),
        "frames": int(df["frame"].nunique()),
        "z_weight": float(args.z_weight),
        "z_mode": args.z_mode,
        "z_tiebreak_margin": float(args.z_tiebreak_margin),
        "z_tiebreak_penalty": float(args.z_tiebreak_penalty),
        "z_tiebreak_cap": float(args.z_tiebreak_cap),
        "z_lap_scale": float(args.z_lap_scale),
        "xy_source": args.xy_source,
        "size_source": args.size_source,
        "gate_size_source": gate_size_source,
        "size_cost_scale": float(args.size_cost_scale),
        "diameter_size_weight": float(args.diameter_size_weight),
        "diameter_size_cost_scale": float(args.diameter_size_cost_scale),
        "diameter_tiebreak_margin": float(args.diameter_tiebreak_margin),
        "diameter_tiebreak_penalty": float(args.diameter_tiebreak_penalty),
        "diameter_tiebreak_cap": float(args.diameter_tiebreak_cap),
        "gap_max_cost": float(args.gap_max_cost),
        "gap_min_hits": int(args.gap_min_hits),
        "gap_assignment_margin": float(args.gap_assignment_margin),
        "gap_diameter_size_weight": float(args.gap_diameter_size_weight),
        "gap_history_min_points": int(args.gap_history_min_points),
        "gap_history_window": int(args.gap_history_window),
        "gap_history_blend": float(args.gap_history_blend),
        "gap_history_z_blend": float(args.gap_history_z_blend),
        "gap_z_tiebreak_scale": float(args.gap_z_tiebreak_scale),
        "embeddings": str(emb_path) if emb_path else "",
        "embedding_shape": list(emb.shape) if emb is not None else None,
        "global_flow": flow_diag,
        "online": online_diag,
        "stitch": stitch_diag,
        "output": output_diag,
        "args": jsonable(vars(args)),
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    (out_dir / "depth3d_link_diagnostics.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(diag, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
