"""Conservative detection of persistent depth discontinuities in linked tracks.

This is an association check, not a smoother: measurements are never changed.
Two local robust lines must disagree in position, at two window sizes, and
outperform continuous motion models. Short excursions back to the other branch
are left to downstream outlier handling. Scores are diagnostics, not p-values.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

MODES = ("off", "audit", "split")
CUT_COLUMNS = [
    "parent_track_id",
    "new_track_id",
    "left_row",
    "right_row",
    "frame_before",
    "frame_after",
    "depth_jump",
    "noise_scale",
    "jump_score",
    "cost_gain",
    "support_fraction",
]


@dataclass(frozen=True)
class DepthRepairConfig:
    """Window lengths count observations; fits always use actual frame times.

    Bounds, when known, use x_lap/y_lap/z_lap units (pixel equivalents in the
    CLI). They are not inferred from observed extrema. Each boundary margin is
    a fraction of its supplied volume extent.
    """

    window: int = 12
    min_segment: int = 24
    z_score: float = 8.0
    min_gain: float = 0.15
    x_bounds: tuple[float, float] | None = None
    y_bounds: tuple[float, float] | None = None
    z_bounds: tuple[float, float] | None = None
    boundary_margin: float = 0.02

    def __post_init__(self) -> None:
        for name in ("window", "min_segment"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"depth repair {name} must be an integer")
        if self.window < 4 or self.min_segment < 2 * self.window:
            raise ValueError("depth repair requires window >= 4 and min_segment >= 2 * window")
        if not np.isfinite(self.z_score) or self.z_score <= 0:
            raise ValueError("depth repair z_score must be finite and positive")
        if not np.isfinite(self.min_gain) or not 0 < self.min_gain < 1:
            raise ValueError("depth repair min_gain must be between 0 and 1")
        if not np.isfinite(self.boundary_margin) or not 0 <= self.boundary_margin < 0.5:
            raise ValueError("depth repair boundary_margin must be in [0, 0.5)")
        for name in ("x_bounds", "y_bounds", "z_bounds"):
            bounds = getattr(self, name)
            if bounds is not None and (len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]):
                raise ValueError(f"depth repair {name} must contain two finite increasing values")
            if bounds is not None:
                object.__setattr__(self, name, tuple(float(v) for v in bounds))
        for name in ("window", "min_segment"):
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in ("z_score", "min_gain", "boundary_margin"):
            object.__setattr__(self, name, float(getattr(self, name)))


@dataclass
class DepthRepairResult:
    track_ids: np.ndarray
    cuts: pd.DataFrame
    diagnostics: dict


@dataclass
class _Evidence:
    index: int
    jump: float
    scale: float
    score: float
    gain: float
    support: float
    left: np.ndarray
    right: np.ndarray
    center: float


def _mad(values: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _noise(frames: np.ndarray, depth: np.ndarray) -> float:
    """MAD of interpolation residuals, normalized for unequal frame intervals."""
    delta = np.diff(frames)
    weight = delta[:-1] / (delta[:-1] + delta[1:])
    residual = depth[1:-1] - (1 - weight) * depth[:-2] - weight * depth[2:]
    return _mad(residual / np.sqrt(1 + (1 - weight) ** 2 + weight**2))


def _line(times: np.ndarray, depth: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(len(times), 1)
    slope = np.median((depth[j] - depth[i]) / (times[j] - times[i]))
    return np.array([np.median(depth - slope * times), slope])


def _huber_fit(design: np.ndarray, depth: np.ndarray, scale: float) -> np.ndarray:
    beta = np.linalg.lstsq(design, depth, rcond=None)[0]
    for _ in range(10):
        residual = np.abs(depth - design @ beta)
        weights = np.ones_like(residual)
        large = residual > 1.345 * scale
        weights[large] = 1.345 * scale / residual[large]
        root = np.sqrt(weights)
        updated = np.linalg.lstsq(design * root[:, None], depth * root, rcond=None)[0]
        change = np.max(np.abs(design @ (updated - beta)))
        beta = updated
        if change < scale * 1e-4:
            break
    return beta


def _cost(residual: np.ndarray, scale: float) -> np.ndarray:
    # Clip BEFORE squaring: even arbitrarily large finite outliers have cost 1.
    return np.minimum(np.abs(residual) / (3 * scale), 1.0) ** 2


def _evidence(
    frames: np.ndarray, depth: np.ndarray, index: int, window: int, floor: float, config: DepthRepairConfig
) -> _Evidence | None:
    if min(index, len(frames) - index) < window:
        return None
    gap = frames[index] - frames[index - 1]
    center = frames[index - 1] + gap / 2
    times = frames[index - window : index + window] - center
    values = depth[index - window : index + window]
    left_span = np.ptp(times[:window]) + np.median(np.diff(times[:window]))
    right_span = np.ptp(times[window:]) + np.median(np.diff(times[window:]))
    if gap > min(left_span, right_span):
        return None  # Too much unobserved time to establish a discontinuity.
    left = _line(times[:window], values[:window])
    right = _line(times[window:], values[window:])
    prediction = np.r_[left[0] + left[1] * times[:window], right[0] + right[1] * times[window:]]
    residual = values - prediction
    scale = max(floor, _mad(residual[:window]), _mad(residual[window:]))
    jump = right[0] - left[0]
    edge_jumps = jump + (right[1] - left[1]) * np.array([-gap / 2, gap / 2])
    if np.sign(edge_jumps[0]) != np.sign(edge_jumps[1]):
        return None  # The branches can meet inside the missing-frame interval.
    score = float(np.min(np.abs(edge_jumps)) / scale)
    if score < config.z_score:
        return None
    support = float(
        min(
            np.mean(np.abs(residual[:window]) <= 3 * scale),
            np.mean(np.abs(residual[window:]) <= 3 * scale),
        )
    )
    if support < 0.75:
        return None
    normalized = times / np.max(np.abs(times))
    # A continuous hinge allows a velocity change; a cubic allows curvature.
    designs = (
        np.column_stack([np.ones(len(times)), normalized, np.maximum(normalized, 0)]),
        np.vander(normalized, 4, increasing=True),
    )
    null_cost = min(float(np.mean(_cost(values - a @ _huber_fit(a, values, scale), scale))) for a in designs)
    gain = null_cost - float(np.mean(_cost(residual, scale)))
    if gain < config.min_gain:
        return None
    return _Evidence(index, float(jump), scale, score, gain, support, left, right, center)


def _refine(frames: np.ndarray, depth: np.ndarray, evidence: _Evidence, minimum: int) -> int:
    # A five-point median may move a transition by up to two observations.
    index = evidence.index
    candidates = np.arange(max(minimum, index - 2), min(len(depth) - minimum, index + 2) + 1)
    lo, hi = int(candidates[0]), int(candidates[-1])
    times = frames[lo:hi] - evidence.center
    left = evidence.left[0] + evidence.left[1] * times
    right = evidence.right[0] + evidence.right[1] * times
    delta = _cost(depth[lo:hi] - left, evidence.scale) - _cost(depth[lo:hi] - right, evidence.scale)
    costs = np.r_[0, np.cumsum(delta)]
    tied = np.flatnonzero(costs <= costs.min() + 1e-10)
    best = min(tied, key=lambda j: (abs(candidates[j] - index), candidates[j]))
    return int(candidates[best])


def _returns(frames: np.ndarray, depth: np.ndarray, evidence: _Evidence, window: int) -> bool:
    index = evidence.index
    block = max(6, window // 2)
    radius = 4 * window
    ranges = (
        (index + block, min(len(frames), index + radius), evidence.left, evidence.right),
        (max(0, index - radius), index - block, evidence.right, evidence.left),
    )
    for lo, hi, other, own in ranges:
        if hi - lo < block:
            continue
        times = frames[lo:hi] - evidence.center
        other_error = np.abs(depth[lo:hi] - other[0] - other[1] * times)
        own_error = np.abs(depth[lo:hi] - own[0] - own[1] * times)
        tolerance = max(3 * evidence.scale, 0.15 * abs(evidence.jump))
        matches = (other_error < tolerance) & (other_error < 0.5 * own_error)
        if np.any(np.convolve(matches.astype(float), np.ones(block), mode="valid") >= 0.8 * block):
            return True
    return False


def _near_boundary(coordinates: dict[str, np.ndarray], index: int, config: DepthRepairConfig) -> bool:
    for axis, values in coordinates.items():
        low, high = getattr(config, f"{axis}_bounds")
        margin = config.boundary_margin * (high - low)
        # Use robust positions on both sides; one coordinate outlier is not a boundary.
        for part in (values[index - config.window : index], values[index : index + config.window]):
            position = np.median(part)
            if position <= low + margin or position >= high - margin:
                return True
    return False


def _detect(
    frames: np.ndarray,
    depth: np.ndarray,
    floor: float,
    coordinates: dict[str, np.ndarray],
    config: DepthRepairConfig,
    counts: dict[str, int],
) -> list[_Evidence]:
    median = median_filter(depth, size=5, mode="nearest")
    delta = np.diff(frames)
    velocity = np.diff(median) / delta
    baseline = median_filter(velocity, size=2 * config.window + 1, mode="nearest")
    candidates = np.flatnonzero(np.abs(np.diff(median) - baseline * delta) > 4 * floor) + 1
    candidates = candidates[(candidates >= config.min_segment) & (candidates <= len(frames) - config.min_segment)]
    counts["candidate_edges"] += len(candidates)
    results = []
    for index in candidates:
        small = _evidence(frames, depth, int(index), config.window, floor, config)
        if small is None:
            continue
        refined = _refine(frames, depth, small, config.min_segment)
        if refined != index:
            small = _evidence(frames, depth, refined, config.window, floor, config)
            if small is None:
                continue
        large = _evidence(frames, depth, refined, 2 * config.window, floor, config)
        if large is None or np.sign(small.jump) != np.sign(large.jump):
            continue
        if _returns(frames, depth, large, config.window):
            counts["return_vetoes"] += 1
            continue
        if _near_boundary(coordinates, refined, config):
            counts["boundary_vetoes"] += 1
            continue
        results.append(small)
    accepted: list[_Evidence] = []
    for candidate in sorted(results, key=lambda e: (-e.gain, -e.score, e.index)):
        if all(abs(candidate.index - other.index) >= config.min_segment for other in accepted):
            accepted.append(candidate)
    return sorted(accepted, key=lambda e: e.index)


def repair_depth_links(
    frame: pd.DataFrame, track_ids: np.ndarray, *, config: DepthRepairConfig | None = None
) -> DepthRepairResult:
    """Propose splits after stitching; output IDs align with the input row order.

    Requires finite ``frame`` and ``z_lap`` columns and one observation per
    (track ID, frame). The first segment keeps its ID; later segments receive
    deterministic unused IDs. Bounds additionally require the matching *_lap
    columns. The input table and IDs are never mutated.
    """
    started = time.perf_counter()
    config = config or DepthRepairConfig()
    ids = np.asarray(track_ids)
    if ids.ndim != 1 or len(ids) != len(frame) or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("track_ids must be row-aligned non-negative integers")
    if len(ids) and ids.max() > np.iinfo(np.int32).max:
        raise ValueError("track_ids exceed the linker's int32 output range")
    if not {"frame", "z_lap"}.issubset(frame):
        raise ValueError("depth repair requires frame and z_lap columns")
    times = frame["frame"].to_numpy(np.float64)
    depth = frame["z_lap"].to_numpy(np.float64)
    if not np.isfinite(times).all() or np.any(times < 0) or np.any(times >= 2**53) or np.any(times != np.floor(times)):
        raise ValueError("depth repair frame values must be finite non-negative integers < 2**53")
    if not np.isfinite(depth).all():
        raise ValueError("depth repair requires finite z_lap values")
    coordinates = {}
    for axis in ("x", "y", "z"):
        if getattr(config, f"{axis}_bounds") is not None:
            name = f"{axis}_lap"
            if name not in frame:
                raise ValueError(f"depth repair {axis}_bounds requires {name}")
            values = frame[name].to_numpy(np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"depth repair requires finite {name} values for volume bounds")
            coordinates[axis] = values
    order = np.lexsort((times, ids))
    ordered_ids = ids[order]
    offsets = np.r_[0, np.flatnonzero(np.diff(ordered_ids) != 0) + 1, len(ids)]
    groups = [order[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True) if b > a]
    estimates = {}
    for number, rows in enumerate(groups):
        t = times[rows]
        if np.any(np.diff(t) <= 0):
            raise ValueError("depth repair requires unique frames within every track")
        if len(rows) >= config.min_segment:
            estimates[number] = _noise(t, depth[rows])
    pooled = float(np.median(list(estimates.values()))) if estimates else 0.0
    # Per-track weighting prevents one long (possibly bad) track setting the scale.
    counts = dict(candidate_edges=0, return_vetoes=0, boundary_vetoes=0, examined_tracks=0)
    repaired = ids.astype(np.int32, copy=True)
    next_id = int(ids.max()) + 1 if len(ids) else 0
    cuts = []
    for number, rows in enumerate(groups):
        if len(rows) < 2 * config.min_segment:
            continue
        counts["examined_tracks"] += 1
        t, z = times[rows], depth[rows]
        # Centering protects regressions against coordinate origins and units.
        z = z - np.median(z)
        numerical_floor = np.finfo(float).eps * max(float(np.max(np.abs(z))), np.finfo(float).tiny) * 128
        floor = max(pooled, estimates[number], numerical_floor, np.finfo(float).tiny)
        found = _detect(t, z, floor, {axis: a[rows] for axis, a in coordinates.items()}, config, counts)
        for evidence in found:
            if next_id > np.iinfo(np.int32).max:
                raise ValueError("depth repair exhausted the linker's int32 track ID range")
            index = evidence.index
            repaired[rows[index:]] = next_id
            cuts.append(
                {
                    "parent_track_id": int(ids[rows[0]]),
                    "new_track_id": next_id,
                    "left_row": int(rows[index - 1]),
                    "right_row": int(rows[index]),
                    "frame_before": int(t[index - 1]),
                    "frame_after": int(t[index]),
                    "depth_jump": evidence.jump,
                    "noise_scale": evidence.scale,
                    "jump_score": evidence.score,
                    "cost_gain": evidence.gain,
                    "support_fraction": evidence.support,
                }
            )
            next_id += 1
    table = pd.DataFrame(cuts, columns=CUT_COLUMNS)
    if "row_id" in frame:
        source_ids = frame["row_id"].to_numpy()
        table["row_id_before"] = source_ids[table.left_row.to_numpy(np.int64)]
        table["row_id_after"] = source_ids[table.right_row.to_numpy(np.int64)]
    diagnostics = {
        "algorithm": "persistent-local-depth-jump-v1",
        "config": asdict(config),
        "depth_column": "z_lap",
        "pooled_noise_scale": pooled,
        "noise_tracks": len(estimates),
        "input_tracks": len(groups),
        "cut_count": len(cuts),
        "changed_tracks": int(table["parent_track_id"].nunique()),
        **{key: int(value) for key, value in counts.items()},
        "elapsed_sec": time.perf_counter() - started,
    }
    return DepthRepairResult(repaired, table, diagnostics)


def add_depth_repair_arguments(parser: argparse.ArgumentParser, *, include_mode: bool = True) -> None:
    group = parser.add_argument_group("persistent depth discontinuities")
    if include_mode:
        group.add_argument(
            "--depth-repair",
            choices=MODES,
            default="split",
            help="Split suspected false links after stitching (default: split); audit records proposals, off disables.",
        )
    group.add_argument(
        "--depth-repair-window", type=int, default=12, help="Observations per side of the smaller local fit."
    )
    group.add_argument(
        "--depth-repair-min-segment", type=int, default=24, help="Minimum observations in every retained segment."
    )
    group.add_argument(
        "--depth-repair-z-score", type=float, default=8.0, help="Minimum jump/noise ratio; not a statistical z-test."
    )
    group.add_argument(
        "--depth-repair-min-gain",
        type=float,
        default=0.15,
        help="Minimum mean bounded-cost improvement over continuous motion.",
    )
    for axis in ("x", "y", "z"):
        group.add_argument(
            f"--depth-repair-{axis}-bounds",
            type=float,
            nargs=2,
            default=None,
            metavar=("LOW", "HIGH"),
            help=f"Known {axis}_lap volume bounds in pixel-equivalent units; never inferred.",
        )
    group.add_argument(
        "--depth-repair-boundary-margin",
        type=float,
        default=0.02,
        help="Excluded fraction at each supplied volume boundary.",
    )


def config_from_args(args: argparse.Namespace) -> DepthRepairConfig:
    return DepthRepairConfig(
        window=args.depth_repair_window,
        min_segment=args.depth_repair_min_segment,
        z_score=args.depth_repair_z_score,
        min_gain=args.depth_repair_min_gain,
        x_bounds=args.depth_repair_x_bounds,
        y_bounds=args.depth_repair_y_bounds,
        z_bounds=args.depth_repair_z_bounds,
        boundary_margin=args.depth_repair_boundary_margin,
    )
