from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from holod3_linker import DepthRepairConfig, repair_depth_links


def track(depth, frames=None, track_id=7):
    n = len(depth)
    return pd.DataFrame(
        {
            "frame": np.arange(n) if frames is None else frames,
            "z_lap": depth,
            "x_lap": np.full(n, 500.0),
            "y_lap": np.linspace(200, 800, n),
            "row_id": np.arange(n) + 1000,
            "diameter_px": np.full(n, 10.0),
        }
    ), np.full(n, track_id, dtype=np.int32)


def noisy_line(n=240, seed=19):
    t = np.arange(n)
    return 100 + 0.1 * t + np.random.default_rng(seed).normal(0, 0.5, n)


def test_persistent_jump_splits_without_changing_measurements():
    depth = noisy_line()
    depth[120:] += 80
    frame, ids = track(depth)
    original = frame.copy(deep=True)
    result = repair_depth_links(frame, ids)
    assert result.cuts.frame_after.tolist() == [120]
    assert result.cuts.left_row.tolist() == [119]
    assert result.cuts.right_row.tolist() == [120]
    assert result.cuts.row_id_before.tolist() == [1119]
    assert result.cuts.row_id_after.tolist() == [1120]
    np.testing.assert_array_equal(result.track_ids[:120], 7)
    np.testing.assert_array_equal(result.track_ids[120:], 8)
    np.testing.assert_array_equal(ids, 7)
    pd.testing.assert_frame_equal(frame, original)
    json.dumps(result.diagnostics, allow_nan=False)


@pytest.mark.parametrize(
    "kind",
    [
        "linear",
        "fast",
        "quadratic",
        "helix",
        "velocity_change",
        "smooth_transition",
        "isolated_outliers",
        "burst",
        "endpoints",
        "heteroscedastic",
        "constant",
    ],
)
def test_preserves_continuous_motion_and_outliers(kind):
    t = np.arange(240)
    depth = noisy_line()
    if kind == "fast":
        depth += 7 * t
    elif kind == "quadratic":
        depth += 0.008 * (t - 120) ** 2
    elif kind == "helix":
        depth += 80 * np.sin(t / 18)
    elif kind == "velocity_change":
        depth += 3 * np.maximum(t - 120, 0)
    elif kind == "smooth_transition":
        depth += 100 / (1 + np.exp(-(t - 120) / 5))
    elif kind == "isolated_outliers":
        depth[[30, 65, 100, 131, 180, 211]] += 1e9
    elif kind == "burst":
        depth[90:128] += 500
    elif kind == "endpoints":
        depth[:20] += 800
        depth[-18:] -= 900
    elif kind == "heteroscedastic":
        depth += np.random.default_rng(31).normal(0, np.linspace(1, 20, len(t)))
    elif kind == "constant":
        depth[:] = 12
    frame, ids = track(depth)
    result = repair_depth_links(frame, ids)
    np.testing.assert_array_equal(result.track_ids, ids)
    assert result.cuts.empty


def test_multiple_jumps_and_minimum_segment_length():
    depth = noisy_line()
    depth[80:] += 80
    depth[160:] += 100
    frame, ids = track(depth)
    result = repair_depth_links(frame, ids)
    assert result.cuts.frame_after.tolist() == [80, 160]
    assert result.cuts.new_track_id.tolist() == [8, 9]
    assert np.unique(result.track_ids, return_counts=True)[1].tolist() == [80, 80, 80]


def test_frame_gaps_use_time_and_abstain_when_unobserved_interval_is_too_long():
    t = np.arange(240)
    t[120:] += 11  # The linker's default recovery permits a 12-frame interval.
    depth = noisy_line() + np.where(np.arange(240) >= 120, 100, 0)
    frame, ids = track(depth, t)
    assert repair_depth_links(frame, ids).cuts.frame_after.tolist() == [131]
    t[120:] += 200
    frame["frame"] = t
    assert repair_depth_links(frame, ids).cuts.empty


def test_reordering_and_unit_changes_preserve_proposed_edges():
    depth = noisy_line()
    depth[120:] -= 80
    frame, ids = track(depth)
    reference = repair_depth_links(frame, ids)
    frame["z_lap"] = frame.z_lap * 100 + 20000
    frame["frame"] += 10**9
    shuffled = frame.sample(frac=1, random_state=3)
    result = repair_depth_links(shuffled, ids)
    assert result.cuts.frame_after.tolist() == [10**9 + 120]
    restored = pd.Series(result.track_ids, index=shuffled.index).sort_index().to_numpy()
    np.testing.assert_array_equal(restored, reference.track_ids)
    assert result.cuts.depth_jump.iloc[0] == pytest.approx(reference.cuts.depth_jump.iloc[0] * 100)


def test_population_scale_is_robust_to_one_long_noisy_track():
    rng = np.random.default_rng(9)
    tables, labels = [], []
    for i in range(10):
        frame, ids = track(noisy_line(seed=i), track_id=i)
        tables.append(frame)
        labels.append(ids)
    frame, ids = track(rng.normal(100, 1000, 10000), track_id=20)
    tables.append(frame)
    labels.append(ids)
    result = repair_depth_links(pd.concat(tables), np.concatenate(labels))
    assert result.diagnostics["pooled_noise_scale"] < 1
    assert result.cuts.empty


def test_multiple_shuffled_tracks_receive_unused_deterministic_ids():
    tables, labels = [], []
    for tid in (19, 100, 4):
        depth = noisy_line(seed=tid)
        if tid != 100:
            depth[120:] += 80
        frame, ids = track(depth, track_id=tid)
        tables.append(frame)
        labels.append(ids)
    frame = pd.concat(tables, ignore_index=True)
    ids = np.concatenate(labels)
    reference = repair_depth_links(frame, ids)
    assert reference.cuts.parent_track_id.tolist() == [4, 19]
    assert reference.cuts.new_track_id.tolist() == [101, 102]
    assert np.all(reference.track_ids[ids == 100] == 100)
    permutation = np.random.default_rng(32).permutation(len(frame))
    result = repair_depth_links(frame.iloc[permutation], ids[permutation])
    np.testing.assert_array_equal(result.track_ids[np.argsort(permutation)], reference.track_ids)
    assert result.cuts[["parent_track_id", "new_track_id", "frame_after"]].equals(
        reference.cuts[["parent_track_id", "new_track_id", "frame_after"]]
    )


def test_known_volume_boundary_veto_is_explicit():
    depth = noisy_line()
    depth[120:] += 100
    frame, ids = track(depth)
    frame["x_lap"] = 5.0
    assert len(repair_depth_links(frame, ids).cuts) == 1
    result = repair_depth_links(frame, ids, config=DepthRepairConfig(x_bounds=(0, 1024)))
    assert result.cuts.empty
    assert result.diagnostics["boundary_vetoes"] > 0


def test_short_tracks_and_zero_noise_are_supported():
    frame, ids = track(np.ones(20))
    assert repair_depth_links(frame, ids).cuts.empty
    frame, ids = track(np.r_[np.ones(80), np.ones(80) * 2])
    assert repair_depth_links(frame, ids).cuts.frame_after.tolist() == [80]
    empty = frame.iloc[:0]
    result = repair_depth_links(empty, np.array([], dtype=int))
    assert len(result.track_ids) == 0
    assert result.diagnostics["input_tracks"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 3},
        {"window": 12.5},
        {"min_segment": 20},
        {"window": True},
        {"z_score": np.nan},
        {"z_score": 0},
        {"min_gain": 1},
        {"x_bounds": (2, 1)},
        {"z_bounds": (0, np.inf)},
        {"boundary_margin": 0.5},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        DepthRepairConfig(**kwargs)


@pytest.mark.parametrize("invalid", ["duplicate_frame", "nan_depth", "negative_id", "float_id", "fractional_frame"])
def test_invalid_inputs_are_rejected(invalid):
    frame, ids = track(noisy_line())
    if invalid == "duplicate_frame":
        frame.loc[3, "frame"] = 2
    elif invalid == "nan_depth":
        frame.loc[3, "z_lap"] = np.nan
    elif invalid == "negative_id":
        ids[3] = -1
    elif invalid == "float_id":
        ids = ids.astype(float)
    else:
        frame["frame"] = frame.frame.astype(float) + 0.5
    with pytest.raises(ValueError):
        repair_depth_links(frame, ids)


def test_id_range_is_checked_before_export():
    depth = noisy_line()
    depth[120:] += 100
    frame, ids = track(depth, track_id=np.iinfo(np.int32).max)
    with pytest.raises(ValueError, match="exhausted"):
        repair_depth_links(frame, ids)
