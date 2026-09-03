"""Trajectory output contract used by the 260712 linker."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def quantiles(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values)
    if array.size == 0:
        return {"min": 0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    return {
        "min": int(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
    }


def write_track_outputs(
    df: pd.DataFrame,
    track_ids: np.ndarray,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and (out_dir / "trajectory3d").exists():
        shutil.rmtree(out_dir / "trajectory3d")
    trajectory_dir = out_dir / "trajectory3d"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    output = df.copy()
    output["track_id"] = track_ids.astype(np.int32)
    output["laptrack_id"] = output["track_id"]
    tracks_depth = out_dir / "tracks_with_diam_depth.csv"
    tracks_diameter = out_dir / "tracks_with_diameter.csv"
    summary_csv = out_dir / "tracks_summary.csv"
    output.to_csv(tracks_depth, index=False)
    drop_columns = [
        column
        for column in ("slice", "depth_um", "model_score", "depth_method", "model_crop_method")
        if column in output.columns
    ]
    output.drop(columns=drop_columns).to_csv(tracks_diameter, index=False)

    summary_rows: list[dict[str, int]] = []
    trajectory_x_column = "seg_xc" if "seg_xc" in output.columns else "x_lap"
    trajectory_y_column = "seg_yc" if "seg_yc" in output.columns else "y_lap"
    for track_id, group in output.groupby("track_id", sort=True):
        summary_rows.append(
            {
                "track_id": int(track_id),
                "len": int(len(group)),
                "frame_first": int(group["frame"].min()),
                "frame_last": int(group["frame"].max()),
            }
        )
        path = trajectory_dir / f"path_{int(track_id):08d}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["frame", "xc", "yc", "slice", "diameter_px"])
            writer.writeheader()
            for row in group.sort_values("frame").itertuples(index=False):
                writer.writerow(
                    {
                        "frame": int(row.frame),
                        "xc": float(getattr(row, trajectory_x_column)),
                        "yc": float(getattr(row, trajectory_y_column)),
                        "slice": float(row.slice),
                        "diameter_px": float(row.diameter_px),
                    }
                )

    summary = pd.DataFrame(summary_rows).sort_values(["len", "track_id"], ascending=[False, True])
    summary.to_csv(summary_csv, index=False)
    lengths = summary["len"].to_numpy(np.int32)
    row_count = int(len(output))

    coverage: dict[str, int | float] = {}
    for minimum_length in (2, 3, 5, 10):
        points = int(lengths[lengths >= minimum_length].sum()) if len(lengths) else 0
        coverage[f"trajectory_points_ge{minimum_length}"] = points
        coverage[f"trajectory_point_fraction_ge{minimum_length}"] = (
            float(points / row_count) if row_count else 0.0
        )

    return {
        "tracks_with_diam_depth": str(tracks_depth),
        "tracks_with_diameter": str(tracks_diameter),
        "tracks_summary": str(summary_csv),
        "trajectory3d": str(trajectory_dir),
        "track_count": int(len(summary)),
        "row_count": row_count,
        "one_frame_tracks": int(np.sum(lengths == 1)),
        "one_frame_points": int(lengths[lengths == 1].sum()) if len(lengths) else 0,
        **coverage,
        "length_quantiles": quantiles(lengths),
    }
