"""Small numerical helpers retained from the 260712 linker."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-8)


def estimate_global_flow(
    df: pd.DataFrame,
    sample_pairs: int = 80,
    min_conf: float = 0.40,
    radius: float = 12.0,
) -> tuple[float, float, dict[str, Any]]:
    frames = sorted(df["frame"].unique())
    grid = [(dx, dy) for dx in np.linspace(-2.5, 2.5, 21) for dy in np.linspace(1.0, 8.0, 29)]
    shifts: list[tuple[float, float]] = []

    for f0, f1 in zip(frames[:sample_pairs], frames[1 : sample_pairs + 1], strict=False):
        a = df[(df.frame == f0) & (df.conf >= min_conf)][["xc", "yc"]].to_numpy(np.float32)
        b = df[(df.frame == f1) & (df.conf >= min_conf)][["xc", "yc"]].to_numpy(np.float32)
        if len(a) < 50 or len(b) < 50:
            continue
        sample = a[:: max(1, len(a) // 260)]
        tree = cKDTree(b)
        best: tuple[float, float, float] | None = None
        for dx, dy in grid:
            distances, _indices = tree.query(
                sample + np.array([dx, dy], dtype=np.float32),
                k=1,
                distance_upper_bound=radius,
            )
            finite = distances[np.isfinite(distances)]
            if len(finite) < len(sample) * 0.65:
                continue
            score = float(np.median(finite) + 0.20 * np.percentile(finite, 90))
            if best is None or score < best[0]:
                best = (score, float(dx), float(dy))
        if best is not None:
            shifts.append((best[1], best[2]))

    if not shifts:
        return -0.75, 4.0, {"estimated": False, "shifts": []}
    values = np.asarray(shifts, dtype=np.float32)
    dx, dy = np.median(values, axis=0).tolist()
    return float(dx), float(dy), {
        "estimated": True,
        "sample_pairs": len(shifts),
        "median_dx": float(dx),
        "median_dy": float(dy),
        "mean_dx": float(values[:, 0].mean()),
        "mean_dy": float(values[:, 1].mean()),
        "q_dx": np.quantile(values[:, 0], [0, 0.25, 0.5, 0.75, 1]).tolist(),
        "q_dy": np.quantile(values[:, 1], [0, 0.25, 0.5, 0.75, 1]).tolist(),
    }


def track_gate(dt: int, area_sqrt: float, speed: float, base_gate: float, gap_gate: float) -> float:
    return float(min(42.0, max(7.0, base_gate + gap_gate * max(0, dt - 1) + 0.22 * speed + 0.18 * area_sqrt)))


class UnionFind:
    def __init__(self, values: np.ndarray) -> None:
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        root = min(left_root, right_root)
        other = max(left_root, right_root)
        self.parent[other] = root
        return True


def finite_positive(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    return np.isfinite(numeric) & (numeric > 0.0)


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
