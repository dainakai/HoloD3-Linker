"""Adapt the stable HoloD3 run contract to the 260712 linker contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from holod3_linker.core import finite_number, finite_positive

REQUIRED_LINKER_COLUMNS = {
    "frame",
    "file",
    "conf",
    "xc",
    "yc",
    "w",
    "h",
    "slice",
    "diameter_px",
}
DEFAULT_PIXEL_PITCH_UM = 10.0


@dataclass(frozen=True)
class HoloD3Source:
    run_dir: Path
    particles_csv: Path
    pipeline_summary_json: Path | None


@dataclass
class PreparedInput:
    source: HoloD3Source
    frame: pd.DataFrame
    pixel_pitch_um: float
    pixel_pitch_source: str
    diameter_rows_updated: int
    z_rows_scaled: int

    @property
    def rows(self) -> int:
        return int(len(self.frame))

    @property
    def frames(self) -> int:
        return int(self.frame["frame"].nunique())

    def write_csv(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(destination, index=False)
        return destination

    def metadata(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.source.run_dir),
            "particles_csv": str(self.source.particles_csv),
            "pipeline_summary_json": (
                str(self.source.pipeline_summary_json) if self.source.pipeline_summary_json is not None else None
            ),
            "rows": self.rows,
            "frames": self.frames,
            "pixel_pitch_um": self.pixel_pitch_um,
            "pixel_pitch_source": self.pixel_pitch_source,
            "diameter_source": "final_diameter_um" if self.diameter_rows_updated else "diameter_px",
            "diameter_rows_updated": self.diameter_rows_updated,
            "z_source": "z_um" if self.z_rows_scaled else "linker_slice_scale",
            "z_rows_scaled": self.z_rows_scaled,
        }


def resolve_holod3_source(path: str | Path) -> HoloD3Source:
    requested = Path(path).expanduser().resolve()
    if requested.is_dir():
        run_dir = requested
        particles_csv = run_dir / "particles_3d.csv"
    elif requested.is_file():
        run_dir = requested.parent
        particles_csv = requested
    else:
        raise FileNotFoundError(f"HoloD3 output does not exist: {requested}")

    if not particles_csv.is_file():
        raise FileNotFoundError(
            f"HoloD3 particles file not found: {particles_csv}. "
            "Pass a HoloD3 run directory or its particles_3d.csv file."
        )
    summary = run_dir / "pipeline_summary.json"
    return HoloD3Source(
        run_dir=run_dir,
        particles_csv=particles_csv,
        pipeline_summary_json=summary if summary.is_file() else None,
    )


def _summary_pixel_pitch(source: HoloD3Source) -> float | None:
    if source.pipeline_summary_json is None:
        return None
    try:
        payload = json.loads(source.pipeline_summary_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read HoloD3 pipeline summary: {source.pipeline_summary_json}") from exc
    optics = payload.get("optics")
    if not isinstance(optics, dict):
        return None
    value = finite_number(optics.get("pixel_pitch_um"))
    return value if value is not None and value > 0 else None


def _infer_pixel_pitch(frame: pd.DataFrame) -> float | None:
    ratios: list[np.ndarray] = []
    for physical, pixels in (("x_um", "seg_xc"), ("y_um", "seg_yc")):
        if physical not in frame.columns or pixels not in frame.columns:
            continue
        physical_values = pd.to_numeric(frame[physical], errors="coerce").to_numpy(np.float64)
        pixel_values = pd.to_numeric(frame[pixels], errors="coerce").to_numpy(np.float64)
        valid = np.isfinite(physical_values) & np.isfinite(pixel_values) & (np.abs(pixel_values) > 1e-6)
        if np.any(valid):
            ratios.append(physical_values[valid] / pixel_values[valid])
    if not ratios:
        return None
    values = np.concatenate(ratios)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if len(values) else None


def resolve_pixel_pitch(
    source: HoloD3Source,
    frame: pd.DataFrame,
    override: float | None,
) -> tuple[float, str]:
    if override is not None:
        if not np.isfinite(override) or override <= 0:
            raise ValueError("--pixel-pitch-um must be a finite positive number")
        return float(override), "command-line"
    summary_value = _summary_pixel_pitch(source)
    if summary_value is not None:
        return summary_value, "pipeline_summary.json"
    inferred = _infer_pixel_pitch(frame)
    if inferred is not None:
        return inferred, "particles_3d.csv"
    return DEFAULT_PIXEL_PITCH_UM, "260712-default"


def _validate_linker_columns(frame: pd.DataFrame, source: Path) -> None:
    missing = sorted(REQUIRED_LINKER_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required HoloD3/linker columns: {missing}")
    if frame.empty:
        raise ValueError(f"HoloD3 output contains no particle detections: {source}")

    numeric_columns = ["frame", "conf", "xc", "yc", "w", "h", "slice", "diameter_px"]
    bad: dict[str, int] = {}
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        count = int(np.count_nonzero(~np.isfinite(values)))
        if count:
            bad[column] = count
    if bad:
        raise ValueError(f"HoloD3 output has non-finite required values: {bad}")

    frames = pd.to_numeric(frame["frame"], errors="raise").to_numpy(np.float64)
    if np.any(frames < 0) or np.any(frames != np.floor(frames)):
        raise ValueError("HoloD3 frame values must be non-negative integers")
    slices = pd.to_numeric(frame["slice"], errors="raise").to_numpy(np.float64)
    if np.any(slices < 1):
        raise ValueError("HoloD3 slice values must be one-based positive numbers")
    for column in ("w", "h", "diameter_px"):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(np.float64)
        if np.any(values <= 0):
            raise ValueError(f"HoloD3 {column} values must be positive")


def prepare_holod3_input(
    holod3_output: str | Path,
    *,
    pixel_pitch_um: float | None = None,
) -> PreparedInput:
    source = resolve_holod3_source(holod3_output)
    frame = pd.read_csv(source.particles_csv)
    _validate_linker_columns(frame, source.particles_csv)
    pitch, pitch_source = resolve_pixel_pitch(source, frame, pixel_pitch_um)

    frame = frame.copy()
    frame["frame"] = pd.to_numeric(frame["frame"], errors="raise").astype(np.int64)
    if "row_id" not in frame.columns:
        frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    else:
        row_ids = pd.to_numeric(frame["row_id"], errors="coerce")
        if row_ids.isna().any() or row_ids.duplicated().any() or np.any(row_ids != np.floor(row_ids)):
            raise ValueError("HoloD3 row_id values must be unique integers")
        frame["row_id"] = row_ids.astype(np.int64)

    diameter_rows_updated = 0
    if "final_diameter_um" in frame.columns:
        good = finite_positive(frame["final_diameter_um"])
        if np.any(good):
            final_um = pd.to_numeric(frame["final_diameter_um"], errors="coerce").to_numpy(np.float64)
            diameter_px = pd.to_numeric(frame["diameter_px"], errors="coerce").to_numpy(np.float64)
            if "diameter_um" in frame.columns:
                diameter_um = pd.to_numeric(frame["diameter_um"], errors="coerce").to_numpy(np.float64)
            else:
                diameter_um = diameter_px * pitch
            frame["diameter_um"] = np.where(good, final_um, diameter_um)
            frame["diameter_px"] = np.where(good, final_um / pitch, diameter_px)
            diameter_rows_updated = int(np.count_nonzero(good))

    z_rows_scaled = 0
    if "z_um" in frame.columns:
        z_um = pd.to_numeric(frame["z_um"], errors="coerce").to_numpy(np.float64)
        good_z = np.isfinite(z_um)
        if np.any(good_z):
            fallback = pd.to_numeric(frame["slice"], errors="raise").to_numpy(np.float64) * 10.0
            frame["z_lap"] = np.where(good_z, z_um / pitch, fallback)
            z_rows_scaled = int(np.count_nonzero(good_z))

    frame = frame.sort_values(["frame", "row_id"], kind="stable").reset_index(drop=True)
    return PreparedInput(
        source=source,
        frame=frame,
        pixel_pitch_um=pitch,
        pixel_pitch_source=pitch_source,
        diameter_rows_updated=diameter_rows_updated,
        z_rows_scaled=z_rows_scaled,
    )
