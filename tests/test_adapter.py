from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from holod3_linker.adapter import prepare_holod3_input, resolve_holod3_source


def write_particles(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "frame": 0,
                "file": "000000.png",
                "row_id": 0,
                "conf": 0.9,
                "xc": 10.0,
                "yc": 20.0,
                "w": 8.0,
                "h": 10.0,
                "seg_xc": 11.0,
                "seg_yc": 19.0,
                "slice": 2,
                "diameter_px": 9.0,
                "diameter_um": 45.0,
                "final_diameter_um": 50.0,
                "x_um": 55.0,
                "y_um": 95.0,
                "z_um": 100.0,
            },
            {
                "frame": 1,
                "file": "000001.png",
                "row_id": 1,
                "conf": 0.8,
                "xc": 9.0,
                "yc": 24.0,
                "w": 8.0,
                "h": 10.0,
                "seg_xc": 10.0,
                "seg_yc": 23.0,
                "slice": 3,
                "diameter_px": 9.0,
                "diameter_um": 45.0,
                "final_diameter_um": np.nan,
                "x_um": 50.0,
                "y_um": 115.0,
                "z_um": 200.0,
            },
        ]
    ).to_csv(path, index=False)


def test_prepare_run_directory_uses_summary_units(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_particles(run_dir / "particles_3d.csv")
    (run_dir / "pipeline_summary.json").write_text(
        json.dumps({"optics": {"pixel_pitch_um": 5.0}}),
        encoding="utf-8",
    )

    prepared = prepare_holod3_input(run_dir)

    assert prepared.pixel_pitch_um == 5.0
    assert prepared.pixel_pitch_source == "pipeline_summary.json"
    assert prepared.diameter_rows_updated == 1
    assert prepared.frame["diameter_px"].tolist() == [10.0, 9.0]
    assert prepared.frame["diameter_um"].tolist() == [50.0, 45.0]
    assert prepared.frame["z_lap"].tolist() == [20.0, 40.0]


def test_prepare_direct_csv_infers_pixel_pitch(tmp_path: Path) -> None:
    csv_path = tmp_path / "particles_3d.csv"
    write_particles(csv_path)

    prepared = prepare_holod3_input(csv_path)

    assert prepared.pixel_pitch_um == pytest.approx(5.0)
    assert prepared.pixel_pitch_source == "particles_3d.csv"
    assert prepared.source.run_dir == tmp_path


def test_explicit_pixel_pitch_has_priority(tmp_path: Path) -> None:
    csv_path = tmp_path / "particles_3d.csv"
    write_particles(csv_path)

    prepared = prepare_holod3_input(csv_path, pixel_pitch_um=2.5)

    assert prepared.pixel_pitch_um == 2.5
    assert prepared.pixel_pitch_source == "command-line"
    assert prepared.frame.loc[0, "diameter_px"] == 20.0


def test_resolve_directory_requires_particles_csv(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="particles_3d.csv"):
        resolve_holod3_source(tmp_path)


def test_missing_linker_column_is_reported(tmp_path: Path) -> None:
    csv_path = tmp_path / "particles_3d.csv"
    pd.DataFrame([{"frame": 0}]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required"):
        prepare_holod3_input(csv_path)
