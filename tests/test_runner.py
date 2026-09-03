from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from holod3_linker.runner import LinkResult, link_holod3_output


def write_synthetic_holod3_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    rows = []
    for frame in range(6):
        for source_id, (x0, y0, z0) in enumerate(((100.0, 80.0, 20.0), (300.0, 120.0, 40.0), (500.0, 160.0, 60.0))):
            x = x0 - 0.75 * frame
            y = y0 + 4.0 * frame
            z_um = (z0 + frame) * 10.0
            rows.append(
                {
                    "frame": frame,
                    "file": f"{frame:06d}.png",
                    "row_id": len(rows),
                    "source_id": source_id,
                    "conf": 0.95,
                    "xc": x,
                    "yc": y,
                    "w": 10.0,
                    "h": 10.0,
                    "seg_xc": x,
                    "seg_yc": y,
                    "slice": z0 + frame + 1,
                    "diameter_px": 10.0,
                    "diameter_um": 100.0,
                    "final_diameter_um": 100.0,
                    "x_um": x * 10.0,
                    "y_um": y * 10.0,
                    "z_um": z_um,
                }
            )
    pd.DataFrame(rows).to_csv(run_dir / "particles_3d.csv", index=False)
    (run_dir / "pipeline_summary.json").write_text(
        json.dumps({"optics": {"pixel_pitch_um": 10.0, "slice_spacing_um": 100.0}}),
        encoding="utf-8",
    )


def test_link_holod3_output_end_to_end(tmp_path: Path) -> None:
    run_dir = tmp_path / "holod3-run"
    write_synthetic_holod3_run(run_dir)

    result = link_holod3_output(run_dir)

    assert isinstance(result, LinkResult)
    tracked = pd.read_csv(result.tracks_csv)
    assert tracked.groupby("source_id")["track_id"].nunique().eq(1).all()
    assert tracked["track_id"].nunique() == 3
    assert len(list(result.trajectory_dir.glob("path_*.csv"))) == 3
    assert not (result.output_dir / "_work").exists()
    assert result.summary()["counts"] == {"rows": 18, "frames": 6, "tracks": 3}


def test_dry_run_validates_without_writing(tmp_path: Path) -> None:
    run_dir = tmp_path / "holod3-run"
    write_synthetic_holod3_run(run_dir)

    plan = link_holod3_output(run_dir, dry_run=True)

    assert isinstance(plan, dict)
    assert plan["source"]["rows"] == 18
    assert plan["parameters"]["linking_mode"] == "fixed-lag"
    assert not (run_dir / "best_tiebreak_z010").exists()


def test_existing_output_requires_overwrite(tmp_path: Path) -> None:
    run_dir = tmp_path / "holod3-run"
    write_synthetic_holod3_run(run_dir)
    output_dir = run_dir / "best_tiebreak_z010"
    output_dir.mkdir()
    (output_dir / "marker.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        link_holod3_output(run_dir)

    assert (output_dir / "marker.txt").read_text(encoding="utf-8") == "keep"


def test_run_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    run_dir = tmp_path / "holod3-run"
    write_synthetic_holod3_run(run_dir)

    with pytest.raises(ValueError, match="must not replace or contain"):
        link_holod3_output(run_dir, output_dir=run_dir, overwrite=True)


def test_ancestor_of_run_cannot_be_overwritten(tmp_path: Path) -> None:
    run_dir = tmp_path / "holod3-run"
    write_synthetic_holod3_run(run_dir)

    with pytest.raises(ValueError, match="must not replace or contain"):
        link_holod3_output(run_dir, output_dir=tmp_path, overwrite=True)


def test_managed_linker_arguments_cannot_be_forwarded(tmp_path: Path) -> None:
    run_dir = tmp_path / "holod3-run"
    write_synthetic_holod3_run(run_dir)

    with pytest.raises(ValueError, match="controlled"):
        link_holod3_output(run_dir, linker_args=["--out-dir=/tmp/elsewhere"], dry_run=True)
