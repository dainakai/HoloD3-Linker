from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def write_depth_jump_run(run_dir: Path) -> None:
    run_dir.mkdir()
    n = 160
    t = np.arange(n)
    z = 100 + 0.03 * t + np.random.default_rng(71).normal(0, 0.3, n)
    z[80:] += 100
    pd.DataFrame(
        {
            "frame": t,
            "file": [f"{f}.png" for f in t],
            "row_id": t + 50,
            "conf": 0.95,
            "xc": 100 + 0.1 * t,
            "yc": 100 + t,
            "w": 10,
            "h": 10,
            "diameter_px": 10,
            "slice": z,
            "z_um": (z - 1) * 100,
            "extra_provenance": "original",
        }
    ).to_csv(run_dir / "particles_3d.csv", index=False)
    (run_dir / "pipeline_summary.json").write_text(json.dumps({"optics": {"pixel_pitch_um": 5}}))


def test_depth_repair_modes_keep_rows_and_downstream_exports_consistent(tmp_path: Path) -> None:
    run_dir = tmp_path / "input"
    write_depth_jump_run(run_dir)
    source_bytes = (run_dir / "particles_3d.csv").read_bytes()
    results = {
        mode: link_holod3_output(run_dir, output_dir=tmp_path / mode, depth_repair=mode)
        for mode in ("off", "audit", "split")
    }
    off, audit, split = (pd.read_csv(results[mode].tracks_csv) for mode in ("off", "audit", "split"))
    assert off.track_id.nunique() == 1
    pd.testing.assert_frame_equal(off, audit)
    assert split.track_id.nunique() == 2
    assert split.groupby("track_id").frame.agg(["min", "max"]).values.tolist() == [[0, 79], [80, 159]]
    pd.testing.assert_frame_equal(
        off.drop(columns=["track_id", "laptrack_id"]),
        split.drop(columns=["track_id", "laptrack_id", "linker_parent_track_id"]),
    )
    np.testing.assert_array_equal(split.linker_parent_track_id, off.track_id)
    assert split.laptrack_id.equals(split.track_id)
    assert (run_dir / "particles_3d.csv").read_bytes() == source_bytes
    audit_cuts = pd.read_csv(results["audit"].output_dir / "depth_link_cuts.csv")
    split_cuts = pd.read_csv(results["split"].output_dir / "depth_link_cuts.csv")
    assert audit_cuts.applied.tolist() == [False]
    assert split_cuts.applied.tolist() == [True]
    pd.testing.assert_frame_equal(audit_cuts.drop(columns="applied"), split_cuts.drop(columns="applied"))
    assert results["split"].summary()["counts"]["depth_cuts"] == 1
    assert results["split"].summary()["linker"]["depth_repair"]["mode"] == "split"
    summary = pd.read_csv(results["split"].tracks_summary_csv)
    assert summary.len.tolist() == [80, 80]
    for tid in split.track_id.unique():
        compact = pd.read_csv(results["split"].trajectory_dir / f"path_{tid:08d}.csv")
        group = split[split.track_id == tid]
        np.testing.assert_allclose(compact.slice, group.slice)
        np.testing.assert_array_equal(compact.frame, group.frame)
    assert not (results["off"].output_dir / "depth_link_cuts.csv").exists()


def test_depth_repair_cli_and_plan_forward_bounds_and_units(tmp_path: Path) -> None:
    run_dir = tmp_path / "input"
    write_depth_jump_run(run_dir)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "holod3_linker.cli",
            str(run_dir),
            "--depth-repair",
            "audit",
            "--depth-repair-z-bounds",
            "0",
            "20000",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["source"]["pixel_pitch_um"] == 5
    assert plan["parameters"]["depth_repair"] == "audit"
    assert plan["parameters"]["depth_repair_config"]["z_bounds"] == [0, 20000]
    assert "--depth-repair-z-bounds" in plan["command"]
    assert not (run_dir / "best_tiebreak_z010").exists()


def test_invalid_depth_repair_mode_cannot_overwrite_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "input"
    write_depth_jump_run(run_dir)
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("original")
    with pytest.raises(ValueError, match="depth_repair"):
        link_holod3_output(run_dir, output_dir=output, depth_repair="unknown", overwrite=True)
    assert marker.read_text() == "original"


def test_depth_repair_arguments_cannot_bypass_public_configuration(tmp_path: Path) -> None:
    run_dir = tmp_path / "input"
    write_synthetic_holod3_run(run_dir)
    with pytest.raises(ValueError, match="controlled"):
        link_holod3_output(run_dir, linker_args=["--depth-repair=split"], dry_run=True)
