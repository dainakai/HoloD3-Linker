"""Public runner for the HoloD3-to-260712 linking workflow."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from holod3_linker.adapter import PreparedInput, prepare_holod3_input
from holod3_linker.depth_repair import MODES, DepthRepairConfig

LINKER_SOURCE_SHA256 = "e2bbd0e3f0f214715a1edc523960392ab98c188fe3185afbf7fcd9e2128c2a8b"
BLOCKED_LINKER_ARGUMENTS = {"--input", "--out-dir", "--linking-mode", "--overwrite"}


@dataclass(frozen=True)
class LinkResult:
    output_dir: Path
    tracks_csv: Path
    tracks_summary_csv: Path
    trajectory_dir: Path
    diagnostics_json: Path
    summary_json: Path
    returncode: int = 0

    def summary(self) -> dict[str, Any]:
        return json.loads(self.summary_json.read_text(encoding="utf-8"))


def _validate_output_dir(prepared: PreparedInput, output_dir: Path) -> None:
    if prepared.source.run_dir.is_relative_to(output_dir):
        raise ValueError("The linker output directory must not replace or contain the HoloD3 run directory")
    if output_dir == Path(output_dir.anchor):
        raise ValueError("Refusing to use a filesystem root as the linker output directory")


def _validate_linker_args(arguments: Sequence[str]) -> list[str]:
    result: list[str] = []
    for argument in arguments:
        value = str(argument)
        option = value.split("=", 1)[0]
        if option in BLOCKED_LINKER_ARGUMENTS or option.startswith("--depth-repair"):
            raise ValueError(f"{option} is controlled by holod3-link and cannot be forwarded")
        result.append(value)
    return result


def _linker_command(
    *,
    input_csv: Path,
    output_dir: Path,
    linking_mode: str,
    xy_source: str,
    size_source: str,
    z_mode: str,
    z_weight: float,
    z_tiebreak_margin: float,
    z_tiebreak_penalty: float,
    z_tiebreak_cap: float,
    z_velocity_alpha: float,
    z_start_velocity_alpha: float,
    start_frame: int | None,
    end_frame: int | None,
    depth_repair: str,
    depth_repair_config: DepthRepairConfig,
    linker_args: Sequence[str],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "holod3_linker.linker",
        "--input",
        str(input_csv),
        "--out-dir",
        str(output_dir),
        "--linking-mode",
        linking_mode,
        "--xy-source",
        xy_source,
        "--size-source",
        size_source,
        "--z-mode",
        z_mode,
        "--z-weight",
        str(z_weight),
        "--z-tiebreak-margin",
        str(z_tiebreak_margin),
        "--z-tiebreak-penalty",
        str(z_tiebreak_penalty),
        "--z-tiebreak-cap",
        str(z_tiebreak_cap),
        "--z-velocity-alpha",
        str(z_velocity_alpha),
        "--z-start-velocity-alpha",
        str(z_start_velocity_alpha),
    ]
    if start_frame is not None:
        command.extend(["--start-frame", str(start_frame)])
    if end_frame is not None:
        command.extend(["--end-frame", str(end_frame)])
    command.extend(["--depth-repair", depth_repair])
    for name, value in asdict(depth_repair_config).items():
        if value is not None:
            command.append(f"--depth-repair-{name.replace('_', '-')}")
            if isinstance(value, (tuple, list)):
                command.extend(str(v) for v in value)
            else:
                command.append(str(value))
    command.extend(_validate_linker_args(linker_args))
    return command


def _result(output_dir: Path) -> LinkResult:
    return LinkResult(
        output_dir=output_dir,
        tracks_csv=output_dir / "tracks_with_diam_depth.csv",
        tracks_summary_csv=output_dir / "tracks_summary.csv",
        trajectory_dir=output_dir / "trajectory3d",
        diagnostics_json=output_dir / "depth3d_link_diagnostics.json",
        summary_json=output_dir / "holod3_link_summary.json",
    )


def link_holod3_output(
    holod3_output: str | Path,
    *,
    output_dir: str | Path | None = None,
    pixel_pitch_um: float | None = None,
    linking_mode: str = "fixed-lag",
    xy_source: str = "seg",
    size_source: str = "bbox_area",
    z_mode: str = "tiebreak",
    z_weight: float = 0.10,
    z_tiebreak_margin: float = 0.04,
    z_tiebreak_penalty: float = 0.10,
    z_tiebreak_cap: float = 1.0,
    z_velocity_alpha: float = 0.08,
    z_start_velocity_alpha: float = 0.20,
    start_frame: int | None = None,
    end_frame: int | None = None,
    depth_repair: str = "off",
    depth_repair_config: DepthRepairConfig | None = None,
    linker_args: Sequence[str] = (),
    overwrite: bool = False,
    keep_link_input: bool = False,
    dry_run: bool = False,
) -> LinkResult | dict[str, Any]:
    started = time.perf_counter()
    if depth_repair not in MODES:
        raise ValueError(f"depth_repair must be one of {MODES}")
    repair_config = depth_repair_config or DepthRepairConfig()
    prepared = prepare_holod3_input(holod3_output, pixel_pitch_um=pixel_pitch_um)
    if depth_repair != "off" and "linker_parent_track_id" in prepared.frame:
        raise ValueError("linker_parent_track_id is reserved for depth repair provenance")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else prepared.source.run_dir / "best_tiebreak_z010"
    )
    destination = destination.resolve()
    _validate_output_dir(prepared, destination)
    work_dir = destination / "_work"
    link_input_csv = work_dir / "link_input.csv"
    command = _linker_command(
        input_csv=link_input_csv,
        output_dir=destination,
        linking_mode=linking_mode,
        xy_source=xy_source,
        size_source=size_source,
        z_mode=z_mode,
        z_weight=z_weight,
        z_tiebreak_margin=z_tiebreak_margin,
        z_tiebreak_penalty=z_tiebreak_penalty,
        z_tiebreak_cap=z_tiebreak_cap,
        z_velocity_alpha=z_velocity_alpha,
        z_start_velocity_alpha=z_start_velocity_alpha,
        start_frame=start_frame,
        end_frame=end_frame,
        depth_repair=depth_repair,
        depth_repair_config=repair_config,
        linker_args=linker_args,
    )
    plan = {
        "source": prepared.metadata(),
        "output_dir": str(destination),
        "link_input_csv": str(link_input_csv),
        "parameters": {
            "linking_mode": linking_mode,
            "xy_source": xy_source,
            "size_source": size_source,
            "z_mode": z_mode,
            "z_weight": z_weight,
            "z_tiebreak_margin": z_tiebreak_margin,
            "z_tiebreak_penalty": z_tiebreak_penalty,
            "z_tiebreak_cap": z_tiebreak_cap,
            "z_velocity_alpha": z_velocity_alpha,
            "z_start_velocity_alpha": z_start_velocity_alpha,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "depth_repair": depth_repair,
            "depth_repair_config": asdict(repair_config),
            "linker_args": list(linker_args),
        },
        "command": command,
    }
    if dry_run:
        return plan

    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Linker output already exists: {destination}; pass --overwrite to replace it")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    prepared.write_csv(link_input_csv)

    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, output=completed.stdout)

    result = _result(destination)
    required = [
        result.tracks_csv,
        result.tracks_summary_csv,
        result.trajectory_dir,
        result.diagnostics_json,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Linker completed without required outputs: {missing}")

    if not keep_link_input:
        link_input_csv.unlink(missing_ok=True)
        if work_dir.exists():
            shutil.rmtree(work_dir)

    diagnostics = json.loads(result.diagnostics_json.read_text(encoding="utf-8"))
    summary = {
        "schema_version": 1,
        "method": "260712-best_tiebreak_z010",
        "source": prepared.metadata(),
        "linker": {
            "upstream_source_sha256": LINKER_SOURCE_SHA256,
            "mode": linking_mode,
            "diagnostics_json": str(result.diagnostics_json),
        },
        "parameters": plan["parameters"],
        "outputs": {
            "tracks_csv": str(result.tracks_csv),
            "tracks_summary_csv": str(result.tracks_summary_csv),
            "trajectory_dir": str(result.trajectory_dir),
            "diagnostics_json": str(result.diagnostics_json),
        },
        "counts": {
            "rows": diagnostics.get("rows"),
            "frames": diagnostics.get("frames"),
            "tracks": diagnostics.get("output", {}).get("track_count"),
        },
        "link_input_retained": keep_link_input,
        "link_input_csv": str(link_input_csv) if keep_link_input else None,
        "elapsed_sec": time.perf_counter() - started,
    }
    if depth_repair != "off":
        summary["counts"]["depth_cuts"] = diagnostics["depth_repair"]["cut_count"]
        summary["outputs"]["depth_link_cuts_csv"] = str(destination / "depth_link_cuts.csv")
        summary["linker"]["depth_repair"] = diagnostics["depth_repair"]
    result.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
