"""Command-line interface for linking one HoloD3 output directory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from holod3_linker import __version__
from holod3_linker.depth_repair import add_depth_repair_arguments, config_from_args
from holod3_linker.runner import LinkResult, link_holod3_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="holod3-link",
        description="Connect HoloD3 particles_3d.csv detections into temporal 3D trajectories.",
    )
    parser.add_argument(
        "holod3_output",
        type=Path,
        help="HoloD3 run directory, or a direct path to particles_3d.csv.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <HoloD3 run>/best_tiebreak_z010",
    )
    parser.add_argument("--pixel-pitch-um", type=float, default=None, help="Override HoloD3 run metadata.")
    parser.add_argument("--linking-mode", choices=("fixed-lag", "legacy"), default="fixed-lag")
    parser.add_argument("--xy-source", choices=("seg", "bbox", "x_lap"), default="seg")
    parser.add_argument("--size-source", choices=("bbox_area", "diameter"), default="bbox_area")
    parser.add_argument("--z-mode", choices=("tiebreak", "distance"), default="tiebreak")
    parser.add_argument("--z-weight", type=float, default=0.10)
    parser.add_argument("--z-tiebreak-margin", type=float, default=0.04)
    parser.add_argument("--z-tiebreak-penalty", type=float, default=0.10)
    parser.add_argument("--z-tiebreak-cap", type=float, default=1.0)
    parser.add_argument("--z-velocity-alpha", type=float, default=0.08)
    parser.add_argument("--z-start-velocity-alpha", type=float, default=0.20)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    add_depth_repair_arguments(parser)
    parser.add_argument(
        "--linker-arg",
        action="append",
        default=[],
        help="Advanced argument forwarded to the bundled linker; repeat as needed.",
    )
    parser.add_argument("--keep-link-input", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outcome = link_holod3_output(
            args.holod3_output,
            output_dir=args.output_dir,
            pixel_pitch_um=args.pixel_pitch_um,
            linking_mode=args.linking_mode,
            xy_source=args.xy_source,
            size_source=args.size_source,
            z_mode=args.z_mode,
            z_weight=args.z_weight,
            z_tiebreak_margin=args.z_tiebreak_margin,
            z_tiebreak_penalty=args.z_tiebreak_penalty,
            z_tiebreak_cap=args.z_tiebreak_cap,
            z_velocity_alpha=args.z_velocity_alpha,
            z_start_velocity_alpha=args.z_start_velocity_alpha,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            depth_repair=args.depth_repair,
            depth_repair_config=config_from_args(args),
            linker_args=args.linker_arg,
            overwrite=args.overwrite,
            keep_link_input=args.keep_link_input,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"holod3-link: error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"holod3-link: linker failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1

    if isinstance(outcome, LinkResult):
        summary = outcome.summary()
        print(
            json.dumps(
                {
                    "output_dir": str(outcome.output_dir),
                    "tracks": summary["counts"]["tracks"],
                    "tracks_csv": str(outcome.tracks_csv),
                    "trajectory_dir": str(outcome.trajectory_dir),
                    "summary_json": str(outcome.summary_json),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
