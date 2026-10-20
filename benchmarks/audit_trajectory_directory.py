"""Audit saved unsmoothed trajectory3d files without rerunning association.

This experiment utility writes only cut proposals and aggregate diagnostics.
Production workflows use holod3-link --depth-repair audit|split instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from holod3_linker import repair_depth_links
from holod3_linker.depth_repair import add_depth_repair_arguments, config_from_args


def audit(source: Path, output: Path, config, z_scale: float) -> dict:
    directory = source / "trajectory3d" if (source / "trajectory3d").is_dir() else source
    paths = sorted(directory.glob("path_*.csv"))
    if not paths:
        raise ValueError(f"No path_*.csv files in {directory}")
    started = time.perf_counter()
    tables, ids, digest = [], [], hashlib.sha256()
    for path in paths:
        raw = path.read_bytes()
        digest.update(path.name.encode() + b"\0" + raw)
        table = pd.read_csv(path)
        columns = {"frame", "xc", "yc", "slice"}
        if not columns.issubset(table):
            raise ValueError(f"{path} is not an unsmoothed trajectory CSV")
        table = table[["frame", "xc", "yc", "slice"]].rename(columns={"xc": "x_lap", "yc": "y_lap", "slice": "z_lap"})
        table["z_lap"] *= z_scale
        tables.append(table)
        ids.append(np.full(len(table), int(path.stem.removeprefix("path_")), dtype=np.int64))
    frame = pd.concat(tables, ignore_index=True)
    load_sec = time.perf_counter() - started
    result = repair_depth_links(frame, np.concatenate(ids), config=config)
    output.mkdir(parents=True, exist_ok=True)
    result.cuts.to_csv(output / "cuts.csv", index=False)
    metrics = {
        "source": str(directory.resolve()),
        "source_sha256": digest.hexdigest(),
        "rows": len(frame),
        "tracks": len(paths),
        "slice_to_z_lap": z_scale,
        "load_sec": load_sec,
        **result.diagnostics,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--z-scale", type=float, default=10.0, help="slice -> z_lap multiplier.")
    add_depth_repair_arguments(parser, include_mode=False)
    args = parser.parse_args()
    if not np.isfinite(args.z_scale) or args.z_scale <= 0:
        parser.error("--z-scale must be finite and positive")
    print(json.dumps(audit(args.source, args.output, config_from_args(args), args.z_scale), indent=2))


if __name__ == "__main__":
    main()
