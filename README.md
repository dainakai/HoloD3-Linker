# HoloD3 Linker

[![CI](https://github.com/dainakai/HoloD3-Linker/actions/workflows/ci.yml/badge.svg)](https://github.com/dainakai/HoloD3-Linker/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

`HoloD3 Linker` connects per-frame 3D particle measurements produced by
[`dainakai/HoloD3`](https://github.com/dainakai/HoloD3) into temporal trajectories.
It packages the fixed-lag `best_tiebreak_z0.10` linker used in the `260712`
pipeline behind a single HoloD3-aware command.

HoloD3 deliberately stops at independent per-frame measurements. This repository
starts from HoloD3's stable `particles_3d.csv` output and adds temporal `track_id`
values without rerunning detection, depth estimation, or diameter estimation.

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/dainakai/HoloD3-Linker.git
cd HoloD3-Linker
uv sync --frozen

uv run holod3-link /absolute/path/to/HoloD3/runs/my-run
```

For a standalone command that does not require cloning the repository:

```bash
uv tool install git+https://github.com/dainakai/HoloD3-Linker.git
holod3-link /absolute/path/to/HoloD3/runs/my-run
```

The positional argument can be either:

- a HoloD3 run directory containing `particles_3d.csv`; or
- the `particles_3d.csv` file itself.

By default, outputs are written to
`<HoloD3 run>/best_tiebreak_z010`. Use `--output-dir` to put them elsewhere:

```bash
uv run holod3-link /data/HoloD3/runs/experiment-01 \
  --output-dir /data/tracking/experiment-01
```

To run HoloD3 and then link its result:

```bash
# In the HoloD3 repository
uv run holod3 infer \
  --acquisition acquisitions/experiment-01/acquisition.yaml \
  --run-dir runs/experiment-01 \
  --preset portable-torch \
  --limit 0

# In this repository
uv run holod3-link /absolute/path/to/HoloD3/runs/experiment-01
```

## Outputs

| Path | Contents |
| --- | --- |
| `tracks_with_diam_depth.csv` | All HoloD3 columns plus temporal `track_id`, `laptrack_id`, and linker coordinates |
| `tracks_with_diameter.csv` | Compatibility copy with depth-model detail columns removed |
| `tracks_summary.csv` | One row per trajectory with length and first/last frame |
| `trajectory3d/path_XXXXXXXX.csv` | One compact CSV per trajectory, compatible with the `260712` downstream tools |
| `depth3d_link_diagnostics.json` | Detailed online-link and tracklet-stitch diagnostics |
| `holod3_link_summary.json` | Source paths, unit conversion, parameters, counts, and output paths |

HoloD3's input `particles_3d.csv` is never modified. Existing linker output is
also left untouched unless `--overwrite` is explicitly supplied.

## HoloD3-aware unit conversion

The `260712` linker works in pixel-equivalent coordinates. `holod3-link` reads
`optics.pixel_pitch_um` from the HoloD3 `pipeline_summary.json` and converts:

- `final_diameter_um` to the linker's `diameter_px`; and
- `z_um` to pixel-equivalent `z_lap`.

This reproduces the original `260712` conversion for its calibrated
`10 µm/px` acquisition while allowing HoloD3 runs with other pixel pitches.
Resolution order is:

1. explicit `--pixel-pitch-um`;
2. HoloD3 `pipeline_summary.json`;
3. inference from `x_um / seg_xc` and `y_um / seg_yc`; then
4. the `260712` compatibility default of `10 µm/px`.

The selected source and value are recorded in `holod3_link_summary.json`.

## Input contract

The adapter consumes HoloD3's tabular output rather than images or model
checkpoints. The required `particles_3d.csv` columns are `frame`, `file`,
`conf`, `xc`, `yc`, `w`, `h`, `slice`, and `diameter_px`. When present,
`seg_xc`, `seg_yc`, `final_diameter_um`, `diameter_um`, `x_um`, `y_um`, and
`z_um` are used to preserve HoloD3's refined coordinates and physical units.

Rows are stably ordered by `(frame, row_id)`. If `row_id` is absent, the
adapter assigns one from the original CSV row order. Invalid required values,
duplicate row IDs, empty inputs, and unsafe output paths fail before linking.

## Linker defaults

The command uses the production settings from
`260712/src/tracking/link_best_tiebreak_z010.py`:

| Setting | Default |
| --- | --- |
| linking mode | `fixed-lag` |
| XY source | segmented centre (`seg_xc`, `seg_yc`) |
| size source | detector bounding-box area |
| Z use | tie-break among near-equal XY candidates |
| Z weight | `0.10` |
| online gap retention | up to 12 frames |
| robust history predictor | enabled |

The old online behavior remains available as `--linking-mode legacy`.

Common overrides are first-class options:

```bash
uv run holod3-link /data/HoloD3/runs/experiment-01 \
  --size-source diameter \
  --start-frame 100 \
  --end-frame 999 \
  --overwrite
```

Any advanced option accepted by the bundled linker can be forwarded. Use the
equals form when the forwarded value starts with `-`:

```bash
uv run holod3-link /data/HoloD3/runs/experiment-01 \
  --linker-arg=--gap-min-hits \
  --linker-arg=3 \
  --linker-arg=--gap-assignment-margin \
  --linker-arg=0.05
```

Inspect resolved paths, conversions, and the exact internal command without
creating files:

```bash
uv run holod3-link /data/HoloD3/runs/experiment-01 --dry-run
```

## Python API

```python
from holod3_linker import link_holod3_output

result = link_holod3_output(
    "/data/HoloD3/runs/experiment-01",
    output_dir="/data/tracking/experiment-01",
)

print(result.tracks_csv)
print(result.trajectory_dir)
print(result.summary()["counts"])
```

## Scope and validation

This package performs temporal association only. It does not rerun HoloD3,
smooth reconstructed Z trajectories, or evaluate against ground truth. The
default motion and gating parameters were selected for the `260712` acquisition
and should be validated before being used with substantially different frame
rates, flow speeds, particle densities, or optical scales.

The bundled implementation was regression-checked against the original
`260712` entry point on a 30-frame HoloD3 result: all 8,624 detections received
the same `track_id`, yielding 2,845 trajectories in both runs. See
[the provenance note](docs/provenance.md) for the exact source hashes and test
conditions.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv build
```

See [docs/provenance.md](docs/provenance.md) for the source snapshot and the
packaging changes made to isolate the linker from the larger `260712` pipeline.
Contribution guidelines are in [CONTRIBUTING.md](CONTRIBUTING.md), and security
reports should follow [SECURITY.md](SECURITY.md).

## License

HoloD3 Linker is released under the [MIT License](LICENSE.md).
