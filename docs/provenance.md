# Linker provenance

The association algorithm in `src/holod3_linker/linker.py` was extracted on
2026-09-03 from the active local `260712` pipeline:

```text
/media/dai-server/M2Disk2TB/repos/260712/
  src/tracking/link_best_tiebreak_z010.py
  src/tracking/laptrack/depth3d_online_link.py
```

The original `depth3d_online_link.py` SHA-256 was:

```text
e2bbd0e3f0f214715a1edc523960392ab98c188fe3185afbf7fcd9e2128c2a8b
```

The integration defaults match `link_best_tiebreak_z010.py`: fixed-lag mode,
segmented XY centres, bounding-box area for the primary size signal, and Z as a
near-tie discriminator with weight `0.10`.

Packaging changes are intentionally narrow:

- imports were changed from sibling scripts to package modules;
- the four required generic helpers were isolated in `core.py` so the linker
  does not import the unused ROI-feature neural network stack;
- the trajectory writer was isolated in `outputs.py` with the same file schemas;
- relative paths now resolve from the caller's working directory, while the
  public runner always supplies absolute paths; and
- `adapter.py` reads HoloD3 metadata and performs acquisition-aware unit
  conversion before invoking the association algorithm.

The HoloD3 output contract was checked against `dainakai/HoloD3` commit:

```text
bdd7cf0d1ce57a86e7e9205b493ecd5fdc04bc50
```

## Equivalence check

The packaged linker and the original `260712` entry point were run independently
against the same 30-frame HoloD3 result on 2026-09-03. The result contained
8,624 detections. All 8,624 `track_id` assignments matched exactly, both runs
produced 2,845 trajectories, and there were zero mismatches.
