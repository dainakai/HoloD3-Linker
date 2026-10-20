# Linker provenance

The association algorithm in `src/holod3_linker/linker.py` was extracted on
2026-09-03 from the active `260712` pipeline source tree:

```text
260712/src/tracking/link_best_tiebreak_z010.py
260712/src/tracking/laptrack/depth3d_online_link.py
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

The HoloD3 output contract was checked against
[`dainakai/HoloD3`](https://github.com/dainakai/HoloD3) commit:

```text
bdd7cf0d1ce57a86e7e9205b493ecd5fdc04bc50
```

## Equivalence check

The packaged linker and the original `260712` entry point were run independently
against the same 30-frame HoloD3 result on 2026-09-03. The result contained
8,624 detections. All 8,624 `track_id` assignments matched exactly, both runs
produced 2,845 trajectories, and there were zero mismatches.

## Optional depth repair

`depth_repair.py` adds a separate, opt-in association check after the extracted
online/stitching algorithm and before output. It is not part of the original
source hash above. The default `off` path keeps the existing associations;
`audit` records proposals without changing them; `split` only changes IDs and
adds provenance. Measurements are preserved.

The [algorithm](depth-repair.md) and
[validation record](depth-repair-validation.md) document synthetic comparisons,
the unsmoothed 260715 acquisition, an end-to-end off/audit/split comparison,
and known failure cases. The output diagnostics identify the new algorithm
version and configuration independently of the upstream source hash.
