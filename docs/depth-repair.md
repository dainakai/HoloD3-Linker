# Persistent depth discontinuities

`holod3-link --depth-repair split` checks for persistent depth changes **after
online association and tracklet stitching, before writing outputs**. It splits
suspected bad connections without changing or removing any measurements.
Smoother is neither called nor installed by Linker.

The default remains `off`. `audit` writes proposals while preserving the
original assignments. Use audit on a new acquisition before selecting split:
a discontinuity is evidence of a possible identity change, not proof.

```bash
uv run holod3-link /data/run --depth-repair audit --output-dir /data/audit
uv run holod3-link /data/run --depth-repair split --output-dir /data/repaired
holod3-smooth /data/repaired
```

## Motion model and shared scale

A shared velocity threshold confounds actual motion with depth noise. One
global polynomial can turn a wrong association into artificial curvature or
classify a whole branch as outliers. A correct rotating trajectory may also
be poorly represented by one polynomial.

This check shares a robust **noise scale**, not a falling direction, zero depth
velocity, common velocity, or whole-track polynomial shape. Each side of a
candidate has its own velocity. Continuous curvature and velocity changes
compete explicitly with the discontinuous position hypothesis.

For unequal frame intervals, set
`w = (t_i - t_(i-1)) / (t_(i+1) - t_(i-1))` and estimate noise from

```text
e_i = [z_i - (1-w) z_(i-1) - w z_(i+1)] / sqrt(1 + (1-w)^2 + w^2)
sigma_track = 1.4826 median |e_i - median(e)|
```

Interpolation cancels constant velocity. The denominator normalizes the noise
in the three observations. Curvature and serial correlation can still affect
this empirical scale. The population reference is the median of per-track
scales from tracks with at least `min_segment` observations. Equal track
weighting prevents one long bad track from dominating.

## Detection

1. Use the larger of the population and track noise scales, plus a relative
   numerical floor. A five-observation median is used **only for discovery**.
   Subtract a local median velocity from its adjacent displacements, accounting
   for actual frame gaps. Only innovations above four scales need further fits.
2. Fit separate Theil–Sen lines to the raw depths on each side, initially
   using 12 observations per side. Increase the scale to either side's residual
   MAD if it is larger. Require at least 75% residual support within three
   scales on **each** side.
3. Require a position discrepancy of at least eight scales throughout the
   interval between the two observations. Lines meeting inside a missing-frame
   interval are compatible with a continuous velocity change. A gap longer
   than either side's observation span plus its typical frame spacing is
   inconclusive.
4. Compare the split with a continuous hinge and a continuous cubic, fitted
   with Huber IRLS, at most ten iterations. Score all hypotheses using the
   same bounded loss `min((residual / (3 sigma))^2, 1)`. Require a mean cost
   reduction of at least 0.15. A single extreme outlier cannot dominate this
   cost.
5. Refine the location within two observations using bounded costs. Require
   the same signed discontinuity at twice the window size.
6. Inspect up to four smaller-window lengths in both directions. A coherent
   return to the other branch vetoes the split. The return block has
   `max(6, window // 2)` observations and needs 80% support. This deliberately
   leaves short returning excursions to downstream outlier handling.
7. If volume bounds were supplied, veto candidates whose median position on
   either side is near a boundary. Select the strongest nonoverlapping
   candidates, retaining at least `min_segment` observations per segment.

All regressions use frame offsets rather than row indices. The discovery
median and fitted positions never replace measured coordinates. A majority
of erroneous depths within a window can defeat a robust fit.

The design draws on bounded-loss robust changepoints
([Fearnhead and Rigaill](https://arxiv.org/abs/1609.07363)), local linear-window
comparisons ([Kim, Oh and Cho](https://arxiv.org/abs/2208.04900)), and continuous
slope changes ([Maidstone, Fearnhead and Letchford](https://arxiv.org/abs/1701.01672)).
It is a custom conservative heuristic, not an implementation of those papers'
estimators. Their optimality and error-control guarantees do not transfer to
these scores.

## Parameters and units

| Option | Default | Interpretation |
| --- | --- | --- |
| `--depth-repair` | `off` | `off`, `audit`, or `split` |
| `--depth-repair-window` | 12 | Observations per side of the smaller fit |
| `--depth-repair-min-segment` | 24 | Minimum retained length; at least twice the window |
| `--depth-repair-z-score` | 8 | Jump/noise ratio, **not a statistical z-test** |
| `--depth-repair-min-gain` | 0.15 | Minimum bounded-cost improvement |
| `--depth-repair-x-bounds LOW HIGH` | absent | Known bounds in `x_lap` units |
| `--depth-repair-y-bounds LOW HIGH` | absent | Known bounds in `y_lap` units |
| `--depth-repair-z-bounds LOW HIGH` | absent | Known bounds in `z_lap` units |
| `--depth-repair-boundary-margin` | 0.02 | Fraction excluded at each supplied boundary |

Coordinates are the association's pixel-equivalent coordinates. The HoloD3
adapter uses `z_lap = z_um / pixel_pitch_um` when physical depths exist.
Otherwise the internal linker uses `slice * z_lap_scale`, normally `slice * 10`.
**Convert slice or micrometre bounds before passing them as `z_lap` bounds.**
Observed extrema are never silently treated as physical boundaries.
Supply bounds when the acquisition volume is known, particularly for particles
near image edges or the axial reconstruction limit. Without bounds, those
boundary locations do not receive a special veto.

Larger windows demand longer persistence and lengthen the return check, but
can miss short identity swaps and include more real curvature. Larger jump
thresholds reduce sensitivity. The pooled floor can miss small jumps in a
quieter subpopulation. Frame rate matters because windows count observations.

## Outputs and API

When enabled, `depth_link_cuts.csv` records the parent ID, proposed new ID,
zero-based input-table row positions, frames bounding the cut, signed depth
jump, noise scale, jump/noise ratio, cost improvement, support, and `applied`.
If available, `row_id_before` and `row_id_after` identify the original
detections. Audit allocates hypothetical IDs but does not apply them.

In split mode, the first segment keeps its ID; later segments receive IDs
above the largest original ID, deterministically by parent ID and time.
`linker_parent_track_id` preserves the assignment before repair.
`track_id`, `laptrack_id`, trajectory filenames, and summary counts all use the
repaired IDs. Cuts happen after stitching, so rejected edges cannot be
reconnected by that pass. No replacement association is guessed.

The diagnostics record the algorithm version, configuration, population
scale, candidate/veto/cut counts, and runtime. The public summary includes
these diagnostics and the cut-table path. Off mode preserves historical
assignments and the CSV schema.

```python
from holod3_linker import DepthRepairConfig, link_holod3_output, repair_depth_links

linked = link_holod3_output("/data/run", depth_repair="split")

# In-memory use on already linked observations; frame and z_lap are required.
result = repair_depth_links(
    observations, original_track_ids, config=DepthRepairConfig()
)
proposed_ids, proposals = result.track_ids, result.cuts
```

The pipeline also accepts `depth_repair_config=DepthRepairConfig(...)`.
Input tables and ID arrays are not mutated. IDs must fit the existing int32
output contract. Bounds additionally require the corresponding `*_lap` column.

## Alternatives and limits

The benchmark compares a one-step threshold, recursive global quadratic M
fits, local fits without a return veto, and the complete method. Global
polynomials cut valid nonpolynomial motion; removing the return veto splits
outlier bursts. A globally optimized robust segmentation may recover weaker
jumps, but still requires a motion model and cannot identify a step's physical
cause. Hard online depth gates would reject valid detections during outliers.

Joint reassignment of complementary crossing tracks is a useful future step,
but it requires evidence for the replacement links, not merely evidence that
the current edge is wrong. This change makes conservative cuts and retains
the evidence for inspection.

An enduring depth-estimator bias and a true identity swap can produce
identical observations. A real short swap returning to the original particle
can be preserved by the return veto. Short segments, long gaps, same-depth
swaps, and transitions blurred by mixed observations may be missed.
Fast continuous displacement can be falsely split when it is unresolved by
the local windows. Known volume bounds help with boundary artifacts but do
not resolve ambiguity in the volume interior.

See [validation and measured limitations](depth-repair-validation.md) before
enabling split on a new acquisition.
