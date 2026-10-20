# Depth repair validation

These experiments evaluate `persistent-local-depth-jump-v1` with its default
parameters. The initial implementation targets **precision of conservative
splits**, accepting missed ambiguous associations. It does not establish
ground-truth tracking accuracy on the experimental acquisition.

## Reproducible synthetic comparison

```bash
uv sync --frozen --extra dev
uv run python benchmarks/evaluate_depth_repair.py \
  --seed 260717 --repeats 100 --output /tmp/depth-evaluation
uv run python benchmarks/evaluate_depth_repair.py \
  --seed 260717 --repeats 100 --stress --output /tmp/depth-stress
```

The primary suite contains 1,900 tracks / 450,565 observations, with 100
repetitions of 19 analytic cases. Identity changes are known by construction.
The 1,300 control tracks include constant and fast velocity, acceleration,
helical depth motion, a continuous velocity change, a resolved sigmoid
transition, isolated outliers, returning bursts of 8/24/40 observations,
endpoint contamination, changing noise, and a long observation gap.
Positive cases include 12/30/100-noise-scale steps, a step with isolated
outliers, a step across a missing-frame interval, and two steps.

All methods see noisy observations, not the known generating noise scale.
The local methods estimate one shared population scale over the batch.
The baselines share fit and bounded-loss primitives with the implementation
to compare decision rules; they are illustrative implementations, not tuned
state-of-the-art competitors. The recursive global M baseline uses quadratic
fits, the same minimum segment length, and candidate screening.

| Method | True cuts found within ±3 observations / 700 | Exact cuts | Intact controls / 1,300 | False cuts on controls | Extra cuts on positive tracks | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adjacent velocity threshold | 599 | 598 | 662 | 5,343 | 2,961 | 0.066 |
| Recursive global quadratic M fit | 700 | 694 | 1,096 | 259 | 0 | 2.185 |
| Local method without return veto | 666 | 663 | 1,100 | 400 | 0 | 1.535 |
| Local method with return veto | **666** | **663** | **1,300** | **0** | **0** | **1.504** |

The full method missed 33 of the 100 weakest (12-scale) steps and one step
with outliers. It found all 30-scale, 100-scale, missing-frame, and two-step
events in this suite. The return veto removed the false cuts in longer
outlier bursts. The global polynomial had better weak-step recall but
incorrectly split valid nonpolynomial motion and contamination.

Timings are one local run, not a timing confidence interval. They include
method-specific preparation and result collection, exclude synthetic-data
generation and CSV writing, and do not include linking. The local timing
includes building the combined table. Detection counts are reproducible;
wall times depend on hardware and system load.

### Deliberate failure cases

The stress command adds 500 tracks to the same population. These cases
expose limitations rather than extending the claim of zero false cuts:

| Added family, 100 tracks each | Result for full method | Interpretation |
| --- | --- | --- |
| Depth-estimator bias lasting 80 observations | 200 false cuts; all 100 tracks split | A long bias is indistinguishable from two identity changes using these coordinates |
| Continuous sigmoid transition with width 0.8 frame | 100 false cuts; all 100 tracks split | Fast motion unresolved by the sampling looks discontinuous |
| Correlated noise, AR(1) coefficient 0.97 | All 100 intact | This particular correlated-noise control is tolerated |
| True identity change after 12 observations | All 100 missed | Minimum segment length intentionally excludes endpoints |
| True swap lasting 32 observations, then returning | All 200 true cuts missed | The return veto cannot distinguish this from an outlier burst |

Thus **0/1,300 false cuts applies only to the primary synthetic controls**.
It is not a general error bound. Adding a different population also changes
the shared scale and can change detection of weak primary-suite events.

## Experimental acquisition 260715

The references in the 260804 experiment directory resolve to the original
unsmoothed `trajectory3d/path_*.csv` files in the 260715 acquisition, under
`2026-07-16-194600/fixedlag_gap12_gg12_zsmooth`. Despite the directory name,
the input used here is the original compact trajectory output, not the
separate smoothed results.

There are 17,790,777 observations and 194,596 tracks across 14 scenes.
The whole acquisition was checked with the default configuration, one
population scale per scene, and no supplied physical bounds. Cached raw
coordinates were used to separate algorithm time from thousands of CSV reads.
Depth was measured in slices; multiplying it by ten reproduces the same cuts
in the linker's pixel-equivalent units.

| Scene suffix | Observations | Proposed cuts | Changed tracks | Detection seconds |
| --- | ---: | ---: | ---: | ---: |
| S0012 | 1,184,308 | 76 | 74 | 1.004 |
| S0013 | 1,671,141 | 190 | 184 | 1.482 |
| S0015 | 1,295,544 | 97 | 96 | 0.935 |
| S0017 | 1,355,818 | 112 | 112 | 0.985 |
| S0018 | 1,249,636 | 84 | 82 | 0.913 |
| S0020 | 936,655 | 53 | 51 | 0.690 |
| S0021 | 1,170,685 | 91 | 90 | 0.853 |
| S0022 | 1,305,642 | 82 | 81 | 0.975 |
| S0024 | 1,130,440 | 74 | 71 | 0.807 |
| S0025 | 1,128,904 | 72 | 70 | 0.859 |
| S0026 | 1,265,544 | 76 | 74 | 0.952 |
| S0028 | 1,284,336 | 75 | 71 | 0.930 |
| S0029 | 1,499,154 | 125 | 124 | 1.292 |
| S0030 | 1,312,970 | 99 | 98 | 1.010 |
| **Total** | **17,790,777** | **1,306** | **1,278** | **13.685** |

99.34% of original tracks retain their assignments. This is a change rate,
**not** the fraction of correct tracks. Many tracks are too short to test.
The routine includes grouping and sorting; only candidate windows need
robust fits. Fixed windows avoid fitting a polynomial to every complete
trajectory or searching all possible global segmentations.

To reproduce an audit on available original trajectories without changing
source files:

```bash
uv run python benchmarks/audit_trajectory_directory.py \
  /path/to/fixedlag_gap12_gg12_zsmooth/trajectory3d \
  --z-scale 10 --output /tmp/depth-audit
```

The utility writes proposals, configuration, source hash, separate load time,
and algorithm time. A direct S0012 run read 11,296 files in 7.181 s, then
found the same 76 cuts in 0.906 s. Its ordered filename-and-content SHA-256:
`d5e163766effb0333d7da7edd8bbe6176035df9a616a8260902a5b041f8154cd`.

### Visual cases and selection bias

Before developing the detector, large interior depth jumps in S0012 were
ranked using a separate nine-point median rule and inspected in raw depth
and XY plots. Ten clear suspected switches and 16 controls were chosen.
All ten suspected transitions were cut and all 16 controls stayed intact.
Controls included endpoint contamination, isolated depth outliers, returning
outlier bursts, and smooth genuine depth motion.

Extending inspection to S0020 and S0029 gives 52 reviewed cases:
31 suspected switches, 17 controls, and four ambiguous cases.
The detector found 30/31 reference transitions within three observations,
preserved all 17 controls, and split three of the four ambiguous cases.
The missed transition had mixed/interleaved depths that defeated clean
two-sided support. One suspected track also had an additional proposed cut.

These are **nonblind visual labels selected for conspicuous features**.
They are not independent particle-identity ground truth, a random sample of
all tracks, or an estimate of precision for all 1,306 proposals. In particular,
the three split ambiguous cases are unresolved, not counted as successes.
Experimental coordinates and plots remain outside the public repository.

A further sample of 18 accepted edges was drawn from the other 11 scenes
with seed 260718, without changing the detector afterward. Fifteen showed
coherent persistent transitions. Three were left unresolved because of
boundary proximity or a longer returning depth pattern. This additional
inspection reinforces the need for known volume bounds and audit; it does
not turn visual agreement into identity ground truth.

### Current Linker pipeline and downstream Smoother

The current public runner was also executed in off, audit, and split modes
on S0012 frames 5500–5950: 451 frames / 49,254 detections. The input schema
was HoloD3 `particles_3d.csv`, with its original detection IDs and an explicit
10 µm pixel pitch matching the saved legacy association coordinates.

| Mode | Output tracks | Proposals / applied cuts | Total seconds |
| --- | ---: | --- | ---: |
| off | 447 | 0 / 0 | 13.229 |
| audit | 447 | 2 / 0 | 12.956 |
| split | 449 | 2 / 2 | 13.099 |

The added detection pass took 0.0345 s. Total-run timing differences are
ordinary run variation, not a speedup claim. Off and audit tracked tables
were exactly equal. Split preserved all non-ID columns and observations,
and `linker_parent_track_id` exactly matched the original assignment.
Source files were unchanged.

The two cuts occur at frames 5708 and 5711, corresponding to the conspicuous
reciprocal depth swap in the saved trajectories. The before/after observations
remain available to Smoother as separate tracks.

An unchanged HoloD3-Smoother was run before and after splitting, with fixed
degree two and the same GMM configuration, on 153 selected tracks including
the initial positives, controls, and a deterministic background sample.
Both EM fits converged. In one curved wrong association, maximum fitted
absolute depth velocity fell from 0.4038 to 0.00848 slices/frame. Another
case previously treated a whole depth branch as outliers; splitting retained
both branches. **Lower velocity is not itself the acceptance criterion**,
and not every corrected track has a lower maximum fitted velocity.
Control assignments were unchanged; their fitted numbers can change slightly
because Smoother re-estimates shared priors and noise over the changed set.

## Reproduction environment and release decision

- Linker base: `fbe935a5d7fcd6b0c04ad0b2b2b9786c23f08cdf`.
- Smoother reference: `bdba80971b9ff530b3d8dcebf07b352de6b6e511`.
- Python 3.12.12; NumPy 2.5.2; pandas 3.0.5; SciPy 1.18.1; Linux;
  Intel Core i7-14700K.
- No GPU, Smoother dependency, or additional runtime package is needed by
  the repair pass.
- Automated regression tests cover actual frame gaps, units and row order,
  boundary vetoes, unchanged observations, provenance, ID allocation,
  invalid inputs, and consistent CLI/summary/trajectory outputs.

The default stays **off**. Use `audit` to review a new acquisition and `split`
when this conservative discontinuity model fits it. Enabling it by default
would require broader independently labeled acquisitions, especially real
rapid/rotating motion, sustained depth-estimation biases, and short swaps.
