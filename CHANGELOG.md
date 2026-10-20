# Changelog

All notable changes to HoloD3 Linker are documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## Unreleased

- Add opt-in `--depth-repair audit|split` after tracklet stitching. Robust local
  position-discontinuity checks preserve measurements, record cut evidence and
  parent IDs, and protect short returning depth outliers.
- Add reproducible synthetic comparisons, experimental 260715 validation, and
  regression tests for continuous motion, gaps, units, provenance, and exports.

## [0.1.0] - 2026-09-03

### Added

- HoloD3 run-directory and direct-CSV input resolution.
- Physical-unit adaptation from HoloD3 metadata to the `260712` linker.
- Bundled fixed-lag `best_tiebreak_z0.10` trajectory association.
- Safe output handling, dry runs, parameter forwarding, and JSON provenance.
- Synthetic end-to-end tests and exact regression comparison against the
  original `260712` entry point.

[0.1.0]: https://github.com/dainakai/HoloD3-Linker/releases/tag/v0.1.0
