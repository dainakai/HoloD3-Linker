# Contributing

Bug reports and focused pull requests are welcome. Before proposing a change,
please search the existing issues and include the HoloD3 commit, the
HoloD3 Linker version, and a minimal description of the input schema involved.
Do not attach private research data or model weights to a public issue.

## Development setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/dainakai/HoloD3-Linker.git
cd HoloD3-Linker
uv sync --frozen --extra dev
uv run --frozen pytest
uv run --frozen ruff check .
uv build
```

Keep changes narrowly scoped and add regression tests for behavior changes.
Changes to the association algorithm or its production defaults should also
document how equivalence or improved behavior was evaluated against the source
listed in [docs/provenance.md](docs/provenance.md).
