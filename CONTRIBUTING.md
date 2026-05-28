# Contributing

Thanks for contributing to `llm-inference-recipe-lab`.

## What to contribute

- New recipe specs in `recipes/` (PTQ/QAT/sparsity variants)
- Engine adapter improvements in `benchmarks/` and `engines/`
- Benchmark workloads/metrics and analysis tooling
- Tests for correctness, determinism, and regression detection
- Documentation improvements and reproducible examples

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Development workflow

1. Create a branch from `main`.
2. Keep changes scoped to one theme (recipe, adapter, benchmark, or tooling).
3. Add or update tests for behavior changes.
4. Run:
  - `pytest -q`
  - `python3 benchmarks/fp16_baseline.py --engine mock-vllm`
5. Open a pull request with:
  - summary of change
  - quality/performance impact
  - test evidence

## Style and quality bar

- Prefer small, reviewable pull requests.
- Keep benchmark outputs reproducible and clearly versioned.
- Document assumptions (tokenizer, calibration data, hardware target).
- Avoid breaking existing benchmark interfaces without migration notes.

