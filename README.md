# LLM Inference Recipe Lab

A collaborative engineering repository for building, validating, and benchmarking quantized/sparse LLM inference recipes in the open.

## Scope

The lab centers on recipe-driven optimization work relevant to production inference stacks:

- **Recipes**: low-precision/sparsity transformations (including calibration and scaling choices)
- **Engine integrations**: vLLM, TRT-LLM, SGLang adapters and prototype hooks
- **Validation**: correctness checks, numeric debugging helpers, and acceptance gates
- **Benchmarking**: prefill/decode metrics, tokens/sec, and tail latency

## Project goals

- Provide reproducible reference implementations for PTQ, QAT, and sparsity recipes.
- Make engine integration patterns portable across vLLM, TRT-LLM, and SGLang.
- Standardize benchmarking and numerics-debugging workflows for community learning.
- Create a contribution-friendly space for experimentation, review, and iteration.

## Repository layout

- `recipes/` recipe specs, configs, and calibration settings
- `engines/` engine-specific implementation shims/prototypes
- `benchmarks/` benchmark harnesses and workloads
- `tools/` numeric analysis and visualization utilities
- `tests/` unit/integration checks for recipe correctness
- `.github/workflows/` CI checks for lint/test/smoke

## Suggested first milestones

1. Wire a real vLLM/TRT-LLM/SGLang client call path into benchmark adapters.
2. Add reference benchmark outputs (JSON artifacts) for baseline comparisons.
3. Expand engine adapter correctness/integration tests.
4. Add sparsity-aware benchmarking and compare against baseline.
5. Add basic CI (lint + tests + benchmark smoke).

## Notes

This repo intentionally starts lightweight so it can evolve with concrete recipe experiments and kernel/model integration work.

## Contributing

Contributions are welcome. See `CONTRIBUTING.md` for setup, contribution workflow, and review expectations.