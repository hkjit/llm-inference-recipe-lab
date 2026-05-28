# LLM Inference Recipe Lab

A focused engineering repo for prototyping, validating, and benchmarking quantized/sparse LLM inference recipes.

## Scope

The lab centers on recipe-driven optimization work relevant to production inference stacks:

- **Recipes**: low-precision/sparsity transformations (including calibration and scaling choices)
- **Engine integrations**: vLLM, TRT-LLM, SGLang adapters and prototype hooks
- **Validation**: correctness checks, numeric debugging helpers, and acceptance gates
- **Benchmarking**: prefill/decode metrics, tokens/sec, and tail latency

## Repository layout

- `recipes/` recipe specs, configs, and calibration settings
- `engines/` engine-specific implementation shims/prototypes
- `benchmarks/` benchmark harnesses and workloads
- `tools/` numeric analysis and visualization utilities
- `tests/` unit/integration checks for recipe correctness
- `.github/workflows/` CI checks for lint/test/smoke

## Suggested first milestones

1. Add a baseline FP16 benchmark script and reference outputs.
2. Add a PTQ recipe (e.g., W8A8/GPTQ-like) with calibration config.
3. Implement one inference-engine adapter path and correctness tests.
4. Add sparsity-aware benchmarking and compare against baseline.
5. Add basic CI (lint + tests + benchmark smoke).

## Notes

This repo intentionally starts lightweight so it can evolve with concrete recipe experiments and kernel/model integration work.