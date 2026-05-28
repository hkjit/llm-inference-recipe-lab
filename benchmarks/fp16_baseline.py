#!/usr/bin/env python3
"""Baseline FP16 benchmark harness (v1).

This script is intentionally lightweight: it provides a repeatable harness shape
you can later wire into real backends (vLLM/TRT-LLM/SGLang) without changing
reporting semantics.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

try:
    from benchmarks.adapters import InferenceAdapter
    from benchmarks.adapters import get_adapter
except ModuleNotFoundError:
    # Support direct script execution: `python3 benchmarks/fp16_baseline.py`
    from adapters import InferenceAdapter
    from adapters import get_adapter


@dataclass
class BenchmarkResult:
    total_requests: int
    prompt_tokens: int
    generated_tokens: int
    elapsed_s: float
    ttft_ms_p50: float
    ttft_ms_p95: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline FP16 benchmark.")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("benchmarks/sample_prompts.jsonl"),
        help="Path to JSONL prompts, each line: {\"prompt\": \"...\"}",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--engine",
        type=str,
        default="mock-vllm",
        choices=["mock-vllm", "mock-trtllm", "mock-sglang"],
        help="Engine adapter to benchmark.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write benchmark metrics JSON.",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    prompts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        item = json.loads(raw_line)
        prompts.append(item["prompt"])
    if not prompts:
        raise ValueError("No prompts found.")
    return prompts


def rough_token_count(text: str) -> int:
    # Cheap tokenizer proxy so harness stays dependency-free.
    return max(1, len(text.split()))


def run_benchmark(
    prompts: list[str],
    max_new_tokens: int,
    adapter: InferenceAdapter,
) -> BenchmarkResult:
    ttft_ms: list[float] = []
    prompt_tokens = 0
    generated_tokens = 0

    t0 = time.perf_counter()
    for prompt in prompts:
        prompt_tokens += rough_token_count(prompt)
        sample = adapter.infer(prompt=prompt, max_new_tokens=max_new_tokens)
        ttft_ms.append(sample.ttft_ms)
        generated_tokens += sample.generated_tokens
    elapsed_s = time.perf_counter() - t0

    p50 = statistics.median(ttft_ms)
    # simple percentile without numpy
    p95 = sorted(ttft_ms)[int(0.95 * (len(ttft_ms) - 1))]

    return BenchmarkResult(
        total_requests=len(prompts),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        elapsed_s=elapsed_s,
        ttft_ms_p50=p50,
        ttft_ms_p95=p95,
    )


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    adapter = get_adapter(engine=args.engine, seed=args.seed)
    result = run_benchmark(prompts, args.max_new_tokens, adapter)

    total_tokens = result.prompt_tokens + result.generated_tokens
    req_per_s = result.total_requests / max(result.elapsed_s, 1e-9)
    tok_per_s = total_tokens / max(result.elapsed_s, 1e-9)

    print(f"=== FP16 Baseline (Harness v1) [{args.engine}] ===")
    print(f"requests:         {result.total_requests}")
    print(f"prompt_tokens:    {result.prompt_tokens}")
    print(f"generated_tokens: {result.generated_tokens}")
    print(f"elapsed_s:        {result.elapsed_s:.6f}")
    print(f"req_per_s:        {req_per_s:.2f}")
    print(f"tok_per_s:        {tok_per_s:.2f}")
    print(f"ttft_ms_p50:      {result.ttft_ms_p50:.2f}")
    print(f"ttft_ms_p95:      {result.ttft_ms_p95:.2f}")
    print()
    print("Next step: replace mock adapter implementation with real engine client calls.")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(result)
        payload["engine"] = args.engine
        payload["req_per_s"] = req_per_s
        payload["tok_per_s"] = tok_per_s
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote metrics JSON to {args.output_json}")


if __name__ == "__main__":
    main()
