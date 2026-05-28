from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.adapters import get_adapter
from benchmarks.fp16_baseline import load_prompts
from benchmarks.fp16_baseline import run_benchmark


def test_load_prompts_reads_jsonl(tmp_path: Path) -> None:
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(
        '\n'.join([json.dumps({"prompt": "hello"}), json.dumps({"prompt": "world"})]),
        encoding="utf-8",
    )
    prompts = load_prompts(prompts_file)
    assert prompts == ["hello", "world"]


def test_benchmark_with_mock_adapter_is_deterministic() -> None:
    prompts = ["one prompt", "two prompt"]
    adapter1 = get_adapter("mock-vllm", seed=123)
    adapter2 = get_adapter("mock-vllm", seed=123)

    result1 = run_benchmark(prompts=prompts, max_new_tokens=64, adapter=adapter1)
    result2 = run_benchmark(prompts=prompts, max_new_tokens=64, adapter=adapter2)

    assert result1.total_requests == 2
    assert result1.prompt_tokens == 4
    assert result1.generated_tokens == result2.generated_tokens
    assert result1.ttft_ms_p50 == result2.ttft_ms_p50
    assert result1.ttft_ms_p95 == result2.ttft_ms_p95
