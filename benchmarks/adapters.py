"""Engine adapter abstractions for benchmark harnesses."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class InferenceSample:
    ttft_ms: float
    generated_tokens: int


class InferenceAdapter:
    """Minimal interface that benchmark runners depend on."""

    def infer(self, prompt: str, max_new_tokens: int) -> InferenceSample:
        raise NotImplementedError


class MockFp16Adapter(InferenceAdapter):
    """Deterministic mock adapter for local harness iteration."""

    def __init__(self, seed: int, engine_flavor: str = "mock-vllm") -> None:
        self._rng = random.Random(seed)
        self._engine_flavor = engine_flavor

    def infer(self, prompt: str, max_new_tokens: int) -> InferenceSample:
        # Flavor-specific jitter helps simulate engine variance while
        # preserving deterministic output for a fixed seed.
        flavor_offset = {
            "mock-vllm": 0.0,
            "mock-trtllm": -2.0,
            "mock-sglang": 1.5,
        }.get(self._engine_flavor, 0.0)

        ttft = max(5.0, self._rng.gauss(28.0 + flavor_offset, 6.0))
        decode_len = max(8, min(max_new_tokens, int(self._rng.gauss(96, 20))))
        return InferenceSample(ttft_ms=ttft, generated_tokens=decode_len)


def get_adapter(engine: str, seed: int) -> InferenceAdapter:
    supported = {"mock-vllm", "mock-trtllm", "mock-sglang"}
    if engine not in supported:
        raise ValueError(f"Unsupported engine: {engine}. Expected one of: {sorted(supported)}")
    return MockFp16Adapter(seed=seed, engine_flavor=engine)
