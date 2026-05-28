#!/usr/bin/env python3
"""Create a simple JSONL calibration stub dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ROWS = [
    "Summarize the following support conversation in three bullet points.",
    "Write a concise explanation of activation clipping in PTQ.",
    "Explain why per-channel scaling can reduce quantization error.",
    "Compare structured and unstructured sparsity for deployment.",
    "Provide a short answer with one reason and one caveat.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate calibration_stub_v1 JSONL data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration_data/calibration_stub_v1.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument("--repeat", type=int, default=100, help="Repeat default rows N times.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = DEFAULT_ROWS * args.repeat
    with args.output.open("w", encoding="utf-8") as f:
        for idx, text in enumerate(rows):
            f.write(json.dumps({"id": idx, "text": text}) + "\n")

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
