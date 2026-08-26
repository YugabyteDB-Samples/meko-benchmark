#!/usr/bin/env python3
"""Run the two published RecipeBench arms over all 320 questions.

Models are serialized within a provider and providers run concurrently. Each
run uses its own resumable ledger under publication/data/ledgers.
"""
from __future__ import annotations

import subprocess
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
QUEUES = {
    "ollama": [
        ("minimax-m2.7:cloud", "full320-minimaxm27"),
        ("glm-5.2:cloud", "full320-glm52"),
    ],
    "claude": [
        ("claude-sonnet-5", "full320-sonnet5"),
        ("claude-opus-5", "full320-opus5"),
    ],
    "agy": [
        ("gemini-3.7-flash-low", "full320-gem37"),
    ],
    "codex": [
        ("gpt-5.5", "full320-gpt55"),
        ("gpt-5.6-luna", "full320-luna"),
        ("gpt-5.6-terra", "full320-terra"),
        ("gpt-5.6-sol", "full320-sol"),
    ],
}


def run_queue(provider: str, models: list[tuple[str, str]],
              common: list[str]) -> tuple[str, bool]:
    for model, label in models:
        print(f"QUEUE {provider}: START {label} ({model})", flush=True)
        cmd = common + [
            "--provider", provider, "--model", model, "--model-label", label,
        ]
        result = subprocess.run(cmd, cwd=HERE)
        if result.returncode:
            print(f"QUEUE {provider}: FAILED {label} rc={result.returncode}", flush=True)
            return provider, False
        print(f"QUEUE {provider}: COMPLETE {label}", flush=True)
    return provider, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", required=True,
                        help="retrieval ledger stamp produced by run_recipebench.py")
    args = parser.parse_args()
    common = [
        sys.executable, "run_cloud.py", "--stamp", args.stamp,
        "--arms", "meko_optimized,bm25_recipe,full_history",
        "--per-type", "999", "--batch-size", "5",
        "--timeout", "900", "--attempts", "3",
    ]
    ok = True
    with ThreadPoolExecutor(max_workers=len(QUEUES)) as pool:
        jobs = [pool.submit(run_queue, provider, models, common)
                for provider, models in QUEUES.items()]
        for job in as_completed(jobs):
            provider, passed = job.result()
            print(f"PROVIDER {provider}: {'COMPLETE' if passed else 'FAILED'}", flush=True)
            ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
