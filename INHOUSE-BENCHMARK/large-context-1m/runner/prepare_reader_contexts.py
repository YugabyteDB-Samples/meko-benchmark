#!/usr/bin/env python3
"""Cap fast packet contexts to each reader's own compaction budget."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "fixtures"
GENERATED = ROOT / "generated"
FIX = GENERATED / "fast-reader"
BASELINE_FIX = GENERATED / "baseline-reader"
STAMP = "xc1-c500-fast"

DIGEST_LABELS = {
    "gpt-5.5": "codex-55",
    "gpt-5.6-luna": "codex-luna",
    "gpt-5.6-sol": "codex-sol",
    "gpt-5.6-terra": "codex-terra",
    "gemini-3.7-flash": "gemini-37",
    "glm-5.2": "glm",
    "opus-5": "opus5",
    "sonnet-5": "sonnet5",
}


def main() -> None:
    FIX.mkdir(parents=True, exist_ok=True)
    BASELINE_FIX.mkdir(parents=True, exist_ok=True)
    questions = (SOURCE / "questions.json").read_bytes()
    (FIX / "pol_questions.json").write_bytes(questions)
    (FIX / "pol_canaries.json").write_text("[]\n")
    (BASELINE_FIX / "pol_questions.json").write_bytes(questions)
    (BASELINE_FIX / "pol_canaries.json").write_text("[]\n")
    contexts = json.loads(
        (GENERATED / f"meko_fast_ctx-{STAMP}.json").read_text())
    rows = []
    for digest_path in sorted((ROOT / "digests").glob("*.txt")):
        label = DIGEST_LABELS[digest_path.stem]
        cap = len(digest_path.read_text())
        (BASELINE_FIX / f"digest-20260812c-{label}.txt").write_bytes(
            digest_path.read_bytes())
        capped = {key: value[:cap] for key, value in contexts.items()}
        stamp = f"{STAMP}-{label}"
        output = FIX / f"fast_packet_ctx-{stamp}.json"
        output.write_text(json.dumps(capped))
        rows.append({"label": label, "stamp": stamp, "context_chars_cap": cap,
                     "digest_sha256": hashlib.sha256(
                         digest_path.read_bytes()).hexdigest(),
                     "context_file": str(output.relative_to(ROOT))})
    manifest = FIX / f"reader-context-manifest-{STAMP}.json"
    manifest.write_text(json.dumps({
        "source_context": str((GENERATED /
                              f"meko_fast_ctx-{STAMP}.json").relative_to(ROOT)),
        "questions_sha256": hashlib.sha256(questions).hexdigest(),
        "rule": "prefix capped to that reader's own compaction digest chars",
        "readers": rows,
    }, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
