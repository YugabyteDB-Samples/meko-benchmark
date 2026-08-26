#!/usr/bin/env python3
"""Stage 1 — each coding client compacts the 1M-token history ITSELF.

The edge-context experiment built one digest with Claude and handed the same
digest to every reader. That measures one compactor. Here each client compacts
with its own model, which is what actually happens when a coding agent hits
context overflow, and it is the only way "different clients try to compact and
fail" is a claim about the clients rather than about Claude.

The history is split into chunks that fit the smallest window in the roster,
each chunk is summarized fact-preservingly, and the summaries concatenate into
that client's digest. Cost, tokens, and wall time of the compaction itself are
recorded, because that overhead is the hidden tax the retrieval arm avoids.

Truncation guard: a client that silently drops half the chunk would look like a
great compactor and a terrible reader. Every call records the tokens the
provider says it received, so under-reading shows up instead of hiding.

Run:  python run_compact_own.py <stamp> --labels glm,kimi-code,opus5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from clients import ROSTER, call_label

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "runs-live"
FIX = ROOT / "generated"
HISTORY = ROOT / "fixtures" / "history.txt"

# 200k chars (~50k tokens) per chunk. Measured ceiling: the Ollama cloud
# endpoint serves 200k-char prompts intact (48.7k of ~50k expected tokens seen)
# and returns 502 at 400k, so this is the largest chunk the whole roster can
# take. Uniform across clients so no client compacts in fewer passes than another.
CHUNK_CHARS = 200_000

# The output budget is the point of compaction, not a detail. Unconstrained,
# these models transcribe rather than compact: GLM returned 20,300 output
# tokens per chunk and MiniMax 60,731, which across 20 chunks is a "digest"
# larger than the window it was supposed to fit. Capping each chunk summary
# keeps the concatenated digest inside a single context window, which is the
# whole premise of the compaction arm.
WORDS_PER_CHUNK = 1800

def summarize_prompt(words_per_chunk: int) -> str:
    return (
        "Summarize these notes for later question-answering. CRITICAL: preserve "
        "every specific fact, identifier, number, code, name, and value EXACTLY "
        "as written. Each fact belongs to a named person — always keep the "
        "person's name attached to their value. If a value changed over time, "
        "keep every value in the order it appeared and mark which one is current. "
        "Drop generic filler chatter. Output dense factual bullet lines and "
        f"nothing else, in at most {words_per_chunk} words. You must stay within "
        "that budget, so if you cannot keep everything, keep the specific "
        "identifiers and values and drop prose.\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stamp")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS)
    ap.add_argument("--words-per-chunk", type=int, default=WORDS_PER_CHUNK)
    ap.add_argument("--history", type=Path, default=HISTORY)
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()

    RUNS.mkdir(exist_ok=True)
    FIX.mkdir(exist_ok=True)
    history = args.history.read_text()
    chunks = [history[i:i + args.chunk_chars]
              for i in range(0, len(history), args.chunk_chars)]
    print(f"history {len(history):,} chars -> {len(chunks)} chunks "
          f"of {args.chunk_chars:,} (~{args.chunk_chars // 4:,} tokens)", flush=True)

    for label in [x.strip() for x in args.labels.split(",") if x.strip()]:
        if label not in ROSTER:
            raise SystemExit(f"unknown label {label}")
        ledger = RUNS / f"compact-{args.stamp}-{label}.jsonl"
        done = {}
        if ledger.exists():
            for line in ledger.open():
                rec = json.loads(line)
                if rec.get("event") == "chunk" and rec.get("ok"):
                    done[rec["chunk"]] = rec["summary"]

        t0 = time.monotonic()
        with ledger.open("a") as lf:
            for i, chunk in enumerate(chunks):
                if i in done:
                    continue
                res = call_label(label, summarize_prompt(args.words_per_chunk) + chunk,
                                 timeout=args.timeout,
                                 workdir="/tmp/junie-probe")
                rec = {"event": "chunk", "chunk": i, "ok": res["ok"],
                       "in_chars": len(chunk),
                       "expected_in_tokens": len(chunk) // 4,
                       "usage": res["usage"], "latency_s": res["latency_s"]}
                if res["ok"]:
                    rec["summary"] = res["text"]
                    rec["out_chars"] = len(res["text"])
                    done[i] = res["text"]
                else:
                    rec["error"] = res.get("error", "")[:300]
                lf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                lf.flush()
                got = res["usage"]["billed_input_tokens"]
                print(f"  {label} chunk {i + 1}/{len(chunks)}: "
                      f"{'ok' if res['ok'] else 'FAIL'} "
                      f"{len(chunk):,}->{len(res['text']):,} chars, "
                      f"provider saw {got:,} tok "
                      f"(expected ~{len(chunk) // 4:,}), "
                      f"{res['latency_s']:.0f}s", flush=True)

        if len(done) != len(chunks):
            print(f"  {label}: INCOMPLETE {len(done)}/{len(chunks)} chunks, "
                  f"no digest written", flush=True)
            continue

        digest = "\n".join(done[i] for i in sorted(done))
        (FIX / f"digest-{args.stamp}-{label}.txt").write_text(digest)
        records = [json.loads(l) for l in ledger.open()
                   if json.loads(l).get("event") == "chunk"]
        ok_recs = [r for r in records if r.get("ok")]
        summary = {
            "label": label, "model": ROSTER[label][1], "chunks": len(chunks),
            "digest_chars": len(digest), "digest_tokens_est": len(digest) // 4,
            "compaction_cost_usd": round(
                sum(r["usage"].get("cost_usd", 0) for r in ok_recs), 4),
            "compaction_input_tokens": sum(
                r["usage"].get("billed_input_tokens", 0) for r in ok_recs),
            "compaction_output_tokens": sum(
                r["usage"].get("output_tokens", 0) for r in ok_recs),
            "expected_input_tokens": sum(r["expected_in_tokens"] for r in records
                                         if r.get("ok")),
            "usage_basis": ok_recs[0]["usage"]["basis"] if ok_recs else "none",
            "wall_s": round(time.monotonic() - t0, 1),
        }
        (RUNS / f"compact-summary-{args.stamp}-{label}.json").write_text(
            json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
