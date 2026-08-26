#!/usr/bin/env python3
"""Answering matrix: every client answers every question from every arm.

Design constraints, and what each one prevents (see
../../METHODOLOGY_LARGE_CONTEXT.md sections 10-12):

* **One question per call.** Batched questions correlate errors inside a single
  completion — one format slip loses the whole batch — which both understates
  models and breaks the independence a paired test assumes.
* **Isolation.** Each call runs in a fresh empty directory outside this
  repository, with `XC1_*` and `MEKO_*` stripped from the environment, so
  neither the corpus path nor the Meko credential is reachable from the
  answering process.
* **Canary probes.** Isolation is not proof: several clients in the roster run
  with sandbox-bypass flags and could read the corpus off disk. Canary facts
  are planted in the history, never appear in any context or digest, and are
  asked with an empty context. A correct canary answer means the client read
  the corpus rather than its context, and the run is marked `leak_detected`.
* **Ceilings first.** Every arm's ceiling — whether the gold answer is present
  in what that arm hands the reader — is computed before any call and written
  into the summary next to the accuracy, so no result can be read without it.
* **Repeats.** `--repeats N` re-asks every question, giving a spread per cell
  instead of a single point with no variance estimate.

Arms:
  own_digest    the client's own compaction of the history
  meko          cached Meko retrieval  (retrieve_meko.py)
  bm25          cached keyword retrieval, the control  (retrieve_bm25.py)
  shared_store  memories retrieved from a store another writer populated.
                Named for what it measures. It is NOT a cross-agent claim
                unless the ingest was performed by a different client; pass
                --shared-writer to record which writer produced it.
  full_history  the whole history in the prompt, no retrieval and no summary —
                the no-memory baseline, and the only arm whose ceiling is 1.0
                by construction. Meaningful only while the history fits the
                reader's window, so the runner refuses it above
                --full-history-max-chars rather than truncating quietly and
                reporting a baseline it did not run.
  fast_packet   evidence selected by deterministic local BM25, persisted as a
                verified Meko artifact, then fetched by hash. This measures
                prepared-record delivery through Meko, not semantic retrieval.

Run:  python3 run_answers.py <stamp> --labels opus5,glm --arms meko,bm25,own_digest
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from clients import ROSTER, call_label, isolated_env
from scoring import gold_in_context, score_answer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = Path(os.environ.get("XC1_RUNS", ROOT / "runs-live" / "answers"))
FIX = Path(os.environ.get("XC1_FIXTURES", ROOT / "generated" / "fast-reader"))

PROMPT = (
    "Answer the question using ONLY the notes below. Do not use any other "
    "source, and do not read any files.\n"
    "Reply with the value or values only, no explanation and no restatement "
    "of the question. If a question asks for several values, separate them "
    "with spaces in the order requested. If the notes do not contain the "
    "answer, reply exactly: NOT FOUND\n\n"
    "NOTES:\n{ctx}\n\nQUESTION: {q}\nANSWER:")


def load_arm_contexts(stamp: str, arm: str) -> dict[int, str]:
    """Context per question index for one arm."""
    if arm in ("own_digest", "full_history"):
        return {}  # filled per label / read from the corpus
    fname = {"meko": f"meko_ctx-{stamp}.json",
             "meko_decomposed": f"meko_ctx-{stamp}-decomposed.json",
             "bm25": f"bm25_ctx-{stamp}.json",
             "bm25_decomposed": f"bm25_ctx-{stamp}-decomposed.json",
             "shared_store": f"shared_ctx-{stamp}.json",
             "fast_packet": f"fast_packet_ctx-{stamp}.json"}[arm]
    path = FIX / fname
    if not path.exists():
        raise SystemExit(f"missing {path} — run the retrieval stage for {arm} first")
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def ceiling_of(questions: list[dict], ctx_for) -> dict:
    by_type: dict[str, list[int]] = {}
    for i, q in enumerate(questions):
        present = gold_in_context(q, ctx_for(i))
        if present is None:
            continue
        hit, total = by_type.setdefault(q["type"], [0, 0])
        by_type[q["type"]] = [hit + int(present), total + 1]
    hits = sum(v[0] for v in by_type.values())
    total = sum(v[1] for v in by_type.values())
    return {"ceiling": round(hits / total, 4) if total else None,
            "by_type": {t: f"{v[0]}/{v[1]}" for t, v in sorted(by_type.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stamp")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--arms", default="meko,bm25,own_digest")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--canary-probes", type=int, default=4)
    ap.add_argument("--full-history-max-chars", type=int, default=400_000,
                    help="refuse the full_history arm above this corpus size, "
                         "where it stops being an arm and becomes a truncation")
    ap.add_argument("--shared-writer", default="",
                    help="client label that populated the shared_store arm")
    args = ap.parse_args()

    RUNS.mkdir(exist_ok=True)
    questions = json.loads((FIX / "pol_questions.json").read_text())
    if args.limit:
        questions = questions[:args.limit]
    canaries = json.loads((FIX / "pol_canaries.json").read_text())
    manifest_path = FIX / "pol_manifest.json"
    corpus = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    for label in labels:
        if label not in ROSTER:
            raise SystemExit(f"unknown label {label}")

    contexts = {arm: load_arm_contexts(args.stamp, arm)
                for arm in arms if arm not in ("own_digest", "full_history")}

    full_history = ""
    if "full_history" in arms:
        full_history = (FIX / "pol_history_1m.txt").read_text()
        if len(full_history) > args.full_history_max_chars:
            raise SystemExit(
                f"full_history arm refused: the corpus is {len(full_history):,} "
                f"chars, above --full-history-max-chars "
                f"({args.full_history_max_chars:,}). Pasting a history that does "
                f"not fit measures truncation, not the no-memory baseline.")

    ledger_path = RUNS / f"answers-{args.stamp}.jsonl"
    done: set[tuple] = set()
    if ledger_path.exists():
        for line in ledger_path.open():
            rec = json.loads(line)
            if rec.get("event") == "answer" and rec.get("ok"):
                done.add((rec["label"], rec["arm"], rec["qidx"], rec["repeat"]))
    print(f"{len(done)} answers already in the ledger", flush=True)

    summaries: list[dict] = []
    with ledger_path.open("a") as lf:
        for label in labels:
            digest = ""
            digest_path = FIX / f"digest-{args.stamp}-{label}.txt"
            if "own_digest" in arms:
                if not digest_path.exists():
                    print(f"  {label}: no digest, skipping own_digest", flush=True)
                else:
                    digest = digest_path.read_text()

            for arm in arms:
                if arm == "own_digest" and not digest:
                    continue
                if arm == "full_history":
                    ctx_for = (lambda i: full_history)
                elif arm == "own_digest":
                    ctx_for = (lambda i: digest)
                else:
                    ctx_for = (lambda i, _a=arm: contexts[_a].get(i, ""))
                ceil = ceiling_of(questions, ctx_for)
                rows: list[dict] = []
                usage_by_basis: dict[str, dict] = {}
                t0 = time.monotonic()

                # Canary probes: empty context, answer only obtainable off disk.
                leaks = []
                for c in canaries[:args.canary_probes]:
                    with tempfile.TemporaryDirectory(prefix="xc1-canary-") as wd:
                        res = call_label(label, PROMPT.format(ctx="", q=c["question"]),
                                         timeout=args.timeout, cwd=wd,
                                         env=isolated_env(), workdir=wd)
                    leaked = res["ok"] and score_answer(c, res["text"])
                    leaks.append(leaked)
                    lf.write(json.dumps({"event": "canary", "label": label,
                                         "arm": arm, "question": c["question"],
                                         "ok": res["ok"], "leaked": leaked,
                                         "text": res["text"][:300]}) + "\n")
                    lf.flush()
                leak_detected = any(leaks)
                if leak_detected:
                    print(f"  {label}/{arm}: LEAK DETECTED — answered a canary "
                          f"with an empty context; results for this cell are void",
                          flush=True)

                for repeat in range(args.repeats):
                    for qidx, q in enumerate(questions):
                        key = (label, arm, qidx, repeat)
                        if key in done:
                            continue
                        with tempfile.TemporaryDirectory(prefix="xc1-answer-") as wd:
                            res = call_label(
                                label,
                                PROMPT.format(ctx=ctx_for(qidx), q=q["question"]),
                                timeout=args.timeout, cwd=wd,
                                env=isolated_env(), workdir=wd)
                        correct = bool(res["ok"] and score_answer(q, res["text"]))
                        rec = {"event": "answer", "label": label, "arm": arm,
                               "qidx": qidx, "repeat": repeat, "type": q["type"],
                               "ok": res["ok"], "correct": correct,
                               "text": res["text"][:400],
                               "usage": res["usage"],
                               "latency_s": res["latency_s"]}
                        if not res["ok"]:
                            rec["error"] = res.get("error", "")[:300]
                        lf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        lf.flush()
                        rows.append(rec)
                        b = res["usage"]["basis"]
                        acc = usage_by_basis.setdefault(
                            b, {"calls": 0, "billed_input_tokens": 0,
                                "output_tokens": 0, "cost_usd": 0.0})
                        acc["calls"] += 1
                        acc["billed_input_tokens"] += res["usage"]["billed_input_tokens"]
                        acc["output_tokens"] += res["usage"]["output_tokens"]
                        acc["cost_usd"] = round(
                            acc["cost_usd"] + res["usage"]["cost_usd"], 6)
                        if len(rows) % 40 == 0:
                            ok_n = sum(r["correct"] for r in rows)
                            print(f"  {label}/{arm}: {len(rows)} asked, "
                                  f"{ok_n} correct, "
                                  f"{time.monotonic() - t0:.0f}s", flush=True)

                scored = [r for r in rows if r["ok"]]
                by_type: dict[str, list[int]] = {}
                for r in scored:
                    hit, total = by_type.setdefault(r["type"], [0, 0])
                    by_type[r["type"]] = [hit + int(r["correct"]), total + 1]
                summary = {
                    "stamp": args.stamp, "label": label, "model": ROSTER[label][1],
                    "arm": arm, "repeats": args.repeats,
                    "questions": len(questions),
                    "asked": len(rows), "failed_calls": len(rows) - len(scored),
                    "correct": sum(r["correct"] for r in scored),
                    "accuracy": (round(sum(r["correct"] for r in scored) / len(scored), 4)
                                 if scored else None),
                    "by_type": {t: f"{v[0]}/{v[1]}" for t, v in sorted(by_type.items())},
                    "arm_ceiling": ceil["ceiling"],
                    "arm_ceiling_by_type": ceil["by_type"],
                    "leak_detected": leak_detected,
                    "canary_probes": len(leaks),
                    "usage_by_basis": usage_by_basis,
                    "wall_s": round(time.monotonic() - t0, 1),
                    "shared_writer": args.shared_writer or None,
                    "corpus_seed": corpus.get("seed"),
                    "corpus_chars": corpus.get("chars"),
                }
                summaries.append(summary)
                lf.write(json.dumps({"event": "cell_summary", **summary}) + "\n")
                lf.flush()
                print(json.dumps({k: summary[k] for k in
                                  ("label", "arm", "accuracy", "arm_ceiling",
                                   "leak_detected")}), flush=True)

    out = RUNS / f"answers-summary-{args.stamp}.json"
    prior = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(prior + summaries, indent=2))
    print(f"\n{len(summaries)} cells written to {out}", flush=True)
    print("Accuracy is uninterpretable without arm_ceiling — publish both.",
          flush=True)


if __name__ == "__main__":
    main()
