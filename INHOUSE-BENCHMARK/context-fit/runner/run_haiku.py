#!/usr/bin/env python3
"""Stratified RecipeBench validation through the normal Haiku CLI harness.

Uses the exact contexts saved by run_recipebench.py. Questions are batched to
avoid paying the Claude Code harness/system prefix once per MCQ.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs-live"
sys.path.insert(0, str(HERE))
from run_recipebench import (contexts, load_fixture, metadata_index,
                             read_jsonl, retrieval_rows)  # noqa: E402

DEFAULT_ARMS = "meko_raw25,meko_recipe,bm25_recipe,full_history"


def sample_rows(rows: list[dict], per_type: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_type = {}
    for row in rows:
        by_type.setdefault(row["type"], []).append(row)
    chosen = []
    for kind in sorted(by_type):
        chosen.extend(rng.sample(by_type[kind], min(per_type, len(by_type[kind]))))
    return sorted(chosen, key=lambda x: x["qid"])


def batch_prompt(items: list[tuple[dict, dict]]) -> str:
    blocks = []
    for row, ctx in items:
        q = row["question_row"]
        blocks.append(f"""ID: {q['qid']}
Session notes:
{ctx['text']}
Question: {q['question']}
A) {q['options'][0]}
B) {q['options'][1]}
C) {q['options'][2]}
D) {q['options'][3]}""")
    return """Answer every multiple-choice item using only its session notes.
Later dated statements replace earlier conflicting statements. Return exactly
one JSON object mapping each ID to one letter. No explanation or markdown.

""" + "\n\n---\n\n".join(blocks)


def parse_answers(text: str, qids: list[str]) -> dict[str, str]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    payload = json.loads(match.group(0))
    out = {qid: str(payload.get(qid, "?")).strip().upper()[:1] for qid in qids}
    if any(value not in "ABCD" for value in out.values()):
        raise ValueError(f"invalid answer mapping: {out}")
    return out


def call_haiku(prompt: str, model: str) -> dict:
    cmd = ["claude", "-p", prompt, "--model", model,
           "--output-format", "json", "--max-turns", "1", "--tools", ""]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=300)
    if proc.returncode:
        raise RuntimeError(f"claude exit {proc.returncode}: {proc.stderr[-500:]}")
    payload = json.loads(proc.stdout)
    if payload.get("is_error"):
        raise RuntimeError(str(payload)[:500])
    return {"result": payload.get("result", ""),
            "usage": payload.get("usage", {}),
            "cost_usd": payload.get("total_cost_usd", 0),
            "latency_s": round(time.monotonic() - t0, 3),
            "session_id": payload.get("session_id")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--model-label", default="haiku")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()

    rows = retrieval_rows(args.stamp)
    if len(rows) != 320:
        raise SystemExit(f"retrieval incomplete: {len(rows)}/320")
    selected = sample_rows(rows, args.per_type, args.seed)
    statements, _ = load_fixture()
    by_persona = {}
    for row in statements:
        by_persona.setdefault(row["persona"], []).append(row["text"])
    meta = metadata_index(statements)
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    ledger = RUNS / f"haiku-{args.stamp}-{args.model_label}.jsonl"
    done = set()
    if ledger.exists():
        done = {(x["arm"], x["batch"]) for x in read_jsonl(ledger)
                if x.get("event") == "batch" and x.get("ok")}
    out = open(ledger, "a")
    batch_no = 0
    for arm in arms:
        for start in range(0, len(selected), args.batch_size):
            batch_no += 1
            if (arm, batch_no) in done:
                continue
            subset = selected[start:start + args.batch_size]
            items = [(row, contexts(row, by_persona, meta)[arm]) for row in subset]
            prompt = batch_prompt(items)
            errors = []
            record = None
            for attempt in range(1, 4):
                try:
                    call = call_haiku(prompt, args.model)
                    answers = parse_answers(call["result"], [r["qid"] for r in subset])
                    record = {"event": "batch", "arm": arm, "batch": batch_no,
                        "ok": True, "attempts": attempt, "errors": errors,
                        "model": args.model, "qids": [r["qid"] for r in subset],
                        "answers": answers, **call}
                    break
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {str(exc)[:300]}")
                    time.sleep(2 * attempt)
            if record is None:
                record = {"event": "batch", "arm": arm, "batch": batch_no,
                          "ok": False, "attempts": 3, "errors": errors,
                          "model": args.model, "qids": [r["qid"] for r in subset]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  haiku batch {batch_no}: {arm} {record['ok']}", flush=True)
            if not record["ok"]:
                raise SystemExit("Haiku batch failed; resume after inspection")
    out.close()
    summarize(args.stamp, args.model_label, selected)


def summarize(stamp: str, label: str, selected: list[dict]) -> dict:
    batches = [x for x in read_jsonl(RUNS / f"haiku-{stamp}-{label}.jsonl")
               if x.get("event") == "batch" and x.get("ok")]
    gold = {r["qid"]: r["question_row"] for r in selected}
    summary = {"stamp": stamp, "model_label": label, "questions": len(gold),
               "cost_usd": round(sum(x.get("cost_usd", 0) for x in batches), 4),
               "arms": {}}
    for arm in sorted(set(x["arm"] for x in batches)):
        answers = {}
        relevant = [x for x in batches if x["arm"] == arm]
        for batch in relevant:
            answers.update(batch["answers"])
        valid = sorted(set(answers) & set(gold))
        correct = sum(answers[q] == gold[q]["answer"] for q in valid)
        summary["arms"][arm] = {"answered": len(valid), "correct": correct,
            "accuracy": round(correct / len(valid), 4) if valid else None,
            "by_type": {kind: f"{sum(answers[q]==gold[q]['answer']
                                    for q in valid if gold[q]['type']==kind)}/"
                                  f"{sum(gold[q]['type']==kind for q in valid)}"
                        for kind in sorted(set(gold[q]["type"] for q in valid))},
            "cost_usd": round(sum(x.get("cost_usd", 0) for x in relevant), 4),
            "latency_s": round(sum(x.get("latency_s", 0) for x in relevant), 2)}
    path = RUNS / f"haiku-summary-{stamp}-{label}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
