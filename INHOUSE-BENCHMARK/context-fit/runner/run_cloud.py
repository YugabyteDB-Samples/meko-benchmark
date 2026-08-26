#!/usr/bin/env python3
"""Cross-reader RecipeBench validation on cloud models.

Reuses the EXACT cached retrieval contexts and the Haiku harness's sampling,
batching, and JSON parsing. Only the reader CLI changes. This is the clean
cross-reader test: does meko_optimized keep winning when a different model
reads the identical evidence?

Providers
  claude   Claude Code CLI          (real usage + real cost_usd)
  agy      Antigravity CLI          (real usage; Gemini flash / others)
  ollama   Ollama cloud /api/chat   (real prompt_eval_count / eval_count)
  codex    Codex CLI                (no usage reported)

Every provider records normalized token usage so tokens/question and
tokens/correct are comparable across readers. Batches that never parse are
recorded as failures and disclosed in the summary instead of silently
shrinking the denominator.

Run:  python run_cloud.py --stamp run1 --provider ollama \
        --model gpt-oss:20b-cloud --model-label gptoss20 \
        --arms meko_raw10,meko_optimized,meko_recipe,full_history --per-type 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs-live"
sys.path.insert(0, str(HERE))
from run_haiku import batch_prompt, sample_rows  # noqa: E402
from run_recipebench import (contexts, load_fixture, metadata_index,  # noqa: E402
                             read_jsonl, retrieval_rows)

CLAUDE = os.environ.get("XC1_CLAUDE", "claude")
AGY = os.environ.get("XC1_AGY", "agy")
CODEX = os.environ.get("XC1_CODEX", "codex")
# Agentic CLIs (codex with sandbox bypass, agy with skipped permissions) will
# read answer-bearing files from the working directory when a question's
# context is thin — demonstrated by an empty-context canary probe: 5/5 and
# 4/5 fixture-only answers recovered from the benchmark cwd, 0/10 from an
# empty one. Every agentic call therefore runs in an isolated empty workdir.
ISOLATED_WORKDIR = "/tmp/rc-isolated-workdir"
os.makedirs(ISOLATED_WORKDIR, exist_ok=True)
OLLAMA_CHAT = "http://127.0.0.1:11434/api/chat"


def _usage(input_tokens=0, output_tokens=0, cache_read=0, cache_write=0,
           source="none") -> dict:
    """Normalized usage. `billed_input` is what the provider charged for."""
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_read_tokens": cache_read, "cache_write_tokens": cache_write,
            "billed_input_tokens": input_tokens + cache_read + cache_write,
            "total_tokens": input_tokens + cache_read + cache_write + output_tokens,
            "usage_source": source}


def extract_json_object(text: str) -> str | None:
    """First balanced JSON object in the text.

    The greedy brace-to-brace regex this replaces swallowed everything between the first
    and the LAST brace, so a valid object followed by any trailing text that
    contained a brace became unparseable "extra data" — a parser-invented
    failure on a well-formed answer.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_partial(text: str, qids: list[str]) -> tuple[dict[str, str], list[str]]:
    """Salvage every legible answer; never let format rigidity invent a failure.

    Accepts the JSON mapping the prompt asks for, and falls back to bare
    `QID: letter` lines — a reader that answers every question in a perfectly
    unambiguous line format has answered, whether or not it wrapped the
    mapping in braces.
    """
    payload = {}
    obj = extract_json_object(text)
    if obj:
        payload = json.loads(obj)
    if not payload:
        for qid in qids:
            m = re.search(rf"{re.escape(qid)}\s*[:=-]\s*([A-Da-d])\b", text)
            if m:
                payload[qid] = m.group(1)
    answers, missing = {}, []
    for qid in qids:
        letter = str(payload.get(qid, "")).strip().upper()[:1]
        if letter in {"A", "B", "C", "D"}:
            answers[qid] = letter
        else:
            missing.append(qid)
    if not answers:
        raise ValueError(f"no parseable answers: {text[:200]}")
    return answers, missing


def call(provider: str, model: str, prompt: str, timeout: int, seed: int) -> dict:
    t0 = time.monotonic()

    if provider == "claude":
        cmd = [CLAUDE, "-p", prompt, "--model", model,
               "--output-format", "json", "--max-turns", "1", "--tools", ""]
        proc = subprocess.run(cmd, text=True, capture_output=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
        if proc.returncode:
            raise RuntimeError(f"claude {proc.returncode}: {proc.stderr[-300:]}")
        payload = json.loads(proc.stdout)
        if payload.get("is_error"):
            raise RuntimeError(str(payload)[:300])
        u = payload.get("usage", {})
        return {"result": payload.get("result", ""),
                "cost_usd": payload.get("total_cost_usd", 0),
                "usage": _usage(u.get("input_tokens", 0), u.get("output_tokens", 0),
                                u.get("cache_read_input_tokens", 0),
                                u.get("cache_creation_input_tokens", 0), "cli_json"),
                "latency_s": round(time.monotonic() - t0, 2)}

    if provider == "agy":
        cmd = [AGY, "--dangerously-skip-permissions", "--new-project",
               "--output-format", "json", "--model", model, "--print", prompt]
        proc = subprocess.run(cmd, text=True, capture_output=True,
                              timeout=timeout, stdin=subprocess.DEVNULL,
                              cwd=ISOLATED_WORKDIR)
        if proc.returncode:
            raise RuntimeError(f"agy {proc.returncode}: {proc.stderr[-300:]}")
        payload = json.loads(proc.stdout)
        if payload.get("status") != "SUCCESS":
            raise RuntimeError(f"agy status {payload.get('status')}: "
                               f"{str(payload)[:300]}")
        u = payload.get("usage", {})
        return {"result": payload.get("response", ""), "cost_usd": 0,
                "usage": _usage(u.get("input_tokens", 0),
                                u.get("output_tokens", 0) + u.get("thinking_tokens", 0),
                                u.get("cache_read_tokens", 0), 0, "cli_json"),
                "latency_s": round(time.monotonic() - t0, 2)}

    if provider == "ollama":
        body = json.dumps({"model": model, "stream": False, "think": False,
                           "messages": [{"role": "user", "content": prompt}],
                           "options": {"temperature": 0, "seed": seed,
                                       "num_ctx": 65536}}).encode()
        req = urllib.request.Request(OLLAMA_CHAT, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        text = ((payload.get("message") or {}).get("content") or "").strip()
        if not text:
            raise RuntimeError(f"empty response: {str(payload)[:300]}")
        return {"result": text, "cost_usd": 0,
                "usage": _usage(payload.get("prompt_eval_count", 0),
                                payload.get("eval_count", 0), 0, 0, "api"),
                "latency_s": round(time.monotonic() - t0, 2)}

    if provider == "codex":
        cmd = [CODEX, "exec", "--skip-git-repo-check",
               "--dangerously-bypass-approvals-and-sandbox", "--model", model,
               prompt]
        proc = subprocess.run(cmd, text=True, capture_output=True,
                              timeout=timeout, stdin=subprocess.DEVNULL,
                              cwd=ISOLATED_WORKDIR)
        if proc.returncode:
            raise RuntimeError(f"codex {proc.returncode}: {proc.stderr[-300:]}")
        # Codex reports its token count on stderr ("tokens used: N"), input
        # and output not separated — the cli_text basis. Scrape it rather
        # than publishing the reader with no usage at all.
        m = re.search(r"tokens used:?\s*([\d,]+)", proc.stderr)
        toks = int(m.group(1).replace(",", "")) if m else 0
        return {"result": proc.stdout, "cost_usd": 0,
                "usage": _usage(toks, 0, 0, 0, "cli_text" if toks else "none"),
                "latency_s": round(time.monotonic() - t0, 2)}

    raise ValueError(provider)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--provider", required=True,
                    choices=["claude", "codex", "agy", "ollama"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--arms", default="meko_raw10,meko_optimized,meko_recipe,full_history")
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--attempts", type=int, default=3)
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

    # Batch numbers only mean something under the sampling config that built
    # them. Resuming a ledger written with a different per-type/batch-size/seed
    # would silently reuse answers for a different set of questions.
    config = {"per_type": args.per_type, "batch_size": args.batch_size,
              "seed": args.seed, "model": args.model, "provider": args.provider}
    ledger = RUNS / f"haiku-{args.stamp}-{args.model_label}.jsonl"
    done = set()
    if ledger.exists():
        prior = read_jsonl(ledger)
        for rec in prior:
            if rec.get("event") == "config" and rec["config"] != config:
                raise SystemExit(
                    f"ledger {ledger.name} was written with {rec['config']}, "
                    f"refusing to resume under {config}; use a new --model-label")
        done = {(x["arm"], x["batch"]) for x in prior
                if x.get("event") == "batch" and x.get("ok")}

    with ledger.open("a") as lf:
        lf.write(json.dumps({"event": "config", "config": config}) + "\n")
        for arm in arms:
            for bi, start in enumerate(range(0, len(selected), args.batch_size)):
                if (arm, bi) in done:
                    continue
                subset = selected[start:start + args.batch_size]
                items = [(r, contexts(r, by_persona, meta)[arm]) for r in subset]
                prompt = batch_prompt(items)
                qids = [r["qid"] for r in subset]
                gold = {r["qid"]: r["question_row"]["answer"] for r in subset}
                rec = {"event": "batch", "arm": arm, "batch": bi,
                       "provider": args.provider, "model": args.model,
                       "qids": qids, "prompt_chars": len(prompt),
                       "context_chars": sum(c["chars"] for _, c in items),
                       "context_documents": sum(c["documents"] for _, c in items)}
                errors, res = [], None
                for attempt in range(1, args.attempts + 1):
                    try:
                        res = call(args.provider, args.model, prompt,
                                   args.timeout, args.seed)
                        ans, missing = parse_partial(res["result"], qids)
                        # A complete batch wins; an incomplete one is retried
                        # and only accepted, partially, on the last attempt.
                        if missing and attempt < args.attempts:
                            errors.append(f"IncompleteMapping: missing {missing}")
                            res = None
                            time.sleep(3 * attempt)
                            continue
                        rec.update(ok=True, attempts=attempt, errors=errors,
                                   answers=ans, partial=bool(missing),
                                   missing_qids=missing,
                                   correct=sum(ans[q] == gold[q] for q in ans),
                                   n=len(qids), cost_usd=res.get("cost_usd", 0),
                                   usage=res["usage"], latency_s=res["latency_s"])
                        break
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"{type(e).__name__}: {str(e)[:300]}")
                        res = None
                        time.sleep(3 * attempt)
                if res is None:
                    rec.update(ok=False, attempts=args.attempts, errors=errors,
                               n=len(qids))
                lf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                lf.flush()
                print(f"{args.model_label} {arm} b{bi}: "
                      f"{'ok' if rec.get('ok') else 'FAIL'} "
                      f"(attempt {rec.get('attempts')})", flush=True)

    summarize_cloud(args.stamp, args.model_label, selected, args.provider,
                    args.model)


def summarize_cloud(stamp: str, label: str, selected: list[dict],
                    provider: str = "", model: str = "") -> dict:
    """Accuracy + full token accounting, with failed batches disclosed."""
    records = [x for x in read_jsonl(RUNS / f"haiku-{stamp}-{label}.jsonl")
               if x.get("event") == "batch"]
    gold = {r["qid"]: r["question_row"] for r in selected}
    summary = {"stamp": stamp, "model_label": label, "provider": provider,
               "model": model, "questions": len(gold),
               "cost_usd": round(sum(x.get("cost_usd", 0) for x in records
                                     if x.get("ok")), 4),
               "arms": {}}
    for arm in sorted({x["arm"] for x in records}):
        rel = [x for x in records if x["arm"] == arm]
        ok = [x for x in rel if x.get("ok")]
        failed = [x for x in rel if not x.get("ok")]
        # A batch retried successfully later must not count as a failure.
        failed = [x for x in failed
                  if x["batch"] not in {y["batch"] for y in ok}]
        answers = {}
        for batch in ok:
            answers.update(batch["answers"])
        valid = sorted(set(answers) & set(gold))
        correct = sum(answers[q] == gold[q]["answer"] for q in valid)
        usage = [x.get("usage", {}) for x in ok]
        billed_in = sum(u.get("billed_input_tokens", 0) for u in usage)
        out_tok = sum(u.get("output_tokens", 0) for u in usage)
        total = billed_in + out_tok
        summary["arms"][arm] = {
            "answered": len(valid), "attempted": len(gold), "correct": correct,
            "accuracy": round(correct / len(valid), 4) if valid else None,
            "by_type": {kind: f"{sum(answers[q]==gold[q]['answer']
                                    for q in valid if gold[q]['type']==kind)}/"
                              f"{sum(gold[q]['type']==kind for q in valid)}"
                        for kind in sorted({gold[q]["type"] for q in valid})},
            "failed_batches": len(failed),
            "partial_batches": sum(1 for x in ok if x.get("partial")),
            "unanswered_qids": sorted(set(gold) - set(valid)),
            "cost_usd": round(sum(x.get("cost_usd", 0) for x in ok), 4),
            "latency_s": round(sum(x.get("latency_s", 0) for x in ok), 2),
            "usage_source": next((u.get("usage_source") for u in usage
                                  if u.get("usage_source")), "none"),
            "billed_input_tokens": billed_in,
            "fresh_input_tokens": sum(u.get("input_tokens", 0) for u in usage),
            "cache_write_tokens": sum(u.get("cache_write_tokens", 0) for u in usage),
            "cache_read_tokens": sum(u.get("cache_read_tokens", 0) for u in usage),
            "output_tokens": out_tok,
            "total_tokens": total,
            "tokens_per_question": round(total / len(valid), 1) if valid else None,
            "tokens_per_correct": round(total / correct, 1) if correct else None,
            "context_chars_mean": round(sum(x.get("context_chars", 0) for x in ok)
                                        / max(len(valid), 1), 1),
            "retries": sum(x.get("attempts", 1) - 1 for x in ok),
        }
    path = RUNS / f"haiku-summary-{stamp}-{label}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
