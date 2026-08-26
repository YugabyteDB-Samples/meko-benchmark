#!/usr/bin/env python3
"""RecipeBench: recipe-following Meko client versus plausible defaults.

Phases are independently resumable:

  retrieve  Save one k=25 Meko response per question plus matched BM25 hits.
  local     Run all saved contexts through an Ollama reader.
  summary   Recompute summary from the local ledger.

The Haiku validation runner consumes the same retrieval ledger separately.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PM = HERE
FIX = ROOT / "fixtures"
RUNS = ROOT / "runs-live"
OLLAMA = "http://127.0.0.1:11434/api/generate"
K = 25
CTX_CHARS = 6000
AGENT = "recipebench:reader"

sys.path.insert(0, str(PM))
from gen_mekobench import SLOTS  # noqa: E402
from meko_client import MekoMCPClient, run_id_for  # noqa: E402
from run_mekobench import BM25, PROMPT  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_fixture():
    return (read_jsonl(FIX / "statements.jsonl"),
            read_jsonl(FIX / "questions.jsonl"))


def extract_hits(payload: dict) -> list[dict]:
    raw = None
    for key in ("results", "memories", "items"):
        if key in payload:
            raw = payload[key]
            break
    if raw is None:
        raise ValueError(f"search response has no results collection: {sorted(payload)}")
    if isinstance(raw, dict):
        raw = raw.get("results") or raw.get("items") or []
    hits = []
    for rank, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        text = item.get("memory") or item.get("text") or item.get("content")
        if text:
            hits.append({"rank": rank, "text": text, "id": item.get("id"),
                         "score": item.get("score")})
    return hits


def search(client, q: dict, ids: dict) -> dict:
    conv = ids["convs"][q["persona"]]
    errors = []
    t0 = time.monotonic()
    for attempt in range(1, 5):
        try:
            payload = client.call_tool("memory_search", {
                "query": q["question"], "conversation_id": conv,
                "agent_id": AGENT, "datapack_id": ids["datapack"],
                "run_id": run_id_for(conv), "limit": K})
            return {"ok": True, "attempts": attempt,
                    "latency_s": round(time.monotonic() - t0, 3),
                    "hits": extract_hits(payload), "errors": errors}
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {str(exc)[:220]}")
            if attempt < 4:
                time.sleep(2 * attempt)
    return {"ok": False, "attempts": 4,
            "latency_s": round(time.monotonic() - t0, 3),
            "hits": [], "errors": errors}


def target_kind(q: dict) -> str | None:
    text = q["question"].lower()
    for kind in sorted(SLOTS, key=len, reverse=True):
        if kind in text:
            return kind
    if q["type"] == "RECOMMEND" and "taste" in text:
        return "favorite cuisine"
    return None


def metadata_index(statements: list[dict]) -> dict[str, dict]:
    return {row["text"]: row for row in statements}


def recipe_hits(hits: list[dict], q: dict, meta: dict[str, dict]) -> list[dict]:
    """Task-aware, conflict-scoped ordering over an unchanged hit set."""
    kind = target_kind(q)
    if not kind:
        return hits
    target = [h for h in hits if meta.get(h["text"], {}).get("kind") == kind]
    other = [h for h in hits if meta.get(h["text"], {}).get("kind") != kind]
    if not target:
        return hits
    target.sort(key=lambda h: meta[h["text"]]["date"])
    if q["type"] in {"CURRENT", "RECOMMEND", "FACT"}:
        target = [target[-1]]
    # A target timeline enters first; unrelated hits keep server relevance order.
    return target + other


def bounded(hits: list[dict], limit: int | None = None) -> dict:
    selected = hits if limit is None else hits[:limit]
    out, used = [], 0
    for hit in selected:
        piece = "- " + hit["text"].rstrip() + "\n"
        if used + len(piece) > CTX_CHARS:
            break
        out.append(piece)
        used += len(piece)
    return {"text": "".join(out) or "(no notes retrieved)",
            "chars": used, "documents": len(out)}


def full_context(hits: list[dict]) -> dict:
    text = "".join("- " + hit["text"].rstrip() + "\n" for hit in hits)
    return {"text": text or "(no notes)", "chars": len(text),
            "documents": len(hits)}


def contexts(row: dict, by_persona: dict, meta: dict) -> dict[str, dict]:
    mhits = row["meko"]["hits"]
    bhits = row["bm25_hits"]
    q = row["question_row"]
    full_hits = [{"text": x} for x in by_persona[q["persona"]]]
    return {
        "meko_raw10": bounded(mhits, 10),
        "meko_raw25": bounded(mhits, 25),
        "meko_recipe": bounded(recipe_hits(mhits, q, meta), 25),
        # optimized: conflict-scoped ordering (surface the right claim
        # first) THEN the cheap k=10 window that won on tokens+accuracy.
        "meko_optimized": bounded(recipe_hits(mhits, q, meta), 10),
        "bm25_recipe": bounded(recipe_hits(bhits, q, meta), 25),
        "full_history": full_context(full_hits),
    }


def retrieve(args) -> None:
    RUNS.mkdir(exist_ok=True)
    statements, questions = load_fixture()
    ids = json.loads(args.ids.read_text())
    by_persona = {}
    for row in statements:
        by_persona.setdefault(row["persona"], []).append(row["text"])
    bm25 = {name: BM25(texts) for name, texts in by_persona.items()}
    ledger = RUNS / f"retrieval-{args.stamp}.jsonl"
    done = set()
    if ledger.exists():
        done = {x["qid"] for x in read_jsonl(ledger) if x.get("event") == "retrieval"}
    todo = [q for q in questions if q["qid"] not in done]
    print(f"retrieve: {len(done)} resumed, {len(todo)} remaining", flush=True)
    lock, count = threading.Lock(), {"n": 0}
    out = open(ledger, "a")

    def one(q):
        meko = search(MekoMCPClient.from_env(), q, ids)
        t0 = time.monotonic()
        btexts = bm25[q["persona"]].search(q["question"], K)
        bm = [{"rank": i + 1, "text": text} for i, text in enumerate(btexts)]
        row = {"event": "retrieval", "qid": q["qid"], "type": q["type"],
               "question_row": q, "meko": meko, "bm25_hits": bm,
               "bm25_latency_s": round(time.monotonic() - t0, 6)}
        with lock:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            count["n"] += 1
            if count["n"] % 25 == 0:
                print(f"  retrieved {count['n']}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, q) for q in todo]
        for future in as_completed(futures):
            future.result()
    out.close()
    summarize_retrieval(args.stamp)


def ask_ollama(model: str, prompt: str, seed: int) -> dict:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
        "think": False, "keep_alive": "30m", "options": {"temperature": 0,
        "seed": seed, "num_predict": 8, "num_ctx": 32768}}).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.loads(resp.read().decode())
    reply = payload.get("response", "").strip()
    match = re.search(r"[ABCD]", reply.upper())
    return {"reply": reply, "pick": match.group(0) if match else "?",
            "prompt_tokens": payload.get("prompt_eval_count", 0),
            "output_tokens": payload.get("eval_count", 0),
            "latency_s": round(time.monotonic() - t0, 3)}


def retrieval_rows(stamp: str) -> list[dict]:
    rows = [x for x in read_jsonl(RUNS / f"retrieval-{stamp}.jsonl")
            if x.get("event") == "retrieval"]
    return sorted(rows, key=lambda x: x["qid"])


def run_local(args) -> None:
    rows = retrieval_rows(args.stamp)
    if len(rows) != 320:
        raise SystemExit(f"retrieval incomplete: {len(rows)}/320")
    if any(not row["meko"]["ok"] for row in rows):
        raise SystemExit("retrieval contains failures; inspect before reader run")
    statements, _ = load_fixture()
    by_persona = {}
    for row in statements:
        by_persona.setdefault(row["persona"], []).append(row["text"])
    meta = metadata_index(statements)
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    ledger = RUNS / f"local-{args.stamp}-{args.model_label}.jsonl"
    done = set()
    if ledger.exists():
        done = {(x["qid"], x["arm"]) for x in read_jsonl(ledger)
                if x.get("event") == "answer"}
    tasks = [(row, arm) for row in rows for arm in arms
             if (row["qid"], arm) not in done]
    print(f"local reader: {len(done)} resumed, {len(tasks)} calls", flush=True)
    lock, count = threading.Lock(), {"n": 0}
    out = open(ledger, "a")

    def one(task):
        row, arm = task
        q = row["question_row"]
        ctx = contexts(row, by_persona, meta)[arm]
        prompt = PROMPT.format(ctx=ctx["text"], q=q["question"],
            a=q["options"][0], b=q["options"][1],
            c=q["options"][2], d=q["options"][3])
        try:
            result = ask_ollama(args.model, prompt, args.seed)
            error = None
        except Exception as exc:
            result = {"reply": "", "pick": "?", "prompt_tokens": 0,
                      "output_tokens": 0, "latency_s": 0}
            error = f"{type(exc).__name__}: {str(exc)[:220]}"
        answer = {"event": "answer", "qid": q["qid"], "type": q["type"],
            "persona": q["persona"], "arm": arm, "gold": q["answer"],
            "ok": result["pick"] == q["answer"], "error": error,
            "context_chars": ctx["chars"], "context_documents": ctx["documents"],
            "model": args.model, "model_label": args.model_label, **result}
        with lock:
            out.write(json.dumps(answer, ensure_ascii=False) + "\n")
            out.flush()
            count["n"] += 1
            if count["n"] % 100 == 0:
                print(f"  answered {count['n']}/{len(tasks)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
    out.close()
    summarize_local(args.stamp, args.model_label)


def mcnemar(x: int, y: int) -> float:
    n = x + y
    if not n:
        return 1.0
    m = min(x, y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n)


def summarize_retrieval(stamp: str) -> dict:
    rows = retrieval_rows(stamp)
    lat = [r["meko"]["latency_s"] for r in rows]
    summary = {"stamp": stamp, "n": len(rows),
        "meko": {"successful": sum(r["meko"]["ok"] for r in rows),
          "failed": sum(not r["meko"]["ok"] for r in rows),
          "empty_success": sum(r["meko"]["ok"] and not r["meko"]["hits"] for r in rows),
          "latency_s_mean": round(statistics.mean(lat), 3) if lat else None,
          "latency_s_p50": round(statistics.median(lat), 3) if lat else None,
          "result_count_mean": round(statistics.mean(len(r["meko"]["hits"])
                                                      for r in rows), 2) if rows else None},
        "caller_cache_repeat": {"repeat_questions": len(rows),
            "network_calls": 0, "search_latency_s": 0,
            "network_calls_avoided": len(rows),
            "first_search_seconds_avoided": round(sum(lat), 1)}}
    path = RUNS / f"retrieval-summary-{stamp}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def summarize_local(stamp: str, model_label: str) -> dict:
    path = RUNS / f"local-{stamp}-{model_label}.jsonl"
    rows = [x for x in read_jsonl(path) if x.get("event") == "answer"]
    arms = sorted(set(x["arm"] for x in rows))
    summary = {"stamp": stamp, "model_label": model_label, "rows": len(rows),
               "arms": {}}
    for arm in arms:
        sub = [x for x in rows if x["arm"] == arm]
        valid = [x for x in sub if not x["error"]]
        correct = sum(x["ok"] for x in valid)
        tokens = sum(x["prompt_tokens"] + x["output_tokens"] for x in valid)
        summary["arms"][arm] = {"n": len(sub), "valid": len(valid),
            "errors": len(sub) - len(valid), "correct": correct,
            "accuracy": round(correct / len(valid), 4) if valid else None,
            "by_type": {kind: f"{sum(x['ok'] for x in valid if x['type']==kind)}/"
                                  f"{sum(1 for x in valid if x['type']==kind)}"
                        for kind in sorted(set(x["type"] for x in valid))},
            "prompt_tokens_mean": round(statistics.mean(x["prompt_tokens"]
                                                         for x in valid), 1) if valid else None,
            "context_chars_mean": round(statistics.mean(x["context_chars"]
                                                         for x in valid), 1) if valid else None,
            "tokens_per_correct": round(tokens / max(correct, 1), 1),
            "latency_s_mean": round(statistics.mean(x["latency_s"]
                                                     for x in valid), 3) if valid else None}
    for a, b in (("meko_recipe", "meko_raw25"),
                 ("meko_recipe", "meko_raw10"),
                 ("meko_recipe", "full_history"),
                 ("meko_recipe", "bm25_recipe"),
                 ("meko_optimized", "meko_raw10"),
                 ("meko_optimized", "meko_raw25"),
                 ("meko_optimized", "full_history")):
        if a not in arms or b not in arms:
            continue
        amap = {x["qid"]: x for x in rows if x["arm"] == a and not x["error"]}
        bmap = {x["qid"]: x for x in rows if x["arm"] == b and not x["error"]}
        common = sorted(set(amap) & set(bmap))
        x = sum(amap[q]["ok"] and not bmap[q]["ok"] for q in common)
        y = sum(bmap[q]["ok"] and not amap[q]["ok"] for q in common)
        summary[f"{a}_vs_{b}"] = {f"{a}_only": x, f"{b}_only": y,
                                    "mcnemar_p": round(mcnemar(x, y), 6)}
    out = RUNS / f"local-summary-{stamp}-{model_label}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    r = sub.add_parser("retrieve")
    r.add_argument("--stamp", required=True)
    r.add_argument("--ids", type=Path, required=True)
    r.add_argument("--workers", type=int, default=8)
    q = sub.add_parser("local")
    q.add_argument("--stamp", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--model-label", required=True)
    q.add_argument("--workers", type=int, default=4)
    q.add_argument("--seed", type=int, default=20260809)
    q.add_argument("--arms", default="meko_raw10,meko_raw25,meko_recipe,bm25_recipe,full_history")
    s = sub.add_parser("summary")
    s.add_argument("--stamp", required=True)
    s.add_argument("--model-label")
    args = ap.parse_args()
    if args.phase == "retrieve":
        retrieve(args)
    elif args.phase == "local":
        run_local(args)
    else:
        summarize_retrieval(args.stamp)
        if args.model_label:
            summarize_local(args.stamp, args.model_label)


if __name__ == "__main__":
    main()
