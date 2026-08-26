#!/usr/bin/env python3
"""MekoBench-P runner — 320 MCQs, three arms, exact token accounting.

Arms (identical reader, identical prompt shape, identical option order):
  meko     tuned client: memory_search k=25, persona-scoped run_id,
           retrieved statements in relevance order, 6000-char budget
  bm25     Okapi BM25 over the persona's own statements, k=25, same budget
  full     the whole persona history in chronological order (PersonaMem's
           "no memory system" condition — context stuffing)

Token accounting is exact: ollama returns prompt_eval_count and
eval_count per call. We report per-arm totals, mean prompt tokens per
question, and tokens-per-correct-answer. Retrieval-side payload sizes are
recorded as characters sent/received.

Run:  python run_mekobench.py <stamp> <ids-json from ingest>
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from meko_client import MekoMCPClient, extract_search_texts, run_id_for

K = 25
CTX_CHARS = 6000
MODEL = "gemma4-e4b-sys:latest"   # cheaper reader: 4B effective vs 12B
OLLAMA = "http://127.0.0.1:11434/api/generate"
AGENT = "mekobench:reader"
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs-live"

PROMPT = """You are answering a multiple-choice question about a specific user,
based ONLY on the dated session notes below. Statements may be updated by
later sessions; trust the most recent dated statement when they conflict.

Session notes:
{ctx}

Question: {q}
A) {a}
B) {b}
C) {c}
D) {d}

Reply with exactly one letter: A, B, C, or D.
"""


# ---------- BM25 (self-contained copy) ----------
_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def tok(t):
    return _TOKEN.findall(t.lower())


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [{"text": t, "toks": Counter(tok(t))} for t in texts]
        self.n = len(self.docs)
        self.avgdl = sum(sum(d["toks"].values()) for d in self.docs) / self.n
        self.df = Counter()
        for d in self.docs:
            self.df.update(d["toks"].keys())

    def search(self, query, k):
        q = tok(query)
        scored = []
        for d in self.docs:
            dl = sum(d["toks"].values())
            s = 0.0
            for t in q:
                tf = d["toks"].get(t)
                if not tf:
                    continue
                idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scored.append((s, d["text"]))
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:k]]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n)


def ask(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "think": False,
                       "options": {"temperature": 0, "num_predict": 8}}).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read().decode())
    reply = out.get("response", "").strip()
    if not reply:
        raise RuntimeError(f"empty generation ({out.get('done_reason')})")
    return reply, out.get("prompt_eval_count", 0), out.get("eval_count", 0)


def build_ctx(texts):
    ctx, used = [], 0
    for t in texts:
        line = f"- {t}\n"
        if used + len(line) > CTX_CHARS:
            break
        ctx.append(line)
        used += len(line)
    return "".join(ctx) if ctx else "(no notes retrieved)"


def meko_search(query, conv, run_id, datapack):
    c = MekoMCPClient.from_env()
    for attempt in range(4):
        try:
            r = c.call_tool("memory_search", {
                "scope": "read", "query": query, "conversation_id": conv,
                "agent_id": AGENT, "datapack_id": datapack,
                "run_id": run_id, "limit": K})
            return extract_search_texts(r)
        except Exception:  # noqa: BLE001
            time.sleep(3 * (attempt + 1))
    return []


def main():
    global MODEL
    stamp, ids_path = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        MODEL = sys.argv[3]
    st = json.loads(open(ids_path).read())
    datapack, convs = st["datapack"], st["convs"]
    qs = [json.loads(l) for l in (ROOT / "fixtures/questions.jsonl").open()]
    stmts = [json.loads(l) for l in (ROOT / "fixtures/statements.jsonl").open()]
    by_persona = {}
    for s in stmts:
        by_persona.setdefault(s["persona"], []).append(s["text"])
    bm25 = {p: BM25(texts) for p, texts in by_persona.items()}

    ledger_path = RUNS / f"pm-run-{stamp}.jsonl"
    done_qids = set()
    if ledger_path.exists():
        prior = [json.loads(l) for l in ledger_path.open() if l.strip()]
        done_qids = {r["qid"] for r in prior if r.get("event") == "q"}
        if done_qids and "--resume" not in sys.argv:
            raise SystemExit(
                f"{ledger_path.name} already holds {len(done_qids)} answered "
                f"questions. Re-running would append a second copy and "
                f"silently double-count in the summary. Use a new stamp, or "
                f"pass --resume to continue this one.")
    if done_qids:
        qs = [q for q in qs if q["qid"] not in done_qids]
        print(f"resuming: {len(done_qids)} done, {len(qs)} to go", flush=True)
    ev = open(ledger_path, "a")

    def rec(**row):
        ev.write(json.dumps(row, ensure_ascii=False) + "\n")
        ev.flush()

    rec(event="start", stamp=stamp, n=len(qs), k=K, ctx_chars=CTX_CHARS,
        model=MODEL, datapack=datapack)

    # retrieval phase (meko arm), 8-way
    def retrieve(q):
        conv = convs[q["persona"]]
        return meko_search(q["question"], conv, run_id_for(conv), datapack)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as ex:
        meko_ctx = list(ex.map(retrieve, qs))
    print(f"meko retrieval: {time.monotonic() - t0:.0f}s", flush=True)

    arms = ("meko", "bm25", "full")
    tokens = {a: {"prompt": 0, "out": 0} for a in arms}
    rows = []
    for i, (q, mtexts) in enumerate(zip(qs, meko_ctx), 1):
        ctxs = {
            "meko": build_ctx(mtexts),
            "bm25": build_ctx(bm25[q["persona"]].search(q["question"], K)),
            "full": build_ctx(by_persona[q["persona"]]),
        }
        row = {"qid": q["qid"], "type": q["type"], "persona": q["persona"],
               "answer": q["answer"]}
        for arm in arms:
            p = PROMPT.format(ctx=ctxs[arm], q=q["question"],
                              a=q["options"][0], b=q["options"][1],
                              c=q["options"][2], d=q["options"][3])
            reply, ptok, otok = ask(p)
            m = re.search(r"[ABCD]", reply.upper())
            pick = m.group(0) if m else "?"
            row[f"{arm}_pick"] = pick
            row[f"{arm}_ok"] = pick == q["answer"]
            row[f"{arm}_ptok"] = ptok
            tokens[arm]["prompt"] += ptok
            tokens[arm]["out"] += otok
        rows.append(row)
        rec(event="q", **row)
        if i % 40 == 0:
            print(f"  {i}/{len(qs)}", flush=True)

    summary = {"stamp": stamp, "n": len(qs), "model": MODEL, "arms": {}}
    for arm in arms:
        ok = sum(r[f"{arm}_ok"] for r in rows)
        by_type = {}
        for t in ("FACT", "CURRENT", "EVOLUTION", "REASON", "RECOMMEND", "UNANSWER"):
            sub = [r for r in rows if r["type"] == t]
            by_type[t] = f"{sum(r[f'{arm}_ok'] for r in sub)}/{len(sub)}"
        summary["arms"][arm] = {
            "correct": ok, "accuracy": round(ok / len(rows), 4),
            "by_type": by_type,
            "prompt_tokens_total": tokens[arm]["prompt"],
            "prompt_tokens_mean": round(tokens[arm]["prompt"] / len(rows), 1),
            "output_tokens_total": tokens[arm]["out"],
            "tokens_per_correct": round(
                (tokens[arm]["prompt"] + tokens[arm]["out"]) / max(ok, 1), 1)}
    for a, b in (("meko", "bm25"), ("meko", "full"), ("bm25", "full")):
        x = sum(1 for r in rows if r[f"{a}_ok"] and not r[f"{b}_ok"])
        y = sum(1 for r in rows if r[f"{b}_ok"] and not r[f"{a}_ok"])
        summary[f"{a}_vs_{b}"] = {f"{a}_only": x, f"{b}_only": y,
                                  "mcnemar_p": round(mcnemar(x, y), 6)}
    rec(event="summary", **summary)
    (RUNS / f"pm-summary-{stamp}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
