#!/usr/bin/env python3
"""Ingest the polluted 1M-token history into a dedicated Meko datapack.

This replaces the BM25 stand-in used by the edge-context experiment. The
retrieval arm has to be answerable by real Meko, or the cross-client overflow
result is a BM25 result wearing a Meko label.

Honesty constraint: Meko receives the SAME raw material the compaction arm
receives — the whole history, filler included, chunked. Ingesting only the
2,400 fact-bearing lines would hand retrieval a pre-filtered corpus and the
comparison would be worthless.

The history is sharded into conversations to mirror an estate that accumulated
over many sessions, which is also what makes it a cross-agent story: any client
holding the datapack id can read what another client wrote.

Run:  python ingest_overflow.py <stamp> [--chunk-chars N] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from meko_client import MekoMCPClient, run_id_for

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "runs-live"
HISTORY = ROOT / "fixtures" / "history.txt"
AGENT = "xc1:writer"
WORKERS = 8          # Meko rate-limits on in-flight concurrency above ~8
RETRIES = 3
CHUNKS_PER_CONV = 100


def create_conv(client, args):
    for _ in range(4):
        result = client.call_tool("conversation_create", args)
        if isinstance(result, dict) and result.get("id"):
            return result["id"]
        time.sleep(4)
    raise RuntimeError("conversation_create returned no id")


def write_one(item, datapack):
    client = MekoMCPClient.from_env()
    cid, text, conv, run_id = item
    err = "unknown"
    for attempt in range(RETRIES + 1):
        t0 = time.monotonic()
        try:
            client.call_tool("memory_add", {
                "scope": "write", "text": text, "conversation_id": conv,
                "agent_id": AGENT, "datapack_id": datapack, "run_id": run_id})
            return {"cid": cid, "ok": True,
                    "latency_s": round(time.monotonic() - t0, 2)}
        except Exception as e:  # noqa: BLE001
            err = str(e)[:200]
            time.sleep(3 * (attempt + 1))
    return {"cid": cid, "ok": False, "error": err}


def chunks_of(text: str, size: int) -> list[str]:
    """Split on line boundaries so no fact line is ever cut in half."""
    out, buf, used = [], [], 0
    for line in text.splitlines(keepends=True):
        if used + len(line) > size and buf:
            out.append("".join(buf))
            buf, used = [], 0
        buf.append(line)
        used += len(line)
    if buf:
        out.append("".join(buf))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stamp")
    ap.add_argument("--chunk-chars", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="ingest only the first N chunks")
    # memory_search only returns memories written under the run_id being
    # searched: a fresh reader conversation against a fully-populated datapack
    # returns zero hits (measured). Sharding the corpus across conversations
    # therefore hides most of it from any single query, so the default keeps
    # the whole estate in one conversation.
    ap.add_argument("--chunks-per-conv", type=int, default=100_000)
    args = ap.parse_args()
    global CHUNKS_PER_CONV
    CHUNKS_PER_CONV = args.chunks_per_conv

    RUNS.mkdir(exist_ok=True)
    text = HISTORY.read_text()
    chunks = chunks_of(text, args.chunk_chars)
    if args.limit:
        chunks = chunks[:args.limit]
    print(f"history {len(text):,} chars -> {len(chunks):,} chunks "
          f"of <={args.chunk_chars} chars", flush=True)

    client = MekoMCPClient.from_env()
    state_path = RUNS / f"xc1-ids-{args.stamp}.json"
    name = f"xc1_overflow_{args.stamp}"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        datapack, convs = state["datapack"], state["convs"]
    else:
        boot = create_conv(client, {"agent_id": AGENT, "title": "xc-1 boot"})
        try:
            datapack = client.call_tool(
                "datapack_create", {"name": name, "conversation_id": boot})["datapack_id"]
        except Exception as e:  # noqa: BLE001
            if "duplicate key" not in str(e):
                raise
            listing = client.call_tool("datapack_list", {"conversation_id": boot})
            datapack = next(d["datapack_id"] for d in listing
                            if d["datapack_name"] == name)
        n_conv = (len(chunks) + CHUNKS_PER_CONV - 1) // CHUNKS_PER_CONV
        convs = {}
        for shard in range(n_conv):
            convs[str(shard)] = create_conv(client, {
                "agent_id": AGENT, "datapack_id": datapack,
                "title": f"xc-1 team channel shard {shard:03d} [{args.stamp}]"})
            print(f"  conv shard {shard}: {convs[str(shard)]}", flush=True)
        state_path.write_text(json.dumps({"datapack": datapack, "convs": convs,
                                          "chunk_chars": args.chunk_chars,
                                          "n_chunks": len(chunks)}, indent=2))
    print(f"DATAPACK {datapack}", flush=True)

    ledger = RUNS / f"xc1-ingest-{args.stamp}.jsonl"
    done = set()
    if ledger.exists():
        for line in ledger.open():
            rec = json.loads(line)
            if rec.get("ok"):
                done.add(rec["cid"])

    items = []
    for i, chunk in enumerate(chunks):
        if i in done:
            continue
        conv = convs[str(i // CHUNKS_PER_CONV)]
        items.append((i, chunk, conv, run_id_for(conv)))
    print(f"{len(done)} already written, {len(items)} to write", flush=True)

    t0, ok, lat = time.monotonic(), 0, []
    with ledger.open("a") as lf:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for n, row in enumerate(pool.map(lambda it: write_one(it, datapack), items), 1):
                lf.write(json.dumps(row) + "\n")
                lf.flush()
                ok += row["ok"]
                if row.get("latency_s"):
                    lat.append(row["latency_s"])
                if n % 25 == 0:
                    rate = n / (time.monotonic() - t0)
                    eta = (len(items) - n) / rate / 60 if rate else 0
                    print(f"  {n}/{len(items)} ({ok} ok) "
                          f"{rate * 60:.0f}/min, eta {eta:.0f} min", flush=True)
    elapsed = time.monotonic() - t0
    lat.sort()
    print(f"ingest: {ok}/{len(items)} ok in {elapsed / 60:.1f} min"
          + (f", write p50 {lat[len(lat) // 2]:.1f}s" if lat else ""), flush=True)


if __name__ == "__main__":
    main()
