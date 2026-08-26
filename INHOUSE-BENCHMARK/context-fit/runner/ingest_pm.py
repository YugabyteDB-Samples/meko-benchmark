#!/usr/bin/env python3
"""MekoBench-P ingest — one datapack, one conversation (= run_id) per persona.

Mirrors real usage: every session utterance (preference-bearing AND filler)
is written, because a memory service does not get to know in advance which
turns matter. Ledgered and resumable.

Run:  python ingest_pm.py <stamp> [resume]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from meko_client import MekoMCPClient, run_id_for

WORKERS = 8
RETRIES = 3
AGENT = "mekobench:writer"
ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs-live"


def create_conv(c, args):
    for _ in range(4):
        r = c.call_tool("conversation_create", args)
        if isinstance(r, dict) and r.get("id"):
            return r["id"]
        time.sleep(4)
    raise RuntimeError("conversation_create returned no id")


def write_one(item, datapack):
    c = MekoMCPClient.from_env()
    sid, text, conv, run_id = item
    for attempt in range(RETRIES + 1):
        t0 = time.monotonic()
        try:
            c.call_tool("memory_add", {
                "scope": "write", "text": text, "conversation_id": conv,
                "agent_id": AGENT, "datapack_id": datapack, "run_id": run_id})
            return {"sid": sid, "ok": True,
                    "latency_s": round(time.monotonic() - t0, 2)}
        except Exception as e:  # noqa: BLE001
            err = str(e)[:150]
            time.sleep(3 * (attempt + 1))
    return {"sid": sid, "ok": False, "error": err}


def main():
    stamp = sys.argv[1]
    RUNS.mkdir(exist_ok=True)
    ledger = RUNS / f"pm-ingest-{stamp}.jsonl"
    stmts = [json.loads(l) for l in (ROOT / "fixtures/statements.jsonl").open()]
    c = MekoMCPClient.from_env()

    state_path = RUNS / f"pm-ids-{stamp}.json"
    if state_path.exists():
        st = json.loads(state_path.read_text())
        datapack, convs = st["datapack"], st["convs"]
    else:
        boot = create_conv(c, {"agent_id": AGENT, "title": "MekoBench-P boot"})
        try:
            dp = c.call_tool("datapack_create", {
                "name": f"mekobench_pm_{stamp[:8]}", "conversation_id": boot})
            datapack = dp["datapack_id"]
        except Exception as e:  # noqa: BLE001
            if "duplicate key" not in str(e):
                raise
            listing = c.call_tool("datapack_list", {"conversation_id": boot})
            datapack = next(d["datapack_id"] for d in listing
                            if d["datapack_name"] == f"mekobench_pm_{stamp[:8]}")
        convs = {}
        for name in sorted({s["persona"] for s in stmts}):
            convs[name] = create_conv(c, {
                "agent_id": AGENT, "datapack_id": datapack,
                "title": f"MekoBench-P history — {name} [{stamp}]"})
        state_path.write_text(json.dumps({"datapack": datapack, "convs": convs}))
    print(f"DATAPACK {datapack}")
    for n, cid in convs.items():
        print(f"CONV {n}: {cid} run_id {run_id_for(cid)}")

    done = set()
    if ledger.exists():
        for l in open(ledger):
            r = json.loads(l)
            if r.get("ok"):
                done.add(r["sid"])
    items = []
    for i, s in enumerate(stmts):
        sid = f"{s['persona']}#{i}"
        if sid in done:
            continue
        conv = convs[s["persona"]]
        items.append((sid, s["text"], conv, run_id_for(conv)))
    print(f"{len(done)} done, {len(items)} to write")

    t0, ok = time.monotonic(), 0
    with open(ledger, "a") as lf:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for i, row in enumerate(ex.map(lambda it: write_one(it, datapack), items)):
                lf.write(json.dumps(row) + "\n")
                lf.flush()
                ok += row["ok"]
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(items)} ({ok} ok, "
                          f"{time.monotonic() - t0:.0f}s)", flush=True)
    print(f"ingest: {ok}/{len(items)} ok in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
