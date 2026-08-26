#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parent
required=[ROOT/"babilong/run_babilong_bm25_800.py",ROOT/"babilong/babilong_bm25_800_evidence.py",ROOT/"babilong/runs/babilong-bm25-800/packets.jsonl",ROOT/"babilong/data/data/qa1/0k.json",ROOT/"babilong/upstream/babilong/prompts.py",ROOT/"babilong/upstream/babilong/metrics.py"]
missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing: raise SystemExit("missing: "+", ".join(missing))
for p in ROOT.rglob("*"):
    if p.is_file() and p.name != "VERIFY.py":
        s=p.read_text(errors="ignore")
        if "/home/" in s or "/Users/" in s or "@gmail.com" in s or "@yahoo.com" in s: raise SystemExit(f"private text: {p}")
print("BABILong handoff verification passed")
