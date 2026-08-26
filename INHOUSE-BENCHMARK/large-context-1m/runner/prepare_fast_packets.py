#!/usr/bin/env python3
"""Build and publish deterministic 500-character evidence packets.

This is the fast storage/delivery arm, not Meko semantic retrieval.  A local
Okapi BM25 index ranks line-safe chunks, one packet is assembled per question,
and the packet is stored as a gzip artifact.  Compression keeps the complete
packet below the artifact size limit.  Every artifact is fetched by its
locally-computed SHA-256 and compared byte-for-byte before it enters the
resumable ledger.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ingest_overflow import chunks_of, create_conv
from meko_client import MekoMCPClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "runs-live"
FIX = ROOT / "generated"
SOURCE_FIX = ROOT / "fixtures"
HISTORY = SOURCE_FIX / "history.txt"
QUESTIONS = SOURCE_FIX / "questions.json"
TOKEN = re.compile(r"\w+")
AGENT = "xc1:fast-packet-writer"
REQUEST_INTERVAL_S = 0.7
RETRIES = 4
_rate_lock = threading.Lock()
_next_request = 0.0


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, chunks: list[str], k1: float = 1.5,
                 b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tokens = [Counter(tokenize(chunk)) for chunk in chunks]
        self.lengths = [sum(tokens.values()) for tokens in self.tokens]
        self.average = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(tokens.keys())
        count = len(chunks)
        self.idf = {
            word: math.log((count - frequency + 0.5) /
                           (frequency + 0.5) + 1)
            for word, frequency in document_frequency.items()
        }

    def top(self, query: str, limit: int) -> list[int]:
        query_tokens = tokenize(query)
        scores = []
        for index, tokens in enumerate(self.tokens):
            norm = self.k1 * (
                1 - self.b + self.b * self.lengths[index] / self.average)
            score = 0.0
            for word in query_tokens:
                frequency = tokens.get(word, 0)
                if frequency:
                    score += (self.idf.get(word, 0.0) * frequency *
                              (self.k1 + 1) / (frequency + norm))
            scores.append((score, index))
        scores.sort(key=lambda pair: (-pair[0], pair[1]))
        return [index for _, index in scores[:limit]]


def gold_in_context(question: dict, context: str) -> bool | None:
    text = context.lower()
    if question["score"] == "contains":
        return question["answer"].lower() in text
    if question["score"] == "contains_all":
        return all(value.lower() in text
                   for value in question["answer_list"])
    return None


def call_limited(client: MekoMCPClient, tool: str,
                 arguments: dict) -> dict:
    global _next_request
    with _rate_lock:
        now = time.monotonic()
        wait = max(0.0, _next_request - now)
        _next_request = max(now, _next_request) + REQUEST_INTERVAL_S
    if wait:
        time.sleep(wait)
    return client.call_tool(tool, arguments)


def artifact_bytes(payload: dict) -> bytes | None:
    encoded = payload.get("content_base64")
    if encoded:
        return base64.b64decode(encoded)
    local_path = payload.get("local_path")
    if local_path and Path(local_path).is_file():
        return Path(local_path).read_bytes()
    return None


def put_verified(item: tuple[int, bytes], conversation: str,
                 datapack: str) -> dict:
    index, raw = item
    expected = sha256(raw)
    client = MekoMCPClient.from_env()
    errors = []
    started = time.monotonic()
    for attempt in range(1, RETRIES + 1):
        try:
            call_limited(client, "artifact_put", {
                "filename": f"xc1-c500-packet-{index:03d}.txt.gz",
                "content_base64": base64.b64encode(raw).decode(),
                "content_type": "application/gzip",
                "conversation_id": conversation,
                "datapack_id": datapack,
            })
            fetched = call_limited(client, "artifact_get", {
                "content_hash": expected,
                "conversation_id": conversation,
                "datapack_id": datapack,
            })
            if artifact_bytes(fetched) == raw:
                return {"idx": index, "ok": True, "hash": expected,
                        "attempts": attempt,
                        "latency_s": round(time.monotonic() - started, 3)}
            errors.append("readback bytes missing or different")
        except Exception as error:  # noqa: BLE001
            message = f"{type(error).__name__}: {str(error)[:180]}"
            errors.append(message)
            time.sleep(65 if "PAT_RATE_LIMITED" in message else attempt)
    return {"idx": index, "ok": False, "hash": expected,
            "errors": errors[-RETRIES:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stamp")
    parser.add_argument("--chunk-chars", type=int, default=500)
    parser.add_argument("--k", type=int, default=300)
    parser.add_argument("--ctx-chars", type=int, default=150_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    RUNS.mkdir(exist_ok=True)
    FIX.mkdir(exist_ok=True)

    history_bytes = HISTORY.read_bytes()
    question_bytes = QUESTIONS.read_bytes()
    questions = json.loads(question_bytes)
    chunks = chunks_of(history_bytes.decode(), args.chunk_chars)
    index = BM25(chunks)
    print(f"index ready: {len(chunks):,} line-safe chunks; "
          f"{len(questions)} questions", flush=True)

    contexts: dict[str, str] = {}
    packets: list[bytes] = []
    packet_rows = []
    ceiling: dict[str, list[int]] = {}
    for question_index, question in enumerate(questions):
        pieces, used, chunk_ids = [], 0, []
        for chunk_index in index.top(question["question"], args.k):
            piece = chunks[chunk_index].rstrip() + "\n"
            if used + len(piece) > args.ctx_chars:
                break
            pieces.append(piece)
            chunk_ids.append(chunk_index)
            used += len(piece)
        context = "".join(pieces)
        contexts[str(question_index)] = context
        raw = gzip.compress(context.encode(), compresslevel=9, mtime=0)
        packets.append(raw)
        present = gold_in_context(question, context)
        if present is not None:
            hit, total = ceiling.setdefault(question["type"], [0, 0])
            ceiling[question["type"]] = [hit + int(present), total + 1]
        packet_rows.append({
            "idx": question_index,
            "question_sha256": sha256(question["question"].encode()),
            "chunk_ids": chunk_ids,
            "context_chars": len(context),
            "context_sha256": sha256(context.encode()),
            "artifact_sha256": sha256(raw),
            "artifact_bytes": len(raw),
            "gold_in_context": present,
        })

    context_path = FIX / f"meko_fast_ctx-{args.stamp}.json"
    manifest_path = RUNS / f"fast-packet-manifest-{args.stamp}.json"
    context_path.write_text(json.dumps(contexts))
    hits = sum(value[0] for value in ceiling.values())
    total = sum(value[1] for value in ceiling.values())
    manifest = {
        "stamp": args.stamp,
        "method": "local_okapi_bm25_then_verified_meko_artifact",
        "meko_semantic_retrieval": False,
        "history_path": "fixtures/history.txt",
        "history_sha256": sha256(history_bytes),
        "questions_path": "fixtures/questions.json",
        "questions_sha256": sha256(question_bytes),
        "chunking": "line-safe chunks_of",
        "chunk_chars": args.chunk_chars,
        "chunks_in_corpus": len(chunks),
        "k": args.k,
        "ctx_chars_cap": args.ctx_chars,
        "questions": len(questions),
        "ceiling": round(hits / total, 4) if total else None,
        "ceiling_count": f"{hits}/{total}",
        "ceiling_by_type": {
            kind: f"{value[0]}/{value[1]}"
            for kind, value in sorted(ceiling.items())
        },
        "mean_context_chars": round(
            sum(row["context_chars"] for row in packet_rows) /
            len(packet_rows)),
        "mean_artifact_bytes": round(
            sum(row["artifact_bytes"] for row in packet_rows) /
            len(packet_rows)),
        "packets": packet_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in (
        "ceiling", "ceiling_count", "ceiling_by_type",
        "mean_context_chars", "mean_artifact_bytes")}, indent=2),
        flush=True)
    print(f"contexts: {context_path}\nmanifest: {manifest_path}", flush=True)
    if args.prepare_only:
        return

    client = MekoMCPClient.from_env()
    state_path = RUNS / f"fast-packet-ids-{args.stamp}.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        conversation, datapack = state["conversation"], state["datapack"]
    else:
        conversation = create_conv(client, {
            "agent_id": AGENT, "title": f"xc-1 fast packets {args.stamp}"})
        result = client.call_tool("datapack_create", {
            "name": f"xc1_fast_packets_{args.stamp}",
            "conversation_id": conversation,
        })
        datapack = result["datapack_id"]
        state_path.write_text(json.dumps({"conversation": conversation,
                                          "datapack": datapack}, indent=2))

    ledger_path = RUNS / f"fast-packet-ingest-{args.stamp}.jsonl"
    done = set()
    if ledger_path.exists():
        for line in ledger_path.open():
            row = json.loads(line)
            if row.get("ok"):
                done.add(row["idx"])
    todo = [(packet_index, raw) for packet_index, raw in enumerate(packets)
            if packet_index not in done]
    print(f"datapack {datapack}; {len(done)} verified; "
          f"{len(todo)} remaining", flush=True)
    started = time.monotonic()
    completed = 0
    with ledger_path.open("a") as output:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(put_verified, item, conversation, datapack)
                       for item in todo]
            for future in as_completed(futures):
                row = future.result()
                output.write(json.dumps(row) + "\n")
                output.flush()
                completed += 1
                if completed % 20 == 0 or completed == len(todo):
                    elapsed = time.monotonic() - started
                    rate = completed / elapsed if elapsed else 0
                    eta = ((len(todo) - completed) / rate / 60
                           if rate else None)
                    print(f"{completed}/{len(todo)}; {rate * 60:.1f}/min; "
                          f"ETA {eta:.1f} min", flush=True)

    latest = {}
    for line in ledger_path.open():
        row = json.loads(line)
        latest[row["idx"]] = row
    verified = sum(bool(latest.get(i, {}).get("ok"))
                   for i in range(len(packets)))
    manifest["meko"] = {"conversation": conversation, "datapack": datapack,
                         "verified": verified,
                         "artifact_count": len(packets)}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"verified {verified}/{len(packets)}", flush=True)
    if verified != len(packets):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
