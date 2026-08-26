#!/usr/bin/env python3
"""Matched BABILong BM25 control: 800-character chunks, top five.

This arm exists to compare plain lexical BM25 with the retained artifact-hybrid
arm under the same chunk size, top-k, source ordering, prompts, and scorer.  It
does not create Meko objects.  Local official bytes are accepted only when their
SHA-256 equals the source hash in the artifact-verified retrieval snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "babilong-bm25-800"
RESULT_ROOT = Path(os.environ.get("BABYLONG_BM25_RESULTS", str(RUN))).expanduser().resolve()
PACKETS = RUN / "packets.jsonl"
PACKET_MANIFEST = RUN / "packet-manifest.json"
SNAPSHOT = HERE / "runs" / "babilong-multireader-full" / "retrieval-snapshot.jsonl"
N = 100
K = 5
CHUNK_CHARS = 800
READERS = (
    "qwen", "sonnet", "deepseek_v4_flash", "codex_luna",
    "gemini_3_7_flash_low",
)
OFFICIAL_CELLS = tuple(
    (f"qa{i}", size) for i in range(1, 11) for size in ("0k", "4k", "16k")
) + tuple((f"qa{i}", "0k") for i in range(11, 21))

sys.path.insert(0, str(HERE))
from babilong_multireader_providers import call_codex, call_deepseek, call_gemini  # noqa: E402
sys.path.insert(0, str(HERE / "upstream"))
from babilong.metrics import TASK_LABELS, compare_answers  # noqa: E402
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402

QWEN_MODEL = os.environ.get("BABYLONG_QWEN_MODEL", "qwen38-bench")
OLLAMA = os.environ.get("BABYLONG_OLLAMA", "http://localhost:11434")
CLAUDE = os.environ.get("BABYLONG_CLAUDE", shutil.which("claude") or "claude")
SONNET_MODEL = os.environ.get("BABYLONG_SONNET_MODEL", "claude-sonnet-5")


def chunk_text(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Frozen sentence-aware chunker; split sentences longer than the cap."""
    clean = text.replace("<", "(").replace(">", ")")
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    pieces: list[str] = []
    for sentence in sentences:
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit + 1)
            cut = cut if cut > 0 else limit
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_top(docs: list[str], query: str, k: int) -> list[int]:
    tokenized = [_tokens(doc) for doc in docs]
    lengths = [len(tokens) for tokens in tokenized]
    average = sum(lengths) / len(lengths)
    frequencies = [Counter(tokens) for tokens in tokenized]
    count = len(docs)
    scores = [0.0] * count
    for term in set(_tokens(query)):
        containing = sum(1 for frequency in frequencies if term in frequency)
        if not containing:
            continue
        inverse = math.log((count - containing + 0.5) / (containing + 0.5) + 1.0)
        for index, frequency in enumerate(frequencies):
            tf = frequency.get(term, 0)
            if not tf:
                continue
            denominator = tf + 1.5 * (1 - 0.75 + 0.75 * lengths[index] / average)
            scores[index] += inverse * tf * 2.5 / denominator
    order = sorted(range(count), key=lambda index: (-scores[index], index))
    return [index for index in order if scores[index] > 0][:k]


def qwen_call(prompt: str) -> dict[str, Any]:
    body = json.dumps({
        "model": QWEN_MODEL, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0, "seed": 7, "num_predict": 512,
                    "num_ctx": min(262_144, int(len(prompt) / 3.2) + 4_000)},
    }).encode()
    request = urllib.request.Request(
        f"{OLLAMA}/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=3600) as response:
        value = json.loads(response.read())
    return {"response": value.get("response", ""),
            "latency_s": round(time.time() - started, 1),
            "prompt_tokens": value.get("prompt_eval_count"),
            "output_tokens": value.get("eval_count")}


def sonnet_call(prompt: str) -> dict[str, Any]:
    error = ""
    for attempt in range(3):
        started = time.time()
        try:
            process = subprocess.run(
                [CLAUDE, "-p", "--model", SONNET_MODEL, "--output-format",
                 "json", "--max-turns", "1", "--tools", "", "--strict-mcp-config"],
                input=prompt, text=True, capture_output=True, timeout=600,
            )
            value = json.loads(process.stdout)
            if value.get("is_error"):
                error = f"is_error: {str(value.get('result'))[:300]}"
                continue
            usage = value.get("usage") or {}
            return {"response": value.get("result", ""),
                    "latency_s": round(time.time() - started, 1),
                    "prompt_tokens": usage.get("input_tokens", 0)
                    + usage.get("cache_creation_input_tokens", 0)
                    + usage.get("cache_read_input_tokens", 0),
                    "output_tokens": usage.get("output_tokens")}
        except Exception as exc:
            error = str(exc)[:300]
        if attempt < 2:
            time.sleep(5)
    return {"response": "", "latency_s": 0.0, "prompt_tokens": None,
            "output_tokens": None, "transport_error": error}


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def formatted_prompt(task: str, packet: str, row: dict[str, Any]) -> str:
    prompt = DEFAULT_PROMPTS[task]
    return get_formatted_input(
        packet, row["question"], prompt["examples"], prompt["instruction"],
        prompt["post_prompt"], template=DEFAULT_TEMPLATE,
    )


def snapshot_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    with SNAPSHOT.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            hashes[value["sample_key"]] = value["source_sha256"]
    if len(hashes) != 4_000:
        raise RuntimeError(f"retrieval snapshot has {len(hashes)} samples, expected 4000")
    return hashes


def build_packets() -> None:
    hashes = snapshot_hashes()
    rows_out: list[dict[str, Any]] = []
    for task, size in OFFICIAL_CELLS:
        rows = json.loads((HERE / "data" / "data" / task / f"{size}.json").read_text())
        if len(rows) < N:
            raise RuntimeError(f"{task}/{size} has only {len(rows)} rows")
        for sample, row in enumerate(rows[:N]):
            key = f"{task}-{size}-{sample:03d}"
            source = row["input"]
            if sha_text(source) != hashes[key]:
                raise RuntimeError(f"{key}: local source differs from artifact-verified hash")
            chunks = chunk_text(source, CHUNK_CHARS)
            ranked = bm25_top(chunks, row["question"], K)
            if not ranked:
                raise RuntimeError(f"{key}: BM25 selected no chunks")
            # Match the artifact-hybrid arm's final packet ordering.
            top = sorted(ranked)
            packet = "\n\n".join(chunks[index] for index in top)
            prompt = formatted_prompt(task, packet, row)
            rows_out.append({
                "schema": "meko.babilong.bm25-800-packet.v1",
                "key": key, "task": task, "size": size, "sample": sample,
                "source_sha256": hashes[key], "question_sha256": sha_text(row["question"]),
                "target_sha256": sha_text(row["target"]), "chunks": len(chunks),
                "chunk_chars": CHUNK_CHARS, "top_k": K, "ranked_top": ranked,
                "top": top, "ordering": "source_order_after_selection",
                "packet_chars": len(packet), "packet_sha256": sha_text(packet),
                "prompt_chars": len(prompt), "prompt_sha256": sha_text(prompt),
            })
    RUN.mkdir(parents=True, exist_ok=True)
    temporary = PACKETS.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(PACKETS)
    manifest = {
        "schema": "meko.babilong.bm25-800-manifest.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "packets_complete", "samples": len(rows_out),
        "cells": [list(cell) for cell in OFFICIAL_CELLS],
        "arm": "bm25_800", "chunk_chars": CHUNK_CHARS, "top_k": K,
        "ranking": "plain Okapi BM25", "ordering": "source_order_after_selection",
        "source_contract": "local bytes must equal artifact-verified snapshot SHA-256",
        "datapack_create_calls": 0, "artifact_put_calls": 0,
        "retrieval_snapshot_sha256": sha_file(SNAPSHOT),
        "packet_jsonl_sha256": sha_file(PACKETS),
        "portable_runner_sha256": sha_file(Path(__file__)),
        "historical_chunker_sha256": "d06aa31b60f39e3431dc23970139be73a6a6e55363db1f4ca415fc91870e9e32",
        "historical_bm25_runner_sha256": "a9de810c3b72eae2407d1c7bc10fa2bf09acc949ebf63fe4a52c70d28a923755",
        "prompt_sha256": sha_file(HERE / "upstream" / "babilong" / "prompts.py"),
        "scorer_sha256": sha_file(HERE / "upstream" / "babilong" / "metrics.py"),
        "readers": list(READERS), "planned_trials": 4_000 * len(READERS),
    }
    atomic_json(PACKET_MANIFEST, manifest)
    print(f"built {len(rows_out)} matched packets sha256={manifest['packet_jsonl_sha256']}")


def packet_index() -> dict[str, dict[str, Any]]:
    if not PACKET_MANIFEST.exists() or not PACKETS.exists():
        raise RuntimeError("build packets first")
    manifest = json.loads(PACKET_MANIFEST.read_text())
    if sha_file(PACKETS) != manifest["packet_jsonl_sha256"]:
        raise RuntimeError("packet JSONL hash mismatch")
    return {row["key"]: row for row in map(json.loads, PACKETS.read_text().splitlines())}


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **row}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def latest_successes(path: Path) -> set[str]:
    successes: set[str] = set()
    if not path.exists():
        return successes
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("kind") == "trial" and row.get("success") is True:
            successes.add(row["key"])
    return successes


def load_trial(key: str, packet_row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    task, size, sample_text = key.rsplit("-", 2)
    sample = int(sample_text)
    row = json.loads((HERE / "data" / "data" / task / f"{size}.json").read_text())[sample]
    chunks = chunk_text(row["input"], CHUNK_CHARS)
    packet = "\n\n".join(chunks[index] for index in packet_row["top"])
    prompt = formatted_prompt(task, packet, row)
    if sha_text(packet) != packet_row["packet_sha256"] or sha_text(prompt) != packet_row["prompt_sha256"]:
        raise RuntimeError(f"{key}: replay mismatch")
    return row, prompt


def provider(reader: str, prompt: str, raw_dir: Path) -> dict[str, Any]:
    if reader == "qwen":
        result = qwen_call(prompt)
        return {"ok": not result.get("transport_error"), **result, "usage": {
            "total_input_tokens": result.get("prompt_tokens"),
            "output_tokens": result.get("output_tokens"),
        }}
    if reader == "sonnet":
        result = sonnet_call(prompt)
        transport_error = result.get("transport_error")
        return {"ok": not transport_error, **result,
                "failure_class": "transport" if transport_error else None,
                "error": transport_error, "usage": {
            "total_input_tokens": result.get("prompt_tokens"),
            "output_tokens": result.get("output_tokens"),
        }}
    if reader == "deepseek_v4_flash":
        result = call_deepseek(prompt, raw_dir)
    elif reader == "codex_luna":
        result = call_codex(prompt, raw_dir)
    else:
        result = call_gemini(prompt, raw_dir)
    return {
        "ok": result.ok, "response": result.response, "latency_s": result.latency_s,
        "usage": result.evaluation_usage, "native_usage": result.native_usage,
        "failure_class": result.failure_class, "error": result.error,
        "model_requested": result.model_requested, "model_resolved": result.model_resolved,
        "retry_after_s": result.retry_after_s, "fatal_contract": result.fatal_contract,
        "raw_paths": result.raw_paths,
    }


def run(reader: str, limit: int | None, only_key: str | None,
        shard_index: int = 0, shard_count: int = 1) -> None:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    packets = packet_index()
    run_dir = RESULT_ROOT / reader
    ledger = run_dir / "ledger.jsonl"
    attempts = run_dir / "attempts"
    successes = latest_successes(ledger)
    keys = [only_key] if only_key else sorted(packets, key=lambda k: (int(k.split('-')[0][2:]), ("0k", "4k", "16k").index(k.split('-')[1]), int(k.rsplit('-', 1)[1])))
    if not only_key and shard_count > 1:
        keys = [key for position, key in enumerate(keys)
                if position % shard_count == shard_index]
    completed = 0
    for key in keys:
        trial_key = f"{key}-bm25_800-{reader}"
        if trial_key in successes:
            continue
        row, prompt = load_trial(key, packets[key])
        final: dict[str, Any] | None = None
        for attempt in range(1, 4):
            attempt_id = uuid.uuid4().hex
            raw_dir = attempts / attempt_id
            append(ledger, {"kind": "attempt", "key": trial_key, "attempt": attempt, "attempt_id": attempt_id})
            try:
                final = provider(reader, prompt, raw_dir)
            except Exception as exc:  # transport failure; answer failures are never retried
                final = {"ok": False, "failure_class": "transport", "error": str(exc)[:500]}
            if final.get("ok"):
                break
            if final.get("fatal_contract"):
                raise RuntimeError(f"{trial_key}: fatal provider contract: {final.get('error')}")
            if attempt < 3:
                time.sleep(min(float(final.get("retry_after_s") or 5), 60))
        assert final is not None
        if not final.get("ok"):
            append(ledger, {"kind": "failure", "key": trial_key, **final})
            print(f"failure {trial_key}: {final.get('failure_class')} {final.get('error')}", flush=True)
            continue
        correct = compare_answers(row["target"], final.get("response", ""), row["question"], TASK_LABELS[packets[key]["task"]])
        append(ledger, {
            "kind": "trial", "key": trial_key, "success": True,
            "task": packets[key]["task"], "size": packets[key]["size"],
            "sample": packets[key]["sample"], "arm": "bm25_800", "reader": reader,
            "correct": bool(correct), "target": row["target"],
            "response": final.get("response", "")[:500],
            "packet_chars": packets[key]["packet_chars"],
            "packet_sha256": packets[key]["packet_sha256"],
            "prompt_sha256": packets[key]["prompt_sha256"],
            "usage": final.get("usage", {}), "native_usage": final.get("native_usage", {}),
            "latency_s": final.get("latency_s"), "model_requested": final.get("model_requested"),
            "model_resolved": final.get("model_resolved"), "raw_paths": final.get("raw_paths", {}),
        })
        successes.add(trial_key)
        completed += 1
        if completed % 10 == 0:
            print(f"{reader}: +{completed} this run, {len(successes)}/4000 complete", flush=True)
        if limit is not None and completed >= limit:
            break
    print(f"{reader}: {len(successes)}/4000 successful trials", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-packets", action="store_true")
    parser.add_argument("--reader", choices=READERS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-key")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.build_packets:
        build_packets()
    elif args.reader:
        run(args.reader, args.limit, args.only_key, args.shard_index, args.shard_count)
    else:
        parser.error("choose --build-packets or --reader")


if __name__ == "__main__":
    main()
