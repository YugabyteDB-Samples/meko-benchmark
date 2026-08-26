#!/usr/bin/env python3
"""Fetch prepared packets by hash and prove their relation to reader fixtures.

The reader runs replayed local frozen fixtures. This verifier does not rewrite
that history: it proves whether the bytes currently fetched from Meko match the
deterministic packet bytes, then derives every capped reader fixture from those
fetched bytes and compares it byte-for-byte with the fixture used by the run.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from meko_client import MekoMCPClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "runs-live"
FIX = ROOT / "generated"
STAMP = "xc1-c500-fast"
REQUEST_INTERVAL_S = 0.75
RETRIES = 4
_rate_lock = threading.Lock()
_next_request = 0.0


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", default=STAMP)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or RUNS / f"meko-fetch-verification-{args.stamp}.json"
    if not os.environ.get("MEKO_API_KEY"):
        report = {"stamp": args.stamp,
                  "status": "not_run_missing_credentials",
                  "claim": "reader fixtures were local; Meko equivalence unverified"}
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        raise SystemExit(2)

    packet_manifest_path = RUNS / f"fast-packet-manifest-{args.stamp}.json"
    packet_manifest = json.loads(packet_manifest_path.read_text())
    meko = packet_manifest["meko"]
    local_context_path = FIX / f"meko_fast_ctx-{args.stamp}.json"
    local_contexts = json.loads(local_context_path.read_text())
    reader_manifest_path = (FIX / "fast-reader" /
                            f"reader-context-manifest-{args.stamp}.json")
    reader_manifest = json.loads(reader_manifest_path.read_text())
    packet_rows = {row["idx"]: row for row in packet_manifest["packets"]}

    def fetch(index: int) -> tuple[int, bytes]:
        row = packet_rows[index]
        client = MekoMCPClient.from_env()
        errors = []
        for attempt in range(1, RETRIES + 1):
            try:
                payload = call_limited(client, "artifact_get", {
                    "content_hash": row["artifact_sha256"],
                    "conversation_id": meko["conversation"],
                    "datapack_id": meko["datapack"],
                })
                raw = artifact_bytes(payload)
                if raw is None:
                    raise RuntimeError("artifact_get returned no bytes")
                return index, raw
            except Exception as error:  # noqa: BLE001
                message = f"{type(error).__name__}: {str(error)[:240]}"
                errors.append(message)
                time.sleep(65 if "PAT_RATE_LIMITED" in message else attempt)
        raise RuntimeError(f"packet {index}: fetch failed: {errors}")

    fetched: dict[int, bytes] = {}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, index) for index in sorted(packet_rows)]
        for completed, future in enumerate(as_completed(futures), 1):
            index, raw = future.result()
            fetched[index] = raw
            if completed % 20 == 0 or completed == len(futures):
                elapsed = time.monotonic() - started
                print(f"fetched {completed}/{len(futures)} in {elapsed:.1f}s",
                      flush=True)

    packet_mismatches = []
    fetched_contexts: dict[str, str] = {}
    for index in sorted(packet_rows):
        row = packet_rows[index]
        raw = fetched[index]
        local_text = local_contexts[str(index)]
        expected = gzip.compress(local_text.encode(), compresslevel=9, mtime=0)
        try:
            fetched_text = gzip.decompress(raw).decode()
        except Exception as error:  # noqa: BLE001
            packet_mismatches.append({"idx": index,
                                      "error": f"gzip decode: {error}"})
            continue
        checks = {
            "artifact_hash": sha256(raw) == row["artifact_sha256"],
            "artifact_bytes": raw == expected,
            "context_hash": sha256(fetched_text.encode()) ==
                            row["context_sha256"],
            "context_bytes": fetched_text == local_text,
        }
        if not all(checks.values()):
            packet_mismatches.append({"idx": index, "checks": checks})
        fetched_contexts[str(index)] = fetched_text

    reader_checks = []
    for reader in reader_manifest["readers"]:
        cap = reader["context_chars_cap"]
        derived = {str(index): fetched_contexts[str(index)][:cap]
                   for index in range(len(packet_rows))}
        derived_bytes = json.dumps(derived).encode()
        fixture_path = ROOT / reader["context_file"]
        fixture_bytes = fixture_path.read_bytes()
        reader_checks.append({
            "label": reader["label"],
            "context_chars_cap": cap,
            "fixture_path": str(fixture_path.relative_to(ROOT)),
            "fixture_sha256": sha256(fixture_bytes),
            "derived_sha256": sha256(derived_bytes),
            "byte_identical": derived_bytes == fixture_bytes,
        })

    report = {
        "stamp": args.stamp,
        "status": "verified" if not packet_mismatches and
                  all(row["byte_identical"] for row in reader_checks)
                  else "failed",
        "scope": {
            "meko_fetch": "artifact_get by manifest SHA-256",
            "reader_delivery": "local frozen fixtures; no claim that reader "
                               "calls fetched Meko directly",
        },
        "packet_manifest_sha256": sha256(packet_manifest_path.read_bytes()),
        "local_uncapped_fixture_sha256": sha256(local_context_path.read_bytes()),
        "packets_expected": len(packet_rows),
        "packets_fetched": len(fetched),
        "packet_mismatches": packet_mismatches,
        "reader_fixtures": reader_checks,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"],
                      "packets_fetched": len(fetched),
                      "packet_mismatches": len(packet_mismatches),
                      "reader_fixtures_identical": sum(
                          row["byte_identical"] for row in reader_checks),
                      "reader_fixtures_total": len(reader_checks),
                      "report": str(report_path)}, indent=2))
    if report["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
