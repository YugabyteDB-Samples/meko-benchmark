#!/usr/bin/env python3
"""Verify, summarize, and merge portable BABILong BM25-800 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "runs" / "babilong-bm25-800"
READERS = (
    "qwen", "sonnet", "deepseek_v4_flash", "codex_luna",
    "gemini_3_7_flash_low",
)
SIZES = ("0k", "4k", "16k")
OFFICIAL_CELLS = tuple(
    (f"qa{i}", size) for i in range(1, 11) for size in SIZES
) + tuple((f"qa{i}", "0k") for i in range(11, 21))


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_packets(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = root / "packet-manifest.json"
    packets_path = root / "packets.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "meko.babilong.bm25-800-manifest.v1":
        raise RuntimeError("unexpected packet manifest schema")
    if manifest.get("chunk_chars") != 800 or manifest.get("top_k") != 5:
        raise RuntimeError("packet manifest is not the frozen 800-character/top-five control")
    actual_hash = sha_file(packets_path)
    if actual_hash != manifest.get("packet_jsonl_sha256"):
        raise RuntimeError("packets.jsonl SHA-256 differs from packet manifest")
    packets: dict[str, dict[str, Any]] = {}
    with packets_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            key = row.get("key")
            if not isinstance(key, str) or key in packets:
                raise RuntimeError(f"invalid or duplicate packet key at line {line_number}")
            packets[key] = row
    expected = {
        f"{task}-{size}-{sample:03d}"
        for task, size in OFFICIAL_CELLS
        for sample in range(100)
    }
    if set(packets) != expected:
        missing = sorted(expected - set(packets))[:5]
        extra = sorted(set(packets) - expected)[:5]
        raise RuntimeError(f"packet key set mismatch: missing={missing}, extra={extra}")
    return manifest, packets


def _trial_sample_key(row: dict[str, Any]) -> str:
    return f"{row['task']}-{row['size']}-{int(row['sample']):03d}"


def load_reader_ledger(
    path: Path,
    reader: str,
    packets: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str], int]:
    trials: dict[str, dict[str, Any]] = {}
    failures: set[str] = set()
    duplicate_trials = 0
    if not path.exists():
        return trials, failures, duplicate_trials
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            key = row.get("key")
            if row.get("kind") == "failure" and isinstance(key, str):
                failures.add(key)
                continue
            if row.get("kind") != "trial" or row.get("success") is not True:
                continue
            if row.get("reader") != reader or row.get("arm") != "bm25_800":
                raise RuntimeError(f"{path}:{line_number}: reader/arm contract mismatch")
            sample_key = _trial_sample_key(row)
            packet = packets.get(sample_key)
            expected_key = f"{sample_key}-bm25_800-{reader}"
            if packet is None or key != expected_key:
                raise RuntimeError(f"{path}:{line_number}: unexpected trial key {key!r}")
            for field in ("packet_sha256", "prompt_sha256"):
                if row.get(field) != packet.get(field):
                    raise RuntimeError(f"{path}:{line_number}: {field} mismatch")
            if key in trials:
                duplicate_trials += 1
                old = trials[key]
                comparable = ("correct", "target", "response", "packet_sha256", "prompt_sha256")
                if any(old.get(field) != row.get(field) for field in comparable):
                    raise RuntimeError(f"conflicting successful duplicate for {key}")
            trials[key] = row
            failures.discard(key)
    return trials, failures, duplicate_trials


def input_tokens(row: dict[str, Any]) -> int | None:
    usage = row.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("total_input_tokens")
    if value is None:
        return None
    return int(value)


def summarize_reader(
    reader: str,
    trials: dict[str, dict[str, Any]],
    failures: set[str],
    duplicates: int,
) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trials.values():
        by_cell[(row["task"], row["size"])].append(row)
    for task, size in OFFICIAL_CELLS:
        rows = by_cell[(task, size)]
        tokens = [value for row in rows if (value := input_tokens(row)) is not None]
        cells[f"{task}/{size}"] = {
            "successful": len(rows),
            "correct": sum(bool(row.get("correct")) for row in rows),
            "accuracy_percent": round(100 * sum(bool(row.get("correct")) for row in rows) / len(rows), 1) if rows else None,
            "input_token_rows": len(tokens),
            "total_input_tokens": sum(tokens) if len(tokens) == len(rows) and rows else None,
            "mean_input_tokens": round(statistics.fmean(tokens), 1) if len(tokens) == len(rows) and rows else None,
            "complete": len(rows) == 100,
        }
    return {
        "reader": reader,
        "successful": len(trials),
        "planned": 4000,
        "complete_cells": sum(cell["complete"] for cell in cells.values()),
        "planned_cells": 40,
        "current_failures": len(failures),
        "duplicate_success_rows": duplicates,
        "cells": cells,
    }


def verify_packet_replay(root: Path, packets: dict[str, dict[str, Any]]) -> None:
    import run_babilong_bm25_800 as runner

    if root.resolve() != runner.RUN.resolve():
        raise RuntimeError("full packet replay requires the package's default packet root")
    for position, (key, packet) in enumerate(sorted(packets.items()), 1):
        task, size, sample_text = key.rsplit("-", 2)
        row = json.loads(
            (runner.HERE / "data" / "data" / task / f"{size}.json").read_text(encoding="utf-8")
        )[int(sample_text)]
        if runner.sha_text(row["input"]) != packet["source_sha256"]:
            raise RuntimeError(f"{key}: source SHA-256 mismatch")
        chunks = runner.chunk_text(row["input"], 800)
        ranked = runner.bm25_top(chunks, row["question"], 5)
        if ranked != packet["ranked_top"] or sorted(ranked) != packet["top"]:
            raise RuntimeError(f"{key}: BM25 selection mismatch")
        selected = "\n\n".join(chunks[index] for index in packet["top"])
        prompt = runner.formatted_prompt(task, selected, row)
        if runner.sha_text(selected) != packet["packet_sha256"]:
            raise RuntimeError(f"{key}: selected packet SHA-256 mismatch")
        if runner.sha_text(prompt) != packet["prompt_sha256"]:
            raise RuntimeError(f"{key}: prompt SHA-256 mismatch")
        if position % 500 == 0:
            print(f"verified {position}/4000 packets", flush=True)


def markdown(summary: dict[str, Any], readers: list[str]) -> str:
    lines = ["# BABILong matched BM25-800 evidence", ""]
    lines += ["| Reader | Successful | Complete cells | Current failures |", "| --- | ---: | ---: | ---: |"]
    for reader in readers:
        row = summary["readers"][reader]
        lines.append(f"| {reader} | {row['successful']}/4,000 | {row['complete_cells']}/40 | {row['current_failures']} |")
    for qa in range(1, 21):
        task = f"qa{qa}"
        sizes = SIZES if qa <= 10 else ("0k",)
        lines += ["", f"## {task.upper()}", "", "| Size | Reader | Accuracy | Mean input tokens | Status |", "| --- | --- | ---: | ---: | --- |"]
        for size in sizes:
            for reader in readers:
                cell = summary["readers"][reader]["cells"][f"{task}/{size}"]
                accuracy = f"{cell['correct']}/{cell['successful']} ({cell['accuracy_percent']}%)" if cell["successful"] else "—"
                tokens = f"{cell['mean_input_tokens']:.1f}" if cell["mean_input_tokens"] is not None else "—"
                status = "COMPLETE" if cell["complete"] else f"PARTIAL {cell['successful']}/100"
                lines.append(f"| {size} | {reader} | {accuracy} | {tokens} | {status} |")
    return "\n".join(lines) + "\n"


def merge_ledgers(reader: str, inputs: Iterable[Path], output: Path) -> None:
    rows: list[dict[str, Any]] = []
    seen_lines: set[str] = set()
    successful: dict[str, dict[str, Any]] = {}
    for path in inputs:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if row.get("reader") not in (None, reader):
                    raise RuntimeError(f"{path}:{line_number}: unexpected reader")
                encoded = canonical(row)
                if encoded in seen_lines:
                    continue
                seen_lines.add(encoded)
                if row.get("kind") == "trial" and row.get("success") is True:
                    key = row["key"]
                    if key in successful:
                        old = successful[key]
                        comparable = ("correct", "target", "response", "packet_sha256", "prompt_sha256")
                        if any(old.get(field) != row.get(field) for field in comparable):
                            raise RuntimeError(f"conflicting successful result for {key}")
                    successful[key] = row
                rows.append(row)
    rows.sort(key=lambda row: (str(row.get("ts", "")), str(row.get("key", "")), str(row.get("kind", ""))))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(output)
    print(f"merged {len(inputs)} ledgers into {output}; {len(successful)} unique successful trials")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--readers", nargs="+", choices=READERS, default=list(READERS))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verify-packets", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--merge-reader", choices=READERS)
    parser.add_argument("--merge-out", type=Path)
    parser.add_argument("--merge-ledgers", nargs="+", type=Path)
    args = parser.parse_args()

    if args.merge_ledgers:
        if not args.merge_reader or not args.merge_out:
            parser.error("--merge-ledgers requires --merge-reader and --merge-out")
        merge_ledgers(args.merge_reader, args.merge_ledgers, args.merge_out)
        return

    root = args.root.resolve()
    manifest, packets = load_packets(root)
    if args.verify_packets:
        verify_packet_replay(root, packets)
    reader_rows: dict[str, Any] = {}
    for reader in args.readers:
        trials, failures, duplicates = load_reader_ledger(root / reader / "ledger.jsonl", reader, packets)
        reader_rows[reader] = summarize_reader(reader, trials, failures, duplicates)
    summary = {
        "schema": "meko.babilong.bm25-800-evidence.v1",
        "packet_manifest_sha256": sha_file(root / "packet-manifest.json"),
        "packets_sha256": manifest["packet_jsonl_sha256"],
        "readers": reader_rows,
    }
    rendered = markdown(summary, args.readers)
    print(rendered, end="")
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.write_text(rendered, encoding="utf-8")
    if args.strict:
        incomplete = [reader for reader in args.readers if reader_rows[reader]["successful"] != 4000]
        if incomplete:
            raise SystemExit(f"strict verification failed; incomplete readers: {', '.join(incomplete)}")


if __name__ == "__main__":
    main()
