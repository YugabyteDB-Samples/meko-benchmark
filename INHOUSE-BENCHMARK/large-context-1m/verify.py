#!/usr/bin/env python3
"""Rebuild packet hashes and rescore the one-million-token result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import prepare
import scoring

ROOT = Path(__file__).resolve().parent

PACKET_FILES = {
    "sonnet5": "sonnet-5.jsonl",
    "opus5": "opus-5.jsonl",
    "codex-55": "gpt-5.5.jsonl",
    "codex-luna": "gpt-5.6-luna.jsonl",
    "codex-terra": "gpt-5.6-terra.jsonl",
    "codex-sol": "gpt-5.6-sol.jsonl",
    "gemini-37": "gemini-3.7-flash.jsonl",
    "glm": "glm-5.2.jsonl",
}

COMPACTION_FILES = {
    "sonnet5": "answers-claude.jsonl",
    "opus5": "answers-claude.jsonl",
    "codex-55": "answers-codex.jsonl",
    "codex-luna": "answers-codex.jsonl",
    "codex-terra": "answers-codex.jsonl",
    "codex-sol": "answers-codex.jsonl",
    "gemini-37": "answers-gemini.jsonl",
    "glm": "answers-glm.jsonl",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def score_ledger(path: Path, label: str, arm: str,
                 questions: list[dict]) -> tuple[int, int]:
    final: dict[int, dict] = {}
    failed_attempts = 0
    for row in read_jsonl(path):
        if row.get("label") != label or row.get("arm") != arm:
            continue
        if row.get("ok"):
            final[row["qidx"]] = row
        else:
            failed_attempts += 1
    assert len(final) == 320, (path, label, arm, len(final))
    correct = 0
    for question_index, row in final.items():
        observed = scoring.score_answer(
            questions[question_index], row.get("text", ""))
        assert observed == bool(row.get("correct")), (path, question_index)
        correct += int(observed)
    return correct, failed_attempts


def portable_manifest(manifest: dict) -> dict:
    """Remove gzip-build metadata before comparing regenerated packets."""
    portable = {key: value for key, value in manifest.items()
                if key != "mean_artifact_bytes"}
    portable["packets"] = [
        {key: value for key, value in row.items()
         if key not in {"artifact_sha256", "artifact_bytes"}}
        for row in manifest["packets"]
    ]
    return portable


def rebuild_packets(questions: list[dict]) -> tuple[dict, list[str]]:
    stored = json.loads((ROOT / "manifests/packet-manifest.json").read_text())
    rebuilt = prepare.build_manifest(ROOT)
    # gzip output is not portable across all zlib builds. The packet text is.
    assert portable_manifest(rebuilt) == portable_manifest(stored)

    history = (ROOT / "fixtures/history.txt").read_text()
    chunks = prepare.chunks_of(history, stored["chunk_chars"])
    contexts: list[str] = []
    for row in stored["packets"]:
        context = "".join(chunks[index].rstrip() + "\n"
                          for index in row["chunk_ids"])
        assert len(context) == row["context_chars"]
        assert sha256(context.encode()) == row["context_sha256"]
        assert prepare.gold_in_context(questions[row["idx"]], context) == \
            row["gold_in_context"]
        contexts.append(context)
    print("PASS rebuilt 320 prepared packets and portable content hashes")
    return stored, contexts


def verify_reader_budgets(questions: list[dict],
                          contexts: list[str]) -> None:
    packet_manifest = json.loads(
        (ROOT / "manifests/packet-manifest.json").read_text())
    expected_hashes = {
        row["idx"]: row["artifact_sha256"]
        for row in packet_manifest["packets"]
    }
    readbacks = read_jsonl(
        ROOT / "manifests/meko-packet-readback.jsonl")
    assert len(readbacks) == 320
    assert {row["idx"] for row in readbacks} == set(range(320))
    assert all(row["ok"] for row in readbacks)
    assert all(row["hash"] == expected_hashes[row["idx"]]
               for row in readbacks)
    print("PASS 320 per-packet Meko put/get records match artifact hashes")

    budgets = json.loads((ROOT / "manifests/reader-budgets.json").read_text())
    report = json.loads((ROOT / "manifests/meko-readback-record.json").read_text())
    assert report["status"] == "recorded_complete"
    assert report["packets_expected"] == report["packets_fetched"] == 320
    assert report["packet_mismatches"] == []
    fetched = {row["label"]: row for row in report["reader_fixtures"]}
    assert len(fetched) == 8

    for reader in budgets["readers"]:
        digest = ROOT / reader["digest"]
        cap = len(digest.read_text())
        assert cap == reader["context_chars_cap"]
        assert sha256(digest.read_bytes()) == reader["digest_sha256"]
        capped = {str(index): context[:cap]
                  for index, context in enumerate(contexts)}
        fixture_hash = sha256(json.dumps(capped).encode())
        recorded = fetched[reader["label"]]
        assert recorded["context_chars_cap"] == cap
        assert recorded["byte_identical"]
        assert fixture_hash == recorded["fixture_sha256"]
        assert fixture_hash == recorded["derived_sha256"]

        hits = 0
        total = 0
        for index, question in enumerate(questions):
            present = prepare.gold_in_context(question, capped[str(index)])
            if present is not None:
                hits += int(present)
                total += 1
        assert f"{hits}/{total}" == reader["evidence_ceiling"]
    print("PASS eight digest caps, evidence ceilings, and fixture hashes")
    print("PASS saved Meko report: 320/320 fetches, no reported mismatches; "
          "eight fixture hashes reproduce")


def verify_answers(questions: list[dict]) -> None:
    results = json.loads((ROOT / "results.json").read_text())
    complete = [row for row in results["readers"]
                if row["prepared_packet_correct"] is not None]
    assert len(complete) == 8
    for row in complete:
        label = row["label"]
        packet_score, _ = score_ledger(
            ROOT / "ledgers/packet" / PACKET_FILES[label],
            label, "fast_packet", questions)
        compaction_score, _ = score_ledger(
            ROOT / "ledgers/compaction" / COMPACTION_FILES[label],
            label, "own_digest", questions)
        assert packet_score == row["prepared_packet_correct"]
        assert compaction_score == row["compaction_correct"]
        print(f"PASS {row['reader']}: prepared {packet_score}/320; "
              f"compaction {compaction_score}/320")

    missing = json.loads((ROOT / "manifests/minimax-status.json").read_text())
    assert missing["status"] == "incomplete"
    assert missing["digest_chunks_completed"] == 19
    assert missing["digest_chunks_expected"] == 21
    assert missing["missing_chunk_indexes"] == [0, 10]
    minimax = next(row for row in results["readers"]
                   if row["label"] == "minimax")
    assert minimax["prepared_packet_correct"] is None
    assert minimax["compaction_correct"] is None
    print("PASS MiniMax is recorded as incomplete, with no score")


def verify_compaction_usage() -> None:
    results = json.loads((ROOT / "results.json").read_text())
    for reader, filename in (("Claude Sonnet 5", "sonnet-5.json"),
                             ("Claude Opus 5", "opus-5.json")):
        row = json.loads((ROOT / "compaction-usage" / filename).read_text())
        tokens = row["compaction_input_tokens"] + row["compaction_output_tokens"]
        expected = results["compaction_usage"][reader]
        assert tokens == expected["reported_tokens"]
        assert row["compaction_cost_usd"] == expected["cost_usd"]
    print("PASS Sonnet and Opus compaction token and cost totals")


def main() -> None:
    questions = json.loads((ROOT / "fixtures/questions.json").read_text())
    assert len(questions) == 320
    assert len((ROOT / "fixtures/history.txt").read_text()) == 4_000_017
    _, contexts = rebuild_packets(questions)
    verify_reader_budgets(questions, contexts)
    verify_answers(questions)
    verify_compaction_usage()


if __name__ == "__main__":
    main()
