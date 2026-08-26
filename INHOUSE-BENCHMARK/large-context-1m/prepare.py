#!/usr/bin/env python3
"""Rebuild the deterministic 500-character evidence packets."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"\w+")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def chunks_of(text: str, size: int) -> list[str]:
    """Split on line boundaries, keeping each fact line intact."""
    chunks: list[str] = []
    buffer: list[str] = []
    used = 0
    for line in text.splitlines(keepends=True):
        if used + len(line) > size and buffer:
            chunks.append("".join(buffer))
            buffer = []
            used = 0
        buffer.append(line)
        used += len(line)
    if buffer:
        chunks.append("".join(buffer))
    return chunks


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, chunks: list[str], k1: float = 1.5,
                 b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
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
        scores: list[tuple[float, int]] = []
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


def deterministic_gzip(raw: bytes) -> bytes:
    """Match the stored artifact bytes without a timestamp in the header."""
    encoded = bytearray(gzip.compress(raw, compresslevel=9, mtime=0))
    encoded[9] = 255
    return bytes(encoded)


def build_manifest(base: Path) -> dict:
    history_path = base / "fixtures" / "history.txt"
    questions_path = base / "fixtures" / "questions.json"
    history_bytes = history_path.read_bytes()
    question_bytes = questions_path.read_bytes()
    questions = json.loads(question_bytes)
    chunks = chunks_of(history_bytes.decode(), 500)
    index = BM25(chunks)

    packet_rows: list[dict] = []
    ceiling: dict[str, list[int]] = {}
    for question_index, question in enumerate(questions):
        pieces: list[str] = []
        chunk_ids: list[int] = []
        used = 0
        for chunk_index in index.top(question["question"], 300):
            piece = chunks[chunk_index].rstrip() + "\n"
            if used + len(piece) > 150_000:
                break
            pieces.append(piece)
            chunk_ids.append(chunk_index)
            used += len(piece)
        context = "".join(pieces)
        artifact = deterministic_gzip(context.encode())
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
            "artifact_sha256": sha256(artifact),
            "artifact_bytes": len(artifact),
            "gold_in_context": present,
        })

    hits = sum(value[0] for value in ceiling.values())
    total = sum(value[1] for value in ceiling.values())
    return {
        "schema_version": 1,
        "selection": "local_deterministic_okapi_bm25",
        "storage": "meko_artifact_put_then_hash_fetch",
        "reader_delivery": "local_fixture_replay",
        "meko_semantic_retrieval": False,
        "history_path": "fixtures/history.txt",
        "history_sha256": sha256(history_bytes),
        "questions_path": "fixtures/questions.json",
        "questions_sha256": sha256(question_bytes),
        "chunking": "line-safe",
        "chunk_chars": 500,
        "chunks_in_corpus": len(chunks),
        "k": 300,
        "context_chars_cap": 150_000,
        "questions": len(questions),
        "ceiling": round(hits / total, 4),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    manifest = build_manifest(base)
    if args.output:
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "questions", "chunks_in_corpus", "ceiling_count",
        "mean_context_chars", "mean_artifact_bytes")}, indent=2))


if __name__ == "__main__":
    main()
