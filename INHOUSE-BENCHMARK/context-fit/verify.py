#!/usr/bin/env python3
"""Recompute the context-fit results from the published JSONL ledgers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARMS = ("meko_optimized", "bm25_recipe", "full_history")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_arm(records: list[dict], gold: dict[str, str], arm: str) -> dict:
    batches = [row for row in records if row["arm"] == arm]
    assert len(batches) == 64, (arm, len(batches))
    assert all(row["event"] == "batch" and row["ok"] is True for row in batches)
    assert all("errors" not in row for row in batches)

    requested = [qid for row in batches for qid in row["qids"]]
    assert len(requested) == len(set(requested)) == len(gold)
    assert set(requested) == set(gold)

    answers: dict[str, str] = {}
    for row in batches:
        overlap = set(answers) & set(row["answers"])
        assert not overlap, (arm, sorted(overlap))
        answers.update(row["answers"])
    assert not (set(answers) - set(gold))

    billed_input = sum(row["usage"]["billed_input_tokens"] for row in batches)
    output = sum(row["usage"]["output_tokens"] for row in batches)
    context_characters = sum(row["context_chars"] for row in batches)
    return {
        "answered": len(answers),
        "correct": sum(answer == gold[qid] for qid, answer in answers.items()),
        "billed_input_tokens": billed_input,
        "output_tokens": output,
        "total_tokens": billed_input + output,
        "cost_usd": round(sum(row["cost_usd"] for row in batches), 4),
        "context_characters": context_characters,
        "mean_context_characters_per_question": context_characters / len(gold),
    }


def validate_fixtures() -> tuple[list[dict], dict[str, str]]:
    fixtures = ROOT / "fixtures"
    meta = json.loads((fixtures / "meta.json").read_text())
    statements_path = fixtures / "statements.jsonl"
    questions_path = fixtures / "questions.jsonl"
    statements = read_jsonl(statements_path)
    questions = read_jsonl(questions_path)

    assert len(statements) == meta["statements"] == 615
    assert len(questions) == meta["questions"] == 320
    assert sha256(statements_path) == meta["statements_sha256"]
    assert sha256(questions_path) == meta["questions_sha256"]
    assert len({row["qid"] for row in questions}) == len(questions)
    gold = {row["qid"]: row["answer"] for row in questions}
    return questions, gold


def validate_retrieval(questions: list[dict]) -> None:
    rows = read_jsonl(ROOT / "retrieval.jsonl")
    by_qid = {row["qid"]: row for row in questions}
    assert len(rows) == len(by_qid) == 320
    assert {row["qid"] for row in rows} == set(by_qid)
    for row in rows:
        assert row["event"] == "retrieval"
        assert row["question_row"] == by_qid[row["qid"]]
        assert row["meko"]["ok"] is True
        assert "errors" not in row["meko"]
        assert len(row["meko"]["hits"]) == 25
        assert len(row["bm25_hits"]) == 25
        assert all("id" not in hit for hit in row["meko"]["hits"])


def main() -> None:
    expected = json.loads((ROOT / "results.json").read_text())
    questions, gold = validate_fixtures()
    validate_retrieval(questions)

    observed_by_reader: dict[str, dict] = {}
    context_means: dict[str, set[float]] = {arm: set() for arm in ARMS}
    for reader in expected["readers"]:
        records = read_jsonl(ROOT / "ledgers" / reader["file"])
        assert len(records) == 192, (reader["reader"], len(records))
        observed = {arm: score_arm(records, gold, arm) for arm in ARMS}
        for arm in ARMS:
            assert observed[arm]["correct"] == reader["correct"][arm]
            assert observed[arm]["answered"] == reader["answered"][arm]
            context_means[arm].add(
                observed[arm]["mean_context_characters_per_question"]
            )

        reduction = round(
            100
            * (
                1
                - observed["meko_optimized"]["total_tokens"]
                / observed["full_history"]["total_tokens"]
            ),
            1,
        )
        assert reduction == reader["meko_token_reduction_vs_full_history_percent"]
        observed_by_reader[reader["reader"]] = observed

    for arm in ARMS:
        assert len(context_means[arm]) == 1, (arm, context_means[arm])
    expected_context = expected["mean_context_characters_per_question"]
    meko_context = context_means["meko_optimized"].pop()
    full_context = context_means["full_history"].pop()
    assert meko_context == expected_context["meko_optimized"]
    assert full_context == expected_context["full_history"]
    context_reduction = round(100 * (1 - meko_context / full_context), 1)
    assert context_reduction == expected_context["meko_reduction_vs_full_history_percent"]

    for reader_name, costs in expected["claude_billed_cost_usd"].items():
        observed = observed_by_reader[reader_name]
        assert observed["meko_optimized"]["cost_usd"] == costs["meko_optimized"]
        assert observed["full_history"]["cost_usd"] == costs["full_history"]
        reduction = round(
            100
            * (
                1
                - observed["meko_optimized"]["cost_usd"]
                / observed["full_history"]["cost_usd"]
            ),
            1,
        )
        assert reduction == costs["reduction_percent"]

    print("PASS context-fit fixtures: 615 statements, 320 questions")
    print("PASS retrieval snapshot: 320 Meko and BM25 result sets")
    print("PASS reader ledgers: 9 readers, 3 arms, 320 questions per arm")
    print("PASS 27 accuracy cells and 9 token-reduction values")
    print("PASS context-size means and Claude billed-cost reductions")


if __name__ == "__main__":
    main()
