#!/usr/bin/env python3
"""Answer scoring for the polluted-corpus questions — the published reference.

Exact containment against globally unique values, so there is no judge model
and no partial-credit argument. Three ways a reader can pass such a scorer
without knowing the answer, and what stops each:

1. **Fabrication.** A reader that invents a well-formed value for a fact that
   was never stated must score wrong, not creative. The value-token pattern is
   deliberately wider than the real scheme so invented codes of the wrong
   length still register as fabrications.
2. **Spray-and-pray.** Because values are globally unique, "the answer is one
   of A, B, C, D" approaches 100% on a naive containment check. A single-answer
   question therefore fails when more than one distinct value token appears,
   and a list question fails when more appear than the gold list holds.
3. **Hedging into abstention.** "I cannot be fully certain" satisfies a loose
   absence-phrase check. Abstention requires an absence phrase AND zero value
   tokens.

A fourth, for questions that fix an order: a set-containment check accepts the
right values listed in the wrong sequence, which is precisely what a question
asking for a history oldest-to-newest is testing. Questions carrying
``ordered: true`` are scored on the order of the values in the answer.

A scorer carried over from a different corpus is unvalidated until exploits
have been attempted against it. Run ``python3 scoring.py`` for the regression
tests that hold these three shut.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Value-shaped tokens. VZ\d{4,7} is deliberately wider than the real 5-digit
# scheme so invented codes of the wrong length still count as fabrications.
# The non-VZ alternatives are earlier value schemes, kept so the scorer stays
# valid for corpora that used them.
VALUE_TOKEN = re.compile(r"\b(VZ\d{4,7}|Project \w+|reg-\w+|rota-\w+|ff-\w+|"
                         r"cohort-\w+|ckpt-\d+|tb-\d+|pool-\w+|host-\w+|"
                         r"ttl-\d+\w*)\b")

_CODENAME_POOL: list[str] | None = None


def _codename_pool() -> list[str]:
    """Natural-answer value vocabulary, shipped by the generator. Absent
    fixture means a corpus without natural values; the guard then never
    fires, which keeps this scorer valid for the older corpora too."""
    global _CODENAME_POOL
    if _CODENAME_POOL is None:
        fix = Path(os.environ.get("XC1_FIXTURES",
                                  Path(__file__).resolve().parent / "fixtures"))
        p = fix / "pol_codenames.json"
        _CODENAME_POOL = (json.loads(p.read_text())["codenames"]
                          if p.exists() else [])
    return _CODENAME_POOL


ABSENT_PHRASES = ("not found", "no information", "not in", "not provided",
                  "cannot find", "cannot answer", "cannot locate",
                  "cannot determine", "cannot say", "don't have", "no record",
                  "not available", "not mention")


def score_answer(q: dict, text: str, codename_pool: list[str] | None = None) -> bool:
    t = text.lower()
    s = q["score"]
    tokens = {m.group(0).lower() for m in VALUE_TOKEN.finditer(text)}
    # Natural-language values are value tokens too: a reader listing several
    # codenames is spraying exactly as one listing several VZ codes is.
    pool = codename_pool if codename_pool is not None else _codename_pool()
    tokens |= {c.lower() for c in pool if c.lower() in t}
    if s == "contains":
        # One question, one answer: extra value tokens are spray, not knowledge.
        return q["answer"].lower() in t and len(tokens) <= 1
    if s == "contains_all":
        gold = [v.lower() for v in q["answer_list"]]
        if not all(v in t for v in gold) or len(tokens) > len(set(gold)):
            return False
        # When the question fixes an order ("oldest to newest"), listing the
        # right values in the wrong order is a wrong answer: it is the ordering
        # that the question tests.
        if q.get("ordered"):
            at = [t.index(v) for v in gold]
            return at == sorted(at)
        return True
    if s == "abstain":
        return not tokens and any(p in t for p in ABSENT_PHRASES)
    return False


def gold_in_context(q: dict, ctx: str) -> bool | None:
    """Arm ceiling probe: is the gold answer even present in the context an
    arm hands the reader? Measurable with zero readers; publish it alongside
    every arm's results. Returns None for UNANSWERABLE (no gold exists)."""
    t = ctx.lower()
    if q["score"] == "contains":
        return q["answer"].lower() in t
    if q["score"] == "contains_all":
        return all(v.lower() in t for v in q["answer_list"])
    return None


if __name__ == "__main__":
    qc = {"score": "contains", "answer": "VZ15260"}
    ql = {"score": "contains_all", "answer_list": ["VZ15261", "VZ15262"]}
    qo = {"score": "contains_all", "answer_list": ["VZ15261", "VZ15262", "VZ15263"],
          "ordered": True}
    qa = {"score": "abstain"}
    checks = [
        ("fabricated code in abstain rejected",
         not score_answer(qa, "Not found in the notes, though it might be VZ33999")),
        ("spray-and-pray on contains rejected",
         not score_answer(qc, "Candidates: VZ15260 VZ15261 VZ15250 VZ25260")),
        ("bare hedge on abstain rejected",
         not score_answer(qa, "I cannot be fully certain about that one")),
        ("spray on contains_all rejected",
         not score_answer(ql, "VZ15261 VZ15262 VZ15263 VZ15264")),
        ("honest single answer passes",
         score_answer(qc, "The value is VZ15260")),
        ("honest repeated mention passes",
         score_answer(qc, "VZ15260 — confirmed, VZ15260")),
        ("honest list answer passes",
         score_answer(ql, "Oldest to newest: VZ15261, then VZ15262")),
        ("ordered question accepts the requested order",
         score_answer(qo, "VZ15261, VZ15262, VZ15263")),
        ("ordered question rejects the reversed order",
         not score_answer(qo, "VZ15263, VZ15262, VZ15261")),
        ("ordered question rejects a shuffled order",
         not score_answer(qo, "VZ15262 VZ15261 VZ15263")),
        ("unordered list still ignores order",
         score_answer(ql, "VZ15262 and VZ15261")),
        ("honest abstention passes",
         score_answer(qa, "No record of that fact in the notes.")),
        ("six-digit invented code still counts as fabrication",
         not score_answer(qa, "not found, possibly VZ339999")),
    ]
    # Natural-answer values: the same three exploits, in ordinary language.
    NP = ["Amber Falcon", "Cobalt Heron", "Umber Lynx"]
    qn = {"score": "contains", "answer": "Amber Falcon"}
    qno = {"score": "contains_all", "answer_list": ["Amber Falcon", "Cobalt Heron"],
           "ordered": True}
    checks += [
        ("honest single codename answer passes",
         score_answer(qn, "Their room is Amber Falcon.", codename_pool=NP)),
        ("codename spray on contains rejected",
         not score_answer(qn, "It is Amber Falcon or maybe Cobalt Heron",
                          codename_pool=NP)),
        ("mixed code and codename spray rejected",
         not score_answer(qn, "Amber Falcon, or VZ123456", codename_pool=NP)),
        ("codename mention defeats a hedged abstention",
         not score_answer(qa, "no record, though Umber Lynx comes up",
                          codename_pool=NP)),
        ("ordered codename sequence accepted in order",
         score_answer(qno, "Amber Falcon then Cobalt Heron", codename_pool=NP)),
        ("ordered codename sequence rejected reversed",
         not score_answer(qno, "Cobalt Heron then Amber Falcon",
                          codename_pool=NP)),
    ]
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    raise SystemExit(1 if failed else 0)
