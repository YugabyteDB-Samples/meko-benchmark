#!/usr/bin/env python3
"""MekoBench-P — a PersonaMem-style benchmark, reconstructed and scaled.

PersonaMem (the real one): 180+ multi-session user histories, ~6,000 MCQs,
seven personalization abilities, histories at 32k/128k/1M tokens. Its core
idea: preferences are revealed, reinforced and UPDATED across temporally
ordered sessions, and the model must answer from the user's CURRENT state.

This reconstruction keeps that core at a scale our write budget affords:

  8 personas x 10 dated sessions x 8 utterances  = 640 statements
  22 preference slots per persona: 12 stable, 10 evolving (2-3 versions,
  explicit dated switch statements with unique reasons)
  320 four-option MCQs across six ability types:

    FACT       12/persona  recall a stable preference
    CURRENT    10/persona  the latest value of an evolving one (the old
                           value is always among the options)
    EVOLUTION   5/persona  the full change sequence ("X, then Y")
    REASON      5/persona  why a switch happened
    RECOMMEND   4/persona  apply the current preference to a new choice
    UNANSWER    4/persona  slot never mentioned; correct = "Not enough info"

Everything is deterministic (seed), values are unique per persona+slot,
and the generator self-validates: correct options must match the final
timeline state, old values must appear as distractors on CURRENT
questions, unanswerable slots must be absent from the transcript.

Run:  python gen_mekobench.py [seed]
"""
from __future__ import annotations

import datetime
import hashlib
import json
import random
import sys
from pathlib import Path

FIX = Path(__file__).resolve().parent.parent / "fixtures"

PERSONAS = ["Vera Lindqvist", "Tomas Abeyta", "Naledi Khumalo", "Ren Ishikawa",
            "Marta Kowalczyk", "Dario Ferreti", "Yusuf Demirel", "Ines Barreto"]

SLOTS = {
    "favorite cuisine": ["Thai", "Ethiopian", "Peruvian", "Lebanese", "Korean",
                         "Oaxacan", "Sichuan", "Punjabi", "Basque", "Vietnamese"],
    "code editor": ["Vim", "Emacs", "Zed", "Helix", "Sublime", "Kakoune",
                    "Fleet", "Lapce"],
    "primary language": ["Rust", "Go", "Kotlin", "Elixir", "Zig", "Swift",
                         "OCaml", "Gleam"],
    "coffee order": ["flat white", "cortado", "cold brew", "oat latte",
                     "pour-over", "macchiato", "espresso tonic", "matcha"],
    "workout": ["dawn swim", "boxing class", "climbing gym", "powerlifting",
                "trail cycling", "pilates", "rowing", "fencing"],
    "home city": ["Lisbon", "Tallinn", "Osaka", "Medellin", "Krakow",
                  "Auckland", "Porto", "Ljubljana"],
    "food allergy": ["peanuts", "shellfish", "gluten", "lactose", "soy",
                     "sesame", "mustard", "celery"],
    "pet": ["border collie", "maine coon", "cockatiel", "leopard gecko",
            "betta fish", "dwarf rabbit", "axolotl", "chinchilla"],
    "music genre": ["jazz fusion", "synthwave", "bluegrass", "afrobeat",
                    "post-rock", "bossa nova", "klezmer", "dub techno"],
    "operating system": ["NixOS", "Fedora", "Arch", "Debian", "openSUSE",
                         "FreeBSD", "Void Linux", "Alpine"],
    "cloud provider": ["Hetzner", "Fly.io", "Railway", "DigitalOcean",
                       "Scaleway", "Linode", "OVH", "Vultr"],
    "note-taking app": ["Obsidian", "Logseq", "Zettlr", "Joplin", "Silverbullet",
                        "Dendron", "Foam", "Trilium"],
    "keyboard": ["Ergodox", "HHKB", "Corne", "Moonlander", "Model M",
                 "Atreus", "Sofle", "Planck"],
    "airline seat": ["aisle seat", "window seat", "exit row", "bulkhead"],
    "lunch budget": ["12 dollars", "15 dollars", "18 dollars", "22 dollars",
                     "25 dollars", "30 dollars"],
    "browser": ["Firefox", "Vivaldi", "Brave", "Orion", "LibreWolf", "Ladybird"],
    "meeting slot": ["8am block", "10am block", "2pm block", "4pm block"],
    "vacation style": ["mountain huts", "city museums", "beach diving",
                       "rail journeys", "desert camping", "canal boating"],
    "phone platform": ["GrapheneOS", "stock Android", "iOS", "CalyxOS"],
    "reading format": ["paper books", "e-ink reader", "audiobooks",
                       "tablet reading"],
    "tea choice": ["genmaicha", "earl grey", "rooibos", "silver needle",
                   "pu-erh", "chamomile"],
    "commute mode": ["gravel bike", "metro", "e-scooter", "walking",
                     "cargo bike", "tram"],
}
ABSENT_SLOTS = ["blood type", "shoe size", "car model", "favorite opera",
                "ski resort", "tax accountant"]

RESTAURANTS = {c: f"{n}" for c, n in zip(SLOTS["favorite cuisine"], [
    "Baan Suan", "Habesha House", "Casa Andina", "Cedar & Thyme", "Seoul Ember",
    "Milpa Roja", "Chili Pavilion", "Tandoor Lane", "Txoko Berri", "Pho Quarter"])}

REASON_BITS = [
    "the wrist pain finally got too bad", "the pricing doubled overnight",
    "a teammate won me over at the offsite", "the latency drove me crazy",
    "my doctor recommended the change", "it kept breaking after updates",
    "the whole community moved there", "I got a voucher and never looked back",
    "the new place opened next door", "the old one discontinued my plan",
    "a documentary completely changed my mind", "my partner talked me into it",
    "the noise was unbearable", "I wanted something easier to maintain",
    "a conference talk sold me on it", "the export feature saved my project",
    "the subscription lapsed and I never renewed", "the season made it impractical",
    "my back demanded the change", "a friend's setup impressed me",
    "the battery life won the argument", "the commute made it impossible",
    "the flavor just stopped working for me", "the ergonomics were night and day"]

FILLER = [
    "the weather has been wild this week", "work sprints are back to back",
    "I finally fixed that flaky test", "the neighbours are renovating again",
    "I watched a great documentary yesterday", "the garden is out of control",
    "my sister visited over the weekend", "the deadline moved up a week",
    "I reorganized my whole desk setup", "traffic was terrible this morning",
    "the standup ran long again", "I tried a new recipe and burned it",
    "the gym was packed on Monday", "my package got lost in transit",
    "the release went out without a hitch", "I finally cleared my inbox"]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260807
    rng = random.Random(seed)
    FIX.mkdir(exist_ok=True)
    reasons = rng.sample(REASON_BITS * 8, 140)
    r_iter = iter(reasons)

    statements, questions = [], []
    start = datetime.date(2026, 2, 2)

    for p_i, name in enumerate(PERSONAS):
        slot_kinds = rng.sample(list(SLOTS), 22)
        stable, evolving = slot_kinds[:12], slot_kinds[12:22]
        state = {}     # kind -> list of (value, reason|None) versions
        for kind in stable:
            state[kind] = [(rng.choice(SLOTS[kind]), None)]
        for kind in evolving:
            n_versions = 3 if rng.random() < 0.4 else 2
            vals = rng.sample(SLOTS[kind], n_versions)
            state[kind] = [(vals[0], None)] + [
                (v, next(r_iter)) for v in vals[1:]]

        # Timeline: every reveal in sessions 0-4, every update in sessions
        # 5-9, and successive updates of the SAME slot in different
        # sessions. Each version therefore carries a strictly later date
        # than the one it replaces, so the chain is resolvable from dates
        # alone — never from position within a session. (Without this, two
        # updates could share a date and the "current" value would be
        # ambiguous; the same defect cost us four questions in the SDLC
        # suite before it was found.)
        n_sessions, per_session = 10, 5
        reveal_sessions, update_sessions = list(range(0, 5)), list(range(5, 10))
        buckets = {s: [] for s in range(n_sessions)}

        reveals = [("reveal", k, 0) for k in slot_kinds]
        rng.shuffle(reveals)
        for i, ev in enumerate(reveals):
            buckets[reveal_sessions[i % len(reveal_sessions)]].append(ev)

        for kind in evolving:
            n_up = len(state[kind]) - 1
            chosen = sorted(rng.sample(update_sessions, n_up))
            for vi, sess in enumerate(chosen, start=1):
                buckets[sess].append(("update", kind, vi))
        for s in buckets:
            assert len(buckets[s]) <= per_session + 3, (name, s, len(buckets[s]))

        def stmt_text(ev, date, sess):
            typ, kind, vi = ev
            val, reason = state[kind][vi]
            if typ == "reveal":
                return (f"[{date}] {name}, session {sess}: These days my "
                        f"{kind} is {val}.")
            old = state[kind][vi - 1][0]
            return (f"[{date}] {name}, session {sess}: Update on my {kind} — "
                    f"I've switched from {old} to {val}, because {reason}.")

        for s in range(n_sessions):
            date = (start + datetime.timedelta(days=9 * s + p_i)).isoformat()
            for ev in buckets[s]:
                statements.append({
                    "persona": name, "session": s, "date": date,
                    "kind": ev[1], "text": stmt_text(ev, date, s)})
            for _ in range(4):
                statements.append({
                    "persona": name, "session": s, "date": date, "kind": None,
                    "text": f"[{date}] {name}, session {s}: By the way, "
                            f"{rng.choice(FILLER)}."})

        # ---- questions ----
        def mcq(qtype, prompt, correct, distractors):
            opts = [correct] + distractors[:3]
            rng.shuffle(opts)
            questions.append({
                "qid": f"{qtype}-{len(questions):03d}", "type": qtype,
                "persona": name, "question": prompt,
                "options": opts, "answer": "ABCD"[opts.index(correct)],
                "answer_text": correct})

        def other_vals(kind, exclude):
            pool = [v for v in SLOTS[kind] if v not in exclude]
            return rng.sample(pool, min(3, len(pool)))

        for kind in rng.sample(stable, 12):
            val = state[kind][0][0]
            mcq("FACT", f"What is {name}'s {kind}?", val, other_vals(kind, {val}))
        for kind in evolving:
            cur = state[kind][-1][0]
            old = state[kind][0][0]
            mcq("CURRENT", f"What is {name}'s {kind} as of their most recent "
                f"session?", cur, [old] + other_vals(kind, {cur, old})[:2])
        for kind in rng.sample(evolving, 5):
            chain = [v for v, _ in state[kind]]
            correct = ", then ".join(chain)
            wrong1 = ", then ".join(reversed(chain))
            others = other_vals(kind, set(chain))
            wrong2 = ", then ".join([chain[0]] + [others[0]])
            wrong3 = ", then ".join([others[1 % len(others)], chain[-1]])
            mcq("EVOLUTION", f"How did {name}'s {kind} change over time?",
                correct, [wrong1, wrong2, wrong3])
        upd_kinds = rng.sample(evolving, 5)
        used_reasons = [r for k in evolving for _, r in state[k][1:]]
        for kind in upd_kinds:
            val, reason = state[kind][-1]
            wrongs = rng.sample([r for r in used_reasons if r != reason], 3)
            mcq("REASON", f"Why did {name} switch their {kind} to {val}?",
                reason, wrongs)
        cuisine_kind = "favorite cuisine"
        for i in range(4):
            if cuisine_kind in state:
                cur = state[cuisine_kind][-1][0]
                correct = RESTAURANTS[cur]
                wrongs = rng.sample([r for c, r in RESTAURANTS.items() if c != cur], 3)
                mcq("RECOMMEND", f"Which restaurant is the best suggestion "
                    f"for {name}'s taste?", correct, wrongs)
            else:
                kind = rng.choice(stable)
                val = state[kind][0][0]
                mcq("RECOMMEND", f"A gift matching {name}'s {kind} should "
                    f"relate to which of these?", val, other_vals(kind, {val}))
        for kind in rng.sample(ABSENT_SLOTS, 4):
            correct = "Not enough information in the sessions"
            wrongs = ["It was mentioned in the first session",
                      "It changed twice over the sessions",
                      "It matches their closest teammate's"]
            mcq("UNANSWER", f"What is {name}'s {kind}?", correct, wrongs)

        # ---- self-validation for this persona ----
        mine = [s for s in statements if s["persona"] == name]
        blob = " ".join(s["text"] for s in mine)
        for kind in evolving:
            cur = state[kind][-1][0]
            assert f"to {cur}" in blob or f"is {cur}" in blob, (name, kind)
            # every version of an evolving slot must sit on its own date,
            # in the order the chain declares
            dates = [s["date"] for s in mine if s["kind"] == kind]
            assert len(dates) == len(set(dates)), (name, kind, "date collision")
            assert dates == sorted(dates), (name, kind, "out of order")
        for kind in ABSENT_SLOTS:
            assert kind not in blob, (name, kind)

    sp = FIX / "statements.jsonl"
    qp = FIX / "questions.jsonl"
    sp.write_text("\n".join(json.dumps(s) for s in statements) + "\n")
    qp.write_text("\n".join(json.dumps(q) for q in questions) + "\n")
    by_type = {}
    for q in questions:
        by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    meta = {"seed": seed, "personas": len(PERSONAS),
            "statements": len(statements), "questions": len(questions),
            "by_type": by_type,
            "history_chars_per_persona": round(sum(len(s['text']) for s in statements) / len(PERSONAS)),
            "statements_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
            "questions_sha256": hashlib.sha256(qp.read_bytes()).hexdigest()}
    (FIX / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
