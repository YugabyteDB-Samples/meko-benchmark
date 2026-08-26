# AGENTS.md

Guidance for AI agents (and humans) working in this repository.

## What this repo is

Reproducible context-infrastructure benchmarks. Two families:

- `INHOUSE-BENCHMARK/` — synthetic in-house tests (`context-fit/` = 320-question
  memory benchmark; `large-context-1m/` = ~1M-token prepared-delivery vs
  compaction).
- `BABILONG-BENCHMARK/` — the public BABILong benchmark with Full, BM25, and
  Meko-artifact delivery arms.

Each sub-directory is self-contained: fixtures, append-only result ledgers,
a deterministic scorer, and a `README.md` with exact commands.

## How to verify (offline, no model or network)

```bash
cd INHOUSE-BENCHMARK/context-fit && python3 verify.py
cd INHOUSE-BENCHMARK/large-context-1m && python3 verify.py
cd BABILONG-BENCHMARK && python3 VERIFY.py
```

All three must pass before any change to fixtures, ledgers, or scorers is
committed. A number is publishable only if a scorer prints it from a ledger.

## Ground rules

1. **Never commit secrets or personal data.** No API keys, tokens, `.env` files,
   emails, usernames, home paths, or session/account identifiers. Credentials are
   read from the environment at runtime only; templates live in `*.env.example`
   with placeholder values. `BABILONG-BENCHMARK/VERIFY.py` fails if `/home/`,
   `/Users/`, or personal email domains appear in shipped text — keep it that way.
2. **Never commit build junk.** No `__pycache__/`, `*.pyc`, or `.DS_Store`
   (see `.gitignore`).
3. **Fixtures are synthetic and stay synthetic.** Do not add real user data,
   customer content, or non-public documents.
4. **Ledgers are append-only.** Do not rewrite historical result rows. To add a
   run, append; the scorers reduce by last-write-wins trial key.
5. **Keep the "Meko" arm honestly labeled.** Meko supplies verified storage and
   delivery; in BABILong the client does the chunking and ranking. Do not claim
   Meko semantic search where the client performed retrieval.
6. **Reproducibility over convenience.** Do not hardcode machine-specific paths
   or logins. Resolve executables from `PATH` or an environment variable; run
   external reader CLIs in a hermetic, hook-free configuration so results are
   identical on any machine.

## Conventions

- Python 3.11+; standard library only for the offline verifiers.
- Token counts are compared only between arms of the same reader (tokenizers and
  client wrappers differ); never rank raw token totals across readers.
- Accuracy is reported with counts; percentages accompany, not replace, them.

## Layout

```
INHOUSE-BENCHMARK/
  context-fit/      fixtures/ ledgers/ runner/ verify.py results.json
  large-context-1m/ fixtures/ digests/ ledgers/ manifests/ runner/ verify.py
BABILONG-BENCHMARK/
  VERIFY.py
  babilong/         data/ runs/ upstream/ *.py  (runner, scorer, providers)
  LICENSES/
```
