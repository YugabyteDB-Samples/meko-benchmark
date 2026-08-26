# BABILong results and clients

This bundle contains the official BABILong rows, upstream prompt/scorer,
retained Full, BM25, and Meko-artifact result ledgers, the BM25-800 packet
builder, and provider adapters. It covers QA1-QA10 at 0k, 4k, and 16k and
official QA11-QA20 at 0k. No article, figure, long-context extension, provider
attempt logs, or private machine configuration is included.

## Offline checks

```bash
python3 VERIFY.py
cd babilong
python3 babilong_bm25_800_evidence.py --verify-packets --readers qwen deepseek_v4_flash codex_luna
```

## Rebuild and run

```bash
cd babilong
python3 run_babilong_bm25_800.py --build-packets
```

Set endpoint, model, and executable variables in `.env.example` or the
environment. Credentials, model weights, and hosted aliases are external.
Future responses may differ while data, prompts, scorer, chunking, top-k policy,
and the client contract remain reproducible.

## Meko locator

The original artifact-backed evaluation used datapack
`bdec33d9-5e95-4611-bd63-f0482e80a955`. The UUID is a locator only; access is
ACL-controlled. A recipient without access can reproduce BM25 and Full from
the retained local source bytes, or create an authorized datapack for a fresh
Meko-artifact run. The Meko arm fetches complete bytes by SHA-256, verifies
them, then ranks 800-character chunks locally with BM25 plus embeddings.
