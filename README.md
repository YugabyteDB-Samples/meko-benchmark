# Meko context-infrastructure benchmarks

Reproducible benchmarks measuring what a bounded, verified context delivery does
to reader accuracy and reader-input tokens, compared against sending full history
and against a local BM25 baseline. Everything needed to re-verify the published
results offline, and to re-run them live, is in this repository.

The fixtures are synthetic. The result ledgers, scorers, and clients are the
exact ones used to produce the numbers.

## What is here

| Directory | Test | What it measures |
|---|---|---|
| [`INHOUSE-BENCHMARK/context-fit/`](INHOUSE-BENCHMARK/context-fit/) | 320-question memory benchmark, 9 readers | Meko memory search vs local BM25 vs full history: accuracy and reader-input reduction |
| [`INHOUSE-BENCHMARK/large-context-1m/`](INHOUSE-BENCHMARK/large-context-1m/) | ~1M-token history, 320 questions | Prepared Meko-delivered evidence vs each model's own compaction |
| [`BABILONG-BENCHMARK/`](BABILONG-BENCHMARK/) | Public BABILong, 4 readers | Full vs BM25 vs Meko-artifact-backed delivery, QA1–QA10 at 0k/4k/16k plus QA11–QA20 at 0k |

Each directory has its own `README.md` with the exact commands, the frozen
fixtures, the append-only result ledgers, and the deterministic scorer.

## Reproduce

Two promises, described in each sub-README.

1. **Recompute the published results offline** (no model calls, no network):

   ```bash
   # In-house
   cd INHOUSE-BENCHMARK/context-fit && python3 verify.py
   cd ../large-context-1m && python3 verify.py

   # BABILong
   cd BABILONG-BENCHMARK && python3 VERIFY.py
   ```

2. **Re-run live** — needs your own model-provider access and, for the Meko arms,
   a Meko API key and datapack. Provider credentials, model CLIs, and Meko access
   are intentionally external; copy each `*.env.example` to your own environment
   and fill in your values. Hosted model aliases change over time, so fresh
   responses are not guaranteed to be byte-identical, while the data, prompts,
   scorer, chunking, and top-k policy stay fixed.

## What is NOT here

No article drafts, figures, private machine configuration, provider attempt
logs, session identifiers, credentials, or personal data. The Meko datapack UUID
referenced in `BABILONG-BENCHMARK` is an access-controlled locator only.

## On the Meko arm

Where a result is labeled "Meko," Meko provides durable, byte-exact storage and
delivery (memory search in the in-house tests; content-addressed artifact fetch
verified by SHA-256 in BABILong). In the BABILong arm the chunking and ranking
run on the client, so that arm tests Meko-backed delivery plus the client
retrieval policy, not a Meko semantic-search API. Each sub-README states exactly
what its "Meko" arm did.

## License

This repository's own code and fixtures are licensed under Apache-2.0
([`LICENSE`](LICENSE)). Upstream BABILong retains its own license under
[`BABILONG-BENCHMARK/LICENSES/`](BABILONG-BENCHMARK/LICENSES/).
