# Meko context-infrastructure benchmarks

Reproducible benchmarks measuring what a bounded, verified context delivery does
to reader accuracy and reader-input tokens, compared against sending full history
and against a local BM25 baseline. Everything needed to re-verify the published
results offline, and to re-run them live, is in this repository.

The fixtures are synthetic. The result ledgers, scorers, and clients are the
exact ones used to produce the numbers.

## Why an in-house benchmark

Meko is a context-infrastructure product with four storage types, each with its
own API:

- **conversations** — the exact words of past sessions
- **memories** — current decisions and durable facts
- **knowledge base** — team documents
- **artifacts** — exact files, content-addressed by SHA-256

Public memory benchmarks such as LongMemEval and LoCoMo score one slice of
this: an assistant's recall of conversational history. They do not exercise
these storage APIs, and they do not hold retrieval policy, chunk size, and
token accounting fixed across arms. They cannot answer the question asked
here: for one storage type and one reader at a time, what does a delivery path
cost in input tokens, and what accuracy does it preserve?

So coverage is split two ways:

- **In-house benchmarks** — synthetic fixtures, frozen ledgers, deterministic
  scorers. Fully controlled: the corpus, the answer key, the retrieval policy,
  and the token accounting are all fixed and replayable offline.
- **BABILong** — a public long-context benchmark, used where public data fits
  the job: reading evidence out of long documents through the artifacts API.

## API coverage today

| Meko API | Benchmark | Status |
|---|---|---|
| memories | [`INHOUSE-BENCHMARK/context-fit/`](INHOUSE-BENCHMARK/context-fit/) — corpus ingest plus memory search retrieval | covered |
| artifacts | [`INHOUSE-BENCHMARK/large-context-1m/`](INHOUSE-BENCHMARK/large-context-1m/) — 320 packet put/get records verified by SHA-256; [`BABILONG-BENCHMARK/`](BABILONG-BENCHMARK/) — content-addressed fetch on public data | covered |
| conversations | none yet | planned |
| knowledge base | none yet | planned |

## What is here

| Directory | Test | What it measures | Meko APIs used |
|---|---|---|---|
| [`INHOUSE-BENCHMARK/context-fit/`](INHOUSE-BENCHMARK/context-fit/) | 320-question memory benchmark, 9 readers | Meko memory search vs local BM25 vs full history: accuracy and reader-input reduction | `memory_add` (corpus ingest), `memory_search` (per-question retrieval) |
| [`INHOUSE-BENCHMARK/large-context-1m/`](INHOUSE-BENCHMARK/large-context-1m/) | ~1M-token history, 320 questions | Prepared Meko-delivered evidence vs each model's own compaction | `artifact_put`, `artifact_get` (320 packet store/fetch pairs, SHA-256-verified) |
| [`BABILONG-BENCHMARK/`](BABILONG-BENCHMARK/) | Public BABILong, 4 readers | Full vs BM25 vs Meko-artifact-backed delivery, QA1–QA10 at 0k/4k/16k plus QA11–QA20 at 0k | `artifact_put`, `artifact_get` (recorded Meko-artifact arm; the BM25-800 rebuild makes no Meko calls) |

Every benchmark also calls `datapack_create` once as setup plumbing; it is not
part of the measured path.

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

## Roadmap

1. Extend in-house coverage to the two APIs with no benchmark yet:
   conversations and the knowledge base.
2. Build a v2 in-house benchmark — more aggressive and more accurate — folding
   in lessons from the BABILong, LongMemEval, and LoCoMo results.
3. Grow toward covering every Meko MCP API with at least one verified,
   offline-replayable result.

## License

This repository's own code and fixtures are licensed under Apache-2.0
([`LICENSE`](LICENSE)). Upstream BABILong retains its own license under
[`BABILONG-BENCHMARK/LICENSES/`](BABILONG-BENCHMARK/LICENSES/).
