# Context-fit benchmark

This directory supports the 320-question table in the article. It tests whether a reader can answer from a bounded evidence packet when the complete history also fits in the prompt.

Each of the nine readers answered the same questions through three arms:

- `meko_optimized`: Meko retrieval followed by a bounded client-side packet.
- `bm25_recipe`: local BM25 retrieval followed by the same bounded packet builder.
- `full_history`: the complete history for the relevant persona.

Run the offline check from this directory:

```bash
python3 verify.py
```

The script rebuilds all 27 accuracy cells, the nine Meko-versus-full-history token reductions, the mean context sizes, and the two Claude billed-cost reductions from the JSONL ledgers. The denominator is 320 in every accuracy cell. Gemini returned 318 full-history answers; its two unanswered questions count as incorrect.

## Files

- `fixtures/`: the generated statements, questions, answer key, and corpus hashes.
- `retrieval.jsonl`: the frozen top-25 Meko and BM25 results for every question.
- `ledgers/`: successful model response batches, including answers, provider-reported usage, cost, and context size.
- `results.json`: the values checked by `verify.py`.

The public ledgers omit failed provider payloads and internal retrieval-record UUIDs. Successful rows retain the attempt count, so retries remain visible without publishing provider error text. Those removed fields do not enter any published calculation.

The offline check verifies the reported scores from the recorded retrieval and
reader ledgers. It does not replay the external calls.

## Run a new 320-question matrix

The `runner/` directory contains the full companion code:

- `gen_mekobench.py` creates the 615-record corpus and 320 questions;
- `ingest_pm.py` writes the corpus to a new Meko datapack;
- `run_recipebench.py` implements Meko retrieval, local Okapi BM25, the
  bounded client, and the dated-change conflict resolver;
- `run_cloud.py` calls one reader and records answers, usage, and cost;
- `run_full320_matrix.py` runs the nine readers named in the blog through all
  three published arms;
- `meko_client.py` is the dependency-free Meko client used by the run.

From `context-fit/`:

```bash
# Optional: regenerate the frozen corpus in a copy of this directory.
python3 runner/gen_mekobench.py 20260807

# Create a new datapack and ingest the records.
MEKO_API_KEY=... python3 runner/ingest_pm.py my-run

# Record Meko and BM25 retrieval for all questions.
MEKO_API_KEY=... python3 runner/run_recipebench.py retrieve \
  --stamp my-run --ids runs-live/pm-ids-my-run.json

# Run the current nine-reader matrix through all three arms.
python3 runner/run_full320_matrix.py --stamp my-run
```

The model runner expects the provider CLIs named in `runner/run_cloud.py`.
Those calls cost money and can differ from the saved run.

The saved Gemini ledger is `gemini-3.7-flash.jsonl` and records model
`gemini-3.7-flash-low`. Older Gemini rounds are not part of this release.
