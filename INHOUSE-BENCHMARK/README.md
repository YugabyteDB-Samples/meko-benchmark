# In-house benchmark results and clients

This bundle contains only the two in-house context-infrastructure tests used
for the benchmark: the 320-question context-fit test and the one-million-token
prepared-delivery versus model-compaction test. It includes their synthetic
fixtures, retained result ledgers, deterministic scorers, provider clients,
and Meko clients.

## Verify retained results without network calls

```bash
cd context-fit && python3 verify.py
cd ../large-context-1m && python3 verify.py
```

## Re-run the 320-question test

From `context-fit/`, generate the synthetic corpus, create a new Meko datapack,
record Meko and local BM25 retrieval, then run provider readers:

```bash
python3 runner/gen_mekobench.py RUN_TAG
MEKO_API_KEY=... python3 runner/ingest_pm.py RUN_TAG
MEKO_API_KEY=... python3 runner/run_recipebench.py retrieve --stamp RUN_TAG --ids runs-live/pm-ids-RUN_TAG.json
python3 runner/run_full320_matrix.py --stamp RUN_TAG
```

Provider credentials, model CLIs, and Meko access are intentionally external.

## Re-run the one-million-token test

From `large-context-1m/`, `runner/prepare_fast_packets.py` creates local BM25
packets and stores/fetches them by SHA-256 through a new authorized Meko
datapack. `run_compact_own.py` creates model digests, `prepare_reader_contexts.py`
equalizes the packet size, and `run_answers.py` scores both arms. The README in
that directory has the exact commands and reader labels.

No API key, private account, host path, conversation ID, or private datapack
access is included. The saved results are synthetic and contain no personal
data.
