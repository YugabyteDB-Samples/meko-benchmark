# One-million-token benchmark

This test starts with a 4,000,017-character synthetic history and 320
questions. Two reader inputs were compared:

- A prepared packet built from 500-character chunks ranked by local Okapi
  BM25.
- A digest produced by the model that later answered the questions.

The prepared path isn't a Meko semantic-search result. Local BM25 selected the
evidence. The 320 base packets were stored in Meko and fetched by their local
SHA-256 hashes. The saved client report records 320 of 320 fetches and no
reported mismatch. Reader calls used frozen local fixtures built from the same
packet bytes.

Each reader's packet was capped to the character length of its own digest. The
cap makes the two inputs comparable in size. `results.json` records the eight
complete reader pairs. MiniMax produced 19 of 21 digest chunks, so it has no
score.

Run:

```bash
python3 verify.py
```

The check rebuilds all 320 base packets, verifies their portable text hashes,
recreates the eight capped fixture hashes, and rescores every answer in both
arms. It also checks the Sonnet and Opus compaction usage totals. Saved gzip
hashes describe the compressed bytes used in the recorded run; they are not
recomputed because gzip output can differ across zlib builds.

The Meko readback file is a saved client report. The offline check rebuilds the
packet and fixture hashes, then checks the report's count and mismatch fields.
It can't independently verify past Meko calls. Repeating them requires a Meko
account and a new datapack.

## Recorded Meko packet checks

`manifests/meko-packet-readback.jsonl` contains one sanitized put/get record
for each packet. Every row has the packet index, local artifact hash, success
state, attempt count, and elapsed time. `verify.py` matches all 320 hashes to
`packet-manifest.json` before checking the aggregate fetch report and reader
fixture hashes.

## Run a new experiment

The exact runner source is under `runner/`:

- `prepare_fast_packets.py` builds the 500-character BM25 packets, uploads
  them with `artifact_put`, fetches each by SHA-256, and records byte matches;
- `run_compact_own.py` makes one digest per reader and records provider usage;
- `prepare_reader_contexts.py` caps every packet to the matching digest size;
- `run_answers.py` records one answer per call and writes evidence ceilings;
- `verify_fast_packet_delivery.py` fetches all packet artifacts again and
  compares the derived reader fixtures byte for byte;
- `clients.py` names the model aliases used in the current run, including
  `gemini-3.7-flash-low`.

From `large-context-1m/`, first build and store the base packets:

```bash
MEKO_API_KEY=... python3 runner/prepare_fast_packets.py xc1-c500-fast
```

Create each reader's compaction digest:

```bash
python3 runner/run_compact_own.py 20260812c \
  --labels sonnet5,opus5,codex-55,codex-luna,codex-terra,codex-sol,gemini-37,glm,minimax
```

Prepare equal-size packet fixtures:

```bash
python3 runner/prepare_reader_contexts.py
```

Run a packet cell and its matching compaction cell. Replace `sonnet5` with
each label in the list above:

```bash
XC1_FIXTURES=generated/fast-reader \
  python3 runner/run_answers.py xc1-c500-fast-sonnet5 \
  --labels sonnet5 --arms fast_packet --canary-probes 0

XC1_FIXTURES=generated/baseline-reader \
  python3 runner/run_answers.py 20260812c \
  --labels sonnet5 --arms own_digest --canary-probes 0
```

Finally, repeat the Meko readback and fixture comparison:

```bash
MEKO_API_KEY=... python3 runner/verify_fast_packet_delivery.py
```

These are live provider and Meko calls. They need the corresponding CLIs and
accounts, incur cost, and need not return the same answers as the saved run.
