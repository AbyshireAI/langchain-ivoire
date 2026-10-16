# Benchmarks

Scores a `langchain-ivoire` backend (normally the remote API) on translation and speech-recognition
test sets, with set sizes and bootstrap 95% confidence intervals.

| file | what |
|---|---|
| `metrics.py` | chrF++ (sacrebleu-equivalent, checked against sacrebleu 2.6 to < 1e-4), CER/WER, percentile bootstrap |
| `run_benchmark.py` | command line: runs a backend over a JSONL test set and writes a results JSON |
| `data/sample_mt.jsonl` | 12 public phrasebook pairs, a **format sample** for smoke runs (not a benchmark) |
| `results/` | results files: scores, CIs, request latency and run details only, never texts or hypotheses |

The full test sets are **not** in this repository. They stay with the project data and are passed by
path. Do not commit them, and do not commit hypotheses.

## Running

```bash
# translation, both directions (server from IVOIRE_BASE_URL / IVOIRE_API_KEY, or --base-url)
uv run python benchmarks/run_benchmark.py mt --test /path/to/everyday_test.jsonl \
    --name everyday_test --direction bci-fra --direction fra-bci \
    --out benchmarks/results/everyday_test_<model>_<date>.json

# speech recognition; --exclude-overlap drops test sentences that occur in another manifest
uv run python benchmarks/run_benchmark.py asr --test /path/to/speech_test.jsonl --audio-root /path/to/audio \
    --name speech_test --out benchmarks/results/speech_test_<model>_<date>.json

# offline smoke run (fake backend, no network)
uv run python benchmarks/run_benchmark.py mt --test benchmarks/data/sample_mt.jsonl --backend fake
```

Translation rows need `id` and the two language fields (`bci`, `fra`, `eng`). Speech rows need
`audio` (relative to `--audio-root`) and `text` or `text_asr`, and optionally `speaker`.

## Method

- **chrF++**: corpus-level, character n-grams up to 6 and word n-grams up to 2, β = 2. `raw` scores
  the texts as they are. `norm` applies NFC and unifies apostrophes and quotes, and for Baoulé also
  drops tone marks and maps ɲ → ny. view.
- **CER/WER**: corpus-level (total edits / total reference length). Text is lower-cased, NFC, with
  punctuation removed, apostrophes unified and ɡ (U+0261) mapped to g, tone marks and ɛ/ɔ are kept. `cer` ignores spaces; `cer_with_spaces` counts them.
- **CIs**: 1000 bootstrap resamples of the test items (seed 12345), 2.5th and 97.5th percentiles.
- Each result records the item count, a hash of the item IDs (to identify the exact subset without
  revealing text), the API version and the product name and version of each model from the server's
  `/v1/models` answer, the client version and the date. It does not record the server address.
  Before committing a results file, check it names the models by their product names only.

## Caveats

- There is no standard public test set for Baoulé yet. The test sets are small (hundreds of items),
  so read the confidence intervals.
- A benchmark through the API measures the service as deployed (decoding settings included) and the
  latency as seen by the client, network included.
