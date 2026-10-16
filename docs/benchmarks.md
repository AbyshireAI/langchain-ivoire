# Benchmarks

All results on this page were **measured through the hosted API** (`https://api.abyshire.ci`) on
16 Oct 2026 with [`benchmarks/run_benchmark.py`](https://github.com/AbyshireAI/langchain-ivoire/tree/main/benchmarks),
one item per request. Scores are corpus-level; brackets are bootstrap 95% confidence intervals
(1000 resamples). Latency is the wall-clock time of each request as seen by the client, network
included. The [results files](https://github.com/AbyshireAI/langchain-ivoire/tree/main/benchmarks/results)
hold scores and latency only, never texts or outputs.

There is no standard public test set for Baoulé yet, and these test sets are small: read the
confidence intervals.

## Translation: `abyshire/translate-baoule`

chrF++ (higher is better). `norm` ignores tone marks and spelling variants of ɲ/ny and apostrophes.

| test set | direction | n | chrF++ [95% CI] | chrF++ norm | latency p50 / p90 |
|---|---|---|---|---|---|
| held-out everyday sentences (style `everyday`) | bci → fra | 716 | **36.5** [35.0, 37.9] | 36.6 | 237 / 442 ms |
| | fra → bci | 716 | **37.1** [35.7, 38.4] | 37.1 | 228 / 433 ms |
| held-out sentences from Baoulé tales (style `tale`) | bci → fra | 218 | **16.1** [15.3, 17.0] | 16.3 | 373 / 608 ms |
| | fra → bci | 218 | **19.4** [18.5, 20.4] | 19.9 | 395 / 638 ms |

- The tales are long, literary sentences whose reference translations are often free; they are much
  harder than everyday sentences.
- One sentence per request. Batched requests amortise the latency over several sentences.

## Speech recognition: `abyshire/listen-baoule`

Lower is better. `CER` ignores spaces; `CER, spaces counted` counts them.

| test set | CER [95% CI] | CER, spaces counted | WER [95% CI] | latency p50 / p90 |
|---|---|---|---|---|
| **phone recordings of speakers not seen before** | **8.3 %** [7.7, 8.9] | 7.6 % | **19.3 %** [17.9, 20.7] | 578 / 998 ms |
| read speech, held-out sentences | 7.7 % [6.8, 8.9] | 7.0 % | 16.4 % [15.0, 18.0] | 645 / 825 ms |

- Latency covers uploading the recording (a few seconds of 16 kHz WAV), recognition and the answer.
- Noisy audio, code-switching with French and names are harder than these sets.

## Voice and embeddings

`abyshire/speak-baoule` and `abyshire/embed-baoule` (both by approval) have not been benchmarked
through the API yet. The voice has not yet been rated by native speakers.

## Reproduce

```bash
export IVOIRE_API_KEY=...        # a key from Abyshire
uv run python benchmarks/run_benchmark.py mt --test your_test.jsonl \
    --direction bci-fra --direction fra-bci --batch 1 --out results.json
```

See [`benchmarks/README.md`](https://github.com/AbyshireAI/langchain-ivoire/tree/main/benchmarks)
for the test-set format and the scoring method.
