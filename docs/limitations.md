# Limitations

- **Machine translation makes mistakes.** Baoulé is a low-resource language. Do not use these
  components where an error can cause harm (medical, legal, safety) without a human interpreter.
- **French is the pivot.** The agent reasons in French, and English always goes through French.
  Each translation step loses some nuance; idioms, proverbs and the tale register suffer most.
- **Short sentences work best.** The middleware asks the model for short, simple French and caps
  the answer at `max_sentences` because long sentences translate much worse.
- **Dialects.** Coverage of the regional varieties of Baoulé is uneven and has not been measured
  separately.
- **Speech recognition** is weaker on noisy phone audio, code-switching with French, and names.
  Turns below the confidence threshold are marked `low_confidence`.
- **The voice** has not yet been rated by native speakers. The phrasebook entries were checked
  automatically, not by a native speaker.
- **Benchmarks are small.** The test sets have hundreds of items, not thousands; read the confidence
  intervals on the [Benchmarks](benchmarks.md) page.
- **Licences.** The code is MIT. The models are served through the Abyshire API under its own
  terms; credits for third-party material are in `ATTRIBUTION.md`.
- **Streaming.** Streamed tokens are French; the Baoulé reply arrives when the turn ends.
- **Only Baoulé** is implemented. Other languages raise `NotImplementedError` until their language
  packs ship.
