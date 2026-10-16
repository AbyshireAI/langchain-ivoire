"""Benchmarks: score a langchain-ivoire backend on a translation or speech test set.

Translation (chrF++, corpus level, both "raw" and normalised scores, bootstrap 95% CI)::

    python benchmarks/run_benchmark.py mt --test PATH/everyday_test.jsonl \\
        --direction bci-fra --direction fra-bci --backend remote --out benchmarks/results/x.json

Speech recognition (CER and WER, bootstrap 95% CI, speaker count)::

    python benchmarks/run_benchmark.py asr --test PATH/read_speech_a.jsonl --audio-root PATH/audio \\
        --exclude-overlap PATH/other.jsonl --backend remote --out benchmarks/results/y.json

``--check-overlap`` reports speaker and sentence overlap with a reference manifest without removing
anything::

    python benchmarks/run_benchmark.py asr --test PATH/unseen_phone_speakers.jsonl \\
        --audio-root PATH/audio --check-overlap PATH/other.jsonl --name unseen_phone_speakers \\
        --out benchmarks/results/z.json

Offline smoke run on the bundled sample (fake backend, no network)::

    python benchmarks/run_benchmark.py mt --test benchmarks/data/sample_mt.jsonl --backend fake

Test sets are JSONL. Translation rows need the two language fields (``bci``, ``fra``, ``eng``) and
an ``id``; speech rows need ``audio`` (a path, relative to ``--audio-root``) and ``text`` (or
``text_asr``), optionally ``speaker``, ``clip_id`` and ``duration`` (seconds; otherwise read from
the WAV header). The remote backend reads ``IVOIRE_BASE_URL`` and
``IVOIRE_API_KEY`` (or ``--base-url``).

Each result also records the wall-clock latency of every request (``latency``: p50, p90, mean and
max in milliseconds, and the items per request), measured on the client, so it includes the network.

The results JSON holds scores, set sizes, CIs and run details only: never the texts, hypotheses or
audio, and never the server address. Of the server's ``/v1/models`` answer it keeps the API version
and each model's product name and version (see ``_public_models``).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M
from langchain_ivoire import __version__
from langchain_ivoire._audio import wav_info
from langchain_ivoire.backends import IvoireBackend, get_backend


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _ids_digest(rows: list[dict[str, Any]], key: str) -> str:
    """Hash of the item ids (identifies the exact subset without revealing any text)."""
    h = hashlib.sha256()
    for i, r in enumerate(rows):
        h.update(str(r.get(key, i)).encode("utf-8") + b"\n")
    return h.hexdigest()[:16]


def _backend(args: argparse.Namespace) -> IvoireBackend:
    if args.backend == "remote":
        return get_backend("remote", base_url=args.base_url, max_retries=args.retries)
    return get_backend(args.backend)


async def _translate_all(
    be: IvoireBackend,
    texts: list[str],
    src: str,
    tgt: str,
    style: str,
    batch: int,
    latencies: list[float] | None = None,
) -> list[str]:
    out: list[str] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        t0 = time.perf_counter()
        res = await be.atranslate(chunk, source=src, target=tgt, style=style)
        if latencies is not None:
            latencies.append(1000 * (time.perf_counter() - t0))
        out += [r.text for r in res]
        print(f"  {src}->{tgt} {min(i + batch, len(texts))}/{len(texts)}", file=sys.stderr)
    return out


def run_mt(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(Path(args.test))
    be = _backend(args)
    results: list[dict[str, Any]] = []
    for direction in args.direction or ["bci-fra", "fra-bci"]:
        src, tgt = direction.split("-")
        items = [r for r in rows if r.get(src) and r.get(tgt)]
        if args.limit:
            items = items[: args.limit]
        t0 = time.perf_counter()
        lat: list[float] = []
        hyps = asyncio.run(
            _translate_all(be, [r[src] for r in items], src, tgt, args.style, args.batch, lat)
        )
        secs = time.perf_counter() - t0
        refs = [r[tgt] for r in items]
        entry: dict[str, Any] = {
            "direction": direction,
            "n": len(items),
            "metric": "chrF++",
            "ids_sha256_16": _ids_digest(items, "id"),
            "seconds": round(secs, 1),
            "latency": M.latency_summary(lat, items_per_request=args.batch),
        }
        for view, norm in (("raw", M.MT_NORMALIZERS["raw"]), ("norm", M.MT_NORMALIZERS[tgt])):
            stats = [M.chrf_stats(norm(h), norm(r)) for h, r in zip(hyps, refs, strict=True)]
            score = M.chrf_from_stats(M.sum_stats(stats))
            entry[view] = {
                "score": round(score, 2),
                "ci95": M.bootstrap_ci(stats, M.chrf_from_stats, n_boot=args.n_boot),
            }
        results.append(entry)
        print(
            f"{direction}: chrF++ {entry['raw']['score']} {entry['raw']['ci95']} "
            f"(norm {entry['norm']['score']}) n={entry['n']} "
            f"latency p50 {entry['latency']['p50_ms']} ms, p90 {entry['latency']['p90_ms']} ms",
            file=sys.stderr,
        )
    return {"task": "mt", "style": args.style, "results": results, **_provenance(args, be)}


def _ref(r: dict[str, Any]) -> str:
    return str(r.get("text_asr") or r.get("text") or "")


def _overlap(rows: list[dict[str, Any]], paths: list[str]) -> dict[str, Any]:
    """How many test speakers and sentences also occur in the given reference manifests."""
    speakers: set[str] = set()
    sentences: set[str] = set()
    for p in paths:
        for r in load_jsonl(Path(p)):
            if r.get("speaker"):
                speakers.add(str(r["speaker"]))
            sentences.add(M.normalize_asr(_ref(r)))
    test_speakers = {str(r["speaker"]) for r in rows if r.get("speaker")}
    return {
        "manifests": len(paths),
        "test_speakers_in_reference": len(test_speakers & speakers),
        "test_items_with_sentence_in_reference": sum(
            M.normalize_asr(_ref(r)) in sentences for r in rows
        ),
    }


def run_asr(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(Path(args.test))
    excluded = 0
    if args.exclude_overlap:
        seen = {
            M.normalize_asr(r.get("text_asr") or r.get("text") or "")
            for p in args.exclude_overlap
            for r in load_jsonl(Path(p))
        }
        keep = [r for r in rows if M.normalize_asr(r.get("text_asr") or r["text"]) not in seen]
        excluded = len(rows) - len(keep)
        rows = keep
    if args.limit:
        rows = rows[: args.limit]
    root = Path(args.audio_root) if args.audio_root else Path(args.test).parent
    be = _backend(args)

    seconds: list[float] = []
    lat: list[float] = []

    async def go() -> list[str]:
        hyps = []
        for i, r in enumerate(rows, 1):  # one clip at a time: keeps a shared server responsive
            p = Path(r["audio"])
            data = (p if p.is_absolute() else root / p).read_bytes()
            if r.get("duration") is not None:
                seconds.append(float(r["duration"]))
            else:
                info = wav_info(data)
                seconds.append(info[1] if info else 0.0)
            t1 = time.perf_counter()
            hyps.append((await be.atranscribe(data, lang="bci")).text)
            lat.append(1000 * (time.perf_counter() - t1))
            if i % 25 == 0 or i == len(rows):
                print(f"  asr {i}/{len(rows)}", file=sys.stderr)
        return hyps

    t0 = time.perf_counter()
    hyps = asyncio.run(go())
    stats = [M.error_rate_stats(h, _ref(r)) for h, r in zip(hyps, rows, strict=True)]
    stats_sp = [M.char_error_stats_with_spaces(h, _ref(r)) for h, r in zip(hyps, rows, strict=True)]

    def cer(s: Any) -> float:
        return 100 * s[0] / s[1] if s[1] else 0.0

    def wer(s: Any) -> float:
        return 100 * s[2] / s[3] if s[3] else 0.0

    def cer_sp(s: Any) -> float:
        return 100 * s[0] / s[1] if s[1] else 0.0

    tot = M.sum_stats(stats)
    entry: dict[str, Any] = {
        "n": len(rows),
        "minutes": round(sum(seconds) / 60, 1),
        "speakers": len({r.get("speaker") for r in rows if r.get("speaker")}),
        "excluded_overlap": excluded,
    }
    if args.check_overlap:
        entry["overlap"] = _overlap(rows, args.check_overlap)
    return {
        "task": "asr",
        "results": [
            {
                **entry,
                "ids_sha256_16": _ids_digest(rows, "clip_id"),
                "seconds": round(time.perf_counter() - t0, 1),
                "latency": M.latency_summary(lat, items_per_request=1),
                "cer": {
                    "score": round(cer(tot), 2),
                    "ci95": M.bootstrap_ci(stats, cer, n_boot=args.n_boot),
                },
                "wer": {
                    "score": round(wer(tot), 2),
                    "ci95": M.bootstrap_ci(stats, wer, n_boot=args.n_boot),
                },
                "cer_with_spaces": {  # CER with spaces counted
                    "score": round(cer_sp(M.sum_stats(stats_sp)), 2),
                    "ci95": M.bootstrap_ci(stats_sp, cer_sp, n_boot=args.n_boot),
                },
            }
        ],
        **_provenance(args, be),
    }


_MODEL_KEYS = ("asr", "mt", "mt_commercial", "tts")


def _public_models(models: Any) -> dict[str, Any]:
    """The part of a ``/v1/models`` answer that goes into a results file: the API version, the
    backend kind, and ``name`` / ``version`` of each model (nothing about how a model was built)."""
    if not isinstance(models, dict):
        return {}
    out: dict[str, Any] = {k: models[k] for k in ("api_version", "backend", "error") if k in models}
    for k in _MODEL_KEYS:
        m = models.get(k)
        if isinstance(m, dict):
            out[k] = {f: m[f] for f in ("name", "version") if f in m}
    return out


def _provenance(args: argparse.Namespace, be: IvoireBackend) -> dict[str, Any]:
    try:
        models = _public_models(be.models())
    except Exception as e:  # an older server without /v1/models
        models = {"error": type(e).__name__}
    return {
        "test_set": args.name or Path(args.test).stem,
        "backend": be.name,
        "server_models": models,
        "client": f"langchain-ivoire {__version__}",
        "python": platform.python_version(),
        "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bootstrap": {"n": args.n_boot, "seed": 12345, "ci": "percentile 95%"},
        "normalisation": {
            "raw": "none (sacrebleu-equivalent corpus chrF++)",
            "norm": "NFC, unified apostrophes/quotes; Baoulé: tone marks dropped, ɲ->ny",
            "asr": "lower-case, NFC, punctuation removed, apostrophes unified, ɡ (U+0261) -> g; "
            "CER without spaces (cer_with_spaces: spaces counted)",
        },
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="langchain-ivoire benchmarks")
    ap.add_argument("task", choices=["mt", "asr"])
    ap.add_argument("--test", required=True, help="test set JSONL")
    ap.add_argument("--name", help="test set name in the results (default: file stem)")
    ap.add_argument("--direction", action="append", help="mt: e.g. bci-fra (repeatable)")
    ap.add_argument("--style", default="everyday", choices=["everyday", "tale", "formal"])
    ap.add_argument("--audio-root", help="asr: directory the audio paths are relative to")
    ap.add_argument(
        "--exclude-overlap", action="append", help="asr: drop sentences found in this manifest"
    )
    ap.add_argument(
        "--check-overlap",
        action="append",
        help="asr: report speaker/sentence overlap with this manifest (drops nothing)",
    )
    ap.add_argument("--backend", default="remote", choices=["remote", "fake", "local"])
    ap.add_argument("--base-url", help="remote server (default: IVOIRE_BASE_URL)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N items")
    ap.add_argument("--out", help="write the results JSON here")
    args = ap.parse_args(argv)
    res = run_mt(args) if args.task == "mt" else run_asr(args)
    text = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return res


if __name__ == "__main__":
    main()
