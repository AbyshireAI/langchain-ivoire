"""Scoring for the benchmark: chrF++ (translation), CER/WER (speech recognition), bootstrap CIs,
latency percentiles.

Pure Python, no dependencies. ``chrf_stats``/``chrf_from_stats`` reproduce sacrebleu 2.x
``CHRF(word_order=2)`` (character n-grams up to 6 without whitespace, word n-grams up to 2 with
punctuation split off word edges, beta = 2, no epsilon smoothing); a corpus score is computed from
the summed per-segment statistics, exactly like ``sacrebleu.corpus_chrf``. The unit tests compare
against sacrebleu when it is installed.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

__all__ = [
    "ASR_NORMALIZERS",
    "MT_NORMALIZERS",
    "bootstrap_ci",
    "char_error_stats_with_spaces",
    "chrf",
    "chrf_from_stats",
    "chrf_stats",
    "edit_distance",
    "error_rate_stats",
    "latency_summary",
    "normalize_asr",
    "percentile",
    "sum_stats",
]

CHAR_ORDER = 6
WORD_ORDER = 2
BETA = 2.0
_PUNCTS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


# ---- chrF++ ------------------------------------------------------------------------------------
def _separate_punctuation(sent: str) -> list[str]:
    """sacrebleu's chrF word split: one leading or trailing punctuation mark becomes a token."""
    out: list[str] = []
    for w in sent.split():
        if len(w) == 1:
            out.append(w)
        elif w[-1] in _PUNCTS:
            out += [w[:-1], w[-1]]
        elif w[0] in _PUNCTS:
            out += [w[0], w[1:]]
        else:
            out.append(w)
    return out


def _ngrams(sent: str) -> list[Counter[str]]:
    chars = "".join(sent.split())
    grams = [
        Counter(chars[i : i + n] for i in range(len(chars) - n + 1))
        for n in range(1, CHAR_ORDER + 1)
    ]
    words = _separate_punctuation(sent)
    grams += [
        Counter(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
        for n in range(1, WORD_ORDER + 1)
    ]
    return grams


def chrf_stats(hyp: str, ref: str) -> list[int]:
    """Per-segment statistics: ``[hyp_count, ref_count, matches]`` for each of the 8 n-gram orders."""
    stats: list[int] = []
    for h, r in zip(_ngrams(hyp.strip()), _ngrams(ref.strip()), strict=True):
        stats += [
            sum(h.values()),
            sum(r.values()),
            sum(min(c, r[g]) for g, c in h.items() if g in r),
        ]
    return stats


def chrf_from_stats(stats: Sequence[float]) -> float:
    """chrF++ (0..100) from summed statistics."""
    factor = BETA**2
    avg_p = avg_r = 0.0
    eff = 0
    for i in range(CHAR_ORDER + WORD_ORDER):
        n_hyp, n_ref, n_match = stats[3 * i : 3 * i + 3]
        if n_hyp > 0 and n_ref > 0:
            avg_p += n_match / n_hyp
            avg_r += n_match / n_ref
            eff += 1
    if eff == 0:
        return 0.0
    avg_p /= eff
    avg_r /= eff
    if avg_p + avg_r == 0:
        return 0.0
    return 100 * (1 + factor) * avg_p * avg_r / (factor * avg_p + avg_r)


def chrf(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """Corpus chrF++."""
    return chrf_from_stats(sum_stats([chrf_stats(h, r) for h, r in zip(hyps, refs, strict=True)]))


def sum_stats(rows: Sequence[Sequence[float]]) -> list[float]:
    return [float(sum(col)) for col in zip(*rows, strict=True)] if rows else [0.0] * 24


# ---- CER / WER ---------------------------------------------------------------------------------
def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance between two token sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def normalize_asr(s: str) -> str:
    """Lower-case, NFC, unified apostrophes, script ɡ -> g, punctuation removed (tone marks and ɛ/ɔ
    kept)."""
    s = unicodedata.normalize("NFC", s or "").lower()
    s = re.sub(r"[’‘ʼ`´]", "'", s).replace("ɡ", "g")
    s = re.sub(r"[^\w' ]+", " ", s)
    return " ".join(s.replace("_", " ").split())


def error_rate_stats(hyp: str, ref: str) -> list[int]:
    """``[char_edits, ref_chars, word_edits, ref_words]`` (characters counted without spaces)."""
    h, r = normalize_asr(hyp), normalize_asr(ref)
    hc, rc = list(h.replace(" ", "")), list(r.replace(" ", ""))
    hw, rw = h.split(), r.split()
    return [edit_distance(hc, rc), len(rc), edit_distance(hw, rw), len(rw)]


def char_error_stats_with_spaces(hyp: str, ref: str) -> list[int]:
    """``[char_edits, ref_chars]`` with spaces counted as characters.

    The headline ``error_rate_stats`` CER ignores spaces. Both use :func:`normalize_asr`.
    """
    h, r = normalize_asr(hyp), normalize_asr(ref)
    return [edit_distance(list(h), list(r)), len(r)]


# ---- normalisation for translation scores ------------------------------------------------------
def _common(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[’‘ʼ′`´]", "'", s)
    s = re.sub(r"[«»“”„\"]", "", s)
    return " ".join(s.split())


def _bci(s: str) -> str:
    s = unicodedata.normalize("NFD", _common(s))
    s = unicodedata.normalize("NFC", "".join(c for c in s if not unicodedata.combining(c)))
    return s.replace("ɲ", "ny").replace("Ɲ", "Ny")


MT_NORMALIZERS: dict[str, Callable[[str], str]] = {
    "raw": lambda s: s,
    "fra": _common,
    "eng": _common,
    "bci": _bci,
}
"""``raw``: as is. Per target language: NFC, unified apostrophes and quotes; Baoulé also drops tone
marks and maps ɲ -> ny."""

ASR_NORMALIZERS = {"default": normalize_asr}


# ---- bootstrap ---------------------------------------------------------------------------------
def bootstrap_ci(
    stats: Sequence[Sequence[float]],
    score: Callable[[Sequence[float]], float],
    *,
    n_boot: int = 1000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI of a corpus score, resampling segments with replacement."""
    rows = [list(r) for r in stats]
    n = len(rows)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    width = len(rows[0])
    scores = []
    for _ in range(n_boot):
        tot = [0.0] * width
        for _ in range(n):
            r = rows[rng.randrange(n)]
            for k in range(width):
                tot[k] += r[k]
        scores.append(score(tot))
    scores.sort()
    lo = scores[int((alpha / 2) * n_boot)]
    hi = scores[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (round(lo, 2), round(hi, 2))


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100) with linear interpolation (numpy's default method)."""
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def latency_summary(ms: Sequence[float], *, items_per_request: int | None = None) -> dict[str, Any]:
    """p50 / p90 (and mean, max) of per-request wall-clock latencies in milliseconds."""
    out: dict[str, Any] = {
        "requests": len(ms),
        "p50_ms": round(percentile(ms, 50)),
        "p90_ms": round(percentile(ms, 90)),
        "mean_ms": round(sum(ms) / len(ms)) if ms else 0,
        "max_ms": round(max(ms)) if ms else 0,
    }
    if items_per_request is not None:
        out["items_per_request"] = items_per_request
    return out
