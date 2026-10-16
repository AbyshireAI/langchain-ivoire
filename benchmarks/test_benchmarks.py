from __future__ import annotations

import json
import sys
import wave
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics as M  # noqa: E402
import run_benchmark as RB  # noqa: E402

SAMPLE = HERE / "data" / "sample_mt.jsonl"


def test_chrf_bounds_and_symmetry_of_identity() -> None:
    assert M.chrf(["Venez manger."], ["Venez manger."]) == pytest.approx(100.0)
    assert M.chrf([""], ["abc"]) == 0.0
    assert M.chrf(["xyz"], ["abc"]) == 0.0
    assert 0 < M.chrf(["Venez manger"], ["Venez manger."]) < 100
    assert M.chrf_from_stats([0.0] * 24) == 0.0


def test_chrf_matches_sacrebleu_when_available() -> None:
    sacrebleu = pytest.importorskip("sacrebleu")
    rows = [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines()]
    hyps = [r["bci"].lower().rstrip(".?") for r in rows]
    refs = [r["bci"] for r in rows]
    ref_score = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    assert M.chrf(hyps, refs) == pytest.approx(ref_score, abs=1e-3)


def test_separate_punctuation_like_sacrebleu() -> None:
    assert M._separate_punctuation("(hi) a, b. c") == ["(hi", ")", "a", ",", "b", ".", "c"]


def test_error_rates() -> None:
    assert M.edit_distance(list("kitten"), list("sitting")) == 3
    assert M.edit_distance([], list("abc")) == 3
    assert M.normalize_asr("Bla di  like’s!") == "bla di like's"
    c_e, c_n, w_e, w_n = M.error_rate_stats("bla de like", "Bla di like.")
    assert (c_e, c_n, w_e, w_n) == (1, 9, 1, 3)


def test_bootstrap_ci_brackets_the_score() -> None:
    hyps = ["a b c", "a b", "x y z", "Venez manger.", "N tɛ"]
    refs = ["a b c", "a b d", "x y", "Venez manger !", "N tɛ, an ti sɛn?"]
    stats = [M.chrf_stats(h, r) for h, r in zip(hyps, refs, strict=True)]
    score = M.chrf_from_stats(M.sum_stats(stats))
    lo, hi = M.bootstrap_ci(stats, M.chrf_from_stats, n_boot=300)
    assert lo <= score <= hi
    assert M.bootstrap_ci([], M.chrf_from_stats) == (0.0, 0.0)


def test_mt_run_on_sample_with_fake_backend(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    res = RB.main(
        ["mt", "--test", str(SAMPLE), "--backend", "fake", "--n-boot", "100", "--out", str(out)]
    )
    (a, b) = res["results"]
    assert a["direction"] == "bci-fra" and a["n"] == 12 and b["direction"] == "fra-bci"
    assert 0 <= a["raw"]["score"] <= 100 and len(a["raw"]["ci95"]) == 2
    assert a["latency"]["requests"] == 1 and a["latency"]["items_per_request"] == 32
    assert 0 <= a["latency"]["p50_ms"] <= a["latency"]["p90_ms"] <= a["latency"]["max_ms"]
    saved = out.read_text(encoding="utf-8")
    # scores only: no sentence from the test set leaks into the results
    for line in SAMPLE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert row["bci"] not in saved and row["fra"] not in saved
    assert json.loads(saved)["server_models"] == {"backend": "fake"}


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)


def test_asr_run_with_fake_backend_and_overlap_exclusion(tmp_path: Path) -> None:
    rows = [
        {"audio": "a.wav", "text": "Bla di like.", "speaker": "s1", "clip_id": "a"},
        {"audio": "b.wav", "text": "N tɛ, an ti sɛn?", "speaker": "s2", "clip_id": "b"},
    ]
    for r in rows:
        _wav(tmp_path / r["audio"])
    test = tmp_path / "test.jsonl"
    test.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ref = tmp_path / "ref.jsonl"
    ref.write_text(json.dumps({"text": "n tɛ an ti sɛn"}), encoding="utf-8")
    res = RB.main(
        [
            "asr",
            "--test",
            str(test),
            "--backend",
            "fake",
            "--n-boot",
            "50",
            "--exclude-overlap",
            str(ref),
        ]
    )
    (r,) = res["results"]
    assert r["n"] == 1 and r["excluded_overlap"] == 1 and r["speakers"] == 1
    assert r["cer"]["score"] == 0.0 and r["wer"]["score"] == 0.0  # fake ASR says "bla di like"
    assert r["minutes"] == 0.0 and r["cer_with_spaces"]["score"] == 0.0
    assert r["latency"]["requests"] == 1 and r["latency"]["items_per_request"] == 1


def test_normalize_asr_matches_offline_target() -> None:
    assert M.normalize_asr("Ɔ  ɡa’n, kɛ!") == "ɔ ga'n kɛ"


def test_cer_with_spaces_counts_spaces() -> None:
    assert M.char_error_stats_with_spaces("bladi like", "Bla di like.") == [1, 11]
    assert M.error_rate_stats("bladi like", "Bla di like.")[:2] == [0, 9]


def test_asr_check_overlap_reports_without_dropping(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {"audio": "a.wav", "text": "Bla di like.", "speaker": "new1", "duration": 30.0},
        {"audio": "b.wav", "text": "N tɛ, an ti sɛn?", "speaker": "old1", "duration": 30.0},
    ]
    for r in rows:
        _wav(tmp_path / r["audio"])
    test = tmp_path / "test.jsonl"
    test.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ref = tmp_path / "ref.jsonl"
    ref.write_text(json.dumps({"text": "n tɛ an ti sɛn", "speaker": "old1"}), encoding="utf-8")
    res = RB.main(
        [
            "asr",
            "--test",
            str(test),
            "--backend",
            "fake",
            "--n-boot",
            "50",
            "--check-overlap",
            str(ref),
        ]
    )
    (r,) = res["results"]
    assert r["n"] == 2 and r["minutes"] == 1.0 and r["speakers"] == 2
    assert r["overlap"] == {
        "manifests": 1,
        "test_speakers_in_reference": 1,
        "test_items_with_sentence_in_reference": 1,
    }


def test_percentile_and_latency_summary() -> None:
    assert M.percentile([], 50) == 0.0
    assert M.percentile([10.0], 90) == 10.0
    assert M.percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert M.percentile(list(map(float, range(1, 11))), 90) == pytest.approx(9.1)
    s = M.latency_summary([100.0, 200.0, 300.0])
    assert s == {"requests": 3, "p50_ms": 200, "p90_ms": 280, "mean_ms": 200, "max_ms": 300}
