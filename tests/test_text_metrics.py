from pathlib import Path

from rwkvasr.data import normalize_asr_text
from rwkvasr.eval import (
    compare_prediction_text_sets,
    compute_text_error_decomposition,
    compute_text_error_stats,
    tokenize_for_cer,
    tokenize_for_wer,
)


def _write_jsonl(path: Path, lines: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            import json

            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


def test_tokenization_is_word_and_char_aware() -> None:
    assert tokenize_for_wer("hello world") == ["hello", "world"]
    assert tokenize_for_wer("hello,world") == ["hello", "world"]
    assert tokenize_for_wer("don't stop") == ["dont", "stop"]
    assert tokenize_for_wer("OK okay O.K.") == ["ok", "ok", "ok"]
    assert tokenize_for_wer("你好世界") == ["你", "好", "世", "界"]
    assert tokenize_for_wer("mix中文text") == ["mix", "中", "文", "text"]
    assert tokenize_for_wer("Popayán Hückeswagen Wipperfürth Fortià") == [
        "popayán",
        "hückeswagen",
        "wipperfürth",
        "fortià",
    ]
    assert tokenize_for_wer("a\u0338 mix中café") == ["a\u0338", "mix", "中", "café"]
    assert tokenize_for_cer("a b") == ["a", "b"]
    assert tokenize_for_cer("a,b.") == ["a", "b"]
    assert tokenize_for_cer("okay O.K.") == ["o", "k", "o", "k"]


def test_ctc_text_normalization_removes_punctuation_and_symbols() -> None:
    assert normalize_asr_text("Yes, she's O.K.!", language="en", mode="ctc") == "yes shes o k"
    assert normalize_asr_text("你好，世界！", language="zh", mode="ctc") == "你好世界"
    assert normalize_asr_text("A <COMMA> B <NOISE> C", language="en", mode="ctc") == "a b c"
    assert (
        normalize_asr_text("A [noise] B (laughter) C {music}", language="en", mode="ctc") == "a b c"
    )


def test_ctc_text_normalization_removes_non_pronounced_control_spans() -> None:
    assert (
        normalize_asr_text(
            "WEBVTT\n00:00:01,000 --> 00:00:02,000\nSpeaker 1: Hello [PAD] <|en|>",
            language="en",
            mode="ctc",
        )
        == "hello"
    )
    assert normalize_asr_text("[00:01.23] A 【音乐】 B", language="en", mode="ctc") == "a b"
    assert normalize_asr_text("说话人2：你好【噪声】", language="zh", mode="ctc") == "你好"


def test_ctc_text_normalization_removes_project_language_confirmation_prefix() -> None:
    assert (
        normalize_asr_text("This is English text. I have 21 apples.", language="en", mode="ctc")
        == "i have twenty one apples"
    )
    assert (
        normalize_asr_text("这是中文文字。2024年增长3.5%", language="zh", mode="ctc")
        == "二零二四年增长百分之三点五"
    )


def test_ctc_metric_normalization_can_score_project_language_confirmation_prefix() -> None:
    assert (
        normalize_asr_text(
            "This is English text. I have 21 apples.",
            language="en",
            mode="ctc",
            strip_language_confirmation=False,
        )
        == "this is english text i have twenty one apples"
    )
    assert (
        normalize_asr_text(
            "这是中文文字。2024年增长3.5%",
            language="zh",
            mode="ctc",
            strip_language_confirmation=False,
        )
        == "这是中文文字二零二四年增长百分之三点五"
    )


def test_ctc_text_normalization_verbalizes_english_numbers() -> None:
    assert (
        normalize_asr_text("I paid $12.50 for 21 apples on 3/4.", language="en", mode="ctc")
        == "i paid twelve dollars fifty cents for twenty one apples on three four"
    )
    assert (
        normalize_asr_text("The 1st score was 10% at 03:05.", language="en", mode="ctc")
        == "the first score was ten percent at three oh five"
    )
    assert (
        normalize_asr_text("1,234.56 reasons", language="en", mode="ctc")
        == "one thousand two hundred thirty four point five six reasons"
    )


def test_ctc_text_normalization_verbalizes_chinese_numbers() -> None:
    assert (
        normalize_asr_text("2024年增长3.5%。", language="zh", mode="ctc")
        == "二零二四年增长百分之三点五"
    )
    assert normalize_asr_text("第12次测试", language="zh", mode="ctc") == "第十二次测试"
    assert normalize_asr_text("编号010", language="zh", mode="ctc") == "编号零一零"


def test_compute_text_error_stats(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "preds.jsonl",
        [
            {"utt_id": "u1", "pred_text": "a b c", "ref_text": "a b c"},
            {"utt_id": "u2", "pred_text": "x y", "ref_text": "x z y"},
        ],
    )
    stats = compute_text_error_stats(path, normalization="none")
    assert stats["sample_count"] == 2
    assert stats["avg_wer"] == 1.0 / 6.0
    assert abs(stats["avg_cer"] - (1.0 / 6.0)) < 1e-9


def test_compute_text_error_decomposition_reports_deletions_and_length(
    tmp_path: Path,
) -> None:
    path = _write_jsonl(
        tmp_path / "preds.jsonl",
        [
            {
                "utt_id": "u1",
                "pred_text": "one three extra",
                "ref_text": "one two three four",
            }
        ],
    )

    decomposition = compute_text_error_decomposition(
        path,
        language="en",
        normalization="none",
        metric="wer",
    )

    assert decomposition["reference_units"] == 4
    assert decomposition["prediction_units"] == 3
    assert decomposition["prediction_reference_unit_ratio"] == 0.75
    assert decomposition["deletions"] + decomposition["substitutions"] == 2
    assert decomposition["error_rate"] == 0.5


def test_text_error_stats_ignore_punctuation(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "preds.jsonl",
        [
            {
                "utt_id": "u1",
                "pred_text": "the key word there is guarantee,",
                "ref_text": "the key word there is guarantee.",
            },
            {
                "utt_id": "u2",
                "pred_text": "你好，世界！",
                "ref_text": "你好世界",
            },
        ],
    )
    stats = compute_text_error_stats(path, normalization="none")
    assert stats["avg_wer"] == 0.0
    assert stats["avg_cer"] == 0.0


def test_text_error_stats_canonicalize_common_asr_equivalents(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "preds.jsonl",
        [
            {"utt_id": "u1", "pred_text": "okay.", "ref_text": "ok,"},
            {"utt_id": "u2", "pred_text": "O.K.", "ref_text": "OK"},
        ],
    )
    stats = compute_text_error_stats(path, normalization="none")
    assert stats["avg_wer"] == 0.0
    assert stats["avg_cer"] == 0.0


def test_text_error_stats_ignore_decoder_language_confirmation_prefix(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "preds.jsonl",
        [
            {
                "utt_id": "u1",
                "pred_text": "This is English text. for example,",
                "ref_text": "for example.",
            },
            {
                "utt_id": "u2",
                "pred_text": "这是中文文字。 六",
                "ref_text": "六",
            },
        ],
    )
    stats = compute_text_error_stats(path, normalization="none")
    assert stats["avg_wer"] == 0.0
    assert stats["avg_cer"] == 0.0


def test_text_error_stats_default_metric_normalization_verbalizes_numbers(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "preds.jsonl",
        [
            {
                "utt_id": "en-num",
                "pred_text": "This is English text. I have 21 apples.",
                "ref_text": "I have twenty one apples",
            },
            {
                "utt_id": "zh-num",
                "pred_text": "这是中文文字。2024年增长3.5%",
                "ref_text": "二零二四年增长百分之三点五",
            },
        ],
    )
    en_stats = compute_text_error_stats(path, language="en")
    zh_stats = compute_text_error_stats(path, language="zh")
    mixed_stats = compute_text_error_stats(path)
    assert en_stats["per_sample_wer"]["en-num"] == 0.0
    assert en_stats["per_sample_cer"]["en-num"] == 0.0
    assert zh_stats["per_sample_wer"]["zh-num"] == 0.0
    assert zh_stats["per_sample_cer"]["zh-num"] == 0.0
    assert mixed_stats["avg_wer"] == 0.0
    assert mixed_stats["avg_cer"] == 0.0


def test_compare_prediction_text_sets(tmp_path: Path) -> None:
    baseline_path = _write_jsonl(
        tmp_path / "baseline.jsonl",
        [
            {"utt_id": "u1", "pred_text": "hello there", "ref_text": "hello there"},
            {"utt_id": "u2", "pred_text": "a c", "ref_text": "a b c"},
        ],
    )
    candidate_path = _write_jsonl(
        tmp_path / "candidate.jsonl",
        [
            {"utt_id": "u1", "pred_text": "hello there", "ref_text": "hello there"},
            {"utt_id": "u2", "pred_text": "a b c", "ref_text": "a b c"},
        ],
    )
    result = compare_prediction_text_sets(
        baseline_path,
        candidate_path,
        baseline_label="ctc",
        candidate_label="ar",
        normalization="none",
    )
    assert result["improved_count"] == 1
    assert result["worsened_count"] == 0
    assert result["unchanged_count"] == 1
    assert result["changed_prediction_count"] == 1
    assert result["baseline_avg_wer"] == 1.0 / 5.0
    assert result["candidate_avg_wer"] == 0.0


def test_compare_prediction_text_sets_ignores_punctuation_only_changes(tmp_path: Path) -> None:
    baseline_path = _write_jsonl(
        tmp_path / "baseline.jsonl",
        [{"utt_id": "u1", "pred_text": "guarantee,", "ref_text": "guarantee."}],
    )
    candidate_path = _write_jsonl(
        tmp_path / "candidate.jsonl",
        [{"utt_id": "u1", "pred_text": "guarantee.", "ref_text": "guarantee."}],
    )
    result = compare_prediction_text_sets(
        baseline_path,
        candidate_path,
        baseline_label="ctc",
        candidate_label="ar",
        normalization="none",
    )
    assert result["improved_count"] == 0
    assert result["worsened_count"] == 0
    assert result["unchanged_count"] == 1
    assert result["changed_prediction_count"] == 0
    assert result["baseline_avg_wer"] == 0.0
    assert result["candidate_avg_wer"] == 0.0


def test_compare_prediction_text_sets_ignores_language_confirmation_only_changes(
    tmp_path: Path,
) -> None:
    baseline_path = _write_jsonl(
        tmp_path / "baseline.jsonl",
        [{"utt_id": "u1", "pred_text": "ok", "ref_text": "ok"}],
    )
    candidate_path = _write_jsonl(
        tmp_path / "candidate.jsonl",
        [{"utt_id": "u1", "pred_text": "This is English text. ok", "ref_text": "ok"}],
    )
    result = compare_prediction_text_sets(
        baseline_path,
        candidate_path,
        baseline_label="ctc",
        candidate_label="ar",
        normalization="none",
    )
    assert result["improved_count"] == 0
    assert result["worsened_count"] == 0
    assert result["unchanged_count"] == 1
    assert result["changed_prediction_count"] == 0
    assert result["baseline_avg_wer"] == 0.0
    assert result["candidate_avg_wer"] == 0.0


def test_compare_prediction_text_sets_ignores_common_equivalent_changes(tmp_path: Path) -> None:
    baseline_path = _write_jsonl(
        tmp_path / "baseline.jsonl",
        [{"utt_id": "u1", "pred_text": "ok", "ref_text": "ok"}],
    )
    candidate_path = _write_jsonl(
        tmp_path / "candidate.jsonl",
        [{"utt_id": "u1", "pred_text": "okay.", "ref_text": "ok"}],
    )
    result = compare_prediction_text_sets(
        baseline_path,
        candidate_path,
        baseline_label="ctc",
        candidate_label="ar",
        normalization="none",
    )
    assert result["improved_count"] == 0
    assert result["worsened_count"] == 0
    assert result["unchanged_count"] == 1
    assert result["changed_prediction_count"] == 0
    assert result["baseline_avg_wer"] == 0.0
    assert result["candidate_avg_wer"] == 0.0
