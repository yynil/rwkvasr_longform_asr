from __future__ import annotations

import re

_PUNCT_TAGS = {
    "COMMA": ",",
    "PERIOD": ".",
    "FULLSTOP": ".",
    "DOT": ".",
    "QUESTION": "?",
    "QUESTIONMARK": "?",
    "EXCLAMATION": "!",
    "EXCLAMATIONMARK": "!",
    "EXCLAMATIONPOINT": "!",
    "COLON": ":",
    "SEMICOLON": ";",
    "DASH": "-",
    "HYPHEN": "-",
    "APOSTROPHE": "'",
}


def normalize_asr_text(text: str, *, language: str | None = None, mode: str = "none") -> str:
    if mode == "none":
        return text
    if mode != "runtime":
        raise ValueError(f"Unsupported text normalization mode: {mode!r}")

    normalized = _replace_markup_punctuation(text)
    language = (language or "").strip().lower()
    if language == "en" or language.startswith(("en-", "en_")):
        normalized = _lowercase_outside_markup(normalized)
    return _cleanup_punctuation_spacing(normalized)


def _replace_markup_punctuation(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw_tag = match.group(1)
        tag = re.sub(r"[\s_-]+", "", raw_tag).upper()
        return _PUNCT_TAGS.get(tag, " ")

    return re.sub(r"<([^<>]+)>", replace, text)


def _cleanup_punctuation_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([,.?!:;])", r"\1", text)
    text = re.sub(r"([,.?!:;])(?=\S)", r"\1 ", text)
    return text.strip()


def _lowercase_outside_markup(text: str) -> str:
    pieces: list[str] = []
    cursor = 0
    for match in re.finditer(r"<[^<>]+>", text):
        pieces.append(text[cursor : match.start()].lower())
        pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(text[cursor:].lower())
    return "".join(pieces)
