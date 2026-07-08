from __future__ import annotations

import re
import unicodedata

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

_APOSTROPHE_CHARS = {"'", "\u2019", "\u02bc", "\uff07"}

_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

_EN_SMALL_NUMBERS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_EN_TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}
_EN_ORDINAL_WORDS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}
_ZH_DIGITS = "零一二三四五六七八九"

_NON_SPEECH_ANNOTATION_PATTERN = re.compile(
    r"[\[\(\{【（［]\s*"
    r"(?:"
    r"noise|noises|noisy|background(?:\s+noise)?|"
    r"laugh(?:ter|ing)?|chuckle|giggle|"
    r"music|bgm|applause|clap(?:ping)?|"
    r"sil(?:ence)?|silent|pause|"
    r"breath(?:ing)?|cough(?:ing)?|sneeze|"
    r"inaudible|unintelligible|unknown|unk|"
    r"foreign|non(?:\s|-)?speech|spoken(?:\s|-)?noise|vocalized(?:\s|-)?noise|"
    r"噪声|杂音|笑声|音乐|掌声|静音|沉默|停顿|咳嗽|不可闻|听不清"
    r")"
    r"\s*[\]\)\}】）］]",
    re.IGNORECASE,
)

_SUBTITLE_TIMESTAMP_RANGE_PATTERN = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?(?:[\.,]\d+)?\s*(?:-->|-|–|—)\s*"
    r"\d{1,2}:\d{2}(?::\d{2})?(?:[\.,]\d+)?\b"
)
_BRACKETED_TIMESTAMP_PATTERN = re.compile(
    r"[\[\(\{【（［]\s*\d{1,2}:\d{2}(?::\d{2})?(?:[\.,]\d+)?\s*[\]\)\}】）］]"
)
_BRACKETED_CONTROL_TOKEN_PATTERN = re.compile(
    r"[\[\(\{【（［]\s*(?:pad|cls|sep|mask|bos|eos|sos|unk|null|blank)\s*[\]\)\}】）］]",
    re.IGNORECASE,
)
_PROJECT_LANGUAGE_CONFIRMATION_PATTERN = re.compile(
    r"^\s*(?:this\s+is\s+english\s+text|这是中文文字)\s*[\.\!\?。:：,，;；]*\s*",
    re.IGNORECASE,
)
_LINE_PREFIX_METADATA_PATTERN = re.compile(
    r"(?m)^\s*(?:speaker|spk|说话人)\s*[-_#]?\s*\d{0,4}\s*[:：]\s*",
    re.IGNORECASE,
)
_SUBTITLE_HEADER_PATTERN = re.compile(r"(?m)^\s*(?:webvtt|note)\s*$", re.IGNORECASE)


def normalize_asr_text(text: str, *, language: str | None = None, mode: str = "none") -> str:
    if mode == "none":
        return text
    if mode not in {"runtime", "ctc"}:
        raise ValueError(f"Unsupported text normalization mode: {mode!r}")

    normalized = _replace_markup_punctuation(text)
    language = (language or "").strip().lower()
    if language == "en" or language.startswith(("en-", "en_")):
        normalized = _lowercase_outside_markup(normalized)
    if mode == "ctc":
        return _normalize_ctc_target_text(normalized, language=language)
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


def _normalize_ctc_target_text(text: str, *, language: str = "") -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _remove_ctc_non_pronounced_spans(text)
    text = _NON_SPEECH_ANNOTATION_PATTERN.sub(" ", text)
    text = _normalize_spoken_numbers(text, language=language)
    pieces: list[str] = []
    for ch in text:
        if ch in _APOSTROPHE_CHARS:
            continue
        category = unicodedata.category(ch)
        if category.startswith(("P", "S")):
            pieces.append(" ")
        else:
            pieces.append(ch)
    return _cleanup_punctuation_spacing("".join(pieces))


def _remove_ctc_non_pronounced_spans(text: str) -> str:
    for _ in range(4):
        updated = _PROJECT_LANGUAGE_CONFIRMATION_PATTERN.sub(" ", text, count=1)
        if updated == text:
            break
        text = updated.lstrip()
    text = _SUBTITLE_HEADER_PATTERN.sub(" ", text)
    text = _SUBTITLE_TIMESTAMP_RANGE_PATTERN.sub(" ", text)
    text = _BRACKETED_TIMESTAMP_PATTERN.sub(" ", text)
    text = _BRACKETED_CONTROL_TOKEN_PATTERN.sub(" ", text)
    text = _LINE_PREFIX_METADATA_PATTERN.sub(" ", text)
    return text


def _normalize_spoken_numbers(text: str, *, language: str = "") -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    family = _language_family(text, language=language)
    if family == "zh":
        return _normalize_zh_spoken_numbers(text)
    return _normalize_en_spoken_numbers(text)


def _language_family(text: str, *, language: str = "") -> str:
    language = (language or "").strip().lower().replace("_", "-")
    if language.startswith(("zh", "cmn", "yue")) or language in {"chinese", "mandarin"}:
        return "zh"
    if language.startswith("en") or language == "english":
        return "en"
    if _CJK_PATTERN.search(text):
        return "zh"
    return "en"


def _normalize_en_spoken_numbers(text: str) -> str:
    text = re.sub(
        r"([$£€])\s*(-?\d+(?:\.\d+)?)",
        lambda match: _en_currency_to_words(match.group(1), match.group(2)),
        text,
    )
    text = re.sub(
        r"\b(-?\d+(?:\.\d+)?)\s*%",
        lambda match: f"{_en_number_to_words(match.group(1))} percent",
        text,
    )
    text = re.sub(
        r"\b(\d{1,2}):(\d{2})\b",
        lambda match: _en_time_to_words(match.group(1), match.group(2)),
        text,
    )
    text = re.sub(
        r"\b(-?\d+)(st|nd|rd|th)\b",
        lambda match: _en_ordinal_to_words(match.group(1)),
        text,
    )
    text = re.sub(
        r"\b-?\d+\.\d+\b",
        lambda match: _en_number_to_words(match.group(0)),
        text,
    )
    text = re.sub(
        r"\b-?\d+\b",
        lambda match: _en_number_to_words(match.group(0)),
        text,
    )
    return text


def _en_currency_to_words(symbol: str, value: str) -> str:
    unit_names = {
        "$": ("dollar", "dollars", "cent", "cents"),
        "£": ("pound", "pounds", "penny", "pence"),
        "€": ("euro", "euros", "cent", "cents"),
    }
    singular, plural, cent_singular, cent_plural = unit_names.get(
        symbol, ("unit", "units", "cent", "cents")
    )
    negative = value.startswith("-")
    value = value[1:] if negative else value
    if "." not in value:
        amount = int(value) if value else 0
        unit = singular if amount == 1 else plural
        prefix = "minus " if negative else ""
        return f"{prefix}{_en_int_to_words(amount)} {unit}"

    whole_raw, frac_raw = value.split(".", 1)
    whole = int(whole_raw) if whole_raw else 0
    cents = int((frac_raw + "00")[:2])
    prefix = "minus " if negative else ""
    unit = singular if whole == 1 else plural
    parts = [f"{prefix}{_en_int_to_words(whole)} {unit}"]
    if cents:
        cent_unit = cent_singular if cents == 1 else cent_plural
        parts.append(f"{_en_int_to_words(cents)} {cent_unit}")
    return " ".join(parts)


def _en_time_to_words(hour: str, minute: str) -> str:
    hour_words = _en_int_to_words(int(hour))
    minute_value = int(minute)
    if minute_value == 0:
        return f"{hour_words} o clock"
    if minute.startswith("0"):
        return f"{hour_words} oh {_en_number_to_words(minute[-1])}"
    return f"{hour_words} {_en_number_to_words(minute)}"


def _en_number_to_words(value: str) -> str:
    negative = value.startswith("-")
    value = value[1:] if negative else value
    prefix = "minus " if negative else ""
    if "." in value:
        whole, frac = value.split(".", 1)
        whole_words = _en_number_to_words(whole or "0")
        frac_words = " ".join(_EN_SMALL_NUMBERS[int(ch)] for ch in frac if ch.isdigit())
        return f"{prefix}{whole_words} point {frac_words}".strip()
    if len(value) > 1 and value.startswith("0"):
        return prefix + " ".join(_EN_SMALL_NUMBERS[int(ch)] for ch in value)
    number = int(value or "0")
    if number > 999_999_999_999:
        return prefix + " ".join(_EN_SMALL_NUMBERS[int(ch)] for ch in value if ch.isdigit())
    return prefix + _en_int_to_words(number)


def _en_int_to_words(number: int) -> str:
    number = int(number)
    if number < 0:
        return "minus " + _en_int_to_words(-number)
    if number < 20:
        return _EN_SMALL_NUMBERS[number]
    if number < 100:
        tens = number // 10 * 10
        rest = number % 10
        return _EN_TENS[tens] if rest == 0 else f"{_EN_TENS[tens]} {_EN_SMALL_NUMBERS[rest]}"
    if number < 1000:
        rest = number % 100
        head = f"{_EN_SMALL_NUMBERS[number // 100]} hundred"
        return head if rest == 0 else f"{head} {_en_int_to_words(rest)}"
    for scale, name in (
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1000, "thousand"),
    ):
        if number >= scale:
            head = number // scale
            rest = number % scale
            words = f"{_en_int_to_words(head)} {name}"
            return words if rest == 0 else f"{words} {_en_int_to_words(rest)}"
    raise AssertionError(f"unhandled English number: {number}")


def _en_ordinal_to_words(value: str) -> str:
    words = _en_number_to_words(value).split()
    if not words:
        return ""
    last = words[-1]
    if last in _EN_ORDINAL_WORDS:
        words[-1] = _EN_ORDINAL_WORDS[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _normalize_zh_spoken_numbers(text: str) -> str:
    text = re.sub(
        r"([¥￥])\s*(\d+(?:\.\d+)?)",
        lambda match: f"{_zh_number_to_words(match.group(2))}元",
        text,
    )
    text = re.sub(
        r"([$£€])\s*(\d+(?:\.\d+)?)",
        lambda match: f"{_zh_number_to_words(match.group(2))}外币",
        text,
    )
    text = re.sub(
        r"(\d+(?:\.\d+)?)\s*%",
        lambda match: f"百分之{_zh_number_to_words(match.group(1))}",
        text,
    )
    text = re.sub(
        r"(\d{2,4})(?=年)",
        lambda match: _zh_digits_to_words(match.group(1)),
        text,
    )
    text = re.sub(
        r"\d+\.\d+",
        lambda match: _zh_number_to_words(match.group(0)),
        text,
    )
    text = re.sub(
        r"\d+",
        lambda match: _zh_number_to_words(match.group(0)),
        text,
    )
    return text


def _zh_number_to_words(value: str) -> str:
    if "." in value:
        whole, frac = value.split(".", 1)
        return f"{_zh_number_to_words(whole or '0')}点{_zh_digits_to_words(frac)}"
    if len(value) > 1 and value.startswith("0"):
        return _zh_digits_to_words(value)
    if len(value) > 8:
        return _zh_digits_to_words(value)
    return _zh_int_to_words(int(value or "0"))


def _zh_digits_to_words(value: str) -> str:
    return "".join(_ZH_DIGITS[int(ch)] for ch in value if ch.isdigit())


def _zh_int_to_words(number: int) -> str:
    number = int(number)
    if number == 0:
        return _ZH_DIGITS[0]
    if number < 0:
        return "负" + _zh_int_to_words(-number)
    if number < 10_000:
        return _zh_under_10000_to_words(number)
    if number < 100_000_000:
        high = number // 10_000
        low = number % 10_000
        words = f"{_zh_int_to_words(high)}万"
        if low:
            words += "零" if low < 1000 else ""
            words += _zh_under_10000_to_words(low)
        return words
    high = number // 100_000_000
    low = number % 100_000_000
    words = f"{_zh_int_to_words(high)}亿"
    if low:
        words += "零" if low < 10_000_000 else ""
        words += _zh_int_to_words(low)
    return words


def _zh_under_10000_to_words(number: int) -> str:
    units = ((1000, "千"), (100, "百"), (10, "十"), (1, ""))
    pieces: list[str] = []
    pending_zero = False
    remaining = int(number)
    for unit_value, unit_name in units:
        digit = remaining // unit_value
        remaining %= unit_value
        if digit:
            if pending_zero and pieces:
                pieces.append("零")
            pieces.append(f"{_ZH_DIGITS[digit]}{unit_name}")
            pending_zero = False
        elif pieces and remaining:
            pending_zero = True
    result = "".join(pieces)
    if 10 <= number < 20 and result.startswith("一十"):
        result = result[1:]
    return result
