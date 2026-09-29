from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

from stage4.features import normalize_text


CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "y",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
)
GENERIC_TOKENS = {
    "chateau",
    "estate",
    "hozyaystvo",
    "pomeste",
    "usadba",
    "vineyard",
    "vineyards",
    "vino",
    "vinodelnya",
    "winery",
    "wine",
    "wines",
    "zmv",
}

# Latin glyphs commonly emitted by OCR for visually identical Cyrillic letters.
# The unmodified OCR spelling remains a candidate as well, so genuine Latin
# manufacturer names are not destroyed by this normalization.
OCR_LATIN_HOMOGLYPHS = str.maketrans(
    {
        "c": "s",  # C -> Cyrillic С
        "h": "n",  # H -> Cyrillic Н
        "p": "r",  # P -> Cyrillic Р
        "x": "h",  # X -> Cyrillic Х
        "y": "u",  # Y -> Cyrillic У
    }
)
AMBIGUOUS_SINGLE_TOKENS = {
    "dagestan",
    "don",
    "krym",
    "kuban",
    "samara",
}


@dataclass(frozen=True)
class TaxonomyMatch:
    value: str | None
    score: float
    runner_up: str | None
    runner_score: float
    margin: float
    mode: str
    matched_text: str | None
    values: tuple[str, ...] = ()
    candidate_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fuzzy_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", normalize_text(value))
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(without_marks.translate(CYRILLIC_TO_LATIN).split())


def _unique_tokens(tokens: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(tokens))


@lru_cache(maxsize=512)
def _aliases(value: str) -> tuple[str, ...]:
    full = fuzzy_key(value)
    tokens = full.split()
    meaningful_tokens = _unique_tokens(
        token for token in tokens if token not in GENERIC_TOKENS
    )
    meaningful = " ".join(meaningful_tokens)
    aliases = {
        alias
        for alias in (full, meaningful)
        if len(alias.replace(" ", "")) >= 4
        and not (" " not in alias and alias in AMBIGUOUS_SINGLE_TOKENS)
    }
    return tuple(sorted(aliases, key=len, reverse=True))


def _source_keys(value: str) -> tuple[str, ...]:
    base = fuzzy_key(value)
    if not base:
        return ()
    homoglyph = " ".join(
        token.translate(OCR_LATIN_HOMOGLYPHS) if len(token) >= 6 else token
        for token in base.split()
    )
    return tuple(dict.fromkeys((base, homoglyph)))


def normalized_edit_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


def _prefix_similarity(left: str, right: str) -> float:
    if " " in left or " " in right:
        return 0.0
    compact_left = left.replace(" ", "")
    compact_right = right.replace(" ", "")
    shorter, longer = sorted((compact_left, compact_right), key=len)
    if shorter in GENERIC_TOKENS:
        return 0.0
    if len(shorter) < 6 or not longer.startswith(shorter):
        return 0.0
    ratio = len(shorter) / len(longer)
    if ratio < 0.58:
        return 0.0
    return 0.94 + 0.06 * ratio


def _window_similarity(query: str, alias: str) -> float:
    if not query or not alias:
        return 0.0
    query_tokens = query.split()
    alias_tokens = alias.split()
    if any(
        query_tokens[start : start + len(alias_tokens)] == alias_tokens
        for start in range(len(query_tokens) - len(alias_tokens) + 1)
    ):
        return 1.0
    alias_size = len(alias_tokens)
    scores = [normalized_edit_similarity(query, alias), _prefix_similarity(query, alias)]
    for size in range(max(1, alias_size - 1), min(len(query_tokens), alias_size + 1) + 1):
        for start in range(len(query_tokens) - size + 1):
            window = " ".join(query_tokens[start : start + size])
            scores.extend(
                (
                    normalized_edit_similarity(window, alias),
                    _prefix_similarity(window, alias),
                )
            )
    return max(scores)


@lru_cache(maxsize=8)
def _taxonomy_groups_cached(unique_values: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Merge catalog spellings that expose the same distinctive alias."""
    aliases = {value: set(_aliases(value)) for value in unique_values}
    parents = list(range(len(unique_values)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(unique_values)):
        for right in range(left + 1, len(unique_values)):
            shared = aliases[unique_values[left]] & aliases[unique_values[right]]
            if any(len(alias.replace(" ", "")) >= 6 for alias in shared):
                union(left, right)

    grouped: dict[int, list[str]] = {}
    for index, value in enumerate(unique_values):
        grouped.setdefault(find(index), []).append(value)
    return tuple(tuple(group) for group in grouped.values())


def _taxonomy_groups(values: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    return _taxonomy_groups_cached(tuple(sorted(set(values))))


def resolve_taxonomy_value(
    lines: list[dict[str, Any]],
    values: Iterable[str],
    *,
    hard_threshold: float = 0.86,
    min_margin: float = 0.08,
) -> TaxonomyMatch:
    sources = [
        (key, float(line.get("score") or 0.0))
        for line in lines
        for key in _source_keys(str(line.get("normalized") or line.get("text") or ""))
    ]
    sources = [(text, confidence) for text, confidence in sources if text]
    if not sources:
        return TaxonomyMatch(None, 0.0, None, 0.0, 0.0, "none", None)
    combined = " ".join(text for text, _ in sources)
    sources.append((combined, sum(score for _, score in sources) / len(sources)))

    ranked: list[tuple[float, str, str, tuple[str, ...]]] = []
    for group in _taxonomy_groups(values):
        best_score = 0.0
        best_text = ""
        best_value = group[0]
        for value in group:
            for alias in _aliases(value):
                compact_length = len(alias.replace(" ", ""))
                for text, confidence in sources:
                    similarity = _window_similarity(text, alias)
                    score = similarity * (0.90 + 0.10 * max(0.0, min(confidence, 1.0)))
                    if compact_length <= 4 and similarity < 1.0:
                        score *= 0.92
                    if score > best_score:
                        best_score = score
                        best_text = text
                        best_value = value
        ranked.append((best_score, best_value, best_text, group))
    ranked.sort(reverse=True)
    best_score, best_value, best_text, best_values = ranked[0]
    runner_score, runner_value, _, _ = (
        ranked[1] if len(ranked) > 1 else (0.0, None, "", ())
    )
    margin = best_score - runner_score
    strong_fuzzy = best_score >= 0.82 and margin >= 0.15
    mode = (
        "hard"
        if strong_fuzzy or (best_score >= hard_threshold and margin >= min_margin)
        else "none"
    )
    return TaxonomyMatch(
        best_value if mode == "hard" else None,
        float(best_score),
        runner_value,
        float(runner_score),
        float(margin),
        mode,
        best_text or None,
        best_values if mode == "hard" else (),
    )


def manufacturer_values(catalog_by_slug: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(record.get("manufacturer") or "").strip()
                for record in catalog_by_slug.values()
                if str(record.get("manufacturer") or "").strip()
            }
        )
    )


def apply_manufacturer_gate(
    candidates: np.ndarray,
    visual_scores: np.ndarray,
    ocr_scores: np.ndarray,
    slugs: np.ndarray,
    catalog_by_slug: dict[str, dict[str, Any]],
    matches: list[dict[str, Any]],
) -> tuple[np.ndarray, list[set[int]]]:
    """Inject the matched manufacturer's wines and return the hard allowed sets."""
    gated = candidates.copy()
    allowed_sets: list[set[int]] = []
    manufacturers = [
        str(catalog_by_slug[str(slug)].get("manufacturer") or "") for slug in slugs
    ]
    width = candidates.shape[1]
    for row, match in enumerate(matches):
        if match["mode"] != "hard" or not match["value"]:
            allowed_sets.append(set())
            continue
        matched_values = set(match.get("values") or (match["value"],))
        allowed = [
            index for index, manufacturer in enumerate(manufacturers)
            if manufacturer in matched_values
        ]
        match["candidate_count"] = len(allowed)
        prescore = 0.65 * visual_scores[row] + 0.35 * ocr_scores[row]
        allowed.sort(key=lambda index: float(prescore[index]), reverse=True)
        selected = allowed[:width]
        seen = set(selected)
        selected.extend(int(index) for index in candidates[row] if int(index) not in seen)
        if len(selected) < width:
            selected.extend(
                int(index)
                for index in np.argsort(-visual_scores[row])
                if int(index) not in seen
            )
        gated[row] = np.asarray(selected[:width], dtype=np.int64)
        allowed_sets.append(set(allowed[:width]))
    return gated, allowed_sets
