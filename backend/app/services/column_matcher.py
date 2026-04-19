"""Column matching helpers for schema validation."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Set


def normalize_column_name(name: str) -> str:
    if name is None:
        return ""
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(name: str) -> Set[str]:
    normalized = normalize_column_name(name)
    if not normalized:
        return set()
    return {token for token in normalized.split(" ") if token}


def fuzzy_match_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    left_norm = normalize_column_name(left)
    right_norm = normalize_column_name(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0

    ratio = SequenceMatcher(None, left_norm.replace(" ", ""), right_norm.replace(" ", "")).ratio()
    left_tokens = _tokenize(left_norm)
    right_tokens = _tokenize(right_norm)
    union = left_tokens | right_tokens
    jaccard = (len(left_tokens & right_tokens) / len(union)) if union else 0.0
    score = (0.7 * ratio) + (0.3 * jaccard)
    return float(round(score, 4))


def find_best_matches(
    expected: Iterable[str],
    actual: Iterable[str],
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, object]:
    expected_list = [str(col) for col in expected]
    actual_list = [str(col) for col in actual]
    thresholds = thresholds or {
        "exact": 1.0,
        "fuzzy_high": 0.85,
        "fuzzy_medium": 0.7,
    }

    candidates: List[Dict[str, object]] = []
    for exp in expected_list:
        for act in actual_list:
            score = fuzzy_match_score(exp, act)
            exact = normalize_column_name(exp) == normalize_column_name(act)
            candidates.append({
                "expected": exp,
                "actual": act,
                "score": score,
                "exact": exact,
            })

    candidates.sort(
        key=lambda item: (item["score"], 1 if item["exact"] else 0),
        reverse=True,
    )

    matched_expected: Set[str] = set()
    matched_actual: Set[str] = set()
    matches: List[Dict[str, object]] = []

    for candidate in candidates:
        exp = candidate["expected"]
        act = candidate["actual"]
        score = float(candidate["score"])
        if score < thresholds["fuzzy_medium"]:
            break
        if exp in matched_expected or act in matched_actual:
            continue

        if score >= thresholds["fuzzy_high"]:
            method = "fuzzy_high"
        else:
            method = "fuzzy_medium"
        if candidate["exact"]:
            method = "exact"

        matches.append({
            "expected": exp,
            "actual": act,
            "score": score,
            "method": method,
        })
        matched_expected.add(exp)
        matched_actual.add(act)

    unmatched_expected = sorted([col for col in expected_list if col not in matched_expected])
    unmatched_actual = sorted([col for col in actual_list if col not in matched_actual])

    mapping_actual_to_expected = {match["actual"]: match["expected"] for match in matches}
    mapping_expected_to_actual = {match["expected"]: match["actual"] for match in matches}

    return {
        "matches": matches,
        "unmatched_expected": unmatched_expected,
        "unmatched_actual": unmatched_actual,
        "mapping_actual_to_expected": mapping_actual_to_expected,
        "mapping_expected_to_actual": mapping_expected_to_actual,
        "thresholds": thresholds,
    }
