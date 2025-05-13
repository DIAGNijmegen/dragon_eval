"""
redaction_scoring.py
----------------------------------
Utilities to score PII-redacted text at character level.

Usage
-----
from dragon_eval.redaction_score import (
    evaluate_redaction,                  # weighted, recommended
)

original  = "Op 12 maart 2023 sprak Jan Jansen om 10:00."
ground_tr = "Op <DATUM> sprak <PERSOON> om <TIJD>."
prediction = "Op <PERSOON> sprak <DATUM> om <TIJD>."

print(evaluate_redaction(original, ground_tr, prediction, alpha=0.7))
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Tuple

from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# 1.  Basic helpers
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Collapse runs of any whitespace into a single space."""
    return " ".join(text.split())


def extract_tags(text: str) -> set[str]:
    """Return the set of all <TAG> occurrences in *text*."""
    return set(re.findall(r"<[^<>]+>", text))


# ---------------------------------------------------------------------------
# 2.  Tag-placeholder machinery
# ---------------------------------------------------------------------------


def _replace_tags_with_placeholders(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace every <TAG> with a single private-use-area (PUA) character
    so difflib cannot split a tag in half.

    Returns
    -------
    new_text : str
        The text with placeholders.
    mapping  : dict { placeholder-char -> original <TAG> }
    """
    if any("\ue000" <= ch <= "\uf8ff" for ch in text):
        raise ValueError(
            "The input text contains characters in the Unicode private-use area (U+E000 to U+F8FF), "
            "which are reserved for placeholders and may cause conflicts."
        )

    pieces: List[str] = []
    mapping: Dict[str, str] = {}

    last = 0
    for i, m in enumerate(re.finditer(r"<[^<>]+>", text)):
        pieces.append(text[last:m.start()])

        placeholder = chr(0xE000 + i)  # U+E000 … never occurs in normal text
        pieces.append(placeholder)
        mapping[placeholder] = m.group()

        last = m.end()

    pieces.append(text[last:])
    return "".join(pieces), mapping


def _get_tag_labels(original: str, redacted: str) -> List[str]:
    """
    For every character in *original* return either
        'O'            – normal text
        '<DATUM>' …    – the tag that replaced that character
    """
    original = normalize_whitespace(original)
    redacted = normalize_whitespace(redacted)

    redacted_proc, placeholders = _replace_tags_with_placeholders(redacted)
    sm = difflib.SequenceMatcher(None, original, redacted_proc)

    labels: List[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            labels.extend(["O"] * (i2 - i1))
            continue

        # Any non-equal op (replace / insert / delete)
        # might contain a placeholder → map to its tag
        segment = redacted_proc[j1:j2]
        tag_label = "O"
        for ch in segment:
            if ch in placeholders:
                tag_label = placeholders[ch]
                break
        labels.extend([tag_label] * (i2 - i1))

    return labels


# ---------------------------------------------------------------------------
# 3.  Weighted evaluator
# ---------------------------------------------------------------------------


def _to_binary(labels: List[str]) -> List[str]:
    """Collapse every concrete tag to 'TAG' so we can measure plain redaction."""
    return ["TAG" if lbl != "O" else "O" for lbl in labels]


def evaluate_redaction(
    original: str,
    ground_truth: str,
    prediction: str,
    alpha: float = 0.7,
) -> dict:
    """
    Mixed-severity scoring. Blended Redaction F1: mean over redaction_f1
    and type_f1, where redaction_f1 is the F1 for 'TAG' vs 'O' and type_f1
    is the strict macro-F1 across concrete tags.

    Parameters
    ----------
    alpha : float, default 0.7
        Weight given to *redaction-F1* (PII vs. non-PII).
        1.0 → ignore tag types completely.
        0.0 → equivalent to the strict evaluator.

    Returns
    -------
    dict with keys
        overall_score : weighted combination
        redaction_f1  : F1 for "TAG" vs "O"
        type_f1       : strict macro-F1 across concrete tags
        binary_report : classification_report for the binary case
        type_report   : classification_report for the strict case
        alpha         : the value used
    """
    # ------------------------------------------------------------------
    # character labels
    # ------------------------------------------------------------------
    true_labels = _get_tag_labels(original, ground_truth)
    pred_labels = _get_tag_labels(original, prediction)

    length = max(len(true_labels), len(pred_labels))
    true_labels += ["O"] * (length - len(true_labels))
    pred_labels += ["O"] * (length - len(pred_labels))

    # ------------------------------------------------------------------
    # (1) strict report  – correct tag type
    # ------------------------------------------------------------------
    strict_tags = sorted(set(true_labels + pred_labels) - {"O"})
    strict_report = classification_report(
        true_labels, pred_labels, labels=strict_tags, zero_division=0, output_dict=True
    )
    type_f1 = strict_report["macro avg"]["f1-score"] if strict_tags else 1.0

    # ------------------------------------------------------------------
    # (2) binary report – was the span redacted at all?
    # ------------------------------------------------------------------
    bin_true = _to_binary(true_labels)
    bin_pred = _to_binary(pred_labels)
    bin_report = classification_report(
        bin_true, bin_pred, labels=["TAG"], zero_division=0, output_dict=True
    )
    redaction_f1 = bin_report["TAG"]["f1-score"]

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------
    overall = alpha * redaction_f1 + (1.0 - alpha) * type_f1
    return {
        "blended_redaction_f1": overall,
        "redaction_f1": redaction_f1,
        "type_f1": type_f1,
        "binary_report": bin_report,
        "type_report": strict_report,
        "alpha": alpha,
    }
