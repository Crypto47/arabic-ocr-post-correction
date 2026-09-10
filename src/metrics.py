"""Character and word error rate, dependency-free.

CER is the metric that matters here: OCR correction is judged on how many
characters you fix, and Arabic words are morphologically dense enough that WER
alone hides real progress.
"""

from __future__ import annotations


def _levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate. 0.0 is perfect; can exceed 1.0 on bad insertions."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    if not ref_words:
        return 0.0 if not hypothesis.split() else 1.0
    return _levenshtein(ref_words, hypothesis.split()) / len(ref_words)


def corpus_rates(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Aggregate by total edits over total length, not the mean of per-line
    rates — otherwise short lines dominate the score."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must be the same length")
    c_edits = c_len = w_edits = w_len = 0
    for ref, hyp in zip(references, hypotheses):
        c_edits += _levenshtein(ref, hyp)
        c_len += len(ref)
        rw, hw = ref.split(), hyp.split()
        w_edits += _levenshtein(rw, hw)
        w_len += len(rw)
    return {
        "cer": c_edits / max(c_len, 1),
        "wer": w_edits / max(w_len, 1),
        "n": len(references),
    }
