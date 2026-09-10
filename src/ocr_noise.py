"""A confusion model for realistic Arabic OCR errors.

No public Arabic OCR-correction corpus exists, so training pairs are synthesized:
take clean Arabic text and corrupt it the way a real OCR engine does. The point
is that the corruption is *not* random — it follows the failure modes that come
from Arabic script itself.

The dominant one is i'jam: many Arabic letters share an identical skeleton
(rasm) and differ only in the number and position of dots. A scanner that loses
or invents a dot turns ب into ت into ث. That single class accounts for most real
Arabic OCR error mass, so it gets most of the weight here.

Validate the noise profile against real engine output before trusting it —
see scripts/calibrate_noise.py.
"""

from __future__ import annotations

import random
import re
import unicodedata

# --- Letters sharing a skeleton, separated only by dots (i'jam) ---------------
DOT_GROUPS: list[str] = [
    "بتثنيى",   # single tooth: 1-below, 2-above, 3-above, ن/ي final forms
    "جحخ",      # 1-inside, none, 1-above
    "دذ",
    "رز",
    "سش",
    "صض",
    "طظ",
    "عغ",
    "فق",
    "كل",       # confusable in many scanned fonts
    "هة",       # also a real orthographic confusion, not just OCR
]

# --- Hamza carriers: collapse or swap under poor resolution ------------------
HAMZA_GROUP = "اأإآءئؤ"

# --- Shapes that blur together in degraded scans -----------------------------
SHAPE_GROUPS: list[str] = [
    "وؤ",
    "مه",
    "لا",
]

ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
WESTERN = "0123456789"
TASHKEEL = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652"
TATWEEL = "\u0640"
ARABIC_LETTER = re.compile(r"[\u0621-\u064A]")


def _build_confusions() -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    for group in DOT_GROUPS + SHAPE_GROUPS + [HAMZA_GROUP]:
        for ch in group:
            table.setdefault(ch, [])
            table[ch].extend(c for c in group if c != ch)
    for a, b in zip(ARABIC_INDIC, WESTERN):
        table.setdefault(a, []).append(b)
        table.setdefault(b, []).append(a)
    return {k: sorted(set(v)) for k, v in table.items()}


CONFUSIONS = _build_confusions()

# Relative frequency of each corruption type. Roughly mirrors the error mix
# reported for Arabic OCR on printed text: dot errors dominate, segmentation
# (word splitting/merging) is the next largest bucket because the script is
# cursive and inter-word gaps are narrow.
DEFAULT_WEIGHTS: dict[str, float] = {
    "confuse": 0.42,      # swap a letter for a skeleton-mate
    "split_word": 0.12,   # insert a space mid-word
    "merge_word": 0.10,   # delete a space between words
    "drop_char": 0.10,    # dropped stroke
    "insert_char": 0.06,  # spurious stroke read as a letter
    "tatweel": 0.07,      # kashida elongation misread as a character
    "diacritic": 0.07,    # hallucinated tashkeel from page speckle
    "transpose": 0.06,    # adjacent character swap
}


class OCRNoiser:
    """Corrupt clean Arabic text with OCR-shaped errors.

    Args:
        rate: probability that any given character becomes a corruption site.
              ~0.05 mimics a good scan, ~0.15 a degraded or historical one.
        seed: fixed for reproducible datasets.
        weights: override the corruption-type mix.
    """

    def __init__(self, rate: float = 0.08, seed: int | None = 42,
                 weights: dict[str, float] | None = None) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate
        self.rng = random.Random(seed)
        w = dict(DEFAULT_WEIGHTS)
        if weights:
            unknown = set(weights) - set(DEFAULT_WEIGHTS)
            if unknown:
                raise ValueError(f"unknown corruption types: {sorted(unknown)}")
            w.update(weights)
        total = sum(w.values())
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        self.ops = list(w)
        self.probs = [v / total for v in w.values()]

    # -- individual corruptions ------------------------------------------------
    def _confuse(self, ch: str) -> str:
        options = CONFUSIONS.get(ch)
        return self.rng.choice(options) if options else ch

    def _random_letter(self) -> str:
        return self.rng.choice("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")

    # -- main entry point ------------------------------------------------------
    def corrupt(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        out: list[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if not ARABIC_LETTER.match(ch) and not ch.isdigit():
                out.append(ch)
                i += 1
                continue
            if self.rng.random() >= self.rate:
                out.append(ch)
                i += 1
                continue

            op = self.rng.choices(self.ops, weights=self.probs, k=1)[0]

            if op == "confuse":
                out.append(self._confuse(ch))
            elif op == "split_word":
                out.append(ch)
                out.append(" ")
            elif op == "merge_word":
                out.append(ch)
                # swallow the next run of whitespace, if any
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                i = j - 1
            elif op == "drop_char":
                pass  # stroke lost
            elif op == "insert_char":
                out.append(ch)
                out.append(self._random_letter())
            elif op == "tatweel":
                out.append(ch)
                out.append(TATWEEL)
            elif op == "diacritic":
                out.append(ch)
                out.append(self.rng.choice(TASHKEEL))
            elif op == "transpose" and i + 1 < len(text):
                out.append(text[i + 1])
                out.append(ch)
                i += 1
            else:
                out.append(ch)
            i += 1

        # Collapse any runs of spaces the corruptions introduced, but keep the
        # single spaces that split_word deliberately created.
        return re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()


if __name__ == "__main__":
    import sys

    # Windows consoles default to cp1252 and cannot print Arabic.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sample = "العلم نور يضيء طريق الإنسان في الحياة، والجهل ظلام دامس."
    print("clean :", sample)
    for rate in (0.05, 0.10, 0.20):
        print(f"r={rate:<5}:", OCRNoiser(rate=rate, seed=7).corrupt(sample))
