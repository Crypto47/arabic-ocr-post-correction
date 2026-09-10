"""Turn a clean Arabic corpus into (noisy -> clean) training pairs.

Accepts a file or a directory in whichever shape the Kaggle corpus arrived in:
.txt, .jsonl, .json, .csv or .tsv. Text columns are auto-detected. Segments the
text into sentence-ish chunks, corrupts each with OCRNoiser, and writes
chat-format JSONL ready for src/train.py.

Usage:
    python src/build_dataset.py --input data/raw --output data/processed \
        --max-pairs 60000 --min-rate 0.04 --max-rate 0.18
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

from ocr_noise import OCRNoiser

INSTRUCTION = "صحّح أخطاء المسح الضوئي في النص التالي وأعد كتابته بشكل صحيح:"

# Split on Arabic and Latin sentence enders, keeping the terminator.
SENTENCE_END = re.compile(r"(?<=[.!?؟।\n])\s+")
ARABIC_CHAR = re.compile(r"[\u0621-\u064A]")


SUPPORTED = {".txt", ".jsonl", ".json", ".csv", ".tsv"}


def _pick_text_column(header: list[str], preferred: str) -> str | None:
    """Kaggle Arabic corpora label their text column half a dozen ways."""
    if preferred in header:
        return preferred
    for guess in ("text", "content", "body", "article", "sentence",
                  "Text", "Content", "Article", "نص", "المحتوى"):
        if guess in header:
            return guess
    # Fall back to the widest-looking column rather than giving up.
    return header[0] if header else None


def iter_text(path: Path, text_field: str):
    """Yield raw text blocks from a file or directory.

    Handles the formats Kaggle Arabic corpora actually ship in: loose .txt
    files, JSON Lines, plain JSON arrays, and CSV/TSV exports.
    """
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED)
    else:
        files = [path]
    if not files:
        sys.exit(f"no {'/'.join(sorted(SUPPORTED))} files found under {path}")

    print(f"reading {len(files)} file(s) from {path}")
    for f in files:
        suffix = f.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            delim = "\t" if suffix == ".tsv" else ","
            with f.open(encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.DictReader(fh, delimiter=delim)
                col = _pick_text_column(reader.fieldnames or [], text_field)
                if col is None:
                    continue
                print(f"  {f.name}: using column {col!r}")
                for row in reader:
                    if value := (row.get(col) or "").strip():
                        yield value
        elif suffix == ".jsonl":
            with f.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and (value := row.get(text_field)):
                        yield value
        elif suffix == ".json":
            try:
                blob = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            rows = blob if isinstance(blob, list) else [blob]
            for row in rows:
                if isinstance(row, dict) and (value := row.get(text_field)):
                    yield value
                elif isinstance(row, str):
                    yield row
        else:
            yield f.read_text(encoding="utf-8", errors="replace")


def segments(blocks, min_chars: int, max_chars: int):
    """Split text blocks into chunks small enough for a 0.5B context."""
    for block in blocks:
        for chunk in SENTENCE_END.split(block):
            chunk = re.sub(r"\s+", " ", chunk).strip()
            if len(chunk) < min_chars:
                continue
            # Arabic-dominant only: skips nav junk, English boilerplate, tables.
            arabic = len(ARABIC_CHAR.findall(chunk))
            if arabic / max(len(chunk), 1) < 0.5:
                continue
            while len(chunk) > max_chars:
                cut = chunk.rfind(" ", 0, max_chars)
                cut = cut if cut > min_chars else max_chars
                yield chunk[:cut].strip()
                chunk = chunk[cut:].strip()
            if len(chunk) >= min_chars:
                yield chunk


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path, help="output directory")
    p.add_argument("--text-field", default="text", help="text field/column name; auto-detected if absent")
    p.add_argument("--max-pairs", type=int, default=60_000)
    p.add_argument("--eval-frac", type=float, default=0.05)
    p.add_argument("--min-chars", type=int, default=40)
    p.add_argument("--max-chars", type=int, default=400)
    p.add_argument("--min-rate", type=float, default=0.04,
                   help="lightest corruption (clean scan)")
    p.add_argument("--max-rate", type=float, default=0.18,
                   help="heaviest corruption (degraded/historical scan)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    train_path = args.output / "train.jsonl"
    eval_path = args.output / "eval.jsonl"

    seen: set[str] = set()
    n_train = n_eval = 0
    skipped_identical = 0

    with train_path.open("w", encoding="utf-8") as ftr, \
         eval_path.open("w", encoding="utf-8") as fev:
        for clean in segments(iter_text(args.input, args.text_field),
                              args.min_chars, args.max_chars):
            if n_train + n_eval >= args.max_pairs:
                break
            key = clean[:120]
            if key in seen:
                continue
            seen.add(key)

            # Spread corruption severity so the model sees clean scans and
            # wrecked ones, not just one noise level.
            rate = rng.uniform(args.min_rate, args.max_rate)
            noisy = OCRNoiser(rate=rate, seed=rng.randrange(1 << 30)).corrupt(clean)
            if noisy == clean:
                skipped_identical += 1
                continue

            record = {"messages": [
                {"role": "user", "content": f"{INSTRUCTION}\n\n{noisy}"},
                {"role": "assistant", "content": clean},
            ], "noise_rate": round(rate, 4)}

            line = json.dumps(record, ensure_ascii=False) + "\n"
            if rng.random() < args.eval_frac:
                fev.write(line); n_eval += 1
            else:
                ftr.write(line); n_train += 1

    print(f"train: {n_train:,} pairs -> {train_path}")
    print(f"eval : {n_eval:,} pairs -> {eval_path}")
    if skipped_identical:
        print(f"(skipped {skipped_identical:,} segments where noise was a no-op)")


if __name__ == "__main__":
    main()
