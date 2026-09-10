"""Score the finetuned model against the raw OCR baseline.

The headline number is the CER of the noisy input versus the CER of the model's
correction, both measured against the clean reference. A model that does not
beat the baseline is not doing anything, however good its loss curve looked.

Usage:
    python src/evaluate.py --adapter outputs/arabic-ocr-post-correction/final \
        --eval-file data/processed/eval.jsonl --limit 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from metrics import cer, corpus_rates

# NOTE: infer (and therefore torch/peft) is imported inside main() on purpose,
# so the parsing and scoring helpers below stay importable — and unit-testable —
# on a machine with no GPU stack installed.

BUCKETS = ["light (<8%)", "medium (8-14%)", "heavy (>14%)"]


def read_eval(path: Path, limit: int | None):
    noisy, clean, rates = [], [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if limit and len(noisy) >= limit:
                break
            row = json.loads(line)
            prompt = row["messages"][0]["content"]
            noisy.append(prompt.split("\n\n", 1)[-1])
            clean.append(row["messages"][1]["content"])
            rates.append(row.get("noise_rate", 0.0))
    return noisy, clean, rates


def severity(rate: float) -> str:
    if rate < 0.08:
        return BUCKETS[0]
    return BUCKETS[1] if rate < 0.14 else BUCKETS[2]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None,
                    help="omit to score the untuned base model")
    ap.add_argument("--eval-file", type=Path, default=Path("data/processed/eval.jsonl"))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default: derived from input length")
    ap.add_argument("--out", type=Path, default=Path("results/eval.json"))
    args = ap.parse_args()

    noisy, clean, rates = read_eval(args.eval_file, args.limit)
    if not noisy:
        sys.exit(f"no rows read from {args.eval_file}")
    print(f"Scoring {len(noisy)} segments ...")

    from infer import correct, load  # heavy deps, only needed to actually run

    model, tokenizer = load(args.base, args.adapter)
    preds = correct(model, tokenizer, noisy, batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens)

    baseline = corpus_rates(clean, noisy)   # what the OCR engine handed you
    corrected = corpus_rates(clean, preds)  # what the model handed you

    reduction = (1 - corrected["cer"] / baseline["cer"]) * 100 if baseline["cer"] else 0.0
    regressions = sum(
        1 for c, n, p in zip(clean, noisy, preds) if cer(c, p) > cer(c, n)
    )

    print()
    print(f"{'':<22}{'CER':>9}{'WER':>9}")
    print("-" * 40)
    print(f"{'raw OCR (baseline)':<22}{baseline['cer']:>9.4f}{baseline['wer']:>9.4f}")
    print(f"{'after correction':<22}{corrected['cer']:>9.4f}{corrected['wer']:>9.4f}")
    print("-" * 40)
    print(f"CER reduction: {reduction:.1f}%")
    print(f"Made worse on {regressions}/{len(noisy)} segments "
          f"({regressions / len(noisy) * 100:.1f}%)")

    # Per-severity breakdown: a model can look strong overall while failing on
    # exactly the degraded scans that actually needed correcting.
    print()
    print(f"{'severity':<18}{'n':>5}{'base CER':>10}{'model CER':>11}")
    print("-" * 44)
    per_bucket = {}
    labels = [severity(r) for r in rates]
    for name in BUCKETS:
        idx = [i for i, b in enumerate(labels) if b == name]
        if not idx:
            continue
        b = corpus_rates([clean[i] for i in idx], [noisy[i] for i in idx])
        m = corpus_rates([clean[i] for i in idx], [preds[i] for i in idx])
        per_bucket[name] = {
            "n": len(idx), "baseline_cer": b["cer"], "model_cer": m["cer"],
        }
        print(f"{name:<18}{len(idx):>5}{b['cer']:>10.4f}{m['cer']:>11.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "base_model": args.base,
        "adapter": args.adapter,
        "n": len(noisy),
        "baseline": baseline,
        "corrected": corrected,
        "cer_reduction_pct": reduction,
        "regressions": regressions,
        "by_severity": per_bucket,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
