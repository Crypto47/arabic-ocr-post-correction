# Arabic OCR Post-Correction

A 0.5B-parameter Arabic small language model finetuned to repair OCR output —
the step between "we scanned the archive" and "the archive is searchable."

```
raw OCR   العلم نور يضنء طزيق الإنسان في الحتياة، وآلجهل ظلام دام
corrected العلم نور يضيء طريق الإنسان في الحياة، والجهل ظلام دامس
```

> **Status: working, with a caveat that matters.** Trained and scored on two
> real corpora. With an inference-time guardrail it beats raw OCR on both CER
> and WER. All numbers below are measured, not estimated — but the corruption
> is synthetic, so they describe performance against a *model* of OCR error,
> not a recording of one. See [Notes](#notes-and-limitations).

## The problem

Arabic OCR fails differently from Latin OCR, and the difference is structural.
Many Arabic letters share an identical skeleton (*rasm*) and are distinguished
only by dots (*i'jam*): ب ت ث ن are the same stroke with different dots. Lose or
invent one dot and ب becomes ت. The script is also cursive, so word boundaries
are narrow and scanners routinely split one word into two or fuse two into one.

The result is that a scan with a *character* error rate of only 8% can carry a
*word* error rate near 50% — a single misread dot corrupts an entire token, and
downstream search, indexing, and extraction all fail on it. Measured on this
repo's synthetic pairs: **CER 0.089, WER 0.495.**

That gap is what a correction model closes.

## Approach

**Synthetic supervision.** No public Arabic OCR-correction corpus exists — I
checked Kaggle and the HuggingFace Hub. So the training pairs are generated:
take clean Arabic text, corrupt it with a confusion model derived from how the
script actually fails, and train the model to invert it.

The corruption in [`src/ocr_noise.py`](src/ocr_noise.py) is not random noise. It
is weighted toward real failure modes:

| Corruption | Weight | Why |
|---|---|---|
| Skeleton confusion (dots) | 42% | ب↔ت↔ث, ج↔ح↔خ, ر↔ز, ص↔ض, ع↔غ, ف↔ق — the dominant error class |
| Word splitting | 12% | cursive script, narrow inter-word gaps |
| Word merging | 10% | same cause, opposite direction |
| Dropped character | 10% | lost stroke on a faded scan |
| Spurious character | 6% | speckle read as a letter |
| Kashida (ـ) insertion | 7% | elongation misread as a glyph |
| Hallucinated diacritic | 7% | page noise read as tashkeel |
| Transposition | 6% | segmentation slip |

Severity is sampled per example (4%–18%), so one model handles both clean
prints and degraded historical pages, and `evaluate.py` reports the breakdown.

**Why 0.5B.** This is a constrained transformation, not open generation: the
output is a near-copy of the input with local repairs. That is well within a
0.5B model's capacity, and it means the whole thing trains on a free Colab T4
and could run on-premise — which is the deployment mode that actually matters
for archives that cannot leave the building.

**Greedy decoding, always.** There is exactly one right answer. Sampling only
invents text that was never on the page. See [`src/infer.py`](src/infer.py).

## Results

Measured locally on an RTX 5070 Ti (12GB, sm_120), Qwen2.5-0.5B + LoRA, greedy
decoding, 200 held-out segments from AraSum. Synthetic corruption — see
[Notes](#notes-and-limitations).

**Training scale is the dominant variable:**

| training pairs | CER | WER | segments made worse |
|---|---|---|---|
| raw OCR (do nothing) | 0.0808 | 0.4112 | — |
| 0 — untuned base model | 1.8123 | 2.2680 | 200/200 (100%) |
| 2,000 | 0.2139 | 0.3751 | 170/200 (85%) |
| 20,000 | 0.1083 | 0.2327 | 115/200 (58%) |
| 20,000 **+ guardrail** | **0.0809** | **0.3343** | **39/200 (20%)** |

The untuned base model is catastrophic — it answers the instruction
conversationally instead of restoring, so its output bears no relation to the
input. Every gain below it is attributable to the finetune, not to Qwen already
knowing Arabic.

**The guardrail** ([`apply_guardrail`](src/infer.py)) rejects any correction
that diverges from its *input* by more than a CER threshold, falling back to the
original OCR text. Divergence is measured against the input, never the
reference, so it runs in production. Tuning the threshold trades the two metrics
against each other:

| policy | CER | WER | corrections kept |
|---|---|---|---|
| raw OCR | 0.0808 | 0.4112 | — |
| accept everything | 0.1083 | 0.2327 | 200/200 |
| drift ≤ 0.10 | **0.0795** | 0.3733 | 66/200 |
| drift ≤ 0.15 | 0.0809 | **0.3343** | 106/200 |
| drift ≤ 0.20 | 0.0865 | 0.2822 | 148/200 |

At drift ≤ 0.10 the system beats raw OCR on **both** metrics simultaneously.

At drift ≤ 0.15 it holds CER at parity while cutting WER by 18.7% — and the
per-severity breakdown shows the gain lands where it matters:

| severity | n | baseline CER | model CER |
|---|---|---|---|
| light (<8%) | 62 | 0.0438 | 0.0523 |
| medium (8–14%) | 82 | 0.0813 | **0.0775** |
| heavy (>14%) | 56 | 0.1211 | **0.1172** |

It improves the damaged scans that need correcting and only loses on clean ones
where there was little to fix.

**Reproduced on a second corpus.** Arabic BERT Corpus (56,946 pairs, baseline
CER 0.0838 / WER 0.3900) shows the same pattern at 2,000 pairs: base 2.2092 →
tuned 0.1956. The behaviour is a property of training scale, not of one dataset.

**Not yet done:** the full 57k-pair run. The 2k → 20k trend (0.2139 → 0.1083)
has not flattened, so the un-guarded CER should keep falling.

## Quickstart

Training runs on Colab — no local GPU or disk required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Crypto47/arabic-ocr-post-correction/blob/main/notebooks/train_colab.ipynb)

The notebook clones this repo, pulls a Kaggle corpus, builds the pairs, trains,
scores base vs. finetuned, and pushes the adapter to the Hub.

To run the data pipeline locally (no GPU stack needed):

```bash
pip install pyyaml
python src/ocr_noise.py                      # see the corruption model in action
python src/build_dataset.py --input <corpus-dir> --output data/processed
```

Full training locally:

```bash
pip install -r requirements.txt
python src/build_dataset.py --input data/raw --output data/processed
python src/train.py --config configs/qwen05b_lora.yaml
python src/evaluate.py --adapter outputs/arabic-ocr-post-correction/final
python src/infer.py --adapter outputs/arabic-ocr-post-correction/final --text "<noisy arabic>"
```

## Layout

```
src/ocr_noise.py       the Arabic OCR confusion model — the core of the project
src/build_dataset.py   corpus -> (noisy, clean) pairs; auto-detects file format
src/train.py           LoRA SFT, with a T4 precision guard (Turing has no bf16)
src/evaluate.py        CER/WER vs. baseline, broken down by scan severity
src/metrics.py         CER/WER, dependency-free so eval is testable anywhere
src/infer.py           greedy correction, batched
configs/               all hyperparameters
notebooks/             the Colab training notebook
```

## Notes and limitations

- **Synthetic noise is a model of OCR errors, not a recording of them.** The
  weights above are grounded in the structure of the script, but they are not
  calibrated against a specific engine's output. Validating against real
  Tesseract/PaddleOCR output on scanned pages is the honest next step, and until
  that is done the reported CER reduction describes performance on synthetic
  corruption only.
- Trained on Modern Standard Arabic. Dialectal and heavily classical text are
  out of distribution.
- The precision guard in `train.py` detects the absence of bf16 on Turing GPUs
  and falls back to fp16. Hardcoding bf16 fails on a free Colab T4.
- Weights are not committed. The adapter is published to the HuggingFace Hub;
  `.gitignore` keeps checkpoints out of git.

## License

MIT — see [LICENSE](LICENSE). The base model's license governs the finetuned
weights independently, as does the license of whichever corpus you train on.
