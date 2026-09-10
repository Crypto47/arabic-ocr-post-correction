# Arabic OCR Post-Correction

A 0.5B-parameter Arabic small language model finetuned to repair OCR output —
the step between "we scanned the archive" and "the archive is searchable."

```
raw OCR   العلم نور يضنء طزيق الإنسان في الحتياة، وآلجهل ظلام دام
corrected العلم نور يضيء طريق الإنسان في الحياة، والجهل ظلام دامس
```

**Model:** https://huggingface.co/Sheeda/arabic-ocr-post-correction-0.5b

> **Status: working.** Trained on 57k pairs and scored on two real corpora.
> Cuts word error rate by 50% while improving character accuracy. All numbers
> are measured, not estimated — but the corruption is synthetic, so they
> describe performance against a *model* of OCR error rather than a recording
> of one. See [Notes](#notes-and-limitations).

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

Measured on an RTX 5070 Ti (12GB, sm_120), Qwen2.5-0.5B + LoRA, greedy decoding,
200 held-out AraSum segments. Corruption is synthetic — see
[Notes](#notes-and-limitations).

### Headline

| | CER | WER |
|---|---|---|
| Raw OCR (do nothing) | 0.0808 | 0.4112 |
| Untuned Qwen2.5-0.5B | 1.8123 | 2.2680 |
| **Finetuned (57k pairs)** | **0.0724** | **0.1765** |
| **Finetuned + guardrail** | **0.0671** | **0.2042** |

**Word error rate falls 41.1% → 20.4%, a 50.3% reduction, while character
accuracy simultaneously improves 16.9%.** WER is the metric that matters:
search, indexing and extraction all operate on whole words.

The untuned base model scores 1.81 CER — above 1.0 means its output is not a
damaged version of the truth but unrelated text. It answers the instruction
conversationally instead of restoring. Every gain above is therefore
attributable to the finetune, not to Qwen already knowing Arabic.

### Training scale was the dominant variable

| training pairs | CER | WER | segments made worse |
|---|---|---|---|
| 0 (untuned) | 1.8123 | 2.2680 | 200/200 |
| 2,000 | 0.2139 | 0.3751 | 170/200 |
| 20,000 | 0.1083 | 0.2327 | 115/200 |
| **56,931** | **0.0724** | **0.1765** | **70/200** |

Training loss 1.976 → 1.751 → 1.637; token accuracy 0.667 → 0.700 → 0.717.

### The guardrail

[`apply_guardrail`](src/infer.py) rejects any correction diverging from its
*input* by more than a CER threshold and falls back to the original text.
Divergence is measured against the input, never the reference, so it runs in
production. The threshold is an operating dial:

| policy | CER | WER | corrections kept |
|---|---|---|---|
| raw OCR | 0.0808 | 0.4112 | — |
| accept everything | 0.0724 | **0.1765** | 200/200 |
| drift ≤ 0.15 | **0.0667** | 0.2676 | 133/200 |
| **drift ≤ 0.20** | **0.0671** | **0.2042** | 180/200 |
| drift ≤ 0.30 | 0.0695 | 0.1781 | 195/200 |

`drift ≤ 0.20` is the recommended default: near-best on both metrics at once.
Tighten it for archives where nothing may be made worse; loosen it where
findability outweighs character fidelity.

### Where the gain lands

| severity | n | baseline CER | model CER |
|---|---|---|---|
| light (<8%) | 62 | 0.0438 | **0.0404** |
| medium (8–14%) | 82 | 0.0813 | **0.0713** |
| heavy (>14%) | 56 | 0.1211 | **0.1092** |

It improves every severity band, including the badly degraded scans that
actually need correcting. On the 123/200 segments it helps, CER drops 0.0816 →
0.0377; 20/200 come back exactly correct.

```
noisy  العراـق، لكْب احتمال ظهور مطالب جدد ة من قبل البرلمان ئلعراقي".
fixed  العراق، لكن احتمال ظهور مطالب جديدة من قبل البرلمان العراقي".

noisy  وأضا ف أن "بريطايا يمكن أن تؤثر على الاتحاد الأوروبى عبر الـعمل هع شركائها".
fixed  وأضاف أن "بريطانيا يمكن أن تؤثر على الاتحاد الأوروبي عبر العمل مع شركائها".
```

### Reproduced on a second corpus

Arabic BERT Corpus (56,946 pairs, baseline CER 0.0838 / WER 0.3900) shows the
same trajectory at 2,000 pairs: untuned 2.2092 → tuned 0.1956. The behaviour is
a property of training scale, not of one dataset.

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
- Weights are not committed. The adapter is published to the HuggingFace Hub
  at [Sheeda/arabic-ocr-post-correction-0.5b](https://huggingface.co/Sheeda/arabic-ocr-post-correction-0.5b);
  `.gitignore` keeps checkpoints out of git.

## License

MIT — see [LICENSE](LICENSE). The base model's license governs the finetuned
weights independently, as does the license of whichever corpus you train on.
