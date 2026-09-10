# Arabic OCR Post-Correction

A 0.5B-parameter Arabic small language model finetuned to repair OCR output —
the step between "we scanned the archive" and "the archive is searchable."

```
raw OCR   العلم نور يضنء طزيق الإنسان في الحتياة، وآلجهل ظلام دام
corrected العلم نور يضيء طريق الإنسان في الحياة، والجهل ظلام دامس
```

> **Status: pipeline complete, results pending.** Every stage runs end to end on
> sample data. The numbers in [Results](#results) are filled in after the first
> full training run — they are deliberately left blank rather than estimated.

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

Scored on held-out synthetic pairs. `--limit 200`, greedy decoding.

| | CER | WER |
|---|---|---|
| Raw OCR (baseline) | _pending_ | _pending_ |
| Base Qwen2.5-0.5B, no finetune | _pending_ | _pending_ |
| **Finetuned** | _pending_ | _pending_ |

**CER reduction vs. raw OCR:** _pending_

By scan severity:

| Severity | n | Baseline CER | Model CER |
|---|---|---|---|
| light (<8%) | | | |
| medium (8–14%) | | | |
| heavy (>14%) | | | |

The untuned base model is scored too, not just the finetune. A finetune that
cannot beat its own base model has demonstrated nothing.

## Quickstart

Training runs on Colab — no local GPU or disk required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Crypto47/arabic-ocr-correct/blob/main/notebooks/train_colab.ipynb)

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
python src/evaluate.py --adapter outputs/arabic-ocr-correct/final
python src/infer.py --adapter outputs/arabic-ocr-correct/final --text "<noisy arabic>"
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
