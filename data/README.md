# Data

Everything here except this file and `sample/` is git-ignored. Corpora are
downloaded at training time (see `notebooks/train_colab.ipynb`), never committed.

```
data/
├── sample/        tracked: a few generated pairs, so the repo is inspectable
├── raw/           ignored: the Kaggle corpus, as downloaded
└── processed/     ignored: output of src/build_dataset.py
    ├── train.jsonl
    └── eval.jsonl
```

## Where the training pairs come from

There is **no public Arabic OCR-correction corpus** — not on Kaggle, not on the
HuggingFace Hub. The pairs are synthesized: clean Arabic text is corrupted by
`src/ocr_noise.py` using a confusion model built around how Arabic script
actually fails under OCR, and the model learns to invert the corruption.

That makes the corpus choice a choice of *clean text only*. Any Arabic corpus
works; domain-matching it to the target documents is what matters.

## Corpus candidates (sizes from the Kaggle API)

| Dataset | Size | License | Note |
|---|---|---|---|
| [arabic-official-documents](https://www.kaggle.com/datasets/azharhasannsaif/arabic-official-documents) | 772 MB | CC-BY-4.0 | closest domain match |
| [Arabic BERT Corpus](https://www.kaggle.com/datasets/abedkhooli/arabic-bert-corpus) | 1.7 GB | Original | clean MSA, high volume |
| [Arabic Wikipedia dump 2021](https://www.kaggle.com/datasets/z3rocool/arabic-wikipedia-dump-2021) | 419 MB | Unknown | broad coverage |
| [ANT Corpus](https://www.kaggle.com/datasets/antcorpus/antcorpus) | 10 MB | Other | small — good for a smoke test |
| [Tashkeela Clean](https://www.kaggle.com/datasets/ahmedmohsen2002/tashkeela-clean-arabic-diacritized-corpus) | 176 MB | **GPL-2** | copyleft; avoid if the weights may go commercial |

Record whichever you used, and its license, before publishing results.

## Format

```json
{"messages": [{"role": "user", "content": "<instruction>\n\n<noisy text>"},
              {"role": "assistant", "content": "<clean text>"}],
 "noise_rate": 0.0931}
```

`noise_rate` is retained so `src/evaluate.py` can break scores down by scan
severity — a model can look strong on average while failing on exactly the
degraded pages that needed correcting.
