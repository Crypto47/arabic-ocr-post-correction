"""Run OCR post-correction with the finetuned adapter."""

from __future__ import annotations

import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION = "صحّح أخطاء المسح الضوئي في النص التالي وأعد كتابته بشكل صحيح:"


def load(base: str, adapter: str | None):
    tokenizer = AutoTokenizer.from_pretrained(adapter or base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation

    dtype = torch.bfloat16 if (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), tokenizer


@torch.inference_mode()
def correct(model, tokenizer, texts: list[str], max_new_tokens: int = 512,
            batch_size: int = 8) -> list[str]:
    """Greedy decoding on purpose: this is a restoration task with one right
    answer, so sampling only invents text that was never on the page."""
    results: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{INSTRUCTION}\n\n{t}"}],
                tokenize=False, add_generation_prompt=True,
            )
            for t in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=1024).to(model.device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1, pad_token_id=tokenizer.pad_token_id,
        )
        for i in range(len(batch)):
            gen = out[i][enc["input_ids"].shape[-1]:]
            results.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return results


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--text", help="noisy OCR text to correct")
    ap.add_argument("--file", help="a file of noisy text, one segment per line")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    if not args.text and not args.file:
        ap.error("pass --text or --file")

    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            texts = [ln.strip() for ln in fh if ln.strip()]
    else:
        texts = [args.text]

    model, tokenizer = load(args.base, args.adapter)
    for src, fixed in zip(texts, correct(model, tokenizer, texts,
                                         max_new_tokens=args.max_new_tokens)):
        print(f"in : {src}")
        print(f"out: {fixed}\n")


if __name__ == "__main__":
    main()
