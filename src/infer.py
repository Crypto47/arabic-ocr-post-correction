"""Run OCR post-correction with the finetuned adapter."""

from __future__ import annotations

import argparse
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import cer

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

    param = next(model.parameters())
    print(f"loaded {base} on {param.device} ({param.dtype})")
    if param.device.type == "cpu":
        print("  WARNING: no GPU — generation will be roughly two orders of "
              "magnitude slower.\n"
              "  In Colab: Runtime > Change runtime type > T4 GPU.")
    return model.eval(), tokenizer


def apply_guardrail(sources: list[str], predictions: list[str],
                    max_drift: float) -> tuple[list[str], int]:
    """Discard a correction that rewrote the page instead of repairing it.

    A generative model asked to restore text will sometimes paraphrase or drop
    a clause. Measured on 200 held-out segments, corrections that stay close to
    the input roughly halve CER, while the ones that drift roughly double it —
    so rejecting the drifters recovers most of the gain and none of the harm.

    Drift is measured against the INPUT, never the reference, so this is
    computable at inference time in production.
    """
    kept, out = 0, []
    for src, pred in zip(sources, predictions):
        if pred and cer(src, pred) <= max_drift:
            out.append(pred)
            kept += 1
        else:
            out.append(src)  # keep the original OCR text
    return out, kept


@torch.inference_mode()
def correct(model, tokenizer, texts: list[str], max_new_tokens: int | None = None,
            batch_size: int = 8, hard_cap: int = 384,
            progress: bool = True, max_drift: float | None = None) -> list[str]:
    """Greedy decoding on purpose: this is a restoration task with one right
    answer, so sampling only invents text that was never on the page.

    max_new_tokens defaults to a budget derived from the input length. This
    matters more than it looks: the output of a restoration task is about as
    long as its input, and an *untuned* base model never learns to emit EOS
    here — so a fixed generous budget means every single sequence runs to the
    full allowance. That alone is the difference between minutes and hours on
    the baseline pass.
    """
    results: list[str] = []
    started = time.time()
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
        budget = max_new_tokens or min(
            hard_cap, int(enc["input_ids"].shape[-1] * 1.15) + 16
        )
        out = model.generate(
            **enc, max_new_tokens=budget, do_sample=False,
            num_beams=1, pad_token_id=tokenizer.pad_token_id,
        )
        for i in range(len(batch)):
            gen = out[i][enc["input_ids"].shape[-1]:]
            results.append(tokenizer.decode(gen, skip_special_tokens=True).strip())

        if progress:
            done = len(results)
            elapsed = time.time() - started
            eta = elapsed / done * (len(texts) - done)
            print(f"\r  {done}/{len(texts)} segments | cap {budget} tok "
                  f"| {elapsed:.0f}s elapsed | ~{eta:.0f}s left",
                  end="", flush=True)
    if progress:
        print()
    if max_drift is not None:
        results, kept = apply_guardrail(texts, results, max_drift)
        print(f"  guardrail (drift <= {max_drift}): kept {kept}/{len(texts)} "
              f"corrections, reverted {len(texts) - kept} to the original")
    return results


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--text", help="noisy OCR text to correct")
    ap.add_argument("--file", help="a file of noisy text, one segment per line")
    ap.add_argument("--max-drift", type=float, default=None,
                    help="reject a correction that diverges from the input by "
                         "more than this CER; 0.10-0.15 is the useful range")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                help="default: derived from input length")
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
                                         max_new_tokens=args.max_new_tokens,
                                         max_drift=args.max_drift)):
        print(f"in : {src}")
        print(f"out: {fixed}\n")


if __name__ == "__main__":
    main()
