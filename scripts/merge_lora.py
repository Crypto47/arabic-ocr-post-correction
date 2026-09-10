"""Merge a LoRA adapter into the base weights for standalone deployment.

Note: merging into a 4-bit quantized base loses accuracy. Load the base in
bf16 here (needs enough RAM/VRAM to hold the full model), not in 4-bit.

Usage:
    python scripts/merge_lora.py --base Qwen/Qwen2.5-7B-Instruct \
                                 --adapter outputs/arabic-sft-qlora/final \
                                 --output outputs/arabic-sft-merged
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--push-to-hub", default=None,
                   help="HF repo id, e.g. your-username/arabic-llm-sft")
    args = p.parse_args()

    print(f"Loading base {args.base} in bf16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging adapter ...")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"Merged model written to {args.output}")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
