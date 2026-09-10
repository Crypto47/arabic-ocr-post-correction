"""LoRA finetune of a small Arabic model for OCR post-correction.

Usage:
    python src/train.py --config configs/qwen05b_lora.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def precision_flags() -> dict[str, bool]:
    """Colab's free tier is a T4 (Turing, sm_75) which has no bf16 support.
    Picking bf16 there fails at runtime, so detect rather than hardcode."""
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    if torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    print("bf16 unsupported on this GPU — falling back to fp16.")
    return {"bf16": False, "fp16": True}


def build_model_and_tokenizer(cfg: dict, dtype: torch.dtype):
    mc = cfg["model"]
    quant = None
    if mc.get("load_in_4bit"):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        mc["base_model"], trust_remote_code=mc["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # generation flips this; see infer.py

    model = AutoModelForCausalLM.from_pretrained(
        mc["base_model"],
        quantization_config=quant,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=mc["trust_remote_code"],
        attn_implementation=mc.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    return model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/qwen05b_lora.yaml"))
    ap.add_argument("--train-file", help="override data.train_file")
    ap.add_argument("--output-dir", help="override training.output_dir")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg, lora_cfg, train_cfg = cfg["data"], cfg["lora"], dict(cfg["training"])
    if args.train_file:
        data_cfg["train_file"] = args.train_file
    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir

    prec = precision_flags()
    dtype = torch.bfloat16 if prec["bf16"] else torch.float16

    data_files = {"train": data_cfg["train_file"]}
    eval_file = data_cfg.get("eval_file")
    if eval_file and Path(eval_file).exists():
        data_files["eval"] = eval_file
    else:
        train_cfg["eval_strategy"] = "no"
    dataset = load_dataset("json", data_files=data_files)

    model, tokenizer = build_model_and_tokenizer(cfg, dtype)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        max_length=data_cfg["max_seq_length"],
        packing=False,  # never blend two documents: the target must map 1:1 to its input
        **prec,
        **train_cfg,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    out = Path(train_cfg["output_dir"]) / "final"
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nAdapter saved to {out}")
    print(f"Next: python src/evaluate.py --adapter {out} --eval-file {eval_file}")


if __name__ == "__main__":
    main()
