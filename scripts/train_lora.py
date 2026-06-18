#!/usr/bin/env python3
"""
LoRA/QLoRA training entrypoint for TSQ JSONL datasets.

Default project tests use --dry-run and do not require heavy ML dependencies.
For real training install: torch, transformers, datasets, peft, accelerate.
Install bitsandbytes only when using --load-in-4bit or --load-in-8bit.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tsq import __version__
from tsq.training.validate_dataset import dataset_summary, load_jsonl, print_summary, validate_dataset


REQUIRED_TRAINING_DEPS = ("torch", "transformers", "datasets", "peft", "accelerate")

SUPERVISED_SYSTEM = (
    "You are a TSQ-aware assistant. Follow explicit constraints and preserve "
    "verifier-critical details."
)
REPAIR_SYSTEM = "You are a TSQ repair model. Repair failed outputs using verifier feedback."


def missing_training_deps() -> list[str]:
    return [name for name in REQUIRED_TRAINING_DEPS if importlib.util.find_spec(name) is None]


def format_supervised_example(row: Dict[str, Any]) -> str:
    return (
        f"<System>\n{SUPERVISED_SYSTEM}\n\n"
        f"<User>\n{row['input_text']}\n\n"
        f"<Assistant>\n{row['target_text']}"
    )


def format_repair_example(row: Dict[str, Any]) -> str:
    return (
        f"<System>\n{REPAIR_SYSTEM}\n\n"
        f"<User>\n{row['repair_prompt']}\n\n"
        f"<Assistant>\n{row['repair_target']}"
    )


def format_training_row(row: Dict[str, Any]) -> str:
    if "repair_prompt" in row:
        return format_repair_example(row)
    return format_supervised_example(row)


def load_formatted_examples(
    train_jsonl: str | Path,
    training_mode: str = "supervised",
    repair_jsonl: str | Path | None = None,
    smoke_train: bool = False,
) -> List[Dict[str, str]]:
    rows = [{"text": format_training_row(row)} for row in load_jsonl(train_jsonl)]
    if training_mode == "repair":
        if repair_jsonl is None:
            rows = [{"text": format_training_row(row)} for row in load_jsonl(train_jsonl)]
        else:
            rows = [{"text": format_training_row(row)} for row in load_jsonl(repair_jsonl)]
    elif training_mode == "mixed":
        if repair_jsonl is None:
            raise ValueError("--repair-jsonl is required when --training-mode mixed")
        rows.extend({"text": format_training_row(row)} for row in load_jsonl(repair_jsonl))
    if smoke_train:
        return rows[: min(len(rows), 2)]
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TSQ LoRA/QLoRA training")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--repair-jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-mode", choices=["supervised", "repair", "mixed"], default="supervised")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--smoke-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    train_rows = validate_dataset(args.train_jsonl)
    eval_rows = validate_dataset(args.eval_jsonl)
    if args.repair_jsonl:
        validate_dataset(args.repair_jsonl)
    if args.training_mode == "mixed" and not args.repair_jsonl:
        raise ValueError("--repair-jsonl is required when --training-mode mixed")

    train_examples = load_formatted_examples(
        args.train_jsonl,
        training_mode=args.training_mode,
        repair_jsonl=args.repair_jsonl,
        smoke_train=args.smoke_train,
    )
    eval_examples = [{"text": format_training_row(row)} for row in eval_rows]
    if args.smoke_train:
        eval_examples = eval_examples[: min(len(eval_examples), 2)]
        if args.max_steps == 50:
            args.max_steps = 2

    print("train dataset:")
    print_summary(dataset_summary(args.train_jsonl))
    print("eval dataset:")
    print_summary(dataset_summary(args.eval_jsonl))
    print(f"formatted_train_examples: {len(train_examples)}")
    print(f"formatted_eval_examples: {len(eval_examples)}")

    if args.dry_run:
        print(
            "dry-run ok: datasets validated and formatted; no model loaded and no training "
            f"performed for {args.model_id}"
        )
        return 0

    _assert_training_dependencies(args)
    _run_training(args, train_examples, eval_examples, train_rows=train_rows, eval_rows=eval_rows)
    return 0


def _assert_training_dependencies(args: argparse.Namespace) -> None:
    missing = missing_training_deps()
    if missing:
        raise RuntimeError(
            "Missing optional training dependencies: "
            + ", ".join(missing)
            + ". Install torch, transformers, datasets, peft, and accelerate to train."
        )
    if (args.load_in_4bit or args.load_in_8bit) and importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "bitsandbytes is required for --load-in-4bit or --load-in-8bit. "
            "Install bitsandbytes or omit quantized loading flags."
        )


def _run_training(
    args: argparse.Namespace,
    train_examples: List[Dict[str, str]],
    eval_examples: List[Dict[str, str]],
    train_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit or args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )

    model_kwargs: Dict[str, Any] = {}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
        model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=max(args.lora_r * 2, 1),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_dataset = Dataset.from_list(train_examples).map(
        lambda batch: _tokenize_batch(batch, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=["text"],
    )
    eval_dataset = Dataset.from_list(eval_examples).map(
        lambda batch: _tokenize_batch(batch, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=["text"],
    )

    training_args_kwargs = {
        "output_dir": str(output_dir),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "report_to": [],
        "bf16": args.bf16,
        "fp16": args.fp16,
        "remove_unused_columns": False,
    }
    eval_strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    training_args_kwargs[eval_strategy_key] = "steps" if eval_examples else "no"
    training_args = TrainingArguments(**training_args_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_examples else None,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _write_metadata(output_dir, args, train_rows, eval_rows, train_examples, eval_examples)
    print(f"saved LoRA adapter and metadata: {output_dir}")


def _tokenize_batch(batch: Dict[str, List[str]], tokenizer: Any, max_seq_length: int) -> Dict[str, Any]:
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )
    tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]
    return tokenized


def _write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    train_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    train_examples: Sequence[Dict[str, str]],
    eval_examples: Sequence[Dict[str, str]],
) -> None:
    metadata = {
        "tsq_version": __version__,
        "model_id": args.model_id,
        "training_mode": args.training_mode,
        "train_jsonl": args.train_jsonl,
        "eval_jsonl": args.eval_jsonl,
        "repair_jsonl": args.repair_jsonl,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "formatted_train_examples": len(train_examples),
        "formatted_eval_examples": len(eval_examples),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_seq_length": args.max_seq_length,
    }
    (output_dir / "tsq_training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"train_lora.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
