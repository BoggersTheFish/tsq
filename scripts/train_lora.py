#!/usr/bin/env python3
"""
Minimal LoRA training scaffold for TSQ JSONL datasets.

Default project tests use --dry-run and do not require heavy ML dependencies.
For real training install: torch, transformers, datasets, peft, accelerate.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

from tsq.training.validate_dataset import dataset_summary, print_summary, validate_dataset


REQUIRED_TRAINING_DEPS = ("torch", "transformers", "datasets", "peft", "accelerate")


def missing_training_deps() -> list[str]:
    return [name for name in REQUIRED_TRAINING_DEPS if importlib.util.find_spec(name) is None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TSQ LoRA training scaffold")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    train_rows = validate_dataset(args.train_jsonl)
    eval_rows = validate_dataset(args.eval_jsonl)
    print("train dataset:")
    print_summary(dataset_summary(args.train_jsonl))
    print("eval dataset:")
    print_summary(dataset_summary(args.eval_jsonl))

    if args.dry_run:
        print(
            "dry-run ok: datasets validated; no model loaded and no training performed "
            f"for {args.model_id}"
        )
        return 0

    missing = missing_training_deps()
    if missing:
        raise RuntimeError(
            "Missing optional training dependencies: "
            + ", ".join(missing)
            + ". Install torch, transformers, datasets, peft, and accelerate to train."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "training scaffold reached dependency-ready path; actual trainer wiring is reserved "
        "for the next wave."
    )
    print(
        f"model={args.model_id} train_examples={len(train_rows)} eval_examples={len(eval_rows)} "
        f"max_steps={args.max_steps} learning_rate={args.learning_rate} lora_r={args.lora_r} "
        f"output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"train_lora.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
