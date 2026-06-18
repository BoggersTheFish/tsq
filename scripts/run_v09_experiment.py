#!/usr/bin/env python3
"""
Run or dry-run the TSQ v0.9 adapter experiment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_training_env import dependency_status
from tsq.training.experiment import build_manifest, write_manifest


DATASET_FILES = {
    "supervised_train": "data/generated/tsq_supervised_train.jsonl",
    "supervised_eval": "data/generated/tsq_supervised_eval.jsonl",
    "repair_train": "data/generated/tsq_repair_train.jsonl",
    "preference_pairs": "data/generated/tsq_preference_pairs.jsonl",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TSQ v0.9 adapter experiment")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", default="artifacts/models/tsq-lora-v09")
    parser.add_argument("--report-dir", default="artifacts/reports")
    parser.add_argument("--experiment-dir", default="artifacts/experiments/v09")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--smoke-train", action="store_true")
    parser.add_argument("--training-mode", choices=["supervised", "repair", "mixed"], default="mixed")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands: List[List[str]] = []
    notes: List[str] = []
    dep_status = dependency_status()
    report_dir = Path(args.report_dir)
    experiment_dir = Path(args.experiment_dir)
    manifest_path = experiment_dir / "experiment_manifest.json"
    eval_report_path = report_dir / "tsq_lora_eval_v09.json"
    adapter_dir = Path(args.output_dir)

    build_cmd = [
        sys.executable,
        "-m",
        "tsq.cli",
        "build-dataset",
        "--out-dir",
        "data/generated",
        "--include-example-reports",
        "examples/reports",
    ]
    validate_cmds = [
        [sys.executable, "-m", "tsq.cli", "validate-dataset", "--path", path]
        for key, path in DATASET_FILES.items()
        if key != "preference_pairs"
    ]
    train_cmd = [
        sys.executable,
        "scripts/train_lora.py",
        "--model-id",
        args.model_id,
        "--train-jsonl",
        DATASET_FILES["supervised_train"],
        "--eval-jsonl",
        DATASET_FILES["supervised_eval"],
        "--repair-jsonl",
        DATASET_FILES["repair_train"],
        "--training-mode",
        args.training_mode,
        "--output-dir",
        str(adapter_dir),
        "--max-steps",
        str(args.max_steps),
    ]
    if args.smoke_train:
        train_cmd.append("--smoke-train")
    eval_cmd = [
        sys.executable,
        "scripts/eval_lora.py",
        "--model-id",
        args.model_id,
        "--adapter-dir",
        str(adapter_dir),
        "--report",
        str(eval_report_path),
        "--max-new-tokens",
        "32",
        "--max-tasks",
        "3",
    ]

    commands.extend([build_cmd, *validate_cmds])
    _run_or_print(build_cmd, args.dry_run)
    for cmd in validate_cmds:
        _run_or_print(cmd, args.dry_run)

    training_status = "not_started"
    eval_status = "not_started"
    if args.skip_train:
        training_status = "skipped_by_user"
        notes.append("training skipped by --skip-train")
    elif args.dry_run:
        training_status = "dry_run"
        commands.append([*train_cmd, "--dry-run"])
        _run_or_print([*train_cmd, "--dry-run"], dry_run=False)
    elif not dep_status.get("ready_for_training"):
        training_status = "skipped_missing_dependencies"
        notes.append("training skipped because core optional ML dependencies are missing")
    else:
        commands.append(train_cmd)
        _run_or_print(train_cmd, dry_run=False)
        training_status = "completed"

    if args.skip_eval:
        eval_status = "skipped_by_user"
        notes.append("eval skipped by --skip-eval")
    elif args.dry_run:
        eval_status = "dry_run"
        commands.append([*eval_cmd, "--dry-run"])
        _run_or_print([*eval_cmd, "--dry-run"], dry_run=False)
    elif training_status == "completed":
        commands.append(eval_cmd)
        _run_or_print(eval_cmd, dry_run=False)
        eval_status = "completed"
    elif training_status == "skipped_missing_dependencies":
        eval_status = "skipped_missing_dependencies"
    else:
        eval_status = "skipped_no_completed_adapter"

    manifest = build_manifest(
        model_id=args.model_id,
        adapter_dir=str(adapter_dir),
        dataset_files=DATASET_FILES,
        training_mode=args.training_mode,
        max_steps=args.max_steps,
        smoke_train=args.smoke_train,
        commands_run=commands,
        dependency_status=dep_status,
        training_status=training_status,
        eval_status=eval_status,
        eval_report_path=str(eval_report_path) if eval_status in {"completed", "dry_run"} else None,
        notes=notes,
    )
    write_manifest(manifest_path, manifest)
    print(f"wrote manifest: {manifest_path}")
    return 0


def _run_or_print(cmd: List[str], dry_run: bool) -> None:
    printable = " ".join(cmd)
    if dry_run:
        print(f"dry-run command: {printable}")
        return
    print(f"running: {printable}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
