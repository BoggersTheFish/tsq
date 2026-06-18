#!/usr/bin/env python3
"""
Evaluate a base model and optional LoRA adapter through TSQ.

Default CI uses --dry-run/--check-only and does not import heavy ML dependencies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

from tsq import __version__
from tsq.evals.harness import eval_suite_tasks, run_eval_suite
from tsq.reports import to_jsonable
from tsq.runtime.model_runner import TransformersModelRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a TSQ LoRA adapter")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-tasks", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run or args.check_only:
        report = {
            "mode": "lora-eval-check",
            "model_id": args.model_id,
            "adapter_dir": args.adapter_dir,
            "max_new_tokens": args.max_new_tokens,
            "max_tasks": args.max_tasks,
            "tsq_version": __version__,
            "status": "check-only ok; no model loaded",
        }
        _write_report(args.report, report)
        print(f"check-only ok: wrote report {args.report}")
        return 0

    _assert_eval_dependencies(args.adapter_dir)
    tasks = _limited_tasks(args.max_tasks, args.max_new_tokens)

    base_result = run_eval_suite(
        backend_factory=lambda: TransformersModelRunner.from_pretrained(args.model_id),
        tasks=tasks,
    )
    adapter_result = None
    if args.adapter_dir:
        adapter_result = run_eval_suite(
            backend_factory=lambda: TransformersModelRunner.from_pretrained(
                args.model_id,
                adapter_dir=args.adapter_dir,
            ),
            tasks=tasks,
        )
    report = {
        "mode": "lora-eval",
        "model_id": args.model_id,
        "adapter_dir": args.adapter_dir,
        "max_new_tokens": args.max_new_tokens,
        "max_tasks": len(tasks),
        "base": base_result,
        "adapter": adapter_result,
        "summary": {
            "base_dynamic_passes": base_result["aggregate"]["dynamic_passes"],
            "adapter_dynamic_passes": (
                adapter_result["aggregate"]["dynamic_passes"] if adapter_result else None
            ),
            "base_mean_dynamic_estimated_cost": base_result["aggregate"][
                "mean_dynamic_estimated_cost"
            ],
            "adapter_mean_dynamic_estimated_cost": (
                adapter_result["aggregate"]["mean_dynamic_estimated_cost"]
                if adapter_result
                else None
            ),
        },
        "tsq_version": __version__,
    }
    _write_report(args.report, report)
    print(f"wrote report: {args.report}")
    return 0


def _assert_eval_dependencies(adapter_dir: str | None) -> None:
    missing = [
        name
        for name in ("torch", "transformers")
        if importlib.util.find_spec(name) is None
    ]
    if adapter_dir and importlib.util.find_spec("peft") is None:
        missing.append("peft")
    if missing:
        raise RuntimeError(
            "Missing optional eval dependencies: "
            + ", ".join(sorted(set(missing)))
            + ". Install torch, transformers, and peft for adapter eval."
        )


def _limited_tasks(max_tasks: int, max_new_tokens: int) -> list[Dict[str, Any]]:
    tasks = []
    for task in eval_suite_tasks()[: max(1, max_tasks)]:
        copied = dict(task)
        copied["max_new_tokens"] = min(int(copied["max_new_tokens"]), max_new_tokens)
        tasks.append(copied)
    return tasks


def _write_report(path: str | Path, report: Dict[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(to_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"eval_lora.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
