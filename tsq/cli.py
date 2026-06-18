"""
Command-line entrypoint for TSQ experiments.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .evals.harness import compare_baselines, repair_eval_tasks
from .receipts.store import ReceiptStore
from .reports import (
    build_eval_report,
    build_generation_report,
    build_repair_eval_report,
    write_json_report,
)
from .runtime.generation_loop import run_tsq_generation
from .runtime.model_runner import MockModelRunner, RepairAwareMockRunner, TransformersModelRunner


class CliError(RuntimeError):
    pass


def _add_common_backend_args(parser: argparse.ArgumentParser, default_backend: str = "mock") -> None:
    parser.add_argument("--backend", choices=["mock", "repair-mock", "transformers"], default=default_backend)
    parser.add_argument("--model-id")
    parser.add_argument("--q4-model")
    parser.add_argument("--q8-model")
    parser.add_argument("--fp16-model")


def _build_backend(args: argparse.Namespace):
    if args.backend == "mock":
        return MockModelRunner()
    if args.backend == "repair-mock":
        return RepairAwareMockRunner()
    if args.backend != "transformers":
        raise CliError(f"unsupported backend: {args.backend}")
    if not args.model_id:
        raise CliError("--model-id is required when --backend transformers")
    precision_models = {}
    if args.q4_model:
        precision_models["Q4"] = args.q4_model
    if args.q8_model:
        precision_models["Q8"] = args.q8_model
    if args.fp16_model:
        precision_models["FP16"] = args.fp16_model
    try:
        return TransformersModelRunner.from_pretrained(
            args.model_id,
            precision_models=precision_models or None,
        )
    except ImportError as exc:
        raise CliError(f"{exc}\nInstall optional dependencies with: pip install -e '.[transformers]'") from exc


def _constraints(args: argparse.Namespace) -> list[str]:
    return list(args.constraint or [])


def _cmd_generate(args: argparse.Namespace) -> int:
    runner = _build_backend(args)
    constraints = _constraints(args)
    store = ReceiptStore(args.receipts) if args.receipts else None
    result = run_tsq_generation(
        prompt=args.prompt,
        constraints=constraints,
        max_new_tokens=args.max_new_tokens,
        model=runner,
        receipt_store=store,
    )
    report = build_generation_report(
        result=result,
        backend=args.backend,
        prompt=args.prompt,
        constraints=constraints,
        model_name=getattr(runner, "name", runner.__class__.__name__),
    )
    write_json_report(args.report, report)
    print(f"wrote report: {args.report}")
    if args.receipts:
        print(f"wrote receipts: {args.receipts}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    runner = _build_backend(args)
    constraints = _constraints(args)
    results = compare_baselines(
        prompt=args.prompt,
        constraints=constraints,
        max_new_tokens=args.max_new_tokens,
        model=runner,
    )
    report = build_eval_report(
        results=results,
        backend=args.backend,
        prompt=args.prompt,
        constraints=constraints,
        model_name=getattr(runner, "name", runner.__class__.__name__),
    )
    write_json_report(args.report, report)
    print(f"wrote report: {args.report}")
    return 0


def _cmd_repair_eval(args: argparse.Namespace) -> int:
    task_results = []
    for task in repair_eval_tasks():
        runner = _build_backend(args)
        constraints = list(task["constraints"])
        result = run_tsq_generation(
            prompt=str(task["prompt"]),
            constraints=constraints,
            max_new_tokens=int(task["max_new_tokens"]),
            model=runner,
        )
        task_results.append(
            {
                "name": task["name"],
                "prompt": task["prompt"],
                "constraints": constraints,
                "output": result["output"],
                "stats": result["stats"],
                "verification": result["verification"],
                "original_verification": result["original_verification"],
                "final_verification": result["final_verification"],
                "cognitive_receipts": result["cognitive_receipts"],
                "compute_receipts": result["compute_receipts"],
                "tension_samples": result["tension_samples"],
            }
        )
    report = build_repair_eval_report(
        task_results=task_results,
        backend=args.backend,
        model_name=args.model_id or args.backend,
    )
    write_json_report(args.report, report)
    print(f"wrote report: {args.report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tsq.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="run a single TSQ generation")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--constraint", action="append", default=[])
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--receipts")
    generate.add_argument("--report", required=True)
    _add_common_backend_args(generate)
    generate.set_defaults(func=_cmd_generate)

    eval_parser = subparsers.add_parser("eval", help="compare always_Q4, always_Q8, and TSQ_dynamic")
    eval_parser.add_argument("--prompt", required=True)
    eval_parser.add_argument("--constraint", action="append", default=[])
    eval_parser.add_argument("--max-new-tokens", type=int, default=64)
    eval_parser.add_argument("--report", required=True)
    _add_common_backend_args(eval_parser)
    eval_parser.set_defaults(func=_cmd_eval)

    repair_eval = subparsers.add_parser("repair-eval", help="run built-in repair eval tasks")
    repair_eval.add_argument("--report", required=True)
    _add_common_backend_args(repair_eval, default_backend="repair-mock")
    repair_eval.set_defaults(func=_cmd_repair_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        print(f"tsq: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
