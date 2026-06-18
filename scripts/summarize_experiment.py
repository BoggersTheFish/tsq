#!/usr/bin/env python3
"""
Summarize a TSQ adapter experiment manifest and optional eval report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from tsq.training.experiment import read_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize TSQ v0.9 experiment outputs")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval-report")
    parser.add_argument("--markdown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = read_manifest(args.manifest)
    eval_report = _read_optional(args.eval_report or manifest.get("eval_report_path"))
    summary = build_summary(manifest, eval_report)
    print_summary(summary)
    if args.markdown:
        write_markdown(args.markdown, summary)
        print(f"wrote markdown: {args.markdown}")
    return 0


def build_summary(manifest: Dict[str, Any], eval_report: Dict[str, Any] | None = None) -> Dict[str, Any]:
    base = eval_report.get("base", {}) if eval_report else {}
    adapter = eval_report.get("adapter", {}) if eval_report else {}
    base_aggregate = base.get("aggregate", {})
    adapter_aggregate = adapter.get("aggregate", {}) if adapter else {}
    base_passes = base_aggregate.get("dynamic_passes")
    adapter_passes = adapter_aggregate.get("dynamic_passes")
    base_cost = base_aggregate.get("mean_dynamic_estimated_cost")
    adapter_cost = adapter_aggregate.get("mean_dynamic_estimated_cost")
    return {
        "model_id": manifest.get("model_id"),
        "adapter_dir": manifest.get("adapter_dir"),
        "training_status": manifest.get("training_status"),
        "eval_status": manifest.get("eval_status"),
        "base_dynamic_passes": base_passes,
        "adapter_dynamic_passes": adapter_passes,
        "base_mean_dynamic_cost": base_cost,
        "adapter_mean_dynamic_cost": adapter_cost,
        "adapter_improved_pass_rate": _improved(adapter_passes, base_passes),
        "adapter_improved_cost": _lower(adapter_cost, base_cost),
        "smoke_train": bool(manifest.get("smoke_train")),
        "notes": manifest.get("notes", []),
        "limitation": (
            "Smoke train and seed datasets are wiring proof only; meaningful quality "
            "requires larger data and more steps."
        ),
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"model_id: {summary['model_id']}")
    print(f"adapter_dir: {summary['adapter_dir']}")
    print(f"training_status: {summary['training_status']}")
    print(f"eval_status: {summary['eval_status']}")
    print(f"base_dynamic_passes: {summary['base_dynamic_passes']}")
    print(f"adapter_dynamic_passes: {summary['adapter_dynamic_passes']}")
    print(f"base_mean_dynamic_cost: {summary['base_mean_dynamic_cost']}")
    print(f"adapter_mean_dynamic_cost: {summary['adapter_mean_dynamic_cost']}")
    print(f"adapter_improved_pass_rate: {summary['adapter_improved_pass_rate']}")
    print(f"adapter_improved_cost: {summary['adapter_improved_cost']}")
    print(f"limitation: {summary['limitation']}")


def write_markdown(path: str | Path, summary: Dict[str, Any]) -> Path:
    markdown_path = Path(path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        "# TSQ v0.9 Experiment Summary",
        "",
        f"- Model: `{summary['model_id']}`",
        f"- Adapter: `{summary['adapter_dir']}`",
        f"- Training status: `{summary['training_status']}`",
        f"- Eval status: `{summary['eval_status']}`",
        f"- Base dynamic passes: `{summary['base_dynamic_passes']}`",
        f"- Adapter dynamic passes: `{summary['adapter_dynamic_passes']}`",
        f"- Base mean dynamic cost: `{summary['base_mean_dynamic_cost']}`",
        f"- Adapter mean dynamic cost: `{summary['adapter_mean_dynamic_cost']}`",
        f"- Adapter improved pass rate: `{summary['adapter_improved_pass_rate']}`",
        f"- Adapter improved cost: `{summary['adapter_improved_cost']}`",
        "",
        f"Limitation: {summary['limitation']}",
        "",
    ]
    notes = summary.get("notes") or []
    if notes:
        content.append("Notes:")
        content.extend(f"- {note}" for note in notes)
        content.append("")
    markdown_path.write_text("\n".join(content), encoding="utf-8")
    return markdown_path


def _read_optional(path: str | None) -> Dict[str, Any] | None:
    if not path:
        return None
    report_path = Path(path)
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def _improved(candidate: Any, baseline: Any) -> bool | None:
    if candidate is None or baseline is None:
        return None
    return candidate > baseline


def _lower(candidate: Any, baseline: Any) -> bool | None:
    if candidate is None or baseline is None:
        return None
    return candidate < baseline


if __name__ == "__main__":
    raise SystemExit(main())
