"""
JSON report helpers for TSQ CLI runs and evals.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from . import __version__


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return str(value)


def build_generation_report(
    result: Dict[str, Any],
    backend: str,
    prompt: str,
    constraints: Sequence[str],
    model_name: str,
) -> Dict[str, Any]:
    return to_jsonable(
        {
            "mode": "generate",
            "backend": backend,
            "model": model_name,
            "prompt": prompt,
            "constraints": list(constraints),
            "output": result["output"],
            "stats": result["stats"],
            "precision_histogram": result["stats"].get("precision_histogram", {}),
            "verification": result["verification"],
            "original_verification": result["original_verification"],
            "final_verification": result["final_verification"],
            "cognitive_receipts": result["cognitive_receipts"],
            "compute_receipts": result["compute_receipts"],
            "tension_samples": result["tension_samples"],
            "tsq_version": __version__,
            "created_at": _now(),
        }
    )


def build_eval_report(
    results: Dict[str, Dict[str, Any]],
    backend: str,
    prompt: str,
    constraints: Sequence[str],
    model_name: str,
) -> Dict[str, Any]:
    dynamic = results["TSQ_dynamic"]["metrics"]
    q4 = results["always_Q4"]["metrics"]
    q8 = results["always_Q8"]["metrics"]
    dynamic_cost = float(dynamic.get("estimated_cost_units", 0.0))
    q8_cost = float(q8.get("estimated_cost_units", 0.0))
    fp16_cost = max(float(dynamic.get("total_tokens_generated", 0)) * 4.0, 1e-9)
    return to_jsonable(
        {
            "mode": "eval",
            "backend": backend,
            "model": model_name,
            "prompt": prompt,
            "constraints": list(constraints),
            "results": results,
            "summary": {
                "dynamic_passed": dynamic["verifier_pass"],
                "dynamic_repaired": dynamic["repair_succeeded"],
                "dynamic_escalations": dynamic["escalations"],
                "q4_passed": q4["verifier_pass"],
                "q8_passed": q8["verifier_pass"],
                "dynamic_receipts": dynamic["receipts"],
                "always_Q4_estimated_cost": q4.get("estimated_cost_units", 0.0),
                "always_Q8_estimated_cost": q8.get("estimated_cost_units", 0.0),
                "TSQ_dynamic_estimated_cost": dynamic_cost,
                "dynamic_vs_q8_cost_ratio": dynamic_cost / max(q8_cost, 1e-9),
                "dynamic_vs_fp16_cost_ratio": dynamic_cost / fp16_cost,
                "precision_histogram": dynamic.get("precision_histogram", {}),
            },
            "tsq_version": __version__,
            "created_at": _now(),
        }
    )


def build_repair_eval_report(
    task_results: Sequence[Dict[str, Any]],
    backend: str,
    model_name: str,
) -> Dict[str, Any]:
    aggregate = {
        "total_tasks": len(task_results),
        "dynamic_passes": sum(1 for item in task_results if item["stats"]["final_verifier_pass"]),
        "repair_attempts": sum(1 for item in task_results if item["stats"]["repair_attempted"]),
        "repair_successes": sum(1 for item in task_results if item["stats"]["repair_succeeded"]),
        "total_compute_receipts": sum(len(item["compute_receipts"]) for item in task_results),
        "total_estimated_cost": sum(item["stats"].get("estimated_cost_units", 0.0) for item in task_results),
        "precision_histogram": _merge_precision_histograms(
            item["stats"].get("precision_histogram", {}) for item in task_results
        ),
    }
    return to_jsonable(
        {
            "mode": "repair-eval",
            "backend": backend,
            "model": model_name,
            "tasks": task_results,
            "aggregate": aggregate,
            "tsq_version": __version__,
            "created_at": _now(),
        }
    )


def build_eval_suite_report(
    suite_result: Dict[str, Any],
    backend: str,
    model_name: str,
) -> Dict[str, Any]:
    return to_jsonable(
        {
            "mode": "eval-suite",
            "backend": backend,
            "model": model_name,
            "tasks": suite_result["tasks"],
            "aggregate": {
                **suite_result["aggregate"],
                "precision_histogram": _merge_precision_histograms(
                    item["results"]["TSQ_dynamic"]["metrics"].get("precision_histogram", {})
                    for item in suite_result["tasks"]
                ),
            },
            "tsq_version": __version__,
            "created_at": _now(),
        }
    )


def write_json_report(path: str | Path, report: Dict[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")


def _merge_precision_histograms(histograms: Any) -> Dict[str, Any]:
    merged = {
        "counts": {"Q4": 0, "Q8": 0, "FP16": 0, "residual_unfolded": 0},
        "repair_tokens": {"Q4": 0, "Q8": 0, "FP16": 0, "residual_unfolded": 0},
        "escalation_count": 0,
        "compute_receipt_count": 0,
    }
    for histogram in histograms:
        if not isinstance(histogram, dict):
            continue
        for precision in merged["counts"]:
            merged["counts"][precision] += int(histogram.get("counts", {}).get(precision, 0))
            merged["repair_tokens"][precision] += int(
                histogram.get("repair_tokens", {}).get(precision, 0)
            )
        merged["escalation_count"] += int(histogram.get("escalation_count", 0))
        merged["compute_receipt_count"] += int(histogram.get("compute_receipt_count", 0))
    return merged
