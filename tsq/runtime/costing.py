"""
Cost and precision accounting helpers for TSQ routing reports.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping


PRECISION_LEVELS = ("Q4", "Q8", "FP16", "residual_unfolded")

DEFAULT_COST_MODEL: Dict[str, float] = {
    "Q4": 1.0,
    "Q8": 2.0,
    "FP16": 4.0,
    "residual_unfolded": 4.0,
}

DEFAULT_COST_MODEL_NAME = "default_estimated_routing_cost_v0.9"


def empty_precision_counts() -> Dict[str, int]:
    return {precision: 0 for precision in PRECISION_LEVELS}


def precision_count_key(precision: str) -> str:
    return f"tokens_at_{precision}"


def increment_precision(counts: Dict[str, int], precision: str) -> None:
    if precision not in counts:
        counts[precision] = 0
    counts[precision] += 1


def estimate_cost_units(
    counts: Mapping[str, int],
    cost_model: Mapping[str, float] | None = None,
) -> float:
    model = cost_model or DEFAULT_COST_MODEL
    return float(sum(int(counts.get(precision, 0)) * model[precision] for precision in PRECISION_LEVELS))


def precision_count_stats(counts: Mapping[str, int]) -> Dict[str, int]:
    return {precision_count_key(precision): int(counts.get(precision, 0)) for precision in PRECISION_LEVELS}


def build_precision_histogram(
    counts: Mapping[str, int],
    repair_tokens_by_precision: Mapping[str, int] | None = None,
    escalation_count: int = 0,
    compute_receipt_count: int = 0,
) -> Dict[str, object]:
    repair_counts = repair_tokens_by_precision or {}
    return {
        "counts": {precision: int(counts.get(precision, 0)) for precision in PRECISION_LEVELS},
        "repair_tokens": {
            precision: int(repair_counts.get(precision, 0)) for precision in PRECISION_LEVELS
        },
        "escalation_count": int(escalation_count),
        "compute_receipt_count": int(compute_receipt_count),
    }


def fixed_precision_metrics(
    precision: str,
    token_count: int,
    escalation_count: int = 0,
    compute_receipt_count: int = 0,
) -> Dict[str, object]:
    counts = empty_precision_counts()
    if precision not in counts:
        counts[precision] = 0
    counts[precision] = max(0, int(token_count))
    return {
        **precision_count_stats(counts),
        "repair_tokens_generated": 0,
        "total_tokens_generated": max(0, int(token_count)),
        "estimated_cost_units": estimate_cost_units(counts),
        "cost_model_used": DEFAULT_COST_MODEL_NAME,
        "precision_histogram": build_precision_histogram(
            counts,
            escalation_count=escalation_count,
            compute_receipt_count=compute_receipt_count,
        ),
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
