"""
Dynamic TSQ generation loop.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Sequence

from ..receipts.schema import (
    CognitiveReceipt,
    ComputeReceipt,
    make_cognitive_receipt,
    make_compute_receipt,
)
from ..receipts.store import ReceiptStore
from ..tension.scanner import scan_recent_window
from ..verifier.base import Verifier
from .costing import (
    DEFAULT_COST_MODEL_NAME,
    build_precision_histogram,
    empty_precision_counts,
    estimate_cost_units,
    increment_precision,
    precision_count_stats,
)
from .model_runner import MockModelRunner, ModelRunner
from .precision_router import PrecisionRouter


def run_tsq_generation(
    prompt: str,
    constraints: Sequence[str] | None = None,
    max_new_tokens: int = 64,
    model: ModelRunner | None = None,
    receipt_store: ReceiptStore | None = None,
    repair_on_failure: bool = True,
    repair_max_tokens: int = 8,
) -> Dict[str, Any]:
    """
    Run cheap-first generation with tension scanning, precision routing, and verification.
    """
    start = perf_counter()
    constraints = list(constraints or [])
    runner = model or MockModelRunner()
    router = PrecisionRouter()
    verifier = Verifier()
    generated: List[str] = []
    tension_samples: List[Dict[str, Any]] = []
    cognitive_receipts: List[CognitiveReceipt] = []
    compute_receipts: List[ComputeReceipt] = []
    precision_counts = empty_precision_counts()
    repair_precision_counts = empty_precision_counts()
    repair_attempted = False
    repair_succeeded = False
    repair_tokens_generated = 0

    for index in range(max(0, max_new_tokens)):
        recent_text = " ".join([prompt, *generated])[-500:]
        step_result = runner.step(
            precision=router.current_precision,
            prompt=prompt,
            constraints=constraints,
            generated=generated,
        )
        tension = scan_recent_window(
            recent_text,
            logits=step_result,
            previous_failure=router.previous_failure,
        )
        precision, compute_receipt = router.decide(tension)
        if compute_receipt is not None:
            compute_receipt.metadata.update(
                {
                    "step": index,
                    "tension_components": dict(tension["components"]),
                    "chosen_precision": precision,
                    "previous_precision": step_result.precision,
                    "target": compute_receipt.target,
                }
            )
            compute_receipts.append(compute_receipt)
            if receipt_store is not None:
                receipt_store.append(compute_receipt)
        generated.append(step_result.token_text)
        increment_precision(precision_counts, step_result.precision)
        tension_samples.append(
            {
                "step": index,
                "precision": precision,
                "step_precision": step_result.precision,
                "tension": tension["tension"],
                "components": tension["components"],
            }
        )

    output = " ".join([prompt.strip(), *generated]).strip()
    original_verification = verifier.verify(prompt=prompt, output=output, constraints=constraints)
    final_verification = original_verification

    if not original_verification.passed:
        router.mark_failure()
        tension = scan_recent_window(output[-500:], logits=None, previous_failure=True)
        precision, compute_receipt = router.decide(tension, force_escalate=True)
        if compute_receipt is None:
            compute_receipt = make_compute_receipt(
                reason="verification_failed_repair",
                target="final_generation_output",
                from_precision=4,
                to_precision=8,
                tension=tension["tension"],
                metadata={
                    "tension_components": dict(tension["components"]),
                    "verifier_failures": list(original_verification.failures),
                    "chosen_precision": precision,
                    "previous_precision": "Q4",
                    "target": "final_generation_output",
                    "repair_attempted": bool(repair_on_failure and repair_max_tokens > 0),
                },
            )
        else:
            compute_receipt.reason = "verification_failed_repair"
            compute_receipt.target = "final_generation_output"
            compute_receipt.metadata.update(
                {
                    "tension_components": dict(tension["components"]),
                    "verifier_failures": list(original_verification.failures),
                    "chosen_precision": precision,
                    "previous_precision": compute_receipt.metadata.get("previous_precision", "Q4"),
                    "target": "final_generation_output",
                    "repair_attempted": bool(repair_on_failure and repair_max_tokens > 0),
                }
            )
        compute_receipts.append(compute_receipt)
        tension_samples.append(
            {
                "step": len(generated),
                "precision": precision,
                "tension": tension["tension"],
                "components": tension["components"],
                "verification_failed": True,
            }
        )

        if repair_on_failure and repair_max_tokens > 0:
            repair_attempted = True
            for repair_index in range(repair_max_tokens):
                step_result = runner.step(
                    precision=precision,
                    prompt=prompt,
                    constraints=constraints,
                    generated=generated,
                    repair_mode=True,
                    verification_failures=list(original_verification.failures),
                    current_output=output,
                )
                generated.append(step_result.token_text)
                increment_precision(precision_counts, step_result.precision)
                increment_precision(repair_precision_counts, step_result.precision)
                repair_tokens_generated += 1
                output = " ".join([prompt.strip(), *generated]).strip()
                tension_samples.append(
                    {
                        "step": len(generated) - 1,
                        "precision": precision,
                        "step_precision": step_result.precision,
                        "tension": step_result.entropy_proxy,
                        "components": {"repair_mode": True},
                        "repair": True,
                        "metadata": dict(step_result.metadata),
                    }
                )
                final_verification = verifier.verify(
                    prompt=prompt,
                    output=output,
                    constraints=constraints,
                )
                if final_verification.passed:
                    repair_succeeded = True
                    break
        else:
            final_verification = original_verification

        compute_receipt.metadata["repair_succeeded"] = repair_succeeded
        if receipt_store is not None:
            receipt_store.append(compute_receipt)

    precision_histogram = build_precision_histogram(
        precision_counts,
        repair_tokens_by_precision=repair_precision_counts,
        escalation_count=len(compute_receipts),
        compute_receipt_count=len(compute_receipts),
    )
    if final_verification.passed:
        cognitive_receipt = make_cognitive_receipt(
            prompt=prompt,
            output=output,
            constraints=constraints,
            verified_by=final_verification.details.get("checkers", []),
            verifier_result=final_verification.to_dict(),
            metadata={
                "model": getattr(runner, "name", runner.__class__.__name__),
                "verifier_checkers": final_verification.details.get("checkers", []),
                "repair_attempted": repair_attempted,
                "repair_succeeded": repair_succeeded,
                "precision_histogram": precision_histogram,
                "routing_summary": {
                    "escalations": len(compute_receipts),
                    "estimated_cost_units": estimate_cost_units(precision_counts),
                    "cost_model_used": DEFAULT_COST_MODEL_NAME,
                },
            },
        )
        cognitive_receipts.append(cognitive_receipt)
        if receipt_store is not None:
            receipt_store.append(cognitive_receipt)

    latency = perf_counter() - start
    cost_stats = {
        **precision_count_stats(precision_counts),
        "total_tokens_generated": len(generated),
        "estimated_cost_units": estimate_cost_units(precision_counts),
        "cost_model_used": DEFAULT_COST_MODEL_NAME,
        "precision_histogram": precision_histogram,
    }
    return {
        "output": output,
        "stats": {
            "latency": latency,
            "verifier_pass": final_verification.passed,
            "original_verifier_pass": original_verification.passed,
            "final_verifier_pass": final_verification.passed,
            "escalations": len(compute_receipts),
            "receipts": len(cognitive_receipts) + len(compute_receipts),
            "repair_attempted": repair_attempted,
            "repair_succeeded": repair_succeeded,
            "repair_tokens_generated": repair_tokens_generated,
            "output_length": len(output),
            "model": getattr(runner, "name", runner.__class__.__name__),
            "tokens_generated": len(generated),
            **cost_stats,
        },
        "cognitive_receipts": cognitive_receipts,
        "compute_receipts": compute_receipts,
        "tension_samples": tension_samples,
        "verification": final_verification,
        "original_verification": original_verification,
        "final_verification": final_verification,
    }
