"""
Dynamic TSQ generation loop.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Sequence

from ..receipts.schema import CognitiveReceipt, ComputeReceipt, make_cognitive_receipt
from ..receipts.store import ReceiptStore
from ..tension.scanner import scan_recent_window
from ..verifier.base import Verifier
from .model_runner import MockModelRunner, ModelRunner
from .precision_router import PrecisionRouter


def _constraint_seed_tokens(prompt: str, constraints: Sequence[str]) -> List[str]:
    text = " ".join([prompt, *constraints]).lower()
    tokens: List[str] = []
    if "number" in text or "digit" in text:
        tokens.append("1")
    if "date" in text:
        tokens.append("2026-01-01")
    for raw in constraints:
        for word in raw.replace(".", " ").split():
            cleaned = "".join(ch for ch in word if ch.isalnum() or ch in "_-")
            if len(cleaned) > 4 and cleaned.lower() not in {"include", "exactly", "forbidden"}:
                tokens.append(cleaned)
                break
    return tokens


def _mock_token(index: int, precision: str, seeds: Sequence[str]) -> str:
    if index < len(seeds):
        return seeds[index]
    return f"mock_{precision.lower()}_{index}"


def run_tsq_generation(
    prompt: str,
    constraints: Sequence[str] | None = None,
    max_new_tokens: int = 64,
    model: ModelRunner | None = None,
    receipt_store: ReceiptStore | None = None,
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
    seeds = _constraint_seed_tokens(prompt, constraints)

    for index in range(max(0, max_new_tokens)):
        recent_text = " ".join([prompt, *generated])[-500:]
        logits = runner.step(precision=router.current_precision, prompt=prompt, generated=generated)
        tension = scan_recent_window(recent_text, logits=logits, previous_failure=router.previous_failure)
        precision, compute_receipt = router.decide(tension)
        if compute_receipt is not None:
            compute_receipts.append(compute_receipt)
            if receipt_store is not None:
                receipt_store.append(compute_receipt)
        generated.append(_mock_token(index, precision, seeds))
        tension_samples.append(
            {
                "step": index,
                "precision": precision,
                "tension": tension["tension"],
                "components": tension["components"],
            }
        )

    output = " ".join([prompt.strip(), *generated]).strip()
    verification = verifier.verify(prompt=prompt, output=output, constraints=constraints)

    if not verification.passed:
        router.mark_failure()
        tension = scan_recent_window(output[-500:], logits=None, previous_failure=True)
        precision, compute_receipt = router.decide(tension, force_escalate=True)
        if compute_receipt is not None:
            compute_receipts.append(compute_receipt)
            if receipt_store is not None:
                receipt_store.append(compute_receipt)
        tension_samples.append(
            {
                "step": len(generated),
                "precision": precision,
                "tension": tension["tension"],
                "components": tension["components"],
                "verification_failed": True,
            }
        )
    else:
        cognitive_receipt = make_cognitive_receipt(
            prompt=prompt,
            output=output,
            constraints=constraints,
            verified_by=verification.details.get("checkers", []),
            verifier_result=verification.to_dict(),
            metadata={"model": getattr(runner, "name", runner.__class__.__name__)},
        )
        cognitive_receipts.append(cognitive_receipt)
        if receipt_store is not None:
            receipt_store.append(cognitive_receipt)

    latency = perf_counter() - start
    return {
        "output": output,
        "stats": {
            "latency": latency,
            "verifier_pass": verification.passed,
            "escalations": len(compute_receipts),
            "receipts": len(cognitive_receipts) + len(compute_receipts),
            "output_length": len(output),
            "model": getattr(runner, "name", runner.__class__.__name__),
            "tokens_generated": len(generated),
        },
        "cognitive_receipts": cognitive_receipts,
        "compute_receipts": compute_receipts,
        "tension_samples": tension_samples,
        "verification": verification,
    }
