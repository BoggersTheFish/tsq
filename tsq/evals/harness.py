"""
Small v0.1 evaluation harness for fixed-precision baselines and TSQ dynamic routing.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Sequence

from ..runtime.generation_loop import run_tsq_generation
from ..runtime.model_runner import MockModelRunner, ModelRunner
from ..verifier.base import Verifier


def _fixed_precision_run(
    prompt: str,
    constraints: Sequence[str],
    max_new_tokens: int,
    precision: str,
    model: ModelRunner,
) -> Dict[str, Any]:
    start = perf_counter()
    output = model.generate(prompt=prompt, max_new_tokens=max_new_tokens, precision=precision)
    verification = Verifier().verify(prompt=prompt, output=output, constraints=constraints)
    return {
        "output": output,
        "metrics": {
            "latency": perf_counter() - start,
            "verifier_pass": verification.passed,
            "escalations": 0,
            "receipts": 0,
            "output_length": len(output),
        },
        "verification": verification,
    }


def compare_baselines(
    prompt: str,
    constraints: Sequence[str] | None = None,
    max_new_tokens: int = 64,
    model: ModelRunner | None = None,
) -> Dict[str, Dict[str, Any]]:
    constraints = list(constraints or [])
    base_model = model or MockModelRunner()
    q4 = _fixed_precision_run(prompt, constraints, max_new_tokens, "Q4", base_model)
    q8 = _fixed_precision_run(prompt, constraints, max_new_tokens, "Q8", base_model)
    dynamic = run_tsq_generation(
        prompt=prompt,
        constraints=constraints,
        max_new_tokens=max_new_tokens,
        model=base_model,
    )
    return {
        "always_Q4": q4,
        "always_Q8": q8,
        "TSQ_dynamic": {
            "output": dynamic["output"],
            "metrics": {
                "latency": dynamic["stats"]["latency"],
                "verifier_pass": dynamic["stats"]["verifier_pass"],
                "escalations": dynamic["stats"]["escalations"],
                "receipts": dynamic["stats"]["receipts"],
                "output_length": dynamic["stats"]["output_length"],
            },
            "verification": dynamic["verification"],
        },
    }


def run_eval(
    prompt: str,
    constraints: Sequence[str] | None = None,
    max_new_tokens: int = 64,
    model: ModelRunner | None = None,
) -> Dict[str, Dict[str, Any]]:
    return compare_baselines(prompt, constraints, max_new_tokens, model)
