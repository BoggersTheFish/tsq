"""
Evaluation harness for isolated fixed-precision baselines and TSQ dynamic routing.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict, List, Sequence

from ..runtime.costing import fixed_precision_metrics, mean
from ..runtime.generation_loop import run_tsq_generation
from ..runtime.model_runner import MockModelRunner, ModelRunner
from ..verifier.base import Verifier
from .tasks import V03_REPAIR_TASKS, V06_EVAL_TASKS


BackendFactory = Callable[[], ModelRunner]


def _fixed_precision_run(
    prompt: str,
    constraints: Sequence[str],
    max_new_tokens: int,
    precision: str,
    model: ModelRunner,
) -> Dict[str, Any]:
    start = perf_counter()
    output = model.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        precision=precision,
        constraints=constraints,
    )
    verification = Verifier().verify(prompt=prompt, output=output, constraints=constraints)
    cost_metrics = fixed_precision_metrics(precision, max_new_tokens)
    return {
        "output": output,
        "metrics": {
            "latency": perf_counter() - start,
            "verifier_pass": verification.passed,
            "original_verifier_pass": verification.passed,
            "final_verifier_pass": verification.passed,
            "escalations": 0,
            "receipts": 0,
            "repair_attempted": False,
            "repair_succeeded": False,
            "output_length": len(output),
            **cost_metrics,
        },
        "verification": verification,
    }


def compare_baselines(
    prompt: str,
    constraints: Sequence[str] | None = None,
    max_new_tokens: int = 64,
    model: ModelRunner | None = None,
    backend_factory: BackendFactory | None = None,
) -> Dict[str, Dict[str, Any]]:
    constraints = list(constraints or [])
    if backend_factory is None:
        if model is not None:
            def backend_factory() -> ModelRunner:
                return model.__class__()

        else:
            backend_factory = MockModelRunner

    q4_model = backend_factory()
    q8_model = backend_factory()
    dynamic_model = backend_factory()
    q4 = _fixed_precision_run(prompt, constraints, max_new_tokens, "Q4", q4_model)
    q8 = _fixed_precision_run(prompt, constraints, max_new_tokens, "Q8", q8_model)
    dynamic = run_tsq_generation(
        prompt=prompt,
        constraints=constraints,
        max_new_tokens=max_new_tokens,
        model=dynamic_model,
    )
    return {
        "always_Q4": q4,
        "always_Q8": q8,
        "TSQ_dynamic": {
            "output": dynamic["output"],
            "metrics": {
                "latency": dynamic["stats"]["latency"],
                "verifier_pass": dynamic["stats"]["verifier_pass"],
                "original_verifier_pass": dynamic["stats"]["original_verifier_pass"],
                "final_verifier_pass": dynamic["stats"]["final_verifier_pass"],
                "escalations": dynamic["stats"]["escalations"],
                "receipts": dynamic["stats"]["receipts"],
                "repair_attempted": dynamic["stats"]["repair_attempted"],
                "repair_succeeded": dynamic["stats"]["repair_succeeded"],
                "output_length": dynamic["stats"]["output_length"],
                "tokens_at_Q4": dynamic["stats"]["tokens_at_Q4"],
                "tokens_at_Q8": dynamic["stats"]["tokens_at_Q8"],
                "tokens_at_FP16": dynamic["stats"]["tokens_at_FP16"],
                "tokens_at_residual_unfolded": dynamic["stats"][
                    "tokens_at_residual_unfolded"
                ],
                "repair_tokens_generated": dynamic["stats"]["repair_tokens_generated"],
                "total_tokens_generated": dynamic["stats"]["total_tokens_generated"],
                "estimated_cost_units": dynamic["stats"]["estimated_cost_units"],
                "cost_model_used": dynamic["stats"]["cost_model_used"],
                "precision_histogram": dynamic["stats"]["precision_histogram"],
            },
            "verification": dynamic["verification"],
            "original_verification": dynamic["original_verification"],
            "final_verification": dynamic["final_verification"],
            "compute_receipts": dynamic["compute_receipts"],
            "cognitive_receipts": dynamic["cognitive_receipts"],
            "tension_samples": dynamic["tension_samples"],
        },
    }


def run_eval(
    prompt: str,
    constraints: Sequence[str] | None = None,
    max_new_tokens: int = 64,
    model: ModelRunner | None = None,
) -> Dict[str, Dict[str, Any]]:
    return compare_baselines(prompt, constraints, max_new_tokens, model=model)


def repair_eval_tasks() -> List[Dict[str, object]]:
    return list(V03_REPAIR_TASKS)


def eval_suite_tasks() -> List[Dict[str, object]]:
    return list(V06_EVAL_TASKS)


def run_eval_suite(
    backend_factory: BackendFactory | None = None,
    tasks: Sequence[Dict[str, object]] | None = None,
) -> Dict[str, Any]:
    factory = backend_factory or MockModelRunner
    task_results: List[Dict[str, Any]] = []
    for task in tasks or V06_EVAL_TASKS:
        results = compare_baselines(
            prompt=str(task["prompt"]),
            constraints=list(task["constraints"]),
            max_new_tokens=int(task["max_new_tokens"]),
            backend_factory=factory,
        )
        task_results.append(
            {
                "name": task["name"],
                "prompt": task["prompt"],
                "constraints": list(task["constraints"]),
                "max_new_tokens": task["max_new_tokens"],
                "expected_behavior": task["expected_behavior"],
                "repairable": bool(task["repairable"]),
                "results": results,
            }
        )

    ratios = [
        item["results"]["TSQ_dynamic"]["metrics"]["estimated_cost_units"]
        / max(item["results"]["always_Q8"]["metrics"]["estimated_cost_units"], 1e-9)
        for item in task_results
    ]
    dynamic_costs = [
        item["results"]["TSQ_dynamic"]["metrics"]["estimated_cost_units"] for item in task_results
    ]
    aggregate = {
        "total_tasks": len(task_results),
        "q4_passes": sum(
            1 for item in task_results if item["results"]["always_Q4"]["metrics"]["verifier_pass"]
        ),
        "q8_passes": sum(
            1 for item in task_results if item["results"]["always_Q8"]["metrics"]["verifier_pass"]
        ),
        "dynamic_passes": sum(
            1
            for item in task_results
            if item["results"]["TSQ_dynamic"]["metrics"]["final_verifier_pass"]
        ),
        "dynamic_repairs_attempted": sum(
            1
            for item in task_results
            if item["results"]["TSQ_dynamic"]["metrics"]["repair_attempted"]
        ),
        "dynamic_repairs_succeeded": sum(
            1
            for item in task_results
            if item["results"]["TSQ_dynamic"]["metrics"]["repair_succeeded"]
        ),
        "total_dynamic_compute_receipts": sum(
            len(item["results"]["TSQ_dynamic"].get("compute_receipts", [])) for item in task_results
        ),
        "mean_dynamic_vs_q8_cost_ratio": mean(ratios),
        "mean_dynamic_estimated_cost": mean(dynamic_costs),
    }
    return {"tasks": task_results, "aggregate": aggregate}
