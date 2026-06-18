"""
Build small deterministic TSQ training datasets from eval tasks and reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .. import __version__
from ..evals.tasks import V03_REPAIR_TASKS, V06_EVAL_TASKS
from ..runtime.costing import DEFAULT_COST_MODEL_NAME
from ..runtime.generation_loop import run_tsq_generation
from ..runtime.model_runner import RepairAwareMockRunner
from .schema import PreferenceExample, RepairTrainingExample, TrainingExample


SUPERVISED_TRAIN = "tsq_supervised_train.jsonl"
SUPERVISED_EVAL = "tsq_supervised_eval.jsonl"
REPAIR_TRAIN = "tsq_repair_train.jsonl"
PREFERENCE_PAIRS = "tsq_preference_pairs.jsonl"


def build_datasets(
    out_dir: str | Path = "data/generated",
    include_example_reports: str | Path | None = None,
) -> Dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    supervised = build_supervised_examples()
    if include_example_reports:
        supervised.extend(build_report_examples(Path(include_example_reports)))

    repair_examples, preference_examples = build_repair_and_preference_examples()
    train_examples = [item for item in supervised if item.split == "train"]
    eval_examples = [item for item in supervised if item.split == "eval"]

    paths = {
        "supervised_train": out_path / SUPERVISED_TRAIN,
        "supervised_eval": out_path / SUPERVISED_EVAL,
        "repair_train": out_path / REPAIR_TRAIN,
        "preference_pairs": out_path / PREFERENCE_PAIRS,
    }
    write_jsonl(paths["supervised_train"], (item.to_dict() for item in train_examples))
    write_jsonl(paths["supervised_eval"], (item.to_dict() for item in eval_examples))
    write_jsonl(paths["repair_train"], (item.to_dict() for item in repair_examples))
    write_jsonl(paths["preference_pairs"], (item.to_dict() for item in preference_examples))
    return paths


def build_supervised_examples() -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    for index, task in enumerate(V06_EVAL_TASKS):
        split = "eval" if index % 4 == 0 else "train"
        example_type = infer_example_type(task)
        target = target_for_task(task)
        examples.append(
            TrainingExample(
                id=f"v06-{split}-{index:03d}",
                task_name=str(task["name"]),
                split=split,
                prompt=str(task["prompt"]),
                constraints=list(task["constraints"]),
                input_text=format_constraint_input(str(task["prompt"]), list(task["constraints"])),
                target_text=target,
                example_type=example_type,
                metadata=metadata_for_task(task, example_type),
            )
        )

    for index, task in enumerate(V03_REPAIR_TASKS):
        example_type = infer_example_type(task)
        examples.append(
            TrainingExample(
                id=f"v03-train-{index:03d}",
                task_name=str(task["name"]),
                split="train",
                prompt=str(task["prompt"]),
                constraints=list(task["constraints"]),
                input_text=format_constraint_input(str(task["prompt"]), list(task["constraints"])),
                target_text=target_for_task(task),
                example_type=example_type,
                metadata=metadata_for_task(task, example_type),
            )
        )
    return examples


def build_report_examples(report_dir: Path) -> List[TrainingExample]:
    if not report_dir.exists():
        return []
    examples: List[TrainingExample] = []
    for report_index, path in enumerate(sorted(report_dir.glob("*.json"))):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        mode = str(report.get("mode", "report"))
        cost_model = _find_cost_model(report)
        if mode == "generate" and report.get("output"):
            prompt = str(report.get("prompt", ""))
            constraints = list(report.get("constraints", []))
            examples.append(
                TrainingExample(
                    id=f"report-{report_index:03d}-receipt",
                    task_name=f"report:{path.stem}",
                    split="train",
                    prompt=prompt,
                    constraints=constraints,
                    input_text=format_constraint_input(prompt, constraints),
                    target_text=str(report["output"]),
                    example_type="receipt_aware_response",
                    metadata={
                        "source_report": path.name,
                        "source_task": mode,
                        "repairable": False,
                        "verifier_expected": bool(
                            report.get("final_verification", {}).get("passed", False)
                        ),
                        "precision_hint": _precision_hint(report),
                        "cost_model_used": cost_model,
                        "tsq_version": __version__,
                    },
                )
            )
        if mode in {"eval", "eval-suite"}:
            examples.extend(_report_dynamic_examples(report, path.name, report_index, cost_model))
    return examples


def build_repair_and_preference_examples() -> tuple[List[RepairTrainingExample], List[PreferenceExample]]:
    repair_examples: List[RepairTrainingExample] = []
    preference_examples: List[PreferenceExample] = []
    repair_tasks = [
        task for task in [*V03_REPAIR_TASKS, *V06_EVAL_TASKS] if bool(task.get("repairable", True))
    ]
    for index, task in enumerate(repair_tasks):
        result = run_tsq_generation(
            prompt=str(task["prompt"]),
            constraints=list(task["constraints"]),
            max_new_tokens=int(task["max_new_tokens"]),
            model=RepairAwareMockRunner(),
        )
        original = result["original_verification"]
        final = result["final_verification"]
        if original.passed or not final.passed:
            continue
        failed_output = _failed_prefix(result)
        final_output = str(result["output"])
        repair_target = final_output[len(failed_output) :].strip() or _last_token(final_output)
        repair_prompt = build_repair_prompt(
            constraints=list(task["constraints"]),
            verifier_failures=list(original.failures),
            current_output=failed_output,
        )
        metadata = {
            "source_task": task["name"],
            "repairable": True,
            "verifier_expected": True,
            "precision_hint": _repair_precision(result),
            "cost_model_used": result["stats"].get("cost_model_used"),
            "tsq_version": __version__,
        }
        repair_examples.append(
            RepairTrainingExample(
                id=f"repair-train-{index:03d}",
                task_name=str(task["name"]),
                split="train",
                original_prompt=str(task["prompt"]),
                constraints=list(task["constraints"]),
                failed_output=failed_output,
                verifier_failures=list(original.failures),
                repair_prompt=repair_prompt,
                repair_target=repair_target,
                final_output=final_output,
                metadata=metadata,
            )
        )
        preference_examples.append(
            PreferenceExample(
                id=f"preference-{index:03d}",
                prompt=str(task["prompt"]),
                constraints=list(task["constraints"]),
                chosen_output=final_output,
                rejected_output=failed_output,
                reason="verifier failure repaired into constraint-satisfying output",
                metadata=metadata,
            )
        )
    return repair_examples, preference_examples


def build_repair_prompt(
    constraints: Sequence[str],
    verifier_failures: Sequence[str],
    current_output: str,
) -> str:
    return (
        "Repair the output to satisfy these constraints.\n"
        f"Constraints: {list(constraints)}\n"
        f"Verification failures: {list(verifier_failures)}\n"
        f"Current output: {current_output}\n"
        "Repair:"
    )


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def infer_example_type(task: Dict[str, Any]) -> str:
    name = str(task.get("name", "")).lower()
    constraints = " ".join(str(item) for item in task.get("constraints", [])).lower()
    prompt = str(task.get("prompt", "")).lower()
    text = " ".join([name, constraints, prompt])
    if "forbidden" in text:
        return "forbidden_word_avoidance"
    if "code fence" in text or "```" in text or "code" in text:
        return "code_fence_closure"
    if "date" in text or "2026-" in text:
        return "date_preservation"
    if any(char.isdigit() for char in text):
        return "number_preservation"
    if "repair" in text or bool(task.get("repairable", False)):
        return "repair_from_verifier_feedback"
    return "constraint_following"


def target_for_task(task: Dict[str, Any]) -> str:
    constraints = [str(item) for item in task.get("constraints", [])]
    prompt = str(task.get("prompt", "")).strip()
    lower = " ".join([prompt, *constraints]).lower()
    if "forbidden words: forbidden" in lower and "include forbidden" not in lower:
        return "compliant"
    if "code fence" in lower:
        return "```python\nreturn 7\n```" if "return" in lower else "```python\npass\n```"
    if "2026-06-18" in lower:
        return "2026-06-18"
    if "include 42" in lower:
        return "42"
    if "include 7" in lower:
        return "7"
    if "checksum" in lower:
        return "checksum"
    if "attractor" in lower:
        return "attractor"
    if "constraint" in lower:
        return "constraint"
    if "json" in lower:
        return '{"status":"ok"}'
    return "constraint satisfied"


def format_constraint_input(prompt: str, constraints: Sequence[str]) -> str:
    return f"Prompt: {prompt}\nConstraints: {list(constraints)}\nResponse:"


def metadata_for_task(task: Dict[str, Any], example_type: str) -> Dict[str, Any]:
    return {
        "source_task": task.get("name"),
        "repairable": bool(task.get("repairable", False)),
        "verifier_expected": not _is_intentionally_unrepairable(task),
        "precision_hint": precision_hint_for_type(example_type),
        "cost_model_used": None,
        "tsq_version": __version__,
    }


def precision_hint_for_type(example_type: str) -> str:
    if example_type in {
        "forbidden_word_avoidance",
        "code_fence_closure",
        "repair_from_verifier_feedback",
    }:
        return "Q8"
    return "Q4"


def _is_intentionally_unrepairable(task: Dict[str, Any]) -> bool:
    return bool(task.get("repairable") is False)


def _failed_prefix(result: Dict[str, Any]) -> str:
    repair_tokens = int(result["stats"].get("repair_tokens_generated", 0))
    tokens = str(result["output"]).split()
    if repair_tokens <= 0:
        return str(result["output"])
    return " ".join(tokens[:-repair_tokens])


def _last_token(text: str) -> str:
    parts = text.split()
    return parts[-1] if parts else text


def _repair_precision(result: Dict[str, Any]) -> str:
    for sample in result.get("tension_samples", []):
        if sample.get("repair"):
            return str(sample.get("step_precision", sample.get("precision", "Q8")))
    return "Q8"


def _find_cost_model(report: Dict[str, Any]) -> str | None:
    if "stats" in report:
        return DEFAULT_COST_MODEL_NAME if report["stats"].get("cost_model_used") else None
    if "summary" in report:
        histogram = report["summary"].get("precision_histogram")
        return DEFAULT_COST_MODEL_NAME if histogram else None
    return None


def _precision_hint(report: Dict[str, Any]) -> str:
    histogram = report.get("precision_histogram") or report.get("stats", {}).get("precision_histogram", {})
    counts = histogram.get("counts", {}) if isinstance(histogram, dict) else {}
    if counts.get("FP16", 0):
        return "FP16"
    if counts.get("Q8", 0):
        return "Q8"
    return "Q4"


def _report_dynamic_examples(
    report: Dict[str, Any],
    report_name: str,
    report_index: int,
    cost_model: str | None,
) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    if report.get("mode") == "eval":
        dynamic = report.get("results", {}).get("TSQ_dynamic", {})
        prompt = str(report.get("prompt", ""))
        constraints = list(report.get("constraints", []))
        output = str(dynamic.get("output", ""))
        if output:
            examples.append(
                TrainingExample(
                    id=f"report-{report_index:03d}-dynamic",
                    task_name=f"report:{report_name}",
                    split="train",
                    prompt=prompt,
                    constraints=constraints,
                    input_text=format_constraint_input(prompt, constraints),
                    target_text=output,
                    example_type="dynamic_precision_explanation",
                    metadata={
                        "source_report": report_name,
                        "source_task": report.get("mode"),
                        "repairable": bool(dynamic.get("metrics", {}).get("repair_attempted", False)),
                        "verifier_expected": bool(
                            dynamic.get("metrics", {}).get("final_verifier_pass", False)
                        ),
                        "precision_hint": _precision_hint(dynamic.get("metrics", {})),
                        "cost_model_used": cost_model,
                        "tsq_version": __version__,
                    },
                )
            )
    return examples
