"""
Validate and summarize TSQ JSONL training datasets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


class DatasetValidationError(ValueError):
    pass


COMMON_REQUIRED = {"id", "metadata"}
SUPERVISED_REQUIRED = {
    "id",
    "task_name",
    "split",
    "prompt",
    "constraints",
    "input_text",
    "target_text",
    "example_type",
    "metadata",
}
REPAIR_REQUIRED = {
    "id",
    "task_name",
    "split",
    "original_prompt",
    "constraints",
    "failed_output",
    "verifier_failures",
    "repair_prompt",
    "repair_target",
    "final_output",
    "metadata",
}
PREFERENCE_REQUIRED = {
    "id",
    "prompt",
    "constraints",
    "chosen_output",
    "rejected_output",
    "reason",
    "metadata",
}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise DatasetValidationError(f"line {line_number}: expected object")
            rows.append(row)
    return rows


def validate_dataset(path: str | Path, max_chars: int = 8000) -> List[Dict[str, Any]]:
    rows = load_jsonl(path)
    if not rows:
        raise DatasetValidationError("dataset is empty")

    seen_ids: set[str] = set()
    splits: set[str] = set()
    for index, row in enumerate(rows, start=1):
        _validate_row(index, row, max_chars=max_chars)
        row_id = str(row["id"])
        if row_id in seen_ids:
            raise DatasetValidationError(f"duplicate id: {row_id}")
        seen_ids.add(row_id)
        if "split" in row:
            splits.add(str(row["split"]))

    if splits and "train" not in splits and "eval" not in splits:
        raise DatasetValidationError("dataset split must include train or eval")
    return rows


def dataset_summary(path: str | Path) -> Dict[str, Any]:
    rows = load_jsonl(path)
    split_counts = Counter(str(row.get("split", "unspecified")) for row in rows)
    type_counts = Counter(_row_type(row) for row in rows)
    input_lengths = [_input_text(row) for row in rows]
    target_lengths = [_target_text(row) for row in rows]
    return {
        "path": str(path),
        "examples": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "example_type_counts": dict(sorted(type_counts.items())),
        "average_input_length": _average(len(item) for item in input_lengths),
        "average_target_length": _average(len(item) for item in target_lengths),
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"examples: {summary['examples']}")
    print(f"split_counts: {summary['split_counts']}")
    print(f"example_type_counts: {summary['example_type_counts']}")
    print(f"average_input_length: {summary['average_input_length']:.2f}")
    print(f"average_target_length: {summary['average_target_length']:.2f}")


def _validate_row(index: int, row: Dict[str, Any], max_chars: int) -> None:
    required = _required_fields(row)
    missing = sorted(field for field in required if field not in row)
    if missing:
        raise DatasetValidationError(f"row {index}: missing fields: {', '.join(missing)}")
    if not str(row.get("id", "")).strip():
        raise DatasetValidationError(f"row {index}: empty id")
    if len(json.dumps(row, sort_keys=True)) > max_chars:
        raise DatasetValidationError(f"row {index}: exceeds max chars {max_chars}")

    if "input_text" in row and not str(row["input_text"]).strip():
        raise DatasetValidationError(f"row {index}: empty input_text")
    if "target_text" in row and not str(row["target_text"]).strip():
        raise DatasetValidationError(f"row {index}: empty target_text")
    if "repair_prompt" in row and not str(row["repair_prompt"]).strip():
        raise DatasetValidationError(f"row {index}: empty repair_prompt")
    if "repair_target" in row and not str(row["repair_target"]).strip():
        raise DatasetValidationError(f"row {index}: empty repair_target")
    if "verifier_failures" in row and not row["verifier_failures"]:
        raise DatasetValidationError(f"row {index}: repair example missing verifier_failures")
    if "chosen_output" in row and not str(row["chosen_output"]).strip():
        raise DatasetValidationError(f"row {index}: empty chosen_output")
    if "rejected_output" in row and not str(row["rejected_output"]).strip():
        raise DatasetValidationError(f"row {index}: empty rejected_output")
    if "constraints" in row and not isinstance(row["constraints"], list):
        raise DatasetValidationError(f"row {index}: constraints must be a list")


def _required_fields(row: Dict[str, Any]) -> set[str]:
    if "repair_prompt" in row:
        return REPAIR_REQUIRED
    if "chosen_output" in row:
        return PREFERENCE_REQUIRED
    if "input_text" in row:
        return SUPERVISED_REQUIRED
    return COMMON_REQUIRED


def _row_type(row: Dict[str, Any]) -> str:
    if "example_type" in row:
        return str(row["example_type"])
    if "repair_prompt" in row:
        return "repair_from_verifier_feedback"
    if "chosen_output" in row:
        return "preference_pair"
    return "unknown"


def _input_text(row: Dict[str, Any]) -> str:
    if "input_text" in row:
        return str(row["input_text"])
    if "repair_prompt" in row:
        return str(row["repair_prompt"])
    return str(row.get("prompt", ""))


def _target_text(row: Dict[str, Any]) -> str:
    if "target_text" in row:
        return str(row["target_text"])
    if "repair_target" in row:
        return str(row["repair_target"])
    return str(row.get("chosen_output", ""))


def _average(values: Iterable[int]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tsq.training.validate_dataset")
    parser.add_argument("path")
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--summary", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_dataset(args.path, max_chars=args.max_chars)
        if args.summary:
            print_summary(dataset_summary(args.path))
        else:
            print(f"valid dataset: {args.path}")
    except DatasetValidationError as exc:
        print(f"dataset validation error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
