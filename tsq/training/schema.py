"""
JSONL-friendly schemas for TSQ training data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class TrainingExample:
    id: str
    task_name: str
    split: str
    prompt: str
    constraints: List[str]
    input_text: str
    target_text: str
    example_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepairTrainingExample:
    id: str
    task_name: str
    split: str
    original_prompt: str
    constraints: List[str]
    failed_output: str
    verifier_failures: List[str]
    repair_prompt: str
    repair_target: str
    final_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreferenceExample:
    id: str
    prompt: str
    constraints: List[str]
    chosen_output: str
    rejected_output: str
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
