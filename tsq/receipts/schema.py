"""
Receipt schemas for TSQ runtime decisions and accepted outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BaseReceipt:
    """Common receipt envelope."""

    type: str
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=_utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CognitiveReceipt(BaseReceipt):
    """Receipt emitted when a generated output is accepted by verification."""

    prompt: str = ""
    output: str = ""
    constraints: List[str] = field(default_factory=list)
    verified_by: List[str] = field(default_factory=list)
    verifier_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputeReceipt(BaseReceipt):
    """Receipt emitted when TSQ spends extra precision on a tense span."""

    reason: str = ""
    target: str = ""
    from_precision: int = 4
    to_precision: int = 8
    tension: float = 0.0


def make_cognitive_receipt(
    prompt: str,
    output: str,
    constraints: List[str] | None = None,
    verified_by: List[str] | None = None,
    verifier_result: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> CognitiveReceipt:
    return CognitiveReceipt(
        type="cognitive",
        prompt=prompt,
        output=output,
        constraints=list(constraints or []),
        verified_by=list(verified_by or []),
        verifier_result=dict(verifier_result or {}),
        metadata=dict(metadata or {}),
    )


def make_compute_receipt(
    reason: str,
    target: str,
    from_precision: int,
    to_precision: int,
    tension: float,
    metadata: Dict[str, Any] | None = None,
) -> ComputeReceipt:
    return ComputeReceipt(
        type="compute",
        reason=reason,
        target=target,
        from_precision=from_precision,
        to_precision=to_precision,
        tension=max(0.0, min(1.0, float(tension))),
        metadata=dict(metadata or {}),
    )
