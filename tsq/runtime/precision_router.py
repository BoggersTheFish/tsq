"""
TS Node: PrecisionRouter
Type: routing_plane_component
Description: Decides current precision level based on live tension score.
             v0: simple threshold rules (cheap). Later: learned or graph-propagated policy.
Tension sources: tension score from TensionScanner + previous verifier state
Edges:
  - low_tension → activates(Q4_path)
  - medium_tension → activates(Q8_path)
  - high_tension_or_failure → activates(FP16_or_rerun_path)
Verifier hooks: on failure, force escalate regardless of current tension
Receipt outputs: every routing decision that causes escalation emits a ComputeReceipt
"""

from __future__ import annotations
from typing import Literal, Dict, Any, Optional
from ..receipts.schema import make_compute_receipt, ComputeReceipt


PrecisionLevel = Literal["Q4", "Q8", "FP16", "residual_unfolded"]


class PrecisionRouter:
    """
    Rules-based router for v0.
    Future versions can make thresholds dynamic or learned from receipt history.
    """

    def __init__(
        self,
        low_threshold: float = 0.35,
        medium_threshold: float = 0.60,
        high_threshold: float = 0.80,
    ):
        self.low = low_threshold
        self.medium = medium_threshold
        self.high = high_threshold
        self.current_precision: PrecisionLevel = "Q4"
        self.previous_failure: bool = False
        self.escalation_history: list[ComputeReceipt] = []

    def decide(
        self,
        tension_result: Dict[str, Any],
        force_escalate: bool = False,
    ) -> tuple[PrecisionLevel, Optional[ComputeReceipt]]:
        """
        Main decision function. Returns (chosen_precision, optional_compute_receipt_if_escalated).
        """
        tension = tension_result["tension"]
        components = tension_result["components"]
        previous_precision = self.current_precision

        if force_escalate or self.previous_failure or tension >= self.high:
            new_precision: PrecisionLevel = "FP16" if tension > 0.9 else "Q8"
            receipt = make_compute_receipt(
                reason="high_tension_or_verifier_failure",
                target="current_generation_span",
                from_precision=self._level_to_bits(previous_precision),
                to_precision=self._level_to_bits(new_precision),
                tension=tension,
                metadata={
                    "tension_components": dict(components),
                    "chosen_precision": new_precision,
                    "previous_precision": previous_precision,
                    "target": "current_generation_span",
                },
            )
            self.escalation_history.append(receipt)
            self.current_precision = new_precision
            self.previous_failure = False  # reset after handling
            return new_precision, receipt

        if tension >= self.medium:
            new_precision = "Q8"
            if new_precision != self.current_precision:
                receipt = make_compute_receipt(
                    reason="medium_tension",
                    target="current_generation_span",
                    from_precision=self._level_to_bits(previous_precision),
                    to_precision=8,
                    tension=tension,
                    metadata={
                        "tension_components": dict(components),
                        "chosen_precision": new_precision,
                        "previous_precision": previous_precision,
                        "target": "current_generation_span",
                    },
                )
                self.escalation_history.append(receipt)
                self.current_precision = new_precision
                return new_precision, receipt
            return new_precision, None

        # default cheap path
        self.current_precision = "Q4"
        return "Q4", None

    def _level_to_bits(self, level: PrecisionLevel) -> int:
        return {"Q4": 4, "Q8": 8, "FP16": 16, "residual_unfolded": 16}.get(level, 4)

    def mark_failure(self):
        """Called by verifier when check fails. Next decide() will escalate."""
        self.previous_failure = True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "current_precision": self.current_precision,
            "escalations": len(self.escalation_history),
            "last_tension": None,  # filled by caller if needed
        }
