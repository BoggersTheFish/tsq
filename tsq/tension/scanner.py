"""
TS Node: TensionScanner
Type: tension_plane_component
Description: Computes cheap, near-zero-cost tension signals in the hot path.
             Intentionally crude for v0. Richer signals deferred to offline/failure paths.
Tension sources (v0):
  - next_token_entropy (from logits)
  - lexical_risk (negation, numbers, code/math markers, explicit constraint words)
  - previous_verifier_failure flag
  - prompt_length / requirement_count (static for now)
Edges:
  - scanner_output → routes_to(precision_router)
  - high_tension_signal → activates(escalation_path)
Verifier hooks: none in hot path (verifier is downstream)
Receipt outputs: tension_score + contributing_signals dict (logged to compute receipt on escalation)
"""

from __future__ import annotations
from typing import Any, Dict, List
import math


def cheap_entropy(logits: Any) -> float:
    """
    Approximate next-token entropy from logits.
    For real implementation: use log_softmax + negative sum(p * log p).
    v0 stub returns a normalized proxy (0.0 low tension → 1.0 high tension).
    """
    # Mock/demo runners expose a cheap entropy_proxy; real runners can pass logits.
    if hasattr(logits, "entropy_proxy"):
        return float(getattr(logits, "entropy_proxy"))
    return 0.3  # default low tension


def lexical_risk(text_window: str) -> float:
    """
    Cheap lexical detectors. Scan recent generated text or prompt tail.
    High risk triggers: numbers, dates, negations, code fences, math symbols,
    explicit constraint language ("must", "exactly", "never", "constraint", etc.).
    """
    if not text_window:
        return 0.0
    text = text_window.lower()
    risk = 0.0
    markers = ["must", "exactly", "never", "always", "constraint", "require",
               "forbidden", "only if", "number", "date", "```", "def ", "class ",
               "import ", "return ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for m in markers:
        if m in text:
            risk += 0.15
    return min(risk, 1.0)


def compute_tension(
    entropy: float,
    lexical: float,
    verifier_failure: bool = False,
    prompt_tension: float = 0.0,
    weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Combine cheap signals into a single tension score + breakdown.
    This is the main hot-path function called every generation step (or every N steps).
    """
    if weights is None:
        weights = {"entropy": 0.4, "lexical": 0.35, "failure": 0.2, "prompt": 0.05}

    failure_risk = 1.0 if verifier_failure else 0.0

    tension = (
        weights["entropy"] * entropy +
        weights["lexical"] * lexical +
        weights["failure"] * failure_risk +
        weights["prompt"] * prompt_tension
    )
    tension = max(0.0, min(1.0, tension))

    return {
        "tension": tension,
        "components": {
            "entropy": entropy,
            "lexical_risk": lexical,
            "verifier_failure_risk": failure_risk,
            "prompt_tension": prompt_tension,
        },
        "weights_used": weights,
        "thresholds": {"low": 0.35, "medium": 0.6, "high": 0.8},
    }


def scan_recent_window(recent_text: str, logits: Any = None, previous_failure: bool = False) -> Dict[str, Any]:
    """
    Convenience wrapper for the common case: scan last N tokens + current logits.
    Called from generation_loop.
    """
    entropy = cheap_entropy(logits) if logits is not None else 0.3
    lexical = lexical_risk(recent_text)
    return compute_tension(entropy, lexical, verifier_failure=previous_failure)
