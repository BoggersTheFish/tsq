"""
Small built-in eval fixtures for repair behavior.
"""

from __future__ import annotations

from typing import Dict, List


V03_REPAIR_TASKS: List[Dict[str, object]] = [
    {
        "name": "missing_number_repair",
        "prompt": "Answer tersely.",
        "constraints": ["include 7"],
        "max_new_tokens": 2,
    },
    {
        "name": "forbidden_word_failure",
        "prompt": "Answer tersely.",
        "constraints": ["forbidden words: forbidden"],
        "max_new_tokens": 1,
    },
    {
        "name": "code_fence_closure_repair",
        "prompt": "Return a tiny code fence.",
        "constraints": ["include code fence"],
        "max_new_tokens": 1,
    },
]


V06_EVAL_TASKS: List[Dict[str, object]] = [
    {
        "name": "number_preservation_42",
        "prompt": "Answer with one compact sentence about TSQ and 42.",
        "constraints": ["include 42"],
        "max_new_tokens": 4,
        "expected_behavior": "output preserves the required number",
        "repairable": True,
    },
    {
        "name": "date_preservation_2026",
        "prompt": "Summarize the milestone date 2026-06-18.",
        "constraints": ["include 2026-06-18"],
        "max_new_tokens": 4,
        "expected_behavior": "output preserves the required date",
        "repairable": True,
    },
    {
        "name": "forbidden_word_avoidance",
        "prompt": "Answer tersely.",
        "constraints": ["forbidden words: forbidden"],
        "max_new_tokens": 1,
        "expected_behavior": "output avoids the forbidden word",
        "repairable": False,
    },
    {
        "name": "code_fence_closure",
        "prompt": "Return a tiny code fence.",
        "constraints": ["include code fence"],
        "max_new_tokens": 1,
        "expected_behavior": "output closes any opened code fence",
        "repairable": True,
    },
    {
        "name": "format_compliance_json",
        "prompt": "Return a tiny JSON note.",
        "constraints": ["include json"],
        "max_new_tokens": 3,
        "expected_behavior": "output reflects the requested format keyword",
        "repairable": True,
    },
    {
        "name": "multi_constraint_checksum_7",
        "prompt": "Answer with a short diagnostic.",
        "constraints": ["include checksum", "include 7"],
        "max_new_tokens": 2,
        "expected_behavior": "output reflects both checksum and the required number",
        "repairable": True,
    },
    {
        "name": "repairable_missing_constraint",
        "prompt": "Respond briefly.",
        "constraints": ["include checksum"],
        "max_new_tokens": 1,
        "expected_behavior": "dynamic repair can add the missing constraint token",
        "repairable": True,
    },
    {
        "name": "non_repairable_forbidden_output",
        "prompt": "Answer tersely.",
        "constraints": ["forbidden words: forbidden", "include forbidden"],
        "max_new_tokens": 1,
        "expected_behavior": "contradictory constraints should remain failed",
        "repairable": False,
    },
    {
        "name": "exact_word_attractor",
        "prompt": "Give the exact label.",
        "constraints": ["include attractor"],
        "max_new_tokens": 2,
        "expected_behavior": "output includes the exact required word",
        "repairable": True,
    },
    {
        "name": "short_answer_required_number",
        "prompt": "Short answer only.",
        "constraints": ["include 7"],
        "max_new_tokens": 1,
        "expected_behavior": "short output includes the required number",
        "repairable": True,
    },
    {
        "name": "negation_prompt",
        "prompt": "Do not over-explain; answer with one detail.",
        "constraints": ["include constraint"],
        "max_new_tokens": 2,
        "expected_behavior": "output handles negation while reflecting the constraint",
        "repairable": True,
    },
    {
        "name": "simple_code_requirement",
        "prompt": "Write a minimal Python marker.",
        "constraints": ["include code fence", "include return"],
        "max_new_tokens": 2,
        "expected_behavior": "output reflects code requirement and closes fences",
        "repairable": True,
    },
]
