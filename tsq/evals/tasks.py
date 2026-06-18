"""
Small built-in eval fixtures for v0.3 repair behavior.
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
