"""
Cheap verifier primitives for TSQ v0.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable, Dict, Iterable, List, Sequence


Checker = Callable[[str, str, Sequence[str]], List[str]]


@dataclass
class VerificationResult:
    passed: bool
    failures: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "details": dict(self.details),
        }


class Verifier:
    """Runs a configurable list of cheap output checkers."""

    def __init__(self, checkers: Iterable[Checker] | None = None):
        self.checkers = list(
            checkers
            or [
                constraint_preservation_checker,
                no_forbidden_words_checker,
                number_date_preservation_checker,
                code_fence_closure_checker,
            ]
        )

    def verify(
        self,
        prompt: str,
        output: str,
        constraints: Sequence[str] | None = None,
    ) -> VerificationResult:
        constraints = list(constraints or [])
        failures: List[str] = []
        checker_names: List[str] = []
        for checker in self.checkers:
            checker_names.append(checker.__name__)
            failures.extend(checker(prompt, output, constraints))
        return VerificationResult(
            passed=not failures,
            failures=failures,
            details={"checkers": checker_names, "constraint_count": len(constraints)},
        )


def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def constraint_preservation_checker(
    prompt: str,
    output: str,
    constraints: Sequence[str],
) -> List[str]:
    failures: List[str] = []
    output_tokens = set(_tokens(output))
    for constraint in constraints:
        tokens = [
            token
            for token in _tokens(constraint)
            if token
            not in {
                "a",
                "an",
                "and",
                "be",
                "exactly",
                "include",
                "must",
                "one",
                "only",
                "the",
                "with",
            }
        ]
        required = [token for token in tokens if len(token) > 2]
        if required and not any(token in output_tokens for token in required):
            failures.append(f"constraint not reflected: {constraint}")
    return failures


def no_forbidden_words_checker(
    prompt: str,
    output: str,
    constraints: Sequence[str],
) -> List[str]:
    failures: List[str] = []
    lower_output = output.lower()
    patterns = [
        r"forbidden(?: words?)?\s*[:=]\s*([A-Za-z0-9_, -]+)",
        r"do not (?:say|include|use)\s+([A-Za-z0-9_-]+)",
        r"never (?:say|include|use)\s+([A-Za-z0-9_-]+)",
        r"avoid\s+([A-Za-z0-9_-]+)",
    ]
    for text in constraints:
        for pattern in patterns:
            for match in re.finditer(pattern, text.lower()):
                words = re.split(r"[\s,]+", match.group(1).strip())
                for word in filter(None, words):
                    if re.search(rf"\b{re.escape(word)}\b", lower_output):
                        failures.append(f"forbidden word present: {word}")
    return failures


def number_date_preservation_checker(
    prompt: str,
    output: str,
    constraints: Sequence[str],
) -> List[str]:
    source = " ".join([prompt, *constraints])
    numbers = set(re.findall(r"\b\d+(?:[./-]\d+)*(?:\.\d+)?\b", source))
    missing = sorted(number for number in numbers if number not in output)
    if missing:
        return [f"missing number/date from prompt or constraints: {', '.join(missing)}"]
    return []


def code_fence_closure_checker(
    prompt: str,
    output: str,
    constraints: Sequence[str],
) -> List[str]:
    if output.count("```") % 2:
        return ["unclosed code fence"]
    return []
