"""
TS Node: ModelRunner
Type: runtime_abstraction
Description: Protocol + implementations for pluggable model backends.
             Allows the TSQ loop to remain model-agnostic while supporting
             mock (for dev) and real quantized models (Transformers, llama.cpp, etc.).
             v0.2: backend-owned token generation via StepResult.
Tension sources: none in the runner itself (tension lives in scanner + router)
Edges:
  - runner.step(precision) → produces logits-like object with entropy_proxy
  - runner.generate(...) → full continuation at fixed precision (for baselines)
Verifier hooks: none (verifier is downstream of generation)
Receipt outputs: none directly (runner is pure execution surface)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable, Optional
import importlib.util


@dataclass
class StepResult:
    """One backend-owned generation step."""

    token_text: str
    entropy_proxy: float
    precision: str
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ModelRunner(Protocol):
    """Minimal protocol that any TSQ-compatible model runner must satisfy."""

    name: str

    def step(self, precision: str = "Q4", **kwargs: Any) -> StepResult:
        """
        Perform one backend-owned token step.

        The returned StepResult exposes token_text for the generation loop and
        entropy_proxy for the cheap tension scanner.
        """
        ...

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        precision: str = "Q4",
        **kwargs: Any,
    ) -> str:
        """Generate a full continuation at a *fixed* precision (used for baselines)."""
        ...


class MockModelRunner:
    """
    Cheap mock implementation for development, tests, and demos.
    Simulates varying entropy to exercise the tension router.
    """

    def __init__(self, name: str = "mock-tensionlm-117m-q4q8"):
        self.name = name
        self.call_count: int = 0

    def step(self, precision: str = "Q4", **kwargs: Any) -> StepResult:
        self.call_count += 1

        # Simulate harder steps having higher entropy (proxy for tension)
        entropy = 0.22 + (self.call_count % 9) * 0.075
        token_text = self._next_token_text(precision=precision, **kwargs)
        return StepResult(
            token_text=token_text,
            entropy_proxy=min(entropy, 0.92),
            precision=precision,
            metadata={"call_count": self.call_count},
        )

    def _next_token_text(self, precision: str, **kwargs: Any) -> str:
        prompt = kwargs.get("prompt", "")
        constraints = kwargs.get("constraints", [])
        generated = kwargs.get("generated", [])
        seeds = self._constraint_seed_tokens(prompt, constraints)
        index = len(generated)
        if index < len(seeds):
            return seeds[index]
        return f"mock_{precision.lower()}_{index}"

    def _constraint_seed_tokens(self, prompt: str, constraints: list[str]) -> list[str]:
        text = " ".join([prompt, *constraints]).lower()
        tokens: list[str] = []
        if "number" in text or "digit" in text:
            tokens.append("1")
        if "date" in text:
            tokens.append("2026-01-01")
        for raw in constraints:
            for word in raw.replace(".", " ").split():
                cleaned = "".join(ch for ch in word if ch.isalnum() or ch in "_-")
                if len(cleaned) > 4 and cleaned.lower() not in {"include", "exactly", "forbidden"}:
                    tokens.append(cleaned)
                    break
        return tokens

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        precision: str = "Q4",
        **kwargs: Any,
    ) -> str:
        self.call_count += max_new_tokens
        base = prompt.strip()[-50:] if len(prompt) > 50 else prompt
        return base + f" [MOCK_{precision.upper()}_OUT_{self.call_count}]"


class TransformersModelRunner:
    """
    Adapter surface for real Hugging Face Transformers + quantization backends.
    v0.2 keeps this optional so default tests do not require torch/transformers.
    When dependencies are present, this can be fleshed out to load
    Q4/Q8/FP16 variants of TensionLM or other models.

    Usage (future):
        runner = TransformersModelRunner.from_pretrained(
            "BoggersTheFish/TensionLM-117M-Curriculum",
            quantizations={"Q4": "path/to/q4", "Q8": "path/to/q8"}
        )
    """

    def __init__(self, name: str = "transformers-adapter"):
        self.name = name
        self._model = None
        self._tokenizer = None
        # Lazy import check
        if importlib.util.find_spec("transformers") is None:
            raise ImportError(
                "TransformersModelRunner requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch --upgrade"
            )

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> "TransformersModelRunner":
        """Factory hook for a future multi-precision Transformers backend."""
        instance = cls(name=f"transformers-{model_id.split('/')[-1]}")
        # In real code: load tokenizer + multiple model variants here
        print(f"[TransformersModelRunner] adapter not wired in v0.2: {model_id}")
        return instance

    def step(self, precision: str = "Q4", **kwargs: Any) -> StepResult:
        raise NotImplementedError(
            "TransformersModelRunner.step() is not implemented in v0.2. "
            "Use MockModelRunner for now or implement the real forward pass."
        )

    def generate(self, prompt: str, max_new_tokens: int = 64, precision: str = "Q4", **kwargs: Any) -> str:
        raise NotImplementedError(
            "TransformersModelRunner.generate() is not implemented in v0.2."
        )
