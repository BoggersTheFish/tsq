"""
TS Node: ModelRunner
Type: runtime_abstraction
Description: Protocol + implementations for pluggable model backends.
             Allows the TSQ loop to remain model-agnostic while supporting
             mock (for dev) and real quantized models (Transformers, llama.cpp, etc.).
             v0.1: real-model-ready but does NOT implement custom quantization.
Tension sources: none in the runner itself (tension lives in scanner + router)
Edges:
  - runner.step(precision) → produces logits-like object with entropy_proxy
  - runner.generate(...) → full continuation at fixed precision (for baselines)
Verifier hooks: none (verifier is downstream of generation)
Receipt outputs: none directly (runner is pure execution surface)
"""

from __future__ import annotations
from typing import Any, Protocol, runtime_checkable, Optional
import importlib.util


@runtime_checkable
class ModelRunner(Protocol):
    """Minimal protocol that any TSQ-compatible model runner must satisfy."""

    name: str

    def step(self, precision: str = "Q4", **kwargs: Any) -> Any:
        """
        Perform one forward step and return an object that at minimum
        has an .entropy_proxy attribute (float 0.0-1.0) for the tension scanner.
        Real implementations can return full logits / past_key_values etc.
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

    def step(self, precision: str = "Q4", **kwargs: Any) -> Any:
        self.call_count += 1

        class FakeLogits:
            def __init__(self, entropy_proxy: float):
                self.entropy_proxy = entropy_proxy
                self.precision = precision

        # Simulate harder steps having higher entropy (proxy for tension)
        entropy = 0.22 + (self.call_count % 9) * 0.075
        return FakeLogits(entropy_proxy=min(entropy, 0.92))

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
    v0.1 keeps this optional so default tests do not require torch/transformers.
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
        print(f"[TransformersModelRunner] adapter not wired in v0.1: {model_id}")
        return instance

    def step(self, precision: str = "Q4", **kwargs: Any) -> Any:
        raise NotImplementedError(
            "TransformersModelRunner.step() is not implemented in v0.1. "
            "Use MockModelRunner for now or implement the real forward pass."
        )

    def generate(self, prompt: str, max_new_tokens: int = 64, precision: str = "Q4", **kwargs: Any) -> str:
        raise NotImplementedError(
            "TransformersModelRunner.generate() is not implemented in v0.1."
        )
