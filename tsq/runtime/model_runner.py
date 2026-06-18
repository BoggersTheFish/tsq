"""
TS Node: ModelRunner
Type: runtime_abstraction
Description: Protocol + implementations for pluggable model backends.
             Allows the TSQ loop to remain model-agnostic while supporting
             mock (for dev) and real quantized models (Transformers, llama.cpp, etc.).
             Current wave: CLI/report-ready backends with verifier-gated repair.
Tension sources: none in the runner itself (tension lives in scanner + router)
Edges:
  - runner.step(precision) → produces logits-like object with entropy_proxy
  - runner.generate(...) → full continuation at fixed precision (for baselines)
Verifier hooks: none (verifier is downstream of generation)
Receipt outputs: none directly (runner is pure execution surface)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
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


class RepairAwareMockRunner(MockModelRunner):
    """
    Deterministic mock backend that fails cheaply and repairs at escalated precision.

    It exists to test the TSQ repair boundary without torch/transformers.
    """

    def __init__(self, name: str = "repair-aware-mock-runner"):
        super().__init__(name=name)

    def step(self, precision: str = "Q4", **kwargs: Any) -> StepResult:
        if kwargs.get("repair_mode"):
            self.call_count += 1
            token_text = self._repair_token(precision=precision, **kwargs)
            return StepResult(
                token_text=token_text,
                entropy_proxy=0.18,
                precision=precision,
                metadata={
                    "repair_mode": True,
                    "reason": "verification_failure",
                    "precision": precision,
                    "call_count": self.call_count,
                },
            )
        return super().step(precision=precision, **kwargs)

    def _next_token_text(self, precision: str, **kwargs: Any) -> str:
        constraints = " ".join(kwargs.get("constraints", [])).lower()
        index = len(kwargs.get("generated", []))
        if "forbidden" in constraints and index == 0:
            return "forbidden"
        if "code fence" in constraints and index == 0:
            return "```python"
        return f"cheap_{precision.lower()}_{index}"

    def _repair_token(self, precision: str, **kwargs: Any) -> str:
        failures = " ".join(kwargs.get("verification_failures", [])).lower()
        constraints = " ".join(kwargs.get("constraints", [])).lower()
        if precision not in {"Q8", "FP16", "residual_unfolded"}:
            return "repair_low_precision_noop"
        if "missing number/date" in failures or "include 7" in constraints:
            return "7"
        if "unclosed code fence" in failures:
            return "```"
        if "constraint not reflected" in failures:
            for word in constraints.replace(".", " ").split():
                cleaned = "".join(ch for ch in word if ch.isalnum() or ch in "_-")
                if len(cleaned) > 4 and cleaned.lower() not in {"include", "exactly", "forbidden"}:
                    return cleaned
        return "repair_noop"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        precision: str = "Q4",
        **kwargs: Any,
    ) -> str:
        constraints = kwargs.get("constraints", [])
        text = prompt
        if precision in {"Q8", "FP16"}:
            joined = " ".join(constraints).lower()
            if "include 7" in joined:
                text += " 7"
            if "include checksum" in joined:
                text += " checksum"
            if "code fence" in joined:
                text += " ```python ```"
        elif constraints and "forbidden" in " ".join(constraints).lower():
            text += " forbidden"
        return text


class TransformersModelRunner:
    """
    Optional Hugging Face Transformers backend.

    Precision labels are routing metadata unless precision_models maps labels
    to distinct model IDs or paths. This adapter does not implement native TSQ
    quantization.

    Usage:
        runner = TransformersModelRunner.from_pretrained(
            "sshleifer/tiny-gpt2",
            precision_models={"Q4": "path/to/q4", "Q8": "path/to/q8"},
        )
    """

    def __init__(
        self,
        model_id: str,
        precision_models: dict[str, str] | None = None,
        device: str | None = None,
        adapter_dir: str | None = None,
        **kwargs: Any,
    ):
        if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
            raise ImportError(
                "TransformersModelRunner requires optional dependencies 'transformers' and 'torch'. "
                "Install them with: pip install 'tsq[transformers]'"
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"transformers-{model_id.split('/')[-1]}"
        self.model_id = model_id
        self.adapter_dir = adapter_dir
        self.precision_models = {
            "Q4": model_id,
            "Q8": model_id,
            "FP16": model_id,
            "residual_unfolded": model_id,
        }
        if precision_models:
            self.precision_models.update(precision_models)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._models: dict[str, Any] = {}
        loaded_by_model_id: dict[str, Any] = {}
        for precision, mapped_model_id in self.precision_models.items():
            if mapped_model_id not in loaded_by_model_id:
                model = AutoModelForCausalLM.from_pretrained(mapped_model_id, **kwargs)
                if adapter_dir:
                    if importlib.util.find_spec("peft") is None:
                        raise ImportError(
                            "adapter_dir requires optional dependency 'peft'. "
                            "Install training dependencies or omit --adapter-dir."
                        )
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(model, adapter_dir)
                model.to(self.device)
                model.eval()
                loaded_by_model_id[mapped_model_id] = model
            self._models[precision] = loaded_by_model_id[mapped_model_id]

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        precision_models: dict[str, str] | None = None,
        device: str | None = None,
        adapter_dir: str | None = None,
        **kwargs: Any,
    ) -> "TransformersModelRunner":
        return cls(
            model_id=model_id,
            precision_models=precision_models,
            device=device,
            adapter_dir=adapter_dir,
            **kwargs,
        )

    def step(self, precision: str = "Q4", **kwargs: Any) -> StepResult:
        model, selected_model_id = self._select_model(precision)
        text = self._build_context(**kwargs)
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        metadata = {
            "backend": "transformers",
            "model_id": selected_model_id,
            "adapter_dir": self.adapter_dir,
            "repair_mode": bool(kwargs.get("repair_mode")),
            "device": self.device,
        }
        with self._torch.no_grad():
            outputs = model(**inputs)
            next_logits = outputs.logits[:, -1, :]
            next_token_id = self._torch.argmax(next_logits, dim=-1)
        token_text = self._tokenizer.decode(
            next_token_id.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        entropy_proxy, entropy_metadata = self._entropy_proxy(next_logits)
        metadata.update(entropy_metadata)
        return StepResult(
            token_text=token_text,
            entropy_proxy=entropy_proxy,
            precision=precision,
            metadata=metadata,
        )

    def generate(self, prompt: str, max_new_tokens: int = 64, precision: str = "Q4", **kwargs: Any) -> str:
        model, _selected_model_id = self._select_model(precision)
        text = self._build_context(prompt=prompt, **kwargs)
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }
        with self._torch.no_grad():
            output_ids = model.generate(**inputs, **generate_kwargs)
        return self._tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _select_model(self, precision: str) -> tuple[Any, str]:
        selected_model_id = self.precision_models.get(precision, self.model_id)
        model = self._models.get(precision)
        if model is None:
            model = self._models.get("Q4") or next(iter(self._models.values()))
        return model, selected_model_id

    def _build_context(self, **kwargs: Any) -> str:
        prompt = kwargs.get("prompt", "")
        generated = kwargs.get("generated", [])
        if kwargs.get("repair_mode"):
            constraints = kwargs.get("constraints", [])
            failures = kwargs.get("verification_failures", [])
            current_output = kwargs.get("current_output", "")
            return (
                "Repair the output to satisfy these constraints.\n"
                f"Constraints: {constraints}\n"
                f"Verification failures: {failures}\n"
                f"Current output: {current_output}\n"
                "Repair:"
            )
        return " ".join([prompt.strip(), *generated]).strip()

    def _entropy_proxy(self, next_logits: Any) -> tuple[float, dict[str, Any]]:
        try:
            probs = self._torch.softmax(next_logits.float(), dim=-1)
            log_probs = self._torch.log(probs.clamp_min(1e-12))
            entropy = -(probs * log_probs).sum(dim=-1).item()
            vocab_size = next_logits.shape[-1]
            normalized = entropy / math.log(max(vocab_size, 2))
            return max(0.0, min(1.0, float(normalized))), {"entropy_source": "next_token_logits"}
        except Exception as exc:
            return 0.3, {"entropy_source": "fallback", "entropy_error": str(exc)}
