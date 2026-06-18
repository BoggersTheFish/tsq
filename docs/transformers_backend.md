# TSQ Transformers Backend

TSQ includes an optional Hugging Face Transformers backend adapter through `TransformersModelRunner`.

## What This Backend Supports

- Real causal language model execution with `transformers` and `torch`.
- Backend-owned token generation through `StepResult`.
- One-token `step()` calls for the dynamic TSQ loop.
- Full `generate()` calls for fixed-precision eval baselines.
- Verifier repair mode by wrapping the current output, failures, and constraints into an inspectable repair prompt.
- Optional precision label mapping:

```python
from tsq.runtime.model_runner import TransformersModelRunner

runner = TransformersModelRunner.from_pretrained(
    "sshleifer/tiny-gpt2",
    precision_models={
        "Q4": "path-or-model-id-for-q4",
        "Q8": "path-or-model-id-for-q8",
        "FP16": "path-or-model-id-for-fp16",
    },
)
```

If `precision_models` is omitted, TSQ loads `model_id` once and reuses it for all precision labels.

## What This Backend Does Not Support

- No native TSQ quantization.
- No custom Q4/Q8 conversion.
- No guarantee that precision labels change numerical precision unless you provide distinct backend model variants.
- No mandatory Transformers dependency for default installs or default tests.

Dynamic precision routing can choose labels such as `Q4`, `Q8`, and `FP16`, but actual model precision depends on the model IDs or paths supplied to the backend.

## Installation

Base TSQ remains lightweight:

```bash
pip install -e .
```

Install optional real-model dependencies with:

```bash
pip install -e '.[transformers]'
```

## Running With a Tiny Model

```python
from tsq.runtime.generation_loop import run_tsq_generation
from tsq.runtime.model_runner import TransformersModelRunner

runner = TransformersModelRunner.from_pretrained("sshleifer/tiny-gpt2")

result = run_tsq_generation(
    prompt="Say hello in one short sentence.",
    constraints=[],
    max_new_tokens=8,
    model=runner,
)

print(result["output"])
print(result["stats"])
```

## Optional Integration Test

Default pytest remains mock-only. To run the optional real backend test:

```bash
TSQ_RUN_TRANSFORMERS_TESTS=1 \
TSQ_TEST_MODEL_ID=sshleifer/tiny-gpt2 \
python -m pytest
```

The test skips cleanly unless the environment variable is set, a model ID is provided, and optional dependencies are installed.

## Repair Mode

When `run_tsq_generation()` enters repair mode, it passes:

- `repair_mode=True`
- `verification_failures`
- `current_output`
- `constraints`

The Transformers backend does not hard-code repair rules. It builds a simple repair prompt from those fields and lets the selected model emit the repair token.
