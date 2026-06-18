# TSQ Fine-Tuning Plan

v0.7 is data preparation only. It creates schemas, deterministic JSONL generation, validation tools, and a dry-run training script scaffold. It does not run a heavy training job and does not publish a fine-tuned TSQ model.

## v0.8 Target

The next wave should run a small LoRA or QLoRA fine-tune using the v0.7 datasets.

Recommended small base models for the first pass:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

The exact model should be chosen based on available local hardware and licensing constraints.

## Training Inputs

Initial training should use:

- supervised constraint-following examples
- repair-from-verifier-feedback examples
- optionally preference pairs after a DPO or preference trainer is added

The dry-run scaffold is:

```bash
python scripts/train_lora.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --train-jsonl data/generated/tsq_supervised_train.jsonl \
  --eval-jsonl data/generated/tsq_supervised_eval.jsonl \
  --output-dir artifacts/tsq-lora \
  --dry-run
```

Real training will require optional dependencies: `torch`, `transformers`, `datasets`, `peft`, and `accelerate`.

## Evaluation Through TSQ

The fine-tuned model should be evaluated by plugging it into TSQ as a backend model variant and running:

- `eval-suite`
- repair evals
- cost-accounting reports
- verifier pass-rate comparisons against the base model

## Success Criteria

A useful TSQ-aware fine-tune should show:

- higher verifier pass rate
- fewer repair attempts
- lower estimated routing cost for the same pass rate
- better number/date preservation
- better forbidden-word avoidance
- better code fence and format closure

These are runtime/eval claims. Real precision or energy claims still require backend variants and hardware measurements.
