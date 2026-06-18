# TSQ Fine-Tuning Plan

v0.9 provides the first executable LoRA/QLoRA fine-tuning path. It wires dataset loading, formatting, tokenization, PEFT LoRA configuration, Trainer execution, adapter saving, and adapter evaluation through TSQ.

The seed datasets are intentionally small. They prove the loop; they do not by themselves justify a strong quality claim.

## Recommended First Runs

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

Dry-run:

```bash
python scripts/train_lora.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --train-jsonl data/generated/tsq_supervised_train.jsonl \
  --eval-jsonl data/generated/tsq_supervised_eval.jsonl \
  --output-dir artifacts/tsq-lora \
  --dry-run
```

Real training will require optional dependencies: `torch`, `transformers`, `datasets`, `peft`, and `accelerate`.
QLoRA flags also require `bitsandbytes`.

## Evaluation Through TSQ

The fine-tuned adapter should be evaluated by plugging it into TSQ as an adapter-aware Transformers backend and running:

- `eval-suite`
- repair evals
- cost-accounting reports
- verifier pass-rate comparisons against the base model

Use `scripts/eval_lora.py` for base-vs-adapter reports, or pass `--adapter-dir` directly to the TSQ CLI Transformers backend.

## Success Criteria

A useful TSQ-aware fine-tune should show:

- higher verifier pass rate
- fewer repair attempts
- lower estimated routing cost for the same pass rate
- better number/date preservation
- better forbidden-word avoidance
- better code fence and format closure

These are runtime/eval claims. Real precision or energy claims still require backend variants and hardware measurements.
