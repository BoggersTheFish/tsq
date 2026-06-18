# LoRA And QLoRA Fine-Tuning

`scripts/train_lora.py` trains a PEFT LoRA adapter from TSQ JSONL datasets when optional ML dependencies are installed.

## Real Training Command

```bash
python scripts/train_lora.py \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --train-jsonl data/generated/tsq_supervised_train.jsonl \
  --eval-jsonl data/generated/tsq_supervised_eval.jsonl \
  --output-dir artifacts/models/tsq-lora-v08 \
  --max-steps 50 \
  --learning-rate 2e-4 \
  --lora-r 8
```

Expected artifacts include:

- PEFT adapter files
- tokenizer files
- `tsq_training_metadata.json`
- Trainer checkpoint files if save steps are reached

## QLoRA-Friendly Flags

Supported flags:

- `--load-in-4bit`
- `--load-in-8bit`
- `--bf16`
- `--fp16`
- `--gradient-checkpointing`
- `--max-seq-length`
- `--per-device-train-batch-size`
- `--gradient-accumulation-steps`
- `--warmup-steps`
- `--save-steps`
- `--eval-steps`
- `--logging-steps`

If 4-bit or 8-bit loading is requested without `bitsandbytes`, the script fails with a clear error.

## Prompt Format

Supervised examples are formatted as:

```text
<System>
You are a TSQ-aware assistant. Follow explicit constraints and preserve verifier-critical details.

<User>
{input_text}

<Assistant>
{target_text}
```

Repair examples are formatted as:

```text
<System>
You are a TSQ repair model. Repair failed outputs using verifier feedback.

<User>
{repair_prompt}

<Assistant>
{repair_target}
```

## Current Expectations

The seed dataset is intentionally small. A first adapter may demonstrate that the loop works without showing strong quality gains. Claims should be based on TSQ eval-suite reports, verifier pass rates, repair attempts, and estimated routing cost.
