# TSQ Training

TSQ v0.8 provides the first real LoRA/QLoRA fine-tuning path. Heavy ML dependencies are optional and are not installed by default CI.

## Install Optional Dependencies

For standard LoRA:

```bash
pip install -e '.[training]'
```

For QLoRA-style 4-bit or 8-bit loading:

```bash
pip install -e '.[qlora]'
```

`bitsandbytes` is only required when using `--load-in-4bit` or `--load-in-8bit`.

## Build Data

```bash
python -m tsq.cli build-dataset \
  --out-dir data/generated \
  --include-example-reports examples/reports
```

## Dry Run

```bash
python scripts/train_lora.py \
  --model-id dry-run-model \
  --train-jsonl data/generated/tsq_supervised_train.jsonl \
  --eval-jsonl data/generated/tsq_supervised_eval.jsonl \
  --output-dir artifacts/models/tsq-lora-v08 \
  --dry-run
```

Dry-run validates and formats datasets, then exits before importing `torch`, `transformers`, `datasets`, `peft`, or `accelerate`.

## Smoke Train

```bash
python scripts/train_lora.py \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --train-jsonl data/generated/tsq_supervised_train.jsonl \
  --eval-jsonl data/generated/tsq_supervised_eval.jsonl \
  --output-dir artifacts/models/tsq-lora-v08-smoke \
  --smoke-train \
  --max-steps 2
```

Smoke mode limits examples and is intended to verify trainer wiring, not quality.

## Training Modes

- `supervised`: train on `input_text -> target_text` examples.
- `repair`: train on repair prompts and repair targets.
- `mixed`: combine supervised examples with `--repair-jsonl`.

Mixed example:

```bash
python scripts/train_lora.py \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --train-jsonl data/generated/tsq_supervised_train.jsonl \
  --eval-jsonl data/generated/tsq_supervised_eval.jsonl \
  --repair-jsonl data/generated/tsq_repair_train.jsonl \
  --training-mode mixed \
  --output-dir artifacts/models/tsq-lora-v08
```

## Limitations

The checked-in seed datasets are tiny and mainly prove the loop. Actual quality depends on dataset size, base model choice, hyperparameters, and evaluation through TSQ. v0.8 does not add native TSQ quantization.
