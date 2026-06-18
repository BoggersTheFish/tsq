# Adapter Evaluation

TSQ v0.9 can evaluate a base Transformers model and a PEFT adapter through the TSQ runtime.

## Check-Only

```bash
python scripts/eval_lora.py \
  --model-id dry-run-model \
  --adapter-dir artifacts/models/tsq-lora-v09 \
  --report artifacts/reports/tsq_lora_eval_check.json \
  --dry-run
```

This writes a small report and exits without loading heavy ML dependencies.

## Adapter Eval

```bash
python scripts/eval_lora.py \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --adapter-dir artifacts/models/tsq-lora-v09 \
  --report artifacts/reports/tsq_lora_eval_v09.json \
  --max-new-tokens 32
```

The script compares:

- base model through TSQ eval-suite
- adapter-loaded model through TSQ eval-suite
- dynamic TSQ routing metrics for each run

## CLI Adapter Eval

The standard TSQ CLI also accepts `--adapter-dir` for Transformers backends:

```bash
python -m tsq.cli eval-suite \
  --backend transformers \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --adapter-dir artifacts/models/tsq-lora-v09 \
  --report artifacts/reports/adapter_eval_suite.json
```

## Metrics To Watch

- verifier pass rate
- repair attempts
- repair successes
- estimated routing cost
- dynamic-vs-Q8 cost ratio
- compute receipts and verifier failures

No native TSQ quantization is implemented yet. Precision labels remain routing labels unless backed by distinct model variants.
