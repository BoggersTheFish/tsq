# TSQ v0.9 Adapter Experiment

v0.9 makes the first adapter experiment reproducible. It does not commit model weights or claim adapter quality without a real report.

## Install Training Dependencies

Standard LoRA:

```bash
pip install -e '.[training]'
```

QLoRA add-on:

```bash
pip install -e '.[training,qlora]'
```

Default CI does not install these dependencies.

## Check Training Environment

```bash
python scripts/check_training_env.py --no-fail
```

The checker reports Python version, optional dependency availability, CUDA availability, GPU name when available, and recommended install commands.

## Dry-Run Experiment

```bash
python scripts/run_v09_experiment.py \
  --dry-run \
  --model-id dry-run-model
```

Dry-run builds command intent, validates datasets through the dry-run train/eval paths, and writes:

- `artifacts/experiments/v09/experiment_manifest.json`
- `artifacts/reports/tsq_lora_eval_v09.json`

No model is loaded in dry-run mode.

## Smoke Experiment

```bash
python scripts/run_v09_experiment.py \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --smoke-train \
  --max-steps 2
```

Smoke training is a wiring proof. It is not a meaningful quality run.

## Full Experiment

```bash
python scripts/run_v09_experiment.py \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --training-mode mixed \
  --max-steps 50 \
  --output-dir artifacts/models/tsq-lora-v09 \
  --report-dir artifacts/reports \
  --experiment-dir artifacts/experiments/v09
```

The runner orchestrates:

1. Dataset build.
2. Dataset validation.
3. LoRA adapter training.
4. Base-vs-adapter TSQ eval.
5. Experiment manifest writing.

If training dependencies are missing, the manifest records `training_status: skipped_missing_dependencies`. It does not fake training results.

## Summarize Results

```bash
python scripts/summarize_experiment.py \
  --manifest artifacts/experiments/v09/experiment_manifest.json \
  --eval-report artifacts/reports/tsq_lora_eval_v09.json \
  --markdown artifacts/experiments/v09/summary.md
```

The summary includes:

- model id
- adapter directory
- training and eval status
- base dynamic passes
- adapter dynamic passes
- base and adapter mean dynamic cost
- whether pass rate or cost improved
- limitation notes

## Limitations

- The seed dataset is tiny.
- Smoke training only proves wiring.
- Meaningful model quality requires larger data and more steps.
- No native TSQ quantization exists yet.
- Precision labels are routing labels unless backed by distinct model variants.
