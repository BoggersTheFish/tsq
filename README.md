# TSQ — Tension-Structured Quantization

**Verifier-Gated, Tension-Driven Inference Runtime for Thinking System (TS) models**

TSQ treats inference compute as a *structural* problem: precision should follow *unresolved tension*, not be applied uniformly.

- Cheap by default (Q4 / low-cost mode)
- Near-zero-cost tension signals in the hot path
- Verifier checks constraint satisfaction
- Escalation to higher precision (Q8 / FP16) only on detected tension or verifier failure
- Explicit **cognitive receipts** (accepted meaning) and **compute receipts** (why extra work was spent)
- Long-term: receipt-backed context compression, residual unfolding, attractor codebooks

This is the **inference/runtime layer** complement to [TensionLM](https://github.com/BoggersTheFish/TensionLM) (the model architecture layer using sigmoid tension attention).

**Status**: v0.9 — First reproducible adapter experiment harness. TSQ can now check the training environment, run a tiny LoRA/QLoRA adapter experiment, evaluate base vs adapter through TSQ, and write an experiment manifest/summary.
Mock/backend-runtime demo + full test suite pass. Optional Transformers support, adapter loading, verifier-gated repair, backend-owned token generation, receipts, cost accounting, eval suites, training data generation, and LoRA training remain active.
Still cheap-first: only entropy proxy + lexical risk + verifier failure in the hot path.
No custom quantization implemented yet.
No committed model artifacts or strong adapter quality claim are included.

## Core Thesis (TS Native)
> Precision should follow unresolved tension.  
> High precision is an *exception handler*, not the default mode.

The architecture roadmap lives in this README for the current wave; deeper design notes can be added under `docs/` as the runtime grows.

## Quick Start
```bash
python -c "
from tsq.runtime.generation_loop import run_tsq_generation
from tsq.runtime.model_runner import MockModelRunner
result = run_tsq_generation(
    prompt='Brainstorm tension-aware inference. Must include one number.',
    constraints=['include one number'],
    max_new_tokens=20,
    model=MockModelRunner()
)
print(result['stats'])
print('Cognitive receipts:', len(result['cognitive_receipts']))
print('Compute receipts:', len(result['compute_receipts']))
"
```

## CLI Quick Start

Mock generation with receipts and a JSON report:
```bash
python -m tsq.cli generate \
  --prompt "Brainstorm TSQ. Must include one number." \
  --constraint "include one number" \
  --max-new-tokens 20 \
  --backend mock \
  --receipts artifacts/receipts.jsonl \
  --report artifacts/run_report.json
```

Mock baseline comparison:
```bash
python -m tsq.cli eval \
  --prompt "Explain TSQ with one number." \
  --constraint "include one number" \
  --max-new-tokens 20 \
  --backend mock \
  --report artifacts/eval_report.json
```

Built-in repair eval:
```bash
python -m tsq.cli repair-eval \
  --backend repair-mock \
  --report artifacts/repair_eval_report.json
```

Built-in eval suite:
```bash
python -m tsq.cli eval-suite \
  --backend repair-mock \
  --report artifacts/eval_suite_report.json
```

Optional Transformers backend:
```bash
pip install -e '.[transformers]'
python -m tsq.cli generate \
  --prompt "Explain TSQ briefly." \
  --max-new-tokens 16 \
  --backend transformers \
  --model-id sshleifer/tiny-gpt2 \
  --report artifacts/transformers_report.json
```

Precision labels are routing labels unless mapped to distinct backend models with `--q4-model`, `--q8-model`, and `--fp16-model`.

## Evidence

Run the built-in suite:
```bash
python -m tsq.cli eval-suite --backend repair-mock --report artifacts/eval_suite_report.json
```

Example reports live under `examples/reports/`.

The core eval/cost metrics are:

- verifier pass rates for `always_Q4`, `always_Q8`, and `TSQ_dynamic`
- `estimated_cost_units`
- `dynamic_vs_q8_cost_ratio`
- precision histograms, including repair-token counts
- compute receipts explaining why routing escalated

Cost is estimated routing cost, not measured GPU energy. The default model is Q4 = 1.0, Q8 = 2.0, FP16 = 4.0, and residual unfolding = 4.0.

Current limitations remain explicit: no native TSQ quantization yet, seed data is tiny, adapter quality depends on the chosen model and run, and precision labels are routing labels unless mapped to distinct backend model variants. Default CI is lightweight and does not install heavy ML dependencies.

## Training Data

Build the seed datasets:
```bash
python -m tsq.cli build-dataset \
  --out-dir data/generated \
  --include-example-reports examples/reports
```

Validate a dataset:
```bash
python -m tsq.cli validate-dataset \
  --path data/generated/tsq_supervised_train.jsonl
```

Summarize a dataset:
```bash
python -m tsq.cli dataset-summary \
  --path data/generated/tsq_supervised_train.jsonl
```

## Training

Check training environment:
```bash
python scripts/check_training_env.py --no-fail
```

Dry-run training validation:
```bash
python scripts/train_lora.py \
  --model-id dry-run-model \
  --train-jsonl data/generated/tsq_supervised_train.jsonl \
  --eval-jsonl data/generated/tsq_supervised_eval.jsonl \
  --output-dir artifacts/models/tsq-lora-v09 \
  --dry-run
```

Smoke-train wiring check:
```bash
python scripts/train_lora.py \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --train-jsonl data/generated/tsq_supervised_train.jsonl \
  --eval-jsonl data/generated/tsq_supervised_eval.jsonl \
  --output-dir artifacts/models/tsq-lora-v09-smoke \
  --smoke-train \
  --max-steps 2
```

Real LoRA train command:
```bash
python scripts/train_lora.py \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --train-jsonl data/generated/tsq_supervised_train.jsonl \
  --eval-jsonl data/generated/tsq_supervised_eval.jsonl \
  --output-dir artifacts/models/tsq-lora-v09 \
  --max-steps 50 \
  --learning-rate 2e-4 \
  --lora-r 8
```

Evaluate an adapter:
```bash
python scripts/eval_lora.py \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --adapter-dir artifacts/models/tsq-lora-v09 \
  --report artifacts/reports/tsq_lora_eval_v09.json \
  --max-new-tokens 32
```

Heavy training dependencies are optional. The seed dataset is small and proves the loop; it is not expected to produce a strong model by itself.

## Adapter Experiment

Dry-run the full experiment harness:
```bash
python scripts/run_v09_experiment.py \
  --dry-run \
  --model-id dry-run-model
```

Run a tiny smoke experiment when optional ML dependencies and a small model are available:
```bash
python scripts/run_v09_experiment.py \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --smoke-train \
  --max-steps 2
```

Summarize an experiment:
```bash
python scripts/summarize_experiment.py \
  --manifest artifacts/experiments/v09/experiment_manifest.json \
  --eval-report artifacts/reports/tsq_lora_eval_v09.json \
  --markdown artifacts/experiments/v09/summary.md
```

The experiment harness writes a manifest with commands, dependency status, training status, eval status, dataset files, and notes. If dependencies are missing, training is recorded as skipped rather than faked.

## Repo Structure
```
tsq/
├── tsq/
│   ├── runtime/     # generation_loop, precision_router, model_runner
│   ├── tension/     # scanner (cheap signals)
│   ├── verifier/    # base + cheap checkers
│   ├── receipts/    # schema + store (JSONL)
│   ├── evals/       # harness for baseline vs dynamic comparison
│   ├── training/    # JSONL schemas, builders, validators
├── tests/
├── docs/
└── pyproject.toml
```

## TS Integration
- **Tension Plane** → extends TensionLM's τ (sigmoid tension) concept from weights/attention into live generation-time signals.
- **Receipts** → first-class verifiable traces (aligns with CIG claim/evidence graphs + TS-Reasoner replay).
- **Verifier** → constraint relaxation engine (TS-Reasoner style policy contracts + bounded reasoning).
- **Routing** → tension propagation → activation of higher-resolution "structures" (precision levels as resolution nodes).
- Every component declares explicit **TS headers** (nodes, tension sources, verifier hooks, receipt outputs).

**Wave Goal**: v0.9 makes the first adapter experiment reproducible: check deps, build data, train when possible, evaluate base vs adapter, write a manifest, and summarize the evidence.
Any compliant ModelRunner can be dropped in. The system demonstrates verifier-gated dynamic precision with bounded repair passes, isolated baseline comparisons, estimated routing-cost accounting, persistent receipts, inspectable reports, seed training datasets, adapter-aware Transformers evaluation, and experiment manifests while keeping default CI lightweight.
See `docs/transformers_backend.md` for the optional real-model adapter.
See `docs/reports.md` for report schemas and interpretation.
See `docs/evals.md` for eval suite fixtures.
See `docs/cost_accounting.md` for routing-cost semantics.
See `docs/training_data.md` for dataset schemas and generation.
See `docs/training.md` and `docs/lora_finetune.md` for the training path.
See `docs/model_eval.md` for adapter evaluation.
See `docs/fine_tuning_plan.md` for quality criteria and next steps.
See `docs/v09_experiment.md` for the reproducible adapter experiment harness.

---

*Part of the BoggersTheFish Thinking System (TS) ecosystem — constraint graphs, wave propagation, verifiable receipts, tension-native everything.*
