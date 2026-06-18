# TSQ Reports

TSQ writes reproducible JSON reports from the CLI. Reports are plain JSON and do not contain Python objects.

## Example Commands

Single generation:

```bash
python -m tsq.cli generate \
  --prompt "Brainstorm TSQ. Must include one number." \
  --constraint "include one number" \
  --max-new-tokens 20 \
  --backend mock \
  --receipts artifacts/receipts.jsonl \
  --report artifacts/run_report.json
```

Baseline comparison:

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

## Generation Report

`generate` reports include:

- `mode`: `"generate"`
- `backend`
- `model`
- `prompt`
- `constraints`
- `output`
- `stats`
- `verification`
- `original_verification`
- `final_verification`
- `cognitive_receipts`
- `compute_receipts`
- `tension_samples`
- `precision_histogram`
- `tsq_version`
- `created_at`

Receipts and verifier results are serialized as dictionaries. Dataclasses are converted before JSON is written.

`stats` also includes routing-cost fields:

- `tokens_at_Q4`
- `tokens_at_Q8`
- `tokens_at_FP16`
- `tokens_at_residual_unfolded`
- `total_tokens_generated`
- `estimated_cost_units`
- `cost_model_used`

## Eval Report

`eval` reports include `always_Q4`, `always_Q8`, and `TSQ_dynamic`.

The compact `summary` contains:

- `dynamic_passed`
- `dynamic_repaired`
- `dynamic_escalations`
- `q4_passed`
- `q8_passed`
- `dynamic_receipts`
- `always_Q4_estimated_cost`
- `always_Q8_estimated_cost`
- `TSQ_dynamic_estimated_cost`
- `dynamic_vs_q8_cost_ratio`
- `dynamic_vs_fp16_cost_ratio`
- `precision_histogram`

Interpretation:

- `always_Q4` and `always_Q8` use fixed precision labels through `ModelRunner.generate()`.
- `TSQ_dynamic` uses the live generation loop with tension scanning, precision routing, verification, repair, and receipts.
- A dynamic run can fail initially, repair, and still finish with `final_verifier_pass: true`.

Precision labels are routing labels unless they are mapped to distinct backend model variants. TSQ does not implement native TSQ quantization yet.

## Precision Histogram

Reports include a compact precision histogram:

```json
{
  "counts": {"Q4": 4, "Q8": 2, "FP16": 0, "residual_unfolded": 0},
  "repair_tokens": {"Q4": 0, "Q8": 1, "FP16": 0, "residual_unfolded": 0},
  "escalation_count": 1,
  "compute_receipt_count": 1
}
```

`counts` is based on the backend-owned `StepResult.precision` for generated tokens. `repair_tokens` is a subset showing how many generated tokens happened during repair mode.

## Repair Eval Report

`repair-eval` runs built-in constraint-heavy tasks and writes:

- `tasks`: per-task output, stats, verification states, receipts, and tension samples
- `aggregate.total_tasks`
- `aggregate.dynamic_passes`
- `aggregate.repair_attempts`
- `aggregate.repair_successes`
- `aggregate.total_compute_receipts`
- `aggregate.total_estimated_cost`
- `aggregate.precision_histogram`

## Eval Suite Report

`eval-suite` runs the built-in v0.6 fixture list and writes:

- `tasks`: per-task baseline comparison results plus task metadata
- `aggregate.total_tasks`
- `aggregate.q4_passes`
- `aggregate.q8_passes`
- `aggregate.dynamic_passes`
- `aggregate.dynamic_repairs_attempted`
- `aggregate.dynamic_repairs_succeeded`
- `aggregate.total_dynamic_compute_receipts`
- `aggregate.mean_dynamic_vs_q8_cost_ratio`
- `aggregate.mean_dynamic_estimated_cost`
- `aggregate.precision_histogram`

## Inspecting Compute Receipts

Compute receipts explain why extra work was spent. The common repair receipt reason is:

```json
"reason": "verification_failed_repair"
```

Inspect `compute_receipts[*].tension`, `from_precision`, `to_precision`, and `metadata.verification_failures` to understand the repair trigger.
