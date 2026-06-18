# TSQ Reports

TSQ v0.5 writes reproducible JSON reports from the CLI. Reports are plain JSON and do not contain Python objects.

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
- `tsq_version`
- `created_at`

Receipts and verifier results are serialized as dictionaries. Dataclasses are converted before JSON is written.

## Eval Report

`eval` reports include `always_Q4`, `always_Q8`, and `TSQ_dynamic`.

The compact `summary` contains:

- `dynamic_passed`
- `dynamic_repaired`
- `dynamic_escalations`
- `q4_passed`
- `q8_passed`
- `dynamic_receipts`

Interpretation:

- `always_Q4` and `always_Q8` use fixed precision labels through `ModelRunner.generate()`.
- `TSQ_dynamic` uses the live generation loop with tension scanning, precision routing, verification, repair, and receipts.
- A dynamic run can fail initially, repair, and still finish with `final_verifier_pass: true`.

Precision labels are routing labels unless they are mapped to distinct backend model variants. TSQ v0.5 does not implement native TSQ quantization.

## Repair Eval Report

`repair-eval` runs built-in constraint-heavy tasks and writes:

- `tasks`: per-task output, stats, verification states, receipts, and tension samples
- `aggregate.total_tasks`
- `aggregate.dynamic_passes`
- `aggregate.repair_attempts`
- `aggregate.repair_successes`
- `aggregate.total_compute_receipts`

## Inspecting Compute Receipts

Compute receipts explain why extra work was spent. In v0.5 the common repair receipt reason is:

```json
"reason": "verification_failed_repair"
```

Inspect `compute_receipts[*].tension`, `from_precision`, `to_precision`, and `metadata.verification_failures` to understand the repair trigger.
