# TSQ Evals

TSQ evals are small, deterministic fixtures for checking routing behavior, verifier repair, and report integrity. They are not a training dataset yet.

## Baseline Isolation

`compare_baselines()` accepts `backend_factory`. Each run receives a fresh backend:

- `always_Q4`
- `always_Q8`
- `TSQ_dynamic`

This prevents stateful mock runners, cached model state, or call counters from contaminating comparisons.

## Task Schema

Built-in suite tasks live in `tsq/evals/tasks.py` and use this shape:

```json
{
  "name": "short_answer_required_number",
  "prompt": "Short answer only.",
  "constraints": ["include 7"],
  "max_new_tokens": 1,
  "expected_behavior": "short output includes the required number",
  "repairable": true
}
```

The built-in suite covers number preservation, date preservation, forbidden word avoidance, code fence closure, format compliance, multi-constraint prompts, repairable missing constraints, non-repairable forbidden output, exact-word requirements, short answers with numbers, negation, and simple code requirements.

## Running The Suite

```bash
python -m tsq.cli eval-suite \
  --backend repair-mock \
  --report artifacts/eval_suite_report.json
```

The aggregate reports pass counts, repair counts, compute receipt totals, mean estimated dynamic cost, and mean dynamic-vs-Q8 cost ratio.

## Future Fine-Tuning Data

The report structure intentionally keeps task metadata, baseline outputs, dynamic outputs, verifier failures, repair behavior, tension samples, and receipts together. That makes the next wave able to extract supervised repair examples or routing-policy traces without changing the report shape first.
