# TSQ Cost Accounting

TSQ includes estimated routing-cost accounting. This is not measured GPU energy, wall-clock cost, or native quantization performance.

## Default Cost Model

The default cost weights are:

- `Q4`: 1.0
- `Q8`: 2.0
- `FP16`: 4.0
- `residual_unfolded`: 4.0

`estimated_cost_units` is:

```text
sum(tokens_at_precision * precision_cost_weight)
```

The default model name in reports is `default_estimated_routing_cost_v0.7`.

## Token Counts

Generation stats include:

- `tokens_at_Q4`
- `tokens_at_Q8`
- `tokens_at_FP16`
- `tokens_at_residual_unfolded`
- `repair_tokens_generated`
- `total_tokens_generated`
- `estimated_cost_units`
- `cost_model_used`

Counts are based on backend-owned `StepResult.precision`, not on verifier assumptions.

## Precision Histograms

Reports include:

- count per precision
- repair-token count per precision
- escalation count
- compute receipt count

This makes routing behavior visible without reading every tension sample.

## Interpreting Ratios

`dynamic_vs_q8_cost_ratio` compares TSQ dynamic estimated cost with the fixed `always_Q8` baseline for the same token budget.

- Below 1.0 means dynamic routing spent fewer estimated cost units than fixed Q8.
- Around 1.0 means dynamic routing cost roughly matched fixed Q8.
- Above 1.0 means repair or escalation made dynamic routing more expensive than fixed Q8.

The ratio only measures estimated routing cost. It does not prove real hardware savings unless precision labels are mapped to real backend variants and measured separately.
