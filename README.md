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

**Status**: v0.1 — Real-model-ready runtime (ModelRunner abstraction + receipt persistence + eval harness).
Mock demo + full test suite pass. TransformersModelRunner is a ready stub (not yet wired to real weights).
Still cheap-first: only entropy proxy + lexical risk + verifier failure in hot path.
No custom quantization implemented yet.

## Core Thesis (TS Native)
> Precision should follow unresolved tension.  
> High precision is an *exception handler*, not the default mode.

See `docs/TSQ_Full_Plan.md` for the complete architecture, roadmap (v0–v5), MVP components, risks, and success criteria.

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

## Repo Structure
```
ts q/
├── tsq/
│   ├── runtime/     # generation_loop, precision_router, model_runner
│   ├── tension/     # scanner (cheap signals)
│   ├── verifier/    # base + cheap checkers
│   ├── receipts/    # schema + store (JSONL)
│   ├── evals/       # harness for baseline vs dynamic comparison
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

**Wave Goal**: v0.1 is real-model-ready. Any compliant ModelRunner can be dropped in.
The system demonstrates verifier-gated dynamic precision with persistent, inspectable receipts while keeping the hot path strictly cheap.

---

*Part of the BoggersTheFish Thinking System (TS) ecosystem — constraint graphs, wave propagation, verifiable receipts, tension-native everything.*