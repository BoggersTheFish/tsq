from tsq.evals.harness import compare_baselines
from tsq.receipts.schema import make_cognitive_receipt
from tsq.receipts.store import ReceiptStore
from tsq.runtime.generation_loop import run_tsq_generation
from tsq.runtime.model_runner import MockModelRunner, StepResult
from tsq.runtime.precision_router import PrecisionRouter
from tsq.tension.scanner import scan_recent_window
from tsq.verifier.base import Verifier


class KnownTokenRunner:
    name = "known-token-runner"

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.call_count = 0

    def step(self, precision="Q4", **kwargs):
        index = self.call_count
        self.call_count += 1
        return StepResult(
            token_text=self.tokens[index],
            entropy_proxy=0.2,
            precision=precision,
            metadata={"index": index},
        )

    def generate(self, prompt, max_new_tokens=64, precision="Q4", **kwargs):
        return prompt + " " + " ".join(self.tokens[:max_new_tokens])


def test_scanner_returns_bounded_tension():
    result = scan_recent_window("must include number 42")
    assert 0.0 <= result["tension"] <= 1.0
    assert "components" in result


def test_router_escalates():
    router = PrecisionRouter()
    precision, receipt = router.decide(
        {"tension": 0.7, "components": {"entropy": 0.7, "lexical_risk": 0.7}}
    )
    assert precision == "Q8"
    assert receipt is not None
    assert receipt.to_precision == 8


def test_receipt_store_appends_and_loads(tmp_path):
    store = ReceiptStore(str(tmp_path / "receipts.jsonl"))
    receipt = make_cognitive_receipt(
        prompt="hello",
        output="hello world",
        constraints=["include world"],
        verified_by=["test"],
    )
    store.append(receipt)
    records = store.load_all()
    assert len(records) == 1
    assert records[0]["type"] == "cognitive"


def test_generation_loop_runs():
    runner = MockModelRunner()
    result = run_tsq_generation(
        prompt="Brainstorm tension-aware inference. Must include one number.",
        constraints=["include one number"],
        max_new_tokens=12,
        model=runner,
    )
    assert result["output"]
    assert result["stats"]["tokens_generated"] == 12
    assert result["stats"]["verifier_pass"] is True
    assert "mock_" in result["output"]
    assert runner.call_count == 12
    assert len(result["cognitive_receipts"]) == 1
    assert result["tension_samples"]


def test_generation_loop_uses_runner_tokens_exactly():
    runner = KnownTokenRunner(["alpha", "beta", "gamma"])
    result = run_tsq_generation(
        prompt="Use alpha beta gamma",
        constraints=[],
        max_new_tokens=3,
        model=runner,
    )
    assert result["output"] == "Use alpha beta gamma alpha beta gamma"
    assert runner.call_count == 3


def test_verifier_failure_produces_failed_result():
    result = Verifier().verify(
        prompt="Say hello",
        output="hello forbidden",
        constraints=["forbidden words: forbidden"],
    )
    assert result.passed is False
    assert result.failures


def test_eval_harness_returns_all_three_baselines():
    results = compare_baselines(
        prompt="Explain TSQ with one number.",
        constraints=["include one number"],
        max_new_tokens=8,
        model=MockModelRunner(),
    )
    assert set(results) == {"always_Q4", "always_Q8", "TSQ_dynamic"}
    for run in results.values():
        assert set(run["metrics"]) == {
            "latency",
            "verifier_pass",
            "escalations",
            "receipts",
            "output_length",
        }
