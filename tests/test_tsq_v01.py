import importlib.util
import json
import os
from pathlib import Path

import pytest

from tsq.cli import main as cli_main
from tsq.evals.harness import compare_baselines
from tsq.receipts.schema import make_cognitive_receipt
from tsq.receipts.store import ReceiptStore
from tsq.reports import build_generation_report, to_jsonable
from tsq.runtime.generation_loop import run_tsq_generation
from tsq.runtime.model_runner import (
    MockModelRunner,
    RepairAwareMockRunner,
    StepResult,
    TransformersModelRunner,
)
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


class RepairTokenRunner:
    name = "repair-token-runner"

    def __init__(self):
        self.calls = []

    def step(self, precision="Q4", **kwargs):
        self.calls.append((precision, dict(kwargs)))
        if kwargs.get("repair_mode"):
            return StepResult(
                token_text="checksum",
                entropy_proxy=0.1,
                precision=precision,
                metadata={"repair_mode": True, "precision": precision},
            )
        return StepResult(
            token_text="cheap",
            entropy_proxy=0.1,
            precision=precision,
            metadata={"repair_mode": False},
        )

    def generate(self, prompt, max_new_tokens=64, precision="Q4", **kwargs):
        return prompt


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


def test_verifier_failure_triggers_compute_receipt_without_repair():
    result = run_tsq_generation(
        prompt="Respond briefly.",
        constraints=["include checksum"],
        max_new_tokens=1,
        model=RepairTokenRunner(),
        repair_on_failure=False,
    )
    assert result["stats"]["original_verifier_pass"] is False
    assert result["stats"]["final_verifier_pass"] is False
    assert result["stats"]["repair_attempted"] is False
    assert len(result["compute_receipts"]) == 1
    assert result["compute_receipts"][0].reason == "verification_failed_repair"
    assert result["cognitive_receipts"] == []


def test_repair_pass_runs_at_escalated_precision_and_succeeds():
    runner = RepairAwareMockRunner()
    result = run_tsq_generation(
        prompt="Answer tersely.",
        constraints=["include 7"],
        max_new_tokens=1,
        model=runner,
    )
    assert result["stats"]["original_verifier_pass"] is False
    assert result["stats"]["final_verifier_pass"] is True
    assert result["stats"]["repair_attempted"] is True
    assert result["stats"]["repair_succeeded"] is True
    assert result["stats"]["repair_tokens_generated"] == 1
    repair_samples = [sample for sample in result["tension_samples"] if sample.get("repair")]
    assert repair_samples
    assert repair_samples[0]["step_precision"] == "Q8"
    assert "7" in result["output"]


def test_repair_tokens_are_backend_owned_step_results():
    runner = RepairTokenRunner()
    result = run_tsq_generation(
        prompt="Respond briefly.",
        constraints=["include checksum"],
        max_new_tokens=1,
        model=runner,
    )
    assert result["stats"]["original_verifier_pass"] is False
    assert result["stats"]["final_verifier_pass"] is True
    assert result["output"].endswith("cheap checksum")
    repair_calls = [call for call in runner.calls if call[1].get("repair_mode")]
    assert repair_calls
    assert repair_calls[0][0] == "Q8"
    assert repair_calls[0][1]["verification_failures"]
    assert repair_calls[0][1]["current_output"].endswith("cheap")


def test_no_internal_token_fabrication_in_generation_loop():
    source = Path("tsq/runtime/generation_loop.py").read_text(encoding="utf-8")
    assert "_mock" + "_token" not in source
    assert "mock_" not in source
    assert "generated.append(step_result.token_text)" in source


def test_importing_tsq_does_not_require_transformers():
    import tsq

    assert tsq.__version__


def test_transformers_runner_raises_clear_import_error_when_dependencies_missing(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in {"transformers", "torch"}:
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(ImportError, match="tsq\\[transformers\\]"):
        TransformersModelRunner.from_pretrained("tiny-test-model")


def test_transformers_runner_class_exposes_stepresult_interface():
    assert hasattr(TransformersModelRunner, "from_pretrained")
    assert hasattr(TransformersModelRunner, "step")
    assert hasattr(TransformersModelRunner, "generate")
    result = StepResult(token_text="x", entropy_proxy=0.3, precision="Q4")
    assert result.token_text == "x"
    assert result.entropy_proxy == 0.3
    assert result.precision == "Q4"


def test_optional_transformers_integration():
    if os.environ.get("TSQ_RUN_TRANSFORMERS_TESTS") != "1":
        pytest.skip("set TSQ_RUN_TRANSFORMERS_TESTS=1 to run optional Transformers integration")
    model_id = os.environ.get("TSQ_TEST_MODEL_ID")
    if not model_id:
        pytest.skip("set TSQ_TEST_MODEL_ID to a tiny causal language model")
    if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
        pytest.skip("optional transformers dependencies are not installed")

    runner = TransformersModelRunner.from_pretrained(model_id)
    result = runner.step(prompt="Hello", generated=[], precision="Q4")
    assert isinstance(result, StepResult)
    assert result.metadata["backend"] == "transformers"
    assert 0.0 <= result.entropy_proxy <= 1.0


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
    dynamic_metrics = results["TSQ_dynamic"]["metrics"]
    assert set(dynamic_metrics) == {
        "latency",
        "verifier_pass",
        "original_verifier_pass",
        "final_verifier_pass",
        "escalations",
        "receipts",
        "repair_attempted",
        "repair_succeeded",
        "output_length",
    }


def test_report_serialization_handles_receipts_and_verifier_results():
    result = run_tsq_generation(
        prompt="Respond briefly.",
        constraints=["include checksum"],
        max_new_tokens=1,
        model=RepairTokenRunner(),
    )
    report = build_generation_report(
        result=result,
        backend="mock",
        prompt="Respond briefly.",
        constraints=["include checksum"],
        model_name="repair-token-runner",
    )
    encoded = json.dumps(report)
    decoded = json.loads(encoded)
    assert decoded["verification"]["passed"] is True
    assert decoded["compute_receipts"][0]["type"] == "compute"
    assert to_jsonable(StepResult("x", 0.2, "Q4", {"source": "test"}))["metadata"]["source"] == "test"


def test_cli_generate_mock_writes_json_report(tmp_path):
    report = tmp_path / "run_report.json"
    receipts = tmp_path / "receipts.jsonl"
    code = cli_main(
        [
            "generate",
            "--prompt",
            "Brainstorm TSQ. Must include one number.",
            "--constraint",
            "include one number",
            "--max-new-tokens",
            "6",
            "--backend",
            "mock",
            "--receipts",
            str(receipts),
            "--report",
            str(report),
        ]
    )
    assert code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == "generate"
    assert payload["backend"] == "mock"
    assert payload["stats"]["verifier_pass"] is True
    assert receipts.exists()


def test_cli_eval_mock_writes_all_baselines(tmp_path):
    report = tmp_path / "eval_report.json"
    code = cli_main(
        [
            "eval",
            "--prompt",
            "Explain TSQ with one number.",
            "--constraint",
            "include one number",
            "--max-new-tokens",
            "6",
            "--backend",
            "mock",
            "--report",
            str(report),
        ]
    )
    assert code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == "eval"
    assert set(payload["results"]) == {"always_Q4", "always_Q8", "TSQ_dynamic"}
    assert "dynamic_passed" in payload["summary"]


def test_cli_repair_eval_writes_aggregate_metrics(tmp_path):
    report = tmp_path / "repair_eval_report.json"
    code = cli_main(["repair-eval", "--backend", "repair-mock", "--report", str(report)])
    assert code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == "repair-eval"
    assert payload["aggregate"]["total_tasks"] == 3
    assert payload["aggregate"]["repair_attempts"] >= 1
    assert payload["aggregate"]["total_compute_receipts"] >= 1


def test_cli_transformers_backend_missing_model_or_dependencies_is_clear(tmp_path, capsys):
    report = tmp_path / "transformers_report.json"
    code = cli_main(
        [
            "generate",
            "--prompt",
            "hello",
            "--backend",
            "transformers",
            "--report",
            str(report),
        ]
    )
    assert code == 2
    assert "--model-id is required" in capsys.readouterr().err

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in {"transformers", "torch"}:
            return None
        return original_find_spec(name, *args, **kwargs)

    report_with_model = tmp_path / "transformers_model_report.json"
    from pytest import MonkeyPatch

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    try:
        code = cli_main(
            [
                "generate",
                "--prompt",
                "hello",
                "--backend",
                "transformers",
                "--model-id",
                "tiny-test-model",
                "--report",
                str(report_with_model),
            ]
        )
    finally:
        monkeypatch.undo()
    assert code == 2
    assert "pip install -e '.[transformers]'" in capsys.readouterr().err
