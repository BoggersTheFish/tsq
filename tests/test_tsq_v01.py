import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tsq.cli import main as cli_main
from tsq.evals.harness import compare_baselines, run_eval_suite
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
from tsq.training.dataset_builder import build_datasets
from tsq.training.experiment import build_manifest, read_manifest, write_manifest
from tsq.training.schema import PreferenceExample, RepairTrainingExample, TrainingExample
from tsq.training.validate_dataset import dataset_summary, validate_dataset


def load_script_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


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


class IsolatedRunner:
    instances = []
    name = "isolated-runner"

    def __init__(self):
        self.step_calls = 0
        self.generate_calls = 0
        self.precisions = []
        IsolatedRunner.instances.append(self)

    def step(self, precision="Q4", **kwargs):
        self.step_calls += 1
        self.precisions.append(precision)
        return StepResult(
            token_text="7",
            entropy_proxy=0.2,
            precision=precision,
            metadata={"step_calls": self.step_calls},
        )

    def generate(self, prompt, max_new_tokens=64, precision="Q4", **kwargs):
        self.generate_calls += 1
        self.precisions.append(precision)
        return prompt + " 7"


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
        "tokens_at_Q4",
        "tokens_at_Q8",
        "tokens_at_FP16",
        "tokens_at_residual_unfolded",
        "repair_tokens_generated",
        "total_tokens_generated",
        "estimated_cost_units",
        "cost_model_used",
        "precision_histogram",
    }


def test_compare_baselines_uses_fresh_runner_instances():
    IsolatedRunner.instances = []
    results = compare_baselines(
        prompt="Answer with 7.",
        constraints=["include 7"],
        max_new_tokens=2,
        backend_factory=IsolatedRunner,
    )
    assert set(results) == {"always_Q4", "always_Q8", "TSQ_dynamic"}
    assert len(IsolatedRunner.instances) == 3
    q4_runner, q8_runner, dynamic_runner = IsolatedRunner.instances
    assert q4_runner.generate_calls == 1
    assert q8_runner.generate_calls == 1
    assert dynamic_runner.step_calls == 2
    assert q4_runner is not q8_runner
    assert q8_runner is not dynamic_runner


def test_generation_cost_accounting_and_precision_histogram():
    result = run_tsq_generation(
        prompt="Must include one number 7 exactly.",
        constraints=["include 7"],
        max_new_tokens=10,
        model=MockModelRunner(),
    )
    stats = result["stats"]
    assert stats["tokens_at_Q4"] >= 1
    assert stats["tokens_at_Q8"] >= 1
    assert stats["tokens_at_FP16"] == 0
    assert stats["total_tokens_generated"] == stats["tokens_generated"]
    expected = stats["tokens_at_Q4"] + (stats["tokens_at_Q8"] * 2.0)
    expected += stats["tokens_at_FP16"] * 4.0
    expected += stats["tokens_at_residual_unfolded"] * 4.0
    assert stats["estimated_cost_units"] == expected
    histogram = stats["precision_histogram"]
    assert histogram["counts"]["Q4"] == stats["tokens_at_Q4"]
    assert histogram["counts"]["Q8"] == stats["tokens_at_Q8"]
    assert histogram["compute_receipt_count"] == len(result["compute_receipts"])


def test_compute_receipt_metadata_contains_tension_and_verifier_context():
    result = run_tsq_generation(
        prompt="Respond briefly.",
        constraints=["include checksum"],
        max_new_tokens=1,
        model=RepairTokenRunner(),
    )
    receipt = result["compute_receipts"][0]
    assert "tension_components" in receipt.metadata
    assert "verifier_failures" in receipt.metadata
    assert receipt.metadata["chosen_precision"] == "Q8"
    assert receipt.metadata["target"] == "final_generation_output"
    assert receipt.metadata["repair_attempted"] is True
    assert receipt.metadata["repair_succeeded"] is True


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
    assert decoded["precision_histogram"]["compute_receipt_count"] == 1
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
    assert "precision_histogram" in payload["aggregate"]


def test_cli_eval_suite_writes_valid_report(tmp_path):
    report = tmp_path / "eval_suite_report.json"
    code = cli_main(["eval-suite", "--backend", "repair-mock", "--report", str(report)])
    assert code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == "eval-suite"
    assert payload["aggregate"]["total_tasks"] == 12
    assert "mean_dynamic_vs_q8_cost_ratio" in payload["aggregate"]
    assert "precision_histogram" in payload["aggregate"]


def test_run_eval_suite_aggregate_contains_costs():
    suite = run_eval_suite(backend_factory=RepairAwareMockRunner)
    assert suite["aggregate"]["total_tasks"] == 12
    assert suite["aggregate"]["mean_dynamic_estimated_cost"] > 0


def test_ci_workflow_exists():
    assert Path(".github/workflows/ci.yml").exists()


def test_training_schema_serialization():
    example = TrainingExample(
        id="ex-1",
        task_name="task",
        split="train",
        prompt="Prompt",
        constraints=["include 7"],
        input_text="Prompt: Prompt",
        target_text="7",
        example_type="number_preservation",
        metadata={"source_task": "task"},
    )
    repair = RepairTrainingExample(
        id="repair-1",
        task_name="task",
        split="train",
        original_prompt="Prompt",
        constraints=["include 7"],
        failed_output="Prompt cheap",
        verifier_failures=["missing number/date from prompt or constraints: 7"],
        repair_prompt="Repair:",
        repair_target="7",
        final_output="Prompt cheap 7",
        metadata={},
    )
    preference = PreferenceExample(
        id="pref-1",
        prompt="Prompt",
        constraints=["include 7"],
        chosen_output="Prompt cheap 7",
        rejected_output="Prompt cheap",
        reason="verifier failure",
        metadata={},
    )
    assert example.to_dict()["target_text"] == "7"
    assert repair.to_dict()["verifier_failures"]
    assert preference.to_dict()["chosen_output"] != preference.to_dict()["rejected_output"]


def test_dataset_builder_writes_and_validates_all_files(tmp_path):
    paths = build_datasets(tmp_path / "generated", include_example_reports="examples/reports")
    assert set(paths) == {
        "supervised_train",
        "supervised_eval",
        "repair_train",
        "preference_pairs",
    }
    for path in paths.values():
        assert path.exists()
        rows = validate_dataset(path)
        assert rows
    summary = dataset_summary(paths["supervised_train"])
    assert summary["examples"] > 0
    assert summary["example_type_counts"]


def test_repair_dataset_contains_verifier_failures_and_preference_outputs(tmp_path):
    paths = build_datasets(tmp_path / "generated")
    repair_rows = validate_dataset(paths["repair_train"])
    preference_rows = validate_dataset(paths["preference_pairs"])
    assert repair_rows
    assert all(row["verifier_failures"] for row in repair_rows)
    assert preference_rows
    assert all(row["chosen_output"] and row["rejected_output"] for row in preference_rows)
    assert any(row["chosen_output"] != row["rejected_output"] for row in preference_rows)


def test_cli_build_validate_and_summary_dataset(tmp_path, capsys):
    out_dir = tmp_path / "generated"
    code = cli_main(
        [
            "build-dataset",
            "--out-dir",
            str(out_dir),
            "--include-example-reports",
            "examples/reports",
        ]
    )
    assert code == 0
    train_path = out_dir / "tsq_supervised_train.jsonl"
    code = cli_main(["validate-dataset", "--path", str(train_path)])
    assert code == 0
    assert "valid dataset" in capsys.readouterr().out
    code = cli_main(["dataset-summary", "--path", str(train_path)])
    assert code == 0
    assert "examples:" in capsys.readouterr().out


def test_train_lora_dry_run_without_heavy_deps(tmp_path):
    paths = build_datasets(tmp_path / "generated")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_lora.py",
            "--model-id",
            "dry-run-model",
            "--train-jsonl",
            str(paths["supervised_train"]),
            "--eval-jsonl",
            str(paths["supervised_eval"]),
            "--output-dir",
            str(tmp_path / "lora"),
            "--dry-run",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "dry-run ok" in result.stdout


def test_train_lora_parser_accepts_current_flags():
    train_lora = load_script_module("scripts/train_lora.py", "train_lora_test_parser")
    args = train_lora.build_parser().parse_args(
        [
            "--model-id",
            "model",
            "--train-jsonl",
            "train.jsonl",
            "--eval-jsonl",
            "eval.jsonl",
            "--repair-jsonl",
            "repair.jsonl",
            "--output-dir",
            "out",
            "--training-mode",
            "mixed",
            "--load-in-4bit",
            "--bf16",
            "--gradient-checkpointing",
            "--max-seq-length",
            "128",
            "--per-device-train-batch-size",
            "2",
            "--gradient-accumulation-steps",
            "4",
            "--warmup-steps",
            "1",
            "--save-steps",
            "2",
            "--eval-steps",
            "2",
            "--logging-steps",
            "1",
            "--smoke-train",
        ]
    )
    assert args.training_mode == "mixed"
    assert args.load_in_4bit is True
    assert args.max_seq_length == 128
    assert args.smoke_train is True


def test_training_formatters_handle_supervised_and_repair_rows():
    train_lora = load_script_module("scripts/train_lora.py", "train_lora_test_formatters")
    supervised = {
        "input_text": "Prompt: include 7",
        "target_text": "7",
    }
    repair = {
        "repair_prompt": "Repair the output",
        "repair_target": "7",
    }
    assert "TSQ-aware assistant" in train_lora.format_training_row(supervised)
    assert "<Assistant>\n7" in train_lora.format_training_row(supervised)
    assert "TSQ repair model" in train_lora.format_training_row(repair)
    assert "Repair the output" in train_lora.format_training_row(repair)


def test_mixed_dataset_preparation_combines_supervised_and_repair(tmp_path):
    train_lora = load_script_module("scripts/train_lora.py", "train_lora_test_mixed")
    paths = build_datasets(tmp_path / "generated")
    rows = train_lora.load_formatted_examples(
        paths["supervised_train"],
        training_mode="mixed",
        repair_jsonl=paths["repair_train"],
    )
    supervised_count = len(validate_dataset(paths["supervised_train"]))
    repair_count = len(validate_dataset(paths["repair_train"]))
    assert len(rows) == supervised_count + repair_count
    assert any("TSQ repair model" in row["text"] for row in rows)


def test_train_lora_missing_deps_error_is_clear(tmp_path, monkeypatch):
    train_lora = load_script_module("scripts/train_lora.py", "train_lora_test_missing_deps")
    paths = build_datasets(tmp_path / "generated")
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in {"torch", "transformers", "datasets", "peft", "accelerate"}:
            return None
        return original_find_spec(name)

    monkeypatch.setattr(train_lora.importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(RuntimeError, match="Missing optional training dependencies"):
        train_lora.main(
            [
                "--model-id",
                "model",
                "--train-jsonl",
                str(paths["supervised_train"]),
                "--eval-jsonl",
                str(paths["supervised_eval"]),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_transformers_runner_and_cli_accept_adapter_dir():
    import inspect

    signature = inspect.signature(TransformersModelRunner.from_pretrained)
    assert "adapter_dir" in signature.parameters
    parser = __import__("tsq.cli", fromlist=["build_parser"]).build_parser()
    args = parser.parse_args(
        [
            "eval-suite",
            "--backend",
            "transformers",
            "--model-id",
            "tiny",
            "--adapter-dir",
            "adapter",
            "--report",
            "report.json",
        ]
    )
    assert args.adapter_dir == "adapter"


def test_eval_lora_dry_run_writes_check_report(tmp_path):
    report = tmp_path / "eval_lora_check.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_lora.py",
            "--model-id",
            "dry-run-model",
            "--adapter-dir",
            "adapter",
            "--report",
            str(report),
            "--dry-run",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == "lora-eval-check"
    assert payload["adapter_dir"] == "adapter"


def test_check_training_env_no_fail_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "scripts/check_training_env.py", "--no-fail"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Python:" in result.stdout


def test_experiment_manifest_schema_round_trips(tmp_path):
    manifest = build_manifest(
        model_id="dry-run-model",
        adapter_dir="adapter",
        dataset_files={"train": "train.jsonl"},
        training_mode="mixed",
        max_steps=2,
        smoke_train=True,
        dependency_status={"ready_for_training": False},
        training_status="skipped_missing_dependencies",
        eval_status="skipped_missing_dependencies",
        eval_report_path=None,
        notes=["missing deps"],
    )
    path = write_manifest(tmp_path / "manifest.json", manifest)
    loaded = read_manifest(path)
    required = {
        "tsq_version",
        "created_at",
        "model_id",
        "adapter_dir",
        "dataset_files",
        "training_mode",
        "max_steps",
        "smoke_train",
        "commands_run",
        "dependency_status",
        "training_status",
        "eval_status",
        "eval_report_path",
        "notes",
    }
    assert required <= set(loaded)
    assert loaded["training_status"] == "skipped_missing_dependencies"


def test_run_v09_experiment_dry_run_writes_manifest_without_model_load(tmp_path):
    manifest_dir = tmp_path / "experiment"
    report_dir = tmp_path / "reports"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_v09_experiment.py",
            "--dry-run",
            "--model-id",
            "dry-run-model",
            "--experiment-dir",
            str(manifest_dir),
            "--report-dir",
            str(report_dir),
            "--output-dir",
            str(tmp_path / "adapter"),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((manifest_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_status"] == "dry_run"
    assert manifest["eval_status"] == "dry_run"
    assert "no model loaded" in result.stdout


def test_summarize_experiment_handles_dry_run_manifest(tmp_path):
    manifest = build_manifest(
        model_id="dry-run-model",
        adapter_dir="adapter",
        dataset_files={},
        training_mode="mixed",
        max_steps=2,
        smoke_train=False,
        dependency_status={"ready_for_training": False},
        training_status="dry_run",
        eval_status="dry_run",
        eval_report_path=None,
    )
    manifest_path = write_manifest(tmp_path / "manifest.json", manifest)
    markdown = tmp_path / "summary.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_experiment.py",
            "--manifest",
            str(manifest_path),
            "--markdown",
            str(markdown),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "training_status: dry_run" in result.stdout
    assert "TSQ v0.9 Experiment Summary" in markdown.read_text(encoding="utf-8")


def test_pyproject_has_training_extra():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "training = " in pyproject
    assert '"peft"' in pyproject
    assert "qlora = " in pyproject


def test_optional_lora_smoke_training(tmp_path):
    if os.environ.get("TSQ_RUN_TRAINING_TESTS") != "1":
        pytest.skip("set TSQ_RUN_TRAINING_TESTS=1 to run optional LoRA smoke training")
    model_id = os.environ.get("TSQ_TRAINING_TEST_MODEL_ID")
    if not model_id:
        pytest.skip("set TSQ_TRAINING_TEST_MODEL_ID to a tiny causal LM")
    missing = [
        name
        for name in ("torch", "transformers", "datasets", "peft", "accelerate")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        pytest.skip(f"missing optional training dependencies: {', '.join(missing)}")
    paths = build_datasets(tmp_path / "generated")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_lora.py",
            "--model-id",
            model_id,
            "--train-jsonl",
            str(paths["supervised_train"]),
            "--eval-jsonl",
            str(paths["supervised_eval"]),
            "--output-dir",
            str(tmp_path / "lora"),
            "--smoke-train",
            "--max-steps",
            "1",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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
