"""
Experiment manifest helpers for TSQ adapter runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List

from tsq import __version__


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentManifest:
    tsq_version: str
    created_at: str
    model_id: str
    adapter_dir: str
    dataset_files: Dict[str, str]
    training_mode: str
    max_steps: int
    smoke_train: bool
    commands_run: List[List[str]]
    dependency_status: Dict[str, Any]
    training_status: str
    eval_status: str
    eval_report_path: str | None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_manifest(
    model_id: str,
    adapter_dir: str,
    dataset_files: Dict[str, str],
    training_mode: str,
    max_steps: int,
    smoke_train: bool,
    commands_run: List[List[str]] | None = None,
    dependency_status: Dict[str, Any] | None = None,
    training_status: str = "not_started",
    eval_status: str = "not_started",
    eval_report_path: str | None = None,
    notes: List[str] | None = None,
) -> ExperimentManifest:
    return ExperimentManifest(
        tsq_version=__version__,
        created_at=utc_now(),
        model_id=model_id,
        adapter_dir=adapter_dir,
        dataset_files=dict(dataset_files),
        training_mode=training_mode,
        max_steps=int(max_steps),
        smoke_train=bool(smoke_train),
        commands_run=list(commands_run or []),
        dependency_status=dict(dependency_status or {}),
        training_status=training_status,
        eval_status=eval_status,
        eval_report_path=eval_report_path,
        notes=list(notes or []),
    )


def write_manifest(path: str | Path, manifest: ExperimentManifest | Dict[str, Any]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def read_manifest(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
