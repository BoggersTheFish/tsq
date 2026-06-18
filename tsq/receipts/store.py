"""
TS Node: ReceiptStore
Type: receipt_persistence
Description: Simple JSONL-backed persistent store for cognitive and compute receipts.
             Enables inspection, replay, and future cross-session propagation.
             v0.3: append-only + load_all. Later: indexing, invalidation, graph queries.
Tension sources: none (pure I/O)
Edges:
  - generation_loop → append(receipt)
  - external tools → load_all() for analysis / replay
Verifier hooks: receipts carry their own verified_by list
Receipt outputs: durable JSONL file (default: artifacts/tsq_receipts.jsonl)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime, timezone

from .schema import BaseReceipt, CognitiveReceipt, ComputeReceipt


class ReceiptStore:
    """Append-only JSONL receipt store with load capability."""

    def __init__(self, path: str = "artifacts/tsq_receipts.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, receipt: Union[BaseReceipt, Dict[str, Any], CognitiveReceipt, ComputeReceipt]) -> None:
        """Append a single receipt (dataclass or dict) as one JSON line."""
        if isinstance(receipt, dict):
            record = receipt
        else:
            record = receipt.to_dict() if hasattr(receipt, "to_dict") else vars(receipt)
        # Ensure timestamp
        if "created_at" not in record:
            record["created_at"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_all(self) -> List[Dict[str, Any]]:
        """Load every receipt in the file as a list of dicts."""
        records: List[Dict[str, Any]] = []
        if not self.path.exists():
            return records
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # skip corrupt lines gracefully
        return records

    def load_by_type(self, receipt_type: str) -> List[Dict[str, Any]]:
        """Convenience filter."""
        return [r for r in self.load_all() if r.get("type") == receipt_type]

    def clear(self) -> None:
        """Dangerous: truncate the file (useful in tests)."""
        self.path.write_text("", encoding="utf-8")


def get_default_store() -> ReceiptStore:
    return ReceiptStore(path="artifacts/tsq_receipts.jsonl")
