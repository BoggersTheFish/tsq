#!/usr/bin/env python3
"""
Report optional TSQ training environment readiness.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from typing import Any, Dict, Sequence


CORE_DEPS = ("torch", "transformers", "datasets", "peft", "accelerate")
OPTIONAL_DEPS = ("bitsandbytes",)


def dependency_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "recommended_install": "pip install -e '.[training]'",
        "recommended_qlora_install": "pip install -e '.[qlora]'",
    }
    for name in [*CORE_DEPS, *OPTIONAL_DEPS]:
        status[name] = importlib.util.find_spec(name) is not None

    status["cuda_available"] = False
    status["gpu_name"] = None
    if status["torch"]:
        try:
            import torch

            status["cuda_available"] = bool(torch.cuda.is_available())
            if status["cuda_available"]:
                status["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            status["cuda_error"] = str(exc)
    status["missing_core_deps"] = [name for name in CORE_DEPS if not status[name]]
    status["ready_for_training"] = not status["missing_core_deps"]
    return status


def print_status(status: Dict[str, Any]) -> None:
    print(f"Python: {status['python_version']}")
    for name in [*CORE_DEPS, *OPTIONAL_DEPS]:
        print(f"{name}: {status[name]}")
    print(f"CUDA available: {status['cuda_available']}")
    print(f"GPU name: {status['gpu_name']}")
    if status["missing_core_deps"]:
        print(f"missing core deps: {', '.join(status['missing_core_deps'])}")
        print(f"recommended install: {status['recommended_install']}")
    if not status["bitsandbytes"]:
        print(f"QLoRA install: {status['recommended_qlora_install']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check TSQ training dependencies")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = dependency_status()
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print_status(status)
    if status["ready_for_training"] or args.no_fail:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
