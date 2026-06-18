"""
TS Node: TSQ Package Root
Type: package_entry
Tension sources: all submodules (tension, verifier, receipts, runtime)
Verifier hooks: import-time validation of TS headers on key modules
Receipt outputs: package-level metadata receipt on import
"""

__version__ = "0.5.0"
__ts_wave__ = "TSQ-Wave-5-CLIReports"

from . import tension, verifier, receipts, runtime  # noqa: F401
