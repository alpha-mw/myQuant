"""Pure, measurement-only Macro v2 observer surface."""

from quant_investor.macro.contracts import MacroObservation, MacroSnapshot
from quant_investor.macro.observer import build_macro_observer, persist_macro_observer
from quant_investor.macro.snapshot import build_macro_snapshot

__all__ = [
    "MacroObservation",
    "MacroSnapshot",
    "build_macro_observer",
    "build_macro_snapshot",
    "persist_macro_observer",
]
