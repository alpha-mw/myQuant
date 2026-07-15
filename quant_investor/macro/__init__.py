"""Pure, measurement-only Macro v2 observer surface."""

from quant_investor.macro.contracts import MacroObservation, MacroSnapshot
from quant_investor.macro.acquisition import build_macro_acquisition_plan
from quant_investor.macro.coverage import build_macro_coverage_audit
from quant_investor.macro.forward import record_macro_forward_observation
from quant_investor.macro.observer import build_macro_observer, persist_macro_observer
from quant_investor.macro.replay import run_macro_replay
from quant_investor.macro.snapshot import build_macro_snapshot
from quant_investor.macro.store import load_observations, publish_observations
from quant_investor.macro.tushare_normalizer import (
    normalize_tushare_bundle,
    publish_tushare_normalization,
)

__all__ = [
    "MacroObservation",
    "MacroSnapshot",
    "build_macro_observer",
    "build_macro_coverage_audit",
    "build_macro_acquisition_plan",
    "build_macro_snapshot",
    "load_observations",
    "normalize_tushare_bundle",
    "persist_macro_observer",
    "publish_observations",
    "publish_tushare_normalization",
    "record_macro_forward_observation",
    "run_macro_replay",
]
