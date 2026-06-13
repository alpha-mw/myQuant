from __future__ import annotations

import importlib
import importlib.util

from quant_investor import macro_terminal_tushare as macro_terminal


def test_macro_terminal_types_are_split_and_reexported() -> None:
    spec = importlib.util.find_spec("quant_investor.macro_terminal_types")
    assert spec is not None
    types_module = importlib.import_module("quant_investor.macro_terminal_types")

    assert macro_terminal.DataAcquisitionStep is types_module.DataAcquisitionStep
    assert macro_terminal.AnalysisStep is types_module.AnalysisStep
    assert macro_terminal.IndicatorResult is types_module.IndicatorResult
    assert macro_terminal.ModuleResult is types_module.ModuleResult
    assert macro_terminal.RiskTerminalReport is types_module.RiskTerminalReport

    indicator = macro_terminal.IndicatorResult(
        name="CPI同比",
        acquisition_steps=[
            macro_terminal.DataAcquisitionStep(
                timestamp="2026-06-13T00:00:00",
                data_source="fixture",
                data_type="macro",
                attempt_method="offline",
                params={"series": "cpi"},
                result_status="success",
                result_summary="ok",
            )
        ],
    )

    assert indicator.name == "CPI同比"
    assert indicator.acquisition_steps[0].params == {"series": "cpi"}


def test_us_macro_terminal_is_split_and_reexported() -> None:
    spec = importlib.util.find_spec("quant_investor.macro_terminal_us")
    assert spec is not None
    us_module = importlib.import_module("quant_investor.macro_terminal_us")

    assert macro_terminal.USMacroRiskTerminal is us_module.USMacroRiskTerminal
    terminal = macro_terminal.create_terminal("US", fred_api_key="")
    assert isinstance(terminal, us_module.USMacroRiskTerminal)
    assert terminal.MARKET == "US"
