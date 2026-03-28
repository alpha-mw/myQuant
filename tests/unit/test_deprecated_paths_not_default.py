"""
deprecated 路径不再作为默认入口的回归测试。
"""

from __future__ import annotations

import importlib
import warnings

import quant_investor
import quant_investor.pipeline as pipeline


def test_deprecated_python_aliases_still_resolve_but_warn():
    with warnings.catch_warnings(record=True) as items:
        warnings.simplefilter("always", DeprecationWarning)
        current = quant_investor.QuantInvestorCurrent
        latest = quant_investor.QuantInvestorLatest
        v10 = quant_investor.QuantInvestorV10

    messages = [str(item.message) for item in items]
    assert current is quant_investor.QuantInvestorV11
    assert latest is quant_investor.QuantInvestorV11
    assert v10 is quant_investor.QuantInvestorV10
    assert any("QuantInvestorCurrent 已废弃" in message for message in messages)
    assert any("QuantInvestorLatest" in message for message in messages)

    with warnings.catch_warnings(record=True) as items:
        warnings.simplefilter("always", DeprecationWarning)
        pipeline_current = pipeline.QuantInvestorCurrent
        pipeline_latest = pipeline.QuantInvestorLatest

    pipeline_messages = [str(item.message) for item in items]
    assert pipeline_current is pipeline.QuantInvestorV11
    assert pipeline_latest is pipeline.QuantInvestorV11
    assert any("QuantInvestorCurrent 已废弃" in message for message in pipeline_messages)
    assert any("QuantInvestorLatest" in message for message in pipeline_messages)


def test_deprecated_cli_aliases_normalize_to_explicit_targets():
    cli_main = importlib.import_module("quant_investor.cli.main")

    with warnings.catch_warnings(record=True) as items:
        warnings.simplefilter("always", DeprecationWarning)
        current = cli_main._normalize_architecture("current")
        latest = cli_main._normalize_architecture("latest")

    messages = [str(item.message) for item in items]
    assert current == "v11"
    assert latest == "v11"
    assert any("--architecture current 已废弃" in message for message in messages)
    assert any("--architecture latest 已废弃" in message for message in messages)
