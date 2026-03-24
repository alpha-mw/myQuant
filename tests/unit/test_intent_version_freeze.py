"""
设计意图 8：V8 必须冻结，current 才指向当前稳定主线。

当前预期：
- 包级 current / V8 freeze 语义通过
- CLI 默认 architecture 仍是 latest，会暴露入口未收口 blocker
"""

from __future__ import annotations

import quant_investor
import quant_investor.cli.main as cli_main
from quant_investor.pipeline.legacy_v8_pipeline import LegacyV8ParallelResearchPipeline
from quant_investor.versioning import (
    ARCHITECTURE_VERSION_CURRENT,
    ARCHITECTURE_VERSION_V9,
    LEGACY_BRANCH_ORDER,
)


def test_v8_is_frozen_and_current_points_to_v9() -> None:
    pipeline = LegacyV8ParallelResearchPipeline(stock_pool=["000001.SZ"], verbose=False)

    assert quant_investor.QuantInvestor is quant_investor.QuantInvestorV9
    assert quant_investor.QuantInvestorCurrent is quant_investor.QuantInvestorV9
    assert ARCHITECTURE_VERSION_CURRENT == ARCHITECTURE_VERSION_V9
    assert quant_investor.QuantInvestorV8.__module__.endswith("quant_investor_v8")
    assert pipeline.enable_fundamental is False
    assert pipeline.enable_branch_debate is False
    assert pipeline.BRANCH_ORDER == LEGACY_BRANCH_ORDER


def test_cli_default_architecture_should_point_to_current_stable() -> None:
    args = cli_main._build_parser().parse_args(["research", "run", "--stocks", "000001.SZ"])

    assert args.architecture == "v9", (
        "CLI 默认 architecture 仍是 latest=V10，而不是 current stable=V9；"
        "这是命名/入口未收口，不是 V8 freeze 常量缺失。"
    )

