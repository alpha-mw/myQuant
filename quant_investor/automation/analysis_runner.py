"""Read-only automation adapter for the active unified Mainline generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_investor.mainline import MAINLINE_ARGUMENTS_INVALID, MainlineError, read_public_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AnalysisRunner:
    """Read the active generation without producing or activating new results."""

    def run(
        self,
        config: dict[str, Any],
        recall_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del recall_context
        market = str(config.get("market") or "CN").strip().upper()
        if market != "CN":
            raise MainlineError(
                MAINLINE_ARGUMENTS_INVALID,
                blockers=["MARKET_UNSUPPORTED"],
            )
        strategy_id = str(
            config.get("strategy_id") or config.get("history_strategy") or ""
        ).strip()
        return read_public_run(PROJECT_ROOT, strategy_id=strategy_id)


__all__ = ["AnalysisRunner"]
