"""Public V17-only research facade.

The public package is deliberately a read-only view over the governed V17
mainline.  It never builds a run, selects another protocol, or falls back to
legacy artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

class QuantInvestor:
    """Read the active V17 run for one canonical strategy."""

    def __init__(
        self,
        *,
        workspace_root: str | Path = ".",
        strategy_id: str,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.strategy_id = str(strategy_id)

    def run(self) -> dict[str, Any]:
        """Return the exact governed ``mainline-public-run.v1`` DTO."""

        from quant_investor.v17_mainline import read_public_run

        return read_public_run(
            self.workspace_root,
            strategy_id=self.strategy_id,
        )


__all__ = ["QuantInvestor"]
