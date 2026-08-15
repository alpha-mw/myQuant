"""Public, generation-bound research facade.

The public package is deliberately a read-only view over the active unified
generation.  It never builds a candidate, changes the active pointer, or falls
back to a retired runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class QuantInvestor:
    """Read the active unified Mainline result for one canonical strategy."""

    def __init__(
        self,
        *,
        workspace_root: str | Path = ".",
        strategy_id: str,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.strategy_id = str(strategy_id)

    def run(self) -> dict[str, Any]:
        """Return the exact generation-bound public-run artifact."""

        from quant_investor.mainline import read_public_run

        return read_public_run(
            self.workspace_root,
            strategy_id=self.strategy_id,
        )


__all__ = ["QuantInvestor"]
