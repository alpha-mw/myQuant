"""Offline macro source placeholder used by the DataHub facade."""

from __future__ import annotations

from quant_investor.data.models import MacroData


class MacroDataSource:
    source_name = "manual_offline_snapshot"

    def get_macro(self, market: str = "CN", as_of: str = "") -> MacroData:
        return MacroData(
            market=market,
            as_of=as_of,
            source=self.source_name,
            metadata={"status": "offline_snapshot_missing"},
        )
