"""US public fundamental fallback placeholder."""

from __future__ import annotations

from quant_investor.data.models import FundamentalData


class USFundamentalDataSource:
    source_name = "public_structured_fallback"

    def get_fundamental(self, symbol: str) -> FundamentalData:
        return FundamentalData(symbol=symbol, source="unavailable")
