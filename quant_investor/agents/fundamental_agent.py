"""
FundamentalAgent：对现有基本面分支做 agent 化包装。
"""

from __future__ import annotations

from typing import Any, Mapping

from quant_investor.branch_contracts import (
    CorporateDocumentSnapshot,
    ForecastSnapshot,
    FundamentalSnapshot,
    ManagementSnapshot,
    OwnershipSnapshot,
    UnifiedDataBundle,
)
from quant_investor.enhanced_data_layer import EnhancedDataLayer
from quant_investor.fundamental_branch import FundamentalBranch
from quant_investor.agents.base import BaseAgent


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        import pandas as pd

        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "nat", "none"}


def _float(value: Any, default: float = 0.0) -> float:
    if not _has_value(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


class _BundleFundamentalDataLayer:
    """PIT mart-backed data layer used only when bundle fundamentals exist."""

    def __init__(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        self.records = {str(symbol): dict(payload) for symbol, payload in records.items()}

    def _record(self, symbol: str) -> dict[str, Any]:
        return dict(self.records.get(symbol, {}) or {})

    def get_point_in_time_fundamental_snapshot(self, symbol: str, as_of: Any) -> FundamentalSnapshot:
        row = self._record(symbol)
        if not row:
            return FundamentalSnapshot(
                symbol=symbol,
                as_of=str(as_of),
                available=False,
                source="disabled",
                data_quality={"provider_missing": False, "snapshot_missing": True, "missing_scope": "symbol"},
            )
        source = str(row.get("source") or "local_fundamental_mart")
        return FundamentalSnapshot(
            symbol=symbol,
            as_of=str(as_of or row.get("trade_date", "")),
            available=True,
            source=source,
            publish_time=str(row.get("availability_date", "")),
            effective_time=str(row.get("trade_date", "")),
            revision_id=str(row.get("source_version", "")),
            roe=_float(row.get("fin_roe")),
            roa=_float(row.get("fin_roa")),
            profit_growth=_float(row.get("fin_net_profit_yoy")),
            debt_ratio=_float(row.get("fin_debt_to_assets")),
            cash_flow=_float(row.get("fin_ocf_to_profit")),
            pe=_float(row.get("pe")),
            pb=_float(row.get("pb")),
            ps=_float(row.get("ps")),
            data_quality={
                "status": "provider_snapshot",
                "provider_missing": False,
                "snapshot_missing": False,
                "missing_scope": "",
                "pit_status": "point_in_time",
                "source_priority": row.get("source_priority", "tushare_primary"),
            },
            provenance={
                "source": source,
                "source_priority": row.get("source_priority", "tushare_primary"),
                "availability_date": row.get("availability_date", ""),
            },
        )

    def get_earnings_forecast_snapshot(self, symbol: str, as_of: Any) -> ForecastSnapshot:
        row = self._record(symbol)
        if not row or not _has_value(row.get("forecast_revision")):
            return ForecastSnapshot(
                symbol=symbol,
                as_of=str(as_of),
                available=False,
                source="disabled",
                provider="local_fundamental_mart",
                data_quality={"provider_missing": False, "snapshot_missing": True, "missing_scope": "symbol"},
            )
        source = str(row.get("source") or "local_fundamental_mart")
        return ForecastSnapshot(
            symbol=symbol,
            as_of=str(as_of or row.get("trade_date", "")),
            available=True,
            source=source,
            provider=source,
            forecast_revision=_float(row.get("forecast_revision")),
            eps_growth=_float(row.get("eps_growth")),
            revenue_growth_forecast=_float(row.get("revenue_growth_forecast")),
            coverage_count=int(_float(row.get("forecast_coverage_count"), 1.0)),
            confidence=0.55,
            data_quality={"status": "provider_snapshot", "provider_missing": False, "missing_scope": ""},
            provenance={"source": source, "source_priority": row.get("source_priority", "tushare_primary")},
        )

    def get_management_snapshot(self, symbol: str, as_of: Any) -> ManagementSnapshot:
        return ManagementSnapshot(
            symbol=symbol,
            as_of=str(as_of),
            available=False,
            source="disabled",
            data_quality={"provider_missing": False, "snapshot_missing": True, "missing_scope": "global"},
        )

    def get_ownership_snapshot(self, symbol: str, as_of: Any) -> OwnershipSnapshot:
        return OwnershipSnapshot(
            symbol=symbol,
            as_of=str(as_of),
            available=False,
            source="disabled",
            data_quality={"provider_missing": False, "snapshot_missing": True, "missing_scope": "global"},
        )

    def get_document_semantic_snapshot(self, symbol: str, as_of: Any) -> CorporateDocumentSnapshot:
        return CorporateDocumentSnapshot(
            symbol=symbol,
            as_of=str(as_of),
            available=False,
            source="disabled",
            data_quality={"provider_missing": False, "snapshot_missing": True, "missing_scope": "global"},
        )


class FundamentalAgent(BaseAgent):
    """基本面 research agent。"""

    agent_name = "FundamentalAgent"

    def run(self, payload: Mapping[str, Any]) -> Any:
        envelope = self.ensure_payload(payload)
        data_bundle = envelope.get("data_bundle")
        if not isinstance(data_bundle, UnifiedDataBundle):
            raise TypeError("FundamentalAgent 需要 `data_bundle: UnifiedDataBundle`")

        stock_pool = list(envelope.get("stock_pool") or data_bundle.symbols)
        data_layer = envelope.get("data_layer")
        if data_layer is None:
            mart_records = {
                symbol: payload
                for symbol, payload in dict(data_bundle.fundamentals or {}).items()
                if isinstance(payload, Mapping) and payload
            }
            if mart_records:
                data_layer = _BundleFundamentalDataLayer(mart_records)
            else:
                data_layer = EnhancedDataLayer(
                    market=str(envelope.get("market", data_bundle.market or "CN")),
                    verbose=bool(envelope.get("verbose", False)),
                )

        branch = FundamentalBranch(
            data_layer=data_layer,
            stock_pool=stock_pool,
            enable_document_semantics=bool(envelope.get("enable_document_semantics", True)),
        )
        result = branch.run(data_bundle)

        thesis = self._build_thesis(result)
        verdict = self.branch_result_to_verdict(
            result,
            thesis=thesis,
            metadata={
                "module_coverage": dict(result.module_coverage),
                "data_quality": dict(result.data_quality),
            },
        )
        verdict.investment_risks = [
            item
            for item in verdict.investment_risks
            if "provider_missing" not in item.lower() and "snapshot_missing" not in item.lower()
        ]
        return verdict

    @staticmethod
    def _build_thesis(result) -> str:
        if str(result.conclusion or "").strip():
            return str(result.conclusion).strip()

        active_modules = [
            str(info.get("label", name))
            for name, info in dict(result.module_coverage).items()
            if info.get("status") == "active" and int(info.get("available_symbols", 0)) > 0
        ]
        module_text = "、".join(active_modules[:4]) if active_modules else "可用模块有限"
        return f"基本面分支当前由 {module_text} 参与评分并形成结构化判断。"
