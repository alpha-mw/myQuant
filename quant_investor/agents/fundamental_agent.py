"""
FundamentalAgent：对现有基本面分支做 agent 化包装。
"""

from __future__ import annotations

import hashlib
import json
import math
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
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    except Exception:
        return default


def _available_fields(row: Mapping[str, Any], field_map: Mapping[str, tuple[str, ...]]) -> tuple[dict[str, float], list[str]]:
    values: dict[str, float] = {}
    available: list[str] = []
    for target, candidates in field_map.items():
        source_value = next((row.get(name) for name in candidates if _has_value(row.get(name))), None)
        numeric = _float(source_value, float("nan"))
        values[target] = numeric if math.isfinite(numeric) else 0.0
        if math.isfinite(numeric):
            available.append(target)
    return values, available


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
        field_map = {
            "roe": ("fin_roe", "roe"),
            "roa": ("fin_roa", "roa"),
            "gross_margin": ("fin_gross_margin", "gross_margin"),
            "net_margin": ("fin_net_margin", "net_margin"),
            "revenue_growth": ("fin_revenue_yoy", "fin_revenue_growth", "revenue_growth"),
            "profit_growth": ("fin_net_profit_yoy", "profit_growth"),
            "debt_ratio": ("fin_debt_to_assets", "debt_ratio"),
            "current_ratio": ("fin_current_ratio", "current_ratio"),
            "cash_flow": ("fin_ocf_to_profit", "cash_flow"),
            "pe": ("pe", "pe_ttm"),
            "pb": ("pb",),
            "ps": ("ps", "ps_ttm"),
            "dividend_yield": ("dividend_yield", "dv_ratio", "dv_ttm"),
        }
        values, available_fields = _available_fields(row, field_map)
        valuation_fields = [name for name in ("pe", "pb", "ps", "dividend_yield") if name in available_fields]
        financial_fields = [name for name in available_fields if name not in {"pe", "pb", "ps", "dividend_yield"}]
        return FundamentalSnapshot(
            symbol=symbol,
            as_of=str(as_of or row.get("trade_date", "")),
            available=bool(financial_fields or valuation_fields),
            source=source,
            publish_time=str(row.get("availability_date", "")),
            effective_time=str(row.get("trade_date", "")),
            revision_id=str(row.get("source_version", "")),
            roe=values["roe"],
            roa=values["roa"],
            gross_margin=values["gross_margin"],
            net_margin=values["net_margin"],
            revenue_growth=values["revenue_growth"],
            profit_growth=values["profit_growth"],
            debt_ratio=values["debt_ratio"],
            current_ratio=values["current_ratio"],
            cash_flow=values["cash_flow"],
            pe=values["pe"],
            pb=values["pb"],
            ps=values["ps"],
            dividend_yield=values["dividend_yield"],
            data_quality={
                "status": "provider_snapshot",
                "provider_missing": False,
                "snapshot_missing": False,
                "missing_scope": "",
                "pit_status": "point_in_time",
                "source_priority": row.get("source_priority", "tushare_primary"),
                "available_fields": available_fields,
                "missing_fields": [name for name in field_map if name not in available_fields],
                "valuation_available": bool(valuation_fields),
                "valuation_snapshot_missing": not bool(valuation_fields),
                "field_coverage_ratio": round(len(available_fields) / len(field_map), 4),
            },
            provenance={
                "source": source,
                "source_priority": row.get("source_priority", "tushare_primary"),
                "availability_date": row.get("availability_date", ""),
                "end_date": row.get("end_date", ""),
                "fin_fcf_to_profit": row.get("fin_fcf_to_profit"),
                "fcf_to_price": row.get("fcf_to_price"),
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
        source = str(row.get("forecast_source") or row.get("source") or "local_fundamental_mart")
        field_map = {
            "forecast_revision": ("forecast_revision",),
            "eps_growth": ("eps_growth",),
            "revenue_growth_forecast": ("revenue_growth_forecast",),
            "coverage_count": ("forecast_coverage_count",),
        }
        values, available_fields = _available_fields(row, field_map)
        return ForecastSnapshot(
            symbol=symbol,
            as_of=str(as_of or row.get("trade_date", "")),
            available=bool(available_fields),
            source=source,
            provider=source,
            publish_time=str(row.get("forecast_ann_date") or row.get("availability_date") or ""),
            effective_time=str(row.get("availability_date") or row.get("forecast_ann_date") or row.get("trade_date") or ""),
            revision_id=str(row.get("forecast_ingest_run_id") or row.get("source_version") or ""),
            forecast_revision=values["forecast_revision"],
            eps_growth=values["eps_growth"],
            revenue_growth_forecast=values["revenue_growth_forecast"],
            coverage_count=int(values["coverage_count"]),
            confidence=0.55,
            data_quality={
                "status": "provider_snapshot",
                "provider_missing": False,
                "missing_scope": "",
                "forecast_kind": "corporate_guidance",
                "available_fields": available_fields,
                "missing_fields": [name for name in field_map if name not in available_fields],
                "field_coverage_ratio": round(len(available_fields) / len(field_map), 4),
            },
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
        snapshot_quality_by_symbol = dict(
            result.data_quality.get("snapshot_quality_by_symbol", {}) or {}
        )
        fundamental_generations = {
            str(symbol): "fundamental-"
            + hashlib.sha256(
                json.dumps(
                    {
                        "snapshot_quality": quality,
                        "local_events": list(
                            (data_bundle.event_data or {}).get(str(symbol), []) or []
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()[:24]
            for symbol, quality in snapshot_quality_by_symbol.items()
        }
        verdict = self.branch_result_to_verdict(
            result,
            thesis=thesis,
            metadata={
                "module_coverage": dict(result.module_coverage),
                "data_quality": dict(result.data_quality),
                "structured_signals": dict(result.signals),
                "reliability": float(result.metadata.get("reliability", 0.0)),
                "horizon_days": int(result.horizon_days),
                "fundamental_data_generation_by_symbol": fundamental_generations,
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
