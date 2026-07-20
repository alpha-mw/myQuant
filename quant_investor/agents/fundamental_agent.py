"""
FundamentalAgent：对现有基本面分支做 agent 化包装。
"""

from __future__ import annotations

import math
from pathlib import Path
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
from quant_investor.market.fundamental_generation import (
    FUNDAMENTAL_POINTER_FILENAME,
    FundamentalGenerationError,
    load_fundamental_pointer,
)
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


def _normalized_date(value: Any):
    if not _has_value(value):
        return None
    try:
        import pandas as pd

        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()
    except Exception:
        return None


def _pit_date_blockers(
    *,
    row: Mapping[str, Any],
    as_of: Any,
    publish_fields: tuple[str, ...],
    effective_fields: tuple[str, ...],
) -> list[str]:
    requested = _normalized_date(as_of)
    publish_raw = next(
        (row.get(name) for name in publish_fields if _has_value(row.get(name))),
        None,
    )
    effective_raw = next(
        (row.get(name) for name in effective_fields if _has_value(row.get(name))),
        None,
    )
    publish = _normalized_date(publish_raw)
    effective = _normalized_date(effective_raw)
    blockers: list[str] = []
    if requested is None:
        blockers.append("pit_requested_as_of_missing")
    if publish is None:
        blockers.append("pit_publish_time_missing")
    if effective is None:
        blockers.append("pit_effective_time_missing")
    if requested is not None and publish is not None and publish > requested:
        blockers.append("pit_publish_time_after_as_of")
    if requested is not None and effective is not None and effective > requested:
        blockers.append("pit_effective_time_after_as_of")
    if publish is not None and effective is not None and publish > effective:
        blockers.append("pit_publish_time_after_effective_time")
    return blockers


def _available_fields(
    row: Mapping[str, Any],
    field_map: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, float], list[str]]:
    values: dict[str, float] = {}
    available: list[str] = []
    for target, candidates in field_map.items():
        source_value = next(
            (row.get(name) for name in candidates if _has_value(row.get(name))),
            None,
        )
        numeric = _float(source_value, float("nan"))
        values[target] = numeric if math.isfinite(numeric) else 0.0
        if math.isfinite(numeric):
            available.append(target)
    return values, available


def _canonical_fundamental_generation(
    data_bundle: UnifiedDataBundle,
) -> tuple[str, dict[str, Any]]:
    """Resolve only the verified canonical pointer generation carried by readiness."""

    readiness_report = dict(
        data_bundle.metadata.get("branch_data_readiness", {}) or {}
    )
    readiness_by_branch = dict(readiness_report.get("readiness", {}) or {})
    fundamental_readiness = dict(readiness_by_branch.get("fundamental", {}) or {})
    readiness_metadata = dict(fundamental_readiness.get("metadata", {}) or {})
    manifest = dict(readiness_metadata.get("manifest", {}) or {})

    generation_id = str(manifest.get("generation_id") or "").strip()
    storage_backend = str(manifest.get("storage_backend") or "").strip()
    readiness_status = str(
        fundamental_readiness.get("status") or ""
    ).strip().lower()
    readiness_source_priority = str(
        fundamental_readiness.get("source_priority") or ""
    ).strip()
    manifest_source_priority = str(
        manifest.get("source_priority") or ""
    ).strip()
    pit_status = str(fundamental_readiness.get("pit_status") or "").strip().lower()
    pointer_path = str(manifest.get("pointer_path") or "").strip()
    blockers: list[str] = []
    if not generation_id:
        blockers.append("canonical_generation_id_missing")
    if storage_backend != "parquet_canonical_generation":
        blockers.append("canonical_generation_backend_unconfirmed")
    if readiness_status not in {"pass", "warn"}:
        blockers.append("fundamental_readiness_not_eligible")
    if readiness_source_priority != "tushare_primary":
        blockers.append("fundamental_readiness_source_unconfirmed")
    if manifest_source_priority != "tushare_primary":
        blockers.append("fundamental_manifest_source_unconfirmed")
    if pit_status != "point_in_time":
        blockers.append("fundamental_pit_status_unconfirmed")
    if not pointer_path:
        blockers.append("canonical_fundamental_pointer_missing")

    pointer_verified = False
    resolved_pointer_path: Path | None = None
    if pointer_path:
        resolved_pointer_path = _resolve_readiness_fundamental_pointer(
            pointer_path
        )
        if resolved_pointer_path is not None:
            try:
                pointer = load_fundamental_pointer(resolved_pointer_path.parent)
                pointer_metadata = dict(
                    (pointer or {}).get("metadata", {}) or {}
                )
                generation_metadata = dict(
                    dict((pointer or {}).get("manifest", {}) or {}).get(
                        "metadata", {}
                    )
                    or {}
                )
                pointer_verified = bool(
                    pointer is not None
                    and str(pointer.get("generation_id") or "").strip()
                    == generation_id
                    and str(
                        pointer_metadata.get("source_priority") or ""
                    ).strip()
                    == "tushare_primary"
                    and str(
                        generation_metadata.get("source_priority") or ""
                    ).strip()
                    == "tushare_primary"
                    and pointer.get("primary_provenance_verified") is True
                    and Path(str(pointer.get("pointer_path") or "")).resolve()
                    == resolved_pointer_path.resolve()
                )
            except (FundamentalGenerationError, OSError, ValueError):
                pointer_verified = False
    if not pointer_verified:
        blockers.append("canonical_fundamental_pointer_unverified")

    confirmed = not blockers
    return (
        generation_id if confirmed else "",
        {
            "status": "confirmed" if confirmed else "UNCONFIRMED",
            "source": "canonical_fundamental_pointer" if confirmed else "UNCONFIRMED",
            "storage_backend": storage_backend or "UNCONFIRMED",
            "readiness_status": readiness_status or "UNCONFIRMED",
            "source_priority": (
                readiness_source_priority or "UNCONFIRMED"
            ),
            "pit_status": pit_status or "UNCONFIRMED",
            "pointer_bound": pointer_verified,
            "resolved_pointer_path": (
                str(resolved_pointer_path) if resolved_pointer_path is not None else ""
            ),
            "blockers": blockers,
        },
    )


def _resolve_readiness_fundamental_pointer(value: Any) -> Path | None:
    """Resolve readiness pointers without allowing relative-root escapes."""

    raw_text = str(value or "").strip()
    if not raw_text or raw_text.startswith("~"):
        return None
    try:
        raw_path = Path(raw_text)
        if raw_path.is_absolute():
            candidate = Path(raw_path.anchor)
            for part in raw_path.parts[1:]:
                candidate = candidate / part
                if candidate.is_symlink():
                    return None
            resolved = candidate.resolve(strict=True)
        else:
            if ".." in raw_path.parts:
                return None
            runtime_root = Path.cwd().resolve(strict=True)
            candidate = runtime_root
            for part in raw_path.parts:
                if part in {"", "."}:
                    continue
                candidate = candidate / part
                if candidate.is_symlink():
                    return None
            resolved = candidate.resolve(strict=True)
            if resolved != runtime_root and runtime_root not in resolved.parents:
                return None
        if (
            resolved.name != FUNDAMENTAL_POINTER_FILENAME
            or not resolved.is_file()
        ):
            return None
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


def _symbol_pit_lineage(
    snapshot_quality: Mapping[str, Any],
    *,
    canonical_generation_id: str,
) -> tuple[bool, dict[str, Any]]:
    canonical_generation_id = str(canonical_generation_id or "").strip()
    eligible_modules: list[str] = []
    invalid_modules: list[str] = []
    blockers: list[str] = []
    invalid_lineage_values = {"", "disabled", "neutral", "none", "unconfirmed", "unknown"}
    for module_name in ("financial_quality", "valuation", "forecast_revision"):
        quality = dict(snapshot_quality.get(module_name, {}) or {})
        status = str(quality.get("status") or "")
        pit_blockers = [
            str(item)
            for item in quality.get("pit_blockers", [])
            if str(item)
        ]
        if pit_blockers or str(quality.get("pit_status") or "").lower() == "blocked":
            invalid_modules.append(module_name)
            blockers.extend(
                f"{module_name}:{item}"
                for item in (pit_blockers or ["pit_status_blocked"])
            )
            continue
        if status != "available":
            continue
        requested = _normalized_date(quality.get("requested_as_of"))
        publish = _normalized_date(quality.get("publish_time"))
        effective = _normalized_date(quality.get("effective_time"))
        revision_id = str(quality.get("revision_id") or "").strip()
        revision_date = _normalized_date(revision_id)
        source_lineage = dict(quality.get("source_lineage", {}) or {})
        source = str(
            source_lineage.get("source")
            or source_lineage.get("provider")
            or quality.get("provider_name")
            or ""
        ).strip()
        provenance_source = str(
            source_lineage.get("provenance_source") or ""
        ).strip()
        source_priority = str(source_lineage.get("source_priority") or "").strip()
        lineage_generation_id = str(
            source_lineage.get("canonical_generation_id") or ""
        ).strip()
        module_blockers: list[str] = []
        if requested is None:
            module_blockers.append(f"{module_name}:requested_as_of_missing")
        if publish is None:
            module_blockers.append(f"{module_name}:publish_time_missing")
        if effective is None:
            module_blockers.append(f"{module_name}:effective_time_missing")
        if not revision_id:
            module_blockers.append(f"{module_name}:revision_id_missing")
        if (
            requested is not None
            and revision_date is not None
            and revision_date > requested
        ):
            module_blockers.append(f"{module_name}:revision_id_after_as_of")
        if (
            source.lower() in invalid_lineage_values
            or provenance_source.lower() in invalid_lineage_values
            or source_priority.lower() in invalid_lineage_values
        ):
            module_blockers.append(f"{module_name}:source_lineage_unconfirmed")
        if source_priority != "tushare_primary":
            module_blockers.append(
                f"{module_name}:source_priority_not_tushare_primary"
            )
        if requested is not None and publish is not None and publish > requested:
            module_blockers.append(f"{module_name}:publish_time_after_as_of")
        if requested is not None and effective is not None and effective > requested:
            module_blockers.append(f"{module_name}:effective_time_after_as_of")
        if publish is not None and effective is not None and publish > effective:
            module_blockers.append(f"{module_name}:publish_time_after_effective_time")
        if not lineage_generation_id:
            module_blockers.append(
                f"{module_name}:canonical_generation_id_missing"
            )
        elif lineage_generation_id != canonical_generation_id:
            module_blockers.append(
                f"{module_name}:canonical_generation_mismatch"
            )
        if module_blockers:
            invalid_modules.append(module_name)
            blockers.extend(module_blockers)
        else:
            eligible_modules.append(module_name)
    if not canonical_generation_id:
        blockers.append("canonical_generation_id_unconfirmed")
    confirmed = bool(
        canonical_generation_id
        and eligible_modules
        and not invalid_modules
        and not blockers
    )
    return (
        confirmed,
        {
            "status": "confirmed" if confirmed else "UNCONFIRMED",
            "canonical_generation_id": canonical_generation_id,
            "eligible_modules": eligible_modules,
            "invalid_modules": invalid_modules,
            "blockers": (
                []
                if confirmed
                else ["symbol_pit_lineage_unconfirmed", *blockers]
            ),
        },
    )


class BundleFundamentalDataLayer:
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
        source = str(row.get("source") or "").strip()
        source_priority = str(row.get("source_priority") or "").strip()
        pit_blockers = _pit_date_blockers(
            row=row,
            as_of=as_of,
            publish_fields=("availability_date",),
            effective_fields=("trade_date",),
        )
        if pit_blockers:
            return FundamentalSnapshot(
                symbol=symbol,
                as_of=str(as_of),
                available=False,
                source=source,
                publish_time=str(row.get("availability_date") or ""),
                effective_time=str(row.get("trade_date") or ""),
                revision_id=str(row.get("source_version") or ""),
                notes=list(pit_blockers),
                data_quality={
                    "status": "pit_blocked",
                    "provider_missing": False,
                    "snapshot_missing": False,
                    "missing_scope": "symbol",
                    "pit_status": "blocked",
                    "pit_blockers": list(pit_blockers),
                },
            )
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
        valuation_fields = [
            name
            for name in ("pe", "pb", "ps", "dividend_yield")
            if name in available_fields
        ]
        financial_fields = [
            name
            for name in available_fields
            if name not in {"pe", "pb", "ps", "dividend_yield"}
        ]
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
                "source_priority": source_priority,
                "available_fields": available_fields,
                "missing_fields": [name for name in field_map if name not in available_fields],
                "valuation_available": bool(valuation_fields),
                "valuation_snapshot_missing": not bool(valuation_fields),
                "field_coverage_ratio": round(len(available_fields) / len(field_map), 4),
            },
            provenance={
                "source": source,
                "source_priority": source_priority,
                "availability_date": row.get("availability_date", ""),
                "end_date": row.get("end_date", ""),
                "fin_fcf_to_profit": row.get("fin_fcf_to_profit"),
                "fcf_to_price": row.get("fcf_to_price"),
                "fundamental_generation_id": row.get(
                    "fundamental_generation_id", ""
                ),
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
        source = str(
            row.get("forecast_source") or row.get("source") or ""
        ).strip()
        source_priority = str(row.get("source_priority") or "").strip()
        pit_blockers = _pit_date_blockers(
            row=row,
            as_of=as_of,
            publish_fields=("forecast_ann_date", "availability_date"),
            effective_fields=("trade_date", "availability_date"),
        )
        if pit_blockers:
            return ForecastSnapshot(
                symbol=symbol,
                as_of=str(as_of),
                available=False,
                source=source,
                provider=source,
                publish_time=str(
                    row.get("forecast_ann_date") or row.get("availability_date") or ""
                ),
                effective_time=str(
                    row.get("trade_date") or row.get("availability_date") or ""
                ),
                revision_id=str(
                    row.get("forecast_ingest_run_id") or row.get("source_version") or ""
                ),
                notes=list(pit_blockers),
                data_quality={
                    "status": "pit_blocked",
                    "provider_missing": False,
                    "snapshot_missing": False,
                    "missing_scope": "symbol",
                    "pit_status": "blocked",
                    "pit_blockers": list(pit_blockers),
                },
            )
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
            effective_time=str(row.get("trade_date") or row.get("availability_date") or row.get("forecast_ann_date") or ""),
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
            provenance={
                "source": source,
                "source_priority": source_priority,
                "fundamental_generation_id": row.get(
                    "fundamental_generation_id", ""
                ),
            },
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


# Transitional private alias for existing callers; new governed replay code uses
# the explicit public name above.
_BundleFundamentalDataLayer = BundleFundamentalDataLayer


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
                data_layer = BundleFundamentalDataLayer(mart_records)
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
        generation_id, generation_evidence = _canonical_fundamental_generation(
            data_bundle
        )
        generation_symbols = list(snapshot_quality_by_symbol) or [
            str(symbol) for symbol in stock_pool
        ]
        symbol_pit_evidence: dict[str, dict[str, Any]] = {}
        fundamental_generations: dict[str, str] = {}
        generation_statuses: dict[str, str] = {}
        for symbol in generation_symbols:
            pit_confirmed, pit_evidence = _symbol_pit_lineage(
                dict(snapshot_quality_by_symbol.get(symbol, {}) or {}),
                canonical_generation_id=generation_id,
            )
            symbol_pit_evidence[symbol] = pit_evidence
            symbol_confirmed = bool(generation_id and pit_confirmed)
            fundamental_generations[symbol] = (
                generation_id if symbol_confirmed else ""
            )
            generation_statuses[symbol] = (
                "confirmed" if symbol_confirmed else "UNCONFIRMED"
            )
        generation_evidence["symbol_pit_evidence"] = symbol_pit_evidence
        generation_evidence["all_symbols_confirmed"] = all(
            value == "confirmed" for value in generation_statuses.values()
        )
        if not generation_evidence["all_symbols_confirmed"]:
            existing_reasons = [
                item
                for item in str(result.metadata.get("degraded_reason", "")).split(",")
                if item
            ]
            if "fundamental_generation_UNCONFIRMED" not in existing_reasons:
                existing_reasons.append("fundamental_generation_UNCONFIRMED")
            result.metadata["degraded_reason"] = ",".join(existing_reasons)
            result.coverage_notes = self.dedupe_texts(
                list(result.coverage_notes)
                + ["canonical fundamental generation is UNCONFIRMED; downstream overlays must fail closed."]
            )
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
                "fundamental_data_generation_status_by_symbol": generation_statuses,
                "fundamental_data_generation_evidence": generation_evidence,
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
