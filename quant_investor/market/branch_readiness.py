"""Three-branch data readiness contracts and offline assessment helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from quant_investor.market.fundamental_generation import (
    FundamentalGenerationError,
    load_fundamental_pointer,
    load_fundamental_table,
    resolve_fundamental_table_path,
)
from quant_investor.market.fundamental_provider_contract import (
    FUNDAMENTAL_DERIVATION_CONTRACT,
    FUNDAMENTAL_ENDPOINT_AUDIT_SCHEMA,
    FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA,
    FUNDAMENTAL_PROVIDER_MANIFEST_SCHEMA,
)
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.macro.contracts import parse_timestamp
from quant_investor.macro.release_calendar import (
    CriticalEventGapEvaluation,
    ReleaseCalendarEvidence,
    ReleaseCalendarGenerationProof,
    ReleaseReadinessEvaluation,
    SessionLagEvaluation,
    evaluate_release_readiness,
    is_validated_release_calendar_generation,
)
from quant_investor.versioning import BRANCH_SCHEMA_VERSION


STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_BLOCK = "block"
SOURCE_OFFICIAL = "official_primary"
SOURCE_OFFICIAL_FIRST = "official_first_mixed"
SOURCE_TUSHARE = "tushare_primary"
SOURCE_PUBLIC_FALLBACK = "public_structured_fallback"
SOURCE_OFFLINE = "manual_offline_snapshot"
SOURCE_PRIORITY_ORDER = (
    SOURCE_OFFICIAL,
    SOURCE_TUSHARE,
    SOURCE_PUBLIC_FALLBACK,
    SOURCE_OFFLINE,
)
_MACRO_SOURCE_PRIORITY_BY_SOURCE = {
    SOURCE_OFFICIAL_FIRST: SOURCE_OFFICIAL,
    SOURCE_TUSHARE: SOURCE_TUSHARE,
    SOURCE_PUBLIC_FALLBACK: SOURCE_PUBLIC_FALLBACK,
    SOURCE_OFFLINE: SOURCE_OFFLINE,
}
_MACRO_APPROVED_PRIMARY_PRIORITIES = frozenset(
    {SOURCE_OFFICIAL, SOURCE_TUSHARE}
)

DEFAULT_PARQUET_CN_ROOT = Path("data/parquet/cn")
DEFAULT_FUNDAMENTAL_ROOT = DEFAULT_PARQUET_CN_ROOT / "fundamental_daily"
DEFAULT_MACRO_ROOT = DEFAULT_PARQUET_CN_ROOT / "macro_daily"
DEFAULT_READINESS_ROOT = Path("reports/v15/branch_readiness")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_FROZEN_READINESS_ROOT = (_REPOSITORY_ROOT / "reports/branch_readiness").resolve()

QUANT_REQUIRED_FIELDS = ("open", "high", "low", "close", "volume", "amount")
FUNDAMENTAL_REQUIRED_FIELDS = (
    "fin_roe",
    "fin_roa",
    "fin_debt_to_assets",
    "fin_net_profit_yoy",
    "fin_ocf_to_profit",
    "fin_fcf_to_profit",
    "fcf_to_price",
    "forecast_revision",
)
FUNDAMENTAL_VALUE_AVAILABILITY_SCHEMA = (
    "cn-fundamental-value-availability.v1"
)
FUNDAMENTAL_PROVIDER_ENDPOINTS = frozenset(
    {
        "balancesheet",
        "cashflow",
        "daily_basic",
        "fina_indicator",
        "forecast",
        "income",
    }
)
FUNDAMENTAL_NULLABLE_VALUE_SEMANTICS = {
    "fin_roe": "issuer_value_absent_in_verified_response",
    "fin_roa": "issuer_value_absent_in_verified_response",
    "fin_debt_to_assets": "issuer_value_absent_in_verified_response",
    "fin_net_profit_yoy": "issuer_value_absent_in_verified_response",
    "fin_ocf_to_profit": "undefined_without_positive_profit_denominator",
    "fin_fcf_to_profit": "undefined_without_positive_profit_denominator",
    "fcf_to_price": "undefined_without_finite_fcf_and_positive_market_value",
    "forecast_revision": "no_qualifying_forecast_is_legitimate",
}
MACRO_REQUIRED_FIELDS = (
    "macro_score",
    "liquidity_score",
    "volatility_percentile",
    "policy_signal",
)
MACRO_READINESS_EVIDENCE_SCHEMA = "macro-readiness-evidence.v1"
MACRO_MAX_SESSION_LAG = 2
MACRO_RELEASE_IDENTITY_FIELDS = (
    "macro_release_calendar_generation_id",
    "macro_release_calendar_pointer_sha256",
    "macro_release_calendar_manifest_sha256",
    "macro_release_calendar_semantic_sha256",
    "macro_release_calendar_registry_sha256",
    "macro_release_calendar_plan_sha256",
    "macro_release_calendar_capture_manifest_sha256",
    "macro_release_calendar_market_open_days_sha256",
    "macro_release_calendar_critical_policy_sha256",
    "macro_readiness_evidence_semantic_sha256",
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(as_of: str = "") -> str:
    suffix = _date_text(as_of) or datetime.now().strftime("%Y%m%d")
    return f"branch_readiness_{suffix}_{datetime.now().strftime('%H%M%S')}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalized_utc_timestamp(value: Any) -> str:
    try:
        parsed = parse_timestamp(value, field_name="decision_cutoff_at")
    except (TypeError, ValueError) as exc:
        raise ValueError("macro_decision_cutoff_invalid") from exc
    return parsed.astimezone(timezone.utc).isoformat()


def _date_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _compact_date(value: Any) -> str:
    date = _date_text(value)
    return date.replace("-", "") if date else ""


@dataclass(frozen=True)
class MacroReadinessEvidence:
    """One immutable release-calendar evaluation pinned for a DAG decision."""

    schema_version: str
    market: str
    macro_logical_date: str
    target_session_date: str
    target_decision_cutoff_at: str
    max_session_lag: int
    macro_release_calendar_generation_id: str
    macro_release_calendar_pointer_sha256: str
    macro_release_calendar_manifest_sha256: str
    macro_release_calendar_semantic_sha256: str
    macro_release_calendar_registry_sha256: str
    macro_release_calendar_plan_sha256: str
    macro_release_calendar_capture_manifest_sha256: str
    macro_release_calendar_market_open_days_sha256: str
    macro_release_calendar_critical_policy_sha256: str
    validated_release_calendar_ancestry: tuple[
        ReleaseCalendarGenerationProof, ...
    ]
    evaluation: ReleaseReadinessEvaluation
    semantic_sha256: str

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "market": self.market,
            "macro_logical_date": self.macro_logical_date,
            "target_session_date": self.target_session_date,
            "target_decision_cutoff_at": self.target_decision_cutoff_at,
            "max_session_lag": self.max_session_lag,
            "macro_release_calendar_generation_id": (
                self.macro_release_calendar_generation_id
            ),
            "macro_release_calendar_pointer_sha256": (
                self.macro_release_calendar_pointer_sha256
            ),
            "macro_release_calendar_manifest_sha256": (
                self.macro_release_calendar_manifest_sha256
            ),
            "macro_release_calendar_semantic_sha256": (
                self.macro_release_calendar_semantic_sha256
            ),
            "macro_release_calendar_registry_sha256": (
                self.macro_release_calendar_registry_sha256
            ),
            "macro_release_calendar_plan_sha256": (
                self.macro_release_calendar_plan_sha256
            ),
            "macro_release_calendar_capture_manifest_sha256": (
                self.macro_release_calendar_capture_manifest_sha256
            ),
            "macro_release_calendar_market_open_days_sha256": (
                self.macro_release_calendar_market_open_days_sha256
            ),
            "macro_release_calendar_critical_policy_sha256": (
                self.macro_release_calendar_critical_policy_sha256
            ),
            "validated_release_calendar_ancestry": [
                asdict(item)
                for item in self.validated_release_calendar_ancestry
            ],
            "evaluation": asdict(self.evaluation),
        }

    def identity_binding(self) -> dict[str, str]:
        return {
            "macro_release_calendar_generation_id": (
                self.macro_release_calendar_generation_id
            ),
            "macro_release_calendar_pointer_sha256": (
                self.macro_release_calendar_pointer_sha256
            ),
            "macro_release_calendar_manifest_sha256": (
                self.macro_release_calendar_manifest_sha256
            ),
            "macro_release_calendar_semantic_sha256": (
                self.macro_release_calendar_semantic_sha256
            ),
            "macro_release_calendar_registry_sha256": (
                self.macro_release_calendar_registry_sha256
            ),
            "macro_release_calendar_plan_sha256": (
                self.macro_release_calendar_plan_sha256
            ),
            "macro_release_calendar_capture_manifest_sha256": (
                self.macro_release_calendar_capture_manifest_sha256
            ),
            "macro_release_calendar_market_open_days_sha256": (
                self.macro_release_calendar_market_open_days_sha256
            ),
            "macro_release_calendar_critical_policy_sha256": (
                self.macro_release_calendar_critical_policy_sha256
            ),
            "macro_readiness_evidence_semantic_sha256": self.semantic_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.semantic_payload(),
            "semantic_sha256": self.semantic_sha256,
        }


def build_macro_readiness_evidence(
    *,
    release_calendar_evidence: ReleaseCalendarEvidence,
    macro_logical_date: str,
    target_session_date: str,
    target_decision_cutoff_at: str | datetime,
) -> MacroReadinessEvidence:
    """Pin one pure release-calendar decision without loading mutable state."""

    if type(release_calendar_evidence) is not ReleaseCalendarEvidence:
        raise TypeError("release_calendar_evidence_exact_type_required")
    logical_date = _date_text(macro_logical_date)
    target_date = _date_text(target_session_date)
    if not logical_date:
        raise ValueError("macro_logical_date_invalid")
    if not target_date:
        raise ValueError("macro_target_session_date_invalid")
    cutoff = _normalized_utc_timestamp(target_decision_cutoff_at)
    evaluation = evaluate_release_readiness(
        release_calendar_evidence,
        macro_logical_date=logical_date,
        target_session_date=target_date,
        decision_cutoff_at=cutoff,
        max_session_lag=MACRO_MAX_SESSION_LAG,
    )
    identity = release_calendar_evidence.identity
    if not is_validated_release_calendar_generation(
        release_calendar_evidence,
        generation_id=identity.generation_id,
        pointer_sha256=identity.pointer_sha256,
        manifest_sha256=identity.manifest_sha256,
        semantic_sha256=identity.semantic_sha256,
        plan_sha256=release_calendar_evidence.plan_sha256,
        capture_manifest_sha256=(
            release_calendar_evidence.capture_manifest_sha256
        ),
        market_open_days_sha256=(
            release_calendar_evidence.market_open_days_sha256
        ),
        registry_sha256=release_calendar_evidence.registry_sha256,
        critical_policy_sha256=(
            release_calendar_evidence.critical_policy_sha256
        ),
    ):
        raise ValueError("release_calendar_evidence_ancestry_invalid")
    candidate = MacroReadinessEvidence(
        schema_version=MACRO_READINESS_EVIDENCE_SCHEMA,
        market="CN",
        macro_logical_date=logical_date,
        target_session_date=target_date,
        target_decision_cutoff_at=cutoff,
        max_session_lag=MACRO_MAX_SESSION_LAG,
        macro_release_calendar_generation_id=identity.generation_id,
        macro_release_calendar_pointer_sha256=identity.pointer_sha256,
        macro_release_calendar_manifest_sha256=identity.manifest_sha256,
        macro_release_calendar_semantic_sha256=identity.semantic_sha256,
        macro_release_calendar_registry_sha256=(
            release_calendar_evidence.registry_sha256
        ),
        macro_release_calendar_plan_sha256=release_calendar_evidence.plan_sha256,
        macro_release_calendar_capture_manifest_sha256=(
            release_calendar_evidence.capture_manifest_sha256
        ),
        macro_release_calendar_market_open_days_sha256=(
            release_calendar_evidence.market_open_days_sha256
        ),
        macro_release_calendar_critical_policy_sha256=(
            release_calendar_evidence.critical_policy_sha256
        ),
        validated_release_calendar_ancestry=(
            release_calendar_evidence.validated_ancestry
        ),
        evaluation=evaluation,
        semantic_sha256="",
    )
    return replace(
        candidate,
        semantic_sha256=_canonical_json_sha256(candidate.semantic_payload()),
    )


def _normalize_symbol(symbol: Any) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        code, suffix = text.split(".", 1)
        return f"{code.zfill(6)}.{suffix}"
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        code = digits[:6]
        suffix = "SH" if code.startswith(("6", "9")) else "SZ"
        return f"{code}.{suffix}"
    return text


def _source_priority(source: str = "", explicit: str = "") -> str:
    priority = str(explicit or "").strip()
    if priority in SOURCE_PRIORITY_ORDER:
        return priority
    source_text = str(source or "").lower()
    if "tushare" in source_text:
        return SOURCE_TUSHARE
    if any(token in source_text for token in ("akshare", "public", "rss", "official")):
        return SOURCE_PUBLIC_FALLBACK
    return SOURCE_OFFLINE


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "nat", "none"}


def _latest_frame_date(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return ""
    for column in ("trade_date", "date"):
        if column not in frame.columns:
            continue
        values = [_compact_date(item) for item in frame[column].tail(256)]
        values = [item for item in values if item]
        if values:
            return max(values)
    return ""


@dataclass
class BranchDataReadiness:
    branch: str
    status: str = STATUS_BLOCK
    coverage_ratio: float = 0.0
    freshness_status: str = "unknown"
    pit_status: str = "unknown"
    source_priority: str = SOURCE_OFFLINE
    source: str = ""
    as_of: str = ""
    required_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    affected_symbols: list[str] = field(default_factory=list)
    fallback_used: bool = False
    provider_status: str = "not_requested"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))


@dataclass
class BranchGovernanceReport:
    run_id: str
    market: str
    category: str
    as_of: str
    generated_at: str = field(default_factory=_now_utc)
    readiness: dict[str, BranchDataReadiness] = field(default_factory=dict)
    blocked_symbols: list[str] = field(default_factory=list)
    quantifiable_universe: list[str] = field(default_factory=list)
    investable_universe: list[str] = field(default_factory=list)
    branch_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_branch_data: bool = True) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "market": self.market,
            "category": self.category,
            "as_of": self.as_of,
            "generated_at": self.generated_at,
            "readiness": {key: value.to_dict() for key, value in self.readiness.items()},
            "blocked_symbols": list(self.blocked_symbols),
            "quantifiable_universe": list(self.quantifiable_universe),
            "investable_universe": list(self.investable_universe),
            "metadata": dict(self.metadata),
        }
        if include_branch_data:
            payload["branch_data"] = dict(_json_safe(self.branch_data))
        return dict(_json_safe(payload))


def assess_quant_readiness(
    *,
    frames: Mapping[str, pd.DataFrame],
    read_results: Mapping[str, Any] | None = None,
    symbols: Sequence[str] | None = None,
    as_of: str = "",
) -> BranchDataReadiness:
    universe = [_normalize_symbol(symbol) for symbol in (symbols or frames.keys()) if _normalize_symbol(symbol)]
    target_date = _compact_date(as_of)
    pass_symbols: list[str] = []
    affected: list[str] = []
    missing_fields: set[str] = set()
    blockers: list[str] = []
    latest_dates: dict[str, str] = {}
    for symbol in universe:
        frame = frames.get(symbol)
        if frame is None and "." in symbol:
            frame = frames.get(symbol.split(".", 1)[0])
        if frame is None or frame.empty:
            affected.append(symbol)
            blockers.append("quant_empty_frame")
            continue
        columns = {str(column).strip().lower() for column in frame.columns}
        aliases = {"volume": {"volume", "vol"}, "amount": {"amount", "amt"}}
        symbol_missing: list[str] = []
        for field_name in QUANT_REQUIRED_FIELDS:
            allowed = aliases.get(field_name, {field_name})
            if not columns.intersection(allowed):
                symbol_missing.append(field_name)
                missing_fields.add(field_name)
        latest = _latest_frame_date(frame)
        latest_dates[symbol] = latest
        if target_date and latest != target_date:
            symbol_missing.append("freshness")
            missing_fields.add("freshness")
        if symbol_missing:
            affected.append(symbol)
            blockers.append(f"quant_missing_or_stale:{symbol}")
            continue
        pass_symbols.append(symbol)
    coverage = len(pass_symbols) / max(len(universe), 1)
    unique_blockers = sorted(set(blockers))
    status = STATUS_PASS if coverage >= 0.995 and not unique_blockers else STATUS_BLOCK
    if not universe:
        unique_blockers.append("empty_quant_universe")
        status = STATUS_BLOCK
    freshness = (
        "fresh"
        if not target_date
        or (
            len(latest_dates) == len(universe)
            and all(latest == target_date for latest in latest_dates.values())
        )
        else "stale"
    )
    if read_results:
        issue_count = sum(len(getattr(result, "issues", []) or []) for result in read_results.values())
        if issue_count:
            unique_blockers.append("market_data_read_diagnostics_present")
            status = STATUS_BLOCK
    return BranchDataReadiness(
        branch="quant",
        status=status,
        coverage_ratio=coverage,
        freshness_status=freshness,
        pit_status="daily_bar_snapshot",
        source_priority=SOURCE_TUSHARE,
        source="strict_parquet_canonical_bars",
        as_of=_date_text(as_of),
        required_fields=list(QUANT_REQUIRED_FIELDS),
        missing_fields=sorted(missing_fields),
        blockers=unique_blockers,
        affected_symbols=sorted(set(affected)),
        fallback_used=False,
        provider_status="strict_parquet_snapshot",
        metadata={"latest_dates": latest_dates, "symbol_count": len(universe), "pass_count": len(pass_symbols)},
    )


def _read_latest_manifest(root: Path) -> dict[str, Any]:
    path = root / "latest_manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _resolve_parquet_table_root(root: str | Path, table_name: str) -> Path:
    path = Path(root).expanduser()
    if path.suffix.lower() == ".parquet":
        return path
    if table_name == "fundamental_daily":
        if load_fundamental_pointer(path) is not None:
            return resolve_fundamental_table_path(path, table_name)
    if (path / "part.parquet").exists():
        return path
    if table_name == "fundamental_daily":
        return resolve_fundamental_table_path(path, table_name)
    if path.name == table_name:
        return path
    return path / table_name


def _read_parquet_table(table_path: Path) -> pd.DataFrame:
    if not table_path.exists():
        return pd.DataFrame()
    if table_path.is_file():
        try:
            return pd.read_parquet(table_path)
        except Exception:
            return pd.DataFrame()
    part_path = table_path / "part.parquet"
    if part_path.exists():
        try:
            return pd.read_parquet(part_path)
        except Exception:
            return pd.DataFrame()
    try:
        return pd.read_parquet(table_path)
    except Exception:
        frames: list[pd.DataFrame] = []
        for path in sorted(table_path.rglob("*.parquet")):
            try:
                frames.append(pd.read_parquet(path))
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def _manifest_from_parquet(table_name: str, frame: pd.DataFrame, table_path: Path) -> dict[str, Any]:
    if frame.empty:
        return {}
    source = "parquet_canonical"
    if "source" in frame.columns:
        sources = frame["source"].dropna().astype(str).str.strip()
        source = str(sources.iloc[0]) if not sources.empty else source
    source_priority = ""
    if "source_priority" in frame.columns:
        priorities = frame["source_priority"].dropna().astype(str).str.strip()
        source_priority = str(priorities.iloc[0]) if not priorities.empty else ""
    return {
        "provider_status": source,
        "source": source,
        "source_priority": source_priority or _source_priority(source),
        "storage_backend": "parquet_canonical",
        "table": table_name,
        "table_path": str(table_path),
        "daily_rows": int(len(frame)),
    }


def _latest_records_by_symbol(
    frame: pd.DataFrame,
    *,
    symbols: Sequence[str],
    as_of: str,
) -> dict[str, dict[str, Any]]:
    if frame.empty or "ts_code" not in frame.columns:
        return {}
    working = frame.copy()
    working["ts_code"] = working["ts_code"].map(_normalize_symbol)
    if "trade_date" not in working.columns:
        working["trade_date"] = working.get("date", "")
    working["_trade_date"] = working["trade_date"].map(_compact_date)
    target = _compact_date(as_of)
    if target:
        working = working[working["_trade_date"].astype(str) <= target]
    records: dict[str, dict[str, Any]] = {}
    wanted = {_normalize_symbol(symbol) for symbol in symbols}
    for symbol, group in working[working["ts_code"].isin(wanted)].groupby("ts_code", sort=True):
        row = group.sort_values("_trade_date").iloc[-1].to_dict()
        row.pop("_trade_date", None)
        records[str(symbol)] = dict(_json_safe(row))
    return records


def _fundamental_value_availability_contract(
    pointer: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize when nullable Fundamental values are verified observations.

    Required field names are a structural schema contract.  Numeric values may
    still be unavailable because a ratio is mathematically undefined or an
    issuer returned no qualifying forecast.  That distinction is trusted only
    for an independently validated authoritative v3 generation.
    """

    manifest = dict(pointer.get("manifest", {}) or {})
    metadata = dict(manifest.get("metadata", {}) or {})
    provider_manifest = dict(metadata.get("provider_manifest", {}) or {})
    endpoint_audit = dict(provider_manifest.get("endpoint_audit", {}) or {})
    endpoint_payloads = dict(endpoint_audit.get("endpoints", {}) or {})
    checkpoint = dict(provider_manifest.get("checkpoint", {}) or {})
    derivation = dict(provider_manifest.get("derivation", {}) or {})

    def _nonnegative_int(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if value >= 0 else None

    endpoint_summary: dict[str, dict[str, Any]] = {}
    for endpoint_name, raw_payload in sorted(endpoint_payloads.items()):
        payload = dict(raw_payload or {}) if isinstance(raw_payload, Mapping) else {}
        endpoint_summary[str(endpoint_name)] = {
            "passed": payload.get("passed") is True,
            "request_denominator": _nonnegative_int(
                payload.get("request_denominator")
            ),
            "accounted": _nonnegative_int(payload.get("accounted")),
            "success": _nonnegative_int(payload.get("success")),
            "empty": _nonnegative_int(payload.get("empty")),
            "error": _nonnegative_int(payload.get("error")),
            "malformed": _nonnegative_int(payload.get("malformed")),
            "financial_coverage_failed": _nonnegative_int(
                payload.get("financial_coverage_failed")
            ),
            "legitimate_empty_allowed": (
                payload.get("legitimate_empty_allowed") is True
            ),
        }

    symbol_denominator = _nonnegative_int(
        endpoint_audit.get("symbol_denominator")
    )
    request_denominator = _nonnegative_int(
        endpoint_audit.get("request_denominator")
    )
    requests_accounted = _nonnegative_int(
        endpoint_audit.get("requests_accounted")
    )
    requests_error = _nonnegative_int(
        endpoint_audit.get("requests_error")
    )
    requests_malformed = _nonnegative_int(
        endpoint_audit.get("requests_malformed")
    )
    provider_requests_attempted = _nonnegative_int(
        provider_manifest.get("requests_attempted")
    )
    provider_requests_failed = _nonnegative_int(
        provider_manifest.get("requests_failed")
    )
    provider_requests_malformed = _nonnegative_int(
        provider_manifest.get("requests_malformed")
    )
    endpoint_count_names = (
        "request_denominator",
        "accounted",
        "success",
        "empty",
        "error",
        "malformed",
        "financial_coverage_failed",
    )
    endpoint_counts_valid = bool(endpoint_summary) and all(
        item[count_name] is not None
        for item in endpoint_summary.values()
        for count_name in endpoint_count_names
    )
    endpoint_denominators_reconciled = bool(
        endpoint_counts_valid
        and symbol_denominator
        and request_denominator
        == symbol_denominator * len(FUNDAMENTAL_PROVIDER_ENDPOINTS)
        and all(
            item["request_denominator"] == symbol_denominator
            for item in endpoint_summary.values()
        )
    )
    endpoint_accounting_reconciled = bool(
        endpoint_counts_valid
        and requests_accounted is not None
        and requests_error is not None
        and requests_malformed is not None
        and all(
            item["accounted"]
            == item["success"]
            + item["empty"]
            + item["error"]
            + item["malformed"]
            + item["financial_coverage_failed"]
            == item["request_denominator"]
            for item in endpoint_summary.values()
        )
        and sum(item["accounted"] for item in endpoint_summary.values())
        == requests_accounted
        and sum(item["error"] for item in endpoint_summary.values())
        == requests_error
        and sum(item["malformed"] for item in endpoint_summary.values())
        == requests_malformed
    )
    provider_accounting_reconciled = bool(
        provider_requests_attempted is not None
        and provider_requests_failed is not None
        and provider_requests_malformed is not None
        and provider_requests_attempted == request_denominator
        and provider_requests_failed == requests_error
        and provider_requests_malformed == requests_malformed
    )
    checks = {
        "primary_provenance_verified": (
            pointer.get("primary_provenance_verified") is True
        ),
        "gate2_passed": (
            dict(pointer.get("metadata", {}) or {}).get("gate2_passed")
            is True
        ),
        "provider_manifest_v3": (
            provider_manifest.get("schema_version")
            == FUNDAMENTAL_PROVIDER_MANIFEST_SCHEMA
        ),
        "authoritative_full_rebuild": (
            provider_manifest.get("authoritative_full_rebuild") is True
        ),
        "endpoint_audit_v3": (
            endpoint_audit.get("schema_version")
            == FUNDAMENTAL_ENDPOINT_AUDIT_SCHEMA
        ),
        "endpoint_audit_passed": endpoint_audit.get("passed") is True,
        "endpoint_audit_blockers_empty": not list(
            endpoint_audit.get("blockers", []) or []
        ),
        "endpoint_set_complete": (
            set(endpoint_payloads) == FUNDAMENTAL_PROVIDER_ENDPOINTS
        ),
        "all_endpoints_passed": bool(endpoint_summary)
        and all(item["passed"] for item in endpoint_summary.values()),
        "endpoint_counts_valid": endpoint_counts_valid,
        "endpoint_denominators_reconciled": (
            endpoint_denominators_reconciled
        ),
        "endpoint_accounting_reconciled": (
            endpoint_accounting_reconciled
        ),
        "request_accounting_complete": request_denominator is not None
        and requests_accounted == request_denominator,
        "provider_accounting_reconciled": provider_accounting_reconciled,
        "provider_request_failures_zero": (
            provider_requests_failed == 0
            and provider_requests_malformed == 0
            and requests_error == 0
            and requests_malformed == 0
        ),
        "checkpoint_v3": (
            checkpoint.get("schema_version")
            == FUNDAMENTAL_FETCH_CHECKPOINT_SCHEMA
        ),
        "derivation_v3": (
            derivation.get("contract_version")
            == FUNDAMENTAL_DERIVATION_CONTRACT
        ),
        "forecast_empty_is_legitimate": (
            endpoint_audit.get("forecast_empty_is_legitimate") is True
            and endpoint_summary.get("forecast", {}).get(
                "legitimate_empty_allowed"
            )
            is True
        ),
    }
    blockers = sorted(name for name, passed in checks.items() if not passed)
    verified = not blockers
    return {
        "schema_version": FUNDAMENTAL_VALUE_AVAILABILITY_SCHEMA,
        "status": "verified" if verified else "unverified",
        "nullable_values_allowed": verified,
        "checks": checks,
        "blockers": blockers,
        "symbol_denominator": symbol_denominator,
        "request_denominator": request_denominator,
        "requests_accounted": requests_accounted,
        "requests_error": requests_error,
        "requests_malformed": requests_malformed,
        "provider_requests_attempted": provider_requests_attempted,
        "provider_requests_failed": provider_requests_failed,
        "provider_requests_malformed": provider_requests_malformed,
        "endpoint_summary": endpoint_summary,
        "field_semantics": dict(FUNDAMENTAL_NULLABLE_VALUE_SEMANTICS),
    }


def load_fundamental_records(
    symbols: Sequence[str],
    *,
    as_of: str = "",
    root: str | Path = DEFAULT_FUNDAMENTAL_ROOT,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pointer = load_fundamental_pointer(root)
    if pointer is not None:
        frame, pointer = load_fundamental_table(
            root,
            "fundamental_daily",
        )
        table_path = Path(
            str(
                dict(pointer.get("tables", {}) or {}).get(
                    "fundamental_daily", ""
                )
            )
        )
    else:
        table_path = _resolve_parquet_table_root(root, "fundamental_daily")
        frame = _read_parquet_table(table_path)
    manifest = (
        dict(pointer.get("metadata", {}) or {})
        if pointer is not None
        else _read_latest_manifest(table_path)
    ) or _manifest_from_parquet("fundamental_daily", frame, table_path)
    if pointer is not None:
        manifest.update(
            {
                "generation_id": pointer.get("generation_id"),
                "pointer_path": pointer.get("pointer_path"),
                "storage_backend": "parquet_canonical_generation",
                "value_availability_contract": (
                    _fundamental_value_availability_contract(pointer)
                ),
            }
        )
    records = _latest_records_by_symbol(
        frame,
        symbols=symbols,
        as_of=as_of,
    )
    if pointer is not None:
        generation_id = str(pointer.get("generation_id") or "").strip()
        if not generation_id:
            raise FundamentalGenerationError(
                "fundamental canonical generation_id missing"
            )
        generation_source_priority = str(
            dict(pointer.get("metadata", {}) or {}).get(
                "source_priority"
            )
            or ""
        ).strip()
        for symbol, record in records.items():
            declared = str(
                record.get("fundamental_generation_id") or ""
            ).strip()
            if declared and declared != generation_id:
                raise FundamentalGenerationError(
                    "fundamental row generation mismatch: " + symbol
                )
            declared_source_priority = str(
                record.get("source_priority") or ""
            ).strip()
            if (
                declared_source_priority
                and declared_source_priority != generation_source_priority
            ):
                raise FundamentalGenerationError(
                    "fundamental row source priority mismatch: " + symbol
                )
            record["fundamental_generation_id"] = generation_id
            record["source_priority"] = generation_source_priority
    return records, manifest


def load_macro_record(
    *,
    as_of: str = "",
    root: str | Path = DEFAULT_MACRO_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Macro has a stricter generation contract than generic logical tables.
    # Import lazily because the mart writer uses this module's source constants.
    from quant_investor.market.macro_mart import (
        MacroMartPromotionError,
        read_macro_mart,
    )

    try:
        frame, manifest = read_macro_mart(data_root=root)
    except (MacroMartPromotionError, OSError, ValueError) as exc:
        return {}, {
            "read_error": str(exc) or "macro_catalog_generation_invalid"
        }
    if frame.empty:
        return {}, manifest
    working = frame.copy()
    if "trade_date" not in working.columns:
        working["trade_date"] = working.get("date", "")
    working["_trade_date"] = working["trade_date"].map(_compact_date)
    target = _compact_date(as_of)
    if target:
        working = working[working["_trade_date"].astype(str) <= target]
    if working.empty:
        return {}, manifest
    row = working.sort_values("_trade_date").iloc[-1].to_dict()
    row.pop("_trade_date", None)
    return dict(_json_safe(row)), manifest


def macro_generation_identity(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return the immutable canonical Macro identity used by DAG consumers."""

    identity = {
        "generation_id": str(manifest.get("generation_id") or "").strip(),
        "parquet_sha256": str(manifest.get("parquet_sha256") or "").strip(),
        "generation_manifest_sha256": str(
            manifest.get("generation_manifest_sha256") or ""
        ).strip(),
    }
    for field_name in (
        "v15_controls_sha256",
        "v15_controls_semantic_sha256",
        *MACRO_RELEASE_IDENTITY_FIELDS,
    ):
        field_value = str(manifest.get(field_name) or "").strip()
        if field_value:
            identity[field_name] = field_value
    return identity


def _macro_release_binding_from_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    return {
        field_name: str(manifest.get(field_name) or "").strip()
        for field_name in MACRO_RELEASE_IDENTITY_FIELDS
        if field_name != "macro_readiness_evidence_semantic_sha256"
    }


def _macro_release_binding_from_proof(
    proof: ReleaseCalendarGenerationProof,
) -> dict[str, str]:
    return {
        "macro_release_calendar_generation_id": proof.generation_id,
        "macro_release_calendar_pointer_sha256": proof.pointer_sha256,
        "macro_release_calendar_manifest_sha256": proof.manifest_sha256,
        "macro_release_calendar_semantic_sha256": proof.semantic_sha256,
        "macro_release_calendar_registry_sha256": proof.registry_sha256,
        "macro_release_calendar_plan_sha256": proof.plan_sha256,
        "macro_release_calendar_capture_manifest_sha256": (
            proof.capture_manifest_sha256
        ),
        "macro_release_calendar_market_open_days_sha256": (
            proof.market_open_days_sha256
        ),
        "macro_release_calendar_critical_policy_sha256": (
            proof.critical_policy_sha256
        ),
    }


def _macro_readiness_evidence_blockers(
    *,
    evidence: MacroReadinessEvidence | None,
    manifest: Mapping[str, Any],
    macro_logical_date: str,
    target_session_date: str,
    decision_cutoff_at: str | datetime | None,
) -> tuple[list[str], bool]:
    """Validate pinned release evidence without filesystem I/O."""

    if evidence is None:
        return ["macro_release_readiness_evidence_missing"], False
    if type(evidence) is not MacroReadinessEvidence:
        return ["macro_release_readiness_evidence_type_invalid"], False
    blockers: list[str] = []
    if (
        evidence.schema_version != MACRO_READINESS_EVIDENCE_SCHEMA
        or evidence.market != "CN"
        or evidence.max_session_lag != MACRO_MAX_SESSION_LAG
    ):
        blockers.append("macro_release_readiness_evidence_contract_invalid")
    try:
        semantic_sha256 = _canonical_json_sha256(evidence.semantic_payload())
    except (TypeError, ValueError):
        semantic_sha256 = ""
    if not semantic_sha256 or evidence.semantic_sha256 != semantic_sha256:
        blockers.append("macro_release_readiness_evidence_tampered")
    logical_date = _date_text(macro_logical_date)
    target_date = _date_text(target_session_date)
    if (
        evidence.macro_logical_date != logical_date
        or evidence.target_session_date != target_date
    ):
        blockers.append("macro_release_readiness_evidence_target_mismatch")
    if decision_cutoff_at in (None, ""):
        blockers.append("macro_decision_cutoff_missing")
    else:
        try:
            expected_cutoff = _normalized_utc_timestamp(decision_cutoff_at)
        except ValueError:
            blockers.append("macro_decision_cutoff_invalid")
        else:
            if evidence.target_decision_cutoff_at != expected_cutoff:
                blockers.append(
                    "macro_release_readiness_evidence_cutoff_mismatch"
                )

    raw_ancestry = evidence.validated_release_calendar_ancestry
    ancestry = raw_ancestry if type(raw_ancestry) is tuple else ()
    binding = _macro_release_binding_from_manifest(manifest)
    if any(not value for value in binding.values()):
        blockers.append("macro_release_calendar_binding_missing")
    elif tuple(sorted(binding.items())) not in {
        tuple(sorted(_macro_release_binding_from_proof(proof).items()))
        for proof in ancestry
        if type(proof) is ReleaseCalendarGenerationProof
    }:
        # Compare exact complete source identities; never infer lineage from
        # generation names or a single parent pointer.
        blockers.append("macro_release_calendar_identity_mismatch")

    sha_fields = {
        field_name: value
        for field_name, value in evidence.identity_binding().items()
        if field_name != "macro_release_calendar_generation_id"
    }
    if (
        not evidence.macro_release_calendar_generation_id
        or any(not _is_sha256(value) for value in sha_fields.values())
    ):
        blockers.append("macro_release_readiness_evidence_identity_invalid")
    current_binding = evidence.identity_binding()
    current_binding.pop("macro_readiness_evidence_semantic_sha256")
    if (
        not ancestry
        or any(
            type(item) is not ReleaseCalendarGenerationProof
            for item in ancestry
        )
        or any(
            type(item.generation_id) is not str
            or not item.generation_id
            or any(
                not _is_sha256(value)
                for value in (
                    item.pointer_sha256,
                    item.manifest_sha256,
                    item.semantic_sha256,
                    item.plan_sha256,
                    item.capture_manifest_sha256,
                    item.market_open_days_sha256,
                    item.registry_sha256,
                    item.critical_policy_sha256,
                )
            )
            for item in ancestry
            if type(item) is ReleaseCalendarGenerationProof
        )
        or _macro_release_binding_from_proof(ancestry[-1]) != current_binding
        or len({item.generation_id for item in ancestry}) != len(ancestry)
    ):
        blockers.append("macro_release_readiness_ancestry_invalid")

    evaluation = evidence.evaluation
    if type(evaluation) is not ReleaseReadinessEvaluation:
        blockers.append("macro_release_readiness_evaluation_type_invalid")
        return list(dict.fromkeys(blockers)), False
    lag = evaluation.session_lag
    gap = evaluation.critical_event_gap
    if (
        type(lag) is not SessionLagEvaluation
        or type(gap) is not CriticalEventGapEvaluation
    ):
        blockers.append("macro_release_readiness_evaluation_type_invalid")
        return list(dict.fromkeys(blockers)), False
    if (
        type(evaluation.ready) is not bool
        or type(lag.ready) is not bool
        or type(gap.ready) is not bool
        or type(evaluation.blockers) is not tuple
        or type(lag.blockers) is not tuple
        or type(gap.blockers) is not tuple
        or any(
            type(item) is not str
            for item in (
                *evaluation.blockers,
                *lag.blockers,
                *gap.blockers,
            )
        )
    ):
        blockers.append("macro_release_readiness_evaluation_contract_invalid")
        return list(dict.fromkeys(blockers)), False
    try:
        expected_window_start = _normalized_utc_timestamp(
            f"{logical_date}T15:00:00+08:00"
        )
    except ValueError:
        expected_window_start = ""
    evaluation_consistent = (
        lag.macro_logical_date == logical_date
        and lag.target_session_date == target_date
        and lag.session_lag is not None
        and type(lag.session_lag) is int
        and 0 <= lag.session_lag <= MACRO_MAX_SESSION_LAG
        and gap.window_start_exclusive == expected_window_start
        and gap.window_end_inclusive == evidence.target_decision_cutoff_at
        and lag.ready == (not lag.blockers)
        and gap.ready == (not gap.blockers)
        and evaluation.ready
        == (lag.ready and gap.ready and not evaluation.blockers)
        and tuple(evaluation.blockers)
        == tuple(dict.fromkeys((*lag.blockers, *gap.blockers)))
    )
    if not evaluation_consistent:
        blockers.append("macro_release_readiness_evaluation_inconsistent")
    if not evaluation.ready:
        blockers.append("macro_release_readiness_blocked")
        blockers.extend(str(item) for item in evaluation.blockers)
    return list(dict.fromkeys(blockers)), not blockers


def _assess_symbol_records(
    *,
    branch: str,
    symbols: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    required_fields: Sequence[str],
    manifest: Mapping[str, Any],
    as_of: str,
) -> BranchDataReadiness:
    universe = [_normalize_symbol(symbol) for symbol in symbols if _normalize_symbol(symbol)]
    affected: list[str] = []
    partial_symbols: list[str] = []
    structurally_missing_fields: set[str] = set()
    unavailable_value_fields: set[str] = set()
    schema_pass_count = 0
    value_complete_count = 0
    record_count = 0
    value_empty_symbols: list[str] = []
    field_schema_counts = {field_name: 0 for field_name in required_fields}
    field_value_counts = {field_name: 0 for field_name in required_fields}
    for symbol in universe:
        record = dict(records.get(symbol, {}) or {})
        if not record:
            symbol_structural_missing = list(required_fields) + ["record"]
            symbol_value_unavailable = list(required_fields)
            affected.append(symbol)
        else:
            record_count += 1
            symbol_structural_missing = [
                field_name
                for field_name in required_fields
                if field_name not in record
            ]
            symbol_value_unavailable = [
                field_name
                for field_name in required_fields
                if field_name in record
                and not _is_present(record.get(field_name))
            ]
            for field_name in required_fields:
                if field_name in record:
                    field_schema_counts[field_name] += 1
                if _is_present(record.get(field_name)):
                    field_value_counts[field_name] += 1
            if not symbol_structural_missing:
                schema_pass_count += 1
            else:
                affected.append(symbol)
            if symbol_value_unavailable:
                partial_symbols.append(symbol)
            else:
                value_complete_count += 1
            if not any(
                _is_present(record.get(field_name))
                for field_name in required_fields
            ):
                value_empty_symbols.append(symbol)
                affected.append(symbol)
        if symbol_structural_missing:
            structurally_missing_fields.update(symbol_structural_missing)
        if symbol_value_unavailable:
            unavailable_value_fields.update(symbol_value_unavailable)
    full_coverage = schema_pass_count / max(len(universe), 1)
    record_coverage = record_count / max(len(universe), 1)
    full_value_observation_ratio = value_complete_count / max(
        len(universe), 1
    )
    source = str(manifest.get("provider_status") or manifest.get("source") or "parquet_canonical")
    priority = _source_priority(source, str(manifest.get("source_priority", "")))
    fallback_used = priority != SOURCE_TUSHARE
    value_availability_contract = dict(
        manifest.get("value_availability_contract", {}) or {}
    )
    nullable_values_accepted = (
        value_availability_contract.get("schema_version")
        == FUNDAMENTAL_VALUE_AVAILABILITY_SCHEMA
        and value_availability_contract.get("status") == "verified"
        and value_availability_contract.get("nullable_values_allowed") is True
    )
    blockers = []
    if not records:
        blockers.append(f"{branch}_parquet_table_missing_or_empty")
    if structurally_missing_fields:
        blockers.append(f"{branch}_required_fields_missing")
    if value_empty_symbols:
        blockers.append(f"{branch}_required_values_unavailable")
    if fallback_used:
        blockers.append(f"{branch}_not_tushare_primary")
    if blockers:
        status = STATUS_BLOCK
    elif partial_symbols and not nullable_values_accepted:
        status = STATUS_WARN
    else:
        status = STATUS_PASS if full_coverage >= 1.0 else STATUS_WARN
    return BranchDataReadiness(
        branch=branch,
        status=status,
        coverage_ratio=record_coverage if status == STATUS_WARN else full_coverage,
        freshness_status="fresh_or_pit_asof" if records else "unknown",
        pit_status="point_in_time" if records else "missing",
        source_priority=priority,
        source=source,
        as_of=_date_text(as_of),
        required_fields=list(required_fields),
        missing_fields=sorted(structurally_missing_fields),
        blockers=blockers,
        affected_symbols=sorted(set(affected)),
        fallback_used=fallback_used,
        provider_status=str(manifest.get("provider_status") or "local_snapshot"),
        metadata={
            "manifest": dict(manifest),
            "symbol_count": len(universe),
            "record_count": record_count,
            "pass_count": schema_pass_count,
            "value_complete_count": value_complete_count,
            "partial_symbols": sorted(set(partial_symbols)),
            "value_empty_symbols": sorted(set(value_empty_symbols)),
            "record_coverage_ratio": record_coverage,
            "full_field_coverage_ratio": full_coverage,
            "full_value_observation_ratio": (
                full_value_observation_ratio
            ),
            "field_schema_coverage": {
                field_name: {
                    "count": field_schema_counts[field_name],
                    "ratio": field_schema_counts[field_name]
                    / max(len(universe), 1),
                }
                for field_name in required_fields
            },
            "field_value_coverage": {
                field_name: {
                    "count": field_value_counts[field_name],
                    "ratio": field_value_counts[field_name]
                    / max(len(universe), 1),
                }
                for field_name in required_fields
            },
            "value_unavailable_fields": sorted(
                unavailable_value_fields
            ),
            "nullable_values_accepted": nullable_values_accepted,
            "value_availability_contract": value_availability_contract,
        },
    )


def assess_macro_readiness(
    *,
    macro_record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    as_of: str = "",
    market: str = "CN",
    decision_cutoff_at: str | datetime | None = None,
    macro_readiness_evidence: MacroReadinessEvidence | None = None,
) -> BranchDataReadiness:
    missing = [field_name for field_name in MACRO_REQUIRED_FIELDS if not _is_present(macro_record.get(field_name))]
    source = str(manifest.get("source") or "").strip()
    priority = _source_priority(
        source,
        str(manifest.get("source_priority") or ""),
    )
    declared_priority = str(
        manifest.get("source_priority") or ""
    ).strip()
    source_priority_valid = (
        _MACRO_SOURCE_PRIORITY_BY_SOURCE.get(source) == declared_priority
    )
    provider_fallback_used = manifest.get("provider_fallback_used") is True
    source_not_approved = (
        priority not in _MACRO_APPROVED_PRIMARY_PRIORITIES
        or not source_priority_valid
    )
    fallback_used = provider_fallback_used or source_not_approved
    blockers = []
    read_error = str(manifest.get("read_error") or "").strip()
    if read_error:
        blockers.append(read_error)
    if not macro_record:
        blockers.append("macro_parquet_table_missing_or_empty")
    if missing:
        blockers.append("macro_required_fields_missing")
    if source_not_approved:
        blockers.append("macro_source_not_approved_primary")
    if not source_priority_valid:
        blockers.append("macro_source_priority_mismatch")
    if manifest.get("production_eligible") is not True:
        blockers.append("macro_generation_not_production_eligible")
    if not str(manifest.get("generation_id") or "").strip():
        blockers.append("macro_generation_id_missing")
    if str(manifest.get("provider_status") or "") != "verified_provider_snapshot":
        blockers.append("macro_provider_manifest_unverified")
    record_source = str(macro_record.get("source") or "").strip()
    record_priority = str(macro_record.get("source_priority") or "").strip()
    if macro_record and (
        record_source != source
        or record_priority != str(manifest.get("source_priority") or "").strip()
    ):
        blockers.append("macro_source_lineage_mismatch")
    if str(macro_record.get("pit_status") or "") != "market_point_in_time":
        blockers.append("macro_pit_status_invalid")
    fetched_at = pd.to_datetime(
        str(macro_record.get("fetched_at") or "").strip(),
        errors="coerce",
        utc=True,
    )
    if pd.isna(fetched_at):
        blockers.append("macro_fetched_at_missing_or_invalid")
    target_date = _date_text(as_of)
    record_date = _date_text(macro_record.get("trade_date"))
    if not target_date:
        blockers.append("macro_as_of_missing")
    elif record_date != target_date:
        blockers.append("macro_trade_date_as_of_mismatch")
    evidence_ready = False
    production_cn = (
        str(market or "").upper() == "CN"
        and manifest.get("production_eligible") is True
    )
    if production_cn:
        evidence_blockers, evidence_ready = (
            _macro_readiness_evidence_blockers(
                evidence=macro_readiness_evidence,
                manifest=manifest,
                macro_logical_date=record_date,
                target_session_date=target_date,
                decision_cutoff_at=decision_cutoff_at,
            )
        )
        blockers.extend(evidence_blockers)
        if evidence_ready and record_date != target_date:
            blockers = [
                blocker
                for blocker in blockers
                if blocker != "macro_trade_date_as_of_mismatch"
            ]
    blockers = list(dict.fromkeys(blockers))
    canonical_identity = macro_generation_identity(manifest)
    readiness_evidence_payload: dict[str, Any] = {}
    if type(macro_readiness_evidence) is MacroReadinessEvidence:
        try:
            readiness_evidence_payload = macro_readiness_evidence.to_dict()
            evidence_semantic_sha256 = _canonical_json_sha256(
                macro_readiness_evidence.semantic_payload()
            )
        except (TypeError, ValueError):
            readiness_evidence_payload = {
                "schema_version": str(
                    macro_readiness_evidence.schema_version or ""
                ),
                "semantic_sha256": str(
                    macro_readiness_evidence.semantic_sha256 or ""
                ),
                "invalid": True,
            }
            evidence_semantic_sha256 = ""
        if (
            evidence_semantic_sha256
            and macro_readiness_evidence.semantic_sha256
            == evidence_semantic_sha256
        ):
            canonical_identity[
                "macro_readiness_evidence_semantic_sha256"
            ] = evidence_semantic_sha256
    session_lag = (
        macro_readiness_evidence.evaluation.session_lag.session_lag
        if type(macro_readiness_evidence) is MacroReadinessEvidence
        and type(macro_readiness_evidence.evaluation)
        is ReleaseReadinessEvaluation
        and type(macro_readiness_evidence.evaluation.session_lag)
        is SessionLagEvaluation
        else None
    )
    return BranchDataReadiness(
        branch="macro",
        status=(
            STATUS_BLOCK
            if blockers
            else STATUS_WARN if provider_fallback_used else STATUS_PASS
        ),
        coverage_ratio=1.0 if not missing and macro_record else 0.0,
        freshness_status=(
            "bounded_open_session_lag"
            if evidence_ready and session_lag
            else "fresh_or_pit_asof" if macro_record else "unknown"
        ),
        pit_status="market_point_in_time" if macro_record else "missing",
        source_priority=priority,
        source=source,
        as_of=_date_text(as_of),
        required_fields=list(MACRO_REQUIRED_FIELDS),
        missing_fields=missing,
        blockers=blockers,
        affected_symbols=[],
        fallback_used=fallback_used,
        provider_status=str(manifest.get("provider_status") or "local_snapshot"),
        metadata={
            "manifest": dict(manifest),
            "macro_record": dict(_json_safe(macro_record)),
            "canonical_identity": canonical_identity,
            "macro_readiness_evidence": readiness_evidence_payload,
            "macro_session_lag": session_lag,
        },
    )


def assess_branch_data_readiness(
    *,
    frames: Mapping[str, pd.DataFrame],
    read_results: Mapping[str, Any] | None = None,
    candidate_symbols: Sequence[str] | None = None,
    market: str = "CN",
    category: str = "full_a",
    as_of: str = "",
    fundamental_root: str | Path = DEFAULT_FUNDAMENTAL_ROOT,
    macro_root: str | Path = DEFAULT_MACRO_ROOT,
    pinned_macro_record: Mapping[str, Any] | None = None,
    pinned_macro_manifest: Mapping[str, Any] | None = None,
    pinned_macro_readiness_evidence: MacroReadinessEvidence | None = None,
    decision_cutoff_at: str | datetime | None = None,
    run_id: str | None = None,
) -> BranchGovernanceReport:
    symbols = [_normalize_symbol(symbol) for symbol in (candidate_symbols or frames.keys()) if _normalize_symbol(symbol)]
    run_id = run_id or make_run_id(as_of)
    quant = assess_quant_readiness(
        frames=frames,
        read_results=read_results,
        symbols=symbols,
        as_of=as_of,
    )
    try:
        fundamentals, fundamental_manifest = load_fundamental_records(
            symbols,
            as_of=as_of,
            root=fundamental_root,
        )
    except FundamentalGenerationError as exc:
        fundamentals = {}
        fundamental_manifest = {
            "read_error": str(exc) or "fundamental_generation_invalid",
            "storage_backend": "parquet_canonical_generation",
        }
    pinned_macro_supplied = (
        pinned_macro_record is not None or pinned_macro_manifest is not None
    )
    if pinned_macro_supplied and (
        pinned_macro_record is None or pinned_macro_manifest is None
    ):
        raise ValueError(
            "pinned_macro_record and pinned_macro_manifest must be supplied together"
        )
    if pinned_macro_supplied:
        macro_record = dict(pinned_macro_record or {})
        macro_manifest = dict(pinned_macro_manifest or {})
    else:
        macro_record, macro_manifest = load_macro_record(
            as_of=as_of,
            root=macro_root,
        )
    if fundamental_manifest.get("read_error"):
        fundamental = BranchDataReadiness(
            branch="fundamental",
            status=STATUS_BLOCK,
            coverage_ratio=0.0,
            freshness_status="unknown",
            pit_status="invalid_canonical_generation",
            source_priority=SOURCE_OFFLINE,
            source="",
            as_of=_date_text(as_of),
            required_fields=list(FUNDAMENTAL_REQUIRED_FIELDS),
            missing_fields=list(FUNDAMENTAL_REQUIRED_FIELDS),
            blockers=["fundamental_generation_invalid"],
            affected_symbols=sorted(set(symbols)),
            fallback_used=False,
            provider_status="blocked_invalid_generation",
            metadata={"manifest": dict(fundamental_manifest)},
        )
    else:
        fundamental = _assess_symbol_records(
            branch="fundamental",
            symbols=symbols,
            records=fundamentals,
            required_fields=FUNDAMENTAL_REQUIRED_FIELDS,
            manifest=fundamental_manifest,
            as_of=as_of,
        )
    macro = assess_macro_readiness(
        macro_record=macro_record,
        manifest=macro_manifest,
        as_of=as_of,
        market=market,
        decision_cutoff_at=decision_cutoff_at,
        macro_readiness_evidence=pinned_macro_readiness_evidence,
    )
    blocked = sorted(
        set(quant.affected_symbols)
        | set(fundamental.affected_symbols)
        | (set(symbols) if macro.status == STATUS_BLOCK else set())
    )
    quantifiable = [
        symbol
        for symbol in symbols
        if _normalize_symbol(symbol) and _normalize_symbol(symbol) not in set(quant.affected_symbols)
    ]
    investable = [symbol for symbol in symbols if symbol not in set(blocked)]
    branch_data = {
        "fundamentals": fundamentals,
        "macro_data": macro_record,
    }
    return BranchGovernanceReport(
        run_id=run_id,
        market=str(market).upper(),
        category=str(category or "full_a"),
        as_of=_date_text(as_of),
        readiness={
            "quant": quant,
            "fundamental": fundamental,
            "macro": macro,
        },
        blocked_symbols=blocked,
        quantifiable_universe=sorted(set(quantifiable)),
        investable_universe=investable,
        branch_data=branch_data,
        metadata={
            "policy": "strict_when_used",
            "source_priority_order": list(SOURCE_PRIORITY_ORDER),
            "branch_schema_version": BRANCH_SCHEMA_VERSION,
            "canonical_branch_order": list(CANONICAL_BRANCH_ORDER),
            "candidate_count": len(symbols),
            "blocked_count": len(blocked),
            "investable_count": len(investable),
        },
    )


def render_branch_readiness_md(report: BranchGovernanceReport) -> str:
    lines = [
        "# Branch Data Readiness",
        "",
        f"- Run: `{report.run_id}`",
        f"- Market: `{report.market}`",
        f"- Category: `{report.category}`",
        f"- As of: `{report.as_of}`",
        f"- Policy: `{report.metadata.get('policy', 'strict_when_used')}`",
        f"- Quantifiable universe: {len(report.quantifiable_universe)}",
        f"- Investable universe: {len(report.investable_universe)}",
        f"- Blocked symbols: {len(report.blocked_symbols)}",
        "",
        "| Branch | Status | Coverage | PIT | Source priority | Blockers |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for branch, readiness in report.readiness.items():
        lines.append(
            "| {branch} | {status} | {coverage:.2%} | {pit} | {source_priority} | {blockers} |".format(
                branch=branch,
                status=readiness.status,
                coverage=readiness.coverage_ratio,
                pit=readiness.pit_status,
                source_priority=readiness.source_priority,
                blockers=", ".join(readiness.blockers) or "-",
            )
        )
    fundamental = report.readiness.get("fundamental")
    if fundamental is not None:
        metadata = dict(fundamental.metadata or {})
        if "full_field_coverage_ratio" in metadata:
            lines.extend(
                [
                    "",
                    "## Fundamental value availability",
                    "",
                    "- Structural field coverage: "
                    f"{float(metadata.get('full_field_coverage_ratio', 0.0)):.2%}",
                    "- All-eight observed value ratio: "
                    f"{float(metadata.get('full_value_observation_ratio', 0.0)):.2%}",
                    "- Nullable values accepted by verified v3 evidence: "
                    f"{bool(metadata.get('nullable_values_accepted', False))}",
                    "- Structurally missing fields: "
                    f"{', '.join(fundamental.missing_fields) or '-'}",
                    "- Unavailable value fields: "
                    f"{', '.join(metadata.get('value_unavailable_fields', [])) or '-'}",
                ]
            )
    return "\n".join(lines) + "\n"


def write_branch_readiness_report(
    report: BranchGovernanceReport,
    *,
    output_dir: str | Path = DEFAULT_READINESS_ROOT,
) -> dict[str, str]:
    out = Path(output_dir)
    resolved_out = out.resolve(strict=False)
    if (
        resolved_out == _FROZEN_READINESS_ROOT
        or _FROZEN_READINESS_ROOT in resolved_out.parents
    ):
        raise ValueError(
            "reports/branch_readiness is frozen v13 retirement evidence; "
            "write current artifacts under reports/v15/branch_readiness"
        )
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{report.run_id}.json"
    md_path = out / f"{report.run_id}.md"
    csv_path = out / f"{report.run_id}.csv"
    json_path.write_text(json.dumps(report.to_dict(include_branch_data=True), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_branch_readiness_md(report), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "branch",
                "status",
                "coverage_ratio",
                "freshness_status",
                "pit_status",
                "source_priority",
                "source",
                "missing_fields",
                "blockers",
                "affected_symbols",
            ],
        )
        writer.writeheader()
        for readiness in report.readiness.values():
            writer.writerow(
                {
                    "branch": readiness.branch,
                    "status": readiness.status,
                    "coverage_ratio": readiness.coverage_ratio,
                    "freshness_status": readiness.freshness_status,
                    "pit_status": readiness.pit_status,
                    "source_priority": readiness.source_priority,
                    "source": readiness.source,
                    "missing_fields": ";".join(readiness.missing_fields),
                    "blockers": ";".join(readiness.blockers),
                    "affected_symbols": ";".join(readiness.affected_symbols[:128]),
                }
            )
    return {"json": str(json_path), "md": str(md_path), "csv": str(csv_path)}


__all__ = [
    "BranchDataReadiness",
    "BranchGovernanceReport",
    "FUNDAMENTAL_REQUIRED_FIELDS",
    "FUNDAMENTAL_VALUE_AVAILABILITY_SCHEMA",
    "MACRO_MAX_SESSION_LAG",
    "MACRO_READINESS_EVIDENCE_SCHEMA",
    "MACRO_REQUIRED_FIELDS",
    "MacroReadinessEvidence",
    "QUANT_REQUIRED_FIELDS",
    "SOURCE_OFFLINE",
    "SOURCE_OFFICIAL",
    "SOURCE_OFFICIAL_FIRST",
    "SOURCE_PRIORITY_ORDER",
    "SOURCE_PUBLIC_FALLBACK",
    "SOURCE_TUSHARE",
    "STATUS_BLOCK",
    "STATUS_PASS",
    "STATUS_WARN",
    "assess_branch_data_readiness",
    "assess_macro_readiness",
    "assess_quant_readiness",
    "build_macro_readiness_evidence",
    "load_fundamental_records",
    "load_macro_record",
    "macro_generation_identity",
    "write_branch_readiness_report",
]
