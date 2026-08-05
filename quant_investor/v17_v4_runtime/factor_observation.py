"""Strict, authority-free V17 v4 forward observations and labels.

The full-universe factor observation and the strategy-pool observation are
different registered artifacts.  Forward labels mature only from explicit
future Shanghai open-session windows and remain diagnostic-only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
from typing import Any, Final, NoReturn
from zoneinfo import ZoneInfo

from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import validate_semantic_sha
from quant_investor.v17_v4_contract.identities import (
    IdentityContractError,
    require_opaque_id,
    require_sha256,
    require_utc_timestamp,
)

FACTOR_UNIVERSE_OBSERVATION_VERSION: Final = "myquant.v17.v4.factor-universe-observation.v1"
FACTOR_OBSERVATION_VERSION: Final = FACTOR_UNIVERSE_OBSERVATION_VERSION
STRATEGY_POOL_OBSERVATION_VERSION: Final = "myquant.v17.v4.strategy-pool-observation.v1"
FORWARD_LABEL_VERSION: Final = "myquant.v17.v4.forward-label.v1"
FACTOR_FORWARD_LABEL_VERSION: Final = FORWARD_LABEL_VERSION
LABEL_HORIZONS: Final = (1, 5, 10, 20, 60)
COST_BASIS_POINTS: Final = 20
COST_RATE: Final = Decimal("0.002")
SHANGHAI: Final = ZoneInfo("Asia/Shanghai")
NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "mainline_authority": False,
    "order": False,
    "production": False,
    "research_only": True,
    "trade": False,
}
_DISABLED: Final = {
    "promotion_eligible": False,
    "provider_authority": False,
    "provider_invoked": False,
    "shadow_only": True,
}
_REF_FIELDS: Final = {
    "artifact_id",
    "artifact_version",
    "byte_sha256",
    "cutoff",
    "relative_path",
    "semantic_sha256",
    "strategy_id",
}
_SYMBOL_SUFFIXES: Final = (".BJ", ".SH", ".SZ")


class FactorObservationError(ValueError):
    """Raised when an observation or forward label is not provable."""

    exit_code = 2


def _blocked(reason: str) -> NoReturn:
    raise FactorObservationError(f"V17_V4_FACTOR_OBSERVATION_BLOCKED:{reason}")


def _identity(value: Any, *, label: str) -> str:
    try:
        return require_opaque_id(value, label=label)
    except IdentityContractError:
        _blocked(f"{label}_invalid")


def _sha256(value: Any, *, label: str) -> str:
    try:
        return require_sha256(value, label=label)
    except IdentityContractError:
        _blocked(f"{label}_invalid")


def _timestamp(value: Any, *, label: str) -> str:
    try:
        return require_utc_timestamp(value, label=label)
    except IdentityContractError:
        _blocked(f"{label}_invalid")


def _instant(value: Any, *, label: str) -> datetime:
    text = _timestamp(value, label=label)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        _blocked(f"{label}_invalid")


def _session(value: Any, *, label: str) -> str:
    if type(value) is not str:
        _blocked(f"{label}_invalid")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        _blocked(f"{label}_invalid")
    if parsed.isoformat() != value:
        _blocked(f"{label}_noncanonical")
    return value


def _relative_path(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        _blocked(f"{label}_invalid")
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        _blocked(f"{label}_non_ascii")
    return value


def _decimal(value: Any, *, label: str) -> Decimal:
    if type(value) is bool or type(value) not in {Decimal, float, int, str}:
        _blocked(f"{label}_invalid")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        _blocked(f"{label}_invalid")
    if not result.is_finite():
        _blocked(f"{label}_nonfinite")
    return result


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        _blocked("decimal_nonfinite")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _symbol(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 9
        or not value[:6].isdigit()
        or not value.endswith(_SYMBOL_SUFFIXES)
    ):
        _blocked(f"{label}_invalid")
    return value


def _artifact_ref(
    value: Mapping[str, Any],
    *,
    strategy_id: str,
    cutoff: str,
    label: str,
    expected_version: str | None = None,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != _REF_FIELDS:
        _blocked(f"{label}_shape")
    artifact_strategy = _identity(
        value["strategy_id"],
        label=f"{label}.strategy_id",
    )
    if artifact_strategy != strategy_id:
        _blocked(f"{label}_strategy")
    artifact_cutoff = _timestamp(value["cutoff"], label=f"{label}.cutoff")
    if artifact_cutoff > cutoff:
        _blocked(f"{label}_after_cutoff")
    version = _identity(
        value["artifact_version"],
        label=f"{label}.artifact_version",
    )
    if expected_version is not None and version != expected_version:
        _blocked(f"{label}_version")
    return {
        "artifact_id": _identity(
            value["artifact_id"],
            label=f"{label}.artifact_id",
        ),
        "artifact_version": version,
        "byte_sha256": _sha256(
            value["byte_sha256"],
            label=f"{label}.byte_sha256",
        ),
        "cutoff": artifact_cutoff,
        "relative_path": _relative_path(
            value["relative_path"],
            label=f"{label}.relative_path",
        ),
        "semantic_sha256": _sha256(
            value["semantic_sha256"],
            label=f"{label}.semantic_sha256",
        ),
        "strategy_id": artifact_strategy,
    }


def _artifact_refs(
    values: Sequence[Mapping[str, Any]],
    *,
    strategy_id: str,
    cutoff: str,
    label: str,
    require_nonempty: bool = True,
) -> list[dict[str, str]]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or (require_nonempty and not values)
    ):
        _blocked(f"{label}_invalid")
    normalized = [
        _artifact_ref(
            value,
            strategy_id=strategy_id,
            cutoff=cutoff,
            label=f"{label}[{index}]",
        )
        for index, value in enumerate(values)
    ]
    normalized.sort(
        key=lambda row: (
            row["relative_path"].encode("ascii"),
            row["byte_sha256"].encode("ascii"),
            row["artifact_id"].encode("utf-8"),
        )
    )
    if len({canonical_bytes(row) for row in normalized}) != len(normalized):
        _blocked(f"{label}_duplicate")
    return normalized


def _registered(document: dict[str, Any], *, label: str) -> dict[str, Any]:
    try:
        validate_artifact(document)
    except Exception as exc:
        _blocked(f"{label}_registered_schema:{exc}")
    return document


def _completeness(statuses: Sequence[str]) -> str:
    available = sum(status == "AVAILABLE" for status in statuses)
    if available == len(statuses):
        return "COMPLETE"
    if available == 0:
        return "UNAVAILABLE"
    return "PARTIAL"


def _observation_rows(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        _blocked("observations_invalid")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        label = f"observations[{index}]"
        if type(value) is not dict or set(value) != {"status", "symbol", "value"}:
            _blocked(f"{label}_shape")
        symbol = _symbol(value["symbol"], label=f"{label}.symbol")
        if symbol in seen:
            _blocked("observations_duplicate_symbol")
        seen.add(symbol)
        status = value["status"]
        if status not in {"AVAILABLE", "UNAVAILABLE"}:
            _blocked(f"{label}_status")
        raw = value["value"]
        if status == "AVAILABLE":
            normalized_value: str | None = _decimal_text(_decimal(raw, label=f"{label}.value"))
        else:
            if raw is not None:
                _blocked(f"{label}_unavailable_value")
            normalized_value = None
        rows.append(
            {
                "status": status,
                "symbol": symbol,
                "value": normalized_value,
            }
        )
    return sorted(rows, key=lambda row: row["symbol"].encode("ascii"))


def build_factor_observation(
    *,
    observation_id: str,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    factor_ref: Mapping[str, Any],
    request_ref: Mapping[str, Any],
    source_refs: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one full-universe observation for one exact factor definition."""

    strategy = _identity(strategy_id, label="strategy_id")
    session = _session(decision_session, label="decision_session")
    cutoff_text = _timestamp(cutoff, label="cutoff")
    if session > cutoff_text[:10]:
        _blocked("decision_session_after_cutoff")
    rows = _observation_rows(observations)
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            **dict(_DISABLED),
            "completeness": _completeness([row["status"] for row in rows]),
            "cutoff": cutoff_text,
            "decision_session": session,
            "factor_ref": _artifact_ref(
                factor_ref,
                strategy_id=strategy,
                cutoff=cutoff_text,
                label="factor_ref",
            ),
            "observation_id": _identity(
                observation_id,
                label="observation_id",
            ),
            "observations": rows,
            "protocol_version": PROTOCOL_VERSION,
            "request_ref": _artifact_ref(
                request_ref,
                strategy_id=strategy,
                cutoff=cutoff_text,
                label="request_ref",
                expected_version="myquant.v17.v4.forward-run-request.v1",
            ),
            "source_refs": _artifact_refs(
                source_refs,
                strategy_id=strategy,
                cutoff=cutoff_text,
                label="source_refs",
            ),
            "strategy_id": strategy,
            "version": FACTOR_UNIVERSE_OBSERVATION_VERSION,
        }
    )
    return _registered(document, label="factor_universe_observation")


def validate_factor_observation(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema, exact refs, sealing, and deterministic replay."""

    try:
        normalized = validate_semantic_sha(document)
    except Exception:
        _blocked("factor_universe_observation_semantic_sha")
    _registered(normalized, label="factor_universe_observation")
    rebuilt = build_factor_observation(
        observation_id=normalized.get("observation_id"),
        strategy_id=normalized.get("strategy_id"),
        decision_session=normalized.get("decision_session"),
        cutoff=normalized.get("cutoff"),
        factor_ref=normalized.get("factor_ref", {}),
        request_ref=normalized.get("request_ref", {}),
        source_refs=normalized.get("source_refs", ()),
        observations=normalized.get("observations", ()),
    )
    if rebuilt != normalized:
        _blocked("factor_universe_observation_replay")
    return normalized


def factor_observation_ref(
    observation: Mapping[str, Any],
    *,
    relative_path: str,
) -> dict[str, str]:
    """Return the shared exact artifact reference for an observation."""

    normalized = validate_factor_observation(observation)
    return {
        "artifact_id": str(normalized["observation_id"]),
        "artifact_version": str(normalized["version"]),
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(normalized)).hexdigest(),
        "cutoff": str(normalized["cutoff"]),
        "relative_path": _relative_path(
            relative_path,
            label="observation_relative_path",
        ),
        "semantic_sha256": str(normalized["semantic_sha256"]),
        "strategy_id": str(normalized["strategy_id"]),
    }


def _pool_rows(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        _blocked("pool_rows_invalid")
    rows: list[dict[str, Any]] = []
    symbols: set[str] = set()
    ranks: set[int] = set()
    for index, value in enumerate(values):
        label = f"pool_rows[{index}]"
        if type(value) is not dict or set(value) != {
            "pool_rank",
            "selected",
            "symbol",
        }:
            _blocked(f"{label}_shape")
        symbol = _symbol(value["symbol"], label=f"{label}.symbol")
        rank = value["pool_rank"]
        if type(rank) is not int or rank < 1:
            _blocked(f"{label}_rank")
        if type(value["selected"]) is not bool:
            _blocked(f"{label}_selected")
        if symbol in symbols or rank in ranks:
            _blocked("pool_rows_duplicate_symbol_or_rank")
        symbols.add(symbol)
        ranks.add(rank)
        rows.append(
            {
                "pool_rank": rank,
                "selected": value["selected"],
                "symbol": symbol,
            }
        )
    rows.sort(key=lambda row: (row["pool_rank"], row["symbol"].encode("ascii")))
    if [row["pool_rank"] for row in rows] != list(range(1, len(rows) + 1)):
        _blocked("pool_rows_noncontiguous_rank")
    return rows


def build_strategy_pool_observation(
    *,
    observation_id: str,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    request_ref: Mapping[str, Any],
    source_refs: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a strategy-pool observation without relabelling it as universe data."""

    strategy = _identity(strategy_id, label="strategy_id")
    session = _session(decision_session, label="decision_session")
    cutoff_text = _timestamp(cutoff, label="cutoff")
    if session > cutoff_text[:10]:
        _blocked("decision_session_after_cutoff")
    rows = _pool_rows(pool_rows)
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            **dict(_DISABLED),
            "completeness": "COMPLETE",
            "cutoff": cutoff_text,
            "decision_session": session,
            "observation_id": _identity(
                observation_id,
                label="observation_id",
            ),
            "pool_rows": rows,
            "protocol_version": PROTOCOL_VERSION,
            "request_ref": _artifact_ref(
                request_ref,
                strategy_id=strategy,
                cutoff=cutoff_text,
                label="request_ref",
                expected_version="myquant.v17.v4.forward-run-request.v1",
            ),
            "source_refs": _artifact_refs(
                source_refs,
                strategy_id=strategy,
                cutoff=cutoff_text,
                label="source_refs",
            ),
            "strategy_id": strategy,
            "version": STRATEGY_POOL_OBSERVATION_VERSION,
        }
    )
    return _registered(document, label="strategy_pool_observation")


def validate_strategy_pool_observation(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        normalized = validate_semantic_sha(document)
    except Exception:
        _blocked("strategy_pool_observation_semantic_sha")
    _registered(normalized, label="strategy_pool_observation")
    rebuilt = build_strategy_pool_observation(
        observation_id=normalized.get("observation_id"),
        strategy_id=normalized.get("strategy_id"),
        decision_session=normalized.get("decision_session"),
        cutoff=normalized.get("cutoff"),
        request_ref=normalized.get("request_ref", {}),
        source_refs=normalized.get("source_refs", ()),
        pool_rows=normalized.get("pool_rows", ()),
    )
    if rebuilt != normalized:
        _blocked("strategy_pool_observation_replay")
    return normalized


def _positive_close_mapping(
    value: Mapping[str, Any],
    *,
    keys: Sequence[str],
    label: str,
) -> dict[str, Decimal]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        _blocked(f"{label}_domain")
    result: dict[str, Decimal] = {}
    for key in keys:
        close = _decimal(value[key], label=f"{label}.{key}")
        if close <= 0:
            _blocked(f"{label}_nonpositive:{key}")
        result[key] = close
    return result


def _label_window(
    values: Sequence[str],
    *,
    origin_session: str,
    horizon_sessions: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        _blocked("shanghai_open_sessions_invalid")
    sessions = tuple(_session(value, label="shanghai_open_session") for value in values)
    if (
        len(sessions) != len(set(sessions))
        or any(left >= right for left, right in zip(sessions, sessions[1:]))
        or origin_session not in sessions
    ):
        _blocked("shanghai_open_sessions_not_strict")
    start = sessions.index(origin_session)
    end = start + horizon_sessions
    if end >= len(sessions):
        _blocked("label_future_session_unavailable")
    return sessions[start : end + 1]


def build_factor_forward_label(
    *,
    strategy_id: str,
    decision_session: str,
    observation_run_ref: Mapping[str, Any],
    horizon_sessions: int,
    shanghai_open_sessions: Sequence[str],
    origin_adjusted_closes: Mapping[str, Any],
    end_adjusted_closes: Mapping[str, Any],
    market_origin_adjusted_close: Any,
    market_end_adjusted_close: Any,
    industry_by_symbol: Mapping[str, str],
    industry_origin_adjusted_closes: Mapping[str, Any],
    industry_end_adjusted_closes: Mapping[str, Any],
    evidence_refs: Sequence[Mapping[str, Any]],
    matured_at: str,
    label_id: str | None = None,
) -> dict[str, Any]:
    """Build an exact 1/5/10/20/60-session total-return diagnostic label."""

    strategy = _identity(strategy_id, label="strategy_id")
    origin_session = _session(decision_session, label="decision_session")
    if type(horizon_sessions) is not int or horizon_sessions not in LABEL_HORIZONS:
        _blocked("label_horizon")
    window = _label_window(
        shanghai_open_sessions,
        origin_session=origin_session,
        horizon_sessions=horizon_sessions,
    )
    label_session = window[-1]
    matured = _timestamp(matured_at, label="matured_at")
    matured_local = _instant(matured, label="matured_at").astimezone(SHANGHAI)
    if matured_local.date().isoformat() != label_session or matured_local.timetz().replace(
        tzinfo=None
    ) < time(15, 0):
        _blocked("label_not_matured_at_end_session_close")
    run_ref = _artifact_ref(
        observation_run_ref,
        strategy_id=strategy,
        cutoff=matured,
        label="observation_run_ref",
        expected_version="myquant.v17.v4.forward-observation-run.v1",
    )
    sources = _artifact_refs(
        evidence_refs,
        strategy_id=strategy,
        cutoff=matured,
        label="evidence_refs",
    )
    symbols = tuple(
        sorted(
            (_symbol(value, label="label_symbol") for value in origin_adjusted_closes),
            key=lambda value: value.encode("ascii"),
        )
    )
    if not symbols:
        _blocked("label_symbols_empty")
    origin_closes = _positive_close_mapping(
        origin_adjusted_closes,
        keys=symbols,
        label="origin_adjusted_closes",
    )
    end_closes = _positive_close_mapping(
        end_adjusted_closes,
        keys=symbols,
        label="end_adjusted_closes",
    )
    if not isinstance(industry_by_symbol, Mapping) or set(industry_by_symbol) != set(symbols):
        _blocked("industry_by_symbol_domain")
    industries_by_symbol = {
        symbol: _identity(
            industry_by_symbol[symbol],
            label=f"industry_by_symbol.{symbol}",
        )
        for symbol in symbols
    }
    industries = tuple(
        sorted(set(industries_by_symbol.values()), key=lambda value: value.encode("utf-8"))
    )
    industry_origin = _positive_close_mapping(
        industry_origin_adjusted_closes,
        keys=industries,
        label="industry_origin_adjusted_closes",
    )
    industry_end = _positive_close_mapping(
        industry_end_adjusted_closes,
        keys=industries,
        label="industry_end_adjusted_closes",
    )
    market_origin = _decimal(
        market_origin_adjusted_close,
        label="market_origin_adjusted_close",
    )
    market_end = _decimal(
        market_end_adjusted_close,
        label="market_end_adjusted_close",
    )
    if market_origin <= 0 or market_end <= 0:
        _blocked("market_adjusted_close_nonpositive")
    with localcontext() as context:
        context.prec = 40
        market_return = +(market_end / market_origin - Decimal("1"))
        industry_returns = {
            industry: +(industry_end[industry] / industry_origin[industry] - Decimal("1"))
            for industry in industries
        }
        rows = []
        for symbol in symbols:
            total_return = +(end_closes[symbol] / origin_closes[symbol] - Decimal("1"))
            industry_id = industries_by_symbol[symbol]
            industry_return = industry_returns[industry_id]
            rows.append(
                {
                    "cost_adjusted_return": _decimal_text(+(total_return - COST_RATE)),
                    "industry_adjusted_return": _decimal_text(+(total_return - industry_return)),
                    "industry_id": industry_id,
                    "industry_return": _decimal_text(industry_return),
                    "market_adjusted_return": _decimal_text(+(total_return - market_return)),
                    "market_return": _decimal_text(market_return),
                    "status": "AVAILABLE",
                    "symbol": symbol,
                    "total_return": _decimal_text(total_return),
                }
            )
    lineage_sha = hashlib.sha256(
        canonical_bytes(
            {
                "evidence_refs": sources,
                "observation_run_ref": run_ref,
                "shanghai_open_sessions": list(window),
            }
        )
    ).hexdigest()
    expected_id = (
        "forward-label-"
        + hashlib.sha256(
            canonical_bytes(
                {
                    "horizon_sessions": horizon_sessions,
                    "source_lineage_sha256": lineage_sha,
                    "strategy_id": strategy,
                }
            )
        ).hexdigest()
    )
    if label_id is not None and label_id != expected_id:
        _blocked("label_id_binding")
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            **dict(_DISABLED),
            "completeness": "COMPLETE",
            "cost_basis_points": COST_BASIS_POINTS,
            "cutoff": matured,
            "decision_session": origin_session,
            "evidence_refs": sources,
            "horizon_sessions": horizon_sessions,
            "label_id": expected_id,
            "label_rows": rows,
            "label_session": label_session,
            "observation_run_ref": run_ref,
            "performance_evidence_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "shanghai_open_sessions": list(window),
            "source_lineage_sha256": lineage_sha,
            "strategy_id": strategy,
            "version": FORWARD_LABEL_VERSION,
        }
    )
    return _registered(document, label="forward_label")


def validate_factor_forward_label(
    document: Mapping[str, Any],
    *,
    observation_run_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate maturity, no-backfill window, exact refs, and return arithmetic."""

    try:
        normalized = validate_semantic_sha(document)
    except Exception:
        _blocked("forward_label_semantic_sha")
    _registered(normalized, label="forward_label")
    horizon = normalized.get("horizon_sessions")
    if type(horizon) is not int or horizon not in LABEL_HORIZONS:
        _blocked("label_horizon")
    origin_session = _session(
        normalized.get("decision_session"),
        label="decision_session",
    )
    window = _label_window(
        normalized.get("shanghai_open_sessions", ()),
        origin_session=origin_session,
        horizon_sessions=horizon,
    )
    if (
        list(window) != normalized.get("shanghai_open_sessions")
        or normalized.get("label_session") != window[-1]
    ):
        _blocked("label_calendar_binding")
    cutoff = _timestamp(normalized.get("cutoff"), label="cutoff")
    matured_local = _instant(cutoff, label="cutoff").astimezone(SHANGHAI)
    if matured_local.date().isoformat() != window[-1] or matured_local.timetz().replace(
        tzinfo=None
    ) < time(15, 0):
        _blocked("label_not_matured_at_end_session_close")
    strategy = _identity(normalized.get("strategy_id"), label="strategy_id")
    run_ref = _artifact_ref(
        normalized.get("observation_run_ref", {}),
        strategy_id=strategy,
        cutoff=cutoff,
        label="observation_run_ref",
        expected_version="myquant.v17.v4.forward-observation-run.v1",
    )
    sources = _artifact_refs(
        normalized.get("evidence_refs", ()),
        strategy_id=strategy,
        cutoff=cutoff,
        label="evidence_refs",
    )
    if sources != normalized.get("evidence_refs"):
        _blocked("evidence_refs_order")
    lineage_sha = hashlib.sha256(
        canonical_bytes(
            {
                "evidence_refs": sources,
                "observation_run_ref": run_ref,
                "shanghai_open_sessions": list(window),
            }
        )
    ).hexdigest()
    if normalized.get("source_lineage_sha256") != lineage_sha:
        _blocked("label_source_lineage")
    rows = normalized.get("label_rows")
    if not isinstance(rows, list) or not rows:
        _blocked("label_rows")
    symbols: list[str] = []
    for index, row in enumerate(rows):
        label = f"label_rows[{index}]"
        expected_fields = {
            "cost_adjusted_return",
            "industry_adjusted_return",
            "industry_id",
            "industry_return",
            "market_adjusted_return",
            "market_return",
            "status",
            "symbol",
            "total_return",
        }
        if type(row) is not dict or set(row) != expected_fields:
            _blocked(f"{label}_shape")
        if row["status"] != "AVAILABLE":
            _blocked(f"{label}_unavailable")
        symbol = _symbol(row["symbol"], label=f"{label}.symbol")
        symbols.append(symbol)
        total_return = _decimal(row["total_return"], label=f"{label}.total_return")
        market_return = _decimal(row["market_return"], label=f"{label}.market_return")
        industry_return = _decimal(
            row["industry_return"],
            label=f"{label}.industry_return",
        )
        expected_adjustments = {
            "cost_adjusted_return": total_return - COST_RATE,
            "industry_adjusted_return": total_return - industry_return,
            "market_adjusted_return": total_return - market_return,
        }
        for field, value in expected_adjustments.items():
            if _decimal(row[field], label=f"{label}.{field}") != value:
                _blocked(f"{label}_arithmetic")
    if symbols != sorted(symbols, key=lambda value: value.encode("ascii")) or len(symbols) != len(
        set(symbols)
    ):
        _blocked("label_symbol_order")
    if observation_run_ref is not None:
        expected = _artifact_ref(
            observation_run_ref,
            strategy_id=strategy,
            cutoff=cutoff,
            label="observation_run_ref_readback",
            expected_version="myquant.v17.v4.forward-observation-run.v1",
        )
        if expected != run_ref:
            _blocked("observation_run_ref_readback")
    return normalized


__all__ = [
    "COST_BASIS_POINTS",
    "FACTOR_FORWARD_LABEL_VERSION",
    "FACTOR_OBSERVATION_VERSION",
    "FACTOR_UNIVERSE_OBSERVATION_VERSION",
    "FORWARD_LABEL_VERSION",
    "FactorObservationError",
    "LABEL_HORIZONS",
    "NO_AUTHORITY",
    "STRATEGY_POOL_OBSERVATION_VERSION",
    "build_factor_forward_label",
    "build_factor_observation",
    "build_strategy_pool_observation",
    "factor_observation_ref",
    "validate_factor_forward_label",
    "validate_factor_observation",
    "validate_strategy_pool_observation",
]
