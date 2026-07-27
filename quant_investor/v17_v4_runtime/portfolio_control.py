"""Holdings-aware, shrink-only V17 v4 portfolio controls.

This module constructs research portfolio artifacts only.  It has no broker,
execution, order, or trade surface.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
from typing import Any, Final, NoReturn

from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import (
    CanonicalContractError,
    load_canonical_resource,
    validate_semantic_sha,
)
from quant_investor.v17_v4_contract.identities import (
    IdentityContractError,
    require_opaque_id,
    require_sha256,
    require_utc_timestamp,
)
from quant_investor.v17_v4_runtime.deep_control import (
    revalidate_deep_evidence_bundle,
)

HOLDINGS_VERSION: Final = "myquant.v17.v4.holdings-snapshot.v1"
RISK_POLICY_VERSION: Final = (
    "myquant.v17.v4.portfolio-risk-policy.v1"
)
PERMISSIONS_VERSION: Final = (
    "myquant.v17.v4.pretrade-permissions.v1"
)
OVERLAY_VERSION: Final = "myquant.v17.v4.portfolio-overlay.v1"
PORTFOLIO_VERSION: Final = "myquant.v17.v4.portfolio-output.v1"
REGIME_EVIDENCE_VERSION: Final = "myquant.v17.v4.regime-evidence.v1"
DEEP_VERSION: Final = "myquant.v17.v4.deep-evidence-bundle.v1"
FUSION_VERSION: Final = "myquant.v17.v4.fusion-top24.v1"
PIT_CATALOG_VERSION: Final = (
    "myquant.v17.v4.pit-generation-catalog.v1"
)
CALENDAR_DATASET_VERSION: Final = (
    "myquant.v17.v4.dataset.cn_open_day_calendar.v1"
)
MIN_PRODUCTION_CALENDAR_SESSIONS: Final = 2520
WEIGHT_QUANTUM: Final = Decimal("0.000000000001")
PERMISSION_RULES_SHA256: Final = hashlib.sha256(
    canonical_bytes(
        {
            "held_truth": "current_target_gt_zero",
            "pool_scope": "fusion_top24_union_outside_holdings",
            "review_only_buy": False,
            "version": 1,
        }
    )
).hexdigest()
ALLOCATION_RULES_SHA256: Final = hashlib.sha256(
    canonical_bytes(
        {
            "deep_for_held": "cannot_reduce_below_current",
            "gross_reduction": "proportional_no_redistribution",
            "group_reduction": "proportional_no_redistribution",
            "overlay": "multiplicative_shrink_only",
            "version": 1,
        }
    )
).hexdigest()
_REF_FIELDS: Final = frozenset(
    {
        "artifact_id",
        "artifact_version",
        "byte_sha256",
        "cutoff",
        "relative_path",
        "semantic_sha256",
        "strategy_id",
    }
)
_NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}


class PortfolioControlError(ValueError):
    """Raised when production portfolio evidence does not close."""

    exit_code = 2


def _blocked(reason: str) -> NoReturn:
    raise PortfolioControlError(
        f"PORTFOLIO_CONTROL_BLOCKED:{reason}"
    )


def _decimal(value: Any, *, label: str) -> Decimal:
    if type(value) not in {str, int, float, Decimal} or type(value) is bool:
        _blocked(f"{label}_invalid")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        _blocked(f"{label}_invalid")
    if not result.is_finite():
        _blocked(f"{label}_nonfinite")
    return result


def _weight(value: Decimal) -> Decimal:
    return value.quantize(WEIGHT_QUANTUM, rounding=ROUND_HALF_EVEN)


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _validate_ref(
    value: Mapping[str, Any],
    *,
    strategy_id: str,
    cutoff: str,
    expected_version: str | None,
    label: str,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != set(_REF_FIELDS):
        _blocked(f"{label}_shape")
    try:
        artifact_id = require_opaque_id(
            value["artifact_id"],
            label=f"{label}.artifact_id",
        )
        version = require_opaque_id(
            value["artifact_version"],
            label=f"{label}.artifact_version",
        )
        ref_cutoff = require_utc_timestamp(
            value["cutoff"],
            label=f"{label}.cutoff",
        )
        require_sha256(value["byte_sha256"], label=f"{label}.byte_sha256")
        require_sha256(
            value["semantic_sha256"],
            label=f"{label}.semantic_sha256",
        )
    except IdentityContractError:
        _blocked(f"{label}_identity")
    path = value["relative_path"]
    if (
        (expected_version is not None and version != expected_version)
        or value["strategy_id"] != strategy_id
        or ref_cutoff > cutoff
        or type(path) is not str
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        _blocked(f"{label}_binding")
    return {
        "artifact_id": artifact_id,
        **{
            field: str(value[field])
            for field in _REF_FIELDS
            if field != "artifact_id"
        },
    }


def _identity(document: Mapping[str, Any]) -> Any:
    return next(
        (
            document.get(field)
            for field in (
                "bundle_id",
                "catalog_id",
                "dossier_id",
                "evidence_id",
                "output_id",
                "overlay_id",
                "permissions_id",
                "policy_id",
                "receipt_id",
                "scan_id",
                "snapshot_id",
            )
            if field in document
        ),
        None,
    )


def _read_exact(
    reference: Mapping[str, str],
    *,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
    label: str,
) -> dict[str, Any]:
    try:
        raw = artifact_loader(reference)
    except Exception as exc:
        raise PortfolioControlError(
            f"PORTFOLIO_CONTROL_BLOCKED:{label}_read_failed"
        ) from exc
    if (
        type(raw) is not bytes
        or hashlib.sha256(raw).hexdigest()
        != reference["byte_sha256"]
    ):
        _blocked(f"{label}_byte_sha")
    try:
        document = load_canonical_resource(raw, label=label)
        sealed = validate_semantic_sha(document)
        validate_artifact(sealed, artifact_loader=artifact_loader)
    except (CanonicalContractError, ValueError, RuntimeError):
        _blocked(f"{label}_native_validation")
    if (
        sealed.get("version") != reference["artifact_version"]
        or sealed.get("strategy_id") != reference["strategy_id"]
        or sealed.get("cutoff") != reference["cutoff"]
        or sealed.get("semantic_sha256")
        != reference["semantic_sha256"]
        or _identity(sealed) != reference["artifact_id"]
    ):
        _blocked(f"{label}_document_binding")
    return sealed


def _read_bound_bytes(
    reference: Mapping[str, str],
    *,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
    label: str,
) -> bytes:
    try:
        raw = artifact_loader(reference)
    except Exception as exc:
        raise PortfolioControlError(
            f"PORTFOLIO_CONTROL_BLOCKED:{label}_read_failed"
        ) from exc
    if (
        type(raw) is not bytes
        or hashlib.sha256(raw).hexdigest()
        != reference["byte_sha256"]
    ):
        _blocked(f"{label}_byte_sha")
    return raw


def _ordered_row_hash(values: Sequence[Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for value in sorted(tuple(tuple(item) for item in values)):
        digest.update(canonical_bytes(list(value)))
        digest.update(b"\n")
    return digest.hexdigest()


def _calendar_row_set_hash(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for row in sorted(
        rows,
        key=lambda item: (str(item["market_id"]), str(item["session"])),
    ):
        digest.update(canonical_bytes(dict(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def _calendar_sessions(
    reference: Mapping[str, str],
    *,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> tuple[list[str], dict[str, Any]]:
    raw = _read_bound_bytes(
        reference,
        artifact_loader=artifact_loader,
        label="canonical_calendar",
    )
    calendar_rows: list[dict[str, Any]]
    try:
        payload = load_canonical_resource(
            raw,
            label="canonical_calendar",
        )
    except CanonicalContractError:
        try:
            import pyarrow.parquet as parquet

            table = parquet.read_table(BytesIO(raw))
            calendar_rows = [
                dict(row) for row in table.to_pylist()
            ]
        except Exception:
            _blocked("canonical_calendar_native_read")
    else:
        rows = (
            payload.get("rows")
            if type(payload) is dict
            else None
        )
        if (
            type(rows) is not list
            or not all(type(item) is dict for item in rows)
        ):
            _blocked("canonical_calendar_native_read")
        calendar_rows = [dict(row) for row in rows]
    sessions: list[str] = []
    latest_available_at = ""
    expected_fields = {
        "available_at",
        "is_open",
        "market_id",
        "session",
    }
    for index, row in enumerate(calendar_rows):
        if (
            set(row) != expected_fields
            or row["market_id"] != "CN"
            or row["is_open"] is not True
            or type(row["session"]) is not str
            or type(row["available_at"]) is not str
        ):
            _blocked("canonical_calendar_row_shape")
        try:
            parsed_session = date.fromisoformat(row["session"])
            available_at = require_utc_timestamp(
                row["available_at"],
                label=f"calendar_rows[{index}].available_at",
            )
        except (ValueError, IdentityContractError):
            _blocked("canonical_calendar_row_value")
        if (
            parsed_session.isoformat() != row["session"]
            or parsed_session.weekday() >= 5
            or available_at > reference["cutoff"]
        ):
            _blocked("canonical_calendar_row_value")
        sessions.append(row["session"])
        latest_available_at = max(latest_available_at, available_at)
    parsed_sessions = [date.fromisoformat(session) for session in sessions]
    if (
        not sessions
        or sessions != sorted(set(sessions))
        or any(
            (current - previous).days > 15
            for previous, current in zip(
                parsed_sessions,
                parsed_sessions[1:],
                strict=False,
            )
        )
    ):
        _blocked("canonical_calendar_session_inventory")
    keys = [("CN", session) for session in sessions]
    ordered_hash = _ordered_row_hash(keys)
    return sessions, {
        "expected_keys_sha256": ordered_hash,
        "latest_available_at": latest_available_at,
        "natural_key_fields": ["market_id", "session"],
        "observed_keys_sha256": ordered_hash,
        "role": "cn_open_day_calendar",
        "row_count": len(sessions),
        "row_set_sha256": _calendar_row_set_hash(calendar_rows),
    }


def _validate_calendar_catalog_binding(
    *,
    catalog: Mapping[str, Any],
    calendar_ref: Mapping[str, str],
    sessions: Sequence[str],
    observed_summary: Mapping[str, Any],
    decision_session: str,
) -> None:
    summaries = {
        row["role"]: row for row in catalog["dataset_summaries"]
    }
    summary = summaries.get("cn_open_day_calendar")
    if (
        catalog["dataset_refs"]["cn_open_day_calendar"]
        != dict(calendar_ref)
        or catalog["decision_session"] != decision_session
        or catalog["history_start"] != sessions[0]
        or sessions[-1] != decision_session
        or summary != observed_summary
        or len(sessions) < MIN_PRODUCTION_CALENDAR_SESSIONS
    ):
        _blocked("canonical_calendar_catalog_binding")


def artifact_ref(
    artifact: Mapping[str, Any],
    *,
    relative_path: str,
) -> dict[str, str]:
    if (
        type(artifact) is not dict
        or type(relative_path) is not str
        or relative_path.startswith("/")
        or "\\" in relative_path
        or any(part in {"", ".", ".."} for part in relative_path.split("/"))
    ):
        _blocked("artifact_ref_invalid")
    identity = _identity(artifact)
    if type(identity) is not str:
        _blocked("artifact_ref_identity")
    return {
        "artifact_id": identity,
        "artifact_version": str(artifact.get("version")),
        "byte_sha256": hashlib.sha256(
            canonical_bytes(artifact) + b"\n"
        ).hexdigest(),
        "cutoff": str(artifact.get("cutoff")),
        "relative_path": relative_path,
        "semantic_sha256": str(artifact.get("semantic_sha256")),
        "strategy_id": str(artifact.get("strategy_id")),
    }


@dataclass(frozen=True)
class HoldingInput:
    symbol: str
    market_value: str


@dataclass(frozen=True)
class PermissionInput:
    symbol: str
    can_buy: bool
    can_sell: bool
    industry: str
    cluster: str


def build_holdings_snapshot(
    *,
    run_id: str,
    strategy_id: str,
    cutoff: str,
    as_of_session: str,
    available_at: str,
    nav: str,
    cash: str,
    positions: Sequence[HoldingInput],
) -> dict[str, Any]:
    require_opaque_id(run_id, label="run_id")
    require_opaque_id(strategy_id, label="strategy_id")
    require_utc_timestamp(cutoff, label="cutoff")
    require_utc_timestamp(available_at, label="available_at")
    rows = sorted(
        (
            {
                "market_value": _decimal_text(
                    _decimal(item.market_value, label=f"{item.symbol}.market_value")
                ),
                "symbol": item.symbol,
            }
            for item in positions
        ),
        key=lambda row: row["symbol"],
    )
    document = seal_semantic(
        {
            "as_of_session": as_of_session,
            "authority": dict(_NO_AUTHORITY),
            "available_at": available_at,
            "cash": _decimal_text(_decimal(cash, label="cash")),
            "created_at": cutoff,
            "cutoff": cutoff,
            "declared_all_cash": not rows,
            "nav": _decimal_text(_decimal(nav, label="nav")),
            "positions": rows,
            "protocol_version": PROTOCOL_VERSION,
            "role": "holdings_snapshot",
            "run_id": run_id,
            "snapshot_id": f"holdings-{run_id}",
            "strategy_id": strategy_id,
            "version": HOLDINGS_VERSION,
        }
    )
    validate_artifact(document)
    return document


def build_risk_policy(
    *,
    strategy_id: str,
    cutoff: str,
    effective_from: str,
    expires_at: str,
    gross_cap: str,
    cash_floor: str,
    single_name_cap: str,
    industry_cap: str,
    cluster_cap: str,
    turnover_cap: str,
) -> dict[str, Any]:
    require_opaque_id(strategy_id, label="strategy_id")
    require_utc_timestamp(cutoff, label="cutoff")
    document = seal_semantic(
        {
            "allocation_rules_sha256": ALLOCATION_RULES_SHA256,
            "authority": dict(_NO_AUTHORITY),
            "cash_floor": _decimal_text(_decimal(cash_floor, label="cash_floor")),
            "cluster_cap": _decimal_text(_decimal(cluster_cap, label="cluster_cap")),
            "created_at": cutoff,
            "cutoff": cutoff,
            "effective_from": effective_from,
            "expires_at": expires_at,
            "gross_cap": _decimal_text(_decimal(gross_cap, label="gross_cap")),
            "industry_cap": _decimal_text(
                _decimal(industry_cap, label="industry_cap")
            ),
            "permission_rules_sha256": PERMISSION_RULES_SHA256,
            "policy_id": f"risk-{strategy_id}-{cutoff[:10].replace('-', '')}",
            "protocol_version": PROTOCOL_VERSION,
            "single_name_cap": _decimal_text(
                _decimal(single_name_cap, label="single_name_cap")
            ),
            "strategy_id": strategy_id,
            "turnover_cap": _decimal_text(
                _decimal(turnover_cap, label="turnover_cap")
            ),
            "version": RISK_POLICY_VERSION,
        }
    )
    validate_artifact(document)
    return document


def build_pretrade_permissions(
    *,
    run_id: str,
    strategy_id: str,
    cutoff: str,
    decision_session: str,
    canonical_calendar_ref: Mapping[str, Any],
    pit_catalog_ref: Mapping[str, Any],
    holdings_snapshot_ref: Mapping[str, Any],
    risk_policy_ref: Mapping[str, Any],
    decisions: Sequence[PermissionInput],
    fusion_symbols: Sequence[str],
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> dict[str, Any]:
    holdings_ref = _validate_ref(
        holdings_snapshot_ref,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=HOLDINGS_VERSION,
        label="holdings_snapshot_ref",
    )
    risk_ref = _validate_ref(
        risk_policy_ref,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=RISK_POLICY_VERSION,
        label="risk_policy_ref",
    )
    calendar_ref = _validate_ref(
        canonical_calendar_ref,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=CALENDAR_DATASET_VERSION,
        label="canonical_calendar_ref",
    )
    catalog_ref = _validate_ref(
        pit_catalog_ref,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=PIT_CATALOG_VERSION,
        label="pit_catalog_ref",
    )
    holdings = _read_exact(
        holdings_ref,
        artifact_loader=artifact_loader,
        label="holdings_snapshot",
    )
    policy = _read_exact(
        risk_ref,
        artifact_loader=artifact_loader,
        label="risk_policy",
    )
    catalog = _read_exact(
        catalog_ref,
        artifact_loader=artifact_loader,
        label="pit_catalog",
    )
    if holdings["run_id"] != run_id:
        _blocked("holdings_run_binding")
    if (
        policy["permission_rules_sha256"] != PERMISSION_RULES_SHA256
        or policy["allocation_rules_sha256"] != ALLOCATION_RULES_SHA256
    ):
        _blocked("risk_policy_rule_binding")
    sessions, calendar_summary = _calendar_sessions(
        calendar_ref,
        artifact_loader=artifact_loader,
    )
    _validate_calendar_catalog_binding(
        catalog=catalog,
        calendar_ref=calendar_ref,
        sessions=sessions,
        observed_summary=calendar_summary,
        decision_session=decision_session,
    )
    if (
        not sessions
        or sessions != sorted(set(sessions))
        or decision_session not in sessions
        or holdings["as_of_session"] not in sessions
    ):
        _blocked("canonical_session_closure")
    age = sessions.index(decision_session) - sessions.index(
        holdings["as_of_session"]
    )
    if age < 0 or age > 1:
        _blocked("holdings_snapshot_stale")
    selected = tuple(fusion_symbols)
    if len(selected) != 24 or len(selected) != len(set(selected)):
        _blocked("fusion_top24_domain")
    selected_set = set(selected)
    position_values = {
        row["symbol"]: _decimal(row["market_value"], label="market_value")
        for row in holdings["positions"]
    }
    expected = selected_set | set(position_values)
    by_symbol = {item.symbol: item for item in decisions}
    if len(by_symbol) != len(decisions) or set(by_symbol) != expected:
        _blocked("permission_domain")
    nav = _decimal(holdings["nav"], label="nav")
    rows: list[dict[str, Any]] = []
    for symbol in sorted(expected):
        item = by_symbol[symbol]
        current = _weight(position_values.get(symbol, Decimal("0")) / nav)
        held = current > 0
        lane = (
            "SELECTION_POOL"
            if symbol in selected_set
            else "REVIEW_ONLY_HOLDING"
        )
        rows.append(
            {
                "can_buy": item.can_buy if lane == "SELECTION_POOL" else False,
                "can_sell": item.can_sell,
                "cluster": require_opaque_id(
                    item.cluster,
                    label=f"{symbol}.cluster",
                ),
                "current_target": _decimal_text(current),
                "held": held,
                "industry": require_opaque_id(
                    item.industry,
                    label=f"{symbol}.industry",
                ),
                "lane": lane,
                "symbol": symbol,
            }
        )
    document = seal_semantic(
        {
            "allocation_rules_sha256": ALLOCATION_RULES_SHA256,
            "authority": dict(_NO_AUTHORITY),
            "canonical_calendar_ref": calendar_ref,
            "created_at": cutoff,
            "cutoff": cutoff,
            "decision_session": decision_session,
            "holdings_snapshot_age_sessions": age,
            "holdings_snapshot_ref": holdings_ref,
            "payload": rows,
            "permission_rules_sha256": PERMISSION_RULES_SHA256,
            "permissions_id": f"permissions-{run_id}",
            "pit_catalog_ref": catalog_ref,
            "portfolio_basis": "HOLDINGS_AWARE",
            "protocol_version": PROTOCOL_VERSION,
            "risk_policy_ref": risk_ref,
            "role": "permissions",
            "run_id": run_id,
            "strategy_id": strategy_id,
            "version": PERMISSIONS_VERSION,
        }
    )
    validate_artifact(document)
    return document


def build_regime_evidence(
    *,
    run_id: str,
    strategy_id: str,
    cutoff: str,
    role: str,
    available_at: str,
    gross_multiplier: str,
) -> dict[str, Any]:
    if role not in {"macro_evidence", "markov_evidence"}:
        _blocked("regime_evidence_role")
    document = seal_semantic(
        {
            "authority": dict(_NO_AUTHORITY),
            "available_at": available_at,
            "created_at": cutoff,
            "cutoff": cutoff,
            "evidence_id": f"{role}-{run_id}",
            "gross_multiplier": _decimal_text(
                _decimal(gross_multiplier, label="gross_multiplier")
            ),
            "protocol_version": PROTOCOL_VERSION,
            "role": role,
            "run_id": run_id,
            "status": "AVAILABLE",
            "strategy_id": strategy_id,
            "version": REGIME_EVIDENCE_VERSION,
        }
    )
    validate_artifact(document)
    return document


def _cap_group(
    targets: dict[str, Decimal],
    permission_rows: Mapping[str, Mapping[str, Any]],
    *,
    field: str,
    cap: Decimal,
) -> None:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for symbol in targets:
        groups[str(permission_rows[symbol][field])].append(symbol)
    for symbols in groups.values():
        total = sum((targets[symbol] for symbol in symbols), Decimal("0"))
        if total <= cap or total == 0:
            continue
        ratio = cap / total
        for symbol in symbols:
            targets[symbol] = _weight(targets[symbol] * ratio)


def _base_targets(
    *,
    deep: Mapping[str, Any],
    permissions: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Decimal]:
    rows = {
        row["symbol"]: row for row in permissions["payload"]
    }
    targets: dict[str, Decimal] = {}
    for deep_row in deep["rows"]:
        symbol = deep_row["symbol"]
        permission = rows.get(symbol)
        if permission is None or permission["lane"] != "SELECTION_POOL":
            _blocked("deep_permission_domain")
        current = _decimal(permission["current_target"], label="current_target")
        target = _decimal(
            deep_row["target_after_deep"],
            label="target_after_deep",
        )
        if permission["held"]:
            target = max(target, current)
            if not permission["can_buy"]:
                target = current
        elif not permission["can_buy"]:
            target = Decimal("0")
        targets[symbol] = target
    for symbol, permission in rows.items():
        if permission["lane"] == "REVIEW_ONLY_HOLDING":
            targets[symbol] = _decimal(
                permission["current_target"],
                label="review_current_target",
            )
    single_cap = _decimal(policy["single_name_cap"], label="single_name_cap")
    for symbol in targets:
        targets[symbol] = min(targets[symbol], single_cap)
    _cap_group(
        targets,
        rows,
        field="industry",
        cap=_decimal(policy["industry_cap"], label="industry_cap"),
    )
    _cap_group(
        targets,
        rows,
        field="cluster",
        cap=_decimal(policy["cluster_cap"], label="cluster_cap"),
    )
    gross_cap = min(
        _decimal(policy["gross_cap"], label="gross_cap"),
        Decimal("1") - _decimal(policy["cash_floor"], label="cash_floor"),
    )
    gross = sum(targets.values(), Decimal("0"))
    if gross > gross_cap:
        ratio = gross_cap / gross
        for symbol in targets:
            targets[symbol] = _weight(targets[symbol] * ratio)
    current_weights = {
        symbol: _decimal(row["current_target"], label="current_target")
        for symbol, row in rows.items()
    }
    turnover = sum(
        (
            abs(targets[symbol] - current_weights[symbol])
            for symbol in targets
        ),
        Decimal("0"),
    )
    if turnover > _decimal(policy["turnover_cap"], label="turnover_cap"):
        _blocked("turnover_cap_infeasible")
    _enforce_sell_permissions(targets, permissions)
    return dict(sorted(targets.items()))


def _enforce_sell_permissions(
    targets: Mapping[str, Decimal],
    permissions: Mapping[str, Any],
) -> None:
    rows = {
        row["symbol"]: row for row in permissions["payload"]
    }
    for symbol, target in targets.items():
        row = rows.get(symbol)
        if row is None:
            _blocked("sell_permission_domain")
        current = _decimal(
            row["current_target"],
            label="current_target",
        )
        if (
            row["held"]
            and not row["can_sell"]
            and target < current
        ):
            _blocked(f"sell_not_permitted:{symbol}")


def _revalidate_permissions_against_holdings(
    *,
    deep: Mapping[str, Any],
    permissions: Mapping[str, Any],
    holdings: Mapping[str, Any],
) -> None:
    selected = {row["symbol"] for row in deep["rows"]}
    position_values = {
        row["symbol"]: _decimal(
            row["market_value"],
            label="holding_market_value",
        )
        for row in holdings["positions"]
    }
    expected = selected | set(position_values)
    permission_rows = {
        row["symbol"]: row for row in permissions["payload"]
    }
    if set(permission_rows) != expected:
        _blocked("permissions_holdings_domain")
    nav = _decimal(holdings["nav"], label="holdings_nav")
    for symbol, row in permission_rows.items():
        current = _weight(
            position_values.get(symbol, Decimal("0")) / nav
        )
        expected_lane = (
            "SELECTION_POOL"
            if symbol in selected
            else "REVIEW_ONLY_HOLDING"
        )
        if (
            _decimal(row["current_target"], label="current_target")
            != current
            or row["held"] != (current > 0)
            or row["lane"] != expected_lane
            or (
                expected_lane == "REVIEW_ONLY_HOLDING"
                and row["can_buy"] is not False
            )
        ):
            _blocked("permissions_holdings_reconciliation")


def _revalidate_holdings_freshness(
    *,
    permissions: Mapping[str, Any],
    holdings: Mapping[str, Any],
    strategy_id: str,
    cutoff: str,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> None:
    calendar_ref = _validate_ref(
        permissions["canonical_calendar_ref"],
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=CALENDAR_DATASET_VERSION,
        label="permissions.canonical_calendar_ref",
    )
    catalog_ref = _validate_ref(
        permissions["pit_catalog_ref"],
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=PIT_CATALOG_VERSION,
        label="permissions.pit_catalog_ref",
    )
    catalog = _read_exact(
        catalog_ref,
        artifact_loader=artifact_loader,
        label="permissions_pit_catalog",
    )
    sessions, calendar_summary = _calendar_sessions(
        calendar_ref,
        artifact_loader=artifact_loader,
    )
    decision = permissions["decision_session"]
    _validate_calendar_catalog_binding(
        catalog=catalog,
        calendar_ref=calendar_ref,
        sessions=sessions,
        observed_summary=calendar_summary,
        decision_session=decision,
    )
    as_of = holdings["as_of_session"]
    if decision not in sessions or as_of not in sessions:
        _blocked("holdings_calendar_domain")
    age = sessions.index(decision) - sessions.index(as_of)
    if (
        age < 0
        or age > 1
        or permissions["holdings_snapshot_age_sessions"] != age
    ):
        _blocked("holdings_snapshot_freshness_replay")


def _read_portfolio_inputs(
    *,
    deep_bundle_ref: Mapping[str, Any],
    permissions_ref: Mapping[str, Any],
    risk_policy_ref: Mapping[str, Any],
    run_id: str,
    strategy_id: str,
    cutoff: str,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    deep_ref = _validate_ref(
        deep_bundle_ref,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=DEEP_VERSION,
        label="deep_bundle_ref",
    )
    permissions_ref_checked = _validate_ref(
        permissions_ref,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=PERMISSIONS_VERSION,
        label="permissions_ref",
    )
    risk_ref = _validate_ref(
        risk_policy_ref,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=RISK_POLICY_VERSION,
        label="risk_policy_ref",
    )
    deep, _fusion = revalidate_deep_evidence_bundle(
        deep_ref,
        artifact_loader=artifact_loader,
    )
    permissions = _read_exact(
        permissions_ref_checked,
        artifact_loader=artifact_loader,
        label="permissions",
    )
    policy = _read_exact(
        risk_ref,
        artifact_loader=artifact_loader,
        label="risk_policy",
    )
    if (
        deep["run_id"] != run_id
        or permissions["run_id"] != run_id
    ):
        _blocked("portfolio_input_run_binding")
    holdings_ref = _validate_ref(
        permissions["holdings_snapshot_ref"],
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=HOLDINGS_VERSION,
        label="permissions.holdings_snapshot_ref",
    )
    holdings = _read_exact(
        holdings_ref,
        artifact_loader=artifact_loader,
        label="permissions_holdings_snapshot",
    )
    _revalidate_permissions_against_holdings(
        deep=deep,
        permissions=permissions,
        holdings=holdings,
    )
    _revalidate_holdings_freshness(
        permissions=permissions,
        holdings=holdings,
        strategy_id=strategy_id,
        cutoff=cutoff,
        artifact_loader=artifact_loader,
    )
    if (
        permissions["risk_policy_ref"] != risk_ref
        or permissions["permission_rules_sha256"] != PERMISSION_RULES_SHA256
        or permissions["allocation_rules_sha256"] != ALLOCATION_RULES_SHA256
        or policy["permission_rules_sha256"] != PERMISSION_RULES_SHA256
        or policy["allocation_rules_sha256"] != ALLOCATION_RULES_SHA256
    ):
        _blocked("portfolio_input_rule_binding")
    return (
        deep_ref,
        permissions_ref_checked,
        risk_ref,
        deep,
        permissions,
        policy,
    )


def _evidence_multiplier(
    references: Sequence[Mapping[str, Any]],
    *,
    strategy_id: str,
    cutoff: str,
    expected_role: str,
    run_id: str,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> tuple[list[dict[str, str]], Decimal]:
    if not references:
        _blocked(f"{expected_role}_unavailable")
    normalized: list[dict[str, str]] = []
    multipliers: list[Decimal] = []
    for index, reference in enumerate(references):
        checked = _validate_ref(
            reference,
            strategy_id=strategy_id,
            cutoff=cutoff,
            expected_version=REGIME_EVIDENCE_VERSION,
            label=f"{expected_role}_refs[{index}]",
        )
        evidence = _read_exact(
            checked,
            artifact_loader=artifact_loader,
            label=f"{expected_role}_evidence[{index}]",
        )
        if evidence["role"] != expected_role:
            _blocked(f"{expected_role}_role")
        if evidence["run_id"] != run_id:
            _blocked(f"{expected_role}_run_binding")
        normalized.append(checked)
        multipliers.append(
            _decimal(
                evidence["gross_multiplier"],
                label="gross_multiplier",
            )
        )
    normalized.sort(
        key=lambda row: (row["relative_path"], row["byte_sha256"])
    )
    return normalized, min(multipliers)


def build_macro_overlay(
    *,
    run_id: str,
    strategy_id: str,
    cutoff: str,
    deep_bundle_ref: Mapping[str, Any],
    permissions_ref: Mapping[str, Any],
    risk_policy_ref: Mapping[str, Any],
    evidence_refs: Sequence[Mapping[str, Any]],
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> dict[str, Any]:
    (
        deep_ref,
        permissions_ref_checked,
        risk_ref,
        deep,
        permissions,
        policy,
    ) = _read_portfolio_inputs(
        deep_bundle_ref=deep_bundle_ref,
        permissions_ref=permissions_ref,
        risk_policy_ref=risk_policy_ref,
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        artifact_loader=artifact_loader,
    )
    evidence, multiplier = _evidence_multiplier(
        evidence_refs,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_role="macro_evidence",
        run_id=run_id,
        artifact_loader=artifact_loader,
    )
    baseline = _base_targets(
        deep=deep,
        permissions=permissions,
        policy=policy,
    )
    return _build_overlay_document(
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        role="macro_overlay",
        baseline_ref=deep_ref,
        permissions_ref=permissions_ref_checked,
        risk_policy_ref=risk_ref,
        baseline=baseline,
        multiplier=multiplier,
        evidence_refs=evidence,
        permissions=permissions,
    )


def _build_overlay_document(
    *,
    run_id: str,
    strategy_id: str,
    cutoff: str,
    role: str,
    baseline_ref: Mapping[str, str],
    permissions_ref: Mapping[str, str],
    risk_policy_ref: Mapping[str, str],
    baseline: Mapping[str, Decimal],
    multiplier: Decimal,
    evidence_refs: Sequence[Mapping[str, str]],
    permissions: Mapping[str, Any],
) -> dict[str, Any]:
    targets = {
        symbol: _weight(value * multiplier)
        for symbol, value in baseline.items()
    }
    _enforce_sell_permissions(targets, permissions)
    input_gross = sum(baseline.values(), Decimal("0"))
    output_gross = sum(targets.values(), Decimal("0"))
    document = seal_semantic(
        {
            "authority": dict(_NO_AUTHORITY),
            "baseline_ref": dict(baseline_ref),
            "created_at": cutoff,
            "cutoff": cutoff,
            "evidence_refs": [dict(row) for row in evidence_refs],
            "input_gross": _decimal_text(input_gross),
            "output_gross": _decimal_text(output_gross),
            "overlay_id": f"{role}-{run_id}",
            "permissions_ref": dict(permissions_ref),
            "protocol_version": PROTOCOL_VERSION,
            "released_to_cash": _decimal_text(
                input_gross - output_gross
            ),
            "risk_policy_ref": dict(risk_policy_ref),
            "role": role,
            "run_id": run_id,
            "status": "APPLIED",
            "strategy_id": strategy_id,
            "target_weights": [
                {"symbol": symbol, "target": _decimal_text(targets[symbol])}
                for symbol in sorted(targets)
            ],
            "version": OVERLAY_VERSION,
        }
    )
    validate_artifact(document)
    return document


def build_markov_overlay(
    *,
    run_id: str,
    strategy_id: str,
    cutoff: str,
    macro_overlay_ref: Mapping[str, Any],
    permissions_ref: Mapping[str, Any],
    risk_policy_ref: Mapping[str, Any],
    evidence_refs: Sequence[Mapping[str, Any]],
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> dict[str, Any]:
    macro_ref = _validate_ref(
        macro_overlay_ref,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=OVERLAY_VERSION,
        label="macro_overlay_ref",
    )
    permissions_ref_checked = _validate_ref(
        permissions_ref,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=PERMISSIONS_VERSION,
        label="permissions_ref",
    )
    risk_ref = _validate_ref(
        risk_policy_ref,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=RISK_POLICY_VERSION,
        label="risk_policy_ref",
    )
    macro = _read_exact(
        macro_ref,
        artifact_loader=artifact_loader,
        label="macro_overlay",
    )
    permissions = _read_exact(
        permissions_ref_checked,
        artifact_loader=artifact_loader,
        label="markov_permissions",
    )
    if (
        macro["role"] != "macro_overlay"
        or macro["run_id"] != run_id
        or macro["permissions_ref"] != permissions_ref_checked
        or macro["risk_policy_ref"] != risk_ref
    ):
        _blocked("macro_overlay_chain")
    evidence, multiplier = _evidence_multiplier(
        evidence_refs,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_role="markov_evidence",
        run_id=run_id,
        artifact_loader=artifact_loader,
    )
    baseline = {
        row["symbol"]: _decimal(row["target"], label="macro_target")
        for row in macro["target_weights"]
    }
    return _build_overlay_document(
        run_id=run_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        role="markov_overlay",
        baseline_ref=macro_ref,
        permissions_ref=permissions_ref_checked,
        risk_policy_ref=risk_ref,
        baseline=baseline,
        multiplier=multiplier,
        evidence_refs=evidence,
        permissions=permissions,
    )


def build_production_portfolio(
    *,
    run_id: str,
    strategy_id: str,
    cutoff: str,
    fusion_top24_ref: Mapping[str, Any],
    deep_bundle_ref: Mapping[str, Any],
    holdings_snapshot_ref: Mapping[str, Any],
    permissions_ref: Mapping[str, Any],
    risk_policy_ref: Mapping[str, Any],
    macro_overlay_ref: Mapping[str, Any],
    markov_overlay_ref: Mapping[str, Any],
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> dict[str, Any]:
    expected = {
        "deep": (deep_bundle_ref, DEEP_VERSION),
        "fusion": (fusion_top24_ref, FUSION_VERSION),
        "holdings": (holdings_snapshot_ref, HOLDINGS_VERSION),
        "macro": (macro_overlay_ref, OVERLAY_VERSION),
        "markov": (markov_overlay_ref, OVERLAY_VERSION),
        "permissions": (permissions_ref, PERMISSIONS_VERSION),
        "risk": (risk_policy_ref, RISK_POLICY_VERSION),
    }
    refs = {
        label: _validate_ref(
            reference,
            strategy_id=strategy_id,
            cutoff=cutoff,
            expected_version=version,
            label=f"{label}_ref",
        )
        for label, (reference, version) in expected.items()
    }
    documents = {
        label: _read_exact(
            refs[label],
            artifact_loader=artifact_loader,
            label=label,
        )
        for label in expected
    }
    replayed_deep, replayed_fusion = revalidate_deep_evidence_bundle(
        refs["deep"],
        artifact_loader=artifact_loader,
    )
    if (
        replayed_deep != documents["deep"]
        or replayed_fusion != documents["fusion"]
    ):
        _blocked("final_deep_replay_binding")
    if any(
        label != "risk" and document["run_id"] != run_id
        for label, document in documents.items()
    ):
        _blocked("final_run_binding")
    macro = documents["macro"]
    markov = documents["markov"]
    permissions = documents["permissions"]
    _revalidate_permissions_against_holdings(
        deep=documents["deep"],
        permissions=permissions,
        holdings=documents["holdings"],
    )
    _revalidate_holdings_freshness(
        permissions=permissions,
        holdings=documents["holdings"],
        strategy_id=strategy_id,
        cutoff=cutoff,
        artifact_loader=artifact_loader,
    )
    if (
        permissions["portfolio_basis"] != "HOLDINGS_AWARE"
        or permissions["holdings_snapshot_ref"] != refs["holdings"]
        or permissions["risk_policy_ref"] != refs["risk"]
        or macro["role"] != "macro_overlay"
        or markov["role"] != "markov_overlay"
        or macro["baseline_ref"] != refs["deep"]
        or markov["baseline_ref"] != refs["macro"]
        or macro["permissions_ref"] != refs["permissions"]
        or markov["permissions_ref"] != refs["permissions"]
        or macro["risk_policy_ref"] != refs["risk"]
        or markov["risk_policy_ref"] != refs["risk"]
    ):
        _blocked("final_portfolio_chain")
    final_targets = {
        row["symbol"]: _decimal(row["target"], label="markov_target")
        for row in markov["target_weights"]
    }
    permission_rows = {
        row["symbol"]: row for row in permissions["payload"]
    }
    if set(final_targets) != set(permission_rows):
        _blocked("final_permission_domain")
    for symbol, target in final_targets.items():
        permission = permission_rows[symbol]
        current = _decimal(
            permission["current_target"],
            label="current_target",
        )
        if (
            (not permission["can_buy"] and target > current)
            or (
                permission["held"]
                and not permission["can_sell"]
                and target < current
            )
            or (
                permission["lane"] == "REVIEW_ONLY_HOLDING"
                and target > current
            )
        ):
            _blocked("final_permission_positive_delta")
    selected = [
        row["symbol"] for row in documents["fusion"]["rows"]
    ]
    deep_symbols = [
        row["symbol"] for row in documents["deep"]["rows"]
    ]
    if selected != deep_symbols:
        _blocked("final_top24_deep_domain")
    target_rows = [
        {
            "current_target": permission_rows[symbol]["current_target"],
            "final_target": _decimal_text(final_targets[symbol]),
            "lane": permission_rows[symbol]["lane"],
            "symbol": symbol,
        }
        for symbol in sorted(final_targets)
    ]
    gross = sum(final_targets.values(), Decimal("0"))
    if gross > 1:
        _blocked("final_gross_exceeds_one")
    document = seal_semantic(
        {
            "authority": dict(_NO_AUTHORITY),
            "cash_weight": _decimal_text(Decimal("1") - gross),
            "created_at": cutoff,
            "cutoff": cutoff,
            "deep_bundle_ref": refs["deep"],
            "fusion_top24_ref": refs["fusion"],
            "gross_weight": _decimal_text(gross),
            "holdings_snapshot_ref": refs["holdings"],
            "macro_overlay_ref": refs["macro"],
            "markov_overlay_ref": refs["markov"],
            "output_id": f"portfolio-{run_id}",
            "permissions_ref": refs["permissions"],
            "portfolio_basis": "HOLDINGS_AWARE",
            "protocol_version": PROTOCOL_VERSION,
            "review_only_holdings": sorted(
                symbol
                for symbol, row in permission_rows.items()
                if row["lane"] == "REVIEW_ONLY_HOLDING"
            ),
            "risk_policy_ref": refs["risk"],
            "run_id": run_id,
            "selection_pool_symbols": selected,
            "status": "COMPLETE",
            "strategy_id": strategy_id,
            "targets": target_rows,
            "version": PORTFOLIO_VERSION,
        }
    )
    validate_artifact(document)
    return document


def revalidate_production_portfolio(
    portfolio_ref: Mapping[str, Any],
    *,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> dict[str, Any]:
    """Replay a persisted portfolio from its exact referenced closure."""

    reference = _validate_ref(
        portfolio_ref,
        strategy_id=str(portfolio_ref.get("strategy_id", "")),
        cutoff=str(portfolio_ref.get("cutoff", "")),
        expected_version=PORTFOLIO_VERSION,
        label="portfolio_ref",
    )
    observed = _read_exact(
        reference,
        artifact_loader=artifact_loader,
        label="portfolio_output",
    )
    replayed = build_production_portfolio(
        run_id=observed["run_id"],
        strategy_id=observed["strategy_id"],
        cutoff=observed["cutoff"],
        fusion_top24_ref=observed["fusion_top24_ref"],
        deep_bundle_ref=observed["deep_bundle_ref"],
        holdings_snapshot_ref=observed["holdings_snapshot_ref"],
        permissions_ref=observed["permissions_ref"],
        risk_policy_ref=observed["risk_policy_ref"],
        macro_overlay_ref=observed["macro_overlay_ref"],
        markov_overlay_ref=observed["markov_overlay_ref"],
        artifact_loader=artifact_loader,
    )
    if observed != replayed:
        _blocked("portfolio_output_replay_mismatch")
    return observed


__all__ = [
    "ALLOCATION_RULES_SHA256",
    "HOLDINGS_VERSION",
    "HoldingInput",
    "PERMISSION_RULES_SHA256",
    "PermissionInput",
    "PortfolioControlError",
    "artifact_ref",
    "build_holdings_snapshot",
    "build_macro_overlay",
    "build_markov_overlay",
    "build_pretrade_permissions",
    "build_production_portfolio",
    "build_regime_evidence",
    "build_risk_policy",
    "revalidate_production_portfolio",
]
