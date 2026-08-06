"""Pure read-only Phase 1 portfolio-cycle readiness projections.

These builders deliberately do not resolve paths, discover artifacts, publish
runs, or mutate any authority domain.  They only project already verified
foundation results and explicit gate evidence into deterministic status DTOs.

``FOUNDATION_VALIDATED`` and
``PUBLIC_CLOSURE_ACTIVE_FOUNDATION_ONLY`` are diagnostic states.  Neither is a
business-readiness or business-cycle-completion claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Final, Mapping

from quant_investor.v17_mainline.constants import (
    ACTIVE_STATE,
    BLOCKED_PREFIX,
    PROTOCOL,
    UNINITIALIZED_STATE,
    MainlineBlocker,
)
from quant_investor.v17_mainline.contracts import (
    MainlineContractError,
    require_identifier,
    require_timestamp,
    validate_ref,
)
from quant_investor.v17_mainline.runtime import MainlineResolution

from .contracts import PortfolioCycleError
from .contracts import (
    HOLDINGS_ACCOUNTING_POLICY_SCHEMA_ID,
    HOLDINGS_LEDGER_SCHEMA_ID,
    HOLDINGS_MANIFEST_SCHEMA_ID,
    HOLDINGS_POINTER_SCHEMA_ID,
    HOLDINGS_PRICE_SOURCE_SCHEMA_ID,
    IDENTITY_DECLARATION_SCHEMA_ID,
    VerifiedHoldingsBaseline,
    VerifiedStrategyIdentity,
)
from .holdings import resolve_holdings_baseline
from .identity import resolve_strategy_identity

DECISION_INPUT_READINESS_SCHEMA_ID: Final = "myquant.v17.v4.decision-input-readiness.v1"
PUBLIC_CYCLE_STATUS_SCHEMA_ID: Final = "myquant.v17.v4.public-cycle-status.v1"
PHASE_CAPABILITY: Final = "FOUNDATION_ONLY"
HISTORICAL_HOLDINGS_LABEL: Final = "aggressive_tech_manufacturing"

PRE_RUN_STATES: Final = frozenset({"BLOCKED", "FOUNDATION_VALIDATED"})
POST_RUN_STATES: Final = frozenset({"BLOCKED", "PUBLIC_CLOSURE_ACTIVE_FOUNDATION_ONLY"})


class ReadinessBlocker(str, Enum):
    """Closed Phase 1 blocker vocabulary outside mainline closure details."""

    V17_STRATEGY_ID_UNCONFIRMED = "V17_STRATEGY_ID_UNCONFIRMED"
    HOLDINGS_BASELINE_UNAVAILABLE = "HOLDINGS_BASELINE_UNAVAILABLE"
    IDENTITY_DECLARATION_INVALID = "IDENTITY_DECLARATION_INVALID"
    HOLDINGS_BASELINE_INVALID = "HOLDINGS_BASELINE_INVALID"
    STRICT_CN_DATA_UNVERIFIED = "STRICT_CN_DATA_UNVERIFIED"
    PIT_UNVERIFIED = "PIT_UNVERIFIED"
    FUNDAMENTAL_UNVERIFIED = "FUNDAMENTAL_UNVERIFIED"
    MACRO_UNVERIFIED = "MACRO_UNVERIFIED"
    RELEASE_CALENDAR_UNVERIFIED = "RELEASE_CALENDAR_UNVERIFIED"
    FACTOR_ACTIVE_SET_UNAVAILABLE = "FACTOR_ACTIVE_SET_UNAVAILABLE"
    RISK_POLICY_UNAVAILABLE = "RISK_POLICY_UNAVAILABLE"
    PORTFOLIO_POLICY_UNAVAILABLE = "PORTFOLIO_POLICY_UNAVAILABLE"
    V17_MAINLINE_UNINITIALIZED = "V17_MAINLINE_UNINITIALIZED"
    CAPABILITY_BLOCKED_V17_MAINLINE_PUBLISHER = "CAPABILITY_BLOCKED_V17_MAINLINE_PUBLISHER"
    PAPER_SIMULATION_UNAVAILABLE = "PAPER_SIMULATION_UNAVAILABLE"
    LEARNING_RUNTIME_UNAVAILABLE = "LEARNING_RUNTIME_UNAVAILABLE"


MAINLINE_BLOCKED_CODES: Final = frozenset(
    f"{BLOCKED_PREFIX}{blocker.value}" for blocker in MainlineBlocker
)
ALL_BLOCKER_CODES: Final = (
    frozenset(blocker.value for blocker in ReadinessBlocker) | MAINLINE_BLOCKED_CODES
)

AUTHORITY_FLAGS: Final = {
    "broker": False,
    "order": False,
    "execution": False,
    "trade": False,
    "provider": False,
    "mainline_write": False,
    "factor_write": False,
    "holdings_write": False,
    "paper_ledger_write_authorized": False,
}


@dataclass(frozen=True)
class GateEvidence:
    """One explicit gate result and its exact immutable evidence reference."""

    verified: bool
    ref: object | None = None


GATE_NAMES: Final = (
    "strict_cn_data",
    "pit",
    "fundamental",
    "macro",
    "release_calendar",
    "factor_active_set",
    "risk_policy",
    "portfolio_policy",
    "mainline_publisher",
    "paper_simulation",
    "learning_runtime",
)

_GATE_BLOCKERS: Final = {
    "strict_cn_data": ReadinessBlocker.STRICT_CN_DATA_UNVERIFIED.value,
    "pit": ReadinessBlocker.PIT_UNVERIFIED.value,
    "fundamental": ReadinessBlocker.FUNDAMENTAL_UNVERIFIED.value,
    "macro": ReadinessBlocker.MACRO_UNVERIFIED.value,
    "release_calendar": ReadinessBlocker.RELEASE_CALENDAR_UNVERIFIED.value,
    "factor_active_set": ReadinessBlocker.FACTOR_ACTIVE_SET_UNAVAILABLE.value,
    "risk_policy": ReadinessBlocker.RISK_POLICY_UNAVAILABLE.value,
    "portfolio_policy": ReadinessBlocker.PORTFOLIO_POLICY_UNAVAILABLE.value,
    "mainline_publisher": (ReadinessBlocker.CAPABILITY_BLOCKED_V17_MAINLINE_PUBLISHER.value),
    "paper_simulation": ReadinessBlocker.PAPER_SIMULATION_UNAVAILABLE.value,
    "learning_runtime": ReadinessBlocker.LEARNING_RUNTIME_UNAVAILABLE.value,
}


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be an exact boolean")
    return value


def _attribute(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _normalize_ref(
    value: object,
    *,
    label: str,
    expected_schema_id: str | None = None,
) -> dict[str, str]:
    if isinstance(value, Mapping):
        candidate = dict(value)
    else:
        candidate = {
            "schema_id": getattr(value, "schema_id", None),
            "relative_path": getattr(value, "relative_path", None),
            "byte_sha256": getattr(value, "byte_sha256", None),
        }
    try:
        normalized = validate_ref(candidate, label=label)
    except MainlineContractError as exc:
        raise ValueError(str(exc)) from exc
    if expected_schema_id is not None and normalized["schema_id"] != expected_schema_id:
        raise ValueError(f"{label} schema_id mismatch")
    return normalized


def _normalize_gate(value: object, *, name: str) -> dict[str, Any]:
    if isinstance(value, GateEvidence):
        verified = value.verified
        ref = value.ref
    elif isinstance(value, Mapping):
        if set(value) != {"verified", "ref"}:
            raise ValueError(f"gate {name} fields must be exact")
        verified = value.get("verified")
        ref = value.get("ref")
    else:
        raise ValueError(f"gate {name} must be GateEvidence or an exact mapping")

    is_verified = _require_bool(verified, label=f"gate {name}.verified")
    if is_verified:
        if ref is None:
            raise ValueError(f"verified gate {name} requires an exact ref")
        normalized_ref: dict[str, str] | None = _normalize_ref(ref, label=f"gate {name}.ref")
    else:
        if ref is not None:
            raise ValueError(f"unverified gate {name} must not carry a ref")
        normalized_ref = None
    return {"verified": is_verified, "ref": normalized_ref}


def _normalize_gates(
    gates: Mapping[str, GateEvidence | Mapping[str, object]] | None,
) -> dict[str, dict[str, Any]]:
    if gates is None:
        return {name: {"verified": False, "ref": None} for name in GATE_NAMES}
    if type(gates) is not dict or set(gates) != set(GATE_NAMES):
        raise ValueError("gates must contain the exact Phase 1 gate names")
    return {name: _normalize_gate(gates[name], name=name) for name in GATE_NAMES}


def _nonempty_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _normalize_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, VerifiedStrategyIdentity) or value.verified is not True:
        raise ValueError("identity must be a verified strategy-identity result")
    return _validate_identity_dto(
        {
            "canonical_strategy_id": value.canonical_strategy_id,
            "historical_label": value.historical_label,
            "declaration_ref": value.declaration_ref.as_dict(),
            "declared_by": value.declared_by,
            "declared_at": value.declared_at,
            "authority_kind": value.authority_kind,
            "provenance": value.provenance,
        }
    )


_IDENTITY_DTO_FIELDS: Final = frozenset(
    {
        "canonical_strategy_id",
        "historical_label",
        "declaration_ref",
        "declared_by",
        "declared_at",
        "authority_kind",
        "provenance",
    }
)


def _validate_identity_dto(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _IDENTITY_DTO_FIELDS:
        raise ValueError("identity status fields must be exact")
    try:
        strategy_id = require_identifier(
            value.get("canonical_strategy_id"),
            label="canonical_strategy_id",
        )
        declared_at = require_timestamp(value.get("declared_at"), label="declared_at")
    except MainlineContractError as exc:
        raise ValueError(str(exc)) from exc
    if value.get("authority_kind") != "owner_declaration":
        raise ValueError("identity authority_kind must be owner_declaration")
    return {
        "canonical_strategy_id": strategy_id,
        "historical_label": _nonempty_text(value.get("historical_label"), label="historical_label"),
        "declaration_ref": _normalize_ref(
            value.get("declaration_ref"),
            label="declaration_ref",
            expected_schema_id=IDENTITY_DECLARATION_SCHEMA_ID,
        ),
        "declared_by": _nonempty_text(value.get("declared_by"), label="declared_by"),
        "declared_at": declared_at,
        "authority_kind": "owner_declaration",
        "provenance": _nonempty_text(value.get("provenance"), label="provenance"),
    }


def _normalize_holdings(value: object) -> dict[str, Any]:
    if not isinstance(value, VerifiedHoldingsBaseline) or value.verified is not True:
        raise ValueError("holdings must be a verified holdings-baseline result")
    return _validate_holdings_dto(
        {
            "canonical_strategy_id": value.canonical_strategy_id,
            "account_id": value.account_id,
            "currency": value.currency,
            "trade_date": value.trade_date,
            "as_of": value.as_of,
            "valuation_at": value.valuation_at,
            "decision_cutoff": value.decision_cutoff,
            "pointer_updated_at": value.pointer_updated_at,
            "pointer_ref": value.pointer_ref.as_dict(),
            "manifest_ref": value.manifest_ref.as_dict(),
            "accounting_policy_ref": value.accounting_policy_ref.as_dict(),
            "price_source_ref": value.price_source_ref.as_dict(),
            "ledger_ref": value.ledger_ref.as_dict(),
        }
    )


_HOLDINGS_DTO_FIELDS: Final = frozenset(
    {
        "canonical_strategy_id",
        "account_id",
        "currency",
        "trade_date",
        "as_of",
        "valuation_at",
        "decision_cutoff",
        "pointer_updated_at",
        "pointer_ref",
        "manifest_ref",
        "accounting_policy_ref",
        "price_source_ref",
        "ledger_ref",
    }
)

_HOLDINGS_REF_SCHEMAS: Final = {
    "pointer_ref": HOLDINGS_POINTER_SCHEMA_ID,
    "manifest_ref": HOLDINGS_MANIFEST_SCHEMA_ID,
    "accounting_policy_ref": HOLDINGS_ACCOUNTING_POLICY_SCHEMA_ID,
    "price_source_ref": HOLDINGS_PRICE_SOURCE_SCHEMA_ID,
    "ledger_ref": HOLDINGS_LEDGER_SCHEMA_ID,
}


def _validate_holdings_dto(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _HOLDINGS_DTO_FIELDS:
        raise ValueError("holdings status fields must be exact")
    try:
        strategy_id = require_identifier(
            value.get("canonical_strategy_id"),
            label="holdings.canonical_strategy_id",
        )
        require_identifier(value.get("account_id"), label="holdings.account_id")
        trade_date = str(value.get("trade_date"))
        as_of_text = require_timestamp(value.get("as_of"), label="holdings.as_of")
        valuation_at_text = require_timestamp(
            value.get("valuation_at"), label="holdings.valuation_at"
        )
        decision_cutoff_text = require_timestamp(
            value.get("decision_cutoff"), label="holdings.decision_cutoff"
        )
        pointer_updated_at_text = require_timestamp(
            value.get("pointer_updated_at"), label="holdings.pointer_updated_at"
        )
    except MainlineContractError as exc:
        raise ValueError(str(exc)) from exc
    try:
        parsed_trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
        as_of = datetime.strptime(as_of_text, "%Y-%m-%dT%H:%M:%SZ")
        valuation_at = datetime.strptime(valuation_at_text, "%Y-%m-%dT%H:%M:%SZ")
        decision_cutoff = datetime.strptime(decision_cutoff_text, "%Y-%m-%dT%H:%M:%SZ")
        pointer_updated_at = datetime.strptime(pointer_updated_at_text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("holdings.trade_date must be a real canonical date") from exc
    if (
        parsed_trade_date.isoformat() != trade_date
        or parsed_trade_date > as_of.date()
        or not (as_of <= valuation_at <= decision_cutoff <= pointer_updated_at)
    ):
        raise ValueError("holdings chronology is invalid")
    if value.get("currency") != "CNY":
        raise ValueError("holdings currency must be CNY")

    refs = {
        name: _normalize_ref(
            value.get(name),
            label=f"holdings.{name}",
            expected_schema_id=schema_id,
        )
        for name, schema_id in _HOLDINGS_REF_SCHEMAS.items()
    }
    return {
        "canonical_strategy_id": strategy_id,
        "account_id": value["account_id"],
        "currency": "CNY",
        "trade_date": trade_date,
        "as_of": value["as_of"],
        "valuation_at": value["valuation_at"],
        "decision_cutoff": value["decision_cutoff"],
        "pointer_updated_at": value["pointer_updated_at"],
        **refs,
    }


def _sorted_blockers(values: list[str] | tuple[str, ...]) -> list[str]:
    unknown = set(values) - ALL_BLOCKER_CODES
    if unknown:
        raise ValueError(f"unknown Phase 1 blockers: {sorted(unknown)!r}")
    return sorted(set(values))


def _identity_status(
    identity: object | None, *, invalid: bool
) -> tuple[dict[str, Any] | None, str | None]:
    if identity is None:
        blocker = (
            ReadinessBlocker.IDENTITY_DECLARATION_INVALID.value
            if invalid
            else ReadinessBlocker.V17_STRATEGY_ID_UNCONFIRMED.value
        )
        return None, blocker
    try:
        return _normalize_identity(identity), None
    except ValueError:
        return None, ReadinessBlocker.IDENTITY_DECLARATION_INVALID.value


def _holdings_status(
    holdings: object | None, *, invalid: bool
) -> tuple[dict[str, Any] | None, str | None]:
    if holdings is None:
        blocker = (
            ReadinessBlocker.HOLDINGS_BASELINE_INVALID.value
            if invalid
            else ReadinessBlocker.HOLDINGS_BASELINE_UNAVAILABLE.value
        )
        return None, blocker
    try:
        return _normalize_holdings(holdings), None
    except ValueError:
        return None, ReadinessBlocker.HOLDINGS_BASELINE_INVALID.value


def _foundation_bindings_match(
    identity: Mapping[str, Any] | None,
    holdings: Mapping[str, Any] | None,
    *,
    decision_cutoff: str | None,
) -> bool:
    if identity is None or holdings is None:
        return True
    return identity["canonical_strategy_id"] == holdings["canonical_strategy_id"] and (
        decision_cutoff is None or holdings["decision_cutoff"] == decision_cutoff
    )


def build_decision_input_readiness(
    *,
    identity: object | None,
    holdings: object | None,
    identity_invalid: bool = False,
    holdings_invalid: bool = False,
    gates: Mapping[str, GateEvidence | Mapping[str, object]] | None = None,
    decision_cutoff: str | None = None,
    synthetic_only: bool,
) -> dict[str, Any]:
    """Build the pre-run Phase 1 diagnostic from explicit verified inputs.

    Missing and invalid foundation inputs are distinct.  Gate evidence is
    caller-supplied and exact; this function never discovers an artifact.
    """

    is_synthetic = _require_bool(synthetic_only, label="synthetic_only")
    identity_is_invalid = _require_bool(identity_invalid, label="identity_invalid")
    holdings_is_invalid = _require_bool(holdings_invalid, label="holdings_invalid")
    if identity_is_invalid and identity is not None:
        raise ValueError("identity_invalid conflicts with an identity result")
    if holdings_is_invalid and holdings is not None:
        raise ValueError("holdings_invalid conflicts with a holdings result")
    if decision_cutoff is not None:
        try:
            requested_decision_cutoff = require_timestamp(decision_cutoff, label="decision_cutoff")
        except MainlineContractError as exc:
            raise ValueError(str(exc)) from exc
    else:
        requested_decision_cutoff = None

    normalized_identity, identity_blocker = _identity_status(identity, invalid=identity_is_invalid)
    normalized_holdings, holdings_blocker = _holdings_status(holdings, invalid=holdings_is_invalid)
    blockers = [blocker for blocker in (identity_blocker, holdings_blocker) if blocker is not None]

    if not _foundation_bindings_match(
        normalized_identity,
        normalized_holdings,
        decision_cutoff=requested_decision_cutoff,
    ):
        normalized_holdings = None
        blockers.append(ReadinessBlocker.HOLDINGS_BASELINE_INVALID.value)

    normalized_gates = _normalize_gates(gates)
    for name in GATE_NAMES:
        if not normalized_gates[name]["verified"]:
            blockers.append(_GATE_BLOCKERS[name])

    foundation_validated = normalized_identity is not None and normalized_holdings is not None
    state = "FOUNDATION_VALIDATED" if foundation_validated else "BLOCKED"
    if state not in PRE_RUN_STATES:  # pragma: no cover - frozen invariant
        raise AssertionError("unexpected pre-run state")

    return {
        "schema_id": DECISION_INPUT_READINESS_SCHEMA_ID,
        "protocol": PROTOCOL,
        "state": state,
        "phase_capability": PHASE_CAPABILITY,
        "synthetic_only": is_synthetic,
        "read_only": True,
        "operational_authority": False,
        "foundation_validated": foundation_validated,
        "business_ready": False,
        "business_cycle_closed": False,
        "canonical_strategy_id": (
            normalized_identity["canonical_strategy_id"]
            if normalized_identity is not None
            else None
        ),
        "decision_cutoff": (
            requested_decision_cutoff
            if requested_decision_cutoff is not None
            else (
                normalized_holdings["decision_cutoff"] if normalized_holdings is not None else None
            )
        ),
        "decision_cutoff_verified": normalized_holdings is not None,
        "identity": normalized_identity,
        "holdings": normalized_holdings,
        "gates": normalized_gates,
        "blockers": _sorted_blockers(blockers),
        "authority": dict(AUTHORITY_FLAGS),
    }


def _require_path_sha_pair(
    path: str | None,
    sha256: str | None,
    *,
    label: str,
) -> None:
    if (path is None) != (sha256 is None):
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_ARGUMENTS_INVALID",
            f"{label} path and SHA-256 must be provided together",
        )


_RERAISE_READ_FAILURES: Final = frozenset(
    {
        "PORTFOLIO_CYCLE_STORAGE_SECURITY",
        "PORTFOLIO_CYCLE_STABLE_READ_FAILED",
        "PORTFOLIO_CYCLE_READ_BOUND_EXCEEDED",
    }
)


def _identity_input(
    workspace_root: str | Path,
    *,
    path: str | None,
    sha256: str | None,
    strategy_id: str | None,
    historical_label: str,
) -> tuple[object | None, bool]:
    if path is None or sha256 is None:
        return None, False
    try:
        identity = resolve_strategy_identity(
            workspace_root,
            declaration_path=path,
            declaration_sha256=sha256,
            expected_historical_label=historical_label,
        )
    except PortfolioCycleError as exc:
        if exc.code in _RERAISE_READ_FAILURES:
            raise
        return None, True
    if strategy_id is not None and identity.canonical_strategy_id != strategy_id:
        return None, True
    return identity, False


def _holdings_input(
    workspace_root: str | Path,
    *,
    path: str | None,
    sha256: str | None,
    identity: object | None,
    decision_cutoff: str,
) -> tuple[object | None, bool]:
    if path is None or sha256 is None:
        return None, False
    if identity is None:
        return None, True
    expected_strategy_id = _attribute(identity, "canonical_strategy_id")
    if type(expected_strategy_id) is not str:
        return None, True
    try:
        holdings = resolve_holdings_baseline(
            workspace_root,
            pointer_path=path,
            pointer_sha256=sha256,
            expected_strategy_id=expected_strategy_id,
        )
    except PortfolioCycleError as exc:
        if exc.code in _RERAISE_READ_FAILURES:
            raise
        return None, True
    if holdings.decision_cutoff != decision_cutoff:
        return None, True
    return holdings, False


def derive_decision_input_readiness(
    workspace_root: str | Path,
    *,
    strategy_id: str | None = None,
    identity_path: str | None = None,
    identity_sha256: str | None = None,
    holdings_pointer_path: str | None = None,
    holdings_pointer_sha256: str | None = None,
    decision_cutoff: str,
    expected_historical_label: str = HISTORICAL_HOLDINGS_LABEL,
    synthetic_only: bool = False,
) -> dict[str, Any]:
    """Resolve only explicitly supplied foundation artifacts, then project status.

    This is the read-only CLI orchestration boundary.  It does not accept
    caller-asserted data/Factor/policy booleans: those gates stay blocked until
    versioned exact-ref validators exist.  The supplied decision cutoff is a
    diagnostic constraint and never substitutes for the holdings manifest's
    cutoff.
    """

    _require_path_sha_pair(identity_path, identity_sha256, label="identity")
    _require_path_sha_pair(
        holdings_pointer_path,
        holdings_pointer_sha256,
        label="holdings pointer",
    )
    try:
        canonical_cutoff = require_timestamp(decision_cutoff, label="decision_cutoff")
        requested_strategy_id = (
            require_identifier(strategy_id, label="strategy_id")
            if strategy_id is not None
            else None
        )
    except MainlineContractError as exc:
        raise PortfolioCycleError("PORTFOLIO_CYCLE_ARGUMENTS_INVALID", str(exc)) from exc
    is_synthetic = _require_bool(synthetic_only, label="synthetic_only")

    identity, identity_invalid = _identity_input(
        workspace_root,
        path=identity_path,
        sha256=identity_sha256,
        strategy_id=requested_strategy_id,
        historical_label=expected_historical_label,
    )
    holdings, holdings_invalid = _holdings_input(
        workspace_root,
        path=holdings_pointer_path,
        sha256=holdings_pointer_sha256,
        identity=identity,
        decision_cutoff=canonical_cutoff,
    )

    return build_decision_input_readiness(
        identity=identity,
        holdings=holdings,
        identity_invalid=identity_invalid,
        holdings_invalid=holdings_invalid,
        gates=None,
        decision_cutoff=canonical_cutoff,
        synthetic_only=is_synthetic,
    )


_PRE_RUN_STATUS_FIELDS: Final = frozenset(
    {
        "schema_id",
        "protocol",
        "state",
        "phase_capability",
        "synthetic_only",
        "read_only",
        "operational_authority",
        "foundation_validated",
        "business_ready",
        "business_cycle_closed",
        "canonical_strategy_id",
        "decision_cutoff",
        "decision_cutoff_verified",
        "identity",
        "holdings",
        "gates",
        "blockers",
        "authority",
    }
)


def _validate_pre_run_status(value: Mapping[str, Any], *, synthetic_only: bool) -> None:
    if type(value) is not dict:
        raise ValueError("decision_input_readiness must be an exact dict")
    if (
        set(value) != _PRE_RUN_STATUS_FIELDS
        or value.get("schema_id") != DECISION_INPUT_READINESS_SCHEMA_ID
        or value.get("protocol") != PROTOCOL
        or value.get("state") not in PRE_RUN_STATES
        or value.get("phase_capability") != PHASE_CAPABILITY
        or value.get("synthetic_only") is not synthetic_only
        or value.get("read_only") is not True
        or value.get("operational_authority") is not False
        or value.get("business_ready") is not False
        or value.get("business_cycle_closed") is not False
        or value.get("authority") != AUTHORITY_FLAGS
    ):
        raise ValueError("decision_input_readiness is not a Phase 1 status")

    identity = None if value.get("identity") is None else _validate_identity_dto(value["identity"])
    holdings = None if value.get("holdings") is None else _validate_holdings_dto(value["holdings"])
    gates = _normalize_gates(value.get("gates"))
    foundation_validated = identity is not None and holdings is not None
    expected_state = "FOUNDATION_VALIDATED" if foundation_validated else "BLOCKED"
    expected_strategy_id = identity["canonical_strategy_id"] if identity is not None else None

    cutoff_value = value.get("decision_cutoff")
    if cutoff_value is not None:
        try:
            normalized_cutoff = require_timestamp(cutoff_value, label="decision_cutoff")
        except MainlineContractError as exc:
            raise ValueError(str(exc)) from exc
    else:
        normalized_cutoff = None
    expected_cutoff_verified = holdings is not None
    if holdings is not None and (
        identity is None
        or holdings["canonical_strategy_id"] != identity["canonical_strategy_id"]
        or normalized_cutoff != holdings["decision_cutoff"]
    ):
        raise ValueError("decision_input_readiness foundation bindings do not match")

    if (
        value.get("state") != expected_state
        or value.get("foundation_validated") is not foundation_validated
        or value.get("canonical_strategy_id") != expected_strategy_id
        or value.get("decision_cutoff_verified") is not expected_cutoff_verified
    ):
        raise ValueError("decision_input_readiness derived fields are inconsistent")

    blockers = value.get("blockers")
    if type(blockers) is not list or blockers != _sorted_blockers(tuple(blockers)):
        raise ValueError("decision_input_readiness blockers are not canonical")
    expected_blockers = {_GATE_BLOCKERS[name] for name in GATE_NAMES if not gates[name]["verified"]}
    if identity is None:
        identity_blockers = {
            ReadinessBlocker.V17_STRATEGY_ID_UNCONFIRMED.value,
            ReadinessBlocker.IDENTITY_DECLARATION_INVALID.value,
        }
        present = identity_blockers.intersection(blockers)
        if len(present) != 1:
            raise ValueError("decision_input_readiness identity blocker is inconsistent")
        expected_blockers.update(present)
    if holdings is None:
        holdings_blockers = {
            ReadinessBlocker.HOLDINGS_BASELINE_UNAVAILABLE.value,
            ReadinessBlocker.HOLDINGS_BASELINE_INVALID.value,
        }
        present = holdings_blockers.intersection(blockers)
        if len(present) != 1:
            raise ValueError("decision_input_readiness holdings blocker is inconsistent")
        expected_blockers.update(present)
    if set(blockers) != expected_blockers:
        raise ValueError("decision_input_readiness blockers do not match its evidence")


def _active_closure_refs(public_run: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    return {
        name: _normalize_ref(public_run.get(name), label=f"public_run.{name}")
        for name in (
            "active_pointer_ref",
            "mainline_run_ref",
            "formal_output_ref",
            "portfolio_output_ref",
            "source_closure_ref",
        )
    }


def build_public_cycle_status(
    *,
    decision_input_readiness: Mapping[str, Any],
    mainline_resolution: MainlineResolution,
    synthetic_only: bool,
) -> dict[str, Any]:
    """Build the post-run public-closure diagnostic without upgrading authority."""

    is_synthetic = _require_bool(synthetic_only, label="synthetic_only")
    _validate_pre_run_status(decision_input_readiness, synthetic_only=is_synthetic)
    if not isinstance(mainline_resolution, MainlineResolution):
        raise ValueError("mainline_resolution must come from derive_mainline_state")

    blockers = list(decision_input_readiness["blockers"])
    public_closure_refs: dict[str, dict[str, str]] | None = None
    active = mainline_resolution.is_active

    if active:
        public_run = mainline_resolution.public_run
        if type(public_run) is not dict:
            raise ValueError("ACTIVE mainline resolution has no public run")
        expected_strategy_id = decision_input_readiness.get("canonical_strategy_id")
        if public_run.get("state") != ACTIVE_STATE or (
            expected_strategy_id is not None
            and public_run.get("canonical_strategy_id") != expected_strategy_id
        ):
            active = False
            blockers.append(f"{BLOCKED_PREFIX}{MainlineBlocker.ACTIVE_RUN_INVALID.value}")
        else:
            try:
                public_closure_refs = _active_closure_refs(public_run)
            except ValueError:
                active = False
                blockers.append(f"{BLOCKED_PREFIX}{MainlineBlocker.ACTIVE_RUN_INVALID.value}")
    elif mainline_resolution.derived_state == UNINITIALIZED_STATE:
        blockers.append(ReadinessBlocker.V17_MAINLINE_UNINITIALIZED.value)
    elif mainline_resolution.derived_state in MAINLINE_BLOCKED_CODES:
        blockers.append(mainline_resolution.derived_state)
    else:
        raise ValueError("mainline_resolution has an unknown derived state")

    # Phase 1 never implements or authorizes these downstream state changes.
    blockers.extend(
        (
            ReadinessBlocker.PAPER_SIMULATION_UNAVAILABLE.value,
            ReadinessBlocker.LEARNING_RUNTIME_UNAVAILABLE.value,
        )
    )
    foundation_validated = decision_input_readiness["state"] == "FOUNDATION_VALIDATED"
    state = (
        "PUBLIC_CLOSURE_ACTIVE_FOUNDATION_ONLY" if active and foundation_validated else "BLOCKED"
    )
    if state not in POST_RUN_STATES:  # pragma: no cover - frozen invariant
        raise AssertionError("unexpected post-run state")

    return {
        "schema_id": PUBLIC_CYCLE_STATUS_SCHEMA_ID,
        "protocol": PROTOCOL,
        "state": state,
        "phase_capability": PHASE_CAPABILITY,
        "synthetic_only": is_synthetic,
        "read_only": True,
        "operational_authority": False,
        "foundation_validated": foundation_validated,
        "public_closure_active": active,
        "business_ready": False,
        "business_cycle_closed": False,
        "paper_state": "UNAVAILABLE",
        "learning_state": "UNAVAILABLE",
        "canonical_strategy_id": decision_input_readiness.get("canonical_strategy_id"),
        "decision_cutoff": decision_input_readiness.get("decision_cutoff"),
        "decision_cutoff_verified": decision_input_readiness.get("decision_cutoff_verified")
        is True,
        "mainline_derived_state": mainline_resolution.derived_state,
        "public_closure_refs": public_closure_refs,
        "blockers": _sorted_blockers(blockers),
        "authority": dict(AUTHORITY_FLAGS),
    }


__all__ = [
    "ALL_BLOCKER_CODES",
    "AUTHORITY_FLAGS",
    "DECISION_INPUT_READINESS_SCHEMA_ID",
    "GATE_NAMES",
    "GateEvidence",
    "HISTORICAL_HOLDINGS_LABEL",
    "MAINLINE_BLOCKED_CODES",
    "PHASE_CAPABILITY",
    "POST_RUN_STATES",
    "PRE_RUN_STATES",
    "PUBLIC_CYCLE_STATUS_SCHEMA_ID",
    "ReadinessBlocker",
    "build_decision_input_readiness",
    "build_public_cycle_status",
    "derive_decision_input_readiness",
]
