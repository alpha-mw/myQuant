"""Pure origin-regime binding inventory for V17 v5 Sprint 1B.

This module accepts only caller-supplied, already verified origin and regime
evidence.  It does not read files, scan directories, call providers, write V4,
or produce governance actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import re
from typing import Any, Final, Mapping, Sequence

from quant_investor.v17_v5_contract.canonical import (
    canonical_bytes,
    seal_semantic,
    validate_semantic_sha,
)
from quant_investor.v17_v5_contract.identities import (
    IdentityContractError,
    require_identifier,
    require_relative_path,
    require_sha256,
)
from quant_investor.v17_v5_contract.validators import (
    FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_ID,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION,
    NO_AUTHORITY,
)

PROTOCOL_VERSION: Final = "myquant.v17.v5"
FACTOR_REGIME_ORIGIN_INVENTORY_VERSION: Final = "myquant.v17.v5.factor-regime-origin-inventory.v1"
HORIZON_SESSIONS: Final = 20
OUTPUT_SCALE: Final = Decimal("0.000000000001")
_SESSION_RE: Final = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$", re.ASCII)
_UTC_RE: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)
_DECIMAL_RE: Final = re.compile(
    r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$",
    re.ASCII,
)
_FORBIDDEN_KEY_PARTS: Final = (
    "factor_weight",
    "recommended_weight",
    "target_weight",
    "production_weight",
    "portfolio_weight",
    "lifecycle_action",
    "validity",
)
_FORBIDDEN_KEYS: Final = {
    "tier",
    "promotion",
    "production_apply",
    "buy_signal",
    "sell_signal",
}


class FactorRegimeOriginInventoryError(ValueError):
    """Raised when a regime-origin inventory input is malformed."""

    exit_code = 2


@dataclass(frozen=True)
class ContentArtifactRef:
    """A pathless V5 content reference or explicit immutable V4 reference."""

    artifact_id: str
    version: str
    byte_sha256: str
    semantic_sha256: str
    cutoff: str
    strategy_id: str
    relative_path: str | None = None


@dataclass(frozen=True)
class RegimeEvidenceSnapshot:
    """Normalized read-only regime evidence at the origin decision point."""

    regime_artifact_ref: ContentArtifactRef
    strategy_id: str
    cutoff: str
    available_at: str
    published_at: str
    regime_state: str
    source_version: str
    decision_session: str | None = None
    effective_session: str | None = None
    state_probabilities: Mapping[str, str] | None = None


@dataclass(frozen=True)
class FactorRegimeOriginInput:
    """One already verified factor origin bound to one origin-time regime."""

    origin_id: str
    strategy_id: str
    factor_name: str
    factor_implementation_sha256: str
    decision_session: str
    origin_cutoff: str
    label_end_session: str
    label_horizon_sessions: int
    label_origin_session: str
    rank_ic: str | None
    eligible_symbol_count: int
    comparable_symbol_count: int
    coverage: str
    factor_observation_ref: ContentArtifactRef
    matured_label_ref: ContentArtifactRef
    factor_evidence_ref: ContentArtifactRef
    observation_run_ref: ContentArtifactRef
    request_ref: ContentArtifactRef
    source_locator_ref: ContentArtifactRef
    regime_evidence: RegimeEvidenceSnapshot


def _fail(message: str) -> None:
    raise FactorRegimeOriginInventoryError(message)


def _timestamp(value: Any, *, label: str) -> datetime:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        _fail(f"{label} must be a second-precision UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise FactorRegimeOriginInventoryError(f"{label} is not a valid UTC timestamp") from exc


def _session(value: Any, *, label: str) -> str:
    if type(value) is not str or _SESSION_RE.fullmatch(value) is None:
        _fail(f"{label} must be an ISO session date")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise FactorRegimeOriginInventoryError(f"{label} is not a valid session date") from exc
    return value


def _decimal(value: Any, *, label: str, minimum: Decimal, maximum: Decimal) -> str:
    if type(value) is not str or _DECIMAL_RE.fullmatch(value) is None:
        _fail(f"{label} must be a canonical finite decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise FactorRegimeOriginInventoryError(f"{label} is not a finite decimal") from exc
    if not parsed.is_finite() or parsed < minimum or parsed > maximum:
        _fail(f"{label} is out of range")
    if parsed.is_zero() and value.startswith("-"):
        _fail(f"{label} is not canonical")
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        rendered = parsed.quantize(OUTPUT_SCALE, rounding=ROUND_HALF_EVEN)
    if rendered.is_zero():
        rendered = abs(rendered)
    return format(rendered, ".12f")


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or any(ord(character) < 32 for character in value):
        _fail(f"{label} must be a nonempty control-free string")
    return value


def _artifact_ref(ref: ContentArtifactRef, *, label: str, v4: bool) -> dict[str, Any]:
    if not isinstance(ref, ContentArtifactRef):
        _fail(f"{label} must be ContentArtifactRef")
    try:
        artifact_id = require_identifier(ref.artifact_id, label=f"{label}.artifact_id")
        version = require_identifier(ref.version, label=f"{label}.version")
        byte_sha = require_sha256(ref.byte_sha256, label=f"{label}.byte_sha256")
        semantic_sha = require_sha256(ref.semantic_sha256, label=f"{label}.semantic_sha256")
        strategy_id = require_identifier(ref.strategy_id, label=f"{label}.strategy_id")
    except IdentityContractError as exc:
        raise FactorRegimeOriginInventoryError(str(exc)) from exc
    cutoff = _timestamp(ref.cutoff, label=f"{label}.cutoff").strftime("%Y-%m-%dT%H:%M:%SZ")
    document: dict[str, Any] = {
        "artifact_id": artifact_id,
        "byte_sha256": byte_sha,
        "cutoff": cutoff,
        "semantic_sha256": semantic_sha,
        "strategy_id": strategy_id,
        "version": version,
    }
    if v4:
        if ref.relative_path is None:
            _fail(f"{label}.relative_path is required for V4 refs")
        try:
            document["relative_path"] = require_relative_path(
                ref.relative_path,
                label=f"{label}.relative_path",
            )
        except IdentityContractError as exc:
            raise FactorRegimeOriginInventoryError(str(exc)) from exc
    elif ref.relative_path is not None:
        _fail(f"{label}.relative_path is not allowed for V5 content refs")
    return document


def _policy_ref(policy_ref: Mapping[str, Any]) -> dict[str, Any]:
    if type(policy_ref) is not dict or set(policy_ref) != {
        "artifact_id",
        "byte_sha256",
        "relative_path",
        "semantic_sha256",
        "version",
    }:
        _fail("policy_ref must be an object")
    try:
        document = {
            "artifact_id": require_identifier(
                policy_ref["artifact_id"], label="policy artifact_id"
            ),
            "byte_sha256": require_sha256(policy_ref["byte_sha256"], label="policy byte_sha256"),
            "semantic_sha256": require_sha256(
                policy_ref["semantic_sha256"],
                label="policy semantic_sha256",
            ),
            "version": require_identifier(policy_ref["version"], label="policy version"),
        }
    except (KeyError, IdentityContractError) as exc:
        raise FactorRegimeOriginInventoryError("policy_ref is invalid") from exc
    try:
        document["relative_path"] = require_relative_path(
            policy_ref["relative_path"],
            label="policy relative_path",
        )
    except IdentityContractError as exc:
        raise FactorRegimeOriginInventoryError(str(exc)) from exc
    if document != {
        "artifact_id": FACTOR_REGIME_DIAGNOSTIC_POLICY_ID,
        "byte_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256,
        "relative_path": FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH,
        "semantic_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256,
        "version": FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION,
    }:
        _fail("policy_ref does not bind the sealed Sprint 1B policy")
    return document


def _probabilities(values: Mapping[str, str] | None) -> list[dict[str, str]] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping) or not values:
        _fail("state_probabilities must be a nonempty mapping")
    result: dict[str, str] = {}
    total = Decimal(0)
    for state, probability in values.items():
        name = _text(state, label="state probability regime_state")
        value = _decimal(
            probability,
            label=f"state_probabilities[{name}]",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        result[name] = value
        total += Decimal(value)
    if total > Decimal("1.000000000001"):
        _fail("state probabilities sum above one")
    return [
        {
            "probability": result[key],
            "regime_state": key,
        }
        for key in sorted(result)
    ]


def _forbidden_key_scan(value: Any, *, path: str = "$") -> None:
    if path == "$.authority":
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is str:
                lowered = key.lower()
                if lowered in _FORBIDDEN_KEYS or any(
                    part in lowered for part in _FORBIDDEN_KEY_PARTS
                ):
                    _fail(f"forbidden governance or weight field at {path}.{key}")
            _forbidden_key_scan(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden_key_scan(child, path=f"{path}[{index}]")


def _origin_row(
    origin: FactorRegimeOriginInput,
    *,
    strategy_id: str,
    factor_name: str,
    factor_implementation_sha256: str,
    horizon_sessions: int,
) -> dict[str, Any]:
    if not isinstance(origin, FactorRegimeOriginInput):
        _fail("origin_rows must contain FactorRegimeOriginInput values")
    try:
        origin_id = require_identifier(origin.origin_id, label="origin_id")
        row_strategy = require_identifier(origin.strategy_id, label="origin.strategy_id")
        row_factor = require_identifier(origin.factor_name, label="origin.factor_name")
        row_implementation = require_sha256(
            origin.factor_implementation_sha256,
            label="origin.factor_implementation_sha256",
        )
    except IdentityContractError as exc:
        raise FactorRegimeOriginInventoryError(str(exc)) from exc
    if (
        row_strategy != strategy_id
        or row_factor != factor_name
        or row_implementation != factor_implementation_sha256
    ):
        _fail("origin row stratum mismatch")
    if (
        origin.label_horizon_sessions != horizon_sessions
        or origin.label_horizon_sessions != HORIZON_SESSIONS
    ):
        _fail("label horizon must equal 20 sessions")
    decision_session = _session(origin.decision_session, label=f"{origin_id}.decision_session")
    label_origin_session = _session(
        origin.label_origin_session,
        label=f"{origin_id}.label_origin_session",
    )
    label_end_session = _session(origin.label_end_session, label=f"{origin_id}.label_end_session")
    if label_origin_session != decision_session:
        _fail("label origin must equal factor observation origin")
    origin_cutoff = _timestamp(origin.origin_cutoff, label=f"{origin_id}.origin_cutoff")
    if type(origin.eligible_symbol_count) is not int or origin.eligible_symbol_count < 0:
        _fail("eligible_symbol_count is invalid")
    if (
        type(origin.comparable_symbol_count) is not int
        or origin.comparable_symbol_count < 0
        or origin.comparable_symbol_count > origin.eligible_symbol_count
    ):
        _fail("comparable_symbol_count is invalid")
    coverage = _decimal(
        origin.coverage,
        label=f"{origin_id}.coverage",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    if origin.eligible_symbol_count == 0:
        if origin.comparable_symbol_count != 0 or Decimal(coverage) != 0:
            _fail("zero eligible symbols require zero coverage")
    elif (Decimal(origin.comparable_symbol_count) / Decimal(origin.eligible_symbol_count)).quantize(
        OUTPUT_SCALE, rounding=ROUND_HALF_EVEN
    ) != Decimal(coverage):
        _fail("coverage must equal comparable_symbol_count / eligible_symbol_count exactly")
    rank_ic = None
    if origin.rank_ic is not None:
        rank_ic = _decimal(
            origin.rank_ic,
            label=f"{origin_id}.rank_ic",
            minimum=Decimal("-1"),
            maximum=Decimal("1"),
        )
    factor_observation_ref = _artifact_ref(
        origin.factor_observation_ref,
        label=f"{origin_id}.factor_observation_ref",
        v4=True,
    )
    label_ref = _artifact_ref(
        origin.matured_label_ref,
        label=f"{origin_id}.matured_label_ref",
        v4=True,
    )
    factor_evidence_ref = _artifact_ref(
        origin.factor_evidence_ref,
        label=f"{origin_id}.factor_evidence_ref",
        v4=True,
    )
    observation_run_ref = _artifact_ref(
        origin.observation_run_ref,
        label=f"{origin_id}.observation_run_ref",
        v4=True,
    )
    request_ref = _artifact_ref(
        origin.request_ref,
        label=f"{origin_id}.request_ref",
        v4=True,
    )
    source_locator_ref = _artifact_ref(
        origin.source_locator_ref,
        label=f"{origin_id}.source_locator_ref",
        v4=True,
    )
    for label, ref in (
        ("factor_observation_ref", factor_observation_ref),
        ("matured_label_ref", label_ref),
        ("factor_evidence_ref", factor_evidence_ref),
        ("observation_run_ref", observation_run_ref),
        ("request_ref", request_ref),
        ("source_locator_ref", source_locator_ref),
    ):
        if ref["strategy_id"] != strategy_id:
            _fail(f"{label} strategy_id mismatch")
    regime = origin.regime_evidence
    if not isinstance(regime, RegimeEvidenceSnapshot):
        _fail("regime_evidence must be RegimeEvidenceSnapshot")
    regime_ref = _artifact_ref(
        regime.regime_artifact_ref,
        label=f"{origin_id}.regime_artifact_ref",
        v4=True,
    )
    try:
        regime_strategy = require_identifier(regime.strategy_id, label="regime.strategy_id")
        regime_source_version = require_identifier(
            regime.source_version,
            label="regime.source_version",
        )
    except IdentityContractError as exc:
        raise FactorRegimeOriginInventoryError(str(exc)) from exc
    if regime_strategy != strategy_id or regime_ref["strategy_id"] != strategy_id:
        _fail("strategy_id mismatch")
    if regime_source_version != regime_ref["version"]:
        _fail("regime source version does not match artifact ref")
    regime_cutoff = _timestamp(regime.cutoff, label=f"{origin_id}.regime.cutoff")
    if regime_ref["cutoff"] != regime.cutoff:
        _fail("regime cutoff does not match artifact ref")
    regime_available = _timestamp(regime.available_at, label=f"{origin_id}.regime.available_at")
    regime_published = _timestamp(regime.published_at, label=f"{origin_id}.regime.published_at")
    if regime_available > origin_cutoff:
        _fail("regime available_at is later than origin cutoff")
    if regime_published > origin_cutoff:
        _fail("regime published_at is later than origin cutoff")
    if regime_cutoff > origin_cutoff:
        _fail("regime cutoff is later than origin cutoff")
    regime_decision_session = None
    if regime.decision_session is not None:
        regime_decision_session = _session(
            regime.decision_session,
            label=f"{origin_id}.regime.decision_session",
        )
        if regime_decision_session > decision_session:
            _fail("regime decision_session is later than factor origin")
    regime_effective_session = None
    if regime.effective_session is not None:
        regime_effective_session = _session(
            regime.effective_session,
            label=f"{origin_id}.regime.effective_session",
        )
        if regime_effective_session > decision_session:
            _fail("regime effective_session is later than factor origin")
    regime_state = _text(regime.regime_state, label=f"{origin_id}.regime_state")
    state_probabilities = _probabilities(regime.state_probabilities)
    if state_probabilities is not None and regime_state not in {
        row["regime_state"] for row in state_probabilities
    }:
        _fail("posterior does not contain the sealed hard regime state")
    row = {
        "comparable_symbol_count": origin.comparable_symbol_count,
        "coverage": coverage,
        "decision_session": decision_session,
        "eligible_symbol_count": origin.eligible_symbol_count,
        "factor_evidence_ref": factor_evidence_ref,
        "factor_name": factor_name,
        "factor_observation_ref": factor_observation_ref,
        "label_end_session": label_end_session,
        "label_horizon_sessions": HORIZON_SESSIONS,
        "matured_label_ref": label_ref,
        "observation_run_ref": observation_run_ref,
        "origin_cutoff": origin.origin_cutoff,
        "origin_id": origin_id,
        "rank_ic": rank_ic,
        "request_ref": request_ref,
        "regime_available_at": regime.available_at,
        "regime_decision_session": regime_decision_session,
        "regime_effective_session": regime_effective_session,
        "regime_evidence_ref": regime_ref,
        "regime_published_at": regime.published_at,
        "regime_source_version": regime_source_version,
        "regime_state": regime_state,
        "source_locator_ref": source_locator_ref,
        "state_probabilities": state_probabilities,
    }
    row_identity = hashlib.sha256(canonical_bytes(row)).hexdigest()
    row["row_identity_sha256"] = row_identity
    return row


def _validate_inventory(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = validate_semantic_sha(document)
        require_identifier(payload["inventory_id"], label="inventory_id")
        require_identifier(payload["strategy_id"], label="strategy_id")
        require_identifier(payload["factor_name"], label="factor_name")
        require_sha256(
            payload["factor_implementation_sha256"],
            label="factor_implementation_sha256",
        )
        _timestamp(payload["cutoff"], label="cutoff")
        _timestamp(payload["created_at"], label="created_at")
    except Exception as exc:
        raise FactorRegimeOriginInventoryError("factor regime origin inventory is invalid") from exc
    if payload.get("version") != FACTOR_REGIME_ORIGIN_INVENTORY_VERSION:
        _fail("factor regime origin inventory version mismatch")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        _fail("factor regime origin inventory protocol mismatch")
    if payload.get("authority") != NO_AUTHORITY:
        _fail("factor regime origin inventory grants authority")
    _forbidden_key_scan(payload)
    rows = payload.get("origin_rows")
    if not isinstance(rows, list):
        _fail("origin_rows must be a list")
    if payload.get("origin_count") != len(rows):
        _fail("origin_count mismatch")
    expected_order = sorted(
        rows,
        key=lambda row: (
            row["decision_session"],
            row["factor_name"],
            row["regime_state"],
            row["factor_observation_ref"]["artifact_id"],
            row["matured_label_ref"]["artifact_id"],
            row["regime_evidence_ref"]["artifact_id"],
        ),
    )
    if rows != expected_order:
        _fail("origin rows are not canonical")
    seen_origins: set[tuple[str, str]] = set()
    seen_origin_ids: set[str] = set()
    regime_counts: dict[str, int] = {}
    for row in rows:
        origin_key = (row["decision_session"], row["factor_name"])
        if origin_key in seen_origins or row["origin_id"] in seen_origin_ids:
            _fail("duplicate origin")
        seen_origins.add(origin_key)
        seen_origin_ids.add(row["origin_id"])
        expected_row_identity = dict(row)
        observed = expected_row_identity.pop("row_identity_sha256")
        if hashlib.sha256(canonical_bytes(expected_row_identity)).hexdigest() != observed:
            _fail("origin row identity mismatch")
        regime_counts[row["regime_state"]] = regime_counts.get(row["regime_state"], 0) + 1
    if payload.get("regime_counts") != [
        {
            "origin_count": regime_counts[key],
            "regime_state": key,
        }
        for key in sorted(regime_counts)
    ]:
        _fail("regime_counts mismatch")
    identity_material = dict(payload)
    identity_material.pop("inventory_id")
    identity_material.pop("semantic_sha256")
    identity = hashlib.sha256(canonical_bytes(identity_material)).hexdigest()
    if payload["inventory_id"] != f"factor-regime-origin-inventory-{identity[:32]}":
        _fail("factor regime origin inventory identity mismatch")
    return payload


def build_factor_regime_origin_inventory(
    *,
    strategy_id: str,
    factor_name: str,
    factor_implementation_sha256: str,
    policy_ref: Mapping[str, Any],
    cutoff: str,
    created_at: str,
    origin_rows: Sequence[FactorRegimeOriginInput],
    horizon_sessions: int = HORIZON_SESSIONS,
) -> dict[str, Any]:
    """Build a deterministic in-memory origin-regime inventory artifact."""

    try:
        subject_strategy = require_identifier(strategy_id, label="strategy_id")
        subject_factor = require_identifier(factor_name, label="factor_name")
        implementation_sha = require_sha256(
            factor_implementation_sha256,
            label="factor_implementation_sha256",
        )
    except IdentityContractError as exc:
        raise FactorRegimeOriginInventoryError(str(exc)) from exc
    if type(horizon_sessions) is not int or horizon_sessions != HORIZON_SESSIONS:
        _fail("horizon_sessions must equal 20")
    cutoff_dt = _timestamp(cutoff, label="cutoff")
    created_dt = _timestamp(created_at, label="created_at")
    if created_dt < cutoff_dt:
        _fail("created_at must not precede cutoff")
    if isinstance(origin_rows, (str, bytes)) or not isinstance(origin_rows, Sequence):
        _fail("origin_rows must be a sequence")
    rows = [
        _origin_row(
            origin,
            strategy_id=subject_strategy,
            factor_name=subject_factor,
            factor_implementation_sha256=implementation_sha,
            horizon_sessions=horizon_sessions,
        )
        for origin in origin_rows
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            row["decision_session"],
            row["factor_name"],
            row["regime_state"],
            row["factor_observation_ref"]["artifact_id"],
            row["matured_label_ref"]["artifact_id"],
            row["regime_evidence_ref"]["artifact_id"],
        ),
    )
    regime_counts: dict[str, int] = {}
    for row in rows:
        regime_counts[row["regime_state"]] = regime_counts.get(row["regime_state"], 0) + 1
    document = {
        "authority": dict(NO_AUTHORITY),
        "created_at": created_at,
        "cutoff": cutoff,
        "factor_implementation_sha256": implementation_sha,
        "factor_name": subject_factor,
        "horizon_sessions": HORIZON_SESSIONS,
        "inventory_id": "",
        "origin_count": len(rows),
        "origin_rows": rows,
        "policy_ref": _policy_ref(policy_ref),
        "protocol_version": PROTOCOL_VERSION,
        "regime_counts": [
            {
                "origin_count": regime_counts[key],
                "regime_state": key,
            }
            for key in sorted(regime_counts)
        ],
        "strategy_id": subject_strategy,
        "version": FACTOR_REGIME_ORIGIN_INVENTORY_VERSION,
    }
    identity_material = dict(document)
    identity_material.pop("inventory_id")
    identity = hashlib.sha256(canonical_bytes(identity_material)).hexdigest()
    document["inventory_id"] = f"factor-regime-origin-inventory-{identity[:32]}"
    return _validate_inventory(seal_semantic(document))


def validate_factor_regime_origin_inventory_replay(
    artifact: Mapping[str, Any],
    *,
    strategy_id: str,
    factor_name: str,
    factor_implementation_sha256: str,
    policy_ref: Mapping[str, Any],
    cutoff: str,
    created_at: str,
    origin_rows: Sequence[FactorRegimeOriginInput],
) -> dict[str, Any]:
    """Rebuild and compare an inventory without reading or writing files."""

    validated = _validate_inventory(artifact)
    rebuilt = build_factor_regime_origin_inventory(
        strategy_id=strategy_id,
        factor_name=factor_name,
        factor_implementation_sha256=factor_implementation_sha256,
        policy_ref=policy_ref,
        cutoff=cutoff,
        created_at=created_at,
        origin_rows=origin_rows,
    )
    if canonical_bytes(validated) != canonical_bytes(rebuilt):
        _fail("factor regime origin inventory replay mismatch")
    return validated


def validate_factor_regime_origin_inventory(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a sealed factor-regime origin inventory artifact."""

    return _validate_inventory(artifact)


__all__ = [
    "ContentArtifactRef",
    "FACTOR_REGIME_ORIGIN_INVENTORY_VERSION",
    "FactorRegimeOriginInput",
    "FactorRegimeOriginInventoryError",
    "RegimeEvidenceSnapshot",
    "build_factor_regime_origin_inventory",
    "validate_factor_regime_origin_inventory",
    "validate_factor_regime_origin_inventory_replay",
]
