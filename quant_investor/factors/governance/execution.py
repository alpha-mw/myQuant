"""Compact, replay-bound prospective execution evidence.

The trusted store computes per-session turnover and returns only a canonical
session-summary hash plus aggregates.  Full sparse weights and matured labels
remain in the 360 capture/observation source closures and are replayed again by
the contextual validator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
import hashlib
from typing import Any, Final

from quant_investor.contracts import canonical_json_bytes, seal_artifact

from .bootstrap import PROSPECTIVE_LANE
from .common import (
    ANNUAL_OPEN_SESSIONS,
    COST_BPS,
    SIGNAL_OPEN_SESSIONS,
    artifact_ref,
    business_identity,
    canonical_timestamp,
    decimal_text,
    decimal_value,
    exact_payload,
    require_sha256,
    validate_artifact_ref,
)
from .errors import FactorGovernanceError
from .prospective import (
    SIGNAL_CAPTURE_KIND,
    validate_configuration_selection,
    validate_preregistration,
)

EXECUTION_EVIDENCE_KIND: Final = "factor.execution_turnover_evidence"

_MAX_EXECUTION_BYTES: Final = 2 * 1024 * 1024
_COST_RATE: Final = Decimal("0.00005")
_EVIDENCE_FIELDS: Final = {
    "execution_evidence_id",
    "preregistration_id",
    "selection_id",
    "lane",
    "signal_sessions_sha256",
    "signal_session_count",
    "signal_capture_refs",
    "observation_refs",
    "configuration_rows",
    "cost_contract",
    "execution_state",
    "blockers",
    "authority",
}
_CONFIGURATION_FIELDS: Final = {
    "configuration_id",
    "factor_id",
    "session_summary_sha256",
    "session_summary_count",
    "initial_entry_turnover",
    "rebalance_turnover",
    "terminal_exit_turnover",
    "total_turnover",
    "annualized_turnover",
    "total_estimated_cost",
    "gross_labeled_return_count",
    "gross_labeled_return_sum",
    "net_labeled_return_sum",
}
_COST_CONTRACT: Final = {
    "round_trip_cost_bps": decimal_text(COST_BPS),
    "absolute_weight_change_cost_rate": decimal_text(_COST_RATE),
    "annual_open_sessions": ANNUAL_OPEN_SESSIONS,
    "observed_sessions": SIGNAL_OPEN_SESSIONS,
    "annualization": "TOTAL_TURNOVER_X_252_DIV_360",
    "unlisted_universe_weight": "EXACT_ZERO",
}
_REF_FIELDS: Final = (
    "kind",
    "contract_sha256",
    "artifact_id",
    "semantic_sha256",
    "byte_sha256",
)


def _fail(detail: str, *, code: str = "FACTOR_VALIDATION_FAILED") -> FactorGovernanceError:
    return FactorGovernanceError(detail, code=code)


def _check_size(envelope: Mapping[str, Any]) -> None:
    if len(canonical_json_bytes(dict(envelope))) > _MAX_EXECUTION_BYTES:
        raise _fail(
            "execution evidence exceeds its canonical byte limit",
            code="ARTIFACT_SIZE_LIMIT_EXCEEDED",
        )


def _blockers(values: Any) -> list[str]:
    if type(values) is not list or len(values) != len(set(values)):
        raise _fail("execution blockers are not a unique list")
    rows: list[str] = []
    for value in values:
        if (
            type(value) is not str
            or not value
            or value != value.strip()
            or not value.replace("_", "").isalnum()
            or value != value.upper()
        ):
            raise _fail("execution blocker code is invalid")
        rows.append(value)
    if rows != sorted(rows):
        raise _fail("execution blockers are not canonical")
    return rows


def _selected_factor_ids(selection: Mapping[str, Any]) -> dict[str, str]:
    return {
        row["selected_configuration_id"]: row["selected_factor_id"]
        for row in selection["payload"]["selected_configurations"]
    }


def _decimal_field(value: Any, *, label: str, minimum: Decimal | None = None) -> Decimal:
    parsed = decimal_value(value, label=label, minimum=minimum)
    if value != decimal_text(parsed, label=label):
        raise _fail(f"{label} is not canonical 12-decimal text")
    return parsed


def _optional_decimal(value: Any, *, label: str) -> Decimal | None:
    if value is None:
        return None
    return _decimal_field(value, label=label)


def _configuration_row(value: Any, *, factor_id: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _CONFIGURATION_FIELDS:
        raise _fail("execution configuration fields are not exact")
    row = dict(value)
    if row["factor_id"] != factor_id:
        raise _fail("execution factor binding differs")
    require_sha256(row["session_summary_sha256"], label="session_summary_sha256")
    if row["session_summary_count"] != SIGNAL_OPEN_SESSIONS:
        raise _fail("execution session-summary count differs")
    initial = _decimal_field(
        row["initial_entry_turnover"], label="initial_entry_turnover", minimum=Decimal("0")
    )
    rebalance = _decimal_field(
        row["rebalance_turnover"], label="rebalance_turnover", minimum=Decimal("0")
    )
    terminal = _decimal_field(
        row["terminal_exit_turnover"], label="terminal_exit_turnover", minimum=Decimal("0")
    )
    total = _decimal_field(row["total_turnover"], label="total_turnover", minimum=Decimal("0"))
    _decimal_field(row["annualized_turnover"], label="annualized_turnover", minimum=Decimal("0"))
    cost = _decimal_field(
        row["total_estimated_cost"], label="total_estimated_cost", minimum=Decimal("0")
    )
    if total != initial + rebalance + terminal:
        raise _fail("execution turnover components do not sum")
    expected_annualized = total * Decimal(ANNUAL_OPEN_SESSIONS) / Decimal(SIGNAL_OPEN_SESSIONS)
    if row["annualized_turnover"] != decimal_text(expected_annualized):
        raise _fail("execution annualized turnover differs")
    if row["total_estimated_cost"] != decimal_text(total * _COST_RATE):
        raise _fail("execution cost differs")
    gross_count = row["gross_labeled_return_count"]
    if (
        type(gross_count) is not int
        or isinstance(gross_count, bool)
        or not 0 <= gross_count <= SIGNAL_OPEN_SESSIONS
    ):
        raise _fail("execution gross-return count is invalid")
    gross = _optional_decimal(row["gross_labeled_return_sum"], label="gross_labeled_return_sum")
    net = _optional_decimal(row["net_labeled_return_sum"], label="net_labeled_return_sum")
    if (gross is None) != (net is None) or (gross_count == 0) != (gross is None):
        raise _fail("execution gross/net return projection is incomplete")
    if gross is not None and row["net_labeled_return_sum"] != decimal_text(gross - cost):
        raise _fail("execution net return does not include exact cost")
    return row


def _reference_rows(values: Any, *, kind: str, label: str) -> list[dict[str, str]]:
    if type(values) is not list or len(values) != SIGNAL_OPEN_SESSIONS:
        raise _fail(f"{label} must contain exactly 360 refs")
    rows = [
        validate_artifact_ref(value, label=f"{label}[{index}]", expected_kind=kind)
        for index, value in enumerate(values)
    ]
    if len({tuple(row[field] for field in _REF_FIELDS) for row in rows}) != len(rows):
        raise _fail(f"{label} contain duplicate refs")
    return rows


def _execution_identity(payload: Mapping[str, Any]) -> str:
    body = {field: payload[field] for field in _EVIDENCE_FIELDS if field != "execution_evidence_id"}
    return business_identity("factor-execution-turnover-evidence", body)


def _validate_execution_state(payload: Mapping[str, Any]) -> None:
    blockers = _blockers(payload["blockers"])
    state = payload["execution_state"]
    if state == "COMPLETE":
        if blockers:
            raise _fail("complete execution may not carry blockers")
        if any(
            row["gross_labeled_return_count"] != SIGNAL_OPEN_SESSIONS
            or row["gross_labeled_return_sum"] is None
            or row["net_labeled_return_sum"] is None
            for row in payload["configuration_rows"]
        ):
            raise _fail("complete execution lacks all matured returns")
        return
    if state != "INCOMPLETE" or not blockers:
        raise _fail("execution state/blockers are inconsistent")


def _validate_artifact_refs(
    payload: Mapping[str, Any],
    *,
    signal_captures: Sequence[Mapping[str, Any] | bytes] | None,
    observations: Sequence[Mapping[str, Any] | bytes] | None,
) -> None:
    capture_refs = _reference_rows(
        payload["signal_capture_refs"], kind=SIGNAL_CAPTURE_KIND, label="signal_capture_refs"
    )
    observation_refs = _reference_rows(
        payload["observation_refs"],
        kind="factor.prospective_observation",
        label="observation_refs",
    )
    if signal_captures is not None:
        observed = [artifact_ref(value) for value in signal_captures]
        if observed != capture_refs:
            raise _fail("execution capture refs differ from supplied closure")
    if observations is not None:
        observed = [artifact_ref(value) for value in observations]
        if observed != observation_refs:
            raise _fail("execution observation refs differ from supplied closure")


def _build_execution_turnover_evidence(
    *,
    preregistration: Mapping[str, Any] | bytes,
    selection: Mapping[str, Any] | bytes,
    signal_captures: Sequence[Mapping[str, Any] | bytes],
    observations: Sequence[Mapping[str, Any] | bytes],
    configuration_rows: Sequence[Mapping[str, Any]],
    execution_state: str,
    blockers: Sequence[str],
    trusted_at: str,
) -> dict[str, Any]:
    """Seal trusted compact aggregates after Store-side raw replay."""

    prereg = validate_preregistration(preregistration)
    selected = validate_configuration_selection(selection, preregistration=prereg)
    sessions = prereg["payload"]["signal_sessions"]
    payload: dict[str, Any] = {
        "preregistration_id": prereg["payload"]["preregistration_id"],
        "selection_id": selected["payload"]["selection_id"],
        "lane": PROSPECTIVE_LANE,
        "signal_sessions_sha256": hashlib.sha256(canonical_json_bytes(sessions)).hexdigest(),
        "signal_session_count": SIGNAL_OPEN_SESSIONS,
        "signal_capture_refs": [artifact_ref(value) for value in signal_captures],
        "observation_refs": [artifact_ref(value) for value in observations],
        "configuration_rows": [dict(row) for row in configuration_rows],
        "cost_contract": dict(_COST_CONTRACT),
        "execution_state": execution_state,
        "blockers": sorted(blockers),
        "authority": "NON_AUTHORIZING",
    }
    payload["execution_evidence_id"] = _execution_identity(payload)
    artifact = seal_artifact(
        EXECUTION_EVIDENCE_KIND,
        payload,
        created_at=canonical_timestamp(trusted_at, label="trusted_at"),
    )
    validate_execution_turnover_evidence(
        artifact,
        preregistration=prereg,
        selection=selected,
        signal_captures=signal_captures,
        observations=observations,
    )
    return artifact


def validate_execution_turnover_evidence(
    document: Mapping[str, Any] | bytes,
    *,
    preregistration: Mapping[str, Any] | bytes,
    selection: Mapping[str, Any] | bytes,
    signal_captures: Sequence[Mapping[str, Any] | bytes] | None = None,
    observations: Sequence[Mapping[str, Any] | bytes] | None = None,
) -> dict[str, Any]:
    """Validate compact aggregates; contextual replay proves their raw inputs."""

    envelope, payload = exact_payload(
        document, kind=EXECUTION_EVIDENCE_KIND, fields=_EVIDENCE_FIELDS
    )
    prereg = validate_preregistration(preregistration)
    selected = validate_configuration_selection(selection, preregistration=prereg)
    sessions = prereg["payload"]["signal_sessions"]
    if (
        payload["preregistration_id"] != prereg["payload"]["preregistration_id"]
        or payload["selection_id"] != selected["payload"]["selection_id"]
        or payload["lane"] != PROSPECTIVE_LANE
        or payload["signal_session_count"] != SIGNAL_OPEN_SESSIONS
        or payload["signal_sessions_sha256"]
        != hashlib.sha256(canonical_json_bytes(sessions)).hexdigest()
        or payload["cost_contract"] != _COST_CONTRACT
        or payload["authority"] != "NON_AUTHORIZING"
    ):
        raise _fail("execution fixed policy binding differs")
    _validate_artifact_refs(
        payload,
        signal_captures=signal_captures,
        observations=observations,
    )
    selected_ids = _selected_factor_ids(selected)
    rows = payload["configuration_rows"]
    if type(rows) is not list or [row.get("configuration_id") for row in rows] != sorted(
        selected_ids
    ):
        raise _fail("execution configuration rows are not canonical")
    normalized = [
        _configuration_row(row, factor_id=selected_ids[row["configuration_id"]]) for row in rows
    ]
    if len(normalized) != len(selected_ids):
        raise _fail("execution configuration rows are incomplete")
    _validate_execution_state(payload)
    if payload["execution_evidence_id"] != _execution_identity(payload):
        raise _fail("execution business identity differs")
    _check_size(envelope)
    return envelope


__all__ = [
    "EXECUTION_EVIDENCE_KIND",
    "validate_execution_turnover_evidence",
]
