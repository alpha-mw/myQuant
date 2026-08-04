"""Pure, descriptive Factor lifecycle diagnostics for V17 v5 Sprint 1A."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import re
from typing import Any, Final, Mapping, Sequence

from quant_investor.v17_v5_contract.canonical import (
    CanonicalContractError,
    canonical_bytes,
    seal_semantic,
    validate_semantic_sha,
)
from quant_investor.v17_v5_contract.identities import (
    IdentityContractError,
    require_identifier,
    require_sha256,
)
from quant_investor.v17_v5_contract.schema_validation import validate_artifact
from quant_investor.v17_v5_contract.validators import NO_AUTHORITY

PROTOCOL_VERSION: Final = "myquant.v17.v5"
FACTOR_LIFECYCLE_DIAGNOSTIC_VERSION: Final = "myquant.v17.v5.factor-lifecycle-diagnostic.v1"
FACTOR_DIAGNOSTIC_VERSION: Final = "myquant.v17.v5.factor-diagnostic.v1"

_UTC_RE: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
    re.ASCII,
)


class FactorLifecycleDiagnosticError(ValueError):
    """Raised when caller-supplied lifecycle input is malformed."""

    exit_code = 2


class FactorLifecycleDiagnosticStatus(str, Enum):
    """The only Sprint 1A lifecycle states."""

    UNOBSERVED = "UNOBSERVED"
    ACCUMULATING = "ACCUMULATING"
    UNAVAILABLE = "UNAVAILABLE"


def _fail(message: str) -> None:
    raise FactorLifecycleDiagnosticError(message)


def _canonical_timestamp(value: Any, *, label: str) -> datetime:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        _fail(f"{label} must be a second-precision UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise FactorLifecycleDiagnosticError(f"{label} is not a valid UTC timestamp") from exc


def _validate_blockers(values: Sequence[str], *, label: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        _fail(f"{label} must be a nonempty sequence")
    blockers: list[str] = []
    for value in values:
        try:
            blockers.append(require_identifier(value, label=label))
        except IdentityContractError as exc:
            raise FactorLifecycleDiagnosticError(str(exc)) from exc
    return sorted(set(blockers))


def _validate_lifecycle_diagnostic(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = validate_semantic_sha(document)
        require_identifier(payload["lifecycle_diagnostic_id"], label="lifecycle_diagnostic_id")
        require_identifier(payload["factor_name"], label="factor_name")
        _canonical_timestamp(payload["evaluation_cutoff"], label="evaluation_cutoff")
    except (
        CanonicalContractError,
        IdentityContractError,
        KeyError,
        TypeError,
    ) as exc:
        raise FactorLifecycleDiagnosticError("factor lifecycle diagnostic is invalid") from exc
    if payload.get("version") != FACTOR_LIFECYCLE_DIAGNOSTIC_VERSION:
        _fail("factor lifecycle diagnostic version mismatch")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        _fail("factor lifecycle diagnostic protocol mismatch")
    if payload.get("authority") != NO_AUTHORITY:
        _fail("factor lifecycle diagnostic grants authority")
    blockers = payload.get("blockers")
    if not isinstance(blockers, list) or blockers != sorted(set(blockers)):
        _fail("factor lifecycle blockers are noncanonical")
    for blocker in blockers:
        try:
            require_identifier(blocker, label="factor lifecycle blocker")
        except IdentityContractError as exc:
            raise FactorLifecycleDiagnosticError(str(exc)) from exc
    input_shas = payload.get("input_factor_diagnostic_semantic_sha256s")
    if not isinstance(input_shas, list) or input_shas != sorted(set(input_shas)):
        _fail("input factor diagnostic semantic SHA list is noncanonical")
    for value in input_shas:
        try:
            require_sha256(value, label="input factor diagnostic semantic SHA-256")
        except IdentityContractError as exc:
            raise FactorLifecycleDiagnosticError(str(exc)) from exc
    for field in (
        "effectiveness_claimed",
        "factor_tier_change_eligible",
        "factor_weight_change_eligible",
        "promotion_eligible",
    ):
        if payload.get(field) is not False:
            _fail(f"{field} must remain false")
    if (
        payload.get("lifecycle_conclusion") is not None
        or payload.get("lifecycle_action") is not None
    ):
        _fail("factor lifecycle diagnostic cannot carry lifecycle conclusions")
    status = payload.get("status")
    stratum = payload.get("stratum")
    stratum_sha = payload.get("stratum_sha256")
    unique_origin_count = payload.get("unique_origin_count")
    first_session = payload.get("first_decision_session")
    last_session = payload.get("last_decision_session")
    if status == FactorLifecycleDiagnosticStatus.UNAVAILABLE.value:
        if (
            stratum is not None
            or stratum_sha is not None
            or unique_origin_count != 0
            or first_session is not None
            or last_session is not None
            or payload.get("descriptive_coverage_minimum_met") is not False
            or not blockers
            or "lifecycle_inputs_unavailable" not in blockers
        ):
            _fail("UNAVAILABLE factor lifecycle diagnostic is inconsistent")
    elif status in {
        FactorLifecycleDiagnosticStatus.UNOBSERVED.value,
        FactorLifecycleDiagnosticStatus.ACCUMULATING.value,
    }:
        if type(stratum) is not dict or type(stratum_sha) is not str:
            _fail("observed factor lifecycle diagnostic has no exact stratum")
        try:
            require_sha256(stratum_sha, label="stratum_sha256")
        except IdentityContractError as exc:
            raise FactorLifecycleDiagnosticError(str(exc)) from exc
        if hashlib.sha256(canonical_bytes(stratum)).hexdigest() != stratum_sha:
            _fail("factor lifecycle stratum identity mismatch")
        if stratum.get("factor_name") != payload["factor_name"]:
            _fail("factor lifecycle factor name mismatch")
        if type(unique_origin_count) is not int or unique_origin_count < 0:
            _fail("unique_origin_count is invalid")
        if status == FactorLifecycleDiagnosticStatus.UNOBSERVED.value:
            if (
                unique_origin_count != 0
                or first_session is not None
                or last_session is not None
                or payload.get("descriptive_coverage_minimum_met") is not False
                or "lifecycle_no_observed_origins" not in blockers
            ):
                _fail("UNOBSERVED factor lifecycle diagnostic is inconsistent")
        else:
            if (
                unique_origin_count <= 0
                or type(first_session) is not str
                or type(last_session) is not str
                or "lifecycle_diagnostic_only" not in blockers
            ):
                _fail("ACCUMULATING factor lifecycle diagnostic is inconsistent")
            if first_session > last_session:
                _fail("factor lifecycle decision-session bounds are invalid")
    else:
        _fail("unknown factor lifecycle diagnostic status")
    identity_material = dict(payload)
    identity_material.pop("lifecycle_diagnostic_id")
    identity_material.pop("semantic_sha256")
    identity = hashlib.sha256(canonical_bytes(identity_material)).hexdigest()
    if payload["lifecycle_diagnostic_id"] != f"factor-lifecycle-diagnostic-{identity[:32]}":
        _fail("factor lifecycle diagnostic identity mismatch")
    return payload


def _seal_document(document: dict[str, Any]) -> dict[str, Any]:
    identity_material = dict(document)
    identity_material.pop("lifecycle_diagnostic_id", None)
    identity = hashlib.sha256(canonical_bytes(identity_material)).hexdigest()
    document["lifecycle_diagnostic_id"] = f"factor-lifecycle-diagnostic-{identity[:32]}"
    return _validate_lifecycle_diagnostic(seal_semantic(document))


def _validated_factor_diagnostics(
    factor_diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if (
        isinstance(factor_diagnostics, (str, bytes))
        or not isinstance(factor_diagnostics, Sequence)
        or not factor_diagnostics
    ):
        _fail("factor_diagnostics must be a nonempty sequence")
    result: list[dict[str, Any]] = []
    for index, artifact in enumerate(factor_diagnostics):
        try:
            validated = validate_artifact(artifact)
        except Exception as exc:
            raise FactorLifecycleDiagnosticError(
                f"factor_diagnostics[{index}] is not a valid factor diagnostic"
            ) from exc
        if validated.get("version") != FACTOR_DIAGNOSTIC_VERSION:
            _fail(f"factor_diagnostics[{index}] is not a factor diagnostic")
        result.append(validated)
    return result


def _factor_name(
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    expected_factor_name: str | None,
) -> str:
    names = {diagnostic["subject_factor_name"] for diagnostic in diagnostics}
    if len(names) != 1:
        _fail("factor diagnostics must describe exactly one factor")
    factor_name = next(iter(names))
    try:
        subject = require_identifier(factor_name, label="factor_name")
        if expected_factor_name is not None:
            expected = require_identifier(expected_factor_name, label="expected_factor_name")
        else:
            expected = None
    except IdentityContractError as exc:
        raise FactorLifecycleDiagnosticError(str(exc)) from exc
    if expected is not None and expected != subject:
        _fail("expected_factor_name does not match factor diagnostic subject")
    return subject


def _origin_summary(diagnostics: Sequence[Mapping[str, Any]]) -> tuple[int, str | None, str | None]:
    by_origin_id: dict[str, Mapping[str, Any]] = {}
    by_decision_session: dict[str, str] = {}
    for diagnostic in diagnostics:
        for row in diagnostic["origin_diagnostics"]:
            origin_id = row["origin_id"]
            decision_session = row["decision_session"]
            existing = by_origin_id.get(origin_id)
            if existing is not None:
                if dict(existing) != dict(row):
                    _fail(f"conflicting duplicate origin_id: {origin_id}")
                continue
            previous_origin = by_decision_session.get(decision_session)
            if previous_origin is not None:
                _fail(f"conflicting origin identity for decision session {decision_session}")
            by_origin_id[origin_id] = row
            by_decision_session[decision_session] = origin_id
    if not by_decision_session:
        return 0, None, None
    ordered_sessions = sorted(by_decision_session)
    return len(by_origin_id), ordered_sessions[0], ordered_sessions[-1]


def build_factor_lifecycle_diagnostic(
    *,
    factor_diagnostics: Sequence[Mapping[str, Any]],
    evaluation_cutoff: str,
    expected_factor_name: str | None = None,
) -> dict[str, Any]:
    """Build an in-memory lifecycle diagnostic from sealed factor diagnostics."""

    diagnostics = _validated_factor_diagnostics(factor_diagnostics)
    cutoff = _canonical_timestamp(evaluation_cutoff, label="evaluation_cutoff")
    factor_name = _factor_name(diagnostics, expected_factor_name=expected_factor_name)
    for diagnostic in diagnostics:
        if (
            _canonical_timestamp(
                diagnostic["evaluation_cutoff"],
                label="input evaluation_cutoff",
            )
            > cutoff
        ):
            _fail("evaluation_cutoff precedes an input factor diagnostic cutoff")
    statuses = {diagnostic["status"] for diagnostic in diagnostics}
    has_unavailable = FactorLifecycleDiagnosticStatus.UNAVAILABLE.value in statuses
    has_observed = bool(
        statuses.intersection(
            {
                FactorLifecycleDiagnosticStatus.UNOBSERVED.value,
                FactorLifecycleDiagnosticStatus.ACCUMULATING.value,
            }
        )
    )
    if has_unavailable and has_observed:
        _fail("cannot mix unavailable and observed factor diagnostics")
    stratum: dict[str, Any] | None = None
    stratum_sha: str | None = None
    if has_observed:
        for diagnostic in diagnostics:
            if stratum is None:
                stratum = dict(diagnostic["stratum"])
                stratum_sha = diagnostic["stratum_sha256"]
            elif stratum != diagnostic["stratum"] or stratum_sha != diagnostic["stratum_sha256"]:
                _fail("observed factor diagnostics must share the exact stratum")
    unique_origin_count, first_session, last_session = _origin_summary(diagnostics)
    if statuses == {FactorLifecycleDiagnosticStatus.UNAVAILABLE.value}:
        status = FactorLifecycleDiagnosticStatus.UNAVAILABLE.value
        blockers = sorted(
            set(blocker for diagnostic in diagnostics for blocker in diagnostic["blockers"])
            | {"lifecycle_inputs_unavailable"}
        )
        coverage_met = False
    elif statuses == {FactorLifecycleDiagnosticStatus.UNOBSERVED.value}:
        status = FactorLifecycleDiagnosticStatus.UNOBSERVED.value
        blockers = sorted(
            set(blocker for diagnostic in diagnostics for blocker in diagnostic["blockers"])
            | {"lifecycle_no_observed_origins"}
        )
        coverage_met = False
    else:
        status = FactorLifecycleDiagnosticStatus.ACCUMULATING.value
        blockers = sorted(
            set(blocker for diagnostic in diagnostics for blocker in diagnostic["blockers"])
            | {"lifecycle_diagnostic_only"}
        )
        coverage_met = any(
            diagnostic["descriptive_coverage_minimum_met"] for diagnostic in diagnostics
        )
    input_shas = sorted({diagnostic["semantic_sha256"] for diagnostic in diagnostics})
    return _seal_document(
        {
            "authority": dict(NO_AUTHORITY),
            "blockers": blockers,
            "descriptive_coverage_minimum_met": coverage_met,
            "effectiveness_claimed": False,
            "evaluation_cutoff": evaluation_cutoff,
            "factor_name": factor_name,
            "factor_tier_change_eligible": False,
            "factor_weight_change_eligible": False,
            "first_decision_session": first_session,
            "input_factor_diagnostic_semantic_sha256s": input_shas,
            "last_decision_session": last_session,
            "lifecycle_action": None,
            "lifecycle_conclusion": None,
            "lifecycle_diagnostic_id": "",
            "promotion_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "status": status,
            "stratum": stratum,
            "stratum_sha256": stratum_sha,
            "unique_origin_count": unique_origin_count,
            "version": FACTOR_LIFECYCLE_DIAGNOSTIC_VERSION,
        }
    )


def build_unavailable_factor_lifecycle_diagnostic(
    *,
    factor_name: str,
    evaluation_cutoff: str,
    prerequisites: Sequence[str],
) -> dict[str, Any]:
    """Build a lifecycle diagnostic for an explicit missing prerequisite set."""

    try:
        subject = require_identifier(factor_name, label="factor_name")
    except IdentityContractError as exc:
        raise FactorLifecycleDiagnosticError(str(exc)) from exc
    _canonical_timestamp(evaluation_cutoff, label="evaluation_cutoff")
    blockers = sorted(
        set(_validate_blockers(prerequisites, label="prerequisites"))
        | {"lifecycle_inputs_unavailable"}
    )
    return _seal_document(
        {
            "authority": dict(NO_AUTHORITY),
            "blockers": blockers,
            "descriptive_coverage_minimum_met": False,
            "effectiveness_claimed": False,
            "evaluation_cutoff": evaluation_cutoff,
            "factor_name": subject,
            "factor_tier_change_eligible": False,
            "factor_weight_change_eligible": False,
            "first_decision_session": None,
            "input_factor_diagnostic_semantic_sha256s": [],
            "last_decision_session": None,
            "lifecycle_action": None,
            "lifecycle_conclusion": None,
            "lifecycle_diagnostic_id": "",
            "promotion_eligible": False,
            "protocol_version": PROTOCOL_VERSION,
            "status": FactorLifecycleDiagnosticStatus.UNAVAILABLE.value,
            "stratum": None,
            "stratum_sha256": None,
            "unique_origin_count": 0,
            "version": FACTOR_LIFECYCLE_DIAGNOSTIC_VERSION,
        }
    )


def validate_factor_lifecycle_diagnostic_replay(
    artifact: Mapping[str, Any],
    *,
    evaluation_cutoff: str,
    factor_diagnostics: Sequence[Mapping[str, Any]] = (),
    factor_name: str | None = None,
    expected_factor_name: str | None = None,
    prerequisites: Sequence[str] = (),
) -> dict[str, Any]:
    """Rebuild and compare a lifecycle diagnostic without reading or writing files."""

    validated = _validate_lifecycle_diagnostic(artifact)
    if factor_diagnostics:
        if factor_name is not None or prerequisites:
            _fail("observed lifecycle replay arguments are inconsistent")
        rebuilt = build_factor_lifecycle_diagnostic(
            factor_diagnostics=factor_diagnostics,
            evaluation_cutoff=evaluation_cutoff,
            expected_factor_name=expected_factor_name,
        )
    else:
        if factor_name is None or expected_factor_name is not None:
            _fail("unavailable lifecycle replay arguments are inconsistent")
        rebuilt = build_unavailable_factor_lifecycle_diagnostic(
            factor_name=factor_name,
            evaluation_cutoff=evaluation_cutoff,
            prerequisites=prerequisites,
        )
    if canonical_bytes(validated) != canonical_bytes(rebuilt):
        _fail("factor lifecycle diagnostic replay mismatch")
    return validated


__all__ = [
    "FACTOR_LIFECYCLE_DIAGNOSTIC_VERSION",
    "FactorLifecycleDiagnosticError",
    "FactorLifecycleDiagnosticStatus",
    "build_factor_lifecycle_diagnostic",
    "build_unavailable_factor_lifecycle_diagnostic",
    "validate_factor_lifecycle_diagnostic_replay",
]
