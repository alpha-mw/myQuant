"""Research-only provisional forward evidence over immutable replayable inputs.

This lane deliberately separates replayable research eligibility from Source
Truth and production governance.  It never creates permanent security IDs,
invokes providers, updates selectors, or writes Factor Governance state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Final, NoReturn

from quant_investor.v17_v4_contract.canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    load_canonical_resource,
    seal_semantic,
    strict_json_loads,
    validate_semantic_sha,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field,
    validate_artifact,
)

from .forward_scoring_v3 import (
    average_tie_percentiles_v3,
    score_quant_forward_v3,
)
from .source_storage import GovernedStore, SourceStorageError

PROTOCOL_VERSION: Final = "myquant.v17.v4"
PROFILE: Final = "PROVISIONAL_FORWARD_EVIDENCE"
REQUEST_VERSION: Final = "myquant.v17.v4.provisional-forward-request.v1"
INPUT_VERSION: Final = "myquant.v17.v4.provisional-forward-input.v1"
ARTIFACT_VERSION: Final = "myquant.v17.v4.provisional-forward-artifact.v1"
MANIFEST_VERSION: Final = "myquant.v17.v4.provisional-forward-run-manifest.v1"
RUN_STATE: Final = "PROVISIONAL_FORWARD_EVIDENCE_ACTIVE"
PROVISIONAL_IDENTITY_STATUS: Final = "RUN_SCOPED_REPLAYABLE"
SOURCE_ADMISSIBILITY_STATUS: Final = "DEGRADED_BUT_REPLAYABLE"
DEFAULT_PROTOCOL_STATE: Final = "V15_DEFAULT"
GLOBAL_ACTIVATION_STATE: Final = "INACTIVE"
VARIANTS: Final = (
    "v17-quant-core",
    "v17-quant-plus-industry",
    "v17-quant-plus-industry-theme",
)
LABEL_HORIZONS: Final = (1, 5, 10, 20, 60)
REQUIRED_INPUT_ROLES: Final = frozenset(
    {
        "FACTOR_SET",
        "MARKET_MANIFEST",
        "MARKET_POINTER",
        "PIT_MEMBERSHIP_MANIFEST",
        "PIT_MEMBERSHIP_POINTER",
        "QUANT_INPUT",
        "RESEARCH_UNIVERSE",
    }
)
POINTER_ROLES: Final = frozenset({"MARKET_POINTER", "PIT_MEMBERSHIP_POINTER"})
BASE_LIMITATIONS: Final = (
    "CORPORATE_ACTION_CLOSURE_UNAVAILABLE",
    "DEEP_UNAVAILABLE",
    "FUNDAMENTAL_UNAVAILABLE",
    "GOVERNED_SECURITY_MASTER_UNAVAILABLE",
    "HOLDINGS_UNAVAILABLE",
    "POPULATION_CENSUS_UNAVAILABLE",
    "REGIME_EVIDENCE_UNAVAILABLE",
    "TRUSTED_AUTHORITY_ROOT_UNAVAILABLE",
)
NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
DISABLED: Final = {
    "factor_governance_write": False,
    "formal_activation_eligible": False,
    "production_governance_eligible": False,
    "promotion_eligible": False,
    "provider_calls": False,
    "selector": False,
}


class ProvisionalForwardError(RuntimeError):
    """Fail-closed provisional runtime error with preserved upstream refs."""

    exit_code = 2

    def __init__(
        self,
        code: str,
        *,
        preserved_artifact_refs: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(code)
        self.code = code
        self.preserved_artifact_refs = tuple(dict(row) for row in preserved_artifact_refs)


def _blocked(
    code: str,
    *,
    preserved_artifact_refs: Sequence[Mapping[str, Any]] = (),
) -> NoReturn:
    raise ProvisionalForwardError(
        code,
        preserved_artifact_refs=preserved_artifact_refs,
    )


def _timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        _blocked(f"PROVISIONAL_{label.upper()}_INVALID")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _blocked(f"PROVISIONAL_{label.upper()}_INVALID")
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
        or parsed.microsecond
        or parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value
    ):
        _blocked(f"PROVISIONAL_{label.upper()}_INVALID")
    return value


def _session(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 10:
        _blocked(f"PROVISIONAL_{label.upper()}_INVALID")
    try:
        if datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") != value:
            raise ValueError
    except ValueError:
        _blocked(f"PROVISIONAL_{label.upper()}_INVALID")
    return value


def _sha(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _blocked(f"PROVISIONAL_{label.upper()}_INVALID")
    return value


def _decimal(value: Any, *, label: str) -> Decimal:
    if type(value) is bool or type(value) not in {int, float, str, Decimal}:
        _blocked(f"PROVISIONAL_{label.upper()}_INVALID")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        _blocked(f"PROVISIONAL_{label.upper()}_INVALID")
    if not result.is_finite():
        _blocked(f"PROVISIONAL_{label.upper()}_NONFINITE")
    return result


def _decimal_text(value: Any, *, label: str) -> str:
    decimal = _decimal(value, label=label)
    if decimal == 0:
        return "0"
    text = format(decimal.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _relative_path(value: Any) -> str:
    if type(value) is not str:
        _blocked("PROVISIONAL_PATH_INVALID")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any("*" in part or "?" in part or "[" in part for part in path.parts)
    ):
        _blocked("PROVISIONAL_PATH_ESCAPE")
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        _blocked("PROVISIONAL_PATH_INVALID")
    return value


def _stable_workspace_read(workspace_root: Path, relative_path: str) -> bytes:
    path = workspace_root.joinpath(*PurePosixPath(_relative_path(relative_path)).parts)
    current = workspace_root
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError:
            _blocked("PROVISIONAL_EXPLICIT_INPUT_UNAVAILABLE")
        if stat.S_ISLNK(info.st_mode):
            _blocked("PROVISIONAL_SYMLINK_REJECTED")
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _blocked("PROVISIONAL_HARDLINK_OR_NONREGULAR_REJECTED")
        with path.open("rb") as handle:
            raw = handle.read(64 * 1024 * 1024 + 1)
        after = os.stat(path, follow_symlinks=False)
    except OSError:
        _blocked("PROVISIONAL_EXPLICIT_INPUT_UNAVAILABLE")
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    if len(raw) > 64 * 1024 * 1024 or identity_before != identity_after:
        _blocked("PROVISIONAL_INPUT_CHANGED_DURING_READ")
    return raw


def _artifact_ref(document: Mapping[str, Any], *, relative_path: str) -> dict[str, str]:
    version = str(document["version"])
    identity_field = artifact_identity_field(version)
    return {
        "artifact_id": str(document[identity_field]),
        "artifact_version": version,
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(document)).hexdigest(),
        "cutoff": str(document["cutoff"]),
        "relative_path": _relative_path(relative_path),
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": str(document["strategy_id"]),
    }


def _sort_refs(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(value) for value in values),
        key=lambda row: (
            str(row["relative_path"]).encode("ascii"),
            str(row["byte_sha256"]).encode("ascii"),
        ),
    )


def _artifact_identity(document: Mapping[str, Any], *, field: str) -> str:
    body = dict(document)
    body.pop(field, None)
    body.pop("semantic_sha256", None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def _seal_artifact(
    *,
    artifact_kind: str,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    created_at: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "artifact_kind": artifact_kind,
        "authority": dict(NO_AUTHORITY),
        "created_at": _timestamp(created_at, label="created_at"),
        "cutoff": _timestamp(cutoff, label="cutoff"),
        "decision_session": _session(decision_session, label="decision_session"),
        "payload": canonical_bytes(dict(payload)).decode("utf-8"),
        "production_governance_eligible": False,
        "profile": PROFILE,
        "protocol_version": PROTOCOL_VERSION,
        "research_evaluation_eligible": True,
        "research_runtime_eligible": True,
        "strategy_id": strategy_id,
        "version": ARTIFACT_VERSION,
    }
    document["artifact_id"] = _artifact_identity(document, field="artifact_id")
    sealed = seal_semantic(document)
    validate_artifact(sealed)
    return sealed


def build_provisional_input(
    *,
    role: str,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    available_at: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "authority": dict(NO_AUTHORITY),
        "available_at": _timestamp(available_at, label="available_at"),
        "cutoff": _timestamp(cutoff, label="cutoff"),
        "decision_session": _session(decision_session, label="decision_session"),
        "payload": canonical_bytes(dict(payload)).decode("utf-8"),
        "protocol_version": PROTOCOL_VERSION,
        "provider_calls": False,
        "role": role,
        "strategy_id": strategy_id,
        "version": INPUT_VERSION,
    }
    if document["available_at"] > document["cutoff"]:
        _blocked("PROVISIONAL_PIT_CUTOFF_VIOLATION")
    document["artifact_id"] = _artifact_identity(document, field="artifact_id")
    sealed = seal_semantic(document)
    validate_artifact(sealed)
    return sealed


def validate_provisional_input(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = validate_semantic_sha(document)
        validate_artifact(normalized)
    except Exception:
        _blocked("PROVISIONAL_INPUT_SCHEMA_OR_SEMANTIC_MISMATCH")
    if normalized.get("artifact_id") != _artifact_identity(normalized, field="artifact_id"):
        _blocked("PROVISIONAL_INPUT_IDENTITY_MISMATCH")
    payload = _payload(normalized)
    rebuilt = build_provisional_input(
        role=str(normalized["role"]),
        strategy_id=str(normalized["strategy_id"]),
        decision_session=str(normalized["decision_session"]),
        cutoff=str(normalized["cutoff"]),
        available_at=str(normalized["available_at"]),
        payload=payload,
    )
    if rebuilt != normalized:
        _blocked("PROVISIONAL_INPUT_REPLAY_MISMATCH")
    result = dict(normalized)
    result["payload"] = payload
    return result


def validate_provisional_artifact(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = validate_semantic_sha(document)
        validate_artifact(normalized)
    except Exception:
        _blocked("PROVISIONAL_ARTIFACT_SCHEMA_OR_SEMANTIC_MISMATCH")
    if normalized.get("artifact_id") != _artifact_identity(
        normalized,
        field="artifact_id",
    ):
        _blocked("PROVISIONAL_ARTIFACT_IDENTITY_MISMATCH")
    rebuilt = _seal_artifact(
        artifact_kind=str(normalized["artifact_kind"]),
        strategy_id=str(normalized["strategy_id"]),
        decision_session=str(normalized["decision_session"]),
        cutoff=str(normalized["cutoff"]),
        created_at=str(normalized["created_at"]),
        payload=_payload(normalized),
    )
    if rebuilt != normalized:
        _blocked("PROVISIONAL_ARTIFACT_REPLAY_MISMATCH")
    return normalized


def _payload(document: Mapping[str, Any]) -> dict[str, Any]:
    raw = document.get("payload")
    if type(raw) is not str:
        _blocked("PROVISIONAL_ARTIFACT_PAYLOAD_INVALID")
    try:
        value = strict_json_loads(raw.encode("utf-8"), label="provisional payload")
    except Exception:
        _blocked("PROVISIONAL_ARTIFACT_PAYLOAD_INVALID")
    if type(value) is not dict or canonical_bytes(value).decode("utf-8") != raw:
        _blocked("PROVISIONAL_ARTIFACT_PAYLOAD_NONCANONICAL")
    return value


def build_provisional_request(
    *,
    request_id: str,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    created_at: str,
    input_refs: Sequence[Mapping[str, Any]],
    quant_input_ref: Mapping[str, Any],
) -> dict[str, Any]:
    refs = _sort_refs(input_refs)
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "created_at": _timestamp(created_at, label="created_at"),
            "cutoff": _timestamp(cutoff, label="cutoff"),
            "decision_session": _session(decision_session, label="decision_session"),
            "input_refs": refs,
            "profile": PROFILE,
            "protocol_version": PROTOCOL_VERSION,
            "quant_input_ref": dict(quant_input_ref),
            "request_id": request_id,
            "strategy_id": strategy_id,
            "variants": list(VARIANTS),
            "version": REQUEST_VERSION,
        }
    )
    validate_artifact(document)
    return document


def validate_provisional_request(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = validate_semantic_sha(document)
        validate_artifact(normalized)
    except Exception:
        _blocked("PROVISIONAL_REQUEST_SCHEMA_OR_SEMANTIC_MISMATCH")
    if normalized.get("profile") != PROFILE or tuple(normalized.get("variants", ())) != VARIANTS:
        _blocked("PROVISIONAL_REQUEST_PROFILE_MISMATCH")
    if normalized.get("authority") != NO_AUTHORITY:
        _blocked("PROVISIONAL_AUTHORITY_ESCALATION")
    if normalized.get("decision_session") > str(normalized.get("cutoff", ""))[:10]:
        _blocked("PROVISIONAL_DECISION_SESSION_CUTOFF_MISMATCH")
    if normalized.get("input_refs") != _sort_refs(normalized.get("input_refs", ())):
        _blocked("PROVISIONAL_REQUEST_REF_ORDER")
    if normalized.get("quant_input_ref") not in normalized.get("input_refs", ()):
        _blocked("PROVISIONAL_QUANT_INPUT_NOT_BOUND")
    return normalized


def _ref_document(
    workspace_root: Path,
    reference: Mapping[str, Any],
    *,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
) -> tuple[dict[str, Any], bytes]:
    required = {
        "artifact_id",
        "artifact_version",
        "byte_sha256",
        "cutoff",
        "relative_path",
        "semantic_sha256",
        "strategy_id",
    }
    if type(reference) is not dict or set(reference) != required:
        _blocked("PROVISIONAL_INPUT_REF_SHAPE")
    if reference["strategy_id"] != strategy_id or reference["cutoff"] > cutoff:
        _blocked("PROVISIONAL_INPUT_STRATEGY_OR_CUTOFF_MISMATCH")
    raw = _stable_workspace_read(workspace_root, str(reference["relative_path"]))
    if hashlib.sha256(raw).hexdigest() != _sha(reference["byte_sha256"], label="input_sha"):
        _blocked("PROVISIONAL_IMMUTABLE_INPUT_SHA_MISMATCH")
    try:
        document = load_canonical_resource(raw, label="provisional input")
        normalized = validate_provisional_input(document)
    except Exception:
        _blocked("PROVISIONAL_INPUT_SCHEMA_OR_SEMANTIC_MISMATCH")
    if (
        normalized.get("version") != INPUT_VERSION
        or normalized.get("version") != reference["artifact_version"]
        or normalized.get("artifact_id") != reference["artifact_id"]
        or normalized.get("semantic_sha256") != reference["semantic_sha256"]
        or normalized.get("strategy_id") != strategy_id
        or normalized.get("cutoff") != reference["cutoff"]
    ):
        _blocked("PROVISIONAL_INPUT_IDENTITY_MISMATCH")
    if "decision_session" in normalized and normalized.get("decision_session") != decision_session:
        _blocked("PROVISIONAL_DECISION_SESSION_INPUT_MISMATCH")
    available_at = _timestamp(normalized.get("available_at"), label="input_available_at")
    if available_at > cutoff:
        _blocked("PROVISIONAL_PIT_CUTOFF_VIOLATION")
    if normalized.get("provider_calls") is True:
        _blocked("PROVISIONAL_UNAUTHORIZED_PROVIDER_CALL")
    return normalized, raw


def _input_documents(
    workspace_root: Path,
    request: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    by_role: dict[str, dict[str, Any]] = {}
    raw_by_role: dict[str, bytes] = {}
    for reference in request["input_refs"]:
        document, raw = _ref_document(
            workspace_root,
            reference,
            strategy_id=str(request["strategy_id"]),
            decision_session=str(request["decision_session"]),
            cutoff=str(request["cutoff"]),
        )
        role = document.get("role")
        if type(role) is not str or role in by_role:
            _blocked("PROVISIONAL_INPUT_ROLE_DUPLICATE_OR_INVALID")
        by_role[role] = document
        raw_by_role[role] = raw
    missing = sorted(REQUIRED_INPUT_ROLES - set(by_role))
    if missing:
        _blocked("PROVISIONAL_CURRENT_INPUT_GAP:" + ",".join(missing))
    quant_ref = request["quant_input_ref"]
    if by_role["QUANT_INPUT"].get("artifact_id") != quant_ref["artifact_id"]:
        _blocked("PROVISIONAL_QUANT_INPUT_IDENTITY_MISMATCH")
    return by_role, raw_by_role


def _provisional_key(
    *,
    exchange: str,
    ticker: str,
    listing_interval_ref: Mapping[str, Any],
    source_snapshot_id: str,
) -> str:
    return (
        "provisional-security-"
        + hashlib.sha256(
            canonical_bytes(
                {
                    "exchange_namespace": exchange,
                    "listing_interval_ref": dict(listing_interval_ref),
                    "pit_ticker": ticker,
                    "source_snapshot_id": source_snapshot_id,
                }
            )
        ).hexdigest()
    )


def _source_snapshot(
    *,
    request: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    market_pointer = inputs["MARKET_POINTER"]
    latest_complete_trade_date = _session(
        market_pointer.get("payload", {}).get("latest_complete_trade_date"),
        label="latest_complete_trade_date",
    )
    if latest_complete_trade_date != request["decision_session"]:
        _blocked("PROVISIONAL_DECISION_SESSION_MARKET_MISMATCH")
    snapshot_refs = _sort_refs(request["input_refs"])
    roles = {
        str(inputs_role): str(document["artifact_id"]) for inputs_role, document in inputs.items()
    }
    return _seal_artifact(
        artifact_kind="SOURCE_SNAPSHOT",
        strategy_id=str(request["strategy_id"]),
        decision_session=str(request["decision_session"]),
        cutoff=str(request["cutoff"]),
        created_at=str(request["created_at"]),
        payload={
            "available_at": max(str(document["available_at"]) for document in inputs.values()),
            "exact_input_refs": snapshot_refs,
            "provisional_identity_status": PROVISIONAL_IDENTITY_STATUS,
            "latest_complete_trade_date": latest_complete_trade_date,
            "roles": roles,
            "source_admissibility_status": SOURCE_ADMISSIBILITY_STATUS,
        },
    )


def _quant_observation(
    *,
    request: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    universe_payload = inputs["RESEARCH_UNIVERSE"].get("payload")
    factor_payload = inputs["FACTOR_SET"].get("payload")
    quant_payload = inputs["QUANT_INPUT"].get("payload")
    if not all(
        isinstance(value, Mapping) for value in (universe_payload, factor_payload, quant_payload)
    ):
        _blocked("PROVISIONAL_QUANT_INPUT_SHAPE")
    if (
        quant_payload.get("factor_set_artifact_id") != inputs["FACTOR_SET"]["artifact_id"]
        or quant_payload.get("research_universe_artifact_id")
        != inputs["RESEARCH_UNIVERSE"]["artifact_id"]
    ):
        _blocked("PROVISIONAL_QUANT_INPUT_UNIVERSE_OR_FACTOR_SET_MISMATCH")
    universe_rows = universe_payload.get("securities")
    factors = factor_payload.get("selected_factors")
    if (
        isinstance(universe_rows, (str, bytes))
        or not isinstance(universe_rows, Sequence)
        or not universe_rows
        or isinstance(factors, (str, bytes))
        or not isinstance(factors, Sequence)
        or not factors
    ):
        _blocked("PROVISIONAL_UNIVERSE_OR_FACTOR_SET_EMPTY")
    symbols: list[str] = []
    identities: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(universe_rows):
        if not isinstance(row, Mapping):
            _blocked("PROVISIONAL_UNIVERSE_ROW_SHAPE")
        exchange = row.get("exchange")
        ticker = row.get("pit_ticker")
        symbol = row.get("symbol")
        interval_ref = row.get("listing_interval_ref")
        if (
            type(exchange) is not str
            or exchange not in {"BJ", "SH", "SZ"}
            or type(ticker) is not str
            or type(symbol) is not str
            or symbol != f"{ticker}.{exchange}"
            or not isinstance(interval_ref, Mapping)
        ):
            _blocked(f"PROVISIONAL_UNIVERSE_ROW_INVALID:{index}")
        if symbol in identities:
            _blocked("PROVISIONAL_UNIVERSE_DUPLICATE_SYMBOL")
        key = _provisional_key(
            exchange=exchange,
            ticker=ticker,
            listing_interval_ref=interval_ref,
            source_snapshot_id=str(source_snapshot["artifact_id"]),
        )
        identities[symbol] = {
            "exchange_namespace": exchange,
            "provisional_identity_status": PROVISIONAL_IDENTITY_STATUS,
            "listing_interval_ref": dict(interval_ref),
            "pit_ticker": ticker,
            "provisional_security_key": key,
            "symbol": symbol,
        }
        symbols.append(symbol)
    if symbols != sorted(symbols, key=lambda value: value.encode("ascii")):
        _blocked("PROVISIONAL_UNIVERSE_ORDER")
    scoring = score_quant_forward_v3(
        symbols=symbols,
        selected_factors=factors,
        factor_values=quant_payload.get("factor_values", {}),
        neutralizer_inputs=quant_payload.get("neutralizer_inputs", {}),
        cutoff=str(request["cutoff"]),
    )
    available_scores = {
        str(row["symbol"]): row["effective_score"]
        for row in scoring["records"]
        if row["effective_score"] is not None
    }
    ranks = average_tie_percentiles_v3(available_scores)
    observation_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    factor_implementations = {
        str(row["name"]): _sha(row.get("implementation_sha256"), label="factor_implementation")
        for row in factors
    }
    if quant_payload.get("factor_implementation_sha256") != factor_implementations:
        _blocked("PROVISIONAL_FACTOR_IMPLEMENTATION_SHA_MISMATCH")
    for record in scoring["records"]:
        symbol = str(record["symbol"])
        available_count = int(record["available_factor_count"])
        available_family_count = int(record["available_family_count"])
        family_counts = {
            str(row["family"]): int(row["available_factor_count"])
            for row in record["family_scores"]
        }
        for evidence in record["factor_evidence"]:
            exposure = evidence["exposure"]
            family_count = family_counts.get(str(evidence["family"]), 0)
            renormalized_weight = (
                Decimal("0")
                if not family_count or not available_family_count
                else Decimal("1") / available_family_count / family_count
            )
            observation_rows.append(
                {
                    **identities[symbol],
                    "availability": str(evidence["status"]),
                    "available_factor_count": available_count,
                    "available_family_count": int(record["available_family_count"]),
                    "contribution": (
                        None
                        if exposure is None
                        else _decimal_text(
                            Decimal(str(exposure)) * renormalized_weight,
                            label="contribution",
                        )
                    ),
                    "coverage_penalty": _decimal_text(
                        record["confidence_penalty"],
                        label="coverage_penalty",
                    ),
                    "cutoff": str(request["cutoff"]),
                    "decision_session": str(request["decision_session"]),
                    "factor_family": str(evidence["family"]),
                    "factor_implementation_sha256": factor_implementations[
                        str(evidence["factor_name"])
                    ],
                    "factor_name": str(evidence["factor_name"]),
                    "factor_set_ref": next(
                        dict(ref)
                        for ref in request["input_refs"]
                        if ref["artifact_id"] == inputs["FACTOR_SET"]["artifact_id"]
                    ),
                    "missing_reason": (
                        None
                        if evidence["status"] in {"AVAILABLE", "ZERO_MAD"}
                        else evidence["status"]
                    ),
                    "neutralized_exposure": (
                        None
                        if exposure is None
                        else _decimal_text(exposure, label="neutralized_exposure")
                    ),
                    "rank": (
                        None if symbol not in ranks else _decimal_text(ranks[symbol], label="rank")
                    ),
                    "raw_exposure": (
                        None
                        if evidence["raw_value"] is None
                        else _decimal_text(evidence["raw_value"], label="raw_exposure")
                    ),
                    "renormalized_weight": _decimal_text(
                        renormalized_weight,
                        label="renormalized_weight",
                    ),
                    "source_snapshot_id": str(source_snapshot["artifact_id"]),
                    "universe_ref": next(
                        dict(ref)
                        for ref in request["input_refs"]
                        if ref["artifact_id"] == inputs["RESEARCH_UNIVERSE"]["artifact_id"]
                    ),
                    "weight_coverage": _decimal_text(record["coverage"], label="weight_coverage"),
                    "winsorized_exposure": (
                        None
                        if evidence["winsorized_value"] is None
                        else _decimal_text(
                            evidence["winsorized_value"], label="winsorized_exposure"
                        )
                    ),
                }
            )
        score_rows.append(
            {
                **identities[symbol],
                "quant_score": (
                    None
                    if record["effective_score"] is None
                    else _decimal_text(record["effective_score"], label="quant_score")
                ),
                "rank": None if symbol not in ranks else _decimal_text(ranks[symbol], label="rank"),
                "status": record["status"],
            }
        )
    observation_rows.sort(
        key=lambda row: (row["symbol"].encode("ascii"), row["factor_name"].encode("utf-8"))
    )
    return _seal_artifact(
        artifact_kind="QUANT_OBSERVATION",
        strategy_id=str(request["strategy_id"]),
        decision_session=str(request["decision_session"]),
        cutoff=str(request["cutoff"]),
        created_at=str(request["created_at"]),
        payload={
            "factor_evidence_status": "ACCUMULATING",
            "observation_rows": observation_rows,
            "regime_conditioned_status": "UNAVAILABLE",
            "score_rows": score_rows,
            "selected_factor_count": scoring["selected_factor_count"],
            "selected_family_count": scoring["selected_family_count"],
            "source_snapshot_id": str(source_snapshot["artifact_id"]),
        },
    )


def _optional_scores(
    document: Mapping[str, Any] | None,
    *,
    symbols: set[str],
    role: str,
) -> dict[str, Decimal] | None:
    if document is None:
        return None
    payload = document.get("payload")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("scores"), Mapping):
        _blocked(f"PROVISIONAL_{role}_INPUT_INVALID")
    scores: dict[str, Decimal] = {}
    for symbol, value in payload["scores"].items():
        if symbol not in symbols:
            _blocked(f"PROVISIONAL_{role}_UNIVERSE_MISMATCH")
        scores[str(symbol)] = _decimal(value, label=f"{role}_score")
    if set(scores) != symbols:
        _blocked(f"PROVISIONAL_{role}_UNIVERSE_MISMATCH")
    return scores


def _variant_result(
    *,
    request: Mapping[str, Any],
    observation: Mapping[str, Any],
    variant: str,
    industry_scores: Mapping[str, Decimal] | None,
    theme_scores: Mapping[str, Decimal] | None,
) -> dict[str, Any]:
    quant_rows = _payload(observation)["score_rows"]
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    if variant == VARIANTS[0]:
        status = "COMPLETE"
    elif variant == VARIANTS[1]:
        status = "COMPLETE" if industry_scores is not None else "PARTIAL"
        if industry_scores is None:
            blockers.append("INDUSTRY_CONTEXT_UNAVAILABLE")
    else:
        available = sum(value is not None for value in (industry_scores, theme_scores))
        status = "COMPLETE" if available == 2 else "PARTIAL" if available == 1 else "UNAVAILABLE"
        if industry_scores is None:
            blockers.append("INDUSTRY_CONTEXT_UNAVAILABLE")
        if theme_scores is None:
            blockers.append("THEME_EXPOSURE_UNAVAILABLE")
    for quant_row in quant_rows:
        symbol = str(quant_row["symbol"])
        quant = None if quant_row["quant_score"] is None else Decimal(quant_row["quant_score"])
        components: list[Decimal] = [] if quant is None else [quant]
        if variant != VARIANTS[0] and industry_scores is not None:
            components.append(industry_scores[symbol])
        if variant == VARIANTS[2] and theme_scores is not None:
            components.append(theme_scores[symbol])
        combined = (
            None if not components or status == "UNAVAILABLE" else sum(components) / len(components)
        )
        rows.append(
            {
                "provisional_security_key": quant_row["provisional_security_key"],
                "score": (
                    None if combined is None else _decimal_text(combined, label="variant_score")
                ),
                "symbol": symbol,
            }
        )
    return _seal_artifact(
        artifact_kind="VARIANT_RESULT",
        strategy_id=str(request["strategy_id"]),
        decision_session=str(request["decision_session"]),
        cutoff=str(request["cutoff"]),
        created_at=str(request["created_at"]),
        payload={
            "blocker_codes": blockers,
            "observation_id": observation["artifact_id"],
            "rows": rows,
            "status": status,
            "variant": variant,
        },
    )


def build_provisional_forward_label(
    *,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    created_at: str,
    horizon_sessions: int,
    observation_ref: Mapping[str, Any],
    label_session: str | None = None,
    future_sessions: Sequence[str] = (),
    calendar_ref: Mapping[str, Any] | None = None,
    rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if horizon_sessions not in LABEL_HORIZONS:
        _blocked("PROVISIONAL_LABEL_HORIZON_INVALID")
    origin = _session(decision_session, label="decision_session")
    if label_session is None:
        if future_sessions or calendar_ref is not None or rows:
            _blocked("PROVISIONAL_PENDING_LABEL_HAS_FUTURE_INPUT")
        status = "PENDING"
        normalized_rows: list[dict[str, Any]] = []
        normalized_sessions: list[str] = []
    else:
        end = _session(label_session, label="label_session")
        normalized_sessions = [_session(value, label="future_session") for value in future_sessions]
        if (
            end <= origin
            or len(normalized_sessions) != horizon_sessions
            or normalized_sessions != sorted(set(normalized_sessions))
            or not normalized_sessions
            or normalized_sessions[0] <= origin
            or normalized_sessions[-1] != end
            or calendar_ref is None
        ):
            _blocked("PROVISIONAL_LABEL_FUTURE_WINDOW_VIOLATION")
        status = "MATURED"
        normalized_rows = []
        for row in rows:
            normalized: dict[str, Any] = {
                "cost_adjusted_return": None,
                "industry_excess_return": None,
                "market_excess_return": None,
                "provisional_security_key": row.get("provisional_security_key"),
                "research_close_to_close_return": None,
                "tradable_return": None,
            }
            for field in (
                "cost_adjusted_return",
                "industry_excess_return",
                "market_excess_return",
                "research_close_to_close_return",
                "tradable_return",
            ):
                value = row.get(field)
                normalized[field] = (
                    {"status": "UNAVAILABLE", "value": None}
                    if value is None
                    else {"status": "AVAILABLE", "value": _decimal_text(value, label=field)}
                )
            normalized_rows.append(normalized)
        normalized_rows.sort(key=lambda row: str(row["provisional_security_key"]).encode("ascii"))
    return _seal_artifact(
        artifact_kind="FORWARD_LABEL",
        strategy_id=strategy_id,
        decision_session=origin,
        cutoff=cutoff,
        created_at=created_at,
        payload={
            "historical_backfill_eligible": False,
            "horizon_sessions": horizon_sessions,
            "calendar_ref": None if calendar_ref is None else dict(calendar_ref),
            "future_sessions": normalized_sessions,
            "label_rows": normalized_rows,
            "label_session": label_session,
            "observation_ref": dict(observation_ref),
            "status": status,
        },
    )


def build_provisional_evaluation_receipt(
    *,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    created_at: str,
    receipt_kind: str,
    subject_id: str,
    evidence_refs: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if receipt_kind not in {"FACTOR", "BRANCH", "VARIANT_COMPARISON"}:
        _blocked("PROVISIONAL_EVALUATION_KIND_INVALID")
    metric_rows = [
        {"metric": name, "value": _decimal_text(value, label=f"metric_{name}")}
        for name, value in sorted(metrics.items(), key=lambda item: item[0].encode("utf-8"))
    ]
    return _seal_artifact(
        artifact_kind="EVALUATION_RECEIPT",
        strategy_id=strategy_id,
        decision_session=decision_session,
        cutoff=cutoff,
        created_at=created_at,
        payload={
            "evidence_refs": _sort_refs(evidence_refs),
            "factor_governance_write": False,
            "formal_activation_eligible": False,
            "metric_rows": metric_rows,
            "promotion_eligible": False,
            "receipt_kind": receipt_kind,
            "research_evaluation_eligible": True,
            "subject_id": subject_id,
        },
    )


def _write_artifact(
    store: GovernedStore,
    *,
    root: PurePosixPath,
    name: str,
    artifact: Mapping[str, Any],
) -> dict[str, str]:
    path = root / f"{name}-{artifact['artifact_id']}.json"
    raw = canonical_resource_bytes(artifact)
    try:
        store.write_exact_once(path, raw)
        readback = load_canonical_resource(store.read(path, hashlib.sha256(raw).hexdigest()))
    except SourceStorageError:
        _blocked("PROVISIONAL_EXACT_ONCE_OR_STORAGE_FAILURE")
    if validate_provisional_artifact(readback) != artifact:
        _blocked("PROVISIONAL_READBACK_REPLAY_MISMATCH")
    return _artifact_ref(artifact, relative_path=str(path))


def _stage_receipt(
    *,
    request: Mapping[str, Any],
    stage: str,
    status: str,
    output_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _seal_artifact(
        artifact_kind="STAGE_RECEIPT",
        strategy_id=str(request["strategy_id"]),
        decision_session=str(request["decision_session"]),
        cutoff=str(request["cutoff"]),
        created_at=str(request["created_at"]),
        payload={
            "output_refs": _sort_refs(output_refs),
            "readback_replay": "PASSED",
            "stage": stage,
            "status": status,
        },
    )


def _request_ref(request: Mapping[str, Any], *, relative_path: str) -> dict[str, str]:
    return {
        "artifact_id": str(request["request_id"]),
        "artifact_version": REQUEST_VERSION,
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(request)).hexdigest(),
        "cutoff": str(request["cutoff"]),
        "relative_path": _relative_path(relative_path),
        "semantic_sha256": str(request["semantic_sha256"]),
        "strategy_id": str(request["strategy_id"]),
    }


def _manifest(
    *,
    request: Mapping[str, Any],
    request_path: str,
    artifact_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "artifact_refs": _sort_refs(artifact_refs),
        "authority": dict(NO_AUTHORITY),
        "created_at": str(request["created_at"]),
        "cutoff": str(request["cutoff"]),
        "decision_session": str(request["decision_session"]),
        "default_protocol_state": DEFAULT_PROTOCOL_STATE,
        "factor_evidence_status": "ACCUMULATING",
        "global_activation_state": GLOBAL_ACTIVATION_STATE,
        "profile": PROFILE,
        "protocol_version": PROTOCOL_VERSION,
        "regime_conditioned_status": "UNAVAILABLE",
        "request_ref": _request_ref(request, relative_path=request_path),
        "run_state": RUN_STATE,
        "strategy_id": str(request["strategy_id"]),
        "version": MANIFEST_VERSION,
    }
    document["manifest_id"] = _artifact_identity(document, field="manifest_id")
    sealed = seal_semantic(document)
    validate_artifact(sealed)
    return sealed


def run_provisional_forward(
    workspace_root: str,
    *,
    request_path: str,
    request_sha256: str,
) -> dict[str, Any]:
    """Run one exact immutable provisional request without provider access."""

    root = Path(workspace_root)
    if not root.is_absolute():
        _blocked("PROVISIONAL_WORKSPACE_ROOT_INVALID")
    request_raw = _stable_workspace_read(root, request_path)
    if hashlib.sha256(request_raw).hexdigest() != _sha(request_sha256, label="request_sha"):
        _blocked("PROVISIONAL_REQUEST_SHA_MISMATCH")
    try:
        request = validate_provisional_request(
            load_canonical_resource(request_raw, label="provisional request")
        )
    except ProvisionalForwardError:
        raise
    except Exception:
        _blocked("PROVISIONAL_REQUEST_INVALID")
    inputs, initial_raw = _input_documents(root, request)
    source_snapshot = _source_snapshot(request=request, inputs=inputs)
    run_root = PurePosixPath(
        "results/v17_v4_shadow/provisional_forward",
        str(request["strategy_id"]),
        str(request["request_id"]),
    )
    store = GovernedStore(root)
    bound_request_path = run_root / "request.json"
    try:
        store.write_exact_once(bound_request_path, request_raw)
        if store.read(bound_request_path, request_sha256) != request_raw:
            _blocked("PROVISIONAL_REQUEST_BINDING_READBACK_MISMATCH")
    except SourceStorageError:
        _blocked("PROVISIONAL_EXACT_ONCE_REQUEST_CONFLICT")
    refs: list[dict[str, str]] = []
    source_ref = _write_artifact(
        store,
        root=run_root,
        name="source-snapshot",
        artifact=source_snapshot,
    )
    refs.append(source_ref)
    source_receipt = _stage_receipt(
        request=request,
        stage="source_snapshot",
        status="COMPLETE",
        output_refs=[source_ref],
    )
    refs.append(
        _write_artifact(
            store,
            root=run_root,
            name="stage-source",
            artifact=source_receipt,
        )
    )
    observation = _quant_observation(
        request=request,
        inputs=inputs,
        source_snapshot=source_snapshot,
    )
    observation_ref = _write_artifact(
        store,
        root=run_root,
        name="quant-observation",
        artifact=observation,
    )
    refs.append(observation_ref)
    quant_receipt = _stage_receipt(
        request=request,
        stage="quant_observation",
        status="COMPLETE",
        output_refs=[observation_ref],
    )
    refs.append(_write_artifact(store, root=run_root, name="stage-quant", artifact=quant_receipt))
    symbols = {str(row["symbol"]) for row in _payload(observation)["score_rows"]}
    try:
        industry_scores = _optional_scores(
            inputs.get("INDUSTRY_CONTEXT"),
            symbols=symbols,
            role="INDUSTRY_CONTEXT",
        )
        theme_scores = _optional_scores(
            inputs.get("THEME_EXPOSURE"),
            symbols=symbols,
            role="THEME_EXPOSURE",
        )
    except ProvisionalForwardError as exc:
        raise ProvisionalForwardError(
            exc.code,
            preserved_artifact_refs=refs,
        ) from exc
    variant_refs: list[dict[str, str]] = []
    for variant in VARIANTS:
        artifact = _variant_result(
            request=request,
            observation=observation,
            variant=variant,
            industry_scores=industry_scores,
            theme_scores=theme_scores,
        )
        variant_refs.append(
            _write_artifact(
                store,
                root=run_root,
                name=f"variant-{variant}",
                artifact=artifact,
            )
        )
    refs.extend(variant_refs)
    variant_receipt = _stage_receipt(
        request=request,
        stage="variants",
        status="COMPLETE",
        output_refs=variant_refs,
    )
    refs.append(
        _write_artifact(store, root=run_root, name="stage-variants", artifact=variant_receipt)
    )
    label_refs: list[dict[str, str]] = []
    for horizon in LABEL_HORIZONS:
        label = build_provisional_forward_label(
            strategy_id=str(request["strategy_id"]),
            decision_session=str(request["decision_session"]),
            cutoff=str(request["cutoff"]),
            created_at=str(request["created_at"]),
            horizon_sessions=horizon,
            observation_ref=observation_ref,
        )
        label_refs.append(
            _write_artifact(
                store,
                root=run_root,
                name=f"label-{horizon:02d}",
                artifact=label,
            )
        )
    refs.extend(label_refs)
    label_receipt = _stage_receipt(
        request=request,
        stage="forward_labels",
        status="PENDING",
        output_refs=label_refs,
    )
    refs.append(
        _write_artifact(
            store,
            root=run_root,
            name="stage-labels",
            artifact=label_receipt,
        )
    )
    limitations = list(BASE_LIMITATIONS)
    if industry_scores is None:
        limitations.append("INDUSTRY_CONTEXT_UNAVAILABLE")
    if theme_scores is None:
        limitations.append("THEME_EXPOSURE_UNAVAILABLE")
    for role in sorted(POINTER_ROLES):
        reference = next(
            ref
            for ref in request["input_refs"]
            if ref["artifact_id"] == inputs[role]["artifact_id"]
        )
        current = _stable_workspace_read(root, str(reference["relative_path"]))
        if current != initial_raw[role]:
            limitations.append("CURRENT_POINTER_CHANGED_DURING_RUN")
            break
    limitation_receipt = _seal_artifact(
        artifact_kind="LIMITATION_RECEIPT",
        strategy_id=str(request["strategy_id"]),
        decision_session=str(request["decision_session"]),
        cutoff=str(request["cutoff"]),
        created_at=str(request["created_at"]),
        payload={
            "limitation_codes": sorted(set(limitations), key=lambda value: value.encode("ascii")),
            "source_snapshot_ref": source_ref,
        },
    )
    refs.append(
        _write_artifact(
            store,
            root=run_root,
            name="limitations",
            artifact=limitation_receipt,
        )
    )
    manifest = _manifest(
        request=request,
        request_path=str(bound_request_path),
        artifact_refs=refs,
    )
    manifest_path = run_root / f"manifest-{manifest['manifest_id']}.json"
    manifest_raw = canonical_resource_bytes(manifest)
    try:
        store.write_exact_once(manifest_path, manifest_raw)
        readback = load_canonical_resource(
            store.read(manifest_path, hashlib.sha256(manifest_raw).hexdigest()),
            label="provisional run manifest",
        )
    except SourceStorageError:
        _blocked("PROVISIONAL_MANIFEST_STORAGE_FAILURE", preserved_artifact_refs=refs)
    if readback != manifest:
        _blocked("PROVISIONAL_MANIFEST_READBACK_MISMATCH", preserved_artifact_refs=refs)
    try:
        validate_artifact(readback)
    except Exception:
        _blocked("PROVISIONAL_MANIFEST_SCHEMA_MISMATCH", preserved_artifact_refs=refs)
    if readback.get("manifest_id") != _artifact_identity(readback, field="manifest_id"):
        _blocked("PROVISIONAL_MANIFEST_IDENTITY_MISMATCH", preserved_artifact_refs=refs)
    return {
        "artifact_manifest_ref": _artifact_ref(manifest, relative_path=str(manifest_path)),
        "authority": dict(NO_AUTHORITY),
        **dict(DISABLED),
        "default_protocol_state": DEFAULT_PROTOCOL_STATE,
        "factor_evidence_status": "ACCUMULATING",
        "global_activation_state": GLOBAL_ACTIVATION_STATE,
        "profile": PROFILE,
        "provisional_identity_status": PROVISIONAL_IDENTITY_STATUS,
        "regime_conditioned_status": "UNAVAILABLE",
        "research_evaluation_eligible": True,
        "research_runtime_available": True,
        "research_runtime_default": False,
        "research_runtime_eligible": True,
        "run_state": RUN_STATE,
        "security_master_status": "UNAVAILABLE",
        "source_admissibility_status": SOURCE_ADMISSIBILITY_STATUS,
        "status": "COMPLETE",
        "variant_statuses": {
            VARIANTS[0]: "COMPLETE",
            VARIANTS[1]: "COMPLETE" if industry_scores is not None else "PARTIAL",
            VARIANTS[2]: (
                "COMPLETE"
                if industry_scores is not None and theme_scores is not None
                else (
                    "PARTIAL"
                    if industry_scores is not None or theme_scores is not None
                    else "UNAVAILABLE"
                )
            ),
        },
    }


__all__ = [
    "ARTIFACT_VERSION",
    "DEFAULT_PROTOCOL_STATE",
    "GLOBAL_ACTIVATION_STATE",
    "PROVISIONAL_IDENTITY_STATUS",
    "INPUT_VERSION",
    "LABEL_HORIZONS",
    "MANIFEST_VERSION",
    "NO_AUTHORITY",
    "PROFILE",
    "ProvisionalForwardError",
    "REQUEST_VERSION",
    "RUN_STATE",
    "SOURCE_ADMISSIBILITY_STATUS",
    "VARIANTS",
    "build_provisional_evaluation_receipt",
    "build_provisional_forward_label",
    "build_provisional_input",
    "build_provisional_request",
    "run_provisional_forward",
    "validate_provisional_artifact",
    "validate_provisional_input",
    "validate_provisional_request",
]
