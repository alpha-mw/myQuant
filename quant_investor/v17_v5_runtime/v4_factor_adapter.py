"""Deterministic, read-only V17 v4 forward-evidence adapter for V17 v5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import re
from typing import Any, Final, Mapping, Sequence
from zoneinfo import ZoneInfo

from quant_investor.v17_v5_contract.canonical import canonical_bytes
from quant_investor.v17_v5_contract.identities import (
    IdentityContractError,
    require_identifier,
    require_relative_path,
    require_sha256,
)
from quant_investor.v17_v5_contract.resources import (
    V4_FACTOR_EVIDENCE_ADAPTER_POLICY_PATH,
    load_v4_factor_evidence_adapter_policy,
    read_packaged_asset,
)
from quant_investor.v17_v5_contract.validators import (
    V4_COMPATIBILITY_POLICY_BYTE_SHA256,
    V4_PACKAGE_MANIFEST_SHA256,
    V4_RUNTIME_MANIFEST_SHA256,
    V4_SOURCE_GIT_COMMIT,
)
from quant_investor.v17_v5_runtime.factor_diagnostics import (
    FactorOriginSample,
    FactorSampleStratum,
    build_factor_diagnostic,
    build_unavailable_factor_diagnostic,
)
from quant_investor.v17_v5_runtime.v4_compat_reader import V4CompatibilityRead

EVALUATION_RECEIPT_VERSION: Final = "myquant.v17.v4.forward-evaluation-receipt.v1"
FACTOR_INVENTORY_VERSION: Final = "myquant.v17.v4.existing-factor-inventory.v1"
FACTOR_OBSERVATION_VERSION: Final = "myquant.v17.v4.factor-universe-observation.v1"
FACTOR_SET_VERSION: Final = "myquant.v17.v4.research-shadow-factor-set.v1"
FORWARD_LABEL_VERSION: Final = "myquant.v17.v4.forward-label.v1"
OBSERVATION_RUN_VERSION: Final = "myquant.v17.v4.forward-observation-run.v1"
REQUEST_VERSION: Final = "myquant.v17.v4.forward-run-request.v1"
SOURCE_BUNDLE_VERSION: Final = "myquant.v17.v4.forward-factor-input-bundle.v1"
SOURCE_LOCATOR_VERSION: Final = "myquant.v17.v4.forward-source-locator.v1"
HORIZON_SESSIONS: Final = 20
SHANGHAI_TZ: Final = ZoneInfo("Asia/Shanghai")
_DECIMAL_RE: Final = re.compile(
    r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$",
    re.ASCII,
)


class V4FactorAdapterError(ValueError):
    """Raised when a V4 closure is contradictory or unsuitable for adaptation."""

    exit_code = 2


class V4FactorAdaptationStatus(str, Enum):
    """The only adapter outcomes."""

    ACCUMULATING = "ACCUMULATING"
    UNAVAILABLE = "UNAVAILABLE"
    UNOBSERVED = "UNOBSERVED"


@dataclass(frozen=True)
class V4ArtifactReference:
    """One exact closure-verified V4 artifact reference."""

    artifact_id: str
    artifact_version: str
    byte_sha256: str
    cutoff: str
    relative_path: str
    semantic_sha256: str
    strategy_id: str


@dataclass(frozen=True)
class V4FactorOriginBinding:
    """Exact V4 references and counts retained for one adapted origin."""

    comparable_symbol_count: int
    decision_session: str
    eligible_symbol_count: int
    evaluation_receipt_ref: V4ArtifactReference
    factor_implementation_sha256: str
    factor_observation_ref: V4ArtifactReference
    factor_set_ref: V4ArtifactReference
    forward_label_ref: V4ArtifactReference
    horizon_end_session: str
    observation_run_ref: V4ArtifactReference
    origin_cutoff: str
    origin_id: str
    request_ref: V4ArtifactReference
    source_locator_ref: V4ArtifactReference


@dataclass(frozen=True)
class V4FactorEvidenceAdaptation:
    """Pure in-memory evidence returned to the V5 diagnostic builder."""

    blockers: tuple[str, ...]
    origin_bindings: tuple[V4FactorOriginBinding, ...]
    origins: tuple[FactorOriginSample, ...]
    status: V4FactorAdaptationStatus
    stratum: FactorSampleStratum | None


def _fail(message: str) -> None:
    raise V4FactorAdapterError(message)


def _instant(value: Any, *, label: str) -> datetime:
    if type(value) is not str:
        _fail(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise V4FactorAdapterError(f"{label} must be a second-precision UTC timestamp") from exc
    return parsed


def _session(value: Any, *, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label} must be an ISO session date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise V4FactorAdapterError(f"{label} must be an ISO session date") from exc
    if parsed.isoformat() != value:
        _fail(f"{label} must be canonical")
    return value


def _identifier(value: Any, *, label: str) -> str:
    try:
        return require_identifier(value, label=label)
    except IdentityContractError as exc:
        raise V4FactorAdapterError(str(exc)) from exc


def _document(
    read: V4CompatibilityRead,
    reference: Mapping[str, Any],
    *,
    version: str,
    label: str,
) -> dict[str, Any]:
    if type(reference) is not dict or reference.get("artifact_version") != version:
        _fail(f"{label} version mismatch")
    path = reference.get("relative_path")
    if type(path) is not str:
        _fail(f"{label} path is absent")
    document = read.documents.get(path)
    nodes = {node.relative_path: node for node in read.closure}
    node = nodes.get(path)
    if document is None or node is None:
        _fail(f"{label} is absent from the verified closure")
    if document.get("semantic_sha256") != node.semantic_sha256:
        _fail(f"{label} closure document semantic SHA mismatch")
    expected = {
        "artifact_id": node.artifact_id,
        "artifact_version": node.version,
        "byte_sha256": node.byte_sha256,
        "cutoff": document.get("cutoff"),
        "relative_path": node.relative_path,
        "semantic_sha256": node.semantic_sha256,
        "strategy_id": document.get("strategy_id"),
    }
    if dict(reference) != expected:
        _fail(f"{label} exact reference mismatch")
    return dict(document)


def _single_reference(
    values: Any,
    *,
    version: str,
    label: str,
) -> dict[str, Any]:
    if type(values) is not list:
        _fail(f"{label} must be a reference array")
    matches = [
        value
        for value in values
        if type(value) is dict and value.get("artifact_version") == version
    ]
    if len(matches) != 1:
        _fail(f"{label} must contain exactly one {version} reference")
    return dict(matches[0])


def _reference_equal(left: Mapping[str, Any], right: Mapping[str, Any], *, label: str) -> None:
    if dict(left) != dict(right):
        _fail(f"{label} exact reference mismatch")


def _artifact_reference(
    value: Mapping[str, Any],
    *,
    label: str,
) -> V4ArtifactReference:
    expected = {
        "artifact_id",
        "artifact_version",
        "byte_sha256",
        "cutoff",
        "relative_path",
        "semantic_sha256",
        "strategy_id",
    }
    if type(value) is not dict or set(value) != expected:
        _fail(f"{label} must be an exact V4 artifact reference")
    try:
        return V4ArtifactReference(
            artifact_id=_identifier(value["artifact_id"], label=f"{label}.artifact_id"),
            artifact_version=_identifier(
                value["artifact_version"],
                label=f"{label}.artifact_version",
            ),
            byte_sha256=require_sha256(
                value["byte_sha256"],
                label=f"{label}.byte_sha256",
            ),
            cutoff=_instant(value["cutoff"], label=f"{label}.cutoff").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            relative_path=require_relative_path(
                value["relative_path"],
                label=f"{label}.relative_path",
            ),
            semantic_sha256=require_sha256(
                value["semantic_sha256"],
                label=f"{label}.semantic_sha256",
            ),
            strategy_id=_identifier(value["strategy_id"], label=f"{label}.strategy_id"),
        )
    except (IdentityContractError, KeyError, TypeError) as exc:
        raise V4FactorAdapterError(f"{label} is malformed") from exc


def _canonical_decimal(value: Any, *, label: str) -> str:
    if type(value) is not str or _DECIMAL_RE.fullmatch(value) is None:
        _fail(f"{label} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise V4FactorAdapterError(f"{label} is not a finite decimal") from exc
    if not parsed.is_finite() or (parsed.is_zero() and value.startswith("-")):
        _fail(f"{label} is not a canonical finite decimal")
    return value


def _calendar(
    open_sessions: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, int], str]:
    if isinstance(open_sessions, (str, bytes)) or not isinstance(open_sessions, Sequence):
        _fail("open_sessions must be a sequence")
    sessions = tuple(
        _session(value, label=f"open_sessions[{index}]")
        for index, value in enumerate(open_sessions)
    )
    if not sessions or sessions != tuple(sorted(set(sessions))):
        _fail("open_sessions must be nonempty, unique, and ASCII ascending")
    return (
        sessions,
        {session: index for index, session in enumerate(sessions)},
        hashlib.sha256(canonical_bytes(list(sessions))).hexdigest(),
    )


def _source_locator_and_bundle(
    read: V4CompatibilityRead,
    *,
    factor_observation: Mapping[str, Any],
    factor_set_ref: Mapping[str, Any],
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    observation_locator_ref = _single_reference(
        factor_observation["source_refs"],
        version=SOURCE_LOCATOR_VERSION,
        label="factor_observation.source_refs",
    )
    request_locator_ref = _single_reference(
        request["source_refs"],
        version=SOURCE_LOCATOR_VERSION,
        label="request.source_refs",
    )
    _reference_equal(
        observation_locator_ref,
        request_locator_ref,
        label="source locator",
    )
    locator = _document(
        read,
        observation_locator_ref,
        version=SOURCE_LOCATOR_VERSION,
        label="source locator",
    )
    bundle_ref = locator["factor_input_bundle_ref"]
    bundle = _document(
        read,
        bundle_ref,
        version=SOURCE_BUNDLE_VERSION,
        label="factor input bundle",
    )
    _reference_equal(
        bundle["factor_set_ref"],
        factor_set_ref,
        label="factor input bundle factor set",
    )
    if (
        locator["decision_session"] != factor_observation["decision_session"]
        or bundle["decision_session"] != factor_observation["decision_session"]
        or bundle["source_set_sha256"] != locator["source_set_sha256"]
    ):
        _fail("source locator and factor input bundle lineage mismatch")
    source_series = {
        "factor_input_bundle_version": bundle["version"],
        "factor_slice_fields": sorted(row["field_name"] for row in bundle["factor_slices"]),
        "neutralizer_fields": list(bundle["neutralizer_fields"]),
        "required_fields": list(bundle["required_fields"]),
        "source_locator_version": locator["version"],
    }
    return locator, bundle, source_series


def _factor_definition(
    factor_set: Mapping[str, Any],
    factor_name: str,
) -> dict[str, Any]:
    matches = [row for row in factor_set["selected_factors"] if row.get("name") == factor_name]
    if len(matches) != 1:
        _fail("factor set must contain exactly one matching selected factor")
    return dict(matches[0])


def _origin_from_read(
    read: V4CompatibilityRead,
    *,
    evaluation_cutoff: datetime,
    factor_name: str,
    sessions: Sequence[str],
    session_index: Mapping[str, int],
    market_calendar_sha256: str,
) -> tuple[
    FactorSampleStratum | None,
    FactorOriginSample | None,
    V4FactorOriginBinding | None,
    tuple[str, ...],
]:
    receipt = dict(read.document)
    if receipt.get("version") != EVALUATION_RECEIPT_VERSION:
        _fail("adapter root must be a V4 forward evaluation receipt")
    root_receipt = _document(
        read,
        read.root_ref,
        version=EVALUATION_RECEIPT_VERSION,
        label="evaluation receipt root",
    )
    if root_receipt != receipt:
        _fail("evaluation receipt root readback mismatch")
    if (
        receipt.get("receipt_type") != "factor_evaluation_receipt"
        or receipt.get("subject_id") != factor_name
        or receipt.get("lineage_key", {}).get("factor_name") != factor_name
    ):
        _fail("evaluation receipt factor subject mismatch")
    if _instant(receipt["cutoff"], label="receipt.cutoff") > evaluation_cutoff:
        _fail("evaluation receipt cutoff exceeds evaluation cutoff")
    lineage = dict(receipt["lineage_key"])
    if receipt["lineage_key_sha256"] != hashlib.sha256(canonical_bytes(lineage)).hexdigest():
        _fail("evaluation receipt lineage SHA mismatch")
    if lineage["horizon_sessions"] != HORIZON_SESSIONS:
        return None, None, None, ("exact_20_session_evidence_unavailable",)
    origin_inventory = _document(
        read,
        receipt["evidence_origin_inventory_ref"],
        version="myquant.v17.v4.forward-evidence-origin-inventory.v1",
        label="evidence origin inventory",
    )
    factor_inventory = _document(
        read,
        receipt["existing_factor_inventory_ref"],
        version=FACTOR_INVENTORY_VERSION,
        label="existing factor inventory",
    )
    observation_run = _document(
        read,
        receipt["observation_run_ref"],
        version=OBSERVATION_RUN_VERSION,
        label="observation run",
    )
    request_ref = observation_run["request_ref"]
    request = _document(
        read,
        request_ref,
        version=REQUEST_VERSION,
        label="forward run request",
    )
    decision_session = receipt["decision_session"]
    if any(
        document["decision_session"] != decision_session
        for document in (
            origin_inventory,
            factor_inventory,
            observation_run,
            request,
        )
    ):
        _fail("evaluation closure decision-session mismatch")
    if _instant(receipt["recorded_at"], label="receipt.recorded_at") < _instant(
        receipt["cutoff"],
        label="receipt.cutoff",
    ):
        _fail("evaluation receipt was recorded before its cutoff")
    _reference_equal(
        origin_inventory["request_ref"],
        request_ref,
        label="origin inventory request",
    )
    _reference_equal(
        factor_inventory["request_ref"],
        request_ref,
        label="factor inventory request",
    )
    if (
        request["request_profile"] != "FORWARD_EVIDENCE"
        or observation_run["global_activation_state"] != "INACTIVE"
        or observation_run["run_state"] != "FORWARD_EVIDENCE_ACTIVE"
        or observation_run["research_runtime_default"] is not False
        or observation_run["formal_activation_eligible"] is not False
        or any(
            observation_run[field] is not False
            for field in ("broker", "execution", "order", "trade")
        )
    ):
        _fail("observation run is not an inactive Forward Evidence run")
    factor_rows = [
        row for row in factor_inventory["factors"] if row.get("factor_name") == factor_name
    ]
    if len(factor_rows) != 1:
        return None, None, None, ("matching_factor_inventory_row_unavailable",)
    factor_inventory_row = factor_rows[0]
    if factor_inventory_row["lifecycle"] != "ACTIVE":
        return None, None, None, ("matching_active_factor_unavailable",)
    factor_set_ref = factor_inventory_row["factor_ref"]
    if factor_set_ref["artifact_version"] != FACTOR_SET_VERSION:
        _fail("factor inventory does not bind a research factor set")
    request_factor_ref = _single_reference(
        request["factor_refs"],
        version=FACTOR_SET_VERSION,
        label="request.factor_refs",
    )
    _reference_equal(
        factor_set_ref,
        request_factor_ref,
        label="request factor set",
    )
    factor_set = _document(
        read,
        factor_set_ref,
        version=FACTOR_SET_VERSION,
        label="research factor set",
    )
    selected_factor = _factor_definition(factor_set, factor_name)
    if (
        factor_inventory_row["definition_sha256"] != selected_factor["definition_sha256"]
        or lineage["factor_definition_sha256"] != selected_factor["definition_sha256"]
        or lineage["factor_set_sha256"] != factor_set["semantic_sha256"]
    ):
        _fail("factor definition or factor-set lineage mismatch")
    observation_refs = factor_inventory_row["exposure_observation_refs"]
    matching_observations = [
        reference
        for reference in observation_refs
        if reference.get("artifact_version") == FACTOR_OBSERVATION_VERSION
        and reference.get("cutoff", "") <= receipt["cutoff"]
    ]
    if len(matching_observations) != 1:
        return None, None, None, ("matching_factor_observation_unavailable",)
    factor_observation_ref = matching_observations[0]
    factor_observation = _document(
        read,
        factor_observation_ref,
        version=FACTOR_OBSERVATION_VERSION,
        label="factor observation",
    )
    _reference_equal(
        factor_observation["factor_ref"],
        factor_set_ref,
        label="factor observation factor set",
    )
    _reference_equal(
        factor_observation["request_ref"],
        request_ref,
        label="factor observation request",
    )
    _, _, source_series = _source_locator_and_bundle(
        read,
        factor_observation=factor_observation,
        factor_set_ref=factor_set_ref,
        request=request,
    )
    decision_session = factor_observation["decision_session"]
    if decision_session != receipt["decision_session"]:
        _fail("factor observation decision-session mismatch")
    if (
        factor_set["effective_from_session"] > decision_session
        or factor_set["cutoff"] > factor_observation["cutoff"]
    ):
        _fail("factor set is not effective at the factor observation cutoff")
    source_series_sha = hashlib.sha256(canonical_bytes(source_series)).hexdigest()
    adapter_policy_sha = hashlib.sha256(
        read_packaged_asset(V4_FACTOR_EVIDENCE_ADAPTER_POLICY_PATH)
    ).hexdigest()
    stratum = FactorSampleStratum(
        adapter_policy_byte_sha256=adapter_policy_sha,
        factor_definition_sha256=selected_factor["definition_sha256"],
        factor_implementation_sha256=selected_factor["implementation_sha256"],
        factor_name=factor_name,
        factor_set_sha256=factor_set["semantic_sha256"],
        horizon_sessions=HORIZON_SESSIONS,
        market_calendar_sha256=market_calendar_sha256,
        quant_policy_sha256=lineage["quant_policy_sha256"],
        source_lineage_series_sha256=source_series_sha,
        strategy_id=receipt["strategy_id"],
    )
    if (
        receipt["completeness"] != "COMPLETE"
        or receipt["execution_outcome"] != "SUCCEEDED"
        or receipt["blockers"]
    ):
        return stratum, None, None, ("evaluation_receipt_not_complete",)
    matching_origins = [
        row for row in origin_inventory["origins"] if row.get("lineage_key") == lineage
    ]
    if len(matching_origins) != 1 or receipt["origin_count"] != 1:
        return stratum, None, None, ("exact_single_origin_unavailable",)
    origin_row = matching_origins[0]
    label_ref = origin_row["canonical_evidence_ref"]
    if (
        origin_row["lineage_key_sha256"] != hashlib.sha256(canonical_bytes(lineage)).hexdigest()
        or origin_row["evidence_refs"] != [label_ref]
        or receipt["label_refs"] != [label_ref]
    ):
        _fail("origin inventory and receipt label binding mismatch")
    label = _document(
        read,
        label_ref,
        version=FORWARD_LABEL_VERSION,
        label="forward label",
    )
    _reference_equal(
        label["observation_run_ref"],
        receipt["observation_run_ref"],
        label="label observation run",
    )
    label_locator_ref = _single_reference(
        label["evidence_refs"],
        version=SOURCE_LOCATOR_VERSION,
        label="label.evidence_refs",
    )
    locator_ref = _single_reference(
        factor_observation["source_refs"],
        version=SOURCE_LOCATOR_VERSION,
        label="factor_observation.source_refs",
    )
    _reference_equal(label_locator_ref, locator_ref, label="label source locator")
    if (
        label["completeness"] != "COMPLETE"
        or label["horizon_sessions"] != HORIZON_SESSIONS
        or label["decision_session"] != decision_session
        or label["cost_basis_points"] != 20
    ):
        _fail("forward label contract mismatch")
    if factor_observation["completeness"] != "COMPLETE":
        return stratum, None, None, ("complete_factor_observation_unavailable",)
    origin_session = _session(label["decision_session"], label="label.decision_session")
    label_session = _session(label["label_session"], label="label.label_session")
    if origin_session not in session_index or label_session not in session_index:
        return stratum, None, None, ("market_calendar_session_unavailable",)
    start = session_index[origin_session]
    end = session_index[label_session]
    if end - start != HORIZON_SESSIONS:
        _fail("forward label is not an exact 20-session horizon")
    expected_window = list(sessions[start : end + 1])
    if label["shanghai_open_sessions"] != expected_window:
        _fail("forward label market-calendar window mismatch")
    label_available_at = _instant(label["cutoff"], label="label.cutoff")
    horizon_close = datetime.combine(
        date.fromisoformat(label_session),
        time(hour=15),
        tzinfo=SHANGHAI_TZ,
    )
    if (
        label_available_at.astimezone(SHANGHAI_TZ) < horizon_close
        or label_available_at > evaluation_cutoff
    ):
        return stratum, None, None, ("label_not_naturally_matured",)
    expected_label_lineage = hashlib.sha256(
        canonical_bytes(
            {
                "evidence_refs": label["evidence_refs"],
                "observation_run_ref": label["observation_run_ref"],
                "shanghai_open_sessions": label["shanghai_open_sessions"],
            }
        )
    ).hexdigest()
    if (
        label["source_lineage_sha256"] != expected_label_lineage
        or lineage["source_lineage_sha256"] != expected_label_lineage
    ):
        _fail("forward label source-lineage mismatch")
    factor_rows = factor_observation["observations"]
    label_rows = label["label_rows"]
    factor_symbols = [row["symbol"] for row in factor_rows]
    label_symbols = [row["symbol"] for row in label_rows]
    if (
        factor_symbols != sorted(factor_symbols)
        or len(factor_symbols) != len(set(factor_symbols))
        or label_symbols != sorted(label_symbols)
        or len(label_symbols) != len(set(label_symbols))
    ):
        _fail("factor observation or label symbol order is noncanonical")
    factor_values: dict[str, str] = {}
    for row in factor_rows:
        if row["status"] != "AVAILABLE" or row["value"] is None:
            _fail("COMPLETE factor observation contains an unavailable row")
        factor_values[row["symbol"]] = _canonical_decimal(
            row["value"],
            label=f"factor observation {row['symbol']}",
        )
    forward_returns: dict[str, str] = {}
    for row in label_rows:
        if row["status"] != "AVAILABLE":
            _fail("COMPLETE forward label contains an unavailable row")
        total_return = Decimal(
            _canonical_decimal(
                row["total_return"],
                label=f"forward label {row['symbol']}",
            )
        )
        market_return = Decimal(
            _canonical_decimal(
                row["market_return"],
                label=f"market return {row['symbol']}",
            )
        )
        industry_return = Decimal(
            _canonical_decimal(
                row["industry_return"],
                label=f"industry return {row['symbol']}",
            )
        )
        cost_adjusted = Decimal(
            _canonical_decimal(
                row["cost_adjusted_return"],
                label=f"cost-adjusted return {row['symbol']}",
            )
        )
        market_adjusted = Decimal(
            _canonical_decimal(
                row["market_adjusted_return"],
                label=f"market-adjusted return {row['symbol']}",
            )
        )
        industry_adjusted = Decimal(
            _canonical_decimal(
                row["industry_adjusted_return"],
                label=f"industry-adjusted return {row['symbol']}",
            )
        )
        if (
            cost_adjusted != total_return - Decimal("0.002")
            or market_adjusted != total_return - market_return
            or industry_adjusted != total_return - industry_return
        ):
            _fail("forward label return arithmetic mismatch")
        forward_returns[row["symbol"]] = str(row["total_return"])
    evidence_preimage = {
        "decision_session": origin_session,
        "factor_observation_ref": factor_observation_ref,
        "factor_set_ref": factor_set_ref,
        "forward_label_ref": label_ref,
        "observation_run_ref": receipt["observation_run_ref"],
        "request_ref": request_ref,
        "source_locator_ref": locator_ref,
    }
    evidence_sha = hashlib.sha256(canonical_bytes(evidence_preimage)).hexdigest()
    origin_id = f"v4-factor-origin-{evidence_sha[:32]}"
    comparable_symbol_count = len(set(factor_values).intersection(forward_returns))
    return (
        stratum,
        FactorOriginSample(
            decision_session=origin_session,
            evidence_lineage_sha256=evidence_sha,
            factor_values=factor_values,
            forward_returns=forward_returns,
            horizon_end_session=label_session,
            label_available_at=label["cutoff"],
            origin_id=origin_id,
        ),
        V4FactorOriginBinding(
            comparable_symbol_count=comparable_symbol_count,
            decision_session=origin_session,
            eligible_symbol_count=len(factor_values),
            evaluation_receipt_ref=_artifact_reference(
                read.root_ref,
                label="evaluation receipt ref",
            ),
            factor_implementation_sha256=selected_factor["implementation_sha256"],
            factor_observation_ref=_artifact_reference(
                factor_observation_ref,
                label="factor observation ref",
            ),
            factor_set_ref=_artifact_reference(
                factor_set_ref,
                label="factor set ref",
            ),
            forward_label_ref=_artifact_reference(
                label_ref,
                label="forward label ref",
            ),
            horizon_end_session=label_session,
            observation_run_ref=_artifact_reference(
                receipt["observation_run_ref"],
                label="observation run ref",
            ),
            origin_cutoff=factor_observation["cutoff"],
            origin_id=origin_id,
            request_ref=_artifact_reference(
                request_ref,
                label="request ref",
            ),
            source_locator_ref=_artifact_reference(
                locator_ref,
                label="source locator ref",
            ),
        ),
        (),
    )


def adapt_v4_factor_evidence(
    reads: Sequence[V4CompatibilityRead],
    *,
    evaluation_cutoff: str,
    factor_name: str,
    open_sessions: Sequence[str],
) -> V4FactorEvidenceAdaptation:
    """Adapt exact V4 evaluation closures into one V5 diagnostic sample stratum."""

    load_v4_factor_evidence_adapter_policy()
    cutoff = _instant(evaluation_cutoff, label="evaluation_cutoff")
    subject = _identifier(factor_name, label="factor_name")
    sessions, session_index, calendar_sha = _calendar(open_sessions)
    if isinstance(reads, (str, bytes)) or not isinstance(reads, Sequence):
        _fail("reads must be a sequence")
    if not reads:
        return V4FactorEvidenceAdaptation(
            blockers=("v4_evaluation_receipts_unavailable",),
            origin_bindings=(),
            origins=(),
            status=V4FactorAdaptationStatus.UNAVAILABLE,
            stratum=None,
        )
    stratum: FactorSampleStratum | None = None
    origins: dict[str, FactorOriginSample] = {}
    origin_bindings: dict[str, V4FactorOriginBinding] = {}
    blockers: set[str] = set()
    for read in reads:
        if not isinstance(read, V4CompatibilityRead):
            _fail("reads must contain V4CompatibilityRead values")
        if (
            read.compatibility_policy_byte_sha256 != V4_COMPATIBILITY_POLICY_BYTE_SHA256
            or read.predecessor_git_commit != V4_SOURCE_GIT_COMMIT
            or read.predecessor_package_manifest_byte_sha256 != V4_PACKAGE_MANIFEST_SHA256
            or read.predecessor_runtime_manifest_byte_sha256 != V4_RUNTIME_MANIFEST_SHA256
            or read.predecessor_protocol_version != "myquant.v17.v4"
        ):
            _fail("V4 compatibility read policy or predecessor identity mismatch")
        candidate_stratum, origin, binding, candidate_blockers = _origin_from_read(
            read,
            evaluation_cutoff=cutoff,
            factor_name=subject,
            sessions=sessions,
            session_index=session_index,
            market_calendar_sha256=calendar_sha,
        )
        blockers.update(candidate_blockers)
        if candidate_stratum is not None:
            if stratum is None:
                stratum = candidate_stratum
            elif stratum != candidate_stratum:
                _fail("V4 evidence closures do not share one exact sample stratum")
        if origin is not None:
            if binding is None or binding.origin_id != origin.origin_id:
                _fail("V4 factor origin binding is absent or inconsistent")
            existing = origins.get(origin.origin_id)
            if existing is not None and existing != origin:
                _fail("conflicting duplicate V4 factor origin")
            existing_binding = origin_bindings.get(binding.origin_id)
            if existing_binding is not None and existing_binding != binding:
                _fail("conflicting duplicate V4 factor origin binding")
            origins[origin.origin_id] = origin
            origin_bindings[binding.origin_id] = binding
    ordered_origins = tuple(
        sorted(
            origins.values(),
            key=lambda value: (value.decision_session, value.origin_id),
        )
    )
    if len({origin.decision_session for origin in ordered_origins}) != len(ordered_origins):
        _fail("duplicate V4 factor origin decision session")
    ordered_bindings = tuple(origin_bindings[origin.origin_id] for origin in ordered_origins)
    if ordered_origins:
        return V4FactorEvidenceAdaptation(
            blockers=tuple(sorted(blockers)),
            origin_bindings=ordered_bindings,
            origins=ordered_origins,
            status=V4FactorAdaptationStatus.ACCUMULATING,
            stratum=stratum,
        )
    if stratum is not None:
        return V4FactorEvidenceAdaptation(
            blockers=tuple(sorted(blockers | {"no_naturally_matured_origins"})),
            origin_bindings=(),
            origins=(),
            status=V4FactorAdaptationStatus.UNOBSERVED,
            stratum=stratum,
        )
    return V4FactorEvidenceAdaptation(
        blockers=tuple(sorted(blockers | {"v4_factor_evidence_unavailable"})),
        origin_bindings=(),
        origins=(),
        status=V4FactorAdaptationStatus.UNAVAILABLE,
        stratum=None,
    )


def build_factor_diagnostic_from_v4(
    reads: Sequence[V4CompatibilityRead],
    *,
    evaluation_cutoff: str,
    factor_name: str,
    open_sessions: Sequence[str],
) -> dict[str, Any]:
    """Build a V5 descriptive diagnostic from an exact V4 closure adaptation."""

    adaptation = adapt_v4_factor_evidence(
        reads,
        evaluation_cutoff=evaluation_cutoff,
        factor_name=factor_name,
        open_sessions=open_sessions,
    )
    if adaptation.status == V4FactorAdaptationStatus.UNAVAILABLE:
        return build_unavailable_factor_diagnostic(
            factor_name=factor_name,
            evaluation_cutoff=evaluation_cutoff,
            unavailable_prerequisites=adaptation.blockers,
        )
    if adaptation.stratum is None:
        _fail("observed V4 factor adaptation has no sample stratum")
    return build_factor_diagnostic(
        stratum=adaptation.stratum,
        evaluation_cutoff=evaluation_cutoff,
        open_sessions=open_sessions,
        origins=adaptation.origins,
    )


__all__ = [
    "V4ArtifactReference",
    "V4FactorAdaptationStatus",
    "V4FactorAdapterError",
    "V4FactorEvidenceAdaptation",
    "V4FactorOriginBinding",
    "adapt_v4_factor_evidence",
    "build_factor_diagnostic_from_v4",
]
