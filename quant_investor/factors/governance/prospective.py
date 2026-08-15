"""Closed prospective Factor artifacts built only by the trusted validation store.

This module intentionally exposes validators, not caller-authoritative builders.
The private ``_build_*`` helpers accept already replayed projections and are
used by :class:`FactorValidationStore`; raw arrays, pandas objects, metrics and
timestamps never cross the stable public boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Final

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes, seal_artifact

from .bootstrap import BLEND_W75_CONTROL, PROSPECTIVE_LANE
from .common import (
    BH_Q_CEILING,
    COST_BPS,
    COVERAGE_MINIMUM,
    CPCV_BLOCK_COUNT,
    CPCV_EMBARGO_OPEN_SESSIONS,
    CPCV_PATH_COUNT,
    CPCV_PURGE_OPEN_SESSIONS,
    CPCV_TEST_BLOCK_COUNT,
    DSR_FLOOR,
    LABEL_HORIZON_OPEN_SESSIONS,
    MIN_CLOSED_MONTH_ENDS,
    MIN_DAILY_RANKIC_SESSIONS,
    MIN_DISJOINT_COHORTS,
    PBO_CEILING,
    PBO_MIN_CONFIGURATIONS,
    PBO_SPLIT_COUNT,
    POSITIVE_PATH_RATIO_FLOOR,
    REDUNDANCY_CORRELATION_FLOOR,
    REDUNDANCY_MIN_OVERLAP,
    SHRINKAGE_PSEUDO_COUNT,
    SIGNAL_OPEN_SESSIONS,
    TOTAL_OPEN_SESSIONS,
    T_STAT_HURDLE,
    TURNOVER_CEILING,
    artifact_ref,
    business_identity,
    canonical_identifier,
    canonical_sessions,
    canonical_timestamp,
    decimal_text,
    decimal_value,
    exact_payload,
    observation_lineage_identity,
    require_sha256,
    validate_artifact_ref,
)
from .errors import FactorGovernanceError

PREREGISTRATION_KIND: Final = "factor.preregistration"
SELECTION_KIND: Final = "factor.configuration_selection"
SIGNAL_CAPTURE_KIND: Final = "factor.signal_capture"
OBSERVATION_KIND: Final = "factor.prospective_observation"

_NON_AUTHORIZING: Final = "NON_AUTHORIZING"
_MAX_PREREGISTRATION_BYTES: Final = 512 * 1024
_MAX_SELECTION_BYTES: Final = 128 * 1024
_MAX_CAPTURE_BYTES: Final = 64 * 1024
_MAX_OBSERVATION_BYTES: Final = 64 * 1024

_PREREGISTRATION_FIELDS: Final = {
    "preregistration_id",
    "lane",
    "stamp_source",
    "open_sessions",
    "signal_sessions",
    "maturity_sessions",
    "session_windows",
    "candidates",
    "exchange_calendar_ref",
    "implementation_manifest_ref",
    "source_decode_attestation_ref",
    "factor_validator_manifest_ref",
    "coverage_contract",
    "label_contract",
    "neutralization_contract",
    "maturity_contract",
    "validation_contract",
    "alternate_policy",
    "observation_policy",
    "authority",
}
_SESSION_WINDOW_FIELDS: Final = {
    "ordinal",
    "open_session",
    "opens_at_utc",
    "closes_at_utc",
    "next_opens_at_utc",
}
_CANDIDATE_FIELDS: Final = {
    "candidate_spec_id",
    "configuration_id",
    "factor_id",
    "implementation_id",
    "implementation_component_ref",
    "implementation_sha256",
    "family",
    "primitive",
    "direction",
    "formula",
    "normalized_expression",
    "parameters_json",
    "input_fields",
    "role",
}

_SELECTION_FIELDS: Final = {
    "selection_id",
    "preregistration_id",
    "first_signal_session",
    "source_decode_attestation_ref",
    "configuration_summary_rows",
    "selected_configurations",
    "selected_before_label",
    "label_inputs_used",
    "substitution_allowed",
    "selection_policy",
}
_SELECTION_SUMMARY_FIELDS: Final = {
    "configuration_id",
    "factor_id",
    "normalized_input_sha256",
    "signal_sha256",
    "finite_signal_count",
    "required_input_complete_count",
    "selection_complete_count",
    "denominator_count",
    "signal_coverage",
    "selection_coverage",
    "coverage_gate",
}
_SELECTED_CONFIGURATION_FIELDS: Final = {
    "primary_configuration_id",
    "selected_configuration_id",
    "selected_factor_id",
    "used_alternate",
}

_CAPTURE_FIELDS: Final = {
    "signal_capture_id",
    "observation_lineage_id",
    "previous_signal_capture_ref",
    "preregistration_id",
    "selection_id",
    "ordinal",
    "signal_session",
    "source_decode_attestation_ref",
    "pit_universe_count",
    "pit_universe_sha256",
    "configuration_rows",
    "coverage_minimum",
    "label_inputs_used",
    "unlisted_universe_weight",
    "backfill",
    "authority",
}
_CAPTURE_CONFIGURATION_FIELDS: Final = {
    "configuration_id",
    "factor_id",
    "signal_values_sha256",
    "finite_signal_count",
    "coverage_numerator_count",
    "coverage_denominator_count",
    "coverage",
    "coverage_gate",
    "portfolio_weights_sha256",
    "nonzero_weight_count",
    "long_weight",
    "short_weight",
    "gross_weight",
    "net_weight",
}

_OBSERVATION_FIELDS: Final = {
    "observation_id",
    "observation_lineage_id",
    "previous_observation_ref",
    "preregistration_id",
    "selection_id",
    "signal_capture_ref",
    "source_decode_attestation_ref",
    "ordinal",
    "signal_session",
    "label_start_session",
    "label_end_session",
    "label_formula",
    "neutralization_method",
    "coverage_minimum",
    "pit_universe_count",
    "pit_universe_sha256",
    "label_values_sha256",
    "label_finite_pair_count",
    "configuration_rows",
    "backfill",
    "substitution",
}
_OBSERVATION_CONFIGURATION_FIELDS: Final = {
    "configuration_id",
    "factor_id",
    "signal_values_sha256",
    "coverage_numerator_count",
    "coverage_denominator_count",
    "coverage",
    "coverage_gate",
    "complete_case_count",
    "held_nonzero_symbol_count",
    "held_missing_label_count",
    "gross_labeled_return_symbol_count",
    "gross_labeled_return",
    "rank_ic",
    "rank_ic_p_value",
    "valid_daily_rankic",
}


def _fail(detail: str, *, code: str = "FACTOR_VALIDATION_FAILED") -> FactorGovernanceError:
    return FactorGovernanceError(detail, code=code)


def _canonical_text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 for character in value)
    ):
        raise _fail(f"{label} is not canonical text")
    value.encode("utf-8", errors="strict")
    return value


def _canonical_json_text(value: Any, *, label: str) -> str:
    text = _canonical_text(value, label=label)
    try:
        parse_canonical_json_bytes(text.encode("utf-8"), label=label)
    except Exception as exc:
        raise _fail(f"{label} is not canonical JSON") from exc
    return text


def _ref_key(ref: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        ref[field]
        for field in (
            "kind",
            "contract_sha256",
            "artifact_id",
            "semantic_sha256",
            "byte_sha256",
        )
    )


def _check_size(envelope: Mapping[str, Any], maximum: int, *, label: str) -> None:
    if len(canonical_json_bytes(dict(envelope))) > maximum:
        raise _fail(
            f"{label} exceeds its canonical byte limit", code="ARTIFACT_SIZE_LIMIT_EXCEEDED"
        )


def _iso_time(value: Any, *, label: str) -> str:
    stamp = canonical_timestamp(value, label=label)
    return stamp


def _parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _decimal_text_in_range(
    value: Any,
    *,
    label: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> str:
    parsed = decimal_value(value, label=label, minimum=minimum, maximum=maximum)
    text = decimal_text(parsed, label=label)
    if value != text:
        raise _fail(f"{label} is not canonical 12-decimal text")
    return text


def _policy_payloads() -> dict[str, dict[str, Any]]:
    return {
        "coverage_contract": {
            "cadence": "DAILY_OPEN_SESSION",
            "minimum": decimal_text(COVERAGE_MINIMUM, label="coverage minimum"),
            "denominator": "EXACT_FULL_A_PIT_UNIVERSE_OBJECT_ROWS",
            "numerator": "FINITE_SELECTED_SIGNAL_ONLY",
            "missing_policy": "NO_FILL_NO_BACKFILL",
        },
        "label_contract": {
            "formula": "adj_close[t+30]/adj_close[t+1]-1",
            "price_field": "adj_close",
            "entry_offset_open_sessions": 1,
            "maturity_offset_open_sessions": LABEL_HORIZON_OPEN_SESSIONS,
            "coverage_gate": None,
            "fallback_allowed": False,
        },
        "neutralization_contract": {
            "method": "PIT_INDUSTRY_PLUS_LOG_TOTAL_MV_OLS",
            "intercept": True,
            "industry_exposure": "PIT_INDUSTRY_AT_T_ONE_HOT",
            "industry_columns": "LEXICOGRAPHIC_DROP_FIRST",
            "size_exposure": "log(total_mv[t])",
            "complete_cases_only": True,
            "minimum_cross_section": 20,
            "fill_or_backfill_allowed": False,
        },
        "maturity_contract": {
            "conjunctive": True,
            "minimum_valid_daily_rankic_sessions": MIN_DAILY_RANKIC_SESSIONS,
            "minimum_closed_calendar_month_end_observations": MIN_CLOSED_MONTH_ENDS,
            "minimum_disjoint_30_open_session_cohort_means": MIN_DISJOINT_COHORTS,
            "cohort_open_sessions": LABEL_HORIZON_OPEN_SESSIONS,
        },
        "validation_contract": {
            "t_statistic_strictly_greater_than": decimal_text(T_STAT_HURDLE),
            "dsr_minimum": decimal_text(DSR_FLOOR),
            "pbo_maximum": decimal_text(PBO_CEILING),
            "pbo_block_count": CPCV_BLOCK_COUNT,
            "pbo_split_count": PBO_SPLIT_COUNT,
            "pbo_minimum_configurations": PBO_MIN_CONFIGURATIONS,
            "bh_q_maximum": decimal_text(BH_Q_CEILING),
            "cpcv_block_count": CPCV_BLOCK_COUNT,
            "cpcv_test_block_count": CPCV_TEST_BLOCK_COUNT,
            "cpcv_path_count": CPCV_PATH_COUNT,
            "cpcv_purge_open_sessions": CPCV_PURGE_OPEN_SESSIONS,
            "cpcv_embargo_open_sessions": CPCV_EMBARGO_OPEN_SESSIONS,
            "positive_path_ratio_minimum": decimal_text(POSITIVE_PATH_RATIO_FLOOR),
            "turnover_maximum": decimal_text(TURNOVER_CEILING),
            "cost_bps": decimal_text(COST_BPS),
            "redundancy_absolute_correlation": decimal_text(REDUNDANCY_CORRELATION_FLOOR),
            "redundancy_minimum_overlap": REDUNDANCY_MIN_OVERLAP,
            "shrunk_ic_pseudo_count": decimal_text(SHRINKAGE_PSEUDO_COUNT),
            "weighting_method": "SHRUNK_IC_LARGEST_REMAINDER",
        },
    }


def _candidate_identity(row: Mapping[str, Any]) -> str:
    body = {field: row[field] for field in _CANDIDATE_FIELDS if field != "candidate_spec_id"}
    return business_identity("factor-candidate-spec", body)


def _candidate_row(value: Any, *, index: int) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _CANDIDATE_FIELDS:
        raise _fail(f"candidates[{index}] fields are not exact")
    row = dict(value)
    for field in (
        "candidate_spec_id",
        "configuration_id",
        "factor_id",
        "implementation_id",
        "family",
        "primitive",
    ):
        row[field] = canonical_identifier(row[field], label=f"candidates[{index}].{field}")
    if row["direction"] != "HIGHER_IS_BETTER":
        raise _fail("candidate direction is not installed")
    row["implementation_component_ref"] = validate_artifact_ref(
        row["implementation_component_ref"],
        label=f"candidates[{index}].implementation_component_ref",
        expected_kind="system.installed_component_manifest",
    )
    row["implementation_sha256"] = require_sha256(
        row["implementation_sha256"], label=f"candidates[{index}].implementation_sha256"
    )
    row["formula"] = _canonical_text(row["formula"], label=f"candidates[{index}].formula")
    row["normalized_expression"] = _canonical_json_text(
        row["normalized_expression"], label=f"candidates[{index}].normalized_expression"
    )
    row["parameters_json"] = _canonical_json_text(
        row["parameters_json"], label=f"candidates[{index}].parameters_json"
    )
    inputs = row["input_fields"]
    if type(inputs) is not list or not inputs:
        raise _fail("candidate input_fields must be a nonempty list")
    row["input_fields"] = [
        canonical_identifier(item, label=f"candidates[{index}].input_fields") for item in inputs
    ]
    if row["input_fields"] != sorted(set(row["input_fields"]), key=lambda item: item.encode()):
        raise _fail("candidate input_fields are not sorted and unique")
    role = row["role"]
    if role != "PRIMARY" and not (
        type(role) is str and role.startswith("ALTERNATE_FOR:") and len(role) > 14
    ):
        raise _fail("candidate role is invalid")
    if row["factor_id"] == BLEND_W75_CONTROL:
        raise _fail("CONTROL_ONLY factor is not selectable", code="CONTROL_ONLY_NOT_SELECTABLE")
    if row["candidate_spec_id"] != _candidate_identity(row):
        raise _fail("candidate business identity differs")
    return row


def _candidate_rows(values: Any) -> list[dict[str, Any]]:
    if type(values) is not list or not 2 <= len(values) <= 20:
        raise _fail("preregistration requires 2..20 installed candidates")
    rows = [_candidate_row(value, index=index) for index, value in enumerate(values)]
    if rows != sorted(rows, key=lambda row: row["configuration_id"].encode("utf-8")):
        raise _fail("candidate rows are not UTF-8 sorted")
    for field in ("candidate_spec_id", "configuration_id", "factor_id"):
        if len({row[field] for row in rows}) != len(rows):
            raise _fail(f"candidate {field} is duplicated")
    primaries = {row["configuration_id"]: row for row in rows if row["role"] == "PRIMARY"}
    if not 2 <= len(primaries) <= 10:
        raise _fail("preregistration requires 2..10 primary slots")
    alternate_for: set[str] = set()
    for row in rows:
        if row["role"] == "PRIMARY":
            continue
        primary_id = canonical_identifier(
            row["role"].split(":", 1)[1], label="alternate primary configuration"
        )
        primary = primaries.get(primary_id)
        if primary is None or primary_id in alternate_for or primary["family"] != row["family"]:
            raise _fail("candidate alternate policy differs")
        alternate_for.add(primary_id)
    return rows


def _session_windows(values: Any, sessions: Sequence[str]) -> list[dict[str, Any]]:
    if type(values) is not list or len(values) != TOTAL_OPEN_SESSIONS:
        raise _fail("session_windows must contain exactly 390 rows")
    rows: list[dict[str, Any]] = []
    previous_close: datetime | None = None
    for index, value in enumerate(values):
        if type(value) is not dict or set(value) != _SESSION_WINDOW_FIELDS:
            raise _fail("session window fields are not exact")
        row = dict(value)
        if row["ordinal"] != index or row["open_session"] != sessions[index]:
            raise _fail("session window ordinal/session differs")
        opens_at = _iso_time(row["opens_at_utc"], label="opens_at_utc")
        closes_at = _iso_time(row["closes_at_utc"], label="closes_at_utc")
        next_opens = _iso_time(row["next_opens_at_utc"], label="next_opens_at_utc")
        if not _parse_time(opens_at) < _parse_time(closes_at) < _parse_time(next_opens):
            raise _fail("session window timestamps are not strict")
        if previous_close is not None and previous_close >= _parse_time(opens_at):
            raise _fail("session windows overlap")
        previous_close = _parse_time(closes_at)
        rows.append(row)
    return rows


def _preregistration_identity(payload: Mapping[str, Any]) -> str:
    body = {
        field: payload[field] for field in _PREREGISTRATION_FIELDS if field != "preregistration_id"
    }
    return business_identity("factor-preregistration", body)


def _validate_preregistration_payload(
    envelope: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        payload["lane"] != PROSPECTIVE_LANE
        or payload["stamp_source"] != "TRUSTED_STORE_CLOCK"
        or payload["authority"] != _NON_AUTHORIZING
    ):
        raise _fail("preregistration lane/stamp/authority differs")
    sessions = canonical_sessions(payload["open_sessions"])
    if len(sessions) != TOTAL_OPEN_SESSIONS:
        raise _fail("preregistration requires exactly 390 open sessions")
    if (
        payload["signal_sessions"] != sessions[:SIGNAL_OPEN_SESSIONS]
        or payload["maturity_sessions"] != sessions[SIGNAL_OPEN_SESSIONS:]
    ):
        raise _fail("signal/maturity session projection differs")
    windows = _session_windows(payload["session_windows"], sessions)
    if _parse_time(envelope["created_at"]) >= _parse_time(windows[0]["opens_at_utc"]):
        raise _fail("preregistration was not sealed before the first open")
    _candidate_rows(payload["candidates"])
    for field, kind in (
        ("exchange_calendar_ref", "system.source_object"),
        ("implementation_manifest_ref", "system.source_object"),
        ("source_decode_attestation_ref", "factor.source_decode_attestation"),
        ("factor_validator_manifest_ref", "factor.validator_manifest"),
    ):
        validate_artifact_ref(payload[field], label=field, expected_kind=kind)
    policies = _policy_payloads()
    if any(payload[field] != value for field, value in policies.items()):
        raise _fail("preregistration policy contract differs")
    if payload["alternate_policy"] != {
        "maximum_per_primary": 1,
        "selection_deadline": "FIRST_SIGNAL_CAPTURE",
        "midstream_substitution_allowed": False,
    } or payload["observation_policy"] != {
        "signal_capture_count": 360,
        "label_observation_count": 360,
        "append_in_ordinal_order": True,
        "backfill_allowed": False,
    }:
        raise _fail("preregistration alternate/observation policy differs")
    if payload["preregistration_id"] != _preregistration_identity(payload):
        raise _fail("preregistration business identity differs")
    return dict(payload)


def _build_preregistration(
    *,
    open_sessions: Sequence[str],
    session_windows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    exchange_calendar_ref: Mapping[str, Any],
    implementation_manifest_ref: Mapping[str, Any],
    source_decode_attestation_ref: Mapping[str, Any],
    factor_validator_manifest_ref: Mapping[str, Any],
    trusted_at: str,
) -> dict[str, Any]:
    """Build a trusted preregistration from decoded Store projections."""

    sessions = canonical_sessions(list(open_sessions))
    candidate_rows = _candidate_rows(list(candidates))
    windows = _session_windows(list(session_windows), sessions)
    payload: dict[str, Any] = {
        "lane": PROSPECTIVE_LANE,
        "stamp_source": "TRUSTED_STORE_CLOCK",
        "open_sessions": sessions,
        "signal_sessions": sessions[:SIGNAL_OPEN_SESSIONS],
        "maturity_sessions": sessions[SIGNAL_OPEN_SESSIONS:],
        "session_windows": windows,
        "candidates": candidate_rows,
        "exchange_calendar_ref": validate_artifact_ref(
            dict(exchange_calendar_ref),
            label="exchange_calendar_ref",
            expected_kind="system.source_object",
        ),
        "implementation_manifest_ref": validate_artifact_ref(
            dict(implementation_manifest_ref),
            label="implementation_manifest_ref",
            expected_kind="system.source_object",
        ),
        "source_decode_attestation_ref": validate_artifact_ref(
            dict(source_decode_attestation_ref),
            label="source_decode_attestation_ref",
            expected_kind="factor.source_decode_attestation",
        ),
        "factor_validator_manifest_ref": validate_artifact_ref(
            dict(factor_validator_manifest_ref),
            label="factor_validator_manifest_ref",
            expected_kind="factor.validator_manifest",
        ),
        **_policy_payloads(),
        "alternate_policy": {
            "maximum_per_primary": 1,
            "selection_deadline": "FIRST_SIGNAL_CAPTURE",
            "midstream_substitution_allowed": False,
        },
        "observation_policy": {
            "signal_capture_count": 360,
            "label_observation_count": 360,
            "append_in_ordinal_order": True,
            "backfill_allowed": False,
        },
        "authority": _NON_AUTHORIZING,
    }
    payload["preregistration_id"] = _preregistration_identity(payload)
    envelope = seal_artifact(
        PREREGISTRATION_KIND,
        payload,
        created_at=canonical_timestamp(trusted_at, label="trusted_at"),
    )
    validate_preregistration(envelope)
    _check_size(envelope, _MAX_PREREGISTRATION_BYTES, label="preregistration")
    return envelope


def validate_preregistration(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    envelope, payload = exact_payload(
        document, kind=PREREGISTRATION_KIND, fields=_PREREGISTRATION_FIELDS
    )
    _validate_preregistration_payload(envelope, payload)
    _check_size(envelope, _MAX_PREREGISTRATION_BYTES, label="preregistration")
    return envelope


def _coverage_projection(row: Mapping[str, Any], *, label: str) -> tuple[int, int, str, str]:
    numerator = row["coverage_numerator_count"]
    denominator = row["coverage_denominator_count"]
    if (
        type(numerator) is not int
        or isinstance(numerator, bool)
        or type(denominator) is not int
        or isinstance(denominator, bool)
        or denominator <= 0
        or not 0 <= numerator <= denominator
    ):
        raise _fail(f"{label} coverage counts are invalid")
    coverage = Decimal(numerator) / Decimal(denominator)
    coverage_text = decimal_text(coverage, label=f"{label} coverage")
    gate = "PASSED" if coverage >= COVERAGE_MINIMUM else "FAILED"
    if row["coverage"] != coverage_text or row["coverage_gate"] != gate:
        raise _fail(f"{label} coverage projection differs")
    return numerator, denominator, coverage_text, gate


def _selection_summary_row(value: Any, candidate: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _SELECTION_SUMMARY_FIELDS:
        raise _fail("selection summary fields are not exact")
    row = dict(value)
    if (
        row["configuration_id"] != candidate["configuration_id"]
        or row["factor_id"] != candidate["factor_id"]
    ):
        raise _fail("selection summary candidate binding differs")
    for field in ("normalized_input_sha256", "signal_sha256"):
        row[field] = require_sha256(row[field], label=f"selection summary {field}")
    denominator = row["denominator_count"]
    counts = (
        row["finite_signal_count"],
        row["required_input_complete_count"],
        row["selection_complete_count"],
    )
    if (
        type(denominator) is not int
        or isinstance(denominator, bool)
        or denominator <= 0
        or any(
            type(item) is not int or isinstance(item, bool) or not 0 <= item <= denominator
            for item in counts
        )
        or row["selection_complete_count"] > row["required_input_complete_count"]
        or row["selection_complete_count"] > row["finite_signal_count"]
    ):
        raise _fail("selection summary counts are invalid")
    signal_coverage = Decimal(row["finite_signal_count"]) / Decimal(denominator)
    selection_coverage = Decimal(row["selection_complete_count"]) / Decimal(denominator)
    expected_signal = decimal_text(signal_coverage)
    expected_selection = decimal_text(selection_coverage)
    expected_gate = "PASSED" if selection_coverage >= COVERAGE_MINIMUM else "FAILED"
    if (
        row["signal_coverage"] != expected_signal
        or row["selection_coverage"] != expected_selection
        or row["coverage_gate"] != expected_gate
    ):
        raise _fail("selection summary coverage differs")
    return row


def _selected_rows(values: Any, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primaries = {row["configuration_id"]: row for row in candidates if row["role"] == "PRIMARY"}
    by_id = {row["configuration_id"]: row for row in candidates}
    alternates = {
        row["role"].split(":", 1)[1]: row
        for row in candidates
        if row["role"].startswith("ALTERNATE_FOR:")
    }
    if type(values) is not list or len(values) != len(primaries):
        raise _fail("selected configurations are incomplete")
    rows: list[dict[str, Any]] = []
    for value in values:
        if type(value) is not dict or set(value) != _SELECTED_CONFIGURATION_FIELDS:
            raise _fail("selected configuration fields are not exact")
        row = dict(value)
        primary_id = row["primary_configuration_id"]
        selected_id = row["selected_configuration_id"]
        if (
            primary_id not in primaries
            or selected_id not in by_id
            or type(row["used_alternate"]) is not bool
        ):
            raise _fail("selected configuration identity is invalid")
        expected_selected = (
            alternates.get(primary_id) if row["used_alternate"] else primaries[primary_id]
        )
        if expected_selected is None or selected_id != expected_selected["configuration_id"]:
            raise _fail("selected alternate binding differs")
        if row["selected_factor_id"] != expected_selected["factor_id"]:
            raise _fail("selected factor binding differs")
        rows.append(row)
    if [row["primary_configuration_id"] for row in rows] != sorted(primaries):
        raise _fail("selected configurations are not canonical")
    return rows


def _selection_identity(payload: Mapping[str, Any]) -> str:
    body = {field: payload[field] for field in _SELECTION_FIELDS if field != "selection_id"}
    return business_identity("factor-configuration-selection", body)


def _validate_selection_payload(
    payload: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> None:
    prereg_payload = preregistration["payload"]
    if (
        payload["preregistration_id"] != prereg_payload["preregistration_id"]
        or payload["first_signal_session"] != prereg_payload["signal_sessions"][0]
        or payload["selected_before_label"] is not True
        or payload["label_inputs_used"] is not False
        or payload["substitution_allowed"] is not False
        or payload["selection_policy"] != "PRIMARY_ELSE_SINGLE_PREREGISTERED_ALTERNATE"
    ):
        raise _fail("selection fixed policy differs")
    validate_artifact_ref(
        payload["source_decode_attestation_ref"],
        label="selection.source_decode_attestation_ref",
        expected_kind="factor.source_decode_attestation",
    )
    candidates = prereg_payload["candidates"]
    by_id = {row["configuration_id"]: row for row in candidates}
    summaries = payload["configuration_summary_rows"]
    if type(summaries) is not list or [row.get("configuration_id") for row in summaries] != sorted(
        by_id
    ):
        raise _fail("selection summaries are not canonical")
    normalized_summaries = [
        _selection_summary_row(row, by_id[row["configuration_id"]]) for row in summaries
    ]
    selected = _selected_rows(payload["selected_configurations"], candidates)
    coverage = {
        row["configuration_id"]: Decimal(row["selection_coverage"]) for row in normalized_summaries
    }
    alternate_by_primary = {
        row["role"].split(":", 1)[1]: row["configuration_id"]
        for row in candidates
        if row["role"].startswith("ALTERNATE_FOR:")
    }
    for row in selected:
        primary_id = row["primary_configuration_id"]
        if coverage[primary_id] >= COVERAGE_MINIMUM:
            expected_id = primary_id
            expected_alternate = False
        else:
            expected_id = alternate_by_primary.get(primary_id)
            if expected_id is None or coverage[expected_id] < COVERAGE_MINIMUM:
                raise _fail("primary and its registered alternate fail initial coverage")
            expected_alternate = True
        if (
            row["selected_configuration_id"] != expected_id
            or row["used_alternate"] is not expected_alternate
        ):
            raise _fail("selection coverage choice differs")
    if payload["selection_id"] != _selection_identity(payload):
        raise _fail("selection business identity differs")


def _build_configuration_selection(
    *,
    preregistration: Mapping[str, Any] | bytes,
    source_decode_attestation_ref: Mapping[str, Any],
    configuration_summary_rows: Sequence[Mapping[str, Any]],
    selected_configurations: Sequence[Mapping[str, Any]],
    trusted_at: str,
) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration)
    payload: dict[str, Any] = {
        "preregistration_id": prereg["payload"]["preregistration_id"],
        "first_signal_session": prereg["payload"]["signal_sessions"][0],
        "source_decode_attestation_ref": validate_artifact_ref(
            dict(source_decode_attestation_ref),
            label="source_decode_attestation_ref",
            expected_kind="factor.source_decode_attestation",
        ),
        "configuration_summary_rows": [dict(row) for row in configuration_summary_rows],
        "selected_configurations": [dict(row) for row in selected_configurations],
        "selected_before_label": True,
        "label_inputs_used": False,
        "substitution_allowed": False,
        "selection_policy": "PRIMARY_ELSE_SINGLE_PREREGISTERED_ALTERNATE",
    }
    payload["selection_id"] = _selection_identity(payload)
    artifact = seal_artifact(
        SELECTION_KIND, payload, created_at=canonical_timestamp(trusted_at, label="trusted_at")
    )
    validate_configuration_selection(artifact, preregistration=prereg)
    _check_size(artifact, _MAX_SELECTION_BYTES, label="configuration selection")
    return artifact


def validate_configuration_selection(
    document: Mapping[str, Any] | bytes,
    *,
    preregistration: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    envelope, payload = exact_payload(document, kind=SELECTION_KIND, fields=_SELECTION_FIELDS)
    prereg = validate_preregistration(preregistration)
    _validate_selection_payload(payload, prereg)
    if envelope["created_at"] < prereg["created_at"]:
        raise _fail("selection predates preregistration")
    _check_size(envelope, _MAX_SELECTION_BYTES, label="configuration selection")
    return envelope


def _capture_identity(payload: Mapping[str, Any]) -> str:
    body = {field: payload[field] for field in _CAPTURE_FIELDS if field != "signal_capture_id"}
    return business_identity("factor-signal-capture", body)


def _capture_configuration_row(value: Any, *, denominator: int) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _CAPTURE_CONFIGURATION_FIELDS:
        raise _fail("signal capture configuration fields are not exact")
    row = dict(value)
    canonical_identifier(row["configuration_id"], label="capture.configuration_id")
    canonical_identifier(row["factor_id"], label="capture.factor_id")
    for field in ("signal_values_sha256", "portfolio_weights_sha256"):
        require_sha256(row[field], label=f"capture.{field}")
    if (
        row["coverage_denominator_count"] != denominator
        or row["finite_signal_count"] != row["coverage_numerator_count"]
    ):
        raise _fail("capture signal coverage count differs")
    _coverage_projection(row, label="capture")
    count = row["nonzero_weight_count"]
    if type(count) is not int or isinstance(count, bool) or not 0 <= count <= denominator:
        raise _fail("capture nonzero weight count is invalid")
    long_weight = decimal_value(row["long_weight"], label="long_weight", minimum=Decimal("0"))
    short_weight = decimal_value(row["short_weight"], label="short_weight", maximum=Decimal("0"))
    gross_weight = decimal_value(row["gross_weight"], label="gross_weight", minimum=Decimal("0"))
    net_weight = decimal_value(row["net_weight"], label="net_weight")
    for field in ("long_weight", "short_weight", "gross_weight", "net_weight"):
        _decimal_text_in_range(row[field], label=field)
    if gross_weight != long_weight - short_weight or net_weight != long_weight + short_weight:
        raise _fail("capture sparse-weight summaries differ")
    return row


def _selected_configuration_map(selection: Mapping[str, Any]) -> dict[str, str]:
    return {
        row["selected_configuration_id"]: row["selected_factor_id"]
        for row in selection["payload"]["selected_configurations"]
    }


def _build_signal_capture(
    *,
    preregistration: Mapping[str, Any] | bytes,
    selection: Mapping[str, Any] | bytes,
    previous_signal_capture: Mapping[str, Any] | bytes | None,
    source_decode_attestation_ref: Mapping[str, Any],
    ordinal: int,
    pit_universe_count: int,
    pit_universe_sha256: str,
    configuration_rows: Sequence[Mapping[str, Any]],
    trusted_at: str,
) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration)
    selected = validate_configuration_selection(selection, preregistration=prereg)
    previous = (
        None
        if previous_signal_capture is None
        else _validate_signal_capture_intrinsic(
            previous_signal_capture,
            preregistration=prereg,
            selection=selected,
        )
    )
    payload: dict[str, Any] = {
        "observation_lineage_id": observation_lineage_identity(
            prereg["payload"]["preregistration_id"], selected["payload"]["selection_id"]
        ),
        "previous_signal_capture_ref": artifact_ref(previous) if previous is not None else None,
        "preregistration_id": prereg["payload"]["preregistration_id"],
        "selection_id": selected["payload"]["selection_id"],
        "ordinal": ordinal,
        "signal_session": prereg["payload"]["signal_sessions"][ordinal],
        "source_decode_attestation_ref": validate_artifact_ref(
            dict(source_decode_attestation_ref),
            label="source_decode_attestation_ref",
            expected_kind="factor.source_decode_attestation",
        ),
        "pit_universe_count": pit_universe_count,
        "pit_universe_sha256": require_sha256(pit_universe_sha256, label="pit_universe_sha256"),
        "configuration_rows": [dict(row) for row in configuration_rows],
        "coverage_minimum": decimal_text(COVERAGE_MINIMUM),
        "label_inputs_used": False,
        "unlisted_universe_weight": "EXACT_ZERO",
        "backfill": False,
        "authority": _NON_AUTHORIZING,
    }
    payload["signal_capture_id"] = _capture_identity(payload)
    artifact = seal_artifact(
        SIGNAL_CAPTURE_KIND, payload, created_at=canonical_timestamp(trusted_at, label="trusted_at")
    )
    validate_signal_capture(
        artifact,
        preregistration=prereg,
        selection=selected,
        previous_signal_capture=previous,
    )
    _check_size(artifact, _MAX_CAPTURE_BYTES, label="signal capture")
    return artifact


def _validate_capture_fixed_projection(
    payload: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> int:
    ordinal = payload["ordinal"]
    if (
        type(ordinal) is not int
        or isinstance(ordinal, bool)
        or not 0 <= ordinal < SIGNAL_OPEN_SESSIONS
    ):
        raise _fail("signal capture ordinal is invalid")
    expected_lineage = observation_lineage_identity(
        preregistration["payload"]["preregistration_id"],
        selection["payload"]["selection_id"],
    )
    if (
        payload["preregistration_id"] != preregistration["payload"]["preregistration_id"]
        or payload["selection_id"] != selection["payload"]["selection_id"]
        or payload["observation_lineage_id"] != expected_lineage
        or payload["signal_session"] != preregistration["payload"]["signal_sessions"][ordinal]
        or payload["label_inputs_used"] is not False
        or payload["unlisted_universe_weight"] != "EXACT_ZERO"
        or payload["backfill"] is not False
        or payload["authority"] != _NON_AUTHORIZING
        or payload["coverage_minimum"] != decimal_text(COVERAGE_MINIMUM)
    ):
        raise _fail("signal capture fixed projection differs")
    return ordinal


def _validate_capture_rows(payload: Mapping[str, Any], selection: Mapping[str, Any]) -> None:
    count = payload["pit_universe_count"]
    if type(count) is not int or isinstance(count, bool) or count <= 0:
        raise _fail("signal capture PIT universe count is invalid")
    require_sha256(payload["pit_universe_sha256"], label="pit_universe_sha256")
    selected_map = _selected_configuration_map(selection)
    rows = payload["configuration_rows"]
    if type(rows) is not list or [row.get("configuration_id") for row in rows] != sorted(
        selected_map
    ):
        raise _fail("capture selected configuration rows are not exact")
    normalized = [_capture_configuration_row(row, denominator=count) for row in rows]
    if len(normalized) != len(selected_map) or any(
        row["factor_id"] != selected_map[row["configuration_id"]] for row in normalized
    ):
        raise _fail("capture selected factor binding differs")


def _validate_signal_capture_intrinsic(
    document: Mapping[str, Any] | bytes,
    *,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    envelope, payload = exact_payload(document, kind=SIGNAL_CAPTURE_KIND, fields=_CAPTURE_FIELDS)
    ordinal = _validate_capture_fixed_projection(
        payload, preregistration=preregistration, selection=selection
    )
    validate_artifact_ref(
        payload["source_decode_attestation_ref"],
        label="capture.source_decode_attestation_ref",
        expected_kind="factor.source_decode_attestation",
    )
    if ordinal == 0:
        if payload["previous_signal_capture_ref"] is not None:
            raise _fail("first signal capture must be the lineage root")
    else:
        validate_artifact_ref(
            payload["previous_signal_capture_ref"],
            label="capture.previous_signal_capture_ref",
            expected_kind=SIGNAL_CAPTURE_KIND,
        )
    _validate_capture_rows(payload, selection)
    if payload["signal_capture_id"] != _capture_identity(payload):
        raise _fail("signal capture business identity differs")
    _check_size(envelope, _MAX_CAPTURE_BYTES, label="signal capture")
    return envelope


def _validate_capture_predecessor(
    envelope: Mapping[str, Any],
    *,
    previous_signal_capture: Mapping[str, Any] | bytes | None,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    payload = envelope["payload"]
    ordinal = payload["ordinal"]
    if ordinal == 0:
        if previous_signal_capture is not None:
            raise _fail("first signal capture must not receive a predecessor")
        return
    if previous_signal_capture is None:
        raise _fail("signal capture predecessor is required")
    previous = _validate_signal_capture_intrinsic(
        previous_signal_capture,
        preregistration=preregistration,
        selection=selection,
    )
    if (
        previous["payload"]["ordinal"] != ordinal - 1
        or payload["previous_signal_capture_ref"] != artifact_ref(previous)
        or previous["created_at"] > envelope["created_at"]
    ):
        raise _fail("signal capture predecessor differs")


def validate_signal_capture(
    document: Mapping[str, Any] | bytes,
    *,
    preregistration: Mapping[str, Any] | bytes,
    selection: Mapping[str, Any] | bytes,
    previous_signal_capture: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration)
    selected = validate_configuration_selection(selection, preregistration=prereg)
    return _validate_signal_capture_prevalidated(
        document,
        preregistration=prereg,
        selection=selected,
        previous_signal_capture=previous_signal_capture,
    )


def _validate_signal_capture_prevalidated(
    document: Mapping[str, Any] | bytes,
    *,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
    previous_signal_capture: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    """Validate one capture after its immutable parents were already replayed."""

    envelope = _validate_signal_capture_intrinsic(
        document,
        preregistration=preregistration,
        selection=selection,
    )
    _validate_capture_predecessor(
        envelope,
        previous_signal_capture=previous_signal_capture,
        preregistration=preregistration,
        selection=selection,
    )
    return envelope


def _observation_identity(payload: Mapping[str, Any]) -> str:
    body = {field: payload[field] for field in _OBSERVATION_FIELDS if field != "observation_id"}
    return business_identity("factor-prospective-observation", body)


def _observation_configuration_row(
    value: Any, *, denominator: int, selected_factor: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _OBSERVATION_CONFIGURATION_FIELDS:
        raise _fail("observation configuration fields are not exact")
    row = dict(value)
    canonical_identifier(row["configuration_id"], label="observation.configuration_id")
    if row["factor_id"] != selected_factor:
        raise _fail("observation factor binding differs")
    require_sha256(row["signal_values_sha256"], label="observation.signal_values_sha256")
    if row["coverage_denominator_count"] != denominator:
        raise _fail("observation coverage denominator differs")
    _coverage_projection(row, label="observation")
    integer_fields = (
        "complete_case_count",
        "held_nonzero_symbol_count",
        "held_missing_label_count",
        "gross_labeled_return_symbol_count",
    )
    if any(
        type(row[field]) is not int
        or isinstance(row[field], bool)
        or not 0 <= row[field] <= denominator
        for field in integer_fields
    ):
        raise _fail("observation evidence counts are invalid")
    if (
        row["held_missing_label_count"] > row["held_nonzero_symbol_count"]
        or row["gross_labeled_return_symbol_count"] > row["held_nonzero_symbol_count"]
    ):
        raise _fail("observation held-label counts differ")
    rank_ic = row["rank_ic"]
    rank_p = row["rank_ic_p_value"]
    gross_return = row["gross_labeled_return"]
    if (rank_ic is None) != (rank_p is None):
        raise _fail("observation RankIC pair is incomplete")
    if rank_ic is not None:
        _decimal_text_in_range(
            rank_ic, label="rank_ic", minimum=Decimal("-1"), maximum=Decimal("1")
        )
        _decimal_text_in_range(
            rank_p, label="rank_ic_p_value", minimum=Decimal("0"), maximum=Decimal("1")
        )
    if gross_return is not None:
        _decimal_text_in_range(gross_return, label="gross_labeled_return")
    expected_valid = (
        row["coverage_gate"] == "PASSED"
        and row["complete_case_count"] >= 20
        and rank_ic is not None
        and row["held_missing_label_count"] == 0
    )
    if (
        type(row["valid_daily_rankic"]) is not bool
        or row["valid_daily_rankic"] is not expected_valid
    ):
        raise _fail("observation valid RankIC flag differs")
    return row


def _build_observation(
    *,
    preregistration: Mapping[str, Any] | bytes,
    selection: Mapping[str, Any] | bytes,
    signal_capture: Mapping[str, Any] | bytes,
    previous_observation: Mapping[str, Any] | bytes | None,
    source_decode_attestation_ref: Mapping[str, Any],
    pit_universe_sha256: str,
    label_values_sha256: str,
    label_finite_pair_count: int,
    configuration_rows: Sequence[Mapping[str, Any]],
    trusted_at: str,
) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration)
    selected = validate_configuration_selection(selection, preregistration=prereg)
    capture = _validate_signal_capture_intrinsic(
        signal_capture,
        preregistration=prereg,
        selection=selected,
    )
    ordinal = capture["payload"]["ordinal"]
    payload: dict[str, Any] = {
        "observation_lineage_id": capture["payload"]["observation_lineage_id"],
        "previous_observation_ref": (
            artifact_ref(previous_observation) if previous_observation is not None else None
        ),
        "preregistration_id": prereg["payload"]["preregistration_id"],
        "selection_id": selected["payload"]["selection_id"],
        "signal_capture_ref": artifact_ref(capture),
        "source_decode_attestation_ref": validate_artifact_ref(
            dict(source_decode_attestation_ref),
            label="source_decode_attestation_ref",
            expected_kind="factor.source_decode_attestation",
        ),
        "ordinal": ordinal,
        "signal_session": capture["payload"]["signal_session"],
        "label_start_session": prereg["payload"]["open_sessions"][ordinal + 1],
        "label_end_session": prereg["payload"]["open_sessions"][
            ordinal + LABEL_HORIZON_OPEN_SESSIONS
        ],
        "label_formula": "adj_close[t+30]/adj_close[t+1]-1",
        "neutralization_method": "PIT_INDUSTRY_PLUS_LOG_TOTAL_MV_OLS",
        "coverage_minimum": decimal_text(COVERAGE_MINIMUM),
        "pit_universe_count": capture["payload"]["pit_universe_count"],
        "pit_universe_sha256": require_sha256(pit_universe_sha256, label="pit_universe_sha256"),
        "label_values_sha256": require_sha256(label_values_sha256, label="label_values_sha256"),
        "label_finite_pair_count": label_finite_pair_count,
        "configuration_rows": [dict(row) for row in configuration_rows],
        "backfill": False,
        "substitution": False,
    }
    payload["observation_id"] = _observation_identity(payload)
    artifact = seal_artifact(
        OBSERVATION_KIND, payload, created_at=canonical_timestamp(trusted_at, label="trusted_at")
    )
    validate_observation(
        artifact,
        preregistration=prereg,
        selection=selected,
        signal_capture=capture,
        previous_observation=previous_observation,
    )
    _check_size(artifact, _MAX_OBSERVATION_BYTES, label="prospective observation")
    return artifact


def _validate_observation_projection(
    payload: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
    capture: Mapping[str, Any],
) -> tuple[int, int]:
    ordinal = payload["ordinal"]
    if (
        type(ordinal) is not int
        or isinstance(ordinal, bool)
        or not 0 <= ordinal < SIGNAL_OPEN_SESSIONS
    ):
        raise _fail("observation ordinal is invalid")
    if (
        payload["signal_capture_ref"] != artifact_ref(capture)
        or capture["payload"]["ordinal"] != ordinal
        or payload["observation_lineage_id"] != capture["payload"]["observation_lineage_id"]
        or payload["preregistration_id"] != preregistration["payload"]["preregistration_id"]
        or payload["selection_id"] != selection["payload"]["selection_id"]
        or payload["signal_session"] != capture["payload"]["signal_session"]
        or payload["label_start_session"]
        != preregistration["payload"]["open_sessions"][ordinal + 1]
        or payload["label_end_session"]
        != preregistration["payload"]["open_sessions"][ordinal + LABEL_HORIZON_OPEN_SESSIONS]
        or payload["label_formula"] != "adj_close[t+30]/adj_close[t+1]-1"
        or payload["neutralization_method"] != "PIT_INDUSTRY_PLUS_LOG_TOTAL_MV_OLS"
        or payload["coverage_minimum"] != decimal_text(COVERAGE_MINIMUM)
        or payload["pit_universe_count"] != capture["payload"]["pit_universe_count"]
        or payload["pit_universe_sha256"] != capture["payload"]["pit_universe_sha256"]
        or payload["backfill"] is not False
        or payload["substitution"] is not False
    ):
        raise _fail("observation capture/session policy differs")
    validate_artifact_ref(
        payload["source_decode_attestation_ref"],
        label="observation.source_decode_attestation_ref",
        expected_kind="factor.source_decode_attestation",
    )
    require_sha256(payload["label_values_sha256"], label="label_values_sha256")
    finite_pairs = payload["label_finite_pair_count"]
    denominator = payload["pit_universe_count"]
    if (
        type(finite_pairs) is not int
        or isinstance(finite_pairs, bool)
        or not 0 <= finite_pairs <= denominator
    ):
        raise _fail("observation label finite-pair count is invalid")
    return ordinal, denominator


def _validate_observation_predecessor(
    envelope: Mapping[str, Any],
    *,
    previous_observation: Mapping[str, Any] | bytes | None,
) -> None:
    payload = envelope["payload"]
    ordinal = payload["ordinal"]
    if ordinal == 0:
        if previous_observation is not None or payload["previous_observation_ref"] is not None:
            raise _fail("first observation must be the lineage root")
        return
    if previous_observation is None:
        raise _fail("observation predecessor is required")
    previous_envelope, previous_payload = exact_payload(
        previous_observation, kind=OBSERVATION_KIND, fields=_OBSERVATION_FIELDS
    )
    if (
        previous_payload["ordinal"] != ordinal - 1
        or previous_payload["observation_lineage_id"] != payload["observation_lineage_id"]
        or payload["previous_observation_ref"] != artifact_ref(previous_envelope)
        or previous_envelope["created_at"] > envelope["created_at"]
    ):
        raise _fail("observation predecessor differs")


def _validate_observation_rows(
    payload: Mapping[str, Any], *, selection: Mapping[str, Any], denominator: int
) -> None:
    selected_map = _selected_configuration_map(selection)
    rows = payload["configuration_rows"]
    if type(rows) is not list or [row.get("configuration_id") for row in rows] != sorted(
        selected_map
    ):
        raise _fail("observation configuration rows are not canonical")
    normalized = [
        _observation_configuration_row(
            row,
            denominator=denominator,
            selected_factor=selected_map[row["configuration_id"]],
        )
        for row in rows
    ]
    if len(normalized) != len(selected_map):
        raise _fail("observation configuration rows are incomplete")


def validate_observation(
    document: Mapping[str, Any] | bytes,
    *,
    preregistration: Mapping[str, Any] | bytes,
    selection: Mapping[str, Any] | bytes,
    signal_capture: Mapping[str, Any] | bytes | None = None,
    previous_observation: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration)
    selected = validate_configuration_selection(selection, preregistration=prereg)
    return _validate_observation_prevalidated(
        document,
        preregistration=prereg,
        selection=selected,
        signal_capture=signal_capture,
        previous_observation=previous_observation,
    )


def _validate_observation_prevalidated(
    document: Mapping[str, Any] | bytes,
    *,
    preregistration: Mapping[str, Any],
    selection: Mapping[str, Any],
    signal_capture: Mapping[str, Any] | bytes | None = None,
    previous_observation: Mapping[str, Any] | bytes | None = None,
) -> dict[str, Any]:
    """Validate one observation after its immutable parents were already replayed."""

    envelope, payload = exact_payload(document, kind=OBSERVATION_KIND, fields=_OBSERVATION_FIELDS)
    if signal_capture is None:
        raise _fail("observation requires its signal capture")
    capture = _validate_signal_capture_intrinsic(
        signal_capture,
        preregistration=preregistration,
        selection=selection,
    )
    _, denominator = _validate_observation_projection(
        payload,
        preregistration=preregistration,
        selection=selection,
        capture=capture,
    )
    _validate_observation_predecessor(
        envelope,
        previous_observation=previous_observation,
    )
    _validate_observation_rows(payload, selection=selection, denominator=denominator)
    if payload["observation_id"] != _observation_identity(payload):
        raise _fail("observation business identity differs")
    _check_size(envelope, _MAX_OBSERVATION_BYTES, label="prospective observation")
    return envelope


__all__ = [
    "OBSERVATION_KIND",
    "PREREGISTRATION_KIND",
    "SELECTION_KIND",
    "SIGNAL_CAPTURE_KIND",
    "validate_configuration_selection",
    "validate_observation",
    "validate_preregistration",
    "validate_signal_capture",
]
