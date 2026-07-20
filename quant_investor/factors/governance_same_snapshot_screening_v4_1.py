"""Strict research-only Factor v4.1 same-snapshot screening contracts.

The module is deliberately pure.  It accepts already-bound JSON-like values,
revalidates both v4 ontologies/catalogs, delegates the complete 267-row
family-BH calculation to :mod:`governance_screening_v4`, and constructs two
non-authoritative research artifacts.  Filesystem publication is optional and
owned exclusively by :mod:`governance_private_bundle_io`.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date
from statistics import median
from typing import Any

from quant_investor.factors import governance_private_bundle_io as private_io
from quant_investor.factors import governance_screening_v4 as screening_v4

PROTOCOL_VERSION = "v4.1"
READINESS = "EXPLORATORY_SAME_SNAPSHOT_SCREENING_ONLY"

SCREENING_SCHEMA_VERSION = "factor-governance-same-snapshot-screening.v4.1"
CORRELATION_SCHEMA_VERSION = (
    "factor-governance-same-snapshot-correlation-diagnostic.v4.1"
)
READBACK_SCHEMA_VERSION = "factor-governance-same-snapshot-screening-readback.v4.1"
FUNDAMENTAL_RESOLVED_SCREENING_SCHEMA_VERSION = (
    "factor-governance-same-snapshot-screening.v4.1.1"
)
FUNDAMENTAL_RESOLVED_CORRELATION_SCHEMA_VERSION = (
    "factor-governance-same-snapshot-correlation-diagnostic.v4.1.1"
)
FUNDAMENTAL_RESOLVED_READBACK_SCHEMA_VERSION = (
    "factor-governance-same-snapshot-screening-readback.v4.1.1"
)
FULLY_RESOLVED_SCREENING_SCHEMA_VERSION = (
    "factor-governance-same-snapshot-screening.v4.1.2"
)
FULLY_RESOLVED_CORRELATION_SCHEMA_VERSION = (
    "factor-governance-same-snapshot-correlation-diagnostic.v4.1.2"
)
FULLY_RESOLVED_READBACK_SCHEMA_VERSION = (
    "factor-governance-same-snapshot-screening-readback.v4.1.2"
)

SCREENING_FILENAME = "same_snapshot_screening.v4_1.json"
CORRELATION_FILENAME = "same_snapshot_correlation_diagnostic.v4_1.json"
READBACK_FILENAME = "same_snapshot_screening_readback.v4_1.json"
BUNDLE_INPUT_FILENAMES = (SCREENING_FILENAME, CORRELATION_FILENAME)
BUNDLE_FILENAMES = (*BUNDLE_INPUT_FILENAMES, READBACK_FILENAME)
PRIVATE_ROOT_SUFFIX = (
    "reports",
    "factor_governance",
    "private",
    "v4_1_same_snapshot_screening",
)

BASE_PRIMITIVE_COUNT = 13
FORMAL_PRIMITIVE_COUNT = 18
BASE_CANDIDATE_COUNT = 230
NEW_CANDIDATE_COUNT = 37
FORMAL_CANDIDATE_COUNT = 267
TURNOVER_BLOCKED_COUNT = 2
FUNDAMENTAL_BLOCKED_COUNT = 8
BLOCKED_COUNT = TURNOVER_BLOCKED_COUNT + FUNDAMENTAL_BLOCKED_COUNT
NEW_EVALUATED_COUNT = NEW_CANDIDATE_COUNT - BLOCKED_COUNT

PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES = frozenset(
    {
        "alpha_turnover_low_20d",
        "alpha_turnover_low_60d",
    }
)
PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES = frozenset(
    {
        "alpha_growth_quality_profit_roa",
        "alpha_quality_low_debt_assets",
        "alpha_quality_value_cash_fcf",
        "alpha_vwap_cash_quality_160",
        "alpha_vwap_growth_profit_160",
        "alpha_vwap_low_debt_160",
        "alpha_vwap_quality_roa_160",
        "alpha_vwap_quality_roe_160",
    }
)
PREDECLARED_BLOCKED_CANDIDATE_NAMES = frozenset(
    PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES
    | PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES
)

ACCOUNTING_PROFILE_LEGACY = "legacy"
ACCOUNTING_PROFILE_FUNDAMENTAL_RESOLVED = "fundamental_resolved"
ACCOUNTING_PROFILE_FULLY_RESOLVED = "fully_resolved"
ACCOUNTING_PROFILES = frozenset(
    {
        ACCOUNTING_PROFILE_LEGACY,
        ACCOUNTING_PROFILE_FUNDAMENTAL_RESOLVED,
        ACCOUNTING_PROFILE_FULLY_RESOLVED,
    }
)

STATUS_EVALUATED = screening_v4.EVALUATED_STATUS
STATUS_COMPUTE_FAILED = screening_v4.COMPUTE_FAILED_STATUS
STATUS_TURNOVER_BLOCKED = "turnover_data_blocked"
STATUS_FUNDAMENTAL_BLOCKED = "fundamental_semantic_blocked"
_STATUSES = frozenset(
    {
        STATUS_EVALUATED,
        STATUS_COMPUTE_FAILED,
        STATUS_TURNOVER_BLOCKED,
        STATUS_FUNDAMENTAL_BLOCKED,
    }
)

CORRELATION_METRIC = "median_monthly_cross_sectional_abs_spearman"
CORRELATION_THRESHOLD = 0.70
MIN_VALID_SYMBOL_COUNT_PER_MONTH = 20
MIN_VALID_MONTH_COUNT = 3
EXPLORATORY_SHORTLIST_LIMIT = 10
REBALANCE_POLICY = (
    "closed_natural_month_last_open_session_after_warmup_and_horizon"
)

AUTHORITY_FLAGS = {
    "screening_authority": False,
    "family_bh_authority": False,
    "pre_holdout_evidence_proven": False,
    "qualification_authority": False,
    "maturity_authority": False,
    "walk_forward_authority": False,
    "cost_authority": False,
    "neutralization_authority": False,
    "stability_authority": False,
    "formal_dedup_authority": False,
    "runtime_equivalence_verified": False,
    "replay_authority": False,
    "proposal_eligible": False,
    "registry_entry_created": False,
    "production_apply_enabled": False,
    "new_risk_authorized": False,
}

SIDE_EFFECT_FLAGS = {
    "registry_mutation_performed": False,
    "production_wal_written": False,
    "budget_written": False,
    "proposal_created": False,
    "replay_created": False,
    "transaction_plan_created": False,
    "production_receipt_written": False,
    "production_pointer_written": False,
    "provider_call_performed": False,
    "portfolio_constructed": False,
    "broker_call_performed": False,
    "order_created": False,
    "trade_executed": False,
}

BLOCKERS = (
    "pre_holdout_evidence_not_proven",
    "authoritative_maturity_not_run",
    "purged_walk_forward_not_run",
    "cost_gate_not_run",
    "neutralization_not_run",
    "stability_gate_not_run",
    "formal_dedup_not_proven",
    "verified_v4_replay_not_run",
)

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}")

_EVALUATION_INPUT_FIELDS = frozenset(
    {
        "name",
        "status",
        "signal_sha256",
        "finite_ratio",
        "raw_p_value",
        "failure_reason",
    }
)
_MATRIX_CONTEXT_FIELDS = frozenset(
    {
        "date_axis_sha256",
        "symbol_axis_sha256",
        "eligibility_matrix_sha256",
        "session_scope_sha256",
        "calendar_sha256",
        "closed_month_end_dates",
        "closed_month_end_axis_sha256",
        "session_count",
        "symbol_count",
    }
)
_LEDGER_ROW_FIELDS = frozenset(
    {
        "name",
        "catalog_role",
        "definition_sha256",
        "family",
        "primitive_ids",
        "status",
        "signal_sha256",
        "finite_ratio",
        "diagnostic_signal_sha256",
        "diagnostic_signal_reproduced",
        "raw_p_value",
        "failure_reason",
        "screening_row",
        "row_semantic_sha256",
    }
)
_SCREENING_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "readiness",
        "base_ontology_sha256",
        "formal_ontology_sha256",
        "base_catalog_sha256",
        "formal_catalog_sha256",
        "matrix_context",
        "rebalance_policy",
        "source_bindings",
        "catalog_accounting",
        "status_accounting",
        "screening_evidence",
        "rows",
        "authority",
        "side_effects",
        "blockers",
        "screening_semantic_sha256",
    }
)
_RESOLVED_PROVENANCE_FIELDS = frozenset(
    {"accounting_profile", "input_resolution_semantic_sha256"}
)
_RESOLVED_SCREENING_FIELDS = frozenset(
    _SCREENING_FIELDS | _RESOLVED_PROVENANCE_FIELDS
)
_MONTHLY_INPUT_FIELDS = frozenset(
    {
        "left_name",
        "right_name",
        "month_end",
        "abs_spearman",
        "valid_common_symbol_count",
    }
)
_MONTHLY_PAIR_EVIDENCE_FIELDS = frozenset(
    {
        "left_name",
        "right_name",
        "left_signal_sha256",
        "right_signal_sha256",
        "abs_spearman_by_month",
        "valid_common_symbol_count_by_month",
        "row_semantic_sha256",
    }
)
_PAIR_FIELDS = frozenset(
    {
        "left_name",
        "right_name",
        "left_catalog_role",
        "right_catalog_role",
        "left_definition_sha256",
        "right_definition_sha256",
        "left_primitive_ids",
        "right_primitive_ids",
        "left_signal_sha256",
        "right_signal_sha256",
        "duplicate_primitive",
        "valid_month_count",
        "median_monthly_abs_spearman",
        "threshold_breached",
        "pair_semantic_sha256",
    }
)
_SHORTLIST_ROW_FIELDS = frozenset(
    {
        "rank",
        "name",
        "definition_sha256",
        "family",
        "primitive_ids",
        "signal_sha256",
        "finite_ratio",
        "raw_p_value",
        "bh_q_value",
        "comparison_pair_count",
        "max_median_monthly_abs_spearman",
        "diagnostic_duplicate_primitive",
        "diagnostic_high_correlation",
        "initial_weight",
        "row_semantic_sha256",
    }
)
_CORRELATION_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "readiness",
        "screening_semantic_sha256",
        "formal_catalog_sha256",
        "matrix_context",
        "correlation_contract",
        "evaluated_candidate_count",
        "evaluated_new_candidate_count",
        "expected_pair_count",
        "observed_pair_count",
        "complete_pair_inventory",
        "monthly_pair_evidence",
        "pair_summaries",
        "exploratory_new_candidate_shortlist",
        "authority",
        "side_effects",
        "blockers",
        "correlation_semantic_sha256",
    }
)
_RESOLVED_CORRELATION_FIELDS = frozenset(
    _CORRELATION_FIELDS | _RESOLVED_PROVENANCE_FIELDS
)
_ARTIFACT_BINDING_INPUT_FIELDS = frozenset(
    {"filename", "byte_sha256", "size_bytes", "mode", "uid", "nlink"}
)
_ARTIFACT_BINDING_FIELDS = frozenset(
    {*_ARTIFACT_BINDING_INPUT_FIELDS, "semantic_sha256"}
)
_READBACK_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "cycle_id",
        "run_id",
        "readiness",
        "artifact_bindings",
        "screening_semantic_sha256",
        "correlation_semantic_sha256",
        "catalog_accounting",
        "status_accounting",
        "expected_pair_count",
        "shortlist_count",
        "authority",
        "side_effects",
        "blockers",
        "report_semantic_sha256",
    }
)
_RESOLVED_READBACK_FIELDS = frozenset(
    _READBACK_FIELDS | _RESOLVED_PROVENANCE_FIELDS
)


class FactorGovernanceSameSnapshotScreeningV4_1Error(ValueError):
    """Raised when same-snapshot research evidence fails closed."""


def canonical_json_bytes_v4_1(value: Any) -> bytes:
    try:
        return screening_v4.canonical_json_bytes(value)
    except Exception as exc:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def canonical_file_bytes_v4_1(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes_v4_1(value) + b"\n"


def semantic_sha256_v4_1(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes_v4_1(value)).hexdigest()


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    if field in payload:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"self-hash field already exists: {field}"
        )
    payload[field] = semantic_sha256_v4_1(payload)
    return payload


def _exact(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"{label} must be an object"
        )
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"{label} field names must be strings"
        )
    missing = sorted(fields - set(payload))
    extra = sorted(set(payload) - fields)
    if missing or extra:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"{label} fields mismatch: missing={missing};extra={extra}"
        )
    return payload


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"{label} must be an exact non-empty string"
        )
    return value


def _safe_id(value: Any, label: str) -> str:
    text = _text(value, label)
    if _SAFE_ID_RE.fullmatch(text) is None or ".." in text:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"{label} must be one safe path segment"
        )
    return text


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"{label} must be an exact lowercase SHA-256"
        )
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _canonical_float(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"{label} must be a canonical finite float"
        )
    if not minimum <= value <= maximum:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"{label} must be in [{minimum}, {maximum}]"
        )
    return value


def _builder_float(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"{label} must be numeric"
        )
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"{label} must be finite in [{minimum}, {maximum}]"
        )
    return result


def _matrix_context(value: Any) -> dict[str, Any]:
    row = _exact(value, _MATRIX_CONTEXT_FIELDS, "matrix_context")
    raw_closed_dates = row["closed_month_end_dates"]
    if not isinstance(raw_closed_dates, list) or not raw_closed_dates:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "closed_month_end_dates must be a non-empty list"
        )
    closed_dates = [
        _month_end(item, f"closed_month_end_dates[{index}]")
        for index, item in enumerate(raw_closed_dates)
    ]
    if closed_dates != sorted(closed_dates) or len(closed_dates) != len(set(closed_dates)):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "closed_month_end_dates must be sorted and distinct"
        )
    if len({item[:7] for item in closed_dates}) != len(closed_dates):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "closed_month_end_dates must contain one endpoint per natural month"
        )
    closed_axis_sha = _sha(
        row["closed_month_end_axis_sha256"], "closed month-end axis SHA"
    )
    if closed_axis_sha != semantic_sha256_v4_1(closed_dates):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "closed month-end axis SHA mismatch"
        )
    return {
        "date_axis_sha256": _sha(row["date_axis_sha256"], "date axis SHA"),
        "symbol_axis_sha256": _sha(row["symbol_axis_sha256"], "symbol axis SHA"),
        "eligibility_matrix_sha256": _sha(
            row["eligibility_matrix_sha256"], "eligibility matrix SHA"
        ),
        "session_scope_sha256": _sha(
            row["session_scope_sha256"], "session scope SHA"
        ),
        "calendar_sha256": _sha(row["calendar_sha256"], "calendar SHA"),
        "closed_month_end_dates": closed_dates,
        "closed_month_end_axis_sha256": closed_axis_sha,
        "session_count": _integer(row["session_count"], "session_count", minimum=1),
        "symbol_count": _integer(row["symbol_count"], "symbol_count", minimum=1),
    }


def _catalog_context(
    *,
    base_ontology: Mapping[str, Any],
    formal_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], set[str]]:
    try:
        normalized_base_ontology = screening_v4.validate_primitive_ontology_v4(
            base_ontology
        )
        normalized_formal_ontology = screening_v4.validate_primitive_ontology_v4(
            formal_ontology
        )
        normalized_base_catalog = screening_v4.validate_candidate_catalog_v4(
            base_catalog, ontology=normalized_base_ontology
        )
        normalized_formal_catalog = screening_v4.validate_candidate_catalog_v4(
            formal_catalog, ontology=normalized_formal_ontology
        )
    except Exception as exc:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"v4 ontology/catalog validation failed: {exc}"
        ) from exc

    if len(normalized_base_ontology["primitives"]) != BASE_PRIMITIVE_COUNT:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "base ontology must contain exactly 13 primitives"
        )
    if len(normalized_formal_ontology["primitives"]) != FORMAL_PRIMITIVE_COUNT:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "formal ontology must contain exactly 18 primitives"
        )
    base_rows = normalized_base_catalog["candidates"]
    formal_rows = normalized_formal_catalog["candidates"]
    if len(base_rows) != BASE_CANDIDATE_COUNT or len(formal_rows) != FORMAL_CANDIDATE_COUNT:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "candidate accounting must be exactly 267=230+37"
        )
    base_by_name = {row["name"]: row for row in base_rows}
    formal_by_name = {row["name"]: row for row in formal_rows}
    missing = sorted(set(base_by_name) - set(formal_by_name))
    drifted = sorted(
        name
        for name, row in base_by_name.items()
        if name in formal_by_name
        and canonical_json_bytes_v4_1(row)
        != canonical_json_bytes_v4_1(formal_by_name[name])
    )
    if missing or drifted:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"formal catalog does not preserve base definitions: missing={missing};drifted={drifted}"
        )
    new_names = set(formal_by_name) - set(base_by_name)
    if len(new_names) != NEW_CANDIDATE_COUNT:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "formal catalog must add exactly 37 distinct candidates"
        )
    missing_predeclared = sorted(PREDECLARED_BLOCKED_CANDIDATE_NAMES - new_names)
    if missing_predeclared:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "formal catalog is missing predeclared input-blocked candidates: "
            f"{missing_predeclared}"
        )
    return (
        normalized_base_ontology,
        normalized_formal_ontology,
        normalized_base_catalog,
        normalized_formal_catalog,
        new_names,
    )


def _normalize_diagnostic_signals(
    value: Any,
    *,
    expected_names: set[str],
) -> dict[str, str]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "diagnostic_signal_sha256_by_name must be an object"
        )
    result = {
        _text(name, "diagnostic signal candidate name"): _sha(
            digest, f"diagnostic signal SHA for {name}"
        )
        for name, digest in value.items()
    }
    if set(result) != expected_names:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "diagnostic signal inventory must match the exact 27 new evaluated candidates"
        )
    return result


def _normalize_evaluations(
    value: Any,
    *,
    formal_catalog: Mapping[str, Any],
    new_names: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    if not isinstance(value, list):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "evaluations must be a list"
        )
    by_name: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        row = _exact(raw, _EVALUATION_INPUT_FIELDS, f"evaluations[{index}]")
        name = _text(row["name"], f"evaluations[{index}].name")
        if name in by_name:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                "evaluation candidate names must be distinct"
            )
        status = row["status"]
        if status not in _STATUSES:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                f"unsupported evaluation status for {name}"
            )
        is_evaluated = status == STATUS_EVALUATED
        if is_evaluated:
            signal_sha = _sha(row["signal_sha256"], f"{name} signal SHA")
            finite_ratio = _builder_float(
                row["finite_ratio"], f"{name} finite_ratio", minimum=0.0, maximum=1.0
            )
            if finite_ratio <= 0.0:
                raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                    f"{name} finite_ratio must be positive"
                )
            raw_p = _builder_float(
                row["raw_p_value"], f"{name} raw_p_value", minimum=0.0, maximum=1.0
            )
            if row["failure_reason"] is not None:
                raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                    f"{name} evaluated row failure_reason must be null"
                )
            failure_reason = None
        else:
            if any(row[field] is not None for field in ("signal_sha256", "finite_ratio", "raw_p_value")):
                raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                    f"{name} non-evaluated row signal/statistic fields must be null"
                )
            signal_sha = None
            finite_ratio = None
            raw_p = None
            failure_reason = _text(
                row["failure_reason"], f"{name} failure_reason"
            )
        if status in {STATUS_TURNOVER_BLOCKED, STATUS_FUNDAMENTAL_BLOCKED} and name not in new_names:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                "turnover/fundamental blockers are permitted only for new candidates"
            )
        if name in new_names and status == STATUS_COMPUTE_FAILED:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                "the 27 no-label-computable new candidates must be evaluated; compute_failed is separate base-catalog evidence"
            )
        by_name[name] = {
            "name": name,
            "status": status,
            "signal_sha256": signal_sha,
            "finite_ratio": finite_ratio,
            "raw_p_value": raw_p,
            "failure_reason": failure_reason,
        }

    catalog_names = [row["name"] for row in formal_catalog["candidates"]]
    if set(by_name) != set(catalog_names) or len(by_name) != FORMAL_CANDIDATE_COUNT:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "evaluations must contain exactly one row for every formal catalog candidate"
        )
    rows = [by_name[name] for name in catalog_names]
    counts = {status: sum(row["status"] == status for row in rows) for status in _STATUSES}
    new_evaluated_names = {
        row["name"]
        for row in rows
        if row["name"] in new_names and row["status"] == STATUS_EVALUATED
    }
    turnover_blocked_names = {
        row["name"]
        for row in rows
        if row["status"] == STATUS_TURNOVER_BLOCKED
    }
    fundamental_blocked_names = {
        row["name"]
        for row in rows
        if row["status"] == STATUS_FUNDAMENTAL_BLOCKED
    }
    original_diagnostic_names = new_names - PREDECLARED_BLOCKED_CANDIDATE_NAMES
    profile: str | None = None
    if (
        new_evaluated_names == original_diagnostic_names
        and turnover_blocked_names
        == PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES
        and fundamental_blocked_names
        == PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES
    ):
        profile = ACCOUNTING_PROFILE_LEGACY
    elif (
        new_evaluated_names
        == original_diagnostic_names
        | PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES
        and turnover_blocked_names
        == PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES
        and not fundamental_blocked_names
    ):
        profile = ACCOUNTING_PROFILE_FUNDAMENTAL_RESOLVED
    elif (
        new_evaluated_names == new_names
        and not turnover_blocked_names
        and not fundamental_blocked_names
    ):
        profile = ACCOUNTING_PROFILE_FULLY_RESOLVED
    if profile is None:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "new-candidate statuses must match exactly one predefined accounting "
            "profile: legacy 27/2/8, fundamental-resolved 35/2/0, or fully "
            "resolved 37/0/0"
        )
    if sum(counts.values()) != FORMAL_CANDIDATE_COUNT:
        raise AssertionError("evaluation accounting invariant violated")
    accounting = {
        "evaluated_count": counts[STATUS_EVALUATED],
        "compute_failed_count": counts[STATUS_COMPUTE_FAILED],
        "blocked_count": (
            counts[STATUS_TURNOVER_BLOCKED]
            + counts[STATUS_FUNDAMENTAL_BLOCKED]
        ),
        "turnover_data_blocked_count": counts[STATUS_TURNOVER_BLOCKED],
        "fundamental_semantic_blocked_count": counts[STATUS_FUNDAMENTAL_BLOCKED],
        "bh_denominator_count": FORMAL_CANDIDATE_COUNT,
    }
    return rows, accounting, profile


def _build_screening_evidence(
    *,
    formal_ontology: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    source_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    evaluations = []
    for row in rows:
        evaluated = row["status"] == STATUS_EVALUATED
        evaluations.append(
            {
                "name": row["name"],
                "evaluation_status": (
                    screening_v4.EVALUATED_STATUS
                    if evaluated
                    else screening_v4.COMPUTE_FAILED_STATUS
                ),
                "raw_p_value": row["raw_p_value"] if evaluated else None,
                "failure_reason": (
                    None
                    if evaluated
                    else f"same_snapshot:{row['status']}:{row['failure_reason']}"
                ),
            }
        )
    try:
        return screening_v4.build_screening_evidence_v4(
            ontology=formal_ontology,
            catalog=formal_catalog,
            evaluations=evaluations,
            source_bindings=source_bindings,
            statistic_contract={
                "raw_p_method": screening_v4.RAW_P_METHOD,
                "fdr_method": screening_v4.FDR_METHOD,
                "q": screening_v4.FDR_Q,
            },
        )
    except Exception as exc:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"full-catalog family BH build failed: {exc}"
        ) from exc


def build_same_snapshot_screening_v4_1(
    *,
    cycle_id: str,
    base_ontology: Mapping[str, Any],
    formal_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
    diagnostic_signal_sha256_by_name: Mapping[str, Any],
    matrix_context: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    input_resolution_semantic_sha256: str | None = None,
) -> dict[str, Any]:
    """Build exact 267-row same-snapshot ledger and full-family BH evidence."""

    normalized_cycle = _safe_id(cycle_id, "cycle_id")
    (
        normalized_base_ontology,
        normalized_formal_ontology,
        normalized_base_catalog,
        normalized_formal_catalog,
        new_names,
    ) = _catalog_context(
        base_ontology=base_ontology,
        formal_ontology=formal_ontology,
        base_catalog=base_catalog,
        formal_catalog=formal_catalog,
    )
    evaluation_rows, status_accounting, accounting_profile = _normalize_evaluations(
        list(evaluations), formal_catalog=normalized_formal_catalog, new_names=new_names
    )
    if accounting_profile == ACCOUNTING_PROFILE_LEGACY:
        if input_resolution_semantic_sha256 is not None:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                "legacy accounting profile must not carry input-resolution proof"
            )
        normalized_input_resolution_sha = None
    else:
        normalized_input_resolution_sha = _sha(
            input_resolution_semantic_sha256,
            "input-resolution semantic SHA",
        )
    original_diagnostic_names = new_names - PREDECLARED_BLOCKED_CANDIDATE_NAMES
    diagnostic_signals = _normalize_diagnostic_signals(
        diagnostic_signal_sha256_by_name, expected_names=original_diagnostic_names
    )
    normalized_matrix_context = _matrix_context(matrix_context)
    evidence = _build_screening_evidence(
        formal_ontology=normalized_formal_ontology,
        formal_catalog=normalized_formal_catalog,
        rows=evaluation_rows,
        source_bindings=source_bindings,
    )
    if normalized_matrix_context["calendar_sha256"] != evidence["source_bindings"][
        "calendar_sha256"
    ]:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "matrix-context calendar SHA differs from screening source binding"
        )
    candidates = {row["name"]: row for row in normalized_formal_catalog["candidates"]}
    bh_rows = {row["name"]: row for row in evidence["rows"]}
    ledger_rows: list[dict[str, Any]] = []
    for row in evaluation_rows:
        candidate = candidates[row["name"]]
        diagnostic_signal_sha = diagnostic_signals.get(row["name"])
        diagnostic_reproduced = diagnostic_signal_sha is not None
        if diagnostic_reproduced and diagnostic_signal_sha != row["signal_sha256"]:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                f"same-snapshot signal differs from upstream no-label diagnostic for {row['name']}"
            )
        ledger_rows.append(
            _seal(
                {
                    "name": row["name"],
                    "catalog_role": "new" if row["name"] in new_names else "base",
                    "definition_sha256": candidate["definition_sha256"],
                    "family": candidate["family"],
                    "primitive_ids": list(candidate["primitive_ids"]),
                    "status": row["status"],
                    "signal_sha256": row["signal_sha256"],
                    "finite_ratio": row["finite_ratio"],
                    "diagnostic_signal_sha256": diagnostic_signal_sha,
                    "diagnostic_signal_reproduced": diagnostic_reproduced,
                    "raw_p_value": row["raw_p_value"],
                    "failure_reason": row["failure_reason"],
                    "screening_row": copy.deepcopy(bh_rows[row["name"]]),
                },
                "row_semantic_sha256",
            )
        )
    blockers = list(BLOCKERS)
    if status_accounting["compute_failed_count"]:
        blockers.append(
            f"screening_compute_failed:{status_accounting['compute_failed_count']}"
        )
    turnover_blocked_count = status_accounting["turnover_data_blocked_count"]
    fundamental_blocked_count = status_accounting[
        "fundamental_semantic_blocked_count"
    ]
    if turnover_blocked_count or fundamental_blocked_count:
        blockers.append(
            "data_blocked:"
            f"turnover={turnover_blocked_count}:"
            f"fundamental={fundamental_blocked_count}"
        )
    payload = {
        "schema_version": (
            SCREENING_SCHEMA_VERSION
            if accounting_profile == ACCOUNTING_PROFILE_LEGACY
            else FUNDAMENTAL_RESOLVED_SCREENING_SCHEMA_VERSION
            if accounting_profile == ACCOUNTING_PROFILE_FUNDAMENTAL_RESOLVED
            else FULLY_RESOLVED_SCREENING_SCHEMA_VERSION
        ),
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": normalized_cycle,
        "readiness": READINESS,
        "base_ontology_sha256": normalized_base_ontology["semantic_sha256"],
        "formal_ontology_sha256": normalized_formal_ontology["semantic_sha256"],
        "base_catalog_sha256": normalized_base_catalog["semantic_sha256"],
        "formal_catalog_sha256": normalized_formal_catalog["semantic_sha256"],
        "matrix_context": normalized_matrix_context,
        "rebalance_policy": REBALANCE_POLICY,
        "source_bindings": copy.deepcopy(evidence["source_bindings"]),
        "catalog_accounting": {
            "base_candidate_count": BASE_CANDIDATE_COUNT,
            "new_candidate_count": NEW_CANDIDATE_COUNT,
            "candidate_count": FORMAL_CANDIDATE_COUNT,
        },
        "status_accounting": status_accounting,
        "screening_evidence": evidence,
        "rows": ledger_rows,
        "authority": dict(AUTHORITY_FLAGS),
        "side_effects": dict(SIDE_EFFECT_FLAGS),
        "blockers": blockers,
    }
    if accounting_profile != ACCOUNTING_PROFILE_LEGACY:
        payload.update(
            {
                "accounting_profile": accounting_profile,
                "input_resolution_semantic_sha256": normalized_input_resolution_sha,
            }
        )
    return _seal(payload, "screening_semantic_sha256")


def validate_same_snapshot_screening_v4_1(
    value: Mapping[str, Any],
    *,
    base_ontology: Mapping[str, Any],
    formal_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild every ledger/BH/derived field and reject resealed drift."""

    if not isinstance(value, Mapping):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "same-snapshot screening must be an object"
        )
    schema_version = value.get("schema_version")
    if schema_version == SCREENING_SCHEMA_VERSION:
        payload = _exact(value, _SCREENING_FIELDS, "same-snapshot screening")
        input_resolution_semantic_sha256 = None
    elif schema_version in {
        FUNDAMENTAL_RESOLVED_SCREENING_SCHEMA_VERSION,
        FULLY_RESOLVED_SCREENING_SCHEMA_VERSION,
    }:
        payload = _exact(
            value, _RESOLVED_SCREENING_FIELDS, "resolved same-snapshot screening"
        )
        input_resolution_semantic_sha256 = _sha(
            payload["input_resolution_semantic_sha256"],
            "input-resolution semantic SHA",
        )
        expected_profile = (
            ACCOUNTING_PROFILE_FUNDAMENTAL_RESOLVED
            if schema_version == FUNDAMENTAL_RESOLVED_SCREENING_SCHEMA_VERSION
            else ACCOUNTING_PROFILE_FULLY_RESOLVED
        )
        if payload["accounting_profile"] != expected_profile:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                "resolved screening accounting_profile is invalid"
            )
    else:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "same-snapshot screening schema/protocol/readiness mismatch"
        )
    if (
        payload["protocol_version"] != PROTOCOL_VERSION
        or payload["readiness"] != READINESS
    ):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "same-snapshot screening schema/protocol/readiness mismatch"
        )
    _sha(payload["screening_semantic_sha256"], "screening semantic SHA")
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error("rows must be a list")
    evaluations: list[dict[str, Any]] = []
    diagnostic_signals: dict[str, str] = {}
    for index, raw in enumerate(rows):
        row = _exact(raw, _LEDGER_ROW_FIELDS, f"rows[{index}]")
        _sha(row["row_semantic_sha256"], f"rows[{index}] semantic SHA")
        evaluations.append(
            {field: copy.deepcopy(row[field]) for field in _EVALUATION_INPUT_FIELDS}
        )
        if row["diagnostic_signal_reproduced"] is True:
            diagnostic_signals[row["name"]] = row["diagnostic_signal_sha256"]
        elif row["diagnostic_signal_reproduced"] is not False:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                "diagnostic_signal_reproduced must be boolean"
            )
    expected = build_same_snapshot_screening_v4_1(
        cycle_id=payload["cycle_id"],
        base_ontology=base_ontology,
        formal_ontology=formal_ontology,
        base_catalog=base_catalog,
        formal_catalog=formal_catalog,
        evaluations=evaluations,
        diagnostic_signal_sha256_by_name=diagnostic_signals,
        matrix_context=payload["matrix_context"],
        source_bindings=payload["source_bindings"],
        input_resolution_semantic_sha256=input_resolution_semantic_sha256,
    )
    try:
        screening_v4.validate_screening_evidence_v4(
            payload["screening_evidence"],
            ontology=formal_ontology,
            catalog=formal_catalog,
        )
    except Exception as exc:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"nested full-family BH validation failed: {exc}"
        ) from exc
    if canonical_json_bytes_v4_1(payload) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "same-snapshot screening differs from exact recomputation"
        )
    return expected


def _month_end(value: Any, label: str) -> str:
    text = _text(value, label)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"{label} must be ISO YYYY-MM-DD"
        ) from exc
    if parsed.isoformat() != text:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"{label} must be canonical ISO date"
        )
    return text


def _expected_pairs(
    evaluated_names: Sequence[str], new_evaluated_names: set[str]
) -> set[tuple[str, str]]:
    names = sorted(evaluated_names)
    return {
        (left, right)
        for index, left in enumerate(names)
        for right in names[index + 1 :]
        if left in new_evaluated_names or right in new_evaluated_names
    }


def build_correlation_diagnostic_v4_1(
    *,
    cycle_id: str,
    base_ontology: Mapping[str, Any],
    formal_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
    screening: Mapping[str, Any],
    monthly_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build complete evaluated-pair correlation diagnostics and shortlist."""

    normalized_screening = validate_same_snapshot_screening_v4_1(
        screening,
        base_ontology=base_ontology,
        formal_ontology=formal_ontology,
        base_catalog=base_catalog,
        formal_catalog=formal_catalog,
    )
    normalized_cycle = _safe_id(cycle_id, "cycle_id")
    if normalized_cycle != normalized_screening["cycle_id"]:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "correlation/screening cycle_id mismatch"
        )
    _, _, _, normalized_formal_catalog, _ = _catalog_context(
        base_ontology=base_ontology,
        formal_ontology=formal_ontology,
        base_catalog=base_catalog,
        formal_catalog=formal_catalog,
    )
    ledger = {row["name"]: row for row in normalized_screening["rows"]}
    evaluated_names = sorted(
        name for name, row in ledger.items() if row["status"] == STATUS_EVALUATED
    )
    new_evaluated_names = {
        name
        for name in evaluated_names
        if ledger[name]["catalog_role"] == "new"
    }
    expected_pairs = _expected_pairs(evaluated_names, new_evaluated_names)
    if not isinstance(monthly_rows, Sequence) or isinstance(
        monthly_rows, (str, bytes, bytearray)
    ):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "monthly_rows must be a sequence"
        )
    closed_month_dates = list(
        normalized_screening["matrix_context"]["closed_month_end_dates"]
    )
    month_index = {month: index for index, month in enumerate(closed_month_dates)}
    monthly_by_pair: dict[tuple[str, str], dict[str, list[float | int | None]]] = {
        pair: {
            "abs_spearman_by_month": [None] * len(closed_month_dates),
            "valid_common_symbol_count_by_month": [None] * len(closed_month_dates),
        }
        for pair in sorted(expected_pairs)
    }
    seen_keys: set[tuple[str, str, str]] = set()
    observed_pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(monthly_rows):
        row = _exact(raw, _MONTHLY_INPUT_FIELDS, f"monthly_rows[{index}]")
        left = _text(row["left_name"], f"monthly_rows[{index}].left_name")
        right = _text(row["right_name"], f"monthly_rows[{index}].right_name")
        pair = (left, right)
        if left >= right or pair not in expected_pairs:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                "monthly correlation pair must be canonical, evaluated, and include a new candidate"
            )
        month = _month_end(row["month_end"], f"monthly_rows[{index}].month_end")
        if month not in normalized_screening["matrix_context"]["closed_month_end_dates"]:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                "monthly correlation date is not a bound closed natural-month endpoint"
            )
        key = (left, right, month)
        if key in seen_keys:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                "monthly correlation pair/month rows must be unique"
            )
        seen_keys.add(key)
        observed_pairs.add(pair)
        count = _integer(
            row["valid_common_symbol_count"],
            f"monthly_rows[{index}].valid_common_symbol_count",
            minimum=MIN_VALID_SYMBOL_COUNT_PER_MONTH,
        )
        corr = _builder_float(
            row["abs_spearman"],
            f"monthly_rows[{index}].abs_spearman",
            minimum=0.0,
            maximum=1.0,
        )
        month_offset = month_index[month]
        monthly_by_pair[pair]["abs_spearman_by_month"][month_offset] = corr
        monthly_by_pair[pair]["valid_common_symbol_count_by_month"][month_offset] = count
    complete_pairs = {
        pair
        for pair, rows in monthly_by_pair.items()
        if sum(value is not None for value in rows["abs_spearman_by_month"])
        >= MIN_VALID_MONTH_COUNT
    }
    if complete_pairs != expected_pairs:
        missing = sorted(expected_pairs - complete_pairs)
        extra = sorted(complete_pairs - expected_pairs)
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"correlation pair inventory is incomplete: missing={missing};extra={extra}"
        )
    catalog_by_name = {
        row["name"]: row for row in normalized_formal_catalog["candidates"]
    }
    monthly_pair_evidence: list[dict[str, Any]] = []
    for left, right in sorted(expected_pairs):
        rows = monthly_by_pair[(left, right)]
        monthly_pair_evidence.append(
            _seal(
                {
                    "left_name": left,
                    "right_name": right,
                    "left_signal_sha256": ledger[left]["signal_sha256"],
                    "right_signal_sha256": ledger[right]["signal_sha256"],
                    "abs_spearman_by_month": list(rows["abs_spearman_by_month"]),
                    "valid_common_symbol_count_by_month": list(
                        rows["valid_common_symbol_count_by_month"]
                    ),
                },
                "row_semantic_sha256",
            )
        )
    pair_summaries: list[dict[str, Any]] = []
    for pair_row in monthly_pair_evidence:
        left = pair_row["left_name"]
        right = pair_row["right_name"]
        monthly_values = [
            value
            for value in pair_row["abs_spearman_by_month"]
            if value is not None
        ]
        if len(monthly_values) < MIN_VALID_MONTH_COUNT:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                f"correlation pair {left}/{right} has fewer than three valid months"
            )
        left_candidate = catalog_by_name[left]
        right_candidate = catalog_by_name[right]
        aggregate = float(median(monthly_values))
        pair_summaries.append(
            _seal(
                {
                    "left_name": left,
                    "right_name": right,
                    "left_catalog_role": ledger[left]["catalog_role"],
                    "right_catalog_role": ledger[right]["catalog_role"],
                    "left_definition_sha256": left_candidate["definition_sha256"],
                    "right_definition_sha256": right_candidate["definition_sha256"],
                    "left_primitive_ids": list(left_candidate["primitive_ids"]),
                    "right_primitive_ids": list(right_candidate["primitive_ids"]),
                    "left_signal_sha256": ledger[left]["signal_sha256"],
                    "right_signal_sha256": ledger[right]["signal_sha256"],
                    "duplicate_primitive": (
                        left_candidate["primitive_ids"]
                        == right_candidate["primitive_ids"]
                    ),
                    "valid_month_count": len(monthly_values),
                    "median_monthly_abs_spearman": aggregate,
                    "threshold_breached": aggregate >= CORRELATION_THRESHOLD,
                },
                "pair_semantic_sha256",
            )
        )
    screening_rows = {
        row["name"]: row["screening_row"] for row in normalized_screening["rows"]
    }
    eligible_shortlist: list[dict[str, Any]] = []
    for name in sorted(new_evaluated_names):
        comparisons = [
            row
            for row in pair_summaries
            if name in (row["left_name"], row["right_name"])
        ]
        expected_comparisons = len(evaluated_names) - 1
        if len(comparisons) != expected_comparisons:
            raise AssertionError("new-candidate comparison completeness invariant violated")
        duplicate = any(row["duplicate_primitive"] for row in comparisons)
        high_corr = any(row["threshold_breached"] for row in comparisons)
        bh_row = screening_rows[name]
        if bh_row["bh_pass"] and not duplicate and not high_corr:
            candidate = catalog_by_name[name]
            eligible_shortlist.append(
                {
                    "name": name,
                    "definition_sha256": candidate["definition_sha256"],
                    "family": candidate["family"],
                    "primitive_ids": list(candidate["primitive_ids"]),
                    "signal_sha256": ledger[name]["signal_sha256"],
                    "finite_ratio": ledger[name]["finite_ratio"],
                    "raw_p_value": bh_row["raw_p_value"],
                    "bh_q_value": bh_row["bh_q_value"],
                    "comparison_pair_count": len(comparisons),
                    "max_median_monthly_abs_spearman": max(
                        row["median_monthly_abs_spearman"] for row in comparisons
                    ),
                    "diagnostic_duplicate_primitive": duplicate,
                    "diagnostic_high_correlation": high_corr,
                    "initial_weight": 0.0,
                }
            )
    eligible_shortlist.sort(
        key=lambda row: (
            row["bh_q_value"],
            row["raw_p_value"],
            row["max_median_monthly_abs_spearman"],
            row["name"],
        )
    )
    shortlist = [
        _seal({"rank": rank, **row}, "row_semantic_sha256")
        for rank, row in enumerate(
            eligible_shortlist[:EXPLORATORY_SHORTLIST_LIMIT], start=1
        )
    ]
    resolved_profile = normalized_screening.get("accounting_profile")
    payload = {
        "schema_version": (
            CORRELATION_SCHEMA_VERSION
            if resolved_profile is None
            else FUNDAMENTAL_RESOLVED_CORRELATION_SCHEMA_VERSION
            if resolved_profile == ACCOUNTING_PROFILE_FUNDAMENTAL_RESOLVED
            else FULLY_RESOLVED_CORRELATION_SCHEMA_VERSION
        ),
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": normalized_cycle,
        "readiness": READINESS,
        "screening_semantic_sha256": normalized_screening[
            "screening_semantic_sha256"
        ],
        "formal_catalog_sha256": normalized_screening["formal_catalog_sha256"],
        "matrix_context": copy.deepcopy(normalized_screening["matrix_context"]),
        "correlation_contract": {
            "metric": CORRELATION_METRIC,
            "threshold": CORRELATION_THRESHOLD,
            "minimum_valid_symbol_count_per_month": MIN_VALID_SYMBOL_COUNT_PER_MONTH,
            "minimum_valid_month_count": MIN_VALID_MONTH_COUNT,
            "rebalance_policy": REBALANCE_POLICY,
            "axes_bound": True,
            "eligibility_mask_bound": True,
            "formal_dedup_authority": False,
        },
        "evaluated_candidate_count": len(evaluated_names),
        "evaluated_new_candidate_count": len(new_evaluated_names),
        "expected_pair_count": len(expected_pairs),
        "observed_pair_count": len(complete_pairs),
        "complete_pair_inventory": True,
        "monthly_pair_evidence": monthly_pair_evidence,
        "pair_summaries": pair_summaries,
        "exploratory_new_candidate_shortlist": shortlist,
        "authority": dict(AUTHORITY_FLAGS),
        "side_effects": dict(SIDE_EFFECT_FLAGS),
        "blockers": list(BLOCKERS),
    }
    if resolved_profile is not None:
        payload.update(
            {
                "accounting_profile": resolved_profile,
                "input_resolution_semantic_sha256": normalized_screening[
                    "input_resolution_semantic_sha256"
                ],
            }
        )
    return _seal(payload, "correlation_semantic_sha256")


def validate_correlation_diagnostic_v4_1(
    value: Mapping[str, Any],
    *,
    base_ontology: Mapping[str, Any],
    formal_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
    screening: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute pair inventory, summaries, threshold flags, and shortlist."""

    if not isinstance(value, Mapping):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "correlation diagnostic must be an object"
        )
    schema_version = value.get("schema_version")
    if schema_version == CORRELATION_SCHEMA_VERSION:
        payload = _exact(value, _CORRELATION_FIELDS, "correlation diagnostic")
    elif schema_version in {
        FUNDAMENTAL_RESOLVED_CORRELATION_SCHEMA_VERSION,
        FULLY_RESOLVED_CORRELATION_SCHEMA_VERSION,
    }:
        payload = _exact(
            value,
            _RESOLVED_CORRELATION_FIELDS,
            "resolved correlation diagnostic",
        )
        _sha(
            payload["input_resolution_semantic_sha256"],
            "input-resolution semantic SHA",
        )
        expected_profile = (
            ACCOUNTING_PROFILE_FUNDAMENTAL_RESOLVED
            if schema_version == FUNDAMENTAL_RESOLVED_CORRELATION_SCHEMA_VERSION
            else ACCOUNTING_PROFILE_FULLY_RESOLVED
        )
        if payload["accounting_profile"] != expected_profile:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                "resolved correlation accounting_profile is invalid"
            )
    else:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "correlation schema/protocol/readiness mismatch"
        )
    if (
        payload["protocol_version"] != PROTOCOL_VERSION
        or payload["readiness"] != READINESS
    ):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "correlation schema/protocol/readiness mismatch"
        )
    _sha(payload["correlation_semantic_sha256"], "correlation semantic SHA")
    raw_monthly = payload["monthly_pair_evidence"]
    if not isinstance(raw_monthly, list):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "correlation monthly_pair_evidence must be a list"
        )
    closed_month_dates = _matrix_context(payload["matrix_context"])[
        "closed_month_end_dates"
    ]
    monthly_inputs: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_monthly):
        row = _exact(
            raw, _MONTHLY_PAIR_EVIDENCE_FIELDS, f"monthly_pair_evidence[{index}]"
        )
        _sha(
            row["row_semantic_sha256"],
            f"monthly_pair_evidence[{index}] semantic SHA",
        )
        left = _text(row["left_name"], f"monthly_pair_evidence[{index}].left_name")
        right = _text(row["right_name"], f"monthly_pair_evidence[{index}].right_name")
        _sha(
            row["left_signal_sha256"],
            f"monthly_pair_evidence[{index}].left_signal_sha256",
        )
        _sha(
            row["right_signal_sha256"],
            f"monthly_pair_evidence[{index}].right_signal_sha256",
        )
        raw_corrs = row["abs_spearman_by_month"]
        raw_counts = row["valid_common_symbol_count_by_month"]
        if (
            not isinstance(raw_corrs, list)
            or not isinstance(raw_counts, list)
            or len(raw_corrs) != len(closed_month_dates)
            or len(raw_counts) != len(closed_month_dates)
        ):
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                f"monthly_pair_evidence[{index}] arrays must match closed month axis"
            )
        for month_offset, month_end in enumerate(closed_month_dates):
            corr = raw_corrs[month_offset]
            count = raw_counts[month_offset]
            if corr is None and count is None:
                continue
            if corr is None or count is None:
                raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                    f"monthly_pair_evidence[{index}] nullable arrays must align"
                )
            monthly_inputs.append(
                {
                    "left_name": left,
                    "right_name": right,
                    "month_end": month_end,
                    "abs_spearman": _canonical_float(
                        corr,
                        f"monthly_pair_evidence[{index}].abs_spearman_by_month[{month_offset}]",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    "valid_common_symbol_count": _integer(
                        count,
                        f"monthly_pair_evidence[{index}].valid_common_symbol_count_by_month[{month_offset}]",
                        minimum=MIN_VALID_SYMBOL_COUNT_PER_MONTH,
                    ),
                }
            )
    for index, raw in enumerate(payload["pair_summaries"]):
        _exact(raw, _PAIR_FIELDS, f"pair_summaries[{index}]")
    for index, raw in enumerate(payload["exploratory_new_candidate_shortlist"]):
        _exact(raw, _SHORTLIST_ROW_FIELDS, f"shortlist[{index}]")
    expected = build_correlation_diagnostic_v4_1(
        cycle_id=payload["cycle_id"],
        base_ontology=base_ontology,
        formal_ontology=formal_ontology,
        base_catalog=base_catalog,
        formal_catalog=formal_catalog,
        screening=screening,
        monthly_rows=monthly_inputs,
    )
    if canonical_json_bytes_v4_1(payload) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "correlation diagnostic differs from exact recomputation"
        )
    return expected


def _artifact_semantic_sha(filename: str, artifact: Mapping[str, Any]) -> str:
    fields = {
        SCREENING_FILENAME: "screening_semantic_sha256",
        CORRELATION_FILENAME: "correlation_semantic_sha256",
    }
    try:
        return _sha(artifact[fields[filename]], f"{filename} semantic SHA")
    except KeyError as exc:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            f"unknown bundle artifact: {filename}"
        ) from exc


def _artifact_bindings(
    value: Any, *, artifacts: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "artifact_bindings must be a sequence"
        )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _exact(raw, _ARTIFACT_BINDING_INPUT_FIELDS, f"artifact_bindings[{index}]")
        filename = row["filename"]
        if filename not in BUNDLE_INPUT_FILENAMES or filename not in artifacts:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                "artifact binding filename is not canonical"
            )
        canonical = canonical_file_bytes_v4_1(artifacts[filename])
        normalized = {
            "filename": filename,
            "byte_sha256": _sha(row["byte_sha256"], "artifact byte SHA"),
            "semantic_sha256": _artifact_semantic_sha(filename, artifacts[filename]),
            "size_bytes": _integer(row["size_bytes"], "artifact size", minimum=1),
            "mode": _integer(row["mode"], "artifact mode"),
            "uid": _integer(row["uid"], "artifact uid"),
            "nlink": _integer(row["nlink"], "artifact nlink", minimum=1),
        }
        if normalized["byte_sha256"] != hashlib.sha256(canonical).hexdigest():
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                f"artifact byte SHA mismatch: {filename}"
            )
        if normalized["size_bytes"] != len(canonical):
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                f"artifact size mismatch: {filename}"
            )
        if normalized["mode"] != 0o600 or normalized["nlink"] != 1:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                f"artifact is not owner-private exact-once: {filename}"
            )
        rows.append(normalized)
    if [row["filename"] for row in rows] != list(BUNDLE_INPUT_FILENAMES):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "artifact bindings must follow exact canonical input order"
        )
    return rows


def build_readback_report_v4_1(
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
    base_ontology: Mapping[str, Any],
    formal_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(BUNDLE_INPUT_FILENAMES):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "readback requires exactly two canonical input artifacts"
        )
    screening = validate_same_snapshot_screening_v4_1(
        artifacts[SCREENING_FILENAME],
        base_ontology=base_ontology,
        formal_ontology=formal_ontology,
        base_catalog=base_catalog,
        formal_catalog=formal_catalog,
    )
    correlation = validate_correlation_diagnostic_v4_1(
        artifacts[CORRELATION_FILENAME],
        base_ontology=base_ontology,
        formal_ontology=formal_ontology,
        base_catalog=base_catalog,
        formal_catalog=formal_catalog,
        screening=screening,
    )
    if correlation["cycle_id"] != screening["cycle_id"]:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "bundle input cycle_id mismatch"
        )
    resolved_profile = screening.get("accounting_profile")
    payload = {
        "schema_version": (
            READBACK_SCHEMA_VERSION
            if resolved_profile is None
            else FUNDAMENTAL_RESOLVED_READBACK_SCHEMA_VERSION
            if resolved_profile == ACCOUNTING_PROFILE_FUNDAMENTAL_RESOLVED
            else FULLY_RESOLVED_READBACK_SCHEMA_VERSION
        ),
        "protocol_version": PROTOCOL_VERSION,
        "cycle_id": screening["cycle_id"],
        "run_id": _safe_id(run_id, "run_id"),
        "readiness": READINESS,
        "artifact_bindings": _artifact_bindings(
            artifact_bindings, artifacts=artifacts
        ),
        "screening_semantic_sha256": screening["screening_semantic_sha256"],
        "correlation_semantic_sha256": correlation["correlation_semantic_sha256"],
        "catalog_accounting": copy.deepcopy(screening["catalog_accounting"]),
        "status_accounting": copy.deepcopy(screening["status_accounting"]),
        "expected_pair_count": correlation["expected_pair_count"],
        "shortlist_count": len(correlation["exploratory_new_candidate_shortlist"]),
        "authority": dict(AUTHORITY_FLAGS),
        "side_effects": dict(SIDE_EFFECT_FLAGS),
        "blockers": list(BLOCKERS),
    }
    if resolved_profile is not None:
        payload.update(
            {
                "accounting_profile": resolved_profile,
                "input_resolution_semantic_sha256": screening[
                    "input_resolution_semantic_sha256"
                ],
            }
        )
    return _seal(payload, "report_semantic_sha256")


def validate_readback_report_v4_1(
    value: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    base_ontology: Mapping[str, Any],
    formal_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "same-snapshot readback report must be an object"
        )
    schema_version = value.get("schema_version")
    if schema_version == READBACK_SCHEMA_VERSION:
        payload = _exact(value, _READBACK_FIELDS, "same-snapshot readback report")
    elif schema_version in {
        FUNDAMENTAL_RESOLVED_READBACK_SCHEMA_VERSION,
        FULLY_RESOLVED_READBACK_SCHEMA_VERSION,
    }:
        payload = _exact(
            value,
            _RESOLVED_READBACK_FIELDS,
            "resolved same-snapshot readback report",
        )
        _sha(
            payload["input_resolution_semantic_sha256"],
            "input-resolution semantic SHA",
        )
        expected_profile = (
            ACCOUNTING_PROFILE_FUNDAMENTAL_RESOLVED
            if schema_version == FUNDAMENTAL_RESOLVED_READBACK_SCHEMA_VERSION
            else ACCOUNTING_PROFILE_FULLY_RESOLVED
        )
        if payload["accounting_profile"] != expected_profile:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                "resolved readback accounting_profile is invalid"
            )
    else:
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "same-snapshot readback report schema mismatch"
        )
    _sha(payload["report_semantic_sha256"], "readback report semantic SHA")
    raw_bindings = payload["artifact_bindings"]
    if not isinstance(raw_bindings, list):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "readback artifact_bindings must be a list"
        )
    binding_inputs = []
    for index, raw in enumerate(raw_bindings):
        row = _exact(raw, _ARTIFACT_BINDING_FIELDS, f"artifact_bindings[{index}]")
        binding_inputs.append(
            {field: copy.deepcopy(row[field]) for field in _ARTIFACT_BINDING_INPUT_FIELDS}
        )
    expected = build_readback_report_v4_1(
        run_id=payload["run_id"],
        artifacts=artifacts,
        artifact_bindings=binding_inputs,
        base_ontology=base_ontology,
        formal_ontology=formal_ontology,
        base_catalog=base_catalog,
        formal_catalog=formal_catalog,
    )
    if canonical_json_bytes_v4_1(payload) != canonical_json_bytes_v4_1(expected):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "readback report differs from exact recomputation"
        )
    return expected


def validate_bundle_values_v4_1(
    values: Mapping[str, Mapping[str, Any]],
    *,
    base_ontology: Mapping[str, Any],
    formal_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, Mapping) or set(values) != set(BUNDLE_FILENAMES):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "bundle must contain exactly three canonical artifacts"
        )
    artifacts = {filename: values[filename] for filename in BUNDLE_INPUT_FILENAMES}
    report = validate_readback_report_v4_1(
        values[READBACK_FILENAME],
        artifacts=artifacts,
        base_ontology=base_ontology,
        formal_ontology=formal_ontology,
        base_catalog=base_catalog,
        formal_catalog=formal_catalog,
    )
    screening = validate_same_snapshot_screening_v4_1(
        artifacts[SCREENING_FILENAME],
        base_ontology=base_ontology,
        formal_ontology=formal_ontology,
        base_catalog=base_catalog,
        formal_catalog=formal_catalog,
    )
    correlation = validate_correlation_diagnostic_v4_1(
        artifacts[CORRELATION_FILENAME],
        base_ontology=base_ontology,
        formal_ontology=formal_ontology,
        base_catalog=base_catalog,
        formal_catalog=formal_catalog,
        screening=screening,
    )
    return {
        SCREENING_FILENAME: screening,
        CORRELATION_FILENAME: correlation,
        READBACK_FILENAME: report,
    }


def build_private_bundle_contract_v4_1(
    *,
    expected_artifacts: Mapping[str, Mapping[str, Any]],
    base_ontology: Mapping[str, Any],
    formal_ontology: Mapping[str, Any],
    base_catalog: Mapping[str, Any],
    formal_catalog: Mapping[str, Any],
) -> private_io.PrivateBundleContract:
    """Close shared owner-private I/O over one exact recomputed artifact pair."""

    if not isinstance(expected_artifacts, Mapping) or set(expected_artifacts) != set(
        BUNDLE_INPUT_FILENAMES
    ):
        raise FactorGovernanceSameSnapshotScreeningV4_1Error(
            "expected_artifacts must contain exactly the two input artifacts"
        )
    expected = copy.deepcopy(dict(expected_artifacts))
    base_ont = copy.deepcopy(dict(base_ontology))
    formal_ont = copy.deepcopy(dict(formal_ontology))
    base_cat = copy.deepcopy(dict(base_catalog))
    formal_cat = copy.deepcopy(dict(formal_catalog))

    # Fail before publication if the closed-over pair is not mutually valid.
    screening = validate_same_snapshot_screening_v4_1(
        expected[SCREENING_FILENAME],
        base_ontology=base_ont,
        formal_ontology=formal_ont,
        base_catalog=base_cat,
        formal_catalog=formal_cat,
    )
    validate_correlation_diagnostic_v4_1(
        expected[CORRELATION_FILENAME],
        base_ontology=base_ont,
        formal_ontology=formal_ont,
        base_catalog=base_cat,
        formal_catalog=formal_cat,
        screening=screening,
    )

    def validate_artifact(filename: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if filename == READBACK_FILENAME:
            return validate_readback_report_v4_1(
                value,
                artifacts=expected,
                base_ontology=base_ont,
                formal_ontology=formal_ont,
                base_catalog=base_cat,
                formal_catalog=formal_cat,
            )
        if filename not in expected:
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                f"unexpected bundle artifact: {filename}"
            )
        if filename == SCREENING_FILENAME:
            normalized = validate_same_snapshot_screening_v4_1(
                value,
                base_ontology=base_ont,
                formal_ontology=formal_ont,
                base_catalog=base_cat,
                formal_catalog=formal_cat,
            )
        else:
            normalized = validate_correlation_diagnostic_v4_1(
                value,
                base_ontology=base_ont,
                formal_ontology=formal_ont,
                base_catalog=base_cat,
                formal_catalog=formal_cat,
                screening=expected[SCREENING_FILENAME],
            )
        if canonical_json_bytes_v4_1(normalized) != canonical_json_bytes_v4_1(
            expected[filename]
        ):
            raise FactorGovernanceSameSnapshotScreeningV4_1Error(
                f"bundle artifact differs from closed-over expected bytes: {filename}"
            )
        return copy.deepcopy(expected[filename])

    def validate_complete(
        values: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, Mapping[str, Any]]:
        return validate_bundle_values_v4_1(
            values,
            base_ontology=base_ont,
            formal_ontology=formal_ont,
            base_catalog=base_cat,
            formal_catalog=formal_cat,
        )

    def build_report(
        *,
        run_id: str,
        artifacts: Mapping[str, Mapping[str, Any]],
        artifact_bindings: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        return build_readback_report_v4_1(
            run_id=run_id,
            artifacts=artifacts,
            artifact_bindings=artifact_bindings,
            base_ontology=base_ont,
            formal_ontology=formal_ont,
            base_catalog=base_cat,
            formal_catalog=formal_cat,
        )

    return private_io.PrivateBundleContract(
        root_suffix=PRIVATE_ROOT_SUFFIX,
        input_filenames=BUNDLE_INPUT_FILENAMES,
        readback_report_filename=READBACK_FILENAME,
        canonicalize=canonical_file_bytes_v4_1,
        validate_artifact=validate_artifact,
        validate_complete=validate_complete,
        build_readback_report=build_report,
    )


__all__ = [
    "ACCOUNTING_PROFILES",
    "ACCOUNTING_PROFILE_FULLY_RESOLVED",
    "ACCOUNTING_PROFILE_FUNDAMENTAL_RESOLVED",
    "ACCOUNTING_PROFILE_LEGACY",
    "AUTHORITY_FLAGS",
    "BASE_CANDIDATE_COUNT",
    "BLOCKED_COUNT",
    "BLOCKERS",
    "BUNDLE_FILENAMES",
    "BUNDLE_INPUT_FILENAMES",
    "CORRELATION_FILENAME",
    "CORRELATION_METRIC",
    "CORRELATION_SCHEMA_VERSION",
    "CORRELATION_THRESHOLD",
    "EXPLORATORY_SHORTLIST_LIMIT",
    "FORMAL_CANDIDATE_COUNT",
    "FULLY_RESOLVED_CORRELATION_SCHEMA_VERSION",
    "FULLY_RESOLVED_READBACK_SCHEMA_VERSION",
    "FULLY_RESOLVED_SCREENING_SCHEMA_VERSION",
    "FUNDAMENTAL_BLOCKED_COUNT",
    "FUNDAMENTAL_RESOLVED_CORRELATION_SCHEMA_VERSION",
    "FUNDAMENTAL_RESOLVED_READBACK_SCHEMA_VERSION",
    "FUNDAMENTAL_RESOLVED_SCREENING_SCHEMA_VERSION",
    "FactorGovernanceSameSnapshotScreeningV4_1Error",
    "MIN_VALID_MONTH_COUNT",
    "MIN_VALID_SYMBOL_COUNT_PER_MONTH",
    "NEW_CANDIDATE_COUNT",
    "NEW_EVALUATED_COUNT",
    "PRIVATE_ROOT_SUFFIX",
    "PREDECLARED_BLOCKED_CANDIDATE_NAMES",
    "PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES",
    "PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES",
    "PROTOCOL_VERSION",
    "READBACK_FILENAME",
    "READBACK_SCHEMA_VERSION",
    "READINESS",
    "REBALANCE_POLICY",
    "SCREENING_FILENAME",
    "SCREENING_SCHEMA_VERSION",
    "SIDE_EFFECT_FLAGS",
    "STATUS_COMPUTE_FAILED",
    "STATUS_EVALUATED",
    "STATUS_FUNDAMENTAL_BLOCKED",
    "STATUS_TURNOVER_BLOCKED",
    "TURNOVER_BLOCKED_COUNT",
    "build_correlation_diagnostic_v4_1",
    "build_private_bundle_contract_v4_1",
    "build_readback_report_v4_1",
    "build_same_snapshot_screening_v4_1",
    "canonical_file_bytes_v4_1",
    "canonical_json_bytes_v4_1",
    "semantic_sha256_v4_1",
    "validate_bundle_values_v4_1",
    "validate_correlation_diagnostic_v4_1",
    "validate_readback_report_v4_1",
    "validate_same_snapshot_screening_v4_1",
]
