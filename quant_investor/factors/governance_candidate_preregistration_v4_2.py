"""Pure v4.2 candidate preregistration evidence contract.

This module records a prospective, no-label source contract for a future
FactorGovernanceProtocol v4 cycle.  It consumes only caller-supplied mappings
and sequences, rejects outcome/statistical evidence, and delegates the cycle
state transition to the existing v4.1 state machine.  It does not touch the
filesystem, providers, registries, production receipts, portfolios, or any
execution surface.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any

from quant_investor.factors.governance_cycle_state_v4_1 import (
    DISCOVERY,
    PRECOMMITTED,
    build_next_cycle_state_v4_1,
    byte_sha256 as cycle_state_byte_sha256_v4_1,
    validate_cycle_state_v4_1,
)


SCHEMA_VERSION = "factor-governance-candidate-preregistration.v4.2"
PROTOCOL_VERSION = "v4"
STATE_SCHEMA_VERSION = "factor-governance-cycle-state.v4.1"
SOURCE_ENVELOPE_SCHEMA_VERSION = (
    "factor-governance-future-source-envelope.v4.2"
)
AQUANT_RECEIPT_SCHEMA_VERSION = "factor-governance-aquant-receipt.v4.2"
MYQUANT_RECEIPT_SCHEMA_VERSION = "factor-governance-myquant-receipt.v4.2"
OPERATOR_SEMANTICS_SCHEMA_VERSION = (
    "factor-governance-operator-semantics.v4.2"
)
COMPARISON_CATALOG_RECEIPT_SCHEMA_VERSION = (
    "factor-governance-comparison-catalog-receipt.v4.2"
)
SELECTION_SPEC_SCHEMA_VERSION = "factor-governance-selection-spec.v4.2"
DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION = (
    "factor-governance-definition-identity-collision-audit.v4.2"
)
DISCOVERY_SOURCE_NODE_SCHEMA_VERSION = (
    "factor-governance-prereg-discovery-source-node.v4.2"
)
ORCHESTRATION_SCHEMA_VERSION = (
    "factor-governance-prereg-discovery-orchestration.v4.2"
)

FROZEN_PREVIOUS_CUTOFF = "2026-07-17"
FROZEN_PREVIOUS_SNAPSHOT_DATE = "2026-07-17"
EXPECTED_CANDIDATES = (
    "alpha_range_position_momentum_20d",
    "pv_low_overnight_gap_20d",
    "pv_low_vol_ratio_10_60",
    "pv_price_volume_consistency_20d",
)

AQUANT_COMMIT = "4424dcecc384f614b0e9fd5e36cf094e9244bad5"
AQUANT_PATH = "A_quant/scripts/run_factor_batch_screen.py"
AQUANT_BLOB_OID = "6de605a9ebc6c4b1f9cd730c5ffe350d11e8aef9"
AQUANT_RAW_SHA256 = (
    "011b754f01db87d04f1b924025b65c6c49999de7d20cc924cc9e22812f74c312"
)
AQUANT_MODE = "100644"
AQUANT_RANGE_DEFINITION_SHA256 = (
    "8e486283e2c36a4ecdfcd4059811afb4e42e75f53a6575f972ee17f2665a826f"
)

MYQUANT_PATH = "quant_investor/alpha158.py"
MYQUANT_FULL_SHA256 = (
    "12e6910c793f570b3699c45eb3157b594577c49f56be64d2c27c6287538a9fc8"
)
MYQUANT_COMMIT = "c03d36f115c0865602433183a04139677f2f87fb"
MYQUANT_BLOB_OID = "e2ec6e5456c4bf5970de6b020651fc81e6ce1db7"

MYQUANT_ALIAS_ROWS = (
    {
        "candidate": "pv_low_overnight_gap_20d",
        "source_factor": "OVERNIGHT_GAP_20D",
        "direction": -1,
        "source_ast_sha256": (
            "b34b831028f83f5aa7615d04f5dc81dd6c1b6a8d0a53899922348e68845a6196"
        ),
        "bound_definition_sha256": (
            "a060bd0a52353b218bb963658073e20b1b9bc5cd598c7c4207263c7f45d7dd4e"
        ),
    },
    {
        "candidate": "pv_low_vol_ratio_10_60",
        "source_factor": "VOL_RATIO_10_60",
        "direction": -1,
        "source_ast_sha256": (
            "07327e6bfab4290088a9bbbdb1b92a80e9df23087fd255b8529b878444d32ba6"
        ),
        "bound_definition_sha256": (
            "b8672e8996696c4f820f30cf6c4b97b2641cefe8b6e2ecd72ba1874685f87ac7"
        ),
    },
    {
        "candidate": "pv_price_volume_consistency_20d",
        "source_factor": "PRICE_VOL_CONSISTENCY_20D",
        "direction": 1,
        "source_ast_sha256": (
            "d8b54e3b192002dba5fb4caf5adbe9a4ac26128c9cdc5750cbc71aad39398895"
        ),
        "bound_definition_sha256": (
            "fe70f67577bc2bcd4d7bb4275d2b7aac3f4e2671ffd618cd9400d1f02145a41d"
        ),
    },
)

RUNTIME_SEMANTICS = {
    "python": "3.13.7",
    "pandas": "3.0.1",
    "numpy": "2.4.3",
}
OPERATOR_SEMANTICS = {
    "pit_remask_after_each_node": True,
    "cs_rank": "cross_sectional_rank_percentile",
    "ts_min": {"min_periods": 1},
    "ts_max": {"min_periods": 1},
    "alpha158_complete_windows": True,
    "pct_change": {"fill_method": None},
    "diff": "pandas_diff",
    "sign": "numpy_sign",
    "abs": "absolute_value",
    "std": {"ddof": 1},
    "division": {
        "A_quant": {
            "zero_guard": "none",
            "nonfinite_policy": "native_pandas_numpy_expression_result",
        },
        "alpha158": {
            "denominator_epsilon": 1e-09,
            "nonfinite_policy": "explicit_denominator_epsilon_then_runtime_mask",
        },
    },
    "runtime": RUNTIME_SEMANTICS,
    "runtime_equivalence_verified": False,
    "signal_computability_proven": False,
}

MEASUREMENT_FLAGS = {
    "statistics": "not_run",
    "family_bh": "not_run",
    "maturity": "not_run",
    "walk_forward": "not_run",
    "cost": "not_run",
    "neutralization": "not_run",
    "stability": "not_run",
    "formal_dedup": "not_run",
    "high_correlation_dedup": "not_run",
    "verified_v4_replay": "not_run",
    "transaction_plan": "not_run",
    "readiness": "PROSPECTIVE_PREREGISTRATION_ONLY",
}
AUTHORITY_FLAGS = {
    "healthy_source_receipt": False,
    "screening_authorized": False,
    "family_bh_authorized": False,
    "maturity_authorized": False,
    "candidate_qualified": False,
    "qualification_authorized": False,
    "admission_authorized": False,
    "production_new_risk_authorized": False,
    "production_candidate_authorized": False,
    "registry_write_authorized": False,
    "production_proposal_authorized": False,
    "apply_authorized": False,
}
SIDE_EFFECT_FLAGS = {
    "registry": False,
    "wal": False,
    "budget": False,
    "production_receipt": False,
    "production_pointer": False,
    "proposal": False,
    "apply": False,
    "provider": False,
    "network": False,
    "portfolio": False,
    "broker": False,
    "order": False,
    "trade": False,
    "transaction": False,
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OID_RE = re.compile(r"[0-9a-f]{40}")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SNAPSHOT_ID_RE = re.compile(r"\d{8}T\d{6}Z")
_BANNED_OUTCOME_KEYS = frozenset(
    {
        "label",
        "labels",
        "target",
        "targets",
        "forward_return",
        "realized_return",
        "return",
        "returns",
        "ic",
        "rank_ic",
        "score",
        "pvalue",
        "p_value",
        "qvalue",
        "q_value",
        "outcome",
        "verdict",
        "performance",
        "backtest",
        "replay",
        "pnl",
        "sharpe",
        "drawdown",
        "turnover",
    }
)
_ALLOWED_NEGATIVE_OUTCOME_CLAIM_KEYS = frozenset(
    {
        "artifact_semantic_sha256",
        "catalog_semantic_sha256",
        "catalog_byte_sha256",
        "selection_spec_semantic_sha256",
        "aquant_receipt_semantic_sha256",
        "myquant_receipt_semantic_sha256",
        "operator_semantics_sha256",
        "comparison_catalog_receipt_semantic_sha256",
        "definition_identity_collision_audit_semantic_sha256",
        "predecessor_state_semantic_sha256",
        "predecessor_state_byte_sha256",
        "future_source_envelope_semantic_sha256",
        "code_binding_set_semantic_sha256",
        "strict_source_binding_semantic_sha256",
        "full_a_scope_sha256",
        "cs_rank",
        "label_inputs_absent",
        "statistics",
        "family_bh",
        "maturity",
        "walk_forward",
        "cost",
        "neutralization",
        "stability",
        "formal_dedup",
        "high_correlation_dedup",
        "verified_v4_replay",
        "transaction_plan",
        "outcomes_used_as_evidence",
        "outcome_fields_absent",
    }
)


class FactorGovernanceCandidatePreregistrationV4_2Error(ValueError):
    """Raised when v4.2 preregistration evidence fails closed."""


FactorGovernanceCandidatePreregistrationV42Error = (
    FactorGovernanceCandidatePreregistrationV4_2Error
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact sorted finite JSON bytes without a trailing newline."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorGovernanceCandidatePreregistrationV4_2Error(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def canonical_file_bytes(value: Any) -> bytes:
    """Return canonical artifact bytes with one final newline."""

    return canonical_json_bytes(value) + b"\n"


def semantic_sha256(value: Any) -> str:
    """Hash canonical semantic JSON bytes, explicitly excluding a newline."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def byte_sha256(value: Any) -> str:
    """Hash canonical artifact bytes, including exactly one final newline."""

    return hashlib.sha256(canonical_file_bytes(value)).hexdigest()


canonical_json_bytes_v4_2 = canonical_json_bytes
canonical_file_bytes_v4_2 = canonical_file_bytes
semantic_sha256_v4_2 = semantic_sha256
byte_sha256_v4_2 = byte_sha256


def _self_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "artifact_semantic_sha256"}


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(dict(payload))
    sealed["artifact_semantic_sha256"] = semantic_sha256(_self_payload(sealed))
    return sealed


def _error(message: str) -> FactorGovernanceCandidatePreregistrationV4_2Error:
    return FactorGovernanceCandidatePreregistrationV4_2Error(message)


def _exact(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise _error(f"{label} field names must be strings")
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        parts: list[str] = []
        if missing:
            parts.append("missing=" + ",".join(missing))
        if unknown:
            parts.append("unknown=" + ",".join(unknown))
        raise _error(f"{label} fields invalid: {';'.join(parts)}")
    canonical_json_bytes(payload)
    return payload


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase SHA-256")
    return value


def _oid(value: Any, label: str) -> str:
    if type(value) is not str or _OID_RE.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase git OID")
    return value


def _date(value: Any, label: str) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        raise _error(f"{label} must be YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _error(f"{label} must be a real ISO calendar date") from exc
    if parsed.isoformat() != value:
        raise _error(f"{label} must be a canonical ISO calendar date")
    return value


def _snapshot_id(value: Any, *, snapshot_date: str) -> str:
    if type(value) is not str or _SNAPSHOT_ID_RE.fullmatch(value) is None:
        raise _error("snapshot_id must be exact YYYYMMDDTHHMMSSZ")
    try:
        parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise _error("snapshot_id must be a real UTC timestamp") from exc
    if parsed.strftime("%Y%m%dT%H%M%SZ") != value:
        raise _error("snapshot_id must be a canonical UTC timestamp")
    if parsed.date().isoformat() != snapshot_date:
        raise _error("snapshot_id date must equal snapshot_date")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise _error(f"{label} must be a positive integer")
    return value


def _exact_bool(value: Any, label: str, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        raise _error(f"{label} must be a boolean")
    if expected is not None and value is not expected:
        raise _error(f"{label} must be {expected}")
    return value


def _exact_zero(value: Any, label: str) -> int:
    if type(value) is not int or value != 0:
        raise _error(f"{label} must be exact integer 0")
    return 0


def _artifact_semantic(payload: Mapping[str, Any], label: str) -> str:
    supplied = _sha256(
        payload.get("artifact_semantic_sha256"), f"{label}.artifact_semantic_sha256"
    )
    expected = semantic_sha256(_self_payload(payload))
    if supplied != expected:
        raise _error(f"{label} artifact_semantic_sha256 mismatch")
    return supplied


def _semantic_binding(value: Mapping[str, Any], label: str) -> str:
    return _sha256(value.get("artifact_semantic_sha256"), f"{label}.semantic")


def _validate_definition_identity_inventory(
    value: Any, label: str
) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _error(f"{label} must be a sequence")
    rows: list[dict[str, str]] = []
    previous_name: str | None = None
    seen_names: set[str] = set()
    for index, item in enumerate(value):
        row = _exact(
            item,
            frozenset({"name", "definition_identity_sha256"}),
            f"{label}[{index}]",
        )
        name = row["name"]
        if type(name) is not str or not name:
            raise _error(f"{label}[{index}].name must be non-empty")
        identity = _sha256(
            row["definition_identity_sha256"],
            f"{label}[{index}].definition_identity_sha256",
        )
        if previous_name is not None and name <= previous_name:
            raise _error(f"{label} must be sorted by distinct name")
        if name in seen_names:
            raise _error(f"{label} names must be distinct")
        previous_name = name
        seen_names.add(name)
        rows.append(
            {
                "name": name,
                "definition_identity_sha256": identity,
            }
        )
    if not rows:
        raise _error(f"{label} must not be empty")
    return rows


def _expected_envelope_bindings(
    *,
    selection_spec: Mapping[str, Any],
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    code_binding_set_semantic_sha256: str,
) -> dict[str, str]:
    aquant = validate_aquant_receipt_v4_2(aquant_receipt)
    myquant = validate_myquant_receipt_v4_2(myquant_receipt)
    operators = validate_operator_semantics_v4_2(operator_semantics)
    comparison = validate_comparison_catalog_receipt_v4_2(
        comparison_catalog_receipt
    )
    selection = validate_selection_spec_v4_2(
        selection_spec,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
    )
    return {
        "selection_spec_semantic_sha256": _semantic_binding(
            selection, "selection spec"
        ),
        "aquant_receipt_semantic_sha256": _semantic_binding(
            aquant, "A_quant receipt"
        ),
        "myquant_receipt_semantic_sha256": _semantic_binding(
            myquant, "myQuant receipt"
        ),
        "operator_semantics_sha256": _semantic_binding(
            operators, "operator semantics"
        ),
        "comparison_catalog_receipt_semantic_sha256": _semantic_binding(
            comparison, "comparison catalog receipt"
        ),
        "code_binding_set_semantic_sha256": _sha256(
            code_binding_set_semantic_sha256, "code_binding_set_semantic_sha256"
        ),
    }


def _reject_banned_keys(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise _error(f"{label} contains a non-string key")
            normalized_key = key.lower()
            if (
                normalized_key not in _ALLOWED_NEGATIVE_OUTCOME_CLAIM_KEYS
                and normalized_key in _BANNED_OUTCOME_KEYS
            ):
                raise _error(f"{label} contains banned outcome/stat key {key!r}")
            _reject_banned_keys(item, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_banned_keys(item, f"{label}[{index}]")


def _exact_flags(payload: Mapping[str, Any]) -> None:
    if payload.get("measurement") != MEASUREMENT_FLAGS:
        raise _error("measurement flags must be exact non-authoritative not_run values")
    if payload.get("authority") != AUTHORITY_FLAGS:
        raise _error("authority flags must be exact false values")
    if payload.get("side_effects") != SIDE_EFFECT_FLAGS:
        raise _error("side_effects flags must be exact false values")


def validate_future_source_envelope_v4_2(
    value: Mapping[str, Any],
    *,
    selection_spec: Mapping[str, Any],
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    code_binding_set_semantic_sha256: str,
    strict_source_binding_semantic_sha256: str,
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    """Validate the future strict full-A snapshot envelope."""

    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "cycle_id",
            "analysis_start",
            "cutoff",
            "snapshot_id",
            "snapshot_date",
            "latest_trade_date",
            "latest_complete_trade_date",
            "market",
            "universe",
            "storage_mode",
            "coverage",
            "strict_source_binding_semantic_sha256",
            "full_a_scope_sha256",
            "full_a_scope_count",
            "serving_inventory_count",
            "selection_spec_semantic_sha256",
            "aquant_receipt_semantic_sha256",
            "myquant_receipt_semantic_sha256",
            "operator_semantics_sha256",
            "comparison_catalog_receipt_semantic_sha256",
            "code_binding_set_semantic_sha256",
            "blockers",
            "healthy_source_verified",
            "source_authority",
            "publication_status",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "future source envelope")
    if payload["schema_version"] != SOURCE_ENVELOPE_SCHEMA_VERSION:
        raise _error("future source envelope schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if type(payload["cycle_id"]) is not str or not payload["cycle_id"]:
        raise _error("cycle_id must be a non-empty string")
    analysis_start = _date(payload["analysis_start"], "analysis_start")
    cutoff = _date(payload["cutoff"], "cutoff")
    snapshot = _date(payload["snapshot_date"], "snapshot_date")
    snapshot_id = _snapshot_id(payload["snapshot_id"], snapshot_date=snapshot)
    expected_cycle_id = (
        f"cn_full_a_v4_2_{cutoff.replace('-', '')}_{snapshot_id}"
    )
    if payload["cycle_id"] != expected_cycle_id:
        raise _error(
            "cycle_id must exactly bind v4.2 evidence schema, cutoff, and "
            "snapshot_id"
        )
    if date.fromisoformat(cutoff) <= date.fromisoformat(FROZEN_PREVIOUS_CUTOFF):
        raise _error("cutoff must be later than 2026-07-17")
    if date.fromisoformat(analysis_start) > date.fromisoformat(cutoff):
        raise _error("analysis_start must not be after cutoff")
    if date.fromisoformat(snapshot) < date.fromisoformat(cutoff):
        raise _error("snapshot_date must not be before cutoff")
    if payload["latest_trade_date"] != cutoff:
        raise _error("latest_trade_date must equal cutoff")
    if payload["latest_complete_trade_date"] != cutoff:
        raise _error("latest_complete_trade_date must equal cutoff")
    if payload["market"] != "CN":
        raise _error("market must be CN")
    if payload["universe"] != "full_a":
        raise _error("universe must be full_a")
    if payload["storage_mode"] != "strict_parquet":
        raise _error("storage_mode must be strict_parquet")
    coverage = _exact(
        payload["coverage"],
        frozenset(
            {
                "coverage_ratio",
                "complete_count",
                "expected_scope_count",
            }
        ),
        "future source envelope coverage",
    )
    if type(coverage["coverage_ratio"]) is not float or coverage["coverage_ratio"] != 1.0:
        raise _error("coverage.coverage_ratio must be exact float 1.0")
    complete_count = _positive_int(coverage["complete_count"], "coverage.complete_count")
    expected_count = _positive_int(
        coverage["expected_scope_count"], "coverage.expected_scope_count"
    )
    if complete_count != expected_count:
        raise _error("coverage counts must match exactly")
    expected_full_a_count = _positive_int(
        full_a_scope_count, "full_a_scope_count"
    )
    if (
        payload["full_a_scope_count"] != expected_full_a_count
        or complete_count != expected_full_a_count
    ):
        raise _error("full-A scope and coverage counts must match exactly")
    expected_serving_count = _positive_int(
        serving_inventory_count, "serving_inventory_count"
    )
    if payload["serving_inventory_count"] != expected_serving_count:
        raise _error("serving_inventory_count mismatch")
    if payload["blockers"] != []:
        raise _error("blockers must be empty")
    _exact_bool(payload["healthy_source_verified"], "healthy_source_verified", False)
    if payload["source_authority"] != "NOT_ASSERTED_BY_PURE_CONTRACT":
        raise _error("source_authority must remain non-authoritative")
    if payload["publication_status"] != "PRECOMMIT_INTENT_ONLY":
        raise _error("publication_status must remain precommit intent only")
    for field in (
        "strict_source_binding_semantic_sha256",
        "full_a_scope_sha256",
        "selection_spec_semantic_sha256",
        "aquant_receipt_semantic_sha256",
        "myquant_receipt_semantic_sha256",
        "operator_semantics_sha256",
        "comparison_catalog_receipt_semantic_sha256",
        "code_binding_set_semantic_sha256",
    ):
        _sha256(payload[field], field)
    if payload["strict_source_binding_semantic_sha256"] != _sha256(
        strict_source_binding_semantic_sha256,
        "expected strict_source_binding_semantic_sha256",
    ):
        raise _error("strict source binding semantic SHA mismatch")
    if payload["full_a_scope_sha256"] != _sha256(
        full_a_scope_sha256, "expected full_a_scope_sha256"
    ):
        raise _error("full-A scope SHA mismatch")
    expected = _expected_envelope_bindings(
        selection_spec=selection_spec,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
        code_binding_set_semantic_sha256=code_binding_set_semantic_sha256,
    )
    for field, expected_sha in expected.items():
        if payload[field] != expected_sha:
            raise _error(f"future source envelope {field} mismatch")
    _artifact_semantic(payload, "future source envelope")
    return copy.deepcopy(payload)


def build_future_source_envelope_v4_2(
    *,
    cycle_id: str,
    analysis_start: str,
    cutoff: str,
    snapshot_id: str,
    snapshot_date: str,
    strict_source_binding_semantic_sha256: str,
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
    selection_spec: Mapping[str, Any],
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    code_binding_set_semantic_sha256: str,
) -> dict[str, Any]:
    """Build a future strict-full-A source envelope without a live health claim."""

    validate_selection_spec_v4_2(
        selection_spec,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
    )
    bindings = _expected_envelope_bindings(
        selection_spec=selection_spec,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
        code_binding_set_semantic_sha256=code_binding_set_semantic_sha256,
    )
    return validate_future_source_envelope_v4_2(
        _seal(
            {
                "schema_version": SOURCE_ENVELOPE_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "cycle_id": cycle_id,
                "analysis_start": analysis_start,
                "cutoff": cutoff,
                "snapshot_id": snapshot_id,
                "snapshot_date": snapshot_date,
                "latest_trade_date": cutoff,
                "latest_complete_trade_date": cutoff,
                "market": "CN",
                "universe": "full_a",
                "storage_mode": "strict_parquet",
                "coverage": {
                    "coverage_ratio": 1.0,
                    "complete_count": full_a_scope_count,
                    "expected_scope_count": full_a_scope_count,
                },
                "strict_source_binding_semantic_sha256": (
                    strict_source_binding_semantic_sha256
                ),
                "full_a_scope_sha256": full_a_scope_sha256,
                "full_a_scope_count": full_a_scope_count,
                "serving_inventory_count": serving_inventory_count,
                **bindings,
                "blockers": [],
                "healthy_source_verified": False,
                "source_authority": "NOT_ASSERTED_BY_PURE_CONTRACT",
                "publication_status": "PRECOMMIT_INTENT_ONLY",
            }
        ),
        selection_spec=selection_spec,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
        code_binding_set_semantic_sha256=code_binding_set_semantic_sha256,
        strict_source_binding_semantic_sha256=(
            strict_source_binding_semantic_sha256
        ),
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )


def validate_aquant_receipt_v4_2(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the pinned A_quant source receipt and reject outcomes."""

    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "project",
            "commit",
            "path",
            "blob_oid",
            "raw_sha256",
            "mode",
            "definition",
            "outcome_paths_read",
            "outcomes_used_as_evidence",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "A_quant receipt")
    _reject_banned_keys(payload, "A_quant receipt")
    if payload["schema_version"] != AQUANT_RECEIPT_SCHEMA_VERSION:
        raise _error("A_quant receipt schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if payload["project"] != "A_quant":
        raise _error("A_quant receipt project mismatch")
    if _oid(payload["commit"], "A_quant commit") != AQUANT_COMMIT:
        raise _error("A_quant commit mismatch")
    if payload["path"] != AQUANT_PATH:
        raise _error("A_quant source path mismatch")
    if _oid(payload["blob_oid"], "A_quant blob_oid") != AQUANT_BLOB_OID:
        raise _error("A_quant blob OID mismatch")
    if _sha256(payload["raw_sha256"], "A_quant raw_sha256") != AQUANT_RAW_SHA256:
        raise _error("A_quant raw SHA mismatch")
    if payload["mode"] != AQUANT_MODE:
        raise _error("A_quant file mode mismatch")
    definition = _exact(
        payload["definition"],
        frozenset({"candidate", "expression", "family", "definition_sha256"}),
        "A_quant definition",
    )
    if definition["candidate"] != "alpha_range_position_momentum_20d":
        raise _error("A_quant definition candidate mismatch")
    if (
        definition["expression"]
        != "cs_rank((close - ts_min(close, 20)) / (ts_max(close, 20) - ts_min(close, 20)))"
    ):
        raise _error("A_quant definition expression mismatch")
    if definition["family"] != "price_momentum":
        raise _error("A_quant definition family mismatch")
    if (
        _sha256(definition["definition_sha256"], "A_quant definition_sha256")
        != AQUANT_RANGE_DEFINITION_SHA256
    ):
        raise _error("A_quant definition SHA mismatch")
    if payload["outcome_paths_read"] != []:
        raise _error("outcome_paths_read must be exact empty list")
    _exact_bool(
        payload["outcomes_used_as_evidence"],
        "outcomes_used_as_evidence",
        False,
    )
    _artifact_semantic(payload, "A_quant receipt")
    return copy.deepcopy(payload)


def build_aquant_receipt_v4_2() -> dict[str, Any]:
    """Build the exact pinned A_quant source receipt."""

    return validate_aquant_receipt_v4_2(
        _seal(
            {
                "schema_version": AQUANT_RECEIPT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "project": "A_quant",
                "commit": AQUANT_COMMIT,
                "path": AQUANT_PATH,
                "blob_oid": AQUANT_BLOB_OID,
                "raw_sha256": AQUANT_RAW_SHA256,
                "mode": AQUANT_MODE,
                "definition": {
                    "candidate": "alpha_range_position_momentum_20d",
                    "expression": (
                        "cs_rank((close - ts_min(close, 20)) / "
                        "(ts_max(close, 20) - ts_min(close, 20)))"
                    ),
                    "family": "price_momentum",
                    "definition_sha256": AQUANT_RANGE_DEFINITION_SHA256,
                },
                "outcome_paths_read": [],
                "outcomes_used_as_evidence": False,
            }
        )
    )


def validate_myquant_receipt_v4_2(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the pinned myQuant alpha158 source receipt."""

    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "project",
            "commit",
            "path",
            "blob_oid",
            "full_sha256",
            "alias_rows",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "myQuant receipt")
    if payload["schema_version"] != MYQUANT_RECEIPT_SCHEMA_VERSION:
        raise _error("myQuant receipt schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if payload["project"] != "myQuant":
        raise _error("myQuant receipt project mismatch")
    if _oid(payload["commit"], "myQuant commit") != MYQUANT_COMMIT:
        raise _error("myQuant commit mismatch")
    if payload["path"] != MYQUANT_PATH:
        raise _error("myQuant path mismatch")
    if _oid(payload["blob_oid"], "myQuant blob_oid") != MYQUANT_BLOB_OID:
        raise _error("myQuant blob OID mismatch")
    if _sha256(payload["full_sha256"], "myQuant full_sha256") != MYQUANT_FULL_SHA256:
        raise _error("myQuant full SHA mismatch")
    if payload["alias_rows"] != list(MYQUANT_ALIAS_ROWS):
        raise _error("myQuant alias rows/order mismatch")
    _artifact_semantic(payload, "myQuant receipt")
    return copy.deepcopy(payload)


def build_myquant_receipt_v4_2() -> dict[str, Any]:
    """Build the exact pinned myQuant alpha158 receipt."""

    return validate_myquant_receipt_v4_2(
        _seal(
            {
                "schema_version": MYQUANT_RECEIPT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "project": "myQuant",
                "commit": MYQUANT_COMMIT,
                "path": MYQUANT_PATH,
                "blob_oid": MYQUANT_BLOB_OID,
                "full_sha256": MYQUANT_FULL_SHA256,
                "alias_rows": list(copy.deepcopy(MYQUANT_ALIAS_ROWS)),
            }
        )
    )


def validate_operator_semantics_v4_2(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the hard-coded operator semantics receipt."""

    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "semantics",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "operator semantics")
    if payload["schema_version"] != OPERATOR_SEMANTICS_SCHEMA_VERSION:
        raise _error("operator semantics schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if payload["semantics"] != OPERATOR_SEMANTICS:
        raise _error("operator semantics must be exact")
    _artifact_semantic(payload, "operator semantics")
    return copy.deepcopy(payload)


def build_operator_semantics_v4_2() -> dict[str, Any]:
    """Build the exact operator-semantics receipt."""

    return validate_operator_semantics_v4_2(
        _seal(
            {
                "schema_version": OPERATOR_SEMANTICS_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "semantics": copy.deepcopy(OPERATOR_SEMANTICS),
            }
        )
    )


def validate_comparison_catalog_receipt_v4_2(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a no-label comparison catalog receipt for collision checks."""

    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "catalog_id",
            "catalog_byte_sha256",
            "catalog_semantic_sha256",
            "candidate_count",
            "primitive_count",
            "definition_identity_inventory",
            "label_inputs_absent",
            "outcome_fields_absent",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "comparison catalog receipt")
    _reject_banned_keys(payload, "comparison catalog receipt")
    if payload["schema_version"] != COMPARISON_CATALOG_RECEIPT_SCHEMA_VERSION:
        raise _error("comparison catalog receipt schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if type(payload["catalog_id"]) is not str or not payload["catalog_id"]:
        raise _error("catalog_id must be a non-empty string")
    _sha256(payload["catalog_byte_sha256"], "catalog_byte_sha256")
    _sha256(payload["catalog_semantic_sha256"], "catalog_semantic_sha256")
    inventory = _validate_definition_identity_inventory(
        payload["definition_identity_inventory"],
        "definition_identity_inventory",
    )
    candidate_count = _positive_int(payload["candidate_count"], "candidate_count")
    if candidate_count != len(inventory):
        raise _error(
            "candidate_count must equal definition identity inventory length"
        )
    _positive_int(payload["primitive_count"], "primitive_count")
    _exact_bool(payload["label_inputs_absent"], "label_inputs_absent", True)
    _exact_bool(payload["outcome_fields_absent"], "outcome_fields_absent", True)
    _artifact_semantic(payload, "comparison catalog receipt")
    return copy.deepcopy(payload)


def build_comparison_catalog_receipt_v4_2(
    *,
    catalog_id: str,
    catalog_byte_sha256: str,
    catalog_semantic_sha256: str,
    primitive_count: int,
    definition_identity_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a no-label comparison-catalog receipt."""

    normalized_inventory = _validate_definition_identity_inventory(
        definition_identity_inventory, "definition_identity_inventory"
    )
    return validate_comparison_catalog_receipt_v4_2(
        _seal(
            {
                "schema_version": COMPARISON_CATALOG_RECEIPT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "catalog_id": catalog_id,
                "catalog_byte_sha256": catalog_byte_sha256,
                "catalog_semantic_sha256": catalog_semantic_sha256,
                "candidate_count": len(normalized_inventory),
                "primitive_count": primitive_count,
                "definition_identity_inventory": normalized_inventory,
                "label_inputs_absent": True,
                "outcome_fields_absent": True,
            }
        )
    )


def _candidate_rows(
    *,
    aquant_receipt_sha256: str,
    myquant_receipt_sha256: str,
    operator_semantics_sha256: str,
    comparison_catalog_sha256: str,
) -> list[dict[str, Any]]:
    return [
        {
            "order": 1,
            "name": "alpha_range_position_momentum_20d",
            "source": "A_quant",
            "source_definition_sha256": AQUANT_RANGE_DEFINITION_SHA256,
            "definition_identity_sha256": AQUANT_RANGE_DEFINITION_SHA256,
            "family": "price_momentum",
            "rationale": "prospective range-position momentum idea only",
            "initial_weight": 0,
            "source_receipt_semantic_sha256": aquant_receipt_sha256,
            "operator_semantics_sha256": operator_semantics_sha256,
            "comparison_catalog_semantic_sha256": comparison_catalog_sha256,
        },
        {
            "order": 2,
            "name": "pv_low_overnight_gap_20d",
            "source": "myQuant.alpha158",
            "source_definition_sha256": MYQUANT_ALIAS_ROWS[0][
                "bound_definition_sha256"
            ],
            "definition_identity_sha256": MYQUANT_ALIAS_ROWS[0][
                "bound_definition_sha256"
            ],
            "family": "overnight_gap",
            "rationale": "prospective low overnight-gap reversal idea only",
            "initial_weight": 0,
            "source_receipt_semantic_sha256": myquant_receipt_sha256,
            "operator_semantics_sha256": operator_semantics_sha256,
            "comparison_catalog_semantic_sha256": comparison_catalog_sha256,
        },
        {
            "order": 3,
            "name": "pv_low_vol_ratio_10_60",
            "source": "myQuant.alpha158",
            "source_definition_sha256": MYQUANT_ALIAS_ROWS[1][
                "bound_definition_sha256"
            ],
            "definition_identity_sha256": MYQUANT_ALIAS_ROWS[1][
                "bound_definition_sha256"
            ],
            "family": "realized_volatility_ratio",
            "rationale": "prospective low short-to-long realized-volatility ratio idea only",
            "initial_weight": 0,
            "source_receipt_semantic_sha256": myquant_receipt_sha256,
            "operator_semantics_sha256": operator_semantics_sha256,
            "comparison_catalog_semantic_sha256": comparison_catalog_sha256,
        },
        {
            "order": 4,
            "name": "pv_price_volume_consistency_20d",
            "source": "myQuant.alpha158",
            "source_definition_sha256": MYQUANT_ALIAS_ROWS[2][
                "bound_definition_sha256"
            ],
            "definition_identity_sha256": MYQUANT_ALIAS_ROWS[2][
                "bound_definition_sha256"
            ],
            "family": "price_volume_consistency",
            "rationale": "prospective price-volume consistency idea only",
            "initial_weight": 0,
            "source_receipt_semantic_sha256": myquant_receipt_sha256,
            "operator_semantics_sha256": operator_semantics_sha256,
            "comparison_catalog_semantic_sha256": comparison_catalog_sha256,
        },
    ]


def validate_selection_spec_v4_2(
    value: Mapping[str, Any],
    *,
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact ordered, zero-weight no-label selection spec."""

    aquant = validate_aquant_receipt_v4_2(aquant_receipt)
    myquant = validate_myquant_receipt_v4_2(myquant_receipt)
    operators = validate_operator_semantics_v4_2(operator_semantics)
    comparison = validate_comparison_catalog_receipt_v4_2(
        comparison_catalog_receipt
    )
    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "candidate_count",
            "candidates",
            "claims",
            "measurement",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "selection spec")
    _reject_banned_keys(payload, "selection spec")
    if payload["schema_version"] != SELECTION_SPEC_SCHEMA_VERSION:
        raise _error("selection spec schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if payload["candidate_count"] != 4:
        raise _error("candidate_count must be 4")
    expected_rows = _candidate_rows(
        aquant_receipt_sha256=aquant["artifact_semantic_sha256"],
        myquant_receipt_sha256=myquant["artifact_semantic_sha256"],
        operator_semantics_sha256=operators["artifact_semantic_sha256"],
        comparison_catalog_sha256=comparison["artifact_semantic_sha256"],
    )
    if payload["candidates"] != expected_rows:
        raise _error("candidate rows/order/source bindings mismatch")
    names = [row["name"] for row in payload["candidates"]]
    if tuple(names) != EXPECTED_CANDIDATES or len(set(names)) != 4:
        raise _error("candidate oracle mismatch")
    for index, row in enumerate(payload["candidates"], start=1):
        if row["order"] != index:
            raise _error("candidate order mismatch")
        _exact_zero(row["initial_weight"], f"candidate {row['name']} initial_weight")
    claims = _exact(
        payload["claims"],
        frozenset(
            {
                "artifact_and_builder_label_inputs_absent",
                "outcome_paths_read",
                "outcomes_used_as_evidence",
                "selection_uninfluenced_by_any_external_label",
                "authoritative_evidence_route",
            }
        ),
        "selection claims",
    )
    _exact_bool(
        claims["artifact_and_builder_label_inputs_absent"],
        "artifact_and_builder_label_inputs_absent",
        True,
    )
    _exact_bool(
        claims["outcomes_used_as_evidence"],
        "outcomes_used_as_evidence",
        False,
    )
    if claims["outcome_paths_read"] != []:
        raise _error("selection outcome_paths_read must be exact empty list")
    if claims["selection_uninfluenced_by_any_external_label"] != "UNPROVEN":
        raise _error("selection external-label claim must be UNPROVEN")
    if (
        claims["authoritative_evidence_route"]
        != "prospective_post_preregistration_holdout_only"
    ):
        raise _error("authoritative evidence route mismatch")
    _exact_flags(payload)
    _artifact_semantic(payload, "selection spec")
    return copy.deepcopy(payload)


def build_selection_spec_v4_2(
    *,
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact four-row no-label zero-weight selection spec."""

    aquant = validate_aquant_receipt_v4_2(aquant_receipt)
    myquant = validate_myquant_receipt_v4_2(myquant_receipt)
    operators = validate_operator_semantics_v4_2(operator_semantics)
    comparison = validate_comparison_catalog_receipt_v4_2(
        comparison_catalog_receipt
    )
    return validate_selection_spec_v4_2(
        _seal(
            {
                "schema_version": SELECTION_SPEC_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "candidate_count": 4,
                "candidates": _candidate_rows(
                    aquant_receipt_sha256=aquant["artifact_semantic_sha256"],
                    myquant_receipt_sha256=myquant["artifact_semantic_sha256"],
                    operator_semantics_sha256=operators[
                        "artifact_semantic_sha256"
                    ],
                    comparison_catalog_sha256=comparison[
                        "artifact_semantic_sha256"
                    ],
                ),
                "claims": {
                    "artifact_and_builder_label_inputs_absent": True,
                    "outcome_paths_read": [],
                    "outcomes_used_as_evidence": False,
                    "selection_uninfluenced_by_any_external_label": "UNPROVEN",
                    "authoritative_evidence_route": (
                        "prospective_post_preregistration_holdout_only"
                    ),
                },
                "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
                "authority": copy.deepcopy(AUTHORITY_FLAGS),
                "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
            }
        ),
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
    )


def validate_candidate_preregistration_v4_2(
    value: Mapping[str, Any],
    *,
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility alias for the v4.2 selection-spec validator."""

    return validate_selection_spec_v4_2(
        value,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
    )


def build_candidate_preregistration_v4_2(
    *,
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility alias for building the v4.2 selection spec."""

    return build_selection_spec_v4_2(
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
    )


def _definition_identities_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    return {
        str(row["name"]): str(row["definition_identity_sha256"])
        for row in rows
    }


def _comparison_definition_identity_map(
    comparison: Mapping[str, Any],
) -> dict[str, str]:
    return {
        row["name"]: row["definition_identity_sha256"]
        for row in comparison["definition_identity_inventory"]
    }


def _definition_identity_collision_pairs(
    selected: Mapping[str, str],
    comparison: Mapping[str, str],
) -> list[dict[str, str]]:
    collisions: list[dict[str, str]] = []
    for selected_name, selected_hash in selected.items():
        for comparison_name, comparison_hash in comparison.items():
            if selected_name == comparison_name or selected_hash == comparison_hash:
                collisions.append(
                    {
                        "selected": selected_name,
                        "comparison": comparison_name,
                        "reason": (
                            "name"
                            if selected_name == comparison_name
                            else "definition_identity_sha256"
                        ),
                    }
                )
    return sorted(collisions, key=lambda row: (row["selected"], row["comparison"]))


def validate_definition_identity_collision_audit_v4_2(
    value: Mapping[str, Any],
    *,
    selection_spec: Mapping[str, Any],
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an identity-equality diagnostic, not structural dedup."""

    selection = validate_selection_spec_v4_2(
        selection_spec,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
    )
    comparison = validate_comparison_catalog_receipt_v4_2(
        comparison_catalog_receipt
    )
    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "selection_spec_semantic_sha256",
            "comparison_catalog_semantic_sha256",
            "method",
            "selected_vs_selected",
            "selected_vs_comparison",
            "definition_identity_collision_detected",
            "duplicate_primitive",
            "structural_dedup",
            "formal_dedup",
            "high_correlation_dedup",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "definition identity collision audit")
    if (
        payload["schema_version"]
        != DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION
    ):
        raise _error("definition identity collision audit schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if (
        _sha256(
            payload["selection_spec_semantic_sha256"],
            "selection_spec_semantic_sha256",
        )
        != selection["artifact_semantic_sha256"]
    ):
        raise _error("selection spec binding mismatch")
    if (
        _sha256(
            payload["comparison_catalog_semantic_sha256"],
            "comparison_catalog_semantic_sha256",
        )
        != comparison["artifact_semantic_sha256"]
    ):
        raise _error("comparison catalog binding mismatch")
    if payload["method"] != "definition_identity_equality_only.v1":
        raise _error("definition identity collision method mismatch")
    expected_selected = _definition_identities_from_rows(
        selection["candidates"]
    )
    if len(set(expected_selected.values())) != len(expected_selected):
        raise _error("selected candidates contain definition identity collisions")
    comparison_identities = _comparison_definition_identity_map(comparison)
    comparison_collisions = _definition_identity_collision_pairs(
        expected_selected, comparison_identities
    )
    if comparison_collisions:
        raise _error(
            "selected candidates have definition identity collision with "
            "comparison catalog"
        )
    if payload["selected_vs_selected"] != {
        "checked": True,
        "definition_identities": expected_selected,
        "collisions": [],
    }:
        raise _error("selected-vs-selected audit mismatch")
    if payload["selected_vs_comparison"] != {
        "checked": True,
        "comparison_catalog_id": comparison["catalog_id"],
        "comparison_definition_identities": comparison_identities,
        "collisions": [],
    }:
        raise _error("selected-vs-comparison audit mismatch")
    _exact_bool(
        payload["definition_identity_collision_detected"],
        "definition_identity_collision_detected",
        False,
    )
    if payload["duplicate_primitive"] != "not_authoritative_not_run":
        raise _error("duplicate_primitive must not assert false")
    if payload["structural_dedup"] != "not_run":
        raise _error("structural_dedup must be not_run")
    if payload["formal_dedup"] != "not_run":
        raise _error("formal_dedup must be not_run")
    if payload["high_correlation_dedup"] != "not_run":
        raise _error("high_correlation_dedup must be not_run")
    _artifact_semantic(payload, "definition identity collision audit")
    return copy.deepcopy(payload)


def build_definition_identity_collision_audit_v4_2(
    *,
    selection_spec: Mapping[str, Any],
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an exact definition-identity collision diagnostic."""

    selection = validate_selection_spec_v4_2(
        selection_spec,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
    )
    comparison = validate_comparison_catalog_receipt_v4_2(
        comparison_catalog_receipt
    )
    selected_identities = _definition_identities_from_rows(
        selection["candidates"]
    )
    comparison_identities = _comparison_definition_identity_map(comparison)
    if _definition_identity_collision_pairs(
        selected_identities, comparison_identities
    ):
        raise _error(
            "selected candidates have definition identity collision with "
            "comparison catalog"
        )
    return validate_definition_identity_collision_audit_v4_2(
        _seal(
            {
                "schema_version": (
                    DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION
                ),
                "protocol_version": PROTOCOL_VERSION,
                "selection_spec_semantic_sha256": selection[
                    "artifact_semantic_sha256"
                ],
                "comparison_catalog_semantic_sha256": comparison[
                    "artifact_semantic_sha256"
                ],
                "method": "definition_identity_equality_only.v1",
                "selected_vs_selected": {
                    "checked": True,
                    "definition_identities": selected_identities,
                    "collisions": [],
                },
                "selected_vs_comparison": {
                    "checked": True,
                    "comparison_catalog_id": comparison["catalog_id"],
                    "comparison_definition_identities": (
                        comparison_identities
                    ),
                    "collisions": [],
                },
                "definition_identity_collision_detected": False,
                "duplicate_primitive": "not_authoritative_not_run",
                "structural_dedup": "not_run",
                "formal_dedup": "not_run",
                "high_correlation_dedup": "not_run",
            }
        ),
        selection_spec=selection,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison,
    )


def validate_prereg_discovery_source_node_v4_2(
    value: Mapping[str, Any],
    *,
    predecessor_state: Mapping[str, Any],
    future_source_envelope: Mapping[str, Any],
    selection_spec: Mapping[str, Any],
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    code_binding_set_semantic_sha256: str,
    strict_source_binding_semantic_sha256: str,
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    """Validate the preregistration DISCOVERY source node."""

    aquant = validate_aquant_receipt_v4_2(aquant_receipt)
    myquant = validate_myquant_receipt_v4_2(myquant_receipt)
    operators = validate_operator_semantics_v4_2(operator_semantics)
    comparison = validate_comparison_catalog_receipt_v4_2(
        comparison_catalog_receipt
    )
    selection = validate_selection_spec_v4_2(
        selection_spec,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
    )
    predecessor = validate_cycle_state_v4_1(
        predecessor_state, expected_state=PRECOMMITTED
    )
    envelope = validate_future_source_envelope_v4_2(
        future_source_envelope,
        selection_spec=selection,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
        code_binding_set_semantic_sha256=code_binding_set_semantic_sha256,
        strict_source_binding_semantic_sha256=(
            strict_source_binding_semantic_sha256
        ),
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )
    if envelope["cycle_id"] != predecessor["cycle_id"]:
        raise _error("future source envelope cycle_id mismatch")
    if predecessor["source_chain_node_sha256"] != envelope["artifact_semantic_sha256"]:
        raise _error("PRECOMMITTED source_chain_node_sha256 must bind envelope")
    collision = validate_definition_identity_collision_audit_v4_2(
        definition_identity_collision_audit,
        selection_spec=selection,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
    )
    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "state_schema_version",
            "cycle_id",
            "cycle_root_sha256",
            "predecessor_state_byte_sha256",
            "predecessor_state_semantic_sha256",
            "predecessor_source_chain_node_sha256",
            "future_source_envelope_semantic_sha256",
            "selection_spec_semantic_sha256",
            "aquant_receipt_semantic_sha256",
            "myquant_receipt_semantic_sha256",
            "operator_semantics_sha256",
            "comparison_catalog_receipt_semantic_sha256",
            "code_binding_set_semantic_sha256",
            "definition_identity_collision_audit_semantic_sha256",
            "selection_claims",
            "measurement",
            "authority",
            "side_effects",
            "dual_sha_predecessor_transition_validated",
            "exact_once_publication",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "prereg discovery source node")
    if payload["schema_version"] != DISCOVERY_SOURCE_NODE_SCHEMA_VERSION:
        raise _error("prereg discovery source node schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if payload["state_schema_version"] != STATE_SCHEMA_VERSION:
        raise _error("state schema version mismatch")
    if payload["cycle_id"] != predecessor["cycle_id"]:
        raise _error("cycle_id mismatch")
    if payload["cycle_root_sha256"] != predecessor["cycle_root_sha256"]:
        raise _error("cycle_root_sha256 mismatch")
    predecessor_byte = cycle_state_byte_sha256_v4_1(predecessor)
    bindings = {
        "predecessor_state_byte_sha256": predecessor_byte,
        "predecessor_state_semantic_sha256": predecessor[
            "state_semantic_sha256"
        ],
        "predecessor_source_chain_node_sha256": predecessor[
            "source_chain_node_sha256"
        ],
        "future_source_envelope_semantic_sha256": envelope[
            "artifact_semantic_sha256"
        ],
        "selection_spec_semantic_sha256": selection["artifact_semantic_sha256"],
        "aquant_receipt_semantic_sha256": aquant["artifact_semantic_sha256"],
        "myquant_receipt_semantic_sha256": myquant["artifact_semantic_sha256"],
        "operator_semantics_sha256": operators["artifact_semantic_sha256"],
        "comparison_catalog_receipt_semantic_sha256": comparison[
            "artifact_semantic_sha256"
        ],
        "code_binding_set_semantic_sha256": envelope[
            "code_binding_set_semantic_sha256"
        ],
        "definition_identity_collision_audit_semantic_sha256": collision[
            "artifact_semantic_sha256"
        ],
    }
    for field, expected in bindings.items():
        if _sha256(payload[field], field) != expected:
            raise _error(f"{field} mismatch")
    if payload["selection_claims"] != selection["claims"]:
        raise _error("selection claims must be copied exactly")
    _exact_flags(payload)
    _exact_bool(
        payload["dual_sha_predecessor_transition_validated"],
        "dual_sha_predecessor_transition_validated",
        True,
    )
    if payload["exact_once_publication"] != "NOT_IMPLEMENTED":
        raise _error("exact_once_publication must be NOT_IMPLEMENTED")
    _artifact_semantic(payload, "prereg discovery source node")
    return copy.deepcopy(payload)


def build_prereg_discovery_source_node_v4_2(
    *,
    predecessor_state: Mapping[str, Any],
    future_source_envelope: Mapping[str, Any],
    selection_spec: Mapping[str, Any],
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    code_binding_set_semantic_sha256: str,
    strict_source_binding_semantic_sha256: str,
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    """Build the preregistration DISCOVERY source node."""

    predecessor = validate_cycle_state_v4_1(
        predecessor_state, expected_state=PRECOMMITTED
    )
    aquant = validate_aquant_receipt_v4_2(aquant_receipt)
    myquant = validate_myquant_receipt_v4_2(myquant_receipt)
    operators = validate_operator_semantics_v4_2(operator_semantics)
    comparison = validate_comparison_catalog_receipt_v4_2(
        comparison_catalog_receipt
    )
    selection = validate_selection_spec_v4_2(
        selection_spec,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
    )
    envelope = validate_future_source_envelope_v4_2(
        future_source_envelope,
        selection_spec=selection,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
        code_binding_set_semantic_sha256=code_binding_set_semantic_sha256,
        strict_source_binding_semantic_sha256=(
            strict_source_binding_semantic_sha256
        ),
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )
    collision = validate_definition_identity_collision_audit_v4_2(
        definition_identity_collision_audit,
        selection_spec=selection,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
    )
    return validate_prereg_discovery_source_node_v4_2(
        _seal(
            {
                "schema_version": DISCOVERY_SOURCE_NODE_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "state_schema_version": STATE_SCHEMA_VERSION,
                "cycle_id": predecessor["cycle_id"],
                "cycle_root_sha256": predecessor["cycle_root_sha256"],
                "predecessor_state_byte_sha256": (
                    cycle_state_byte_sha256_v4_1(predecessor)
                ),
                "predecessor_state_semantic_sha256": predecessor[
                    "state_semantic_sha256"
                ],
                "predecessor_source_chain_node_sha256": predecessor[
                    "source_chain_node_sha256"
                ],
                "future_source_envelope_semantic_sha256": envelope[
                    "artifact_semantic_sha256"
                ],
                "selection_spec_semantic_sha256": selection[
                    "artifact_semantic_sha256"
                ],
                "aquant_receipt_semantic_sha256": aquant[
                    "artifact_semantic_sha256"
                ],
                "myquant_receipt_semantic_sha256": myquant[
                    "artifact_semantic_sha256"
                ],
                "operator_semantics_sha256": operators[
                    "artifact_semantic_sha256"
                ],
                "comparison_catalog_receipt_semantic_sha256": comparison[
                    "artifact_semantic_sha256"
                ],
                "code_binding_set_semantic_sha256": envelope[
                    "code_binding_set_semantic_sha256"
                ],
                "definition_identity_collision_audit_semantic_sha256": collision[
                    "artifact_semantic_sha256"
                ],
                "selection_claims": copy.deepcopy(selection["claims"]),
                "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
                "authority": copy.deepcopy(AUTHORITY_FLAGS),
                "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
                "dual_sha_predecessor_transition_validated": True,
                "exact_once_publication": "NOT_IMPLEMENTED",
            }
        ),
        predecessor_state=predecessor,
        future_source_envelope=envelope,
        selection_spec=selection,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
        definition_identity_collision_audit=collision,
        code_binding_set_semantic_sha256=code_binding_set_semantic_sha256,
        strict_source_binding_semantic_sha256=(
            strict_source_binding_semantic_sha256
        ),
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )


def build_preregistration_source_node_v4_2(
    *,
    predecessor_state: Mapping[str, Any],
    future_source_envelope: Mapping[str, Any],
    selection_spec: Mapping[str, Any],
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    code_binding_set_semantic_sha256: str,
    strict_source_binding_semantic_sha256: str,
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    """Compatibility alias for the preregistration source-node builder."""

    return build_prereg_discovery_source_node_v4_2(
        predecessor_state=predecessor_state,
        future_source_envelope=future_source_envelope,
        selection_spec=selection_spec,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
        definition_identity_collision_audit=(
            definition_identity_collision_audit
        ),
        code_binding_set_semantic_sha256=code_binding_set_semantic_sha256,
        strict_source_binding_semantic_sha256=(
            strict_source_binding_semantic_sha256
        ),
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )


def _build_preregistration_discovery_cycle_payload_v4_2(
    *,
    predecessor_state: Mapping[str, Any],
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    future_source_envelope: Mapping[str, Any],
    selection_spec: Mapping[str, Any],
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    code_binding_set_semantic_sha256: str,
    strict_source_binding_semantic_sha256: str,
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    predecessor = validate_cycle_state_v4_1(
        predecessor_state, expected_state=PRECOMMITTED
    )
    supplied_predecessor_byte = _sha256(
        predecessor_byte_sha256, "predecessor_byte_sha256"
    )
    actual_predecessor_byte = cycle_state_byte_sha256_v4_1(predecessor)
    if supplied_predecessor_byte != actual_predecessor_byte:
        raise _error("predecessor byte SHA mismatch")
    if expected_predecessor_byte_sha256 != actual_predecessor_byte:
        raise _error("expected predecessor byte SHA mismatch")
    if (
        expected_predecessor_semantic_sha256
        != predecessor["state_semantic_sha256"]
    ):
        raise _error("expected predecessor semantic SHA mismatch")
    source_node = build_prereg_discovery_source_node_v4_2(
        predecessor_state=predecessor,
        future_source_envelope=future_source_envelope,
        selection_spec=selection_spec,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
        definition_identity_collision_audit=(
            definition_identity_collision_audit
        ),
        code_binding_set_semantic_sha256=code_binding_set_semantic_sha256,
        strict_source_binding_semantic_sha256=(
            strict_source_binding_semantic_sha256
        ),
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )
    selection = validate_selection_spec_v4_2(
        selection_spec,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
    )
    discovery_state = build_next_cycle_state_v4_1(
        predecessor=predecessor,
        predecessor_byte_sha256=actual_predecessor_byte,
        expected_predecessor_byte_sha256=expected_predecessor_byte_sha256,
        expected_predecessor_semantic_sha256=(
            expected_predecessor_semantic_sha256
        ),
        cycle_id=predecessor["cycle_id"],
        cycle_root_sha256=predecessor["cycle_root_sha256"],
        next_state=DISCOVERY,
        source_chain_node_sha256=source_node["artifact_semantic_sha256"],
    )
    payload = _seal(
        {
            "schema_version": ORCHESTRATION_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "state_schema_version": STATE_SCHEMA_VERSION,
            "predecessor_state": predecessor,
            "source_node": source_node,
            "discovery_state": discovery_state,
            "persisted_state_sequence": [PRECOMMITTED, DISCOVERY],
            "precommitted_state_role": "INTRA_BUNDLE_LINEAGE_ONLY",
            "discovery_state_role": "FINAL_CURRENT",
            "external_state_pointer_mutation": False,
            "selection_claims": copy.deepcopy(selection["claims"]),
            "dual_sha_predecessor_transition_validated": True,
            "exact_once_publication": "NOT_IMPLEMENTED",
            "measurement": copy.deepcopy(MEASUREMENT_FLAGS),
            "authority": copy.deepcopy(AUTHORITY_FLAGS),
            "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
        }
    )
    _artifact_semantic(payload, "prereg discovery orchestration")
    return payload


def validate_preregistration_discovery_cycle_v4_2(
    value: Mapping[str, Any],
    *,
    predecessor_state: Mapping[str, Any],
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    future_source_envelope: Mapping[str, Any],
    selection_spec: Mapping[str, Any],
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    code_binding_set_semantic_sha256: str,
    strict_source_binding_semantic_sha256: str,
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    """Rebuild and compare the complete PRECOMMITTED -> DISCOVERY graph."""

    fields = frozenset(
        {
            "schema_version",
            "protocol_version",
            "state_schema_version",
            "predecessor_state",
            "source_node",
            "discovery_state",
            "persisted_state_sequence",
            "precommitted_state_role",
            "discovery_state_role",
            "external_state_pointer_mutation",
            "selection_claims",
            "dual_sha_predecessor_transition_validated",
            "exact_once_publication",
            "measurement",
            "authority",
            "side_effects",
            "artifact_semantic_sha256",
        }
    )
    payload = _exact(value, fields, "prereg discovery orchestration")
    if payload["schema_version"] != ORCHESTRATION_SCHEMA_VERSION:
        raise _error("prereg discovery orchestration schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must be v4")
    if payload["state_schema_version"] != STATE_SCHEMA_VERSION:
        raise _error("state schema version mismatch")
    if payload["persisted_state_sequence"] != [PRECOMMITTED, DISCOVERY]:
        raise _error("persisted state sequence must be PRECOMMITTED then DISCOVERY")
    if payload["precommitted_state_role"] != "INTRA_BUNDLE_LINEAGE_ONLY":
        raise _error("PRECOMMITTED must remain intra-bundle lineage only")
    if payload["discovery_state_role"] != "FINAL_CURRENT":
        raise _error("DISCOVERY must be the final current state")
    _exact_bool(
        payload["external_state_pointer_mutation"],
        "external_state_pointer_mutation",
        False,
    )
    _exact_bool(
        payload["dual_sha_predecessor_transition_validated"],
        "dual_sha_predecessor_transition_validated",
        True,
    )
    if payload["exact_once_publication"] != "NOT_IMPLEMENTED":
        raise _error("exact_once_publication must be NOT_IMPLEMENTED")
    _exact_flags(payload)
    _artifact_semantic(payload, "prereg discovery orchestration")

    expected = _build_preregistration_discovery_cycle_payload_v4_2(
        predecessor_state=predecessor_state,
        predecessor_byte_sha256=predecessor_byte_sha256,
        expected_predecessor_byte_sha256=expected_predecessor_byte_sha256,
        expected_predecessor_semantic_sha256=(
            expected_predecessor_semantic_sha256
        ),
        future_source_envelope=future_source_envelope,
        selection_spec=selection_spec,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
        definition_identity_collision_audit=(
            definition_identity_collision_audit
        ),
        code_binding_set_semantic_sha256=code_binding_set_semantic_sha256,
        strict_source_binding_semantic_sha256=(
            strict_source_binding_semantic_sha256
        ),
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )
    if payload != expected:
        raise _error("prereg discovery orchestration graph mismatch")
    return copy.deepcopy(payload)


def build_preregistration_discovery_cycle_v4_2(
    *,
    predecessor_state: Mapping[str, Any],
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    future_source_envelope: Mapping[str, Any],
    selection_spec: Mapping[str, Any],
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    code_binding_set_semantic_sha256: str,
    strict_source_binding_semantic_sha256: str,
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    """Orchestrate the pure PRECOMMITTED -> DISCOVERY preregistration step."""

    payload = _build_preregistration_discovery_cycle_payload_v4_2(
        predecessor_state=predecessor_state,
        predecessor_byte_sha256=predecessor_byte_sha256,
        expected_predecessor_byte_sha256=expected_predecessor_byte_sha256,
        expected_predecessor_semantic_sha256=(
            expected_predecessor_semantic_sha256
        ),
        future_source_envelope=future_source_envelope,
        selection_spec=selection_spec,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
        definition_identity_collision_audit=(
            definition_identity_collision_audit
        ),
        code_binding_set_semantic_sha256=code_binding_set_semantic_sha256,
        strict_source_binding_semantic_sha256=(
            strict_source_binding_semantic_sha256
        ),
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )
    return validate_preregistration_discovery_cycle_v4_2(
        payload,
        predecessor_state=predecessor_state,
        predecessor_byte_sha256=predecessor_byte_sha256,
        expected_predecessor_byte_sha256=expected_predecessor_byte_sha256,
        expected_predecessor_semantic_sha256=(
            expected_predecessor_semantic_sha256
        ),
        future_source_envelope=future_source_envelope,
        selection_spec=selection_spec,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
        definition_identity_collision_audit=(
            definition_identity_collision_audit
        ),
        code_binding_set_semantic_sha256=code_binding_set_semantic_sha256,
        strict_source_binding_semantic_sha256=(
            strict_source_binding_semantic_sha256
        ),
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )


def build_discovery_cycle_state_v4_2(
    *,
    predecessor_state: Mapping[str, Any],
    predecessor_byte_sha256: str,
    expected_predecessor_byte_sha256: str,
    expected_predecessor_semantic_sha256: str,
    future_source_envelope: Mapping[str, Any],
    selection_spec: Mapping[str, Any],
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    definition_identity_collision_audit: Mapping[str, Any],
    code_binding_set_semantic_sha256: str,
    strict_source_binding_semantic_sha256: str,
    full_a_scope_sha256: str,
    full_a_scope_count: int,
    serving_inventory_count: int,
) -> dict[str, Any]:
    """Compatibility alias for the pure DISCOVERY orchestration."""

    return build_preregistration_discovery_cycle_v4_2(
        predecessor_state=predecessor_state,
        predecessor_byte_sha256=predecessor_byte_sha256,
        expected_predecessor_byte_sha256=expected_predecessor_byte_sha256,
        expected_predecessor_semantic_sha256=(
            expected_predecessor_semantic_sha256
        ),
        future_source_envelope=future_source_envelope,
        selection_spec=selection_spec,
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
        definition_identity_collision_audit=(
            definition_identity_collision_audit
        ),
        code_binding_set_semantic_sha256=code_binding_set_semantic_sha256,
        strict_source_binding_semantic_sha256=(
            strict_source_binding_semantic_sha256
        ),
        full_a_scope_sha256=full_a_scope_sha256,
        full_a_scope_count=full_a_scope_count,
        serving_inventory_count=serving_inventory_count,
    )


__all__ = [
    "AQUANT_BLOB_OID",
    "AQUANT_COMMIT",
    "AQUANT_MODE",
    "AQUANT_PATH",
    "AQUANT_RANGE_DEFINITION_SHA256",
    "AQUANT_RAW_SHA256",
    "AUTHORITY_FLAGS",
    "EXPECTED_CANDIDATES",
    "FROZEN_PREVIOUS_CUTOFF",
    "MEASUREMENT_FLAGS",
    "MYQUANT_ALIAS_ROWS",
    "MYQUANT_BLOB_OID",
    "MYQUANT_COMMIT",
    "MYQUANT_FULL_SHA256",
    "MYQUANT_PATH",
    "OPERATOR_SEMANTICS",
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "SIDE_EFFECT_FLAGS",
    "FactorGovernanceCandidatePreregistrationV4_2Error",
    "FactorGovernanceCandidatePreregistrationV42Error",
    "build_aquant_receipt_v4_2",
    "build_candidate_preregistration_v4_2",
    "build_comparison_catalog_receipt_v4_2",
    "build_definition_identity_collision_audit_v4_2",
    "build_discovery_cycle_state_v4_2",
    "build_future_source_envelope_v4_2",
    "build_myquant_receipt_v4_2",
    "build_operator_semantics_v4_2",
    "build_prereg_discovery_source_node_v4_2",
    "build_preregistration_source_node_v4_2",
    "build_preregistration_discovery_cycle_v4_2",
    "build_selection_spec_v4_2",
    "byte_sha256",
    "byte_sha256_v4_2",
    "canonical_file_bytes",
    "canonical_file_bytes_v4_2",
    "canonical_json_bytes",
    "canonical_json_bytes_v4_2",
    "semantic_sha256",
    "semantic_sha256_v4_2",
    "validate_aquant_receipt_v4_2",
    "validate_candidate_preregistration_v4_2",
    "validate_comparison_catalog_receipt_v4_2",
    "validate_definition_identity_collision_audit_v4_2",
    "validate_future_source_envelope_v4_2",
    "validate_myquant_receipt_v4_2",
    "validate_operator_semantics_v4_2",
    "validate_prereg_discovery_source_node_v4_2",
    "validate_preregistration_discovery_cycle_v4_2",
    "validate_selection_spec_v4_2",
]
