"""Pure future-only strict exact-five computability contract for v4.4.

The module is deliberately filesystem-free and imports only the standard
library.  It validates caller-supplied, already collected evidence.  It cannot
discover a snapshot, read a predecessor, publish a bundle, or authorize Factor
readiness, production, or new risk.
"""

from __future__ import annotations

import base64
import copy
from datetime import date, datetime
import hashlib
import json
import math
import posixpath
import re
from collections.abc import Mapping, Sequence
from typing import Any


PROTOCOL_VERSION = "v4"
EVIDENCE_CONTRACT_VERSION = "v4.4"
FROZEN_PREVIOUS_CUTOFF = "2026-07-19"
READINESS = "NON_AUTHORIZING_STRICT_COMPUTABILITY_ONLY"
READBACK_SCOPE = "SEALED_BUNDLE_GRAPH_ONLY"

MANIFEST_SCHEMA_VERSION = (
    "factor-governance-future-strict-computability-input.v4.4"
)
INPUT_RECEIPT_SCHEMA_VERSION = (
    "factor-governance-future-strict-computability-input-receipt.v4.4"
)
DATA_FIELD_RECEIPT_SCHEMA_VERSION = (
    "factor-governance-future-strict-computability-data-field-receipt.v4.4"
)
TWO_PASS_EQUIVALENCE_RECEIPT_SCHEMA_VERSION = (
    "factor-governance-future-strict-computability-two-pass-equivalence.v4.4"
)
PROOF_SCHEMA_VERSION = (
    "factor-governance-future-strict-signal-computability-proof.v4.4"
)
READBACK_SCHEMA_VERSION = (
    "factor-governance-future-strict-signal-computability-readback.v4.4"
)
RUNTIME_BINDING_SCHEMA_VERSION = (
    "factor-governance-future-strict-runtime-binding.v4.4"
)
OPERATOR_PROGRAM_SET_SCHEMA_VERSION = (
    "factor-governance-future-strict-operator-program-set.v4.4"
)

RUNTIME_EXECUTION_MODE = "sealed_shadow_tree"
RUNTIME_ISOLATED_FLAGS = ("-B", "-I", "-S")
RUNTIME_IMPORT_ROOTS = (
    {"order": 1, "kind": "directory", "relative_path": "numpy"},
    {
        "order": 2,
        "kind": "directory",
        "relative_path": "numpy-2.4.3.dist-info",
    },
    {"order": 3, "kind": "directory", "relative_path": "pandas"},
    {
        "order": 4,
        "kind": "directory",
        "relative_path": "pandas-3.0.1.dist-info",
    },
    {"order": 5, "kind": "directory", "relative_path": "pyarrow"},
    {
        "order": 6,
        "kind": "directory",
        "relative_path": "pyarrow-24.0.0.dist-info",
    },
    {"order": 7, "kind": "directory", "relative_path": "dateutil"},
    {
        "order": 8,
        "kind": "directory",
        "relative_path": "python_dateutil-2.9.0.post0.dist-info",
    },
    {"order": 9, "kind": "directory", "relative_path": "pytz"},
    {
        "order": 10,
        "kind": "directory",
        "relative_path": "pytz-2026.1.post1.dist-info",
    },
    {
        "order": 11,
        "kind": "directory",
        "relative_path": "six-1.17.0.dist-info",
    },
    {"order": 12, "kind": "file", "relative_path": "six.py"},
)

PRIVATE_ROOT_SUFFIX = (
    "reports",
    "factor_governance",
    "private",
    "v4_4_signal_computability_strict",
)

INPUT_MANIFEST_FILENAME = "strict_computability_input_manifest.v4_4.json"
INPUT_RECEIPT_FILENAME = "strict_computability_input_receipt.v4_4.json"
DATA_FIELD_RECEIPT_FILENAME = "strict_data_field_receipt.v4_4.json"
TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME = (
    "strict_two_pass_equivalence_receipt.v4_4.json"
)
PROOF_FILENAME = "strict_exact_five_signal_computability_proof.v4_4.json"
READBACK_FILENAME = "strict_signal_computability_readback.v4_4.json"
READBACK_REPORT_FILENAME = READBACK_FILENAME
ROOT_SUFFIX = PRIVATE_ROOT_SUFFIX
INPUT_FILENAMES = (
    INPUT_MANIFEST_FILENAME,
    INPUT_RECEIPT_FILENAME,
    DATA_FIELD_RECEIPT_FILENAME,
    TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME,
    PROOF_FILENAME,
)
BUNDLE_FILENAMES = (*INPUT_FILENAMES, READBACK_FILENAME)

PREREGISTRATION_ARTIFACT_COUNT = 27
EXPECTED_CANDIDATE_ROWS_SEMANTIC_SHA256 = (
    "147e6bd8dc23d9e38b09a198b110e516830c3908400b0a3d8bb48ec1de16ec39"
)

PREREGISTRATION_FILENAMES = (
    "v4_2_predecessor.aquant_idea_source_receipt.v4_2.json",
    "v4_2_predecessor.myquant_alpha158_source_receipt.v4_2.json",
    "v4_2_predecessor.operator_semantics.v4_2.json",
    "v4_2_predecessor.comparison_catalog_receipt.v4_2.json",
    "v4_2_predecessor.candidate_selection_spec.v4_2.json",
    "v4_2_predecessor.strict_full_a_source_binding.v4_2.json",
    "v4_2_predecessor.code_binding_set.v4_2.json",
    "v4_2_predecessor.future_source_envelope.v4_2.json",
    "v4_2_predecessor.cycle_root.v4_2.json",
    "v4_2_predecessor.definition_identity_collision_audit.v4_2.json",
    "v4_2_predecessor.cycle_state.precommitted.v4_1.json",
    "v4_2_predecessor.discovery_source_node.v4_2.json",
    "v4_2_predecessor.cycle_state.discovery.v4_1.json",
    "v4_2_predecessor.prereg_discovery_orchestration.v4_2.json",
    "code_binding_set.v4_4.json",
    "prior_diagnostic_runtime_binding.v4_3.json",
    "prior_diagnostic_nomination.v4_3.json",
    "prior_diagnostic_nomination_readback.v4_3.json",
    "expanded_candidate_selection.v4_4.json",
    "definition_identity_collision_audit.v4_4.json",
    "cycle_root.v4_4.json",
    "future_source_envelope.v4_4.json",
    "cycle_state.precommitted.v4_1.json",
    "discovery_source_node.v4_4.json",
    "cycle_state.discovery.v4_1.json",
    "prereg_discovery_orchestration.v4_4.json",
    "candidate_preregistration_readback.v4_4.json",
)

CODE_BINDING_PATHS = (
    "quant_investor/factors/governance_future_strict_exact_five_eval_v4_4.py",
    "quant_investor/factors/governance_future_strict_signal_computability_v4_4.py",
    "scripts/build_factor_v4_4_future_strict_signal_computability.py",
    "quant_investor/factors/governance_private_bundle_io.py",
    "quant_investor/factors/governance_candidate_preregistration_v4_4.py",
    "quant_investor/factors/governance_candidate_preregistration_bundle_v4_4.py",
    "quant_investor/factors/governance_candidate_preregistration_v4_2.py",
    "quant_investor/factors/governance_candidate_preregistration_bundle_v4_2.py",
    "quant_investor/factors/governance_prior_diagnostic_nomination_v4_3.py",
    "quant_investor/factors/governance_prior_diagnostic_nomination_bundle_v4_3.py",
    "quant_investor/factors/governance_source_readback_v4_1.py",
    "quant_investor/alpha158.py",
)

PROTECTED_CONTROL_NAMES = (
    "registry",
    "latest_pointer",
    "catalog",
    "fundamental_pointer",
    "latest_manifest",
)

TABLE_PROJECTION = (
    "trade_date",
    "ts_code",
    "open",
    "close",
    "vol",
    "adj_close",
)
INPUT_FIELDS = ("raw_close", "raw_open", "vol", "adj_close")
PASS_IDS = ("fresh_pass_1", "fresh_pass_2")
ENGINE_IDS = (
    "closed_pandas_source_dag.future_strictexact.v4.4",
    "independent_numpy_local_formulas.future_strictexact.v4.4",
)

MANIFEST_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "cycle_id",
        "cutoff",
        "snapshot_id",
        "proof_output_start",
        "preregistration",
        "strict_source_expected",
        "source_definition_bindings",
        "code_binding_set",
        "runtime_binding_expected_semantic_sha256",
        "protected_control_expected_sha256",
        "resource_contract",
        "selection_disclosures",
        "negative_claims",
    }
)

RESOURCE_CONTRACT = {
    "manifest_max_bytes": 65_536,
    "prereg_artifact_max_bytes": 64 * 1024 * 1024,
    "prereg_bundle_max_bytes": 256 * 1024 * 1024,
    "strict_artifact_max_bytes": 16 * 1024 * 1024,
    "strict_bundle_max_bytes": 64 * 1024 * 1024,
    "table_member_count_max": 256,
    "table_member_max_bytes": 64 * 1024 * 1024,
    "table_total_max_bytes": 1024 * 1024 * 1024,
    "pit_max_bytes": 128 * 1024 * 1024,
    "pit_row_count_max": 16_384,
    "source_session_count_max": 4_096,
    "historical_symbol_count_max": 8_192,
    "projected_row_count_per_pass_max": 16_777_216,
    "dense_cell_count_per_block_max": 1_540_096,
    "rss_max_bytes": 3 * 1024 * 1024 * 1024,
    "pass_wall_seconds_max": 1_800,
    "total_wall_seconds_max": 3_900,
    "halo_session_count": 60,
    "output_block_session_count": 128,
}

SELECTION_DISCLOSURES = {
    "outcome_informed_selection": True,
    "external_label_independence": False,
    "prior_statistics_inherited_as_formal_evidence": False,
}

NEGATIVE_CLAIMS = {
    "measurement": {
        "statistics": "not_run",
        "ic": "not_run",
        "fdr": "not_run",
        "family_bh": "not_run",
        "maturity": "not_run",
        "walk_forward": "not_run",
        "cost": "not_run",
        "neutralization": "not_run",
        "stability": "not_run",
        "structural_dedup": "not_run",
        "formal_dedup": "not_run",
        "high_correlation_dedup": "not_run",
        "qualification": "not_run",
        "admission": "not_run",
        "verified_v4_replay": "not_run",
        "transaction_plan": "not_run",
    },
    "authority": {
        "healthy_source_receipt": False,
        "healthy_factor_authorized": False,
        "measurement_authorized": False,
        "screening_authorized": False,
        "statistics_authorized": False,
        "ic_authorized": False,
        "fdr_authorized": False,
        "family_bh_authorized": False,
        "maturity_authorized": False,
        "walk_forward_authorized": False,
        "cost_authorized": False,
        "neutralization_authorized": False,
        "stability_authorized": False,
        "dedup_authorized": False,
        "qualification_authorized": False,
        "admission_authorized": False,
        "replay_authorized": False,
        "transaction_authorized": False,
        "proposal_authorized": False,
        "registry_write_authorized": False,
        "apply_authorized": False,
        "activation_authorized": False,
        "production_candidate_authorized": False,
        "production_new_risk_authorized": False,
        "new_risk_authorized": False,
    },
    "side_effects": {
        "registry": False,
        "wal": False,
        "budget": False,
        "production_receipt": False,
        "production_pointer": False,
        "proposal": False,
        "apply": False,
        "activation": False,
        "provider": False,
        "network": False,
        "portfolio": False,
        "broker": False,
        "order": False,
        "trade": False,
        "transaction": False,
    },
}

POSITIVE_CLAIMS = {
    "strict_snapshot_signal_computability_proven": True,
    "exact_five_atomic": True,
    "independent_engine_equivalence": True,
    "double_fresh_read_reproducibility": True,
    "readiness": READINESS,
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OID_RE = re.compile(r"[0-9a-f]{40}")
_SNAPSHOT_ID_RE = re.compile(r"\d{8}T\d{6}Z")
_DETERMINISTIC_CYCLE_ID_RE = re.compile(
    r"cn_full_a_v4_4_strict_computability_(\d{8})_(\d{8}T\d{6}Z)"
)
_CN_SYMBOL_RE = re.compile(r"\d{6}\.(?:SH|SZ|BJ)")
_SAFE_RELATIVE_COMPONENT_RE = re.compile(r"[A-Za-z0-9_.-]+")
_PROHIBITED_EVIDENCE_KEYS = frozenset(
    {
        "label",
        "labels",
        "target",
        "targets",
        "forward_return",
        "forward_returns",
        "realized_return",
        "realized_returns",
        "outcome",
        "outcomes",
        "rank_ic",
        "p_value",
        "q_value",
        "bonferroni_p",
        "pnl",
        "performance",
        "verdict",
    }
)


class FactorGovernanceFutureStrictSignalComputabilityV4_4Error(ValueError):
    """Raised when strict v4.4 evidence fails closed."""


def _error(
    message: str,
) -> FactorGovernanceFutureStrictSignalComputabilityV4_4Error:
    return FactorGovernanceFutureStrictSignalComputabilityV4_4Error(message)


def _validate_json_value(value: Any, label: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise _error(f"{label} contains non-finite JSON number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise _error(f"{label} contains non-string field name")
            _validate_json_value(item, f"{label}.{key}")
        return
    raise _error(f"{label} contains non-JSON type {type(value).__name__}")


def canonical_json_bytes_v4_4(value: Any) -> bytes:
    """Return strict compact sorted finite UTF-8 JSON bytes."""

    _validate_json_value(value, "value")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # defensive after type walk
        raise _error(f"value is not canonical finite JSON: {exc}") from exc


def canonical_file_bytes_v4_4(value: Any) -> bytes:
    """Return canonical JSON file bytes with exactly one trailing newline."""

    return canonical_json_bytes_v4_4(value) + b"\n"


def semantic_sha256_v4_4(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes_v4_4(value)).hexdigest()


def byte_sha256_v4_4(value: bytes) -> str:
    if type(value) is not bytes:
        raise _error("byte SHA input must be exact bytes")
    return hashlib.sha256(value).hexdigest()


def parse_canonical_json_file_bytes_v4_4(
    raw: bytes, *, max_bytes: int = RESOURCE_CONTRACT["strict_artifact_max_bytes"]
) -> dict[str, Any]:
    """Parse one canonical object while rejecting duplicates and byte drift."""

    if type(raw) is not bytes or not raw or len(raw) > max_bytes:
        raise _error("canonical JSON file byte size is invalid")
    duplicate = False

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, item in values:
            if key in result:
                duplicate = True
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _error(f"canonical JSON file cannot be parsed: {exc}") from exc
    if duplicate or type(value) is not dict:
        raise _error("canonical JSON file must be one object without duplicates")
    if canonical_file_bytes_v4_4(value) != raw:
        raise _error("JSON file bytes are not the canonical encoding")
    return value


def _exact_object(value: Any, fields: frozenset[str] | set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise _error(f"{label} must be an exact JSON object")
    missing = sorted(set(fields) - set(value))
    unknown = sorted(set(value) - set(fields))
    if missing or unknown:
        raise _error(
            f"{label} fields invalid: missing={','.join(missing) or '-'};"
            f"unknown={','.join(unknown) or '-'}"
        )
    _validate_json_value(value, label)
    return copy.deepcopy(value)


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase SHA-256")
    return value


def _oid(value: Any, label: str) -> str:
    if type(value) is not str or _OID_RE.fullmatch(value) is None:
        raise _error(f"{label} must be a lowercase 40-hex Git object ID")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise _error(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise _error(f"{label} must be a non-negative integer")
    return value


def _canonical_date(value: Any, label: str) -> str:
    if type(value) is not str:
        raise _error(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _error(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise _error(f"{label} must be canonical YYYY-MM-DD")
    return value


def _snapshot_id(value: Any, *, cutoff: str) -> str:
    if type(value) is not str or _SNAPSHOT_ID_RE.fullmatch(value) is None:
        raise _error("snapshot_id must be canonical YYYYMMDDTHHMMSSZ")
    try:
        parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise _error("snapshot_id must be a real UTC timestamp") from exc
    if parsed.date().isoformat() != cutoff:
        raise _error("snapshot_id calendar date must exactly equal cutoff")
    return value


def _absolute_normalized_path(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("/")
        or value == "/"
        or "\x00" in value
        or "//" in value
    ):
        raise _error(f"{label} must be an absolute normalized path")
    parts = value.split("/")[1:]
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise _error(f"{label} must be an absolute normalized path")
    if posixpath.normpath(value) != value:
        raise _error(f"{label} must be an absolute normalized path")
    return value


def _relative_path(value: Any, label: str) -> str:
    if type(value) is not str or not value or value.startswith("/") or "\x00" in value:
        raise _error(f"{label} must be a normalized repository-relative path")
    parts = value.split("/")
    if any(
        not part
        or part in {".", ".."}
        or _SAFE_RELATIVE_COMPONENT_RE.fullmatch(part) is None
        for part in parts
    ):
        raise _error(f"{label} must be a normalized repository-relative path")
    return value


def _reject_prohibited_keys(value: Any, label: str) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if key.lower() in _PROHIBITED_EVIDENCE_KEYS:
                raise _error(f"{label} contains prohibited evidence field {key}")
            _reject_prohibited_keys(item, label)
    elif type(value) is list:
        for item in value:
            _reject_prohibited_keys(item, label)


def deterministic_cycle_id_v4_4(*, cutoff: str, snapshot_id: str) -> str:
    normalized_cutoff = _canonical_date(cutoff, "cutoff")
    normalized_snapshot = _snapshot_id(snapshot_id, cutoff=normalized_cutoff)
    return (
        "cn_full_a_v4_4_strict_computability_"
        f"{normalized_cutoff.replace('-', '')}_"
        f"{normalized_snapshot}"
    )


def deterministic_preregistration_cycle_id_v4_4(
    *, cutoff: str, snapshot_id: str
) -> str:
    normalized_cutoff = _canonical_date(cutoff, "cutoff")
    normalized_snapshot = _snapshot_id(snapshot_id, cutoff=normalized_cutoff)
    return (
        f"cn_full_a_v4_4_{normalized_cutoff.replace('-', '')}_"
        f"{normalized_snapshot}"
    )


def _validate_context_free_cycle_identity_v4_4(
    value: Mapping[str, Any], *, label: str
) -> tuple[str, str, str]:
    """Validate the self-contained cutoff/snapshot/cycle identity tuple."""

    cutoff = _canonical_date(value["cutoff"], f"{label} cutoff")
    if date.fromisoformat(cutoff) <= date.fromisoformat(FROZEN_PREVIOUS_CUTOFF):
        raise _error(f"{label} cutoff must be strictly later than 2026-07-19")
    snapshot_id = _snapshot_id(value["snapshot_id"], cutoff=cutoff)
    proof_output_start = _canonical_date(
        value["proof_output_start"], f"{label} proof_output_start"
    )
    if proof_output_start > cutoff:
        raise _error(f"{label} proof_output_start must not be later than cutoff")
    if value["cycle_id"] != deterministic_cycle_id_v4_4(
        cutoff=cutoff, snapshot_id=snapshot_id
    ):
        raise _error(f"{label} cycle_id is not deterministic")
    return cutoff, snapshot_id, proof_output_start


def _validate_embedded_cycle_id_v4_4(value: Any, *, label: str) -> str:
    """Validate a deterministic cycle ID when no separate identity fields exist."""

    if type(value) is not str:
        raise _error(f"{label} must be a deterministic v4.4 cycle ID")
    match = _DETERMINISTIC_CYCLE_ID_RE.fullmatch(value)
    if match is None:
        raise _error(f"{label} must be a deterministic v4.4 cycle ID")
    try:
        cutoff = datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
    except ValueError as exc:
        raise _error(f"{label} contains an invalid cutoff date") from exc
    if date.fromisoformat(cutoff) <= date.fromisoformat(FROZEN_PREVIOUS_CUTOFF):
        raise _error(f"{label} cutoff must be strictly later than 2026-07-19")
    snapshot_id = _snapshot_id(match.group(2), cutoff=cutoff)
    if value != deterministic_cycle_id_v4_4(
        cutoff=cutoff, snapshot_id=snapshot_id
    ):
        raise _error(f"{label} is not deterministic")
    return value


def full_a_scope_sha256_v4_4(symbols: Sequence[str]) -> str:
    """Hash sorted unique symbols joined by newlines without a final newline."""

    if type(symbols) not in {list, tuple} or not symbols:
        raise _error("full-A symbols must be a non-empty list or tuple")
    normalized = list(symbols)
    if any(type(item) is not str or _CN_SYMBOL_RE.fullmatch(item) is None for item in normalized):
        raise _error("full-A symbols must be canonical CN symbols")
    if len(set(normalized)) != len(normalized):
        raise _error("full-A symbols must be unique")
    return hashlib.sha256("\n".join(sorted(normalized)).encode("ascii")).hexdigest()


FIELD_SEMANTICS = (
    {
        "candidate": "alpha_range_position_momentum_20d",
        "canonical_inputs": ["raw_close"],
        "physical_columns": ["close"],
        "source_facing_fields": ["close"],
        "adjustment": "raw_unadjusted",
        "units": ["canonical_price_units"],
        "rescaling": "identity",
        "dtype": "float64",
        "missing_policy": "preserve_nonfinite_then_node_level_pit_remask",
    },
    {
        "candidate": "pv_low_overnight_gap_20d",
        "canonical_inputs": ["raw_open", "raw_close"],
        "physical_columns": ["open", "close"],
        "source_facing_fields": ["open", "close"],
        "adjustment": "raw_unadjusted",
        "units": ["canonical_price_units", "canonical_price_units"],
        "rescaling": "identity",
        "dtype": "float64",
        "missing_policy": "preserve_nonfinite_then_node_level_pit_remask",
    },
    {
        "candidate": "pv_low_vol_ratio_10_60",
        "canonical_inputs": ["raw_close"],
        "physical_columns": ["close"],
        "source_facing_fields": ["close"],
        "adjustment": "raw_unadjusted",
        "units": ["canonical_price_units"],
        "rescaling": "identity",
        "dtype": "float64",
        "missing_policy": "preserve_nonfinite_then_node_level_pit_remask",
    },
    {
        "candidate": "pv_price_volume_consistency_20d",
        "canonical_inputs": ["raw_close", "vol"],
        "physical_columns": ["close", "vol"],
        "source_facing_fields": ["close", "volume"],
        "adjustment": "raw_unadjusted",
        "units": ["canonical_price_units", "canonical_volume_units"],
        "rescaling": "vol_exposed_as_volume_without_scaling",
        "dtype": "float64",
        "missing_policy": "preserve_nonfinite_then_node_level_pit_remask",
    },
    {
        "candidate": "pv_low_vol_of_vol_20d",
        "canonical_inputs": ["adj_close"],
        "physical_columns": ["adj_close"],
        "source_facing_fields": ["close"],
        "adjustment": "exact_adjusted_close",
        "units": ["canonical_adjusted_price_units"],
        "rescaling": "identity",
        "dtype": "float64",
        "missing_policy": "preserve_nonfinite_then_node_level_pit_remask",
    },
)

_SOURCE_DEFINITION_BASE = (
    {
        "order": 1,
        "name": "alpha_range_position_momentum_20d",
        "definition_identity_sha256": "8e486283e2c36a4ecdfcd4059811afb4e42e75f53a6575f972ee17f2665a826f",
        "direction": 1,
        "source_repository": "A_quant",
        "source_commit": "4424dcecc384f614b0e9fd5e36cf094e9244bad5",
        "source_tree_oid": "7365bdd815fb8442bb7b0deece489eb5d8b396da",
        "source_relative_path": "A_quant/scripts/run_factor_batch_screen.py",
        "source_blob_oid": "6de605a9ebc6c4b1f9cd730c5ffe350d11e8aef9",
        "source_raw_sha256": "011b754f01db87d04f1b924025b65c6c49999de7d20cc924cc9e22812f74c312",
        "source_ast_sha256": "8e486283e2c36a4ecdfcd4059811afb4e42e75f53a6575f972ee17f2665a826f",
        "field_semantics_sha256": semantic_sha256_v4_4(FIELD_SEMANTICS[0]),
    },
    {
        "order": 2,
        "name": "pv_low_overnight_gap_20d",
        "definition_identity_sha256": "a060bd0a52353b218bb963658073e20b1b9bc5cd598c7c4207263c7f45d7dd4e",
        "direction": -1,
        "source_repository": "myQuant",
        "source_commit": "c03d36f115c0865602433183a04139677f2f87fb",
        "source_tree_oid": "fbde997f57a1b595d09f2a563d62760a9ef13d85",
        "source_relative_path": "quant_investor/alpha158.py",
        "source_blob_oid": "e2ec6e5456c4bf5970de6b020651fc81e6ce1db7",
        "source_raw_sha256": "12e6910c793f570b3699c45eb3157b594577c49f56be64d2c27c6287538a9fc8",
        "source_ast_sha256": "b34b831028f83f5aa7615d04f5dc81dd6c1b6a8d0a53899922348e68845a6196",
        "field_semantics_sha256": semantic_sha256_v4_4(FIELD_SEMANTICS[1]),
    },
    {
        "order": 3,
        "name": "pv_low_vol_ratio_10_60",
        "definition_identity_sha256": "b8672e8996696c4f820f30cf6c4b97b2641cefe8b6e2ecd72ba1874685f87ac7",
        "direction": -1,
        "source_repository": "myQuant",
        "source_commit": "c03d36f115c0865602433183a04139677f2f87fb",
        "source_tree_oid": "fbde997f57a1b595d09f2a563d62760a9ef13d85",
        "source_relative_path": "quant_investor/alpha158.py",
        "source_blob_oid": "e2ec6e5456c4bf5970de6b020651fc81e6ce1db7",
        "source_raw_sha256": "12e6910c793f570b3699c45eb3157b594577c49f56be64d2c27c6287538a9fc8",
        "source_ast_sha256": "07327e6bfab4290088a9bbbdb1b92a80e9df23087fd255b8529b878444d32ba6",
        "field_semantics_sha256": semantic_sha256_v4_4(FIELD_SEMANTICS[2]),
    },
    {
        "order": 4,
        "name": "pv_price_volume_consistency_20d",
        "definition_identity_sha256": "fe70f67577bc2bcd4d7bb4275d2b7aac3f4e2671ffd618cd9400d1f02145a41d",
        "direction": 1,
        "source_repository": "myQuant",
        "source_commit": "c03d36f115c0865602433183a04139677f2f87fb",
        "source_tree_oid": "fbde997f57a1b595d09f2a563d62760a9ef13d85",
        "source_relative_path": "quant_investor/alpha158.py",
        "source_blob_oid": "e2ec6e5456c4bf5970de6b020651fc81e6ce1db7",
        "source_raw_sha256": "12e6910c793f570b3699c45eb3157b594577c49f56be64d2c27c6287538a9fc8",
        "source_ast_sha256": "d8b54e3b192002dba5fb4caf5adbe9a4ac26128c9cdc5750cbc71aad39398895",
        "field_semantics_sha256": semantic_sha256_v4_4(FIELD_SEMANTICS[3]),
    },
    {
        "order": 5,
        "name": "pv_low_vol_of_vol_20d",
        "definition_identity_sha256": "eb401bc44af71069b87eee44a3c4bb5ba73abe5337dc38a9ab1ac9e6b4bb261a",
        "direction": -1,
        "source_repository": "myQuant",
        "source_commit": "c03d36f115c0865602433183a04139677f2f87fb",
        "source_tree_oid": "fbde997f57a1b595d09f2a563d62760a9ef13d85",
        "source_relative_path": "quant_investor/alpha158.py",
        "source_blob_oid": "e2ec6e5456c4bf5970de6b020651fc81e6ce1db7",
        "source_raw_sha256": "12e6910c793f570b3699c45eb3157b594577c49f56be64d2c27c6287538a9fc8",
        "source_ast_sha256": "295f0b8580b0b77e749da27274b02bcb6662afeff0c6b7b22245e677ed49aa31",
        "field_semantics_sha256": semantic_sha256_v4_4(FIELD_SEMANTICS[4]),
    },
)

OPERATOR_EXECUTION_SEMANTICS = {
    "dtype": "float64",
    "memory_order": "C",
    "matrix_axes": ["sessions", "symbols"],
    "execution_partitioning": (
        "deterministic_block_local_by_manifest_input_block"
    ),
    "rolling_state_lifecycle": "reset_before_every_manifest_input_block",
    "historical_halo_session_count": 60,
    "maximum_output_session_count_per_block": 128,
    "pandas_fixed_window_accumulator_semantics": (
        "pandas_3.0.1_within_each_manifest_input_block"
    ),
    "monolithic_pandas_bit_equivalence_claimed": False,
    "pit_remask": "after_every_node",
    "nonfinite_policy": (
        "preserve_nan_and_convert_both_infinities_to_nan_after_every_node"
    ),
    "rolling_axis": "sessions",
    "rolling_nan_policy": "skip_nan_subject_to_explicit_min_periods",
    "division": "native_float64_only_explicit_constant_epsilon_nodes",
    "pct_change_lowering": (
        "subtract(divide(x,shift(x,1)),constant_1)_fill_method_none"
    ),
    "diff_lowering": "subtract(x,shift(x,1))",
    "structural_cse": (
        "exact_opcode_ordered_input_ids_canonical_parameters_only_no_algebraic_rewrite"
    ),
}

_OPERATOR_PROGRAM_SET_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "execution_semantics",
        "candidate_count",
        "candidates",
        "artifact_semantic_sha256",
    }
)
_OPERATOR_PROGRAM_FIELDS = frozenset(
    {
        "order",
        "name",
        "direction",
        "definition_identity_sha256",
        "source_repository",
        "source_commit",
        "source_tree_oid",
        "source_relative_path",
        "source_blob_oid",
        "source_raw_sha256",
        "source_ast_sha256",
        "field_semantics_sha256",
        "field_adapter",
        "nodes",
        "output_node_id",
        "program_semantic_sha256",
    }
)
_OPERATOR_NODE_FIELDS = frozenset({"node_id", "opcode", "inputs", "parameters"})
_OPERATOR_OPCODES = frozenset(
    {
        "source",
        "constant",
        "shift",
        "add",
        "subtract",
        "multiply",
        "divide",
        "absolute",
        "sign",
        "rolling_min",
        "rolling_max",
        "rolling_mean",
        "rolling_std",
        "cross_section_rank",
    }
)
_OPERATOR_CONSTANTS = frozenset(
    {"3ff0000000000000", "3e112e0be826d695"}
)
_RANK_PARAMETERS = {
    "axis": "symbols",
    "method": "average",
    "na_option": "keep",
    "pct": True,
    "ascending": True,
}


def _operator_node(
    node_id: str,
    opcode: str,
    inputs: Sequence[str] = (),
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "opcode": opcode,
        "inputs": list(inputs),
        "parameters": copy.deepcopy(dict(parameters or {})),
    }


_GOLDEN_PROGRAM_NODES = (
    (
        _operator_node("n000", "source", parameters={"source_field": "close", "canonical_input": "raw_close"}),
        _operator_node("n001", "rolling_min", ("n000",), {"window": 20, "min_periods": 1}),
        _operator_node("n002", "subtract", ("n000", "n001")),
        _operator_node("n003", "rolling_max", ("n000",), {"window": 20, "min_periods": 1}),
        _operator_node("n004", "subtract", ("n003", "n001")),
        _operator_node("n005", "divide", ("n002", "n004")),
        _operator_node("n006", "cross_section_rank", ("n005",), _RANK_PARAMETERS),
    ),
    (
        _operator_node("n000", "source", parameters={"source_field": "open", "canonical_input": "raw_open"}),
        _operator_node("n001", "source", parameters={"source_field": "close", "canonical_input": "raw_close"}),
        _operator_node("n002", "shift", ("n001",), {"periods": 1}),
        _operator_node("n003", "subtract", ("n000", "n002")),
        _operator_node("n004", "constant", parameters={"float64_be_hex": "3e112e0be826d695"}),
        _operator_node("n005", "add", ("n002", "n004")),
        _operator_node("n006", "divide", ("n003", "n005")),
        _operator_node("n007", "absolute", ("n006",)),
        _operator_node("n008", "rolling_mean", ("n007",), {"window": 20, "min_periods": 20}),
    ),
    (
        _operator_node("n000", "source", parameters={"source_field": "close", "canonical_input": "raw_close"}),
        _operator_node("n001", "shift", ("n000",), {"periods": 1}),
        _operator_node("n002", "divide", ("n000", "n001")),
        _operator_node("n003", "constant", parameters={"float64_be_hex": "3ff0000000000000"}),
        _operator_node("n004", "subtract", ("n002", "n003")),
        _operator_node("n005", "rolling_std", ("n004",), {"window": 10, "min_periods": 10, "ddof": 1}),
        _operator_node("n006", "rolling_std", ("n004",), {"window": 60, "min_periods": 60, "ddof": 1}),
        _operator_node("n007", "constant", parameters={"float64_be_hex": "3e112e0be826d695"}),
        _operator_node("n008", "add", ("n006", "n007")),
        _operator_node("n009", "divide", ("n005", "n008")),
    ),
    (
        _operator_node("n000", "source", parameters={"source_field": "close", "canonical_input": "raw_close"}),
        _operator_node("n001", "shift", ("n000",), {"periods": 1}),
        _operator_node("n002", "subtract", ("n000", "n001")),
        _operator_node("n003", "sign", ("n002",)),
        _operator_node("n004", "source", parameters={"source_field": "volume", "canonical_input": "vol"}),
        _operator_node("n005", "shift", ("n004",), {"periods": 1}),
        _operator_node("n006", "subtract", ("n004", "n005")),
        _operator_node("n007", "sign", ("n006",)),
        _operator_node("n008", "multiply", ("n003", "n007")),
        _operator_node("n009", "rolling_mean", ("n008",), {"window": 20, "min_periods": 20}),
    ),
    (
        _operator_node("n000", "source", parameters={"source_field": "close", "canonical_input": "adj_close"}),
        _operator_node("n001", "shift", ("n000",), {"periods": 1}),
        _operator_node("n002", "divide", ("n000", "n001")),
        _operator_node("n003", "constant", parameters={"float64_be_hex": "3ff0000000000000"}),
        _operator_node("n004", "subtract", ("n002", "n003")),
        _operator_node("n005", "rolling_std", ("n004",), {"window": 5, "min_periods": 5, "ddof": 1}),
        _operator_node("n006", "rolling_std", ("n005",), {"window": 20, "min_periods": 20, "ddof": 1}),
    ),
)


def _operator_program_content(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "program_semantic_sha256"
    }


def _operator_program_set_content(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "artifact_semantic_sha256"
    }


def _build_golden_operator_program_set() -> dict[str, Any]:
    programs: list[dict[str, Any]] = []
    for definition, adapter, nodes in zip(
        _SOURCE_DEFINITION_BASE,
        FIELD_SEMANTICS,
        _GOLDEN_PROGRAM_NODES,
        strict=True,
    ):
        program = {
            "order": definition["order"],
            "name": definition["name"],
            "direction": float(definition["direction"]),
            "definition_identity_sha256": definition[
                "definition_identity_sha256"
            ],
            "source_repository": definition["source_repository"],
            "source_commit": definition["source_commit"],
            "source_tree_oid": definition["source_tree_oid"],
            "source_relative_path": definition["source_relative_path"],
            "source_blob_oid": definition["source_blob_oid"],
            "source_raw_sha256": definition["source_raw_sha256"],
            "source_ast_sha256": definition["source_ast_sha256"],
            "field_semantics_sha256": definition["field_semantics_sha256"],
            "field_adapter": copy.deepcopy(adapter),
            "nodes": [copy.deepcopy(node) for node in nodes],
            "output_node_id": nodes[-1]["node_id"],
        }
        program["program_semantic_sha256"] = semantic_sha256_v4_4(program)
        programs.append(program)
    program_set = {
        "schema_version": OPERATOR_PROGRAM_SET_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "execution_semantics": copy.deepcopy(OPERATOR_EXECUTION_SEMANTICS),
        "candidate_count": len(programs),
        "candidates": programs,
    }
    program_set["artifact_semantic_sha256"] = semantic_sha256_v4_4(
        program_set
    )
    return program_set


def _validate_operator_node_v4_4(
    value: Any,
    *,
    index: int,
    adapter: Mapping[str, Any],
) -> dict[str, Any]:
    node = _exact_object(value, _OPERATOR_NODE_FIELDS, f"operator node[{index}]")
    expected_id = f"n{index:03d}"
    if node["node_id"] != expected_id or node["opcode"] not in _OPERATOR_OPCODES:
        raise _error("operator node id/opcode is not exact")
    inputs = node["inputs"]
    if type(inputs) is not list or any(type(item) is not str for item in inputs):
        raise _error("operator node inputs must be an ordered string list")
    prior_ids = {f"n{position:03d}" for position in range(index)}
    if any(item not in prior_ids for item in inputs):
        raise _error("operator node inputs must reference prior nodes only")
    parameters = node["parameters"]
    if type(parameters) is not dict or any(type(key) is not str for key in parameters):
        raise _error("operator node parameters must be an exact object")
    opcode = node["opcode"]
    arity = 0 if opcode in {"source", "constant"} else (
        2 if opcode in {"add", "subtract", "multiply", "divide"} else 1
    )
    if len(inputs) != arity:
        raise _error("operator node input arity mismatch")
    if opcode == "source":
        expected_fields = {"source_field", "canonical_input"}
        allowed_pairs = set(
            zip(
                adapter["source_facing_fields"],
                adapter["canonical_inputs"],
                strict=True,
            )
        )
        if set(parameters) != expected_fields or (
            parameters.get("source_field"), parameters.get("canonical_input")
        ) not in allowed_pairs:
            raise _error("operator source node differs from the exact field adapter")
    elif opcode == "constant":
        if set(parameters) != {"float64_be_hex"} or parameters.get(
            "float64_be_hex"
        ) not in _OPERATOR_CONSTANTS:
            raise _error("operator constant is not an allowed float64 BE value")
    elif opcode == "shift":
        if parameters != {"periods": 1}:
            raise _error("operator shift parameters must be exact")
    elif opcode in {"add", "subtract", "multiply", "divide", "absolute", "sign"}:
        if parameters != {}:
            raise _error("operator arithmetic parameters must be empty")
    elif opcode in {"rolling_min", "rolling_max", "rolling_mean", "rolling_std"}:
        expected_fields = {"window", "min_periods"}
        if opcode == "rolling_std":
            expected_fields.add("ddof")
        if set(parameters) != expected_fields:
            raise _error("operator rolling parameter fields are not exact")
        window = parameters.get("window")
        min_periods = parameters.get("min_periods")
        if (
            type(window) is not int
            or type(min_periods) is not int
            or window <= 0
            or min_periods <= 0
            or min_periods > window
            or (opcode == "rolling_std" and parameters.get("ddof") != 1)
            or (opcode == "rolling_std" and type(parameters.get("ddof")) is not int)
        ):
            raise _error("operator rolling parameters are invalid")
    elif opcode == "cross_section_rank":
        if parameters != _RANK_PARAMETERS:
            raise _error("operator cross-section rank parameters are not exact")
    return node


def _validate_operator_program_set_structure_v4_4(value: Any) -> dict[str, Any]:
    payload = _exact_object(
        value, _OPERATOR_PROGRAM_SET_FIELDS, "operator program set"
    )
    if (
        payload["schema_version"] != OPERATOR_PROGRAM_SET_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
    ):
        raise _error("operator program-set schema mismatch")
    if payload["execution_semantics"] != OPERATOR_EXECUTION_SEMANTICS:
        raise _error("operator execution semantics mismatch")
    programs = payload["candidates"]
    if type(programs) is not list or len(programs) != len(_SOURCE_DEFINITION_BASE):
        raise _error("operator program set must contain the ordered exact five")
    _positive_int(
        payload["candidate_count"], "operator program-set candidate count"
    )
    if payload["candidate_count"] != len(_SOURCE_DEFINITION_BASE):
        raise _error("operator program-set candidate count mismatch")
    normalized_programs: list[dict[str, Any]] = []
    for index, (value_program, definition, adapter) in enumerate(
        zip(programs, _SOURCE_DEFINITION_BASE, FIELD_SEMANTICS, strict=True)
    ):
        program = _exact_object(
            value_program, _OPERATOR_PROGRAM_FIELDS, f"operator program[{index}]"
        )
        exact_metadata = {
            "order": definition["order"],
            "name": definition["name"],
            "direction": float(definition["direction"]),
            "definition_identity_sha256": definition["definition_identity_sha256"],
            "source_repository": definition["source_repository"],
            "source_commit": definition["source_commit"],
            "source_tree_oid": definition["source_tree_oid"],
            "source_relative_path": definition["source_relative_path"],
            "source_blob_oid": definition["source_blob_oid"],
            "source_raw_sha256": definition["source_raw_sha256"],
            "source_ast_sha256": definition["source_ast_sha256"],
            "field_semantics_sha256": definition["field_semantics_sha256"],
            "field_adapter": adapter,
        }
        for field, expected in exact_metadata.items():
            actual = program[field]
            if field == "direction" and type(actual) is not float:
                raise _error("operator program direction must be an exact float")
            if canonical_json_bytes_v4_4(actual) != canonical_json_bytes_v4_4(expected):
                raise _error(f"operator program {field} differs from the fixed source")
        for field in (
            "definition_identity_sha256",
            "source_raw_sha256",
            "source_ast_sha256",
            "field_semantics_sha256",
            "program_semantic_sha256",
        ):
            _sha256(program[field], f"operator program {field}")
        for field in ("source_commit", "source_tree_oid", "source_blob_oid"):
            _oid(program[field], f"operator program {field}")
        _relative_path(program["source_relative_path"], "operator source path")
        nodes = program["nodes"]
        if type(nodes) is not list or not nodes:
            raise _error("operator program nodes must be a non-empty list")
        normalized_nodes: list[dict[str, Any]] = []
        structural_nodes: set[bytes] = set()
        for node_index, node_value in enumerate(nodes):
            node = _validate_operator_node_v4_4(
                node_value, index=node_index, adapter=adapter
            )
            structural_key = canonical_json_bytes_v4_4(
                {
                    "opcode": node["opcode"],
                    "inputs": node["inputs"],
                    "parameters": node["parameters"],
                }
            )
            if structural_key in structural_nodes:
                raise _error("operator program contains an exact structural duplicate")
            structural_nodes.add(structural_key)
            normalized_nodes.append(node)
        node_by_id = {node["node_id"]: node for node in normalized_nodes}
        output = program["output_node_id"]
        if type(output) is not str or output not in node_by_id:
            raise _error("operator output node is unavailable")
        reachable: set[str] = set()
        pending = [output]
        while pending:
            node_id = pending.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            pending.extend(node_by_id[node_id]["inputs"])
        if reachable != set(node_by_id):
            raise _error("operator program contains a dead or unreachable node")
        program["nodes"] = normalized_nodes
        supplied_program_sha = _sha256(
            program["program_semantic_sha256"], "operator program semantic SHA"
        )
        if supplied_program_sha != semantic_sha256_v4_4(
            _operator_program_content(program)
        ):
            raise _error("operator program semantic SHA mismatch")
        normalized_programs.append(program)
    payload["candidates"] = normalized_programs
    supplied_set_sha = _sha256(
        payload["artifact_semantic_sha256"], "operator program-set semantic SHA"
    )
    if supplied_set_sha != semantic_sha256_v4_4(
        _operator_program_set_content(payload)
    ):
        raise _error("operator program-set semantic SHA mismatch")
    return payload


OPERATOR_PROGRAM_SET = _validate_operator_program_set_structure_v4_4(
    _build_golden_operator_program_set()
)
OPERATOR_PROGRAM_SET_SEMANTIC_SHA256 = (
    "49a79bcba2bfe960e3cb2ca9846063c140520d36a8d5567ea60e0bc3d1c04f17"
)
if (
    OPERATOR_PROGRAM_SET["artifact_semantic_sha256"]
    != OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
):
    raise RuntimeError("frozen operator program-set semantic SHA drift")
_OPERATOR_PROGRAM_SET_CANONICAL_BYTES = canonical_json_bytes_v4_4(
    OPERATOR_PROGRAM_SET
)


def validate_operator_program_set_v4_4(value: Any) -> dict[str, Any]:
    """Validate the exact independently frozen five-program execution plan."""

    payload = _validate_operator_program_set_structure_v4_4(value)
    if canonical_json_bytes_v4_4(payload) != _OPERATOR_PROGRAM_SET_CANONICAL_BYTES:
        raise _error("operator program set differs from the fixed golden exact five")
    return payload


def operator_program_set_v4_4() -> dict[str, Any]:
    return validate_operator_program_set_v4_4(copy.deepcopy(OPERATOR_PROGRAM_SET))


_PROGRAM_BY_NAME = {
    program["name"]: program for program in OPERATOR_PROGRAM_SET["candidates"]
}
SOURCE_DEFINITION_BINDINGS = tuple(
    {
        **copy.deepcopy(definition),
        "operator_program_sha256": _PROGRAM_BY_NAME[definition["name"]][
            "program_semantic_sha256"
        ],
        "operator_program_set_sha256": OPERATOR_PROGRAM_SET_SEMANTIC_SHA256,
    }
    for definition in _SOURCE_DEFINITION_BASE
)


def _validate_preregistration(
    value: Any, *, expected_preregistration_cycle_id: str
) -> dict[str, Any]:
    fields = frozenset(
        {
            "bundle_path",
            "readback_byte_sha256",
            "readback_semantic_sha256",
            "artifact_count",
            "cycle_id",
            "candidate_rows_semantic_sha256",
        }
    )
    payload = _exact_object(value, fields, "preregistration")
    _absolute_normalized_path(payload["bundle_path"], "preregistration bundle_path")
    _sha256(payload["readback_byte_sha256"], "preregistration readback byte SHA")
    _sha256(
        payload["readback_semantic_sha256"],
        "preregistration readback semantic SHA",
    )
    _positive_int(payload["artifact_count"], "preregistration artifact count")
    if payload["artifact_count"] != PREREGISTRATION_ARTIFACT_COUNT:
        raise _error("preregistration artifact_count must be exactly 27")
    if payload["cycle_id"] != expected_preregistration_cycle_id:
        raise _error("preregistration cycle_id mismatch")
    if (
        payload["candidate_rows_semantic_sha256"]
        != EXPECTED_CANDIDATE_ROWS_SEMANTIC_SHA256
    ):
        raise _error("preregistration candidate-set SHA mismatch")
    return payload


def _validate_strict_source_expected(value: Any) -> dict[str, Any]:
    fields = frozenset(
        {
            "strict_source_binding_semantic_sha256",
            "snapshot_manifest_byte_sha256",
            "pit_generation_manifest_byte_sha256",
            "pit_membership_byte_sha256",
            "table_inventory_semantic_sha256",
            "full_a_scope_count",
            "full_a_scope_sha256",
            "source_calendar_semantic_sha256",
            "recorded_latest_pointer_byte_sha256",
            "recorded_components_byte_sha256",
        }
    )
    payload = _exact_object(value, fields, "strict_source_expected")
    for key, item in payload.items():
        if key.endswith("sha256"):
            _sha256(item, f"strict_source_expected.{key}")
    count = _positive_int(payload["full_a_scope_count"], "full-A scope count")
    if count > RESOURCE_CONTRACT["historical_symbol_count_max"]:
        raise _error("full-A scope count exceeds the fixed resource contract")
    return payload


def _validate_source_definition_bindings(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != len(SOURCE_DEFINITION_BINDINGS):
        raise _error("source_definition_bindings must be the ordered exact five")
    fields = frozenset(SOURCE_DEFINITION_BINDINGS[0])
    normalized: list[dict[str, Any]] = []
    for index, (actual, expected) in enumerate(
        zip(value, SOURCE_DEFINITION_BINDINGS, strict=True), start=1
    ):
        row = _exact_object(actual, fields, f"source_definition_bindings[{index}]")
        _positive_int(row["order"], f"source definition {index} order")
        _sha256(row["definition_identity_sha256"], "definition identity SHA")
        if type(row["direction"]) is not int or row["direction"] not in {-1, 1}:
            raise _error("source definition direction must be exact integer -1 or 1")
        _oid(row["source_commit"], "source commit")
        _oid(row["source_tree_oid"], "source tree OID")
        _relative_path(row["source_relative_path"], "source relative path")
        _oid(row["source_blob_oid"], "source blob OID")
        _sha256(row["source_raw_sha256"], "source raw SHA")
        _sha256(row["source_ast_sha256"], "source AST SHA")
        _sha256(row["field_semantics_sha256"], "field semantics SHA")
        _sha256(row["operator_program_sha256"], "operator program SHA")
        _sha256(row["operator_program_set_sha256"], "operator program-set SHA")
        if canonical_json_bytes_v4_4(row) != canonical_json_bytes_v4_4(expected):
            raise _error(f"source definition row {index} differs from fixed oracle")
        normalized.append(row)
    _reject_prohibited_keys(normalized, "source definition bindings")
    return normalized


def _validate_code_binding_set(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != len(CODE_BINDING_PATHS):
        raise _error("code_binding_set must contain the exact ordered path set")
    normalized: list[dict[str, Any]] = []
    for index, (actual, expected_path) in enumerate(
        zip(value, CODE_BINDING_PATHS, strict=True), start=1
    ):
        row = _exact_object(
            actual,
            frozenset({"relative_path", "byte_sha256"}),
            f"code_binding_set[{index}]",
        )
        _relative_path(row["relative_path"], "code binding path")
        _sha256(row["byte_sha256"], "code binding byte SHA")
        if row["relative_path"] != expected_path:
            raise _error(f"code binding row {index} path/order mismatch")
        normalized.append(row)
    return normalized


def _validate_protected_control_sha256(value: Any) -> dict[str, str]:
    payload = _exact_object(
        value, frozenset(PROTECTED_CONTROL_NAMES), "protected_control_expected_sha256"
    )
    for name in PROTECTED_CONTROL_NAMES:
        _sha256(payload[name], f"protected control {name}")
    return payload


def _validate_fixed_object(value: Any, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    payload = _exact_object(value, frozenset(expected), label)
    if canonical_json_bytes_v4_4(payload) != canonical_json_bytes_v4_4(expected):
        raise _error(f"{label} differs from the immutable contract")
    return payload


def validate_input_manifest_v4_4(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete stage-1 manifest without reading external state."""

    payload = _exact_object(value, MANIFEST_TOP_LEVEL_FIELDS, "input manifest")
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise _error("input manifest schema mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise _error("protocol_version must remain v4")
    if payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION:
        raise _error("evidence_contract_version must remain v4.4")
    cutoff, snapshot_id, _proof_output_start = (
        _validate_context_free_cycle_identity_v4_4(
            payload, label="input manifest"
        )
    )

    payload["preregistration"] = _validate_preregistration(
        payload["preregistration"],
        expected_preregistration_cycle_id=(
            deterministic_preregistration_cycle_id_v4_4(
                cutoff=cutoff, snapshot_id=snapshot_id
            )
        ),
    )
    payload["strict_source_expected"] = _validate_strict_source_expected(
        payload["strict_source_expected"]
    )
    payload["source_definition_bindings"] = _validate_source_definition_bindings(
        payload["source_definition_bindings"]
    )
    payload["code_binding_set"] = _validate_code_binding_set(
        payload["code_binding_set"]
    )
    _sha256(
        payload["runtime_binding_expected_semantic_sha256"],
        "runtime binding expected semantic SHA",
    )
    payload["protected_control_expected_sha256"] = (
        _validate_protected_control_sha256(
            payload["protected_control_expected_sha256"]
        )
    )
    payload["resource_contract"] = _validate_fixed_object(
        payload["resource_contract"], RESOURCE_CONTRACT, "resource_contract"
    )
    payload["selection_disclosures"] = _validate_fixed_object(
        payload["selection_disclosures"],
        SELECTION_DISCLOSURES,
        "selection_disclosures",
    )
    payload["negative_claims"] = _validate_fixed_object(
        payload["negative_claims"], NEGATIVE_CLAIMS, "negative_claims"
    )
    if len(canonical_file_bytes_v4_4(payload)) > RESOURCE_CONTRACT["manifest_max_bytes"]:
        raise _error("input manifest exceeds the fixed 65536-byte limit")
    return payload


def build_input_manifest_v4_4(
    *,
    cutoff: str,
    snapshot_id: str,
    proof_output_start: str,
    preregistration: Mapping[str, Any],
    strict_source_expected: Mapping[str, Any],
    code_binding_set: Sequence[Mapping[str, Any]],
    runtime_binding_expected_semantic_sha256: str,
    protected_control_expected_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build the only accepted stage-1 manifest shape."""

    cycle_id = deterministic_cycle_id_v4_4(
        cutoff=cutoff, snapshot_id=snapshot_id
    )
    return validate_input_manifest_v4_4(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "cycle_id": cycle_id,
            "cutoff": cutoff,
            "snapshot_id": snapshot_id,
            "proof_output_start": proof_output_start,
            "preregistration": copy.deepcopy(dict(preregistration)),
            "strict_source_expected": copy.deepcopy(dict(strict_source_expected)),
            "source_definition_bindings": copy.deepcopy(
                list(SOURCE_DEFINITION_BINDINGS)
            ),
            "code_binding_set": [copy.deepcopy(dict(row)) for row in code_binding_set],
            "runtime_binding_expected_semantic_sha256": (
                runtime_binding_expected_semantic_sha256
            ),
            "protected_control_expected_sha256": copy.deepcopy(
                dict(protected_control_expected_sha256)
            ),
            "resource_contract": copy.deepcopy(RESOURCE_CONTRACT),
            "selection_disclosures": copy.deepcopy(SELECTION_DISCLOSURES),
            "negative_claims": copy.deepcopy(NEGATIVE_CLAIMS),
        }
    )


def _self_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "artifact_semantic_sha256"
    }


def _seal_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    if "artifact_semantic_sha256" in payload:
        raise _error("artifact body must not supply its own semantic SHA")
    payload["artifact_semantic_sha256"] = semantic_sha256_v4_4(payload)
    return payload


def _validate_self_sha(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    supplied = _sha256(
        payload.get("artifact_semantic_sha256"),
        f"{label} artifact semantic SHA",
    )
    if supplied != semantic_sha256_v4_4(_self_payload(payload)):
        raise _error(f"{label} artifact_semantic_sha256 mismatch")
    return payload


_INPUT_MANIFEST_BINDING_FIELDS = frozenset(
    {"filename", "byte_sha256", "semantic_sha256", "size_bytes"}
)
_STAGE1_CLAIMS = {
    "manifest_canonical_and_exact": True,
    "preregistration_bundle_revalidated": True,
    "strict_source_expectations_bound": True,
    "source_definitions_revalidated": True,
    "code_bindings_revalidated": True,
    "runtime_binding_revalidated": True,
    "protected_controls_revalidated": True,
    "resource_contract_enforced": True,
    "outcome_inputs_absent": True,
    "external_state_authority_claimed": False,
}
_INPUT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "cycle_id",
        "cutoff",
        "snapshot_id",
        "proof_output_start",
        "input_manifest_binding",
        "preregistration",
        "strict_source_expected",
        "source_definition_bindings",
        "code_binding_set",
        "runtime_binding_semantic_sha256",
        "protected_control_sha256",
        "resource_contract",
        "stage1_claims",
        "selection_disclosures",
        "negative_claims",
        "artifact_semantic_sha256",
    }
)


def _expected_manifest_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_input_manifest_v4_4(manifest)
    raw = canonical_file_bytes_v4_4(normalized)
    return {
        "filename": INPUT_MANIFEST_FILENAME,
        "byte_sha256": byte_sha256_v4_4(raw),
        "semantic_sha256": semantic_sha256_v4_4(normalized),
        "size_bytes": len(raw),
    }


def validate_input_receipt_v4_4(
    value: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the stage-1 receipt against the exact copied manifest."""

    manifest_payload = validate_input_manifest_v4_4(manifest)
    payload = _exact_object(value, _INPUT_RECEIPT_FIELDS, "input receipt")
    payload = _validate_self_sha(payload, "input receipt")
    if (
        payload["schema_version"] != INPUT_RECEIPT_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
    ):
        raise _error("input receipt schema/protocol mismatch")
    for field in (
        "cycle_id",
        "cutoff",
        "snapshot_id",
        "proof_output_start",
    ):
        if payload[field] != manifest_payload[field]:
            raise _error(f"input receipt {field} does not bind the manifest")

    manifest_binding = _exact_object(
        payload["input_manifest_binding"],
        _INPUT_MANIFEST_BINDING_FIELDS,
        "input receipt manifest binding",
    )
    _sha256(manifest_binding["byte_sha256"], "manifest byte SHA")
    _sha256(manifest_binding["semantic_sha256"], "manifest semantic SHA")
    _positive_int(manifest_binding["size_bytes"], "manifest size")
    if manifest_binding != _expected_manifest_binding(manifest_payload):
        raise _error("input receipt copied-manifest binding mismatch")
    payload["input_manifest_binding"] = manifest_binding

    expected_equal = {
        "preregistration": manifest_payload["preregistration"],
        "strict_source_expected": manifest_payload["strict_source_expected"],
        "source_definition_bindings": manifest_payload[
            "source_definition_bindings"
        ],
        "code_binding_set": manifest_payload["code_binding_set"],
        "runtime_binding_semantic_sha256": manifest_payload[
            "runtime_binding_expected_semantic_sha256"
        ],
        "protected_control_sha256": manifest_payload[
            "protected_control_expected_sha256"
        ],
        "resource_contract": RESOURCE_CONTRACT,
        "stage1_claims": _STAGE1_CLAIMS,
        "selection_disclosures": SELECTION_DISCLOSURES,
        "negative_claims": NEGATIVE_CLAIMS,
    }
    for field, expected in expected_equal.items():
        if canonical_json_bytes_v4_4(payload[field]) != canonical_json_bytes_v4_4(
            expected
        ):
            raise _error(f"input receipt {field} mismatch")
    _validate_source_definition_bindings(payload["source_definition_bindings"])
    _validate_code_binding_set(payload["code_binding_set"])
    _validate_protected_control_sha256(payload["protected_control_sha256"])
    _sha256(
        payload["runtime_binding_semantic_sha256"],
        "input receipt runtime semantic SHA",
    )
    return payload


def build_input_receipt_v4_4(
    *,
    manifest: Mapping[str, Any],
    observed_preregistration: Mapping[str, Any],
    observed_code_binding_set: Sequence[Mapping[str, Any]],
    runtime_binding: Mapping[str, Any],
    observed_protected_control_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build a receipt only after every observed stage-1 identity matches."""

    normalized = validate_input_manifest_v4_4(manifest)
    runtime = validate_runtime_binding_v4_4(runtime_binding)
    runtime_semantic_sha256 = runtime["artifact_semantic_sha256"]
    prereg = copy.deepcopy(dict(observed_preregistration))
    code = [copy.deepcopy(dict(row)) for row in observed_code_binding_set]
    protected = copy.deepcopy(dict(observed_protected_control_sha256))
    comparisons = (
        ("preregistration", prereg, normalized["preregistration"]),
        ("code binding set", code, normalized["code_binding_set"]),
        (
            "runtime binding",
            runtime_semantic_sha256,
            normalized["runtime_binding_expected_semantic_sha256"],
        ),
        (
            "protected controls",
            protected,
            normalized["protected_control_expected_sha256"],
        ),
    )
    for label, actual, expected in comparisons:
        if canonical_json_bytes_v4_4(actual) != canonical_json_bytes_v4_4(expected):
            raise _error(f"observed {label} differs from stage-1 manifest")
    body = {
        "schema_version": INPUT_RECEIPT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "cycle_id": normalized["cycle_id"],
        "cutoff": normalized["cutoff"],
        "snapshot_id": normalized["snapshot_id"],
        "proof_output_start": normalized["proof_output_start"],
        "input_manifest_binding": _expected_manifest_binding(normalized),
        "preregistration": prereg,
        "strict_source_expected": copy.deepcopy(
            normalized["strict_source_expected"]
        ),
        "source_definition_bindings": copy.deepcopy(
            normalized["source_definition_bindings"]
        ),
        "code_binding_set": code,
        "runtime_binding_semantic_sha256": runtime_semantic_sha256,
        "protected_control_sha256": protected,
        "resource_contract": copy.deepcopy(RESOURCE_CONTRACT),
        "stage1_claims": copy.deepcopy(_STAGE1_CLAIMS),
        "selection_disclosures": copy.deepcopy(SELECTION_DISCLOSURES),
        "negative_claims": copy.deepcopy(NEGATIVE_CLAIMS),
    }
    return validate_input_receipt_v4_4(
        _seal_artifact(body), manifest=normalized
    )


_RUNTIME_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "python_implementation",
        "python_version",
        "python_executable_byte_sha256",
        "platform_system",
        "platform_release",
        "machine",
        "byteorder",
        "distributions",
        "execution_mode",
        "isolated_flags",
        "import_roots",
        "import_root_tree_sha256",
        "artifact_semantic_sha256",
    }
)
_RUNTIME_DISTRIBUTION_FIELDS = frozenset(
    {
        "name",
        "version",
        "distribution_file_count",
        "distribution_inventory_sha256",
        "native_binary_count",
        "native_binary_inventory_sha256",
    }
)
RUNTIME_DISTRIBUTION_NAMES = (
    "numpy",
    "pandas",
    "pyarrow",
    "python-dateutil",
    "pytz",
    "six",
)


def validate_runtime_binding_v4_4(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed, non-cyclic runtime inventory descriptor."""

    payload = _exact_object(value, _RUNTIME_BINDING_FIELDS, "runtime binding")
    payload = _validate_self_sha(payload, "runtime binding")
    if (
        payload["schema_version"] != RUNTIME_BINDING_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
    ):
        raise _error("runtime binding schema/protocol mismatch")
    for key in (
        "python_implementation",
        "python_version",
        "platform_system",
        "platform_release",
        "machine",
    ):
        if type(payload[key]) is not str or not payload[key]:
            raise _error(f"runtime binding {key} must be a non-empty string")
    _sha256(
        payload["python_executable_byte_sha256"],
        "Python executable byte SHA",
    )
    if payload["byteorder"] != "little":
        raise _error("strict runtime requires little-endian byte order")
    if payload["execution_mode"] != RUNTIME_EXECUTION_MODE:
        raise _error("runtime execution mode is not the sealed shadow mode")
    if payload["isolated_flags"] != list(RUNTIME_ISOLATED_FLAGS):
        raise _error("runtime isolated flags must be exact -B -I -S")
    if canonical_json_bytes_v4_4(payload["import_roots"]) != canonical_json_bytes_v4_4(
        list(RUNTIME_IMPORT_ROOTS)
    ):
        raise _error("runtime import roots differ from the frozen contract inventory")
    _sha256(payload["import_root_tree_sha256"], "runtime import-root tree SHA")
    rows = payload["distributions"]
    if type(rows) is not list or len(rows) != len(RUNTIME_DISTRIBUTION_NAMES):
        raise _error(
            "runtime binding requires the exact six-distribution closure"
        )
    normalized_rows: list[dict[str, Any]] = []
    for index, (row_value, expected_name) in enumerate(
        zip(rows, RUNTIME_DISTRIBUTION_NAMES, strict=True), start=1
    ):
        row = _exact_object(
            row_value,
            _RUNTIME_DISTRIBUTION_FIELDS,
            f"runtime distribution[{index}]",
        )
        if row["name"] != expected_name:
            raise _error("runtime distribution name/order mismatch")
        if type(row["version"]) is not str or not row["version"]:
            raise _error("runtime distribution version must be non-empty")
        _positive_int(
            row["distribution_file_count"],
            "runtime distribution file count",
        )
        _sha256(
            row["distribution_inventory_sha256"],
            "runtime distribution inventory SHA",
        )
        _nonnegative_int(
            row["native_binary_count"], "runtime native binary count"
        )
        _sha256(
            row["native_binary_inventory_sha256"],
            "runtime native binary inventory SHA",
        )
        normalized_rows.append(row)
    payload["distributions"] = normalized_rows
    return payload


def build_runtime_binding_v4_4(
    *,
    python_implementation: str,
    python_version: str,
    python_executable_byte_sha256: str,
    platform_system: str,
    platform_release: str,
    machine: str,
    byteorder: str,
    distributions: Sequence[Mapping[str, Any]],
    import_root_tree_sha256: str,
) -> dict[str, Any]:
    """Build a runtime identity that never embeds or hashes the manifest."""

    body = {
        "schema_version": RUNTIME_BINDING_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "python_implementation": python_implementation,
        "python_version": python_version,
        "python_executable_byte_sha256": python_executable_byte_sha256,
        "platform_system": platform_system,
        "platform_release": platform_release,
        "machine": machine,
        "byteorder": byteorder,
        "distributions": [copy.deepcopy(dict(row)) for row in distributions],
        "execution_mode": RUNTIME_EXECUTION_MODE,
        "isolated_flags": list(RUNTIME_ISOLATED_FLAGS),
        "import_roots": copy.deepcopy(list(RUNTIME_IMPORT_ROOTS)),
        "import_root_tree_sha256": import_root_tree_sha256,
    }
    return validate_runtime_binding_v4_4(_seal_artifact(body))


MATRIX_DESCRIPTOR_SCHEMA_VERSION = (
    "factor-governance-future-strict-matrix-descriptor.v4.4"
)
BINARY_MASK_DESCRIPTOR_SCHEMA_VERSION = (
    "factor-governance-future-strict-binary-mask-descriptor.v4.4"
)
CANDIDATE_NON_NULL_MASK_SET_SCHEMA_VERSION = (
    "factor-governance-future-strict-candidate-non-null-mask-set.v4.4"
)
BLOCK_MANIFEST_SCHEMA_VERSION = (
    "factor-governance-future-strict-block-manifest.v4.4"
)
STRICT_SOURCE_EVIDENCE_STATUS = "HEALTHY_STRICT_SOURCE_EVIDENCE_ONLY"
AXIS_HASH_ALGORITHM = "sha256_utf8_lines_with_trailing_newline"
FULL_A_HASH_ALGORITHM = (
    "sha256_utf8_sorted_unique_symbols_newline_join_without_trailing_newline"
)

_AXIS_DESCRIPTOR_FIELDS = frozenset({"count", "sha256", "first", "last"})
_MATRIX_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "dtype",
        "layout",
        "row_count",
        "column_count",
        "date_axis",
        "symbol_axis",
        "matrix_sha256",
        "bit_pattern_sha256",
        "magnitude_bits_sha256",
        "elementwise_negated_sha256",
        "finite_count",
        "nan_count",
        "positive_infinity_count",
        "negative_infinity_count",
        "positive_finite_count",
        "negative_finite_count",
        "positive_zero_count",
        "negative_zero_count",
        "byte_count",
    }
)
_BINARY_MASK_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "value_type",
        "layout",
        "encoding",
        "row_count",
        "column_count",
        "date_axis",
        "symbol_axis",
        "bit_count",
        "zero_count",
        "one_count",
        "packed_byte_count",
        "padding_bit_count",
        "padding_bits_zero",
        "packed_bits_base64",
        "packed_bits_sha256",
        "binary_values_only",
    }
)


def _axis_descriptor(values: Sequence[str]) -> dict[str, Any]:
    items = list(values)
    raw = b"".join(item.encode("utf-8") + b"\n" for item in items)
    return {
        "count": len(items),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "first": items[0] if items else None,
        "last": items[-1] if items else None,
    }


def _canonical_symbol_values(
    values: Sequence[str], label: str
) -> list[str]:
    if type(values) not in {list, tuple} or not values:
        raise _error(f"{label} must be a non-empty list or tuple")
    normalized = list(values)
    if any(
        type(item) is not str or _CN_SYMBOL_RE.fullmatch(item) is None
        for item in normalized
    ):
        raise _error(f"{label} must contain canonical CN symbols")
    if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
        raise _error(f"{label} must be sorted and unique")
    return normalized


def _decode_binary_mask_descriptor_v4_4(
    value: Any,
    *,
    label: str = "binary mask descriptor",
    expected_date_axis: Mapping[str, Any] | None = None,
    expected_symbol_axis: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Validate and decode one canonical row-major little-bit-order bitmap."""

    payload = _exact_object(value, _BINARY_MASK_DESCRIPTOR_FIELDS, label)
    if (
        payload["schema_version"] != BINARY_MASK_DESCRIPTOR_SCHEMA_VERSION
        or payload["value_type"] != "bool"
        or payload["layout"] != "row-major"
        or payload["encoding"] != "base64_packbits_little_bit_order"
        or payload["binary_values_only"] is not True
        or payload["padding_bits_zero"] is not True
    ):
        raise _error(f"{label} schema/dtype/encoding mismatch")
    for field in (
        "row_count",
        "column_count",
        "bit_count",
        "zero_count",
        "one_count",
        "packed_byte_count",
        "padding_bit_count",
    ):
        _nonnegative_int(payload[field], f"{label}.{field}")
    if payload["row_count"] <= 0 or payload["column_count"] <= 0:
        raise _error(f"{label} shape must be non-empty")
    date_axis = _validate_axis_descriptor(
        payload["date_axis"], f"{label} date axis"
    )
    symbol_axis = _validate_axis_descriptor(
        payload["symbol_axis"], f"{label} symbol axis"
    )
    cells = payload["row_count"] * payload["column_count"]
    expected_bytes = (cells + 7) // 8
    expected_padding = expected_bytes * 8 - cells
    if (
        date_axis["count"] != payload["row_count"]
        or symbol_axis["count"] != payload["column_count"]
        or payload["bit_count"] != cells
        or payload["packed_byte_count"] != expected_bytes
        or payload["padding_bit_count"] != expected_padding
        or payload["zero_count"] + payload["one_count"] != cells
    ):
        raise _error(f"{label} shape/value accounting mismatch")
    if expected_bytes > (
        RESOURCE_CONTRACT["projected_row_count_per_pass_max"] + 7
    ) // 8:
        raise _error(f"{label} decoded bitmap exceeds the fixed resource cap")
    if expected_date_axis is not None and date_axis != dict(expected_date_axis):
        raise _error(f"{label} date axis mismatch")
    if expected_symbol_axis is not None and symbol_axis != dict(expected_symbol_axis):
        raise _error(f"{label} symbol axis mismatch")
    encoded = payload["packed_bits_base64"]
    if type(encoded) is not str or not encoded:
        raise _error(f"{label} packed bitmap must be non-empty base64")
    try:
        packed = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise _error(f"{label} packed bitmap is not canonical base64") from exc
    if (
        len(packed) != expected_bytes
        or base64.b64encode(packed).decode("ascii") != encoded
    ):
        raise _error(f"{label} packed bitmap length/encoding mismatch")
    if expected_padding:
        used_low_bits = 8 - expected_padding
        if packed[-1] & ~((1 << used_low_bits) - 1):
            raise _error(f"{label} non-zero padding bits are forbidden")
    if sum(byte.bit_count() for byte in packed) != payload["one_count"]:
        raise _error(f"{label} bitmap popcount mismatch")
    supplied_sha = _sha256(
        payload["packed_bits_sha256"], f"{label} packed-bits SHA"
    )
    if supplied_sha != byte_sha256_v4_4(packed):
        raise _error(f"{label} packed-bits SHA mismatch")
    payload["date_axis"] = date_axis
    payload["symbol_axis"] = symbol_axis
    return payload, packed


def validate_binary_mask_descriptor_v4_4(
    value: Any,
    *,
    label: str = "binary mask descriptor",
    expected_date_axis: Mapping[str, Any] | None = None,
    expected_symbol_axis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a self-contained canonical packed binary bitmap."""

    payload, _packed = _decode_binary_mask_descriptor_v4_4(
        value,
        label=label,
        expected_date_axis=expected_date_axis,
        expected_symbol_axis=expected_symbol_axis,
    )
    return payload


def build_packed_binary_mask_descriptor_v4_4(
    *,
    packed_bits: bytes,
    bit_count: int,
    dates: Sequence[str],
    symbols: Sequence[str],
) -> dict[str, Any]:
    """Build a self-contained descriptor from canonical little-order bits."""

    if type(packed_bits) is not bytes:
        raise _error("packed binary mask must be exact bytes")
    normalized_dates = [
        _canonical_date(item, f"binary mask date[{index}]")
        for index, item in enumerate(dates)
    ]
    if (
        not normalized_dates
        or normalized_dates != sorted(normalized_dates)
        or len(set(normalized_dates)) != len(normalized_dates)
    ):
        raise _error("binary mask dates must be strictly increasing and unique")
    normalized_symbols = _canonical_symbol_values(symbols, "binary mask symbols")
    cells = len(normalized_dates) * len(normalized_symbols)
    if type(bit_count) is not int or bit_count != cells:
        raise _error("packed binary mask bit count must equal the matrix cell count")
    expected_bytes = (cells + 7) // 8
    padding = expected_bytes * 8 - cells
    if len(packed_bits) != expected_bytes:
        raise _error("packed binary mask byte count mismatch")
    if padding and packed_bits[-1] & ~((1 << (8 - padding)) - 1):
        raise _error("packed binary mask padding bits must be zero")
    one_count = sum(byte.bit_count() for byte in packed_bits)
    body = {
        "schema_version": BINARY_MASK_DESCRIPTOR_SCHEMA_VERSION,
        "value_type": "bool",
        "layout": "row-major",
        "encoding": "base64_packbits_little_bit_order",
        "row_count": len(normalized_dates),
        "column_count": len(normalized_symbols),
        "date_axis": _axis_descriptor(normalized_dates),
        "symbol_axis": _axis_descriptor(normalized_symbols),
        "bit_count": cells,
        "zero_count": cells - one_count,
        "one_count": one_count,
        "packed_byte_count": expected_bytes,
        "padding_bit_count": padding,
        "padding_bits_zero": True,
        "packed_bits_base64": base64.b64encode(packed_bits).decode("ascii"),
        "packed_bits_sha256": byte_sha256_v4_4(packed_bits),
        "binary_values_only": True,
    }
    return validate_binary_mask_descriptor_v4_4(body)


def build_binary_mask_descriptor_v4_4(
    *,
    uint8_values: bytes,
    dates: Sequence[str],
    symbols: Sequence[str],
) -> dict[str, Any]:
    """Pack exact row-major 0/1 bytes into a canonical bitmap descriptor."""

    if type(uint8_values) is not bytes:
        raise _error("binary mask values must be exact bytes")
    normalized_dates = [
        _canonical_date(item, f"binary mask date[{index}]")
        for index, item in enumerate(dates)
    ]
    if (
        not normalized_dates
        or normalized_dates != sorted(normalized_dates)
        or len(set(normalized_dates)) != len(normalized_dates)
    ):
        raise _error("binary mask dates must be strictly increasing and unique")
    normalized_symbols = _canonical_symbol_values(symbols, "binary mask symbols")
    cells = len(normalized_dates) * len(normalized_symbols)
    if len(uint8_values) != cells or any(item not in {0, 1} for item in uint8_values):
        raise _error("binary mask bytes must contain exactly one 0/1 byte per cell")
    packed = bytearray((cells + 7) // 8)
    for index, item in enumerate(uint8_values):
        if item:
            packed[index // 8] |= 1 << (index % 8)
    return build_packed_binary_mask_descriptor_v4_4(
        packed_bits=bytes(packed),
        bit_count=cells,
        dates=normalized_dates,
        symbols=normalized_symbols,
    )


def source_calendar_semantic_sha256_v4_4(
    open_sessions: Sequence[str], *, cutoff: str
) -> str:
    sessions = [
        _canonical_date(item, f"source calendar[{index}]")
        for index, item in enumerate(open_sessions)
    ]
    if not sessions:
        raise _error("source calendar must not be empty")
    if sessions != sorted(sessions) or len(set(sessions)) != len(sessions):
        raise _error("source calendar must be strictly increasing and unique")
    normalized_cutoff = _canonical_date(cutoff, "source calendar cutoff")
    if sessions[-1] != normalized_cutoff:
        raise _error("source calendar must end at cutoff")
    return semantic_sha256_v4_4(
        {
            "analysis_start": sessions[0],
            "cutoff_date": normalized_cutoff,
            "open_session_count": len(sessions),
            "open_sessions": sessions,
        }
    )


def _validate_axis_descriptor(
    value: Any,
    label: str,
    *,
    expected_values: Sequence[str] | None = None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    payload = _exact_object(value, _AXIS_DESCRIPTOR_FIELDS, label)
    count = _nonnegative_int(payload["count"], f"{label} count")
    if count == 0 and not allow_empty:
        raise _error(f"{label} must not be empty")
    _sha256(payload["sha256"], f"{label} SHA")
    if count == 0:
        if payload["first"] is not None or payload["last"] is not None:
            raise _error(f"{label} empty endpoints must be null")
    elif type(payload["first"]) is not str or type(payload["last"]) is not str:
        raise _error(f"{label} non-empty endpoints must be strings")
    if expected_values is not None and payload != _axis_descriptor(expected_values):
        raise _error(f"{label} does not bind the supplied ordered values")
    return payload


def validate_matrix_descriptor_v4_4(
    value: Any,
    *,
    label: str = "matrix descriptor",
    expected_date_axis: Mapping[str, Any] | None = None,
    expected_symbol_axis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Structurally validate an evaluator bit-pattern descriptor."""

    payload = _exact_object(value, _MATRIX_DESCRIPTOR_FIELDS, label)
    if (
        payload["schema_version"] != MATRIX_DESCRIPTOR_SCHEMA_VERSION
        or payload["dtype"] != "float64-le"
        or payload["layout"] != "row-major"
    ):
        raise _error(f"{label} schema/dtype/layout mismatch")
    integer_fields = (
        "row_count",
        "column_count",
        "finite_count",
        "nan_count",
        "positive_infinity_count",
        "negative_infinity_count",
        "positive_finite_count",
        "negative_finite_count",
        "positive_zero_count",
        "negative_zero_count",
        "byte_count",
    )
    for field in integer_fields:
        _nonnegative_int(payload[field], f"{label}.{field}")
    if payload["row_count"] <= 0 or payload["column_count"] <= 0:
        raise _error(f"{label} shape must be non-empty")
    date_axis = _validate_axis_descriptor(payload["date_axis"], f"{label} date axis")
    symbol_axis = _validate_axis_descriptor(
        payload["symbol_axis"], f"{label} symbol axis"
    )
    if (
        date_axis["count"] != payload["row_count"]
        or symbol_axis["count"] != payload["column_count"]
    ):
        raise _error(f"{label} shape/axis count mismatch")
    if expected_date_axis is not None and date_axis != dict(expected_date_axis):
        raise _error(f"{label} date axis mismatch")
    if expected_symbol_axis is not None and symbol_axis != dict(expected_symbol_axis):
        raise _error(f"{label} symbol axis mismatch")
    for field in (
        "matrix_sha256",
        "bit_pattern_sha256",
        "magnitude_bits_sha256",
        "elementwise_negated_sha256",
    ):
        _sha256(payload[field], f"{label}.{field}")
    if payload["matrix_sha256"] != payload["bit_pattern_sha256"]:
        raise _error(f"{label} bit-pattern identity mismatch")
    cells = payload["row_count"] * payload["column_count"]
    if payload["byte_count"] != cells * 8:
        raise _error(f"{label} byte count mismatch")
    if (
        payload["finite_count"]
        + payload["nan_count"]
        + payload["positive_infinity_count"]
        + payload["negative_infinity_count"]
        != cells
    ):
        raise _error(f"{label} observation accounting mismatch")
    if (
        payload["positive_finite_count"]
        + payload["negative_finite_count"]
        + payload["positive_zero_count"]
        + payload["negative_zero_count"]
        != payload["finite_count"]
    ):
        raise _error(f"{label} finite-sign accounting mismatch")
    return payload


_HISTORICAL_SYMBOL_AXIS_FIELDS = frozenset(
    {
        "scope",
        "cutoff_only",
        "contains_all_cutoff_full_a",
        "historical_only_symbol_count",
        "hash_algorithm",
        "descriptor",
    }
)
_PIT_MEMBERSHIP_FIELDS = frozenset(
    {
        "row_count",
        "distinct_symbol_count",
        "historical_union_symbol_count",
        "duplicate_symbol_count",
        "one_row_per_symbol",
        "effective_from_semantics",
        "effective_to_semantics",
        "blank_effective_to_semantics",
        "membership_byte_sha256",
    }
)
_CUTOFF_FULL_A_FIELDS = frozenset({"count", "sha256", "hash_algorithm"})
_SOURCE_CALENDAR_FIELDS = frozenset({"open_sessions", "descriptor"})
_SOURCE_ACCESS_FIELDS = frozenset(
    {
        "recorded_latest_pointer_byte_sha256",
        "recorded_components_byte_sha256",
        "current_pointer_read",
        "current_components_read",
        "serving_read",
        "csv_read",
    }
)
_BLOCK_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "halo",
        "output_block",
        "source_row_count",
        "proof_output_row_count",
        "source_calendar",
        "proof_output_calendar",
        "date_axis",
        "proof_output_date_axis",
        "symbol_axis",
        "full_historical_symbols",
        "block_count",
        "blocks",
        "manifest_semantic_sha256",
    }
)
_BLOCK_ROW_FIELDS = frozenset(
    {
        "block_index",
        "input_start_offset",
        "input_end_offset",
        "output_start_offset",
        "output_end_offset",
        "local_output_start_offset",
        "local_output_end_offset",
        "input_row_count",
        "output_row_count",
        "input_first_date",
        "input_last_date",
        "output_first_date",
        "output_last_date",
        "symbol_axis",
        "future_halo_row_count",
    }
)
_DATA_FIELD_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "cycle_id",
        "cutoff",
        "snapshot_id",
        "proof_output_start",
        "input_manifest_semantic_sha256",
        "input_receipt_semantic_sha256",
        "strict_source_binding",
        "source_calendar",
        "proof_output_calendar",
        "historical_date_axis_descriptor",
        "historical_symbol_axis",
        "pit_membership_contract",
        "pit_mask_descriptor",
        "cutoff_full_a_scope",
        "block_manifest",
        "table_projection",
        "field_adapters",
        "operator_program_set",
        "operator_program_set_semantic_sha256",
        "field_missing_counts",
        "bars_outside_pit_interval_count",
        "ignored_pre_analysis_row_count",
        "outside_pit_non_null_counts",
        "projected_row_count_per_pass",
        "source_access",
        "strict_source_evidence_status",
        "selection_disclosures",
        "negative_claims",
        "artifact_semantic_sha256",
    }
)


def _expected_block_manifest_v4_4(
    source_sessions: Sequence[str], symbols: Sequence[str]
) -> dict[str, Any]:
    sessions = list(source_sessions)
    full_symbols = list(symbols)
    halo = RESOURCE_CONTRACT["halo_session_count"]
    output_block = RESOURCE_CONTRACT["output_block_session_count"]
    symbol_axis = _axis_descriptor(full_symbols)
    blocks: list[dict[str, Any]] = []
    for block_index, output_start in enumerate(
        range(halo, len(sessions), output_block)
    ):
        output_end = min(output_start + output_block, len(sessions))
        input_start = output_start - halo
        input_end = output_end
        blocks.append(
            {
                "block_index": block_index,
                "input_start_offset": input_start,
                "input_end_offset": input_end,
                "output_start_offset": output_start,
                "output_end_offset": output_end,
                "local_output_start_offset": halo,
                "local_output_end_offset": halo + output_end - output_start,
                "input_row_count": input_end - input_start,
                "output_row_count": output_end - output_start,
                "input_first_date": sessions[input_start],
                "input_last_date": sessions[input_end - 1],
                "output_first_date": sessions[output_start],
                "output_last_date": sessions[output_end - 1],
                "symbol_axis": copy.deepcopy(symbol_axis),
                "future_halo_row_count": 0,
            }
        )
    body = {
        "schema_version": BLOCK_MANIFEST_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "halo": halo,
        "output_block": output_block,
        "source_row_count": len(sessions),
        "proof_output_row_count": len(sessions) - halo,
        "source_calendar": sessions,
        "proof_output_calendar": sessions[halo:],
        "date_axis": _axis_descriptor(sessions),
        "proof_output_date_axis": _axis_descriptor(sessions[halo:]),
        "symbol_axis": symbol_axis,
        "full_historical_symbols": full_symbols,
        "block_count": len(blocks),
        "blocks": blocks,
    }
    body["manifest_semantic_sha256"] = semantic_sha256_v4_4(body)
    return body


def validate_block_manifest_v4_4(
    value: Any,
    *,
    source_sessions: Sequence[str],
    expected_symbol_axis: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the exact 60/128 partition and reject any row-level drift."""

    payload = _exact_object(value, _BLOCK_MANIFEST_FIELDS, "block manifest")
    symbols_value = payload["full_historical_symbols"]
    if type(symbols_value) is not list or len(symbols_value) > RESOURCE_CONTRACT[
        "historical_symbol_count_max"
    ]:
        raise _error("block manifest historical symbol inventory is invalid")
    symbols = _canonical_symbol_values(
        symbols_value, "block manifest historical symbols"
    )
    if _axis_descriptor(symbols) != dict(expected_symbol_axis):
        raise _error("block manifest historical symbol axis mismatch")
    rows = payload["blocks"]
    if type(rows) is not list:
        raise _error("block manifest blocks must be a list")
    for index, row in enumerate(rows):
        _exact_object(row, _BLOCK_ROW_FIELDS, f"block manifest row[{index}]")
    expected = _expected_block_manifest_v4_4(source_sessions, symbols)
    if canonical_json_bytes_v4_4(payload) != canonical_json_bytes_v4_4(expected):
        raise _error(
            "block manifest must be the exact-once past-only deterministic partition"
        )
    return payload


def _validate_source_calendar(
    value: Any, *, cutoff: str, proof_output_start: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _exact_object(value, _SOURCE_CALENDAR_FIELDS, "source calendar")
    sessions = payload["open_sessions"]
    if type(sessions) is not list or len(sessions) < RESOURCE_CONTRACT[
        "halo_session_count"
    ] + 1:
        raise _error("source calendar requires at least 61 open sessions")
    normalized = [
        _canonical_date(item, f"source calendar[{index}]")
        for index, item in enumerate(sessions)
    ]
    if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
        raise _error("source calendar must be strictly increasing and unique")
    if len(normalized) > RESOURCE_CONTRACT["source_session_count_max"]:
        raise _error("source calendar exceeds the fixed session limit")
    if normalized[-1] != cutoff:
        raise _error("source calendar must end exactly at cutoff")
    halo = RESOURCE_CONTRACT["halo_session_count"]
    if normalized[halo] != proof_output_start:
        raise _error(
            "proof_output_start must equal accepted source calendar open_sessions[60]"
        )
    descriptor = _validate_axis_descriptor(
        payload["descriptor"],
        "source calendar descriptor",
        expected_values=normalized,
    )
    payload["open_sessions"] = normalized
    payload["descriptor"] = descriptor
    return payload, _axis_descriptor(normalized[halo:])


def _validate_historical_symbol_axis(
    value: Any, *, cutoff_full_a_count: int
) -> dict[str, Any]:
    payload = _exact_object(
        value, _HISTORICAL_SYMBOL_AXIS_FIELDS, "historical symbol axis"
    )
    descriptor = _validate_axis_descriptor(
        payload["descriptor"], "historical symbol axis descriptor"
    )
    historical_only = _positive_int(
        payload["historical_only_symbol_count"],
        "historical-only symbol count",
    )
    if (
        payload["scope"] != "all_historical_pit_symbols"
        or payload["cutoff_only"] is not False
        or payload["contains_all_cutoff_full_a"] is not True
        or payload["hash_algorithm"] != AXIS_HASH_ALGORITHM
        or descriptor["count"] != cutoff_full_a_count + historical_only
    ):
        raise _error("historical symbol axis is a cutoff-only or malformed axis")
    if descriptor["count"] > RESOURCE_CONTRACT["historical_symbol_count_max"]:
        raise _error("historical symbol axis exceeds fixed resource contract")
    payload["descriptor"] = descriptor
    return payload


def _validate_pit_membership(
    value: Any, *, historical_symbol_count: int, expected_byte_sha256: str
) -> dict[str, Any]:
    payload = _exact_object(value, _PIT_MEMBERSHIP_FIELDS, "PIT membership")
    row_count = _positive_int(payload["row_count"], "PIT membership row count")
    distinct = _positive_int(
        payload["distinct_symbol_count"], "PIT distinct symbol count"
    )
    historical_union = _positive_int(
        payload["historical_union_symbol_count"],
        "PIT historical-union symbol count",
    )
    duplicates = _nonnegative_int(
        payload["duplicate_symbol_count"], "PIT duplicate symbol count"
    )
    if (
        row_count != distinct
        or duplicates != 0
        or payload["one_row_per_symbol"] is not True
        or historical_union != historical_symbol_count
        or historical_union > distinct
    ):
        raise _error(
            "PIT membership must contain one row per PIT symbol and bind the "
            "calendar-overlapping historical union"
        )
    if row_count > RESOURCE_CONTRACT["pit_row_count_max"]:
        raise _error("PIT membership exceeds the fixed row limit")
    if (
        payload["effective_from_semantics"] != "inclusive"
        or payload["effective_to_semantics"] != "exclusive"
        or payload["blank_effective_to_semantics"] != "positive_infinity"
    ):
        raise _error("PIT interval semantics mismatch")
    if payload["membership_byte_sha256"] != expected_byte_sha256:
        raise _error("PIT membership byte SHA differs from strict source binding")
    return payload


def _validate_count_map(value: Any, label: str, *, require_zero: bool) -> dict[str, int]:
    payload = _exact_object(value, frozenset(INPUT_FIELDS), label)
    for field in INPUT_FIELDS:
        count = _nonnegative_int(payload[field], f"{label}.{field}")
        if require_zero and count != 0:
            raise _error(f"{label}.{field} must be zero")
    return payload


def validate_data_field_receipt_v4_4(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    input_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate strict source, historical axes, PIT, and field semantics."""

    manifest_payload = validate_input_manifest_v4_4(manifest)
    input_payload = validate_input_receipt_v4_4(
        input_receipt, manifest=manifest_payload
    )
    payload = _exact_object(
        value, _DATA_FIELD_RECEIPT_FIELDS, "data/field receipt"
    )
    payload = _validate_self_sha(payload, "data/field receipt")
    if (
        payload["schema_version"] != DATA_FIELD_RECEIPT_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
    ):
        raise _error("data/field receipt schema/protocol mismatch")
    for field in ("cycle_id", "cutoff", "snapshot_id", "proof_output_start"):
        if payload[field] != manifest_payload[field]:
            raise _error(f"data/field receipt {field} mismatch")
    if payload["input_manifest_semantic_sha256"] != semantic_sha256_v4_4(
        manifest_payload
    ):
        raise _error("data/field receipt manifest semantic SHA mismatch")
    if payload["input_receipt_semantic_sha256"] != input_payload[
        "artifact_semantic_sha256"
    ]:
        raise _error("data/field receipt input-receipt SHA mismatch")
    if payload["strict_source_binding"] != manifest_payload[
        "strict_source_expected"
    ]:
        raise _error("data/field receipt strict source binding mismatch")
    strict_source = manifest_payload["strict_source_expected"]
    source_calendar, expected_proof_calendar = _validate_source_calendar(
        payload["source_calendar"],
        cutoff=payload["cutoff"],
        proof_output_start=payload["proof_output_start"],
    )
    proof_calendar = _validate_axis_descriptor(
        payload["proof_output_calendar"], "proof output calendar"
    )
    if proof_calendar != expected_proof_calendar:
        raise _error("proof output calendar does not equal source calendar[60:]")
    historical_date_axis = _validate_axis_descriptor(
        payload["historical_date_axis_descriptor"], "historical date axis"
    )
    if historical_date_axis != source_calendar["descriptor"]:
        raise _error("historical date axis must equal the full source calendar")
    if strict_source["source_calendar_semantic_sha256"] != (
        source_calendar_semantic_sha256_v4_4(
            source_calendar["open_sessions"], cutoff=payload["cutoff"]
        )
    ):
        raise _error("source calendar semantic SHA differs from preregistration")

    cutoff_scope = _exact_object(
        payload["cutoff_full_a_scope"],
        _CUTOFF_FULL_A_FIELDS,
        "cutoff full-A scope",
    )
    _positive_int(cutoff_scope["count"], "cutoff full-A scope count")
    if (
        cutoff_scope["count"] != strict_source["full_a_scope_count"]
        or cutoff_scope["sha256"] != strict_source["full_a_scope_sha256"]
        or cutoff_scope["hash_algorithm"] != FULL_A_HASH_ALGORITHM
    ):
        raise _error("cutoff full-A scope/count/SHA contract mismatch")
    historical_symbols = _validate_historical_symbol_axis(
        payload["historical_symbol_axis"],
        cutoff_full_a_count=cutoff_scope["count"],
    )
    if historical_symbols["descriptor"]["sha256"] == cutoff_scope["sha256"]:
        raise _error("full-A scope SHA must be distinct from matrix-axis SHA")
    pit_membership = _validate_pit_membership(
        payload["pit_membership_contract"],
        historical_symbol_count=historical_symbols["descriptor"]["count"],
        expected_byte_sha256=strict_source["pit_membership_byte_sha256"],
    )
    pit_descriptor = validate_binary_mask_descriptor_v4_4(
        payload["pit_mask_descriptor"],
        label="PIT mask descriptor",
        expected_date_axis=historical_date_axis,
        expected_symbol_axis=historical_symbols["descriptor"],
    )
    if (
        pit_descriptor["one_count"] <= 0
        or pit_descriptor["zero_count"]
        < historical_symbols["historical_only_symbol_count"]
    ):
        raise _error("PIT bitmap has invalid eligible/ineligible accounting")

    block_manifest = validate_block_manifest_v4_4(
        payload["block_manifest"],
        source_sessions=source_calendar["open_sessions"],
        expected_symbol_axis=historical_symbols["descriptor"],
    )
    if payload["table_projection"] != list(TABLE_PROJECTION):
        raise _error("table projection must be the exact ordered six fields")
    if payload["field_adapters"] != list(FIELD_SEMANTICS):
        raise _error("field adapters differ from the fixed exact-five semantics")
    operator_program_set = validate_operator_program_set_v4_4(
        payload["operator_program_set"]
    )
    if (
        payload["operator_program_set_semantic_sha256"]
        != operator_program_set["artifact_semantic_sha256"]
        or payload["operator_program_set_semantic_sha256"]
        != OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
    ):
        raise _error("data/field receipt operator program-set SHA mismatch")
    missing = _validate_count_map(
        payload["field_missing_counts"], "field missing counts", require_zero=False
    )
    outside = _validate_count_map(
        payload["outside_pit_non_null_counts"],
        "outside-PIT non-null counts",
        require_zero=True,
    )
    _nonnegative_int(
        payload["bars_outside_pit_interval_count"],
        "bars outside valid PIT interval count",
    )
    _nonnegative_int(
        payload["ignored_pre_analysis_row_count"],
        "ignored pre-analysis row count",
    )
    projected_rows = _positive_int(
        payload["projected_row_count_per_pass"],
        "projected row count per pass",
    )
    expected_rows = historical_date_axis["count"] * historical_symbols[
        "descriptor"
    ]["count"]
    if projected_rows != expected_rows or projected_rows > RESOURCE_CONTRACT[
        "projected_row_count_per_pass_max"
    ]:
        raise _error("projected row count per pass mismatch or resource overflow")
    if pit_descriptor["bit_count"] != projected_rows:
        raise _error("PIT bitmap cell count differs from projected rows")
    source_access = _exact_object(
        payload["source_access"], _SOURCE_ACCESS_FIELDS, "source access"
    )
    if (
        source_access["recorded_latest_pointer_byte_sha256"]
        != strict_source["recorded_latest_pointer_byte_sha256"]
        or source_access["recorded_components_byte_sha256"]
        != strict_source["recorded_components_byte_sha256"]
        or source_access["current_pointer_read"] is not False
        or source_access["current_components_read"] is not False
        or source_access["serving_read"] is not False
        or source_access["csv_read"] is not False
    ):
        raise _error("source access must bind prereg-recorded bytes without fallbacks")
    if payload["strict_source_evidence_status"] != STRICT_SOURCE_EVIDENCE_STATUS:
        raise _error("healthy strict source may be recorded only as evidence status")
    payload["selection_disclosures"] = _validate_fixed_object(
        payload["selection_disclosures"],
        SELECTION_DISCLOSURES,
        "data/field receipt selection disclosures",
    )
    payload["negative_claims"] = _validate_fixed_object(
        payload["negative_claims"],
        NEGATIVE_CLAIMS,
        "data/field receipt negative claims",
    )

    payload["source_calendar"] = source_calendar
    payload["proof_output_calendar"] = proof_calendar
    payload["historical_date_axis_descriptor"] = historical_date_axis
    payload["historical_symbol_axis"] = historical_symbols
    payload["pit_membership_contract"] = pit_membership
    payload["pit_mask_descriptor"] = pit_descriptor
    payload["block_manifest"] = block_manifest
    payload["operator_program_set"] = operator_program_set
    payload["field_missing_counts"] = missing
    payload["outside_pit_non_null_counts"] = outside
    payload["source_access"] = source_access
    return payload


def build_data_field_receipt_v4_4(
    *,
    manifest: Mapping[str, Any],
    input_receipt: Mapping[str, Any],
    source_calendar_open_sessions: Sequence[str],
    historical_symbol_axis: Mapping[str, Any],
    pit_membership_contract: Mapping[str, Any],
    pit_mask_descriptor: Mapping[str, Any],
    block_manifest: Mapping[str, Any],
    field_missing_counts: Mapping[str, int],
    bars_outside_pit_interval_count: int,
    ignored_pre_analysis_row_count: int,
    outside_pit_non_null_counts: Mapping[str, int],
    projected_row_count_per_pass: int,
) -> dict[str, Any]:
    manifest_payload = validate_input_manifest_v4_4(manifest)
    input_payload = validate_input_receipt_v4_4(
        input_receipt, manifest=manifest_payload
    )
    source_sessions = list(source_calendar_open_sessions)
    source_calendar = {
        "open_sessions": source_sessions,
        "descriptor": _axis_descriptor(source_sessions),
    }
    halo = RESOURCE_CONTRACT["halo_session_count"]
    strict_source = manifest_payload["strict_source_expected"]
    body = {
        "schema_version": DATA_FIELD_RECEIPT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "cycle_id": manifest_payload["cycle_id"],
        "cutoff": manifest_payload["cutoff"],
        "snapshot_id": manifest_payload["snapshot_id"],
        "proof_output_start": manifest_payload["proof_output_start"],
        "input_manifest_semantic_sha256": semantic_sha256_v4_4(
            manifest_payload
        ),
        "input_receipt_semantic_sha256": input_payload[
            "artifact_semantic_sha256"
        ],
        "strict_source_binding": copy.deepcopy(strict_source),
        "source_calendar": source_calendar,
        "proof_output_calendar": _axis_descriptor(source_sessions[halo:]),
        "historical_date_axis_descriptor": _axis_descriptor(source_sessions),
        "historical_symbol_axis": copy.deepcopy(dict(historical_symbol_axis)),
        "pit_membership_contract": copy.deepcopy(dict(pit_membership_contract)),
        "pit_mask_descriptor": copy.deepcopy(dict(pit_mask_descriptor)),
        "cutoff_full_a_scope": {
            "count": strict_source["full_a_scope_count"],
            "sha256": strict_source["full_a_scope_sha256"],
            "hash_algorithm": FULL_A_HASH_ALGORITHM,
        },
        "block_manifest": copy.deepcopy(dict(block_manifest)),
        "table_projection": list(TABLE_PROJECTION),
        "field_adapters": copy.deepcopy(list(FIELD_SEMANTICS)),
        "operator_program_set": operator_program_set_v4_4(),
        "operator_program_set_semantic_sha256": (
            OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
        ),
        "field_missing_counts": copy.deepcopy(dict(field_missing_counts)),
        "bars_outside_pit_interval_count": bars_outside_pit_interval_count,
        "ignored_pre_analysis_row_count": ignored_pre_analysis_row_count,
        "outside_pit_non_null_counts": copy.deepcopy(
            dict(outside_pit_non_null_counts)
        ),
        "projected_row_count_per_pass": projected_row_count_per_pass,
        "source_access": {
            "recorded_latest_pointer_byte_sha256": strict_source[
                "recorded_latest_pointer_byte_sha256"
            ],
            "recorded_components_byte_sha256": strict_source[
                "recorded_components_byte_sha256"
            ],
            "current_pointer_read": False,
            "current_components_read": False,
            "serving_read": False,
            "csv_read": False,
        },
        "strict_source_evidence_status": STRICT_SOURCE_EVIDENCE_STATUS,
        "selection_disclosures": copy.deepcopy(SELECTION_DISCLOSURES),
        "negative_claims": copy.deepcopy(NEGATIVE_CLAIMS),
    }
    return validate_data_field_receipt_v4_4(
        _seal_artifact(body),
        manifest=manifest_payload,
        input_receipt=input_payload,
    )


ENGINE_PASS_SCHEMA_VERSION = (
    "factor-governance-future-strict-engine-pass.v4.4"
)
COLLECTION_DESCRIPTOR_SCHEMA_VERSION = (
    "factor-governance-future-strict-collection-descriptor.v4.4"
)
_INPUT_MATRIX_DESCRIPTOR_ROW_FIELDS = frozenset({"field", "descriptor"})
_COLLECTION_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "pass_id",
        "strict_source_binding",
        "source_calendar_descriptor",
        "historical_symbol_axis_descriptor",
        "pit_mask_semantic_sha256",
        "block_manifest_semantic_sha256",
        "operator_program_set_semantic_sha256",
        "input_matrix_descriptors",
        "projected_row_count_per_pass",
        "collection_sha256",
    }
)
_ENGINE_PASS_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "engine_id",
        "pass_id",
        "collection_sha256",
        "row_count",
        "column_count",
        "date_axis",
        "symbol_axis",
        "proof_pit_mask_semantic_sha256",
        "operator_program_set_semantic_sha256",
        "candidates",
        "signal_computability_proven",
        "labels_present",
        "outcomes_present",
        "statistics_run",
        "tolerance_used",
        "rounding_used",
        "authority",
        "result_semantic_sha256",
    }
)
_ENGINE_CANDIDATE_FIELDS = frozenset(
    {
        "order",
        "name",
        "direction",
        "raw_matrix",
        "direction_adjusted_matrix",
        "non_null_mask_semantic_sha256",
        "operator_program_semantic_sha256",
    }
)
_CANDIDATE_NON_NULL_MASK_ROW_FIELDS = frozenset(
    {
        "order",
        "name",
        "raw_matrix_bit_pattern_sha256",
        "mask",
        "outside_pit_non_null_count",
    }
)
_CANDIDATE_NON_NULL_MASK_SET_FIELDS = frozenset(
    {"schema_version", "candidate_count", "rows", "set_semantic_sha256"}
)
_PASS_DESCRIPTOR_FIELDS = frozenset({"pass_id", "collection", "engines"})
_TWO_PASS_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "cycle_id",
        "cutoff",
        "snapshot_id",
        "proof_output_start",
        "input_manifest_semantic_sha256",
        "input_receipt_semantic_sha256",
        "data_field_receipt_semantic_sha256",
        "operator_program_set_semantic_sha256",
        "proof_pit_mask",
        "candidate_non_null_masks",
        "passes",
        "claims",
        "selection_disclosures",
        "negative_claims",
        "artifact_semantic_sha256",
    }
)
_EQUIVALENCE_CLAIMS = {
    "exact_five_atomic": True,
    "independent_engine_equivalence": True,
    "double_fresh_read_reproducibility": True,
}


def _collection_content(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key not in {"pass_id", "collection_sha256"}
    }


def validate_collection_descriptor_v4_4(
    value: Mapping[str, Any],
    *,
    pass_id: str,
    data_field_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    data = _exact_object(
        data_field_receipt,
        _DATA_FIELD_RECEIPT_FIELDS,
        "collection-bound data/field receipt",
    )
    _validate_self_sha(data, "collection-bound data/field receipt")
    program_set = validate_operator_program_set_v4_4(
        data["operator_program_set"]
    )
    if (
        data["operator_program_set_semantic_sha256"]
        != program_set["artifact_semantic_sha256"]
        or data["operator_program_set_semantic_sha256"]
        != OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
    ):
        raise _error("collection-bound operator program-set SHA mismatch")
    payload = _exact_object(
        value, _COLLECTION_DESCRIPTOR_FIELDS, f"{pass_id} collection"
    )
    if (
        payload["schema_version"] != COLLECTION_DESCRIPTOR_SCHEMA_VERSION
        or payload["pass_id"] != pass_id
        or pass_id not in PASS_IDS
    ):
        raise _error("collection descriptor schema/pass mismatch")
    if payload["strict_source_binding"] != data["strict_source_binding"]:
        raise _error("collection strict-source binding mismatch")
    if payload["source_calendar_descriptor"] != data["source_calendar"][
        "descriptor"
    ]:
        raise _error("collection source calendar mismatch")
    symbol_axis = data["historical_symbol_axis"]["descriptor"]
    if payload["historical_symbol_axis_descriptor"] != symbol_axis:
        raise _error("collection historical symbol axis mismatch")
    if payload["pit_mask_semantic_sha256"] != semantic_sha256_v4_4(
        data["pit_mask_descriptor"]
    ):
        raise _error("collection PIT bitmap binding mismatch")
    if payload["block_manifest_semantic_sha256"] != data["block_manifest"][
        "manifest_semantic_sha256"
    ]:
        raise _error("collection block-manifest SHA mismatch")
    if payload["operator_program_set_semantic_sha256"] != program_set[
        "artifact_semantic_sha256"
    ]:
        raise _error("collection operator program-set SHA mismatch")
    _positive_int(
        payload["projected_row_count_per_pass"],
        f"{pass_id} collection projected row count",
    )
    if payload["projected_row_count_per_pass"] != data[
        "projected_row_count_per_pass"
    ]:
        raise _error("collection projected-row count mismatch")
    rows = payload["input_matrix_descriptors"]
    if type(rows) is not list or len(rows) != len(INPUT_FIELDS):
        raise _error("collection requires exact ordered four input descriptors")
    pit = data["pit_mask_descriptor"]
    outside_cells = pit["zero_count"]
    normalized_rows: list[dict[str, Any]] = []
    for index, (row_value, expected_field) in enumerate(
        zip(rows, INPUT_FIELDS, strict=True), start=1
    ):
        row = _exact_object(
            row_value,
            _INPUT_MATRIX_DESCRIPTOR_ROW_FIELDS,
            f"collection input descriptor[{index}]",
        )
        if row["field"] != expected_field:
            raise _error("collection input descriptor field/order mismatch")
        descriptor = validate_matrix_descriptor_v4_4(
            row["descriptor"],
            label=f"collection {expected_field} descriptor",
            expected_date_axis=data["historical_date_axis_descriptor"],
            expected_symbol_axis=symbol_axis,
        )
        if (
            descriptor["positive_infinity_count"] != 0
            or descriptor["negative_infinity_count"] != 0
            or descriptor["nan_count"]
            != outside_cells + data["field_missing_counts"][expected_field]
        ):
            raise _error("collection input missing/nonfinite accounting mismatch")
        row["descriptor"] = descriptor
        normalized_rows.append(row)
    payload["input_matrix_descriptors"] = normalized_rows
    supplied = _sha256(payload["collection_sha256"], "collection SHA")
    if supplied != semantic_sha256_v4_4(_collection_content(payload)):
        raise _error("collection SHA does not bind the full collection descriptor")
    return payload


def build_collection_descriptor_v4_4(
    *,
    pass_id: str,
    data_field_receipt: Mapping[str, Any],
    input_matrix_descriptors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    data = copy.deepcopy(dict(data_field_receipt))
    body = {
        "schema_version": COLLECTION_DESCRIPTOR_SCHEMA_VERSION,
        "pass_id": pass_id,
        "strict_source_binding": copy.deepcopy(data["strict_source_binding"]),
        "source_calendar_descriptor": copy.deepcopy(
            data["source_calendar"]["descriptor"]
        ),
        "historical_symbol_axis_descriptor": copy.deepcopy(
            data["historical_symbol_axis"]["descriptor"]
        ),
        "pit_mask_semantic_sha256": semantic_sha256_v4_4(
            data["pit_mask_descriptor"]
        ),
        "block_manifest_semantic_sha256": data["block_manifest"][
            "manifest_semantic_sha256"
        ],
        "operator_program_set_semantic_sha256": data[
            "operator_program_set_semantic_sha256"
        ],
        "input_matrix_descriptors": [
            copy.deepcopy(dict(row)) for row in input_matrix_descriptors
        ],
        "projected_row_count_per_pass": data[
            "projected_row_count_per_pass"
        ],
    }
    body["collection_sha256"] = semantic_sha256_v4_4(
        _collection_content(body)
    )
    return validate_collection_descriptor_v4_4(
        body, pass_id=pass_id, data_field_receipt=data
    )


def _validate_direction_adjustment(
    raw: Mapping[str, Any], adjusted: Mapping[str, Any], direction: int
) -> None:
    if direction == 1:
        if canonical_json_bytes_v4_4(raw) != canonical_json_bytes_v4_4(adjusted):
            raise _error("positive-direction raw/adjusted descriptors must be exact")
        return
    if direction != -1:
        raise _error("candidate direction must be +1 or -1")
    equal_fields = (
        "schema_version",
        "dtype",
        "layout",
        "row_count",
        "column_count",
        "date_axis",
        "symbol_axis",
        "magnitude_bits_sha256",
        "finite_count",
        "nan_count",
        "byte_count",
    )
    if any(raw[field] != adjusted[field] for field in equal_fields):
        raise _error("negative-direction magnitude/axis accounting drift")
    if (
        adjusted["matrix_sha256"] != raw["elementwise_negated_sha256"]
        or adjusted["bit_pattern_sha256"] != raw["elementwise_negated_sha256"]
        or adjusted["elementwise_negated_sha256"] != raw["matrix_sha256"]
        or adjusted["positive_infinity_count"] != raw["negative_infinity_count"]
        or adjusted["negative_infinity_count"] != raw["positive_infinity_count"]
        or adjusted["positive_finite_count"] != raw["negative_finite_count"]
        or adjusted["negative_finite_count"] != raw["positive_finite_count"]
        or adjusted["positive_zero_count"] != raw["negative_zero_count"]
        or adjusted["negative_zero_count"] != raw["positive_zero_count"]
    ):
        raise _error("negative-direction bitwise descriptor relation mismatch")


def _slice_packed_suffix(
    packed: bytes, *, start_bit: int, result_bit_count: int
) -> bytes:
    if (
        type(packed) is not bytes
        or type(start_bit) is not int
        or type(result_bit_count) is not int
        or start_bit < 0
        or result_bit_count <= 0
    ):
        raise _error("packed bitmap slice arguments are invalid")
    shifted = int.from_bytes(packed, "little") >> start_bit
    return shifted.to_bytes((result_bit_count + 7) // 8, "little")


def _validate_proof_pit_mask_v4_4(
    value: Any, *, data_field_receipt: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes]:
    data = dict(data_field_receipt)
    proof_axis = data["proof_output_calendar"]
    symbol_axis = data["historical_symbol_axis"]["descriptor"]
    payload, packed = _decode_binary_mask_descriptor_v4_4(
        value,
        label="proof-output PIT bitmap",
        expected_date_axis=proof_axis,
        expected_symbol_axis=symbol_axis,
    )
    full, full_packed = _decode_binary_mask_descriptor_v4_4(
        data["pit_mask_descriptor"],
        label="full-source PIT bitmap",
        expected_date_axis=data["historical_date_axis_descriptor"],
        expected_symbol_axis=symbol_axis,
    )
    start_bit = (
        RESOURCE_CONTRACT["halo_session_count"] * symbol_axis["count"]
    )
    if full["bit_count"] - start_bit != payload["bit_count"]:
        raise _error("proof-output PIT bitmap length is not the halo suffix")
    expected_packed = _slice_packed_suffix(
        full_packed,
        start_bit=start_bit,
        result_bit_count=payload["bit_count"],
    )
    if packed != expected_packed:
        raise _error("proof-output PIT bitmap is not the exact full-mask halo suffix")
    if payload["one_count"] <= 0:
        raise _error("proof-output PIT bitmap has no eligible observation")
    return payload, packed


def _candidate_mask_set_content(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "set_semantic_sha256"
    }


def _validate_candidate_non_null_mask_set_v4_4(
    value: Any,
    *,
    proof_pit_mask: Mapping[str, Any],
    proof_pit_packed: bytes,
    data_field_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], list[bytes]]:
    payload = _exact_object(
        value,
        _CANDIDATE_NON_NULL_MASK_SET_FIELDS,
        "candidate non-null mask set",
    )
    _positive_int(
        payload["candidate_count"],
        "candidate non-null mask set candidate count",
    )
    if (
        payload["schema_version"]
        != CANDIDATE_NON_NULL_MASK_SET_SCHEMA_VERSION
        or payload["candidate_count"] != len(SOURCE_DEFINITION_BINDINGS)
    ):
        raise _error("candidate non-null mask set identity/count mismatch")
    rows = payload["rows"]
    if type(rows) is not list or len(rows) != len(SOURCE_DEFINITION_BINDINGS):
        raise _error("candidate non-null mask set must contain the exact five")
    proof_axis = data_field_receipt["proof_output_calendar"]
    symbol_axis = data_field_receipt["historical_symbol_axis"]["descriptor"]
    normalized_rows: list[dict[str, Any]] = []
    packed_rows: list[bytes] = []
    for index, (row_value, expected) in enumerate(
        zip(rows, SOURCE_DEFINITION_BINDINGS, strict=True), start=1
    ):
        row = _exact_object(
            row_value,
            _CANDIDATE_NON_NULL_MASK_ROW_FIELDS,
            f"candidate non-null mask[{index}]",
        )
        _positive_int(row["order"], f"candidate non-null mask {index} order")
        if row["order"] != expected["order"] or row["name"] != expected["name"]:
            raise _error("candidate non-null mask identity/order mismatch")
        _sha256(
            row["raw_matrix_bit_pattern_sha256"],
            f"{row['name']} bound raw-matrix bit-pattern SHA",
        )
        _nonnegative_int(
            row["outside_pit_non_null_count"],
            f"candidate non-null mask {index} outside-PIT count",
        )
        if row["outside_pit_non_null_count"] != 0:
            raise _error("candidate non-null mask claims an outside-PIT observation")
        mask, packed = _decode_binary_mask_descriptor_v4_4(
            row["mask"],
            label=f"{row['name']} non-null bitmap",
            expected_date_axis=proof_axis,
            expected_symbol_axis=symbol_axis,
        )
        outside_count = sum(
            (candidate_byte & (~pit_byte & 0xFF)).bit_count()
            for candidate_byte, pit_byte in zip(
                packed, proof_pit_packed, strict=True
            )
        )
        if outside_count != 0:
            raise _error(f"{row['name']} has a non-null observation outside PIT")
        row["mask"] = mask
        normalized_rows.append(row)
        packed_rows.append(packed)
    payload["rows"] = normalized_rows
    supplied = _sha256(
        payload["set_semantic_sha256"],
        "candidate non-null mask-set semantic SHA",
    )
    if supplied != semantic_sha256_v4_4(_candidate_mask_set_content(payload)):
        raise _error("candidate non-null mask-set semantic SHA mismatch")
    return payload, packed_rows


def build_candidate_non_null_mask_set_v4_4(
    *,
    proof_pit_mask: Mapping[str, Any],
    candidate_masks: Sequence[Mapping[str, Any]],
    raw_matrix_descriptors: Sequence[Mapping[str, Any]],
    data_field_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    proof, proof_packed = _validate_proof_pit_mask_v4_4(
        proof_pit_mask, data_field_receipt=data_field_receipt
    )
    if len(candidate_masks) != len(SOURCE_DEFINITION_BINDINGS):
        raise _error("candidate mask builder requires the ordered exact five")
    if len(raw_matrix_descriptors) != len(SOURCE_DEFINITION_BINDINGS):
        raise _error("candidate mask builder requires five raw matrix descriptors")
    body = {
        "schema_version": CANDIDATE_NON_NULL_MASK_SET_SCHEMA_VERSION,
        "candidate_count": len(SOURCE_DEFINITION_BINDINGS),
        "rows": [
            {
                "order": expected["order"],
                "name": expected["name"],
                "raw_matrix_bit_pattern_sha256": _sha256(
                    raw_descriptor.get("bit_pattern_sha256"),
                    f"{expected['name']} raw matrix bit-pattern SHA",
                ),
                "mask": copy.deepcopy(dict(mask)),
                "outside_pit_non_null_count": 0,
            }
            for expected, mask, raw_descriptor in zip(
                SOURCE_DEFINITION_BINDINGS,
                candidate_masks,
                raw_matrix_descriptors,
                strict=True,
            )
        ],
    }
    body["set_semantic_sha256"] = semantic_sha256_v4_4(body)
    normalized, _packed = _validate_candidate_non_null_mask_set_v4_4(
        body,
        proof_pit_mask=proof,
        proof_pit_packed=proof_packed,
        data_field_receipt=data_field_receipt,
    )
    return normalized


def validate_engine_pass_result_v4_4(
    value: Mapping[str, Any],
    *,
    pass_id: str,
    engine_id: str,
    collection_sha256: str,
    data_field_receipt: Mapping[str, Any],
    proof_pit_mask: Mapping[str, Any],
    candidate_non_null_masks: Mapping[str, Any],
) -> dict[str, Any]:
    data = _exact_object(
        data_field_receipt,
        _DATA_FIELD_RECEIPT_FIELDS,
        "engine-bound data/field receipt",
    )
    _validate_self_sha(data, "engine-bound data/field receipt")
    program_set = validate_operator_program_set_v4_4(
        data["operator_program_set"]
    )
    if (
        data["operator_program_set_semantic_sha256"]
        != program_set["artifact_semantic_sha256"]
        or data["operator_program_set_semantic_sha256"]
        != OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
    ):
        raise _error("engine-bound operator program-set SHA mismatch")
    _sha256(collection_sha256, "engine-bound collection SHA")
    payload = _exact_object(
        value, _ENGINE_PASS_FIELDS, f"{pass_id} {engine_id} engine result"
    )
    if (
        payload["schema_version"] != ENGINE_PASS_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
        or payload["engine_id"] != engine_id
        or payload["pass_id"] != pass_id
        or payload["collection_sha256"] != collection_sha256
    ):
        raise _error("engine result identity mismatch")
    if payload["operator_program_set_semantic_sha256"] != program_set[
        "artifact_semantic_sha256"
    ]:
        raise _error("engine result operator program-set SHA mismatch")
    if (
        payload["signal_computability_proven"] is not True
        or payload["labels_present"] is not False
        or payload["outcomes_present"] is not False
        or payload["statistics_run"] is not False
        or payload["tolerance_used"] is not False
        or payload["rounding_used"] is not False
        or payload["authority"] is not False
    ):
        raise _error("engine result promoted a prohibited claim or approximation")
    proof_axis = data["proof_output_calendar"]
    symbol_axis = data["historical_symbol_axis"]["descriptor"]
    date_axis = _validate_axis_descriptor(payload["date_axis"], "engine date axis")
    engine_symbols = _validate_axis_descriptor(
        payload["symbol_axis"], "engine symbol axis"
    )
    _positive_int(payload["row_count"], "engine result row count")
    _positive_int(payload["column_count"], "engine result column count")
    if (
        date_axis != proof_axis
        or engine_symbols != symbol_axis
        or payload["row_count"] != date_axis["count"]
        or payload["column_count"] != engine_symbols["count"]
    ):
        raise _error("engine result output axes/shape mismatch")
    pit, pit_packed = _validate_proof_pit_mask_v4_4(
        proof_pit_mask,
        data_field_receipt=data,
    )
    if payload["proof_pit_mask_semantic_sha256"] != semantic_sha256_v4_4(pit):
        raise _error("engine result proof-PIT bitmap binding mismatch")
    mask_set, _mask_packed = _validate_candidate_non_null_mask_set_v4_4(
        candidate_non_null_masks,
        proof_pit_mask=pit,
        proof_pit_packed=pit_packed,
        data_field_receipt=data,
    )
    mask_rows = mask_set["rows"]
    rows = payload["candidates"]
    if type(rows) is not list or len(rows) != len(SOURCE_DEFINITION_BINDINGS):
        raise _error("engine result must contain the ordered exact five candidates")
    normalized_rows: list[dict[str, Any]] = []
    for index, (row_value, expected, program, mask_row) in enumerate(
        zip(
            rows,
            SOURCE_DEFINITION_BINDINGS,
            program_set["candidates"],
            mask_rows,
            strict=True,
        ),
        start=1,
    ):
        row = _exact_object(
            row_value,
            _ENGINE_CANDIDATE_FIELDS,
            f"engine candidate[{index}]",
        )
        _positive_int(row["order"], f"engine candidate {index} order")
        if (
            row["order"] != expected["order"]
            or row["name"] != expected["name"]
            or type(row["direction"]) is not float
            or row["direction"] != float(expected["direction"])
            or row["operator_program_semantic_sha256"]
            != program["program_semantic_sha256"]
            or row["operator_program_semantic_sha256"]
            != expected["operator_program_sha256"]
        ):
            raise _error(
                "engine candidate identity/order/direction/program mismatch"
            )
        raw = validate_matrix_descriptor_v4_4(
            row["raw_matrix"],
            label=f"{row['name']} raw output",
            expected_date_axis=proof_axis,
            expected_symbol_axis=symbol_axis,
        )
        adjusted = validate_matrix_descriptor_v4_4(
            row["direction_adjusted_matrix"],
            label=f"{row['name']} adjusted output",
            expected_date_axis=proof_axis,
            expected_symbol_axis=symbol_axis,
        )
        if (
            raw["finite_count"] <= 0
            or raw["positive_infinity_count"] != 0
            or raw["negative_infinity_count"] != 0
            or raw["nan_count"] < pit["zero_count"]
        ):
            raise _error("engine candidate is not finitely computable within PIT")
        mask = validate_binary_mask_descriptor_v4_4(
            mask_row["mask"],
            label=f"{row['name']} engine-bound non-null bitmap",
            expected_date_axis=proof_axis,
            expected_symbol_axis=symbol_axis,
        )
        if (
            row["non_null_mask_semantic_sha256"]
            != semantic_sha256_v4_4(mask)
            or mask_row["raw_matrix_bit_pattern_sha256"]
            != raw["bit_pattern_sha256"]
            or raw["finite_count"] != mask["one_count"]
            or raw["nan_count"] != mask["zero_count"]
        ):
            raise _error("engine candidate matrix/non-null bitmap accounting mismatch")
        _validate_direction_adjustment(raw, adjusted, expected["direction"])
        row["raw_matrix"] = raw
        row["direction_adjusted_matrix"] = adjusted
        normalized_rows.append(row)
    payload["candidates"] = normalized_rows
    supplied = _sha256(
        payload["result_semantic_sha256"], "engine result semantic SHA"
    )
    if supplied != semantic_sha256_v4_4(
        {
            key: item
            for key, item in payload.items()
            if key != "result_semantic_sha256"
        }
    ):
        raise _error("engine result semantic SHA mismatch")
    return payload


def _exact_candidate_descriptor_mapping_v4_4(
    value: Any, *, label: str
) -> dict[str, dict[str, Any]]:
    names = tuple(row["name"] for row in SOURCE_DEFINITION_BINDINGS)
    if type(value) is not dict or tuple(value) != names:
        raise _error(f"{label} must be the exact ordered name-to-descriptor mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for name in names:
        descriptor = value[name]
        if type(descriptor) is not dict:
            raise _error(f"{label}.{name} must be an exact descriptor object")
        normalized[name] = copy.deepcopy(descriptor)
    return normalized


def build_engine_pass_result_v4_4(
    *,
    pass_id: str,
    engine_id: str,
    collection_sha256: str,
    data_field_receipt: Mapping[str, Any],
    operator_program_set: Mapping[str, Any],
    proof_pit_mask: Mapping[str, Any],
    candidate_non_null_masks: Mapping[str, Any],
    raw_matrix_descriptors: Mapping[str, Mapping[str, Any]],
    adjusted_matrix_descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the sole contract-owned engine-pass evidence shape."""

    if pass_id not in PASS_IDS or engine_id not in ENGINE_IDS:
        raise _error("engine-pass builder identity is not registered")
    _sha256(collection_sha256, "engine-pass builder collection SHA")
    data = _exact_object(
        data_field_receipt,
        _DATA_FIELD_RECEIPT_FIELDS,
        "engine-pass builder data/field receipt",
    )
    _validate_self_sha(data, "engine-pass builder data/field receipt")
    program_set = validate_operator_program_set_v4_4(operator_program_set)
    data_program_set = validate_operator_program_set_v4_4(
        data["operator_program_set"]
    )
    if (
        canonical_json_bytes_v4_4(program_set)
        != canonical_json_bytes_v4_4(data_program_set)
        or data["operator_program_set_semantic_sha256"]
        != program_set["artifact_semantic_sha256"]
    ):
        raise _error("engine-pass builder operator program-set substitution")
    proof_pit, proof_pit_packed = _validate_proof_pit_mask_v4_4(
        proof_pit_mask, data_field_receipt=data
    )
    masks, _packed_masks = _validate_candidate_non_null_mask_set_v4_4(
        candidate_non_null_masks,
        proof_pit_mask=proof_pit,
        proof_pit_packed=proof_pit_packed,
        data_field_receipt=data,
    )
    raw_by_name = _exact_candidate_descriptor_mapping_v4_4(
        raw_matrix_descriptors, label="raw matrix descriptors"
    )
    adjusted_by_name = _exact_candidate_descriptor_mapping_v4_4(
        adjusted_matrix_descriptors, label="adjusted matrix descriptors"
    )
    mask_by_name = {row["name"]: row for row in masks["rows"]}
    candidates = []
    for program in program_set["candidates"]:
        name = program["name"]
        candidates.append(
            {
                "order": program["order"],
                "name": name,
                "direction": program["direction"],
                "raw_matrix": raw_by_name[name],
                "direction_adjusted_matrix": adjusted_by_name[name],
                "non_null_mask_semantic_sha256": semantic_sha256_v4_4(
                    mask_by_name[name]["mask"]
                ),
                "operator_program_semantic_sha256": program[
                    "program_semantic_sha256"
                ],
            }
        )
    body = {
        "schema_version": ENGINE_PASS_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "engine_id": engine_id,
        "pass_id": pass_id,
        "collection_sha256": collection_sha256,
        "row_count": data["proof_output_calendar"]["count"],
        "column_count": data["historical_symbol_axis"]["descriptor"]["count"],
        "date_axis": copy.deepcopy(data["proof_output_calendar"]),
        "symbol_axis": copy.deepcopy(
            data["historical_symbol_axis"]["descriptor"]
        ),
        "proof_pit_mask_semantic_sha256": semantic_sha256_v4_4(proof_pit),
        "operator_program_set_semantic_sha256": program_set[
            "artifact_semantic_sha256"
        ],
        "candidates": candidates,
        "signal_computability_proven": True,
        "labels_present": False,
        "outcomes_present": False,
        "statistics_run": False,
        "tolerance_used": False,
        "rounding_used": False,
        "authority": False,
    }
    body["result_semantic_sha256"] = semantic_sha256_v4_4(body)
    return validate_engine_pass_result_v4_4(
        body,
        pass_id=pass_id,
        engine_id=engine_id,
        collection_sha256=collection_sha256,
        data_field_receipt=data,
        proof_pit_mask=proof_pit,
        candidate_non_null_masks=masks,
    )


def _normalized_engine_result(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key not in {"engine_id", "pass_id", "result_semantic_sha256"}
    }


def validate_two_pass_equivalence_receipt_v4_4(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    input_receipt: Mapping[str, Any],
    data_field_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_payload = validate_input_manifest_v4_4(manifest)
    input_payload = validate_input_receipt_v4_4(
        input_receipt, manifest=manifest_payload
    )
    data = validate_data_field_receipt_v4_4(
        data_field_receipt,
        manifest=manifest_payload,
        input_receipt=input_payload,
    )
    payload = _exact_object(
        value,
        _TWO_PASS_RECEIPT_FIELDS,
        "two-pass equivalence receipt",
    )
    payload = _validate_self_sha(payload, "two-pass equivalence receipt")
    if (
        payload["schema_version"] != TWO_PASS_EQUIVALENCE_RECEIPT_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
    ):
        raise _error("two-pass receipt schema/protocol mismatch")
    for field in ("cycle_id", "cutoff", "snapshot_id", "proof_output_start"):
        if payload[field] != manifest_payload[field]:
            raise _error(f"two-pass receipt {field} mismatch")
    expected_shas = {
        "input_manifest_semantic_sha256": semantic_sha256_v4_4(manifest_payload),
        "input_receipt_semantic_sha256": input_payload[
            "artifact_semantic_sha256"
        ],
        "data_field_receipt_semantic_sha256": data[
            "artifact_semantic_sha256"
        ],
    }
    for field, expected in expected_shas.items():
        if payload[field] != expected:
            raise _error(f"two-pass receipt {field} mismatch")
    if payload["operator_program_set_semantic_sha256"] != data[
        "operator_program_set_semantic_sha256"
    ]:
        raise _error("two-pass receipt operator program-set SHA mismatch")
    payload["claims"] = _validate_fixed_object(
        payload["claims"], _EQUIVALENCE_CLAIMS, "two-pass receipt positive claims"
    )
    payload["selection_disclosures"] = _validate_fixed_object(
        payload["selection_disclosures"],
        SELECTION_DISCLOSURES,
        "two-pass receipt selection disclosures",
    )
    payload["negative_claims"] = _validate_fixed_object(
        payload["negative_claims"],
        NEGATIVE_CLAIMS,
        "two-pass receipt negative claims",
    )

    proof_pit, proof_pit_packed = _validate_proof_pit_mask_v4_4(
        payload["proof_pit_mask"], data_field_receipt=data
    )
    candidate_masks, _candidate_packed = (
        _validate_candidate_non_null_mask_set_v4_4(
            payload["candidate_non_null_masks"],
            proof_pit_mask=proof_pit,
            proof_pit_packed=proof_pit_packed,
            data_field_receipt=data,
        )
    )
    payload["proof_pit_mask"] = proof_pit
    payload["candidate_non_null_masks"] = candidate_masks

    passes = payload["passes"]
    if type(passes) is not list or len(passes) != 2:
        raise _error("two-pass receipt requires exactly two fresh passes")
    normalized_passes: list[dict[str, Any]] = []
    for index, (pass_value, pass_id) in enumerate(
        zip(passes, PASS_IDS, strict=True), start=1
    ):
        pass_row = _exact_object(
            pass_value, _PASS_DESCRIPTOR_FIELDS, f"pass descriptor[{index}]"
        )
        if pass_row["pass_id"] != pass_id:
            raise _error("fresh pass identity/order mismatch")
        collection = validate_collection_descriptor_v4_4(
            pass_row["collection"], pass_id=pass_id, data_field_receipt=data
        )
        engines = pass_row["engines"]
        if type(engines) is not list or len(engines) != len(ENGINE_IDS):
            raise _error("each pass requires both exact independent engines")
        normalized_engines = [
            validate_engine_pass_result_v4_4(
                engine,
                pass_id=pass_id,
                engine_id=engine_id,
                collection_sha256=collection["collection_sha256"],
                data_field_receipt=data,
                proof_pit_mask=proof_pit,
                candidate_non_null_masks=candidate_masks,
            )
            for engine, engine_id in zip(engines, ENGINE_IDS, strict=True)
        ]
        if canonical_json_bytes_v4_4(
            _normalized_engine_result(normalized_engines[0])
        ) != canonical_json_bytes_v4_4(
            _normalized_engine_result(normalized_engines[1])
        ):
            raise _error("independent engines differ at the bitwise descriptor level")
        pass_row["collection"] = collection
        pass_row["engines"] = normalized_engines
        normalized_passes.append(pass_row)

    first, second = normalized_passes
    if first["collection"]["collection_sha256"] != second["collection"][
        "collection_sha256"
    ]:
        raise _error("fresh source/data collections differ")
    if canonical_json_bytes_v4_4(
        _collection_content(first["collection"])
    ) != canonical_json_bytes_v4_4(_collection_content(second["collection"])):
        raise _error("fresh collection descriptors drifted")
    for engine_index in range(len(ENGINE_IDS)):
        if canonical_json_bytes_v4_4(
            _normalized_engine_result(first["engines"][engine_index])
        ) != canonical_json_bytes_v4_4(
            _normalized_engine_result(second["engines"][engine_index])
        ):
            raise _error("fresh-pass engine bitwise descriptors drifted")
    payload["passes"] = normalized_passes
    return payload


def build_two_pass_equivalence_receipt_v4_4(
    *,
    manifest: Mapping[str, Any],
    input_receipt: Mapping[str, Any],
    data_field_receipt: Mapping[str, Any],
    proof_pit_mask: Mapping[str, Any],
    candidate_non_null_masks: Mapping[str, Any],
    collections: Sequence[Mapping[str, Any]],
    engine_results: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    manifest_payload = validate_input_manifest_v4_4(manifest)
    input_payload = validate_input_receipt_v4_4(
        input_receipt, manifest=manifest_payload
    )
    data = validate_data_field_receipt_v4_4(
        data_field_receipt,
        manifest=manifest_payload,
        input_receipt=input_payload,
    )
    if len(collections) != 2 or len(engine_results) != 2:
        raise _error("builder requires exactly two collections and two engine pairs")
    passes = []
    for pass_id, collection, engines in zip(
        PASS_IDS, collections, engine_results, strict=True
    ):
        passes.append(
            {
                "pass_id": pass_id,
                "collection": copy.deepcopy(dict(collection)),
                "engines": [copy.deepcopy(dict(engine)) for engine in engines],
            }
        )
    body = {
        "schema_version": TWO_PASS_EQUIVALENCE_RECEIPT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "cycle_id": manifest_payload["cycle_id"],
        "cutoff": manifest_payload["cutoff"],
        "snapshot_id": manifest_payload["snapshot_id"],
        "proof_output_start": manifest_payload["proof_output_start"],
        "input_manifest_semantic_sha256": semantic_sha256_v4_4(
            manifest_payload
        ),
        "input_receipt_semantic_sha256": input_payload[
            "artifact_semantic_sha256"
        ],
        "data_field_receipt_semantic_sha256": data[
            "artifact_semantic_sha256"
        ],
        "operator_program_set_semantic_sha256": data[
            "operator_program_set_semantic_sha256"
        ],
        "proof_pit_mask": copy.deepcopy(dict(proof_pit_mask)),
        "candidate_non_null_masks": copy.deepcopy(
            dict(candidate_non_null_masks)
        ),
        "passes": passes,
        "claims": copy.deepcopy(_EQUIVALENCE_CLAIMS),
        "selection_disclosures": copy.deepcopy(SELECTION_DISCLOSURES),
        "negative_claims": copy.deepcopy(NEGATIVE_CLAIMS),
    }
    return validate_two_pass_equivalence_receipt_v4_4(
        _seal_artifact(body),
        manifest=manifest_payload,
        input_receipt=input_payload,
        data_field_receipt=data,
    )


_PROOF_MANIFEST_BINDING_FIELDS = frozenset(
    {"filename", "byte_sha256", "semantic_sha256"}
)
_PROOF_PREREG_BINDING_FIELDS = frozenset(
    {"bundle_path", "cycle_id", "byte_sha256", "semantic_sha256"}
)
_PROOF_PREDECESSOR_FIELDS = frozenset(
    {
        "input_manifest",
        "preregistration_readback",
        "input_receipt_semantic_sha256",
        "data_field_receipt_semantic_sha256",
        "two_pass_equivalence_receipt_semantic_sha256",
    }
)
_PROOF_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "cycle_id",
        "cutoff",
        "snapshot_id",
        "proof_output_start",
        "predecessor_bindings",
        "strict_source_binding_semantic_sha256",
        "source_definition_bindings",
        "operator_program_set_semantic_sha256",
        "claims",
        "strict_source_evidence_status",
        "selection_disclosures",
        "negative_claims",
        "artifact_semantic_sha256",
    }
)


def _expected_proof_predecessor_bindings(
    *,
    manifest: Mapping[str, Any],
    input_receipt: Mapping[str, Any],
    data_field_receipt: Mapping[str, Any],
    two_pass_equivalence_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_binding = _expected_manifest_binding(manifest)
    prereg = manifest["preregistration"]
    return {
        "input_manifest": {
            "filename": INPUT_MANIFEST_FILENAME,
            "byte_sha256": manifest_binding["byte_sha256"],
            "semantic_sha256": manifest_binding["semantic_sha256"],
        },
        "preregistration_readback": {
            "bundle_path": prereg["bundle_path"],
            "cycle_id": prereg["cycle_id"],
            "byte_sha256": prereg["readback_byte_sha256"],
            "semantic_sha256": prereg["readback_semantic_sha256"],
        },
        "input_receipt_semantic_sha256": input_receipt[
            "artifact_semantic_sha256"
        ],
        "data_field_receipt_semantic_sha256": data_field_receipt[
            "artifact_semantic_sha256"
        ],
        "two_pass_equivalence_receipt_semantic_sha256": (
            two_pass_equivalence_receipt["artifact_semantic_sha256"]
        ),
    }


def _validate_proof_predecessor_binding_structure_v4_4(
    value: Any,
) -> dict[str, Any]:
    payload = _exact_object(
        value, _PROOF_PREDECESSOR_FIELDS, "proof predecessor bindings"
    )
    manifest_binding = _exact_object(
        payload["input_manifest"],
        _PROOF_MANIFEST_BINDING_FIELDS,
        "proof input-manifest binding",
    )
    prereg = _exact_object(
        payload["preregistration_readback"],
        _PROOF_PREREG_BINDING_FIELDS,
        "proof preregistration binding",
    )
    for binding, label in (
        (manifest_binding, "proof manifest"),
        (prereg, "proof preregistration"),
    ):
        _sha256(binding["byte_sha256"], f"{label} byte SHA")
        _sha256(binding["semantic_sha256"], f"{label} semantic SHA")
    if manifest_binding["filename"] != INPUT_MANIFEST_FILENAME:
        raise _error("proof predecessor manifest filename mismatch")
    _absolute_normalized_path(prereg["bundle_path"], "proof prereg bundle path")
    if type(prereg["cycle_id"]) is not str or not prereg["cycle_id"]:
        raise _error("proof preregistration cycle id must be non-empty")
    for field in (
        "input_receipt_semantic_sha256",
        "data_field_receipt_semantic_sha256",
        "two_pass_equivalence_receipt_semantic_sha256",
    ):
        _sha256(payload[field], f"proof predecessor {field}")
    payload["input_manifest"] = manifest_binding
    payload["preregistration_readback"] = prereg
    return payload


def _validate_proof_predecessor_bindings(
    value: Any, *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _validate_proof_predecessor_binding_structure_v4_4(value)
    if canonical_json_bytes_v4_4(payload) != canonical_json_bytes_v4_4(expected):
        raise _error("proof predecessor graph cross-hash mismatch")
    return payload


def validate_proof_v4_4(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    input_receipt: Mapping[str, Any],
    data_field_receipt: Mapping[str, Any],
    two_pass_equivalence_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the positive claim only against the sealed predecessor graph."""

    manifest_payload = validate_input_manifest_v4_4(manifest)
    input_payload = validate_input_receipt_v4_4(
        input_receipt, manifest=manifest_payload
    )
    data = validate_data_field_receipt_v4_4(
        data_field_receipt,
        manifest=manifest_payload,
        input_receipt=input_payload,
    )
    equivalence = validate_two_pass_equivalence_receipt_v4_4(
        two_pass_equivalence_receipt,
        manifest=manifest_payload,
        input_receipt=input_payload,
        data_field_receipt=data,
    )
    payload = _exact_object(value, _PROOF_FIELDS, "strict computability proof")
    payload = _validate_self_sha(payload, "strict computability proof")
    if (
        payload["schema_version"] != PROOF_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
    ):
        raise _error("strict computability proof schema/protocol mismatch")
    for field in ("cycle_id", "cutoff", "snapshot_id", "proof_output_start"):
        if payload[field] != manifest_payload[field]:
            raise _error(f"strict computability proof {field} mismatch")
    expected_predecessors = _expected_proof_predecessor_bindings(
        manifest=manifest_payload,
        input_receipt=input_payload,
        data_field_receipt=data,
        two_pass_equivalence_receipt=equivalence,
    )
    payload["predecessor_bindings"] = _validate_proof_predecessor_bindings(
        payload["predecessor_bindings"], expected=expected_predecessors
    )
    if payload["strict_source_binding_semantic_sha256"] != manifest_payload[
        "strict_source_expected"
    ]["strict_source_binding_semantic_sha256"]:
        raise _error("proof strict-source semantic SHA mismatch")
    payload["source_definition_bindings"] = _validate_source_definition_bindings(
        payload["source_definition_bindings"]
    )
    if (
        payload["operator_program_set_semantic_sha256"]
        != data["operator_program_set_semantic_sha256"]
        or payload["operator_program_set_semantic_sha256"]
        != equivalence["operator_program_set_semantic_sha256"]
    ):
        raise _error("proof operator program-set SHA mismatch")
    payload["claims"] = _validate_fixed_object(
        payload["claims"], POSITIVE_CLAIMS, "proof positive claims/readiness"
    )
    if payload["strict_source_evidence_status"] != STRICT_SOURCE_EVIDENCE_STATUS:
        raise _error("proof strict-source evidence status mismatch")
    payload["selection_disclosures"] = _validate_fixed_object(
        payload["selection_disclosures"],
        SELECTION_DISCLOSURES,
        "proof selection disclosures",
    )
    payload["negative_claims"] = _validate_fixed_object(
        payload["negative_claims"], NEGATIVE_CLAIMS, "proof negative claims"
    )
    return payload


def build_proof_v4_4(
    *,
    manifest: Mapping[str, Any],
    input_receipt: Mapping[str, Any],
    data_field_receipt: Mapping[str, Any],
    two_pass_equivalence_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_payload = validate_input_manifest_v4_4(manifest)
    input_payload = validate_input_receipt_v4_4(
        input_receipt, manifest=manifest_payload
    )
    data = validate_data_field_receipt_v4_4(
        data_field_receipt,
        manifest=manifest_payload,
        input_receipt=input_payload,
    )
    equivalence = validate_two_pass_equivalence_receipt_v4_4(
        two_pass_equivalence_receipt,
        manifest=manifest_payload,
        input_receipt=input_payload,
        data_field_receipt=data,
    )
    body = {
        "schema_version": PROOF_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "cycle_id": manifest_payload["cycle_id"],
        "cutoff": manifest_payload["cutoff"],
        "snapshot_id": manifest_payload["snapshot_id"],
        "proof_output_start": manifest_payload["proof_output_start"],
        "predecessor_bindings": _expected_proof_predecessor_bindings(
            manifest=manifest_payload,
            input_receipt=input_payload,
            data_field_receipt=data,
            two_pass_equivalence_receipt=equivalence,
        ),
        "strict_source_binding_semantic_sha256": manifest_payload[
            "strict_source_expected"
        ]["strict_source_binding_semantic_sha256"],
        "source_definition_bindings": copy.deepcopy(
            list(SOURCE_DEFINITION_BINDINGS)
        ),
        "operator_program_set_semantic_sha256": data[
            "operator_program_set_semantic_sha256"
        ],
        "claims": copy.deepcopy(POSITIVE_CLAIMS),
        "strict_source_evidence_status": STRICT_SOURCE_EVIDENCE_STATUS,
        "selection_disclosures": copy.deepcopy(SELECTION_DISCLOSURES),
        "negative_claims": copy.deepcopy(NEGATIVE_CLAIMS),
    }
    return validate_proof_v4_4(
        _seal_artifact(body),
        manifest=manifest_payload,
        input_receipt=input_payload,
        data_field_receipt=data,
        two_pass_equivalence_receipt=equivalence,
    )


_READBACK_BINDING_FIELDS = frozenset(
    {
        "filename",
        "byte_sha256",
        "semantic_sha256",
        "size_bytes",
        "mode",
        "uid",
        "nlink",
    }
)
_READBACK_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "evidence_contract_version",
        "run_id",
        "cycle_id",
        "readback_scope",
        "artifact_bindings",
        "proof_semantic_sha256",
        "operator_program_set_semantic_sha256",
        "claims",
        "strict_source_evidence_status",
        "selection_disclosures",
        "negative_claims",
        "external_predecessor_revalidated",
        "immutable_source_revalidated",
        "protected_controls_revalidated",
        "external_state_claimed",
        "artifact_semantic_sha256",
    }
)
_SAFE_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")


def _artifact_semantic_sha(filename: str, artifact: Mapping[str, Any]) -> str:
    if filename == INPUT_MANIFEST_FILENAME:
        return semantic_sha256_v4_4(artifact)
    return _sha256(
        artifact.get("artifact_semantic_sha256"),
        f"{filename} artifact semantic SHA",
    )


def _validate_readback_binding_structure_v4_4(
    value: Any,
) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != len(INPUT_FILENAMES):
        raise _error("readback must bind the exact first five artifacts")
    normalized: list[dict[str, Any]] = []
    uid: int | None = None
    total_size = 0
    for index, (row_value, filename) in enumerate(
        zip(value, INPUT_FILENAMES, strict=True), start=1
    ):
        row = _exact_object(
            row_value, _READBACK_BINDING_FIELDS, f"readback binding[{index}]"
        )
        if row["filename"] != filename:
            raise _error("readback artifact filename/order mismatch")
        _sha256(row["byte_sha256"], "readback artifact byte SHA")
        _sha256(row["semantic_sha256"], "readback artifact semantic SHA")
        size = _positive_int(row["size_bytes"], "readback artifact size")
        mode = _nonnegative_int(row["mode"], "readback artifact mode")
        owner = _nonnegative_int(row["uid"], "readback artifact uid")
        nlink = _positive_int(row["nlink"], "readback artifact nlink")
        if mode != 0o600 or nlink != 1:
            raise _error("readback artifact mode/link identity mismatch")
        if filename == INPUT_MANIFEST_FILENAME and size > RESOURCE_CONTRACT[
            "manifest_max_bytes"
        ]:
            raise _error("readback manifest exceeds fixed size")
        if size > RESOURCE_CONTRACT["strict_artifact_max_bytes"]:
            raise _error("readback artifact exceeds fixed size")
        if uid is None:
            uid = owner
        elif owner != uid:
            raise _error("readback artifacts must have one owner uid")
        total_size += size
        normalized.append(row)
    if total_size > RESOURCE_CONTRACT["strict_bundle_max_bytes"]:
        raise _error("readback bound bundle exceeds fixed size")
    return normalized


def _validate_readback_bindings(
    value: Any, *, artifacts: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    normalized = _validate_readback_binding_structure_v4_4(value)
    for row, filename in zip(normalized, INPUT_FILENAMES, strict=True):
        raw = canonical_file_bytes_v4_4(artifacts[filename])
        if (
            row["byte_sha256"] != byte_sha256_v4_4(raw)
            or row["semantic_sha256"]
            != _artifact_semantic_sha(filename, artifacts[filename])
            or row["size_bytes"] != len(raw)
        ):
            raise _error("readback artifact byte/semantic/file identity mismatch")
    return normalized


def validate_readback_v4_4(
    value: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate only the already sealed six-file bundle graph."""

    if type(artifacts) is not dict or tuple(artifacts) != INPUT_FILENAMES:
        raise _error("readback artifacts must be the exact ordered first five")
    manifest = validate_input_manifest_v4_4(artifacts[INPUT_MANIFEST_FILENAME])
    input_receipt = validate_input_receipt_v4_4(
        artifacts[INPUT_RECEIPT_FILENAME], manifest=manifest
    )
    data = validate_data_field_receipt_v4_4(
        artifacts[DATA_FIELD_RECEIPT_FILENAME],
        manifest=manifest,
        input_receipt=input_receipt,
    )
    equivalence = validate_two_pass_equivalence_receipt_v4_4(
        artifacts[TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME],
        manifest=manifest,
        input_receipt=input_receipt,
        data_field_receipt=data,
    )
    proof = validate_proof_v4_4(
        artifacts[PROOF_FILENAME],
        manifest=manifest,
        input_receipt=input_receipt,
        data_field_receipt=data,
        two_pass_equivalence_receipt=equivalence,
    )
    payload = _exact_object(value, _READBACK_FIELDS, "strict readback")
    payload = _validate_self_sha(payload, "strict readback")
    if (
        payload["schema_version"] != READBACK_SCHEMA_VERSION
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
    ):
        raise _error("strict readback schema/protocol mismatch")
    if (
        type(payload["run_id"]) is not str
        or _SAFE_RUN_ID_RE.fullmatch(payload["run_id"]) is None
        or payload["run_id"] != manifest["cycle_id"]
        or payload["cycle_id"] != manifest["cycle_id"]
    ):
        raise _error("readback run/cycle identity mismatch")
    if payload["readback_scope"] != READBACK_SCOPE:
        raise _error("readback scope must remain SEALED_BUNDLE_GRAPH_ONLY")
    payload["artifact_bindings"] = _validate_readback_bindings(
        payload["artifact_bindings"], artifacts=artifacts
    )
    if payload["proof_semantic_sha256"] != proof[
        "artifact_semantic_sha256"
    ]:
        raise _error("readback proof semantic SHA mismatch")
    if (
        payload["operator_program_set_semantic_sha256"]
        != proof["operator_program_set_semantic_sha256"]
        or payload["operator_program_set_semantic_sha256"]
        != data["operator_program_set_semantic_sha256"]
        or payload["operator_program_set_semantic_sha256"]
        != equivalence["operator_program_set_semantic_sha256"]
    ):
        raise _error("readback operator program-set SHA mismatch")
    payload["claims"] = _validate_fixed_object(
        payload["claims"], POSITIVE_CLAIMS, "readback positive claims"
    )
    if payload["strict_source_evidence_status"] != STRICT_SOURCE_EVIDENCE_STATUS:
        raise _error("readback strict-source evidence status mismatch")
    payload["selection_disclosures"] = _validate_fixed_object(
        payload["selection_disclosures"],
        SELECTION_DISCLOSURES,
        "readback selection disclosures",
    )
    payload["negative_claims"] = _validate_fixed_object(
        payload["negative_claims"], NEGATIVE_CLAIMS, "readback negative claims"
    )
    for field in (
        "external_predecessor_revalidated",
        "immutable_source_revalidated",
        "protected_controls_revalidated",
        "external_state_claimed",
    ):
        if payload[field] is not False:
            raise _error(f"readback {field} must remain false")
    return payload


def build_readback_v4_4(
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if type(artifacts) is not dict or tuple(artifacts) != INPUT_FILENAMES:
        raise _error("readback builder requires the exact ordered first five")
    proof = artifacts[PROOF_FILENAME]
    if len(artifact_bindings) != len(INPUT_FILENAMES):
        raise _error("readback builder binding inventory mismatch")
    normalized_bindings: list[dict[str, Any]] = []
    base_fields = _READBACK_BINDING_FIELDS - {"semantic_sha256"}
    for index, (binding_value, filename) in enumerate(
        zip(artifact_bindings, INPUT_FILENAMES, strict=True), start=1
    ):
        if type(binding_value) is not dict:
            raise _error(f"readback builder binding[{index}] must be an object")
        fields = set(binding_value)
        if fields != set(base_fields) and fields != set(_READBACK_BINDING_FIELDS):
            raise _error("readback builder binding fields mismatch")
        binding = copy.deepcopy(dict(binding_value))
        if binding.get("filename") != filename:
            raise _error("readback builder binding filename/order mismatch")
        derived_semantic = _artifact_semantic_sha(filename, artifacts[filename])
        if "semantic_sha256" in binding and binding["semantic_sha256"] != derived_semantic:
            raise _error("readback builder supplied semantic SHA mismatch")
        binding["semantic_sha256"] = derived_semantic
        normalized_bindings.append(binding)
    body = {
        "schema_version": READBACK_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "run_id": run_id,
        "cycle_id": artifacts[INPUT_MANIFEST_FILENAME]["cycle_id"],
        "readback_scope": READBACK_SCOPE,
        "artifact_bindings": normalized_bindings,
        "proof_semantic_sha256": proof["artifact_semantic_sha256"],
        "operator_program_set_semantic_sha256": proof[
            "operator_program_set_semantic_sha256"
        ],
        "claims": copy.deepcopy(POSITIVE_CLAIMS),
        "strict_source_evidence_status": STRICT_SOURCE_EVIDENCE_STATUS,
        "selection_disclosures": copy.deepcopy(SELECTION_DISCLOSURES),
        "negative_claims": copy.deepcopy(NEGATIVE_CLAIMS),
        "external_predecessor_revalidated": False,
        "immutable_source_revalidated": False,
        "protected_controls_revalidated": False,
        "external_state_claimed": False,
    }
    return validate_readback_v4_4(
        _seal_artifact(body), artifacts=dict(artifacts)
    )


def validate_complete_bundle_v4_4(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate the exact six-file graph without consulting external state."""

    if type(values) is not dict or tuple(values) != BUNDLE_FILENAMES:
        raise _error("complete bundle inventory/order must be the fixed six files")
    artifacts = {
        filename: copy.deepcopy(dict(values[filename]))
        for filename in INPUT_FILENAMES
    }
    readback = validate_readback_v4_4(
        values[READBACK_FILENAME], artifacts=artifacts
    )
    return {**artifacts, READBACK_FILENAME: readback}


# Readability aliases used by the private bundle runner.
build_strict_computability_proof_v4_4 = build_proof_v4_4
validate_strict_computability_proof_v4_4 = validate_proof_v4_4
build_readback_report_v4_4 = build_readback_v4_4
validate_readback_report_v4_4 = validate_readback_v4_4


def _validate_context_free_two_pass_claims_v4_4(
    payload: Mapping[str, Any],
) -> None:
    _validate_fixed_object(
        payload["claims"], _EQUIVALENCE_CLAIMS, "standalone equivalence claims"
    )
    if (
        payload["operator_program_set_semantic_sha256"]
        != OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
    ):
        raise _error("standalone two-pass operator program-set SHA mismatch")
    proof_pit, proof_pit_packed = _decode_binary_mask_descriptor_v4_4(
        payload["proof_pit_mask"], label="standalone proof-output PIT bitmap"
    )
    proof_date_axis = proof_pit["date_axis"]
    proof_symbol_axis = proof_pit["symbol_axis"]
    candidate_masks, _candidate_mask_packed = (
        _validate_candidate_non_null_mask_set_v4_4(
            payload["candidate_non_null_masks"],
            proof_pit_mask=proof_pit,
            proof_pit_packed=proof_pit_packed,
            data_field_receipt={
                "proof_output_calendar": proof_date_axis,
                "historical_symbol_axis": {"descriptor": proof_symbol_axis},
            },
        )
    )
    mask_rows = candidate_masks["rows"]
    passes = payload["passes"]
    if type(passes) is not list or len(passes) != len(PASS_IDS):
        raise _error("standalone two-pass receipt requires the exact two passes")
    normalized_passes: list[dict[str, Any]] = []
    for pass_index, (pass_value, pass_id) in enumerate(
        zip(passes, PASS_IDS, strict=True), start=1
    ):
        pass_row = _exact_object(
            pass_value,
            _PASS_DESCRIPTOR_FIELDS,
            f"standalone pass descriptor[{pass_index}]",
        )
        if pass_row["pass_id"] != pass_id:
            raise _error("standalone pass identity/order mismatch")
        collection = _exact_object(
            pass_row["collection"],
            _COLLECTION_DESCRIPTOR_FIELDS,
            f"standalone {pass_id} collection",
        )
        if (
            collection["schema_version"] != COLLECTION_DESCRIPTOR_SCHEMA_VERSION
            or collection["pass_id"] != pass_id
            or collection["operator_program_set_semantic_sha256"]
            != OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
        ):
            raise _error("standalone collection fixed identity/claim mismatch")
        collection["strict_source_binding"] = _validate_strict_source_expected(
            collection["strict_source_binding"]
        )
        source_axis = _validate_axis_descriptor(
            collection["source_calendar_descriptor"],
            "standalone collection source calendar",
        )
        collection_symbol_axis = _validate_axis_descriptor(
            collection["historical_symbol_axis_descriptor"],
            "standalone collection symbol axis",
        )
        if (
            collection_symbol_axis != proof_symbol_axis
            or source_axis["count"]
            != proof_date_axis["count"]
            + RESOURCE_CONTRACT["halo_session_count"]
            or source_axis["last"] != proof_date_axis["last"]
        ):
            raise _error("standalone collection/proof output axes mismatch")
        _sha256(
            collection["pit_mask_semantic_sha256"],
            "standalone collection PIT-mask SHA",
        )
        _sha256(
            collection["block_manifest_semantic_sha256"],
            "standalone collection block-manifest SHA",
        )
        input_rows = collection["input_matrix_descriptors"]
        if type(input_rows) is not list or len(input_rows) != len(INPUT_FIELDS):
            raise _error("standalone collection requires four input matrices")
        normalized_input_rows: list[dict[str, Any]] = []
        for input_index, (input_value, field) in enumerate(
            zip(input_rows, INPUT_FIELDS, strict=True), start=1
        ):
            input_row = _exact_object(
                input_value,
                _INPUT_MATRIX_DESCRIPTOR_ROW_FIELDS,
                f"standalone collection input[{input_index}]",
            )
            if input_row["field"] != field:
                raise _error("standalone collection input field/order mismatch")
            input_row["descriptor"] = validate_matrix_descriptor_v4_4(
                input_row["descriptor"],
                label=f"standalone collection {field}",
                expected_date_axis=source_axis,
                expected_symbol_axis=collection_symbol_axis,
            )
            normalized_input_rows.append(input_row)
        collection["input_matrix_descriptors"] = normalized_input_rows
        projected_rows = _positive_int(
            collection["projected_row_count_per_pass"],
            "standalone collection projected row count",
        )
        if projected_rows != source_axis["count"] * collection_symbol_axis["count"]:
            raise _error("standalone collection projected row count mismatch")
        collection_sha256 = _sha256(
            collection["collection_sha256"], "standalone collection SHA"
        )
        if collection_sha256 != semantic_sha256_v4_4(
            _collection_content(collection)
        ):
            raise _error("standalone collection semantic SHA mismatch")
        engines = pass_row["engines"]
        if type(engines) is not list or len(engines) != len(ENGINE_IDS):
            raise _error("standalone pass requires both fixed engines")
        normalized_engines: list[dict[str, Any]] = []
        for engine_index, (engine_value, engine_id) in enumerate(
            zip(engines, ENGINE_IDS, strict=True), start=1
        ):
            engine = _exact_object(
                engine_value,
                _ENGINE_PASS_FIELDS,
                f"standalone engine[{pass_index},{engine_index}]",
            )
            if (
                engine["schema_version"] != ENGINE_PASS_SCHEMA_VERSION
                or engine["protocol_version"] != PROTOCOL_VERSION
                or engine["evidence_contract_version"]
                != EVIDENCE_CONTRACT_VERSION
                or engine["pass_id"] != pass_id
                or engine["engine_id"] != engine_id
                or engine["collection_sha256"] != collection_sha256
                or engine["operator_program_set_semantic_sha256"]
                != OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
            ):
                raise _error("standalone engine fixed identity/claim mismatch")
            date_axis = _validate_axis_descriptor(
                engine["date_axis"], "standalone engine date axis"
            )
            symbol_axis = _validate_axis_descriptor(
                engine["symbol_axis"], "standalone engine symbol axis"
            )
            _positive_int(engine["row_count"], "standalone engine row count")
            _positive_int(
                engine["column_count"], "standalone engine column count"
            )
            if (
                date_axis != proof_date_axis
                or symbol_axis != proof_symbol_axis
                or engine["row_count"] != date_axis["count"]
                or engine["column_count"] != symbol_axis["count"]
                or engine["proof_pit_mask_semantic_sha256"]
                != semantic_sha256_v4_4(proof_pit)
            ):
                raise _error("standalone engine proof axes/PIT binding mismatch")
            if (
                engine["signal_computability_proven"] is not True
                or engine["labels_present"] is not False
                or engine["outcomes_present"] is not False
                or engine["statistics_run"] is not False
                or engine["tolerance_used"] is not False
                or engine["rounding_used"] is not False
                or engine["authority"] is not False
            ):
                raise _error("standalone engine promoted a prohibited claim")
            supplied_result_sha = _sha256(
                engine["result_semantic_sha256"],
                "standalone engine result semantic SHA",
            )
            if supplied_result_sha != semantic_sha256_v4_4(
                {
                    key: item
                    for key, item in engine.items()
                    if key != "result_semantic_sha256"
                }
            ):
                raise _error("standalone engine result semantic SHA mismatch")
            candidates = engine["candidates"]
            if type(candidates) is not list or len(candidates) != len(
                SOURCE_DEFINITION_BINDINGS
            ):
                raise _error("standalone engine requires the exact five candidates")
            normalized_candidates: list[dict[str, Any]] = []
            for candidate_index, (candidate_value, expected, mask_row) in enumerate(
                zip(
                    candidates,
                    SOURCE_DEFINITION_BINDINGS,
                    mask_rows,
                    strict=True,
                ),
                start=1,
            ):
                candidate = _exact_object(
                    candidate_value,
                    _ENGINE_CANDIDATE_FIELDS,
                    f"standalone engine candidate[{candidate_index}]",
                )
                _positive_int(
                    candidate["order"],
                    f"standalone engine candidate {candidate_index} order",
                )
                if (
                    candidate["order"] != expected["order"]
                    or candidate["name"] != expected["name"]
                    or type(candidate["direction"]) is not float
                    or candidate["direction"] != float(expected["direction"])
                    or candidate["operator_program_semantic_sha256"]
                    != expected["operator_program_sha256"]
                ):
                    raise _error("standalone engine candidate claim mismatch")
                raw = validate_matrix_descriptor_v4_4(
                    candidate["raw_matrix"],
                    label=f"standalone {candidate['name']} raw output",
                    expected_date_axis=date_axis,
                    expected_symbol_axis=symbol_axis,
                )
                adjusted = validate_matrix_descriptor_v4_4(
                    candidate["direction_adjusted_matrix"],
                    label=f"standalone {candidate['name']} adjusted output",
                    expected_date_axis=date_axis,
                    expected_symbol_axis=symbol_axis,
                )
                mask = validate_binary_mask_descriptor_v4_4(
                    mask_row["mask"],
                    label=f"standalone {candidate['name']} non-null bitmap",
                    expected_date_axis=date_axis,
                    expected_symbol_axis=symbol_axis,
                )
                if (
                    raw["finite_count"] <= 0
                    or raw["positive_infinity_count"] != 0
                    or raw["negative_infinity_count"] != 0
                    or candidate["non_null_mask_semantic_sha256"]
                    != semantic_sha256_v4_4(mask)
                    or mask_row["raw_matrix_bit_pattern_sha256"]
                    != raw["bit_pattern_sha256"]
                    or raw["finite_count"] != mask["one_count"]
                    or raw["nan_count"] != mask["zero_count"]
                ):
                    raise _error(
                        "standalone engine candidate matrix/mask mismatch"
                    )
                _validate_direction_adjustment(
                    raw, adjusted, expected["direction"]
                )
                candidate["raw_matrix"] = raw
                candidate["direction_adjusted_matrix"] = adjusted
                normalized_candidates.append(candidate)
            engine["candidates"] = normalized_candidates
            normalized_engines.append(engine)
        if canonical_json_bytes_v4_4(
            _normalized_engine_result(normalized_engines[0])
        ) != canonical_json_bytes_v4_4(
            _normalized_engine_result(normalized_engines[1])
        ):
            raise _error("standalone independent engines differ")
        pass_row["collection"] = collection
        pass_row["engines"] = normalized_engines
        normalized_passes.append(pass_row)
    first, second = normalized_passes
    if (
        first["collection"]["collection_sha256"]
        != second["collection"]["collection_sha256"]
        or canonical_json_bytes_v4_4(_collection_content(first["collection"]))
        != canonical_json_bytes_v4_4(_collection_content(second["collection"]))
    ):
        raise _error("standalone fresh collections differ")
    for engine_index in range(len(ENGINE_IDS)):
        if canonical_json_bytes_v4_4(
            _normalized_engine_result(first["engines"][engine_index])
        ) != canonical_json_bytes_v4_4(
            _normalized_engine_result(second["engines"][engine_index])
        ):
            raise _error("standalone fresh-pass engine descriptors differ")


def _validate_context_free_fixed_claims_v4_4(
    filename: str, payload: Mapping[str, Any]
) -> None:
    if filename in {
        INPUT_RECEIPT_FILENAME,
        DATA_FIELD_RECEIPT_FILENAME,
        TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME,
        PROOF_FILENAME,
    }:
        _validate_context_free_cycle_identity_v4_4(
            payload, label=f"standalone {filename}"
        )
    _validate_fixed_object(
        payload["selection_disclosures"],
        SELECTION_DISCLOSURES,
        "standalone selection disclosures",
    )
    _validate_fixed_object(
        payload["negative_claims"],
        NEGATIVE_CLAIMS,
        "standalone negative claims",
    )
    if filename == INPUT_RECEIPT_FILENAME:
        cutoff = payload["cutoff"]
        snapshot_id = payload["snapshot_id"]
        payload["preregistration"] = _validate_preregistration(
            payload["preregistration"],
            expected_preregistration_cycle_id=(
                deterministic_preregistration_cycle_id_v4_4(
                    cutoff=cutoff, snapshot_id=snapshot_id
                )
            ),
        )
        manifest_binding = _exact_object(
            payload["input_manifest_binding"],
            _INPUT_MANIFEST_BINDING_FIELDS,
            "input receipt manifest binding",
        )
        if manifest_binding["filename"] != INPUT_MANIFEST_FILENAME:
            raise _error("input receipt manifest binding filename mismatch")
        _sha256(manifest_binding["byte_sha256"], "manifest byte SHA")
        _sha256(manifest_binding["semantic_sha256"], "manifest semantic SHA")
        _positive_int(manifest_binding["size_bytes"], "manifest size")
        _validate_fixed_object(
            payload["stage1_claims"], _STAGE1_CLAIMS, "standalone stage-1 claims"
        )
        _validate_fixed_object(
            payload["resource_contract"],
            RESOURCE_CONTRACT,
            "standalone resource contract",
        )
        _validate_source_definition_bindings(payload["source_definition_bindings"])
        _validate_strict_source_expected(payload["strict_source_expected"])
        _validate_code_binding_set(payload["code_binding_set"])
        _validate_protected_control_sha256(payload["protected_control_sha256"])
        _sha256(
            payload["runtime_binding_semantic_sha256"],
            "standalone input-receipt runtime SHA",
        )
        return
    if filename == DATA_FIELD_RECEIPT_FILENAME:
        _sha256(
            payload["input_manifest_semantic_sha256"],
            "standalone data/field manifest semantic SHA",
        )
        _sha256(
            payload["input_receipt_semantic_sha256"],
            "standalone data/field input-receipt semantic SHA",
        )
        strict_source = _validate_strict_source_expected(
            payload["strict_source_binding"]
        )
        source_calendar, expected_proof_calendar = _validate_source_calendar(
            payload["source_calendar"],
            cutoff=payload["cutoff"],
            proof_output_start=payload["proof_output_start"],
        )
        proof_calendar = _validate_axis_descriptor(
            payload["proof_output_calendar"], "proof output calendar"
        )
        if proof_calendar != expected_proof_calendar:
            raise _error("standalone proof output calendar mismatch")
        historical_date_axis = _validate_axis_descriptor(
            payload["historical_date_axis_descriptor"], "historical date axis"
        )
        if historical_date_axis != source_calendar["descriptor"]:
            raise _error("standalone historical date axis mismatch")
        if strict_source["source_calendar_semantic_sha256"] != (
            source_calendar_semantic_sha256_v4_4(
                source_calendar["open_sessions"], cutoff=payload["cutoff"]
            )
        ):
            raise _error("standalone source calendar semantic SHA mismatch")
        cutoff_scope = _exact_object(
            payload["cutoff_full_a_scope"],
            _CUTOFF_FULL_A_FIELDS,
            "cutoff full-A scope",
        )
        _positive_int(
            cutoff_scope["count"], "standalone cutoff full-A scope count"
        )
        if (
            cutoff_scope["count"] != strict_source["full_a_scope_count"]
            or cutoff_scope["sha256"] != strict_source["full_a_scope_sha256"]
            or cutoff_scope["hash_algorithm"] != FULL_A_HASH_ALGORITHM
        ):
            raise _error("standalone cutoff full-A scope mismatch")
        historical_symbols = _validate_historical_symbol_axis(
            payload["historical_symbol_axis"],
            cutoff_full_a_count=cutoff_scope["count"],
        )
        pit_membership = _validate_pit_membership(
            payload["pit_membership_contract"],
            historical_symbol_count=historical_symbols["descriptor"]["count"],
            expected_byte_sha256=strict_source["pit_membership_byte_sha256"],
        )
        pit_descriptor = validate_binary_mask_descriptor_v4_4(
            payload["pit_mask_descriptor"],
            label="PIT mask descriptor",
            expected_date_axis=historical_date_axis,
            expected_symbol_axis=historical_symbols["descriptor"],
        )
        block_manifest = validate_block_manifest_v4_4(
            payload["block_manifest"],
            source_sessions=source_calendar["open_sessions"],
            expected_symbol_axis=historical_symbols["descriptor"],
        )
        missing = _validate_count_map(
            payload["field_missing_counts"],
            "field missing counts",
            require_zero=False,
        )
        outside = _validate_count_map(
            payload["outside_pit_non_null_counts"],
            "outside-PIT non-null counts",
            require_zero=True,
        )
        _nonnegative_int(
            payload["bars_outside_pit_interval_count"],
            "bars outside valid PIT interval count",
        )
        _nonnegative_int(
            payload["ignored_pre_analysis_row_count"],
            "ignored pre-analysis row count",
        )
        projected_rows = _positive_int(
            payload["projected_row_count_per_pass"],
            "projected row count per pass",
        )
        expected_rows = historical_date_axis["count"] * historical_symbols[
            "descriptor"
        ]["count"]
        if (
            projected_rows != expected_rows
            or pit_descriptor["bit_count"] != expected_rows
            or projected_rows
            > RESOURCE_CONTRACT["projected_row_count_per_pass_max"]
        ):
            raise _error("standalone projected row/bitmap shape mismatch")
        if payload["strict_source_evidence_status"] != STRICT_SOURCE_EVIDENCE_STATUS:
            raise _error("standalone data/field strict-source status mismatch")
        if payload["table_projection"] != list(TABLE_PROJECTION):
            raise _error("standalone data/field table projection mismatch")
        if payload["field_adapters"] != list(FIELD_SEMANTICS):
            raise _error("standalone data/field adapters mismatch")
        program_set = validate_operator_program_set_v4_4(
            payload["operator_program_set"]
        )
        if (
            payload["operator_program_set_semantic_sha256"]
            != program_set["artifact_semantic_sha256"]
            or payload["operator_program_set_semantic_sha256"]
            != OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
        ):
            raise _error("standalone data/field operator program-set mismatch")
        source_access = _exact_object(
            payload["source_access"],
            _SOURCE_ACCESS_FIELDS,
            "standalone data/field source access",
        )
        for field in (
            "current_pointer_read",
            "current_components_read",
            "serving_read",
            "csv_read",
        ):
            if source_access[field] is not False:
                raise _error(f"standalone data/field source access {field} must be false")
        if (
            source_access["recorded_latest_pointer_byte_sha256"]
            != strict_source["recorded_latest_pointer_byte_sha256"]
            or source_access["recorded_components_byte_sha256"]
            != strict_source["recorded_components_byte_sha256"]
        ):
            raise _error("standalone data/field source access SHA mismatch")
        payload["strict_source_binding"] = strict_source
        payload["source_calendar"] = source_calendar
        payload["proof_output_calendar"] = proof_calendar
        payload["historical_date_axis_descriptor"] = historical_date_axis
        payload["historical_symbol_axis"] = historical_symbols
        payload["pit_membership_contract"] = pit_membership
        payload["pit_mask_descriptor"] = pit_descriptor
        payload["block_manifest"] = block_manifest
        payload["field_missing_counts"] = missing
        payload["outside_pit_non_null_counts"] = outside
        payload["source_access"] = source_access
        return
    if filename == TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME:
        for field in (
            "input_manifest_semantic_sha256",
            "input_receipt_semantic_sha256",
            "data_field_receipt_semantic_sha256",
        ):
            _sha256(payload[field], f"standalone two-pass predecessor {field}")
        _validate_context_free_two_pass_claims_v4_4(payload)
        return
    if filename == PROOF_FILENAME:
        _validate_fixed_object(
            payload["claims"], POSITIVE_CLAIMS, "standalone proof claims"
        )
        if payload["strict_source_evidence_status"] != STRICT_SOURCE_EVIDENCE_STATUS:
            raise _error("standalone proof strict-source status mismatch")
        if (
            payload["operator_program_set_semantic_sha256"]
            != OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
        ):
            raise _error("standalone proof operator program-set SHA mismatch")
        predecessors = _validate_proof_predecessor_binding_structure_v4_4(
            payload["predecessor_bindings"]
        )
        expected_preregistration_cycle_id = (
            deterministic_preregistration_cycle_id_v4_4(
                cutoff=payload["cutoff"], snapshot_id=payload["snapshot_id"]
            )
        )
        if (
            predecessors["preregistration_readback"]["cycle_id"]
            != expected_preregistration_cycle_id
        ):
            raise _error("standalone proof preregistration cycle_id mismatch")
        _sha256(
            payload["strict_source_binding_semantic_sha256"],
            "standalone proof strict-source semantic SHA",
        )
        _validate_source_definition_bindings(payload["source_definition_bindings"])
        return
    if filename == READBACK_FILENAME:
        _validate_fixed_object(
            payload["claims"], POSITIVE_CLAIMS, "standalone readback claims"
        )
        if payload["readback_scope"] != READBACK_SCOPE:
            raise _error("standalone readback scope mismatch")
        if payload["strict_source_evidence_status"] != STRICT_SOURCE_EVIDENCE_STATUS:
            raise _error("standalone readback strict-source status mismatch")
        if (
            payload["operator_program_set_semantic_sha256"]
            != OPERATOR_PROGRAM_SET_SEMANTIC_SHA256
        ):
            raise _error("standalone readback operator program-set SHA mismatch")
        if (
            type(payload["run_id"]) is not str
            or _SAFE_RUN_ID_RE.fullmatch(payload["run_id"]) is None
            or payload["run_id"] != payload["cycle_id"]
        ):
            raise _error("standalone readback run/cycle identity mismatch")
        _validate_embedded_cycle_id_v4_4(
            payload["cycle_id"], label="standalone readback cycle_id"
        )
        _sha256(
            payload["proof_semantic_sha256"],
            "standalone readback proof semantic SHA",
        )
        _validate_readback_binding_structure_v4_4(
            payload["artifact_bindings"]
        )
        for field in (
            "external_predecessor_revalidated",
            "immutable_source_revalidated",
            "protected_controls_revalidated",
            "external_state_claimed",
        ):
            if payload[field] is not False:
                raise _error(f"standalone readback {field} must remain false")


def validate_artifact_v4_4(
    filename: str, value: Mapping[str, Any]
) -> dict[str, Any]:
    """Perform the strongest context-free check available for one file."""

    if filename == INPUT_MANIFEST_FILENAME:
        return validate_input_manifest_v4_4(value)
    specifications = {
        INPUT_RECEIPT_FILENAME: (
            _INPUT_RECEIPT_FIELDS,
            INPUT_RECEIPT_SCHEMA_VERSION,
            "input receipt",
        ),
        DATA_FIELD_RECEIPT_FILENAME: (
            _DATA_FIELD_RECEIPT_FIELDS,
            DATA_FIELD_RECEIPT_SCHEMA_VERSION,
            "data/field receipt",
        ),
        TWO_PASS_EQUIVALENCE_RECEIPT_FILENAME: (
            _TWO_PASS_RECEIPT_FIELDS,
            TWO_PASS_EQUIVALENCE_RECEIPT_SCHEMA_VERSION,
            "two-pass receipt",
        ),
        PROOF_FILENAME: (_PROOF_FIELDS, PROOF_SCHEMA_VERSION, "strict proof"),
        READBACK_FILENAME: (
            _READBACK_FIELDS,
            READBACK_SCHEMA_VERSION,
            "strict readback",
        ),
    }
    if filename not in specifications:
        raise _error("artifact filename is outside the fixed six-file inventory")
    fields, schema, label = specifications[filename]
    payload = _exact_object(value, fields, label)
    payload = _validate_self_sha(payload, label)
    if (
        payload["schema_version"] != schema
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["evidence_contract_version"] != EVIDENCE_CONTRACT_VERSION
    ):
        raise _error(f"{label} schema/protocol mismatch")
    _validate_context_free_fixed_claims_v4_4(filename, payload)
    return payload


def private_bundle_contract_v4_4() -> Any:
    """Return the shared private-I/O contract through a lazy project import."""

    from quant_investor.factors import governance_private_bundle_io as private_io

    return private_io.PrivateBundleContract(
        root_suffix=PRIVATE_ROOT_SUFFIX,
        input_filenames=INPUT_FILENAMES,
        readback_report_filename=READBACK_FILENAME,
        canonicalize=canonical_file_bytes_v4_4,
        validate_artifact=validate_artifact_v4_4,
        validate_complete=validate_complete_bundle_v4_4,
        build_readback_report=build_readback_v4_4,
        max_artifact_bytes=RESOURCE_CONTRACT["strict_artifact_max_bytes"],
        max_bundle_bytes=RESOURCE_CONTRACT["strict_bundle_max_bytes"],
    )
