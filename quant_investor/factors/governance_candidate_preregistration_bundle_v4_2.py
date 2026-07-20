"""Owner-private bundle contract for prospective Factor v4.2 preregistration.

The module binds one already-validated strict full-A cutoff, the exact source
and code identities, and the pure PRECOMMITTED -> DISCOVERY graph.  Historical
readback is deliberately self-contained: it validates the recorded pointer
and components bytes and the immutable reopen descriptor without consulting a
current ``_latest.json`` or any mutable production control.

Filesystem publication is delegated to :mod:`governance_private_bundle_io`.
The bundle-internal report therefore records precommit intent only; successful
rename, fsync, durability, and no-clobber evidence belongs to the caller's live
return value and is never asserted inside staged bytes.
"""

from __future__ import annotations

import base64
import binascii
import copy
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from collections.abc import Mapping, Sequence
from typing import Any

from quant_investor.factors import governance_candidate_preregistration_v4_2 as prereg
from quant_investor.factors.governance_cycle_state_v4_1 import (
    DISCOVERY,
    PRECOMMITTED,
    build_genesis_cycle_state_v4_1,
    byte_sha256 as cycle_state_byte_sha256_v4_1,
    validate_cycle_state_v4_1,
)
from quant_investor.factors.governance_private_bundle_io import (
    PrivateBundleContract,
    publish_private_bundle,
    readback_private_bundle,
)
from quant_investor.factors.governance_source_readback_v4_1 import (
    BoundCutoffInputsV4_1,
    INPUT_BINDING_SCHEMA_VERSION,
    SOURCE_USE_PROHIBITED,
    binding_semantic_sha256_v4_1,
)


ROOT_SUFFIX_V4_2 = (
    "reports",
    "factor_governance",
    "private",
    "v4_2_candidate_preregistration",
)

AQUANT_IDEA_SOURCE_RECEIPT_FILENAME_V4_2 = "aquant_idea_source_receipt.v4_2.json"
MYQUANT_ALPHA158_SOURCE_RECEIPT_FILENAME_V4_2 = (
    "myquant_alpha158_source_receipt.v4_2.json"
)
OPERATOR_SEMANTICS_FILENAME_V4_2 = "operator_semantics.v4_2.json"
COMPARISON_CATALOG_RECEIPT_FILENAME_V4_2 = (
    "comparison_catalog_receipt.v4_2.json"
)
CANDIDATE_SELECTION_SPEC_FILENAME_V4_2 = "candidate_selection_spec.v4_2.json"
STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2 = (
    "strict_full_a_source_binding.v4_2.json"
)
CODE_BINDING_SET_FILENAME_V4_2 = "code_binding_set.v4_2.json"
FUTURE_SOURCE_ENVELOPE_FILENAME_V4_2 = "future_source_envelope.v4_2.json"
CYCLE_ROOT_FILENAME_V4_2 = "cycle_root.v4_2.json"
DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_2 = (
    "definition_identity_collision_audit.v4_2.json"
)
PRECOMMITTED_STATE_FILENAME_V4_2 = "cycle_state.precommitted.v4_1.json"
DISCOVERY_SOURCE_NODE_FILENAME_V4_2 = "discovery_source_node.v4_2.json"
DISCOVERY_STATE_FILENAME_V4_2 = "cycle_state.discovery.v4_1.json"
PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_2 = (
    "prereg_discovery_orchestration.v4_2.json"
)
READBACK_REPORT_FILENAME_V4_2 = "candidate_preregistration_readback.v4_2.json"

INPUT_FILENAMES_V4_2 = (
    AQUANT_IDEA_SOURCE_RECEIPT_FILENAME_V4_2,
    MYQUANT_ALPHA158_SOURCE_RECEIPT_FILENAME_V4_2,
    OPERATOR_SEMANTICS_FILENAME_V4_2,
    COMPARISON_CATALOG_RECEIPT_FILENAME_V4_2,
    CANDIDATE_SELECTION_SPEC_FILENAME_V4_2,
    STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2,
    CODE_BINDING_SET_FILENAME_V4_2,
    FUTURE_SOURCE_ENVELOPE_FILENAME_V4_2,
    CYCLE_ROOT_FILENAME_V4_2,
    DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_2,
    PRECOMMITTED_STATE_FILENAME_V4_2,
    DISCOVERY_SOURCE_NODE_FILENAME_V4_2,
    DISCOVERY_STATE_FILENAME_V4_2,
    PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_2,
)

CODE_BINDING_PATHS_V4_2 = (
    "scripts/build_factor_v4_2_candidate_preregistration.py",
    "quant_investor/factors/governance_candidate_preregistration_v4_2.py",
    "quant_investor/factors/governance_candidate_preregistration_bundle_v4_2.py",
    "quant_investor/factors/governance_cycle_state_v4_1.py",
    "quant_investor/factors/governance_private_bundle_io.py",
    "quant_investor/factors/governance_source_readback_v4_1.py",
    "quant_investor/factors/governance_screening_v4.py",
    "quant_investor/codex_review/storage.py",
    "quant_investor/market/pit_universe.py",
    "quant_investor/factors/governance_source_v4_1.py",
)

STRICT_SOURCE_SCHEMA_VERSION = "factor-governance-strict-full-a-source-binding.v4.2"
CODE_BINDING_SET_SCHEMA_VERSION = "factor-governance-code-binding-set.v4.2"
CYCLE_ROOT_SCHEMA_VERSION = "factor-governance-candidate-cycle-root.v4.2"
READBACK_REPORT_SCHEMA_VERSION = (
    "factor-governance-candidate-preregistration-readback.v4.2"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CN_SYMBOL = re.compile(r"[0-9]{6}\.(?:SH|SZ|BJ)")
_SNAPSHOT_ID = re.compile(r"\d{8}T\d{6}Z")


class FactorGovernanceCandidatePreregistrationBundleV4_2Error(ValueError):
    """Raised when a v4.2 private bundle fails closed."""


def _error(message: str) -> FactorGovernanceCandidatePreregistrationBundleV4_2Error:
    return FactorGovernanceCandidatePreregistrationBundleV4_2Error(message)


def _exact(value: Any, fields: set[str] | frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise _error(f"{label} field names must be strings")
    missing = sorted(set(fields) - set(payload))
    unknown = sorted(set(payload) - set(fields))
    if missing or unknown:
        raise _error(
            f"{label} fields invalid: missing={','.join(missing) or '-'};"
            f"unknown={','.join(unknown) or '-'}"
        )
    prereg.canonical_json_bytes_v4_2(payload)
    return payload


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise _error(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise _error(f"{label} must be a nonnegative integer")
    return value


def _iso_date(value: Any, label: str) -> str:
    if type(value) is not str:
        raise _error(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _error(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise _error(f"{label} must be canonical YYYY-MM-DD")
    return value


def _snapshot_id(value: Any) -> str:
    if type(value) is not str or _SNAPSHOT_ID.fullmatch(value) is None:
        raise _error("snapshot_id must be canonical YYYYMMDDTHHMMSSZ")
    try:
        datetime.strptime(value, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise _error("snapshot_id must be a real canonical UTC timestamp") from exc
    return value


def _absolute_path(value: Any, label: str) -> str:
    if type(value) is not str or not value.startswith("/") or "\x00" in value:
        raise _error(f"{label} must be an absolute normalized path")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise _error(f"{label} must be an absolute normalized path")
    if os.path.abspath(value) != value:
        raise _error(f"{label} must be an absolute normalized path")
    return value


def _declared_path_matches(
    value: Any,
    *,
    expected: str,
    anchors: Sequence[Path],
    label: str,
) -> None:
    if type(value) is not str or not value or "\x00" in value:
        raise _error(f"{label} declared path is missing or unsafe")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts if part != "/"):
        raise _error(f"{label} declared path is unsafe")
    expected_path = Path(expected)
    candidates = (
        {Path(_absolute_path(value, label))}
        if path.is_absolute()
        else {Path(os.path.abspath(anchor / path)) for anchor in anchors}
    )
    if expected_path not in candidates:
        raise _error(f"{label} declared path mismatch")


def _self_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "artifact_semantic_sha256"}


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result["artifact_semantic_sha256"] = prereg.semantic_sha256_v4_2(
        _self_payload(result)
    )
    return result


def _validate_self(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    supplied = _sha256(payload.get("artifact_semantic_sha256"), f"{label} self SHA")
    expected = prereg.semantic_sha256_v4_2(_self_payload(payload))
    if supplied != expected:
        raise _error(f"{label} artifact_semantic_sha256 mismatch")
    return payload


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value}")

    def exact_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=exact_object,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise _error(f"{label} must be strict finite JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(f"{label} must be a JSON object")
    return value


def _raw_evidence(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise _error("raw evidence input must be exact bytes")
    parsed = _strict_json_object(raw, "raw evidence")
    return {
        "encoding": "base64",
        "raw_base64": base64.b64encode(raw).decode("ascii"),
        "size_bytes": len(raw),
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "strict_json_semantic_sha256": prereg.semantic_sha256_v4_2(parsed),
    }


def _validate_raw_evidence(
    value: Any,
    label: str,
    *,
    extra_fields: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    core_fields = {
        "encoding",
        "raw_base64",
        "size_bytes",
        "byte_sha256",
        "strict_json_semantic_sha256",
    }
    payload = _exact(
        value,
        core_fields | set(extra_fields),
        label,
    )
    if payload["encoding"] != "base64":
        raise _error(f"{label}.encoding must be base64")
    if type(payload["raw_base64"]) is not str:
        raise _error(f"{label}.raw_base64 must be a string")
    try:
        raw = base64.b64decode(payload["raw_base64"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise _error(f"{label}.raw_base64 is invalid") from exc
    if base64.b64encode(raw).decode("ascii") != payload["raw_base64"]:
        raise _error(f"{label}.raw_base64 must be canonical")
    if _nonnegative_int(payload["size_bytes"], f"{label}.size_bytes") != len(raw):
        raise _error(f"{label}.size_bytes mismatch")
    if _sha256(payload["byte_sha256"], f"{label}.byte_sha256") != hashlib.sha256(raw).hexdigest():
        raise _error(f"{label}.byte_sha256 mismatch")
    parsed = _strict_json_object(raw, label)
    if (
        _sha256(
            payload["strict_json_semantic_sha256"],
            f"{label}.strict_json_semantic_sha256",
        )
        != prereg.semantic_sha256_v4_2(parsed)
    ):
        raise _error(f"{label}.strict_json_semantic_sha256 mismatch")
    return copy.deepcopy(payload), raw, parsed


def _validate_binding_record(value: Any, label: str) -> dict[str, Any]:
    payload = _exact(value, {"absolute_path", "size_bytes", "sha256"}, label)
    return {
        "absolute_path": _absolute_path(payload["absolute_path"], f"{label}.absolute_path"),
        "size_bytes": _nonnegative_int(payload["size_bytes"], f"{label}.size_bytes"),
        "sha256": _sha256(payload["sha256"], f"{label}.sha256"),
    }


def _validate_table_inventory(value: Any) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "absolute_root",
            "regular_file_count",
            "parquet_file_count",
            "inventory_sha256",
            "parquet_inventory",
            "bound_symbol_inventory",
        },
        "backend table binding",
    )
    rows = payload["parquet_inventory"]
    if not isinstance(rows, list) or not rows:
        raise _error("backend table parquet_inventory must be a non-empty list")
    normalized_rows: list[dict[str, Any]] = []
    previous: str | None = None
    for index, item in enumerate(rows):
        row = _exact(
            item,
            {"relative_path", "size_bytes", "sha256", "hard_link_count", "dataset_member"},
            f"backend table parquet_inventory[{index}]",
        )
        relative = row["relative_path"]
        if (
            type(relative) is not str
            or not relative
            or relative.startswith("/")
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
        ):
            raise _error("table inventory relative_path is unsafe")
        if previous is not None and relative <= previous:
            raise _error("table inventory must be sorted by unique relative_path")
        previous = relative
        if type(row["dataset_member"]) is not bool:
            raise _error("table inventory dataset_member must be boolean")
        normalized_rows.append(
            {
                "relative_path": relative,
                "size_bytes": _nonnegative_int(row["size_bytes"], "table file size"),
                "sha256": _sha256(row["sha256"], "table file SHA"),
                "hard_link_count": _positive_int(row["hard_link_count"], "table hard-link count"),
                "dataset_member": row["dataset_member"],
            }
        )
    regular_count = _positive_int(payload["regular_file_count"], "regular_file_count")
    parquet_count = _positive_int(payload["parquet_file_count"], "parquet_file_count")
    if regular_count != len(normalized_rows):
        raise _error("regular_file_count does not match table inventory")
    if parquet_count != sum(row["dataset_member"] for row in normalized_rows):
        raise _error("parquet_file_count does not match table inventory")
    inventory_sha = hashlib.sha256(
        prereg.canonical_json_bytes_v4_2(normalized_rows)
    ).hexdigest()
    if _sha256(payload["inventory_sha256"], "table inventory SHA") != inventory_sha:
        raise _error("table inventory SHA mismatch")
    symbols = _exact(
        payload["bound_symbol_inventory"],
        {"symbol_count", "symbols_newline_sha256", "noncanonical_symbol_count"},
        "bound symbol inventory",
    )
    if (
        type(symbols["noncanonical_symbol_count"]) is not int
        or symbols["noncanonical_symbol_count"] != 0
    ):
        raise _error("bound symbol inventory must contain no noncanonical symbols")
    _positive_int(symbols["symbol_count"], "bound symbol count")
    _sha256(symbols["symbols_newline_sha256"], "bound symbols SHA")
    return {
        "absolute_root": _absolute_path(payload["absolute_root"], "table absolute_root"),
        "regular_file_count": regular_count,
        "parquet_file_count": parquet_count,
        "inventory_sha256": inventory_sha,
        "parquet_inventory": normalized_rows,
        "bound_symbol_inventory": copy.deepcopy(symbols),
    }


def _validate_backend_binding(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "market",
        "snapshot_id",
        "cutoff_date",
        "latest_pointer",
        "snapshot_manifest",
        "components",
        "pit_generation",
        "table",
        "calendar",
        "eligibility_boundary",
        "readiness",
        "side_effects",
    }
    payload = _exact(value, fields, "v4.1 backend binding")
    if payload["schema_version"] != INPUT_BINDING_SCHEMA_VERSION:
        raise _error("v4.1 backend binding schema identity must be preserved")
    if payload["market"] != "CN":
        raise _error("backend binding market must be CN")
    snapshot_id = _snapshot_id(payload["snapshot_id"])
    cutoff = _iso_date(payload["cutoff_date"], "backend cutoff_date")
    snapshot_calendar_date = date.fromisoformat(
        snapshot_id[:4] + "-" + snapshot_id[4:6] + "-" + snapshot_id[6:8]
    )
    if snapshot_calendar_date < date.fromisoformat(cutoff):
        raise _error("snapshot_id calendar date must not precede cutoff")
    latest_pointer = _validate_binding_record(payload["latest_pointer"], "latest pointer binding")
    snapshot_manifest = _validate_binding_record(
        payload["snapshot_manifest"], "snapshot manifest binding"
    )
    components = _exact(
        payload["components"],
        {"absolute_path", "size_bytes", "sha256", "universe", "count", "newline_set_sha256"},
        "components binding",
    )
    component_record = _validate_binding_record(
        {key: components[key] for key in ("absolute_path", "size_bytes", "sha256")},
        "components binding",
    )
    if components["universe"] != "full_a":
        raise _error("components universe must be full_a")
    component_count = _positive_int(components["count"], "components count")
    full_a_sha = _sha256(components["newline_set_sha256"], "full_a scope SHA")
    pit = _exact(
        payload["pit_generation"],
        {
            "generation_id",
            "manifest",
            "membership",
            "row_count",
            "historical_alias_table_evidence",
        },
        "PIT generation binding",
    )
    if type(pit["generation_id"]) is not str or not pit["generation_id"]:
        raise _error("PIT generation_id must be non-empty")
    pit_manifest = _validate_binding_record(pit["manifest"], "PIT generation manifest")
    pit_membership = _validate_binding_record(pit["membership"], "PIT membership")
    _positive_int(pit["row_count"], "PIT row_count")
    if not isinstance(pit["historical_alias_table_evidence"], list):
        raise _error("historical alias table evidence must be a list")
    table = _validate_table_inventory(payload["table"])
    calendar = _exact(
        payload["calendar"],
        {
            "analysis_start",
            "cutoff_date",
            "open_session_count",
            "open_sessions",
            "semantic_sha256",
        },
        "calendar binding",
    )
    analysis_start = _iso_date(calendar["analysis_start"], "analysis_start")
    if calendar["cutoff_date"] != cutoff or analysis_start > cutoff:
        raise _error("calendar cutoff/analysis_start mismatch")
    sessions = calendar["open_sessions"]
    if not isinstance(sessions, list) or not sessions:
        raise _error("open_sessions must be a non-empty list")
    normalized_sessions = [_iso_date(item, "open session") for item in sessions]
    if normalized_sessions != sorted(set(normalized_sessions)):
        raise _error("open_sessions must be sorted and unique")
    if normalized_sessions[0] != analysis_start or normalized_sessions[-1] != cutoff:
        raise _error("calendar endpoints mismatch")
    if _positive_int(
        calendar["open_session_count"], "open_session_count"
    ) != len(normalized_sessions):
        raise _error("open_session_count mismatch")
    calendar_base = {
        key: calendar[key]
        for key in (
            "analysis_start",
            "cutoff_date",
            "open_session_count",
            "open_sessions",
        )
    }
    calendar_sha = hashlib.sha256(
        prereg.canonical_json_bytes_v4_2(calendar_base)
    ).hexdigest()
    if (
        _sha256(calendar["semantic_sha256"], "calendar semantic SHA")
        != calendar_sha
    ):
        raise _error("calendar semantic SHA mismatch")
    eligibility = _exact(
        payload["eligibility_boundary"],
        {"component_source", "pit_source", "bar_source", "serving_inventory"},
        "eligibility boundary",
    )
    if (
        eligibility["component_source"] != component_record["absolute_path"]
        or eligibility["pit_source"] != pit_membership["absolute_path"]
        or eligibility["bar_source"] != table["absolute_root"]
    ):
        raise _error("eligibility source paths do not match backend bindings")
    serving = _exact(
        eligibility["serving_inventory"],
        {"absolute_root", "symbol_count", "use", "was_scanned"},
        "serving historical semantics",
    )
    _absolute_path(serving["absolute_root"], "serving root")
    _positive_int(serving["symbol_count"], "serving symbol_count")
    if serving["use"] != SOURCE_USE_PROHIBITED or serving["was_scanned"] is not False:
        raise _error("serving inventory must remain unscanned and ineligible")
    pointer_path = Path(latest_pointer["absolute_path"])
    if tuple(pointer_path.parts[-4:]) != (
        "data",
        "parquet",
        "cn",
        "_latest.json",
    ):
        raise _error("latest pointer binding is not the myQuant CN control")
    cn_root = pointer_path.parent
    if Path(snapshot_manifest["absolute_path"]) != (
        cn_root / "_snapshots" / f"{snapshot_id}.json"
    ):
        raise _error("snapshot manifest is not the immutable snapshot manifest")
    if Path(table["absolute_root"]) != (
        cn_root / "_snapshots" / snapshot_id / "table" / "bars"
    ):
        raise _error("table root is not the immutable snapshot table")
    if Path(serving["absolute_root"]) != (
        cn_root / "_snapshots" / snapshot_id / "serving" / "bars"
    ):
        raise _error("serving root identity mismatch")
    expected_pit_parent = (
        cn_root
        / "reference"
        / "_generations"
        / str(pit["generation_id"])
    )
    if (
        Path(pit_manifest["absolute_path"]).parent != expected_pit_parent
        or Path(pit_membership["absolute_path"]).parent != expected_pit_parent
    ):
        raise _error("PIT immutable generation paths mismatch")
    if payload["readiness"] != "EXPLORATORY_INPUT_BOUND":
        raise _error("backend binding readiness mismatch")
    if payload["side_effects"] != {
        "registry": False,
        "wal": False,
        "budget": False,
        "apply": False,
        "broker": False,
        "order": False,
        "trade": False,
        "network": False,
    }:
        raise _error("backend binding side_effects mismatch")
    normalized = copy.deepcopy(payload)
    normalized["latest_pointer"] = latest_pointer
    normalized["snapshot_manifest"] = snapshot_manifest
    normalized["components"] = {
        **component_record,
        "universe": "full_a",
        "count": component_count,
        "newline_set_sha256": full_a_sha,
    }
    normalized["pit_generation"] = {
        **copy.deepcopy(pit),
        "manifest": pit_manifest,
        "membership": pit_membership,
    }
    normalized["table"] = table
    return normalized


def validate_strict_full_a_source_binding_v4_2(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate recorded source evidence without reading any live control."""

    fields = {
        "schema_version",
        "protocol_version",
        "market",
        "universe",
        "snapshot_id",
        "analysis_start",
        "cutoff",
        "expected_scope_count",
        "full_a_scope_sha256",
        "serving_inventory_count",
        "backend_binding_schema_version",
        "backend_binding_semantic_sha256",
        "backend_binding",
        "latest_pointer_raw_evidence",
        "components_raw_evidence",
        "component_symbols",
        "serving_historical_semantics",
        "immutable_reopen_descriptor",
        "historical_validation_reads_current_pointer",
        "historical_validation_reads_current_components",
        "artifact_semantic_sha256",
    }
    payload = _exact(value, fields, "strict full-A source binding")
    if payload["schema_version"] != STRICT_SOURCE_SCHEMA_VERSION:
        raise _error("strict source binding schema mismatch")
    if (
        payload["protocol_version"] != "v4"
        or payload["market"] != "CN"
        or payload["universe"] != "full_a"
    ):
        raise _error("strict source binding market/universe/protocol mismatch")
    backend = _validate_backend_binding(payload["backend_binding"])
    backend_sha = binding_semantic_sha256_v4_1(backend)
    if payload["backend_binding_schema_version"] != INPUT_BINDING_SCHEMA_VERSION:
        raise _error("backend schema identity was relabelled")
    if _sha256(payload["backend_binding_semantic_sha256"], "backend binding SHA") != backend_sha:
        raise _error("backend binding semantic SHA mismatch")
    snapshot_id = _snapshot_id(payload["snapshot_id"])
    cutoff = _iso_date(payload["cutoff"], "cutoff")
    analysis_start = _iso_date(payload["analysis_start"], "analysis_start")
    if (snapshot_id, cutoff, analysis_start) != (
        backend["snapshot_id"],
        backend["cutoff_date"],
        backend["calendar"]["analysis_start"],
    ):
        raise _error("strict source identity differs from backend binding")
    pointer_evidence, _pointer_raw, pointer_value = _validate_raw_evidence(
        payload["latest_pointer_raw_evidence"], "latest pointer raw evidence"
    )
    if (
        pointer_evidence["byte_sha256"] != backend["latest_pointer"]["sha256"]
        or pointer_evidence["size_bytes"] != backend["latest_pointer"]["size_bytes"]
    ):
        raise _error("latest pointer raw evidence differs from backend binding")
    if (
        pointer_value.get("snapshot_id") != snapshot_id
        or pointer_value.get("status") != "OK"
        or pointer_value.get("blockers") != []
        or pointer_value.get("latest_available_trade_date") != cutoff.replace("-", "")
        or pointer_value.get("latest_complete_trade_date") != cutoff.replace("-", "")
    ):
        raise _error("recorded latest pointer was not the exact healthy cutoff")
    pointer_path = Path(backend["latest_pointer"]["absolute_path"])
    project_root = pointer_path.parents[3]
    _declared_path_matches(
        pointer_value.get("manifest_path"),
        expected=backend["snapshot_manifest"]["absolute_path"],
        anchors=(project_root, pointer_path.parent),
        label="recorded pointer manifest",
    )
    _declared_path_matches(
        pointer_value.get("table_root"),
        expected=backend["table"]["absolute_root"],
        anchors=(project_root, pointer_path.parent),
        label="recorded pointer table",
    )
    _declared_path_matches(
        pointer_value.get("derived_serving_root"),
        expected=backend["eligibility_boundary"]["serving_inventory"][
            "absolute_root"
        ],
        anchors=(project_root, pointer_path.parent),
        label="recorded pointer serving",
    )
    coverage = pointer_value.get("coverage")
    if not isinstance(coverage, Mapping):
        raise _error("recorded pointer strict coverage is missing")
    compact_cutoff = cutoff.replace("-", "")
    if (
        coverage.get("coverage_schema_version") != "cn-full-a-coverage.v4"
        or coverage.get("complete") is not True
        or coverage.get("coverage_ratio") != 1.0
        or coverage.get("categories_checked") != ["full_a"]
        or coverage.get("expected_scope_count")
        != backend["components"]["count"]
        or coverage.get("coverage_complete_count")
        != backend["components"]["count"]
        or coverage.get("expected_scope_sha256")
        != backend["components"]["newline_set_sha256"]
        or coverage.get("coverage_trade_date") != compact_cutoff
        or coverage.get("latest_available_trade_date") != compact_cutoff
        or coverage.get("latest_complete_trade_date") != compact_cutoff
        or coverage.get("blocking_incomplete_count") != 0
        or coverage.get("classification_sets_disjoint") is not True
        or coverage.get("true_missing_symbols") != []
        or coverage.get("pit_generation_id")
        != backend["pit_generation"]["generation_id"]
        or coverage.get("pit_generation_manifest_sha256")
        != backend["pit_generation"]["manifest"]["sha256"]
        or coverage.get("pit_membership_sha256")
        != backend["pit_generation"]["membership"]["sha256"]
    ):
        raise _error("recorded pointer strict full-A coverage mismatch")
    _declared_path_matches(
        coverage.get("pit_generation_manifest_path"),
        expected=backend["pit_generation"]["manifest"]["absolute_path"],
        anchors=(project_root, pointer_path.parent),
        label="recorded coverage PIT manifest",
    )
    _declared_path_matches(
        coverage.get("pit_membership_path"),
        expected=backend["pit_generation"]["membership"]["absolute_path"],
        anchors=(project_root, pointer_path.parent),
        label="recorded coverage PIT membership",
    )
    components_evidence, _components_raw, components_value = _validate_raw_evidence(
        payload["components_raw_evidence"],
        "components raw evidence",
        extra_fields=frozenset(
            {"normalized_symbols", "symbol_count", "full_a_scope_sha256"}
        ),
    )
    if (
        components_evidence["byte_sha256"] != backend["components"]["sha256"]
        or components_evidence["size_bytes"] != backend["components"]["size_bytes"]
    ):
        raise _error("components raw evidence differs from backend binding")
    component_evidence_details = _exact(
        payload["components_raw_evidence"],
        {
            "encoding",
            "raw_base64",
            "size_bytes",
            "byte_sha256",
            "strict_json_semantic_sha256",
            "normalized_symbols",
            "symbol_count",
            "full_a_scope_sha256",
        },
        "components raw evidence",
    )
    symbols = payload["component_symbols"]
    if (
        not isinstance(symbols, list)
        or any(type(item) is not str or _CN_SYMBOL.fullmatch(item) is None for item in symbols)
        or symbols != sorted(set(symbols))
        or components_value.get("full_a") != symbols
    ):
        raise _error("component_symbols must equal normalized raw full_a")
    expected_count = _positive_int(payload["expected_scope_count"], "expected_scope_count")
    if len(symbols) != expected_count or backend["components"]["count"] != expected_count:
        raise _error("full-A component count mismatch")
    scope_sha = hashlib.sha256("\n".join(symbols).encode("utf-8")).hexdigest()
    if (
        _sha256(payload["full_a_scope_sha256"], "full_a_scope_sha256") != scope_sha
        or backend["components"]["newline_set_sha256"] != scope_sha
    ):
        raise _error("full-A scope SHA mismatch")
    if (
        component_evidence_details["normalized_symbols"] != symbols
        or component_evidence_details["symbol_count"] != expected_count
        or component_evidence_details["full_a_scope_sha256"] != scope_sha
    ):
        raise _error("components raw normalized scope evidence mismatch")
    serving_count = _positive_int(payload["serving_inventory_count"], "serving_inventory_count")
    serving = _exact(
        payload["serving_historical_semantics"],
        {"derived_serving_root", "symbol_count", "was_scanned"},
        "serving historical semantics",
    )
    backend_serving = backend["eligibility_boundary"]["serving_inventory"]
    if serving != {
        "derived_serving_root": backend_serving["absolute_root"],
        "symbol_count": backend_serving["symbol_count"],
        "was_scanned": False,
    } or serving_count != backend_serving["symbol_count"]:
        raise _error("serving historical semantics mismatch")
    reopen = _exact(
        payload["immutable_reopen_descriptor"],
        {"snapshot_manifest", "pit_generation_manifest", "pit_membership", "table_inventory"},
        "immutable reopen descriptor",
    )
    expected_reopen = {
        "snapshot_manifest": backend["snapshot_manifest"],
        "pit_generation_manifest": backend["pit_generation"]["manifest"],
        "pit_membership": backend["pit_generation"]["membership"],
        "table_inventory": backend["table"],
    }
    if reopen != expected_reopen:
        raise _error("immutable reopen descriptor differs from backend binding")
    if (
        payload["historical_validation_reads_current_pointer"] is not False
        or payload["historical_validation_reads_current_components"] is not False
    ):
        raise _error("historical validation must not read mutable current controls")
    _validate_self(payload, "strict full-A source binding")
    return copy.deepcopy(payload)


validate_historical_strict_full_a_source_binding_v4_2 = (
    validate_strict_full_a_source_binding_v4_2
)


def build_strict_full_a_source_binding_v4_2(
    *,
    bound_inputs: BoundCutoffInputsV4_1,
    latest_pointer_raw: bytes,
    components_raw: bytes,
) -> dict[str, Any]:
    """Build one historical-self-contained wrapper around the v4.1 binder."""

    if not isinstance(bound_inputs, BoundCutoffInputsV4_1):
        raise _error("bound_inputs must be BoundCutoffInputsV4_1")
    backend = _validate_backend_binding(bound_inputs.binding)
    symbols = list(bound_inputs.component_symbols)
    if (
        len(symbols) != backend["components"]["count"]
        or symbols != sorted(set(symbols))
        or any(_CN_SYMBOL.fullmatch(item) is None for item in symbols)
    ):
        raise _error("bound input symbols must be sorted and unique")
    if tuple(bound_inputs.calendar_sessions) != tuple(
        backend["calendar"]["open_sessions"]
    ):
        raise _error("bound input calendar differs from backend binding")
    if len(bound_inputs.pit_records) != backend["pit_generation"]["row_count"]:
        raise _error("bound PIT records differ from backend binding")
    table_symbols = [item[0] for item in bound_inputs.bound_table_symbol_row_counts]
    if (
        table_symbols != sorted(set(table_symbols))
        or len(table_symbols)
        != backend["table"]["bound_symbol_inventory"]["symbol_count"]
        or hashlib.sha256("\n".join(table_symbols).encode("ascii")).hexdigest()
        != backend["table"]["bound_symbol_inventory"][
            "symbols_newline_sha256"
        ]
    ):
        raise _error("bound table symbol inventory differs from backend binding")
    serving = backend["eligibility_boundary"]["serving_inventory"]
    components_evidence = _raw_evidence(components_raw)
    components_evidence.update(
        {
            "normalized_symbols": symbols,
            "symbol_count": len(symbols),
            "full_a_scope_sha256": backend["components"][
                "newline_set_sha256"
            ],
        }
    )
    return validate_strict_full_a_source_binding_v4_2(
        _seal(
            {
                "schema_version": STRICT_SOURCE_SCHEMA_VERSION,
                "protocol_version": "v4",
                "market": "CN",
                "universe": "full_a",
                "snapshot_id": backend["snapshot_id"],
                "analysis_start": backend["calendar"]["analysis_start"],
                "cutoff": backend["cutoff_date"],
                "expected_scope_count": backend["components"]["count"],
                "full_a_scope_sha256": backend["components"]["newline_set_sha256"],
                "serving_inventory_count": serving["symbol_count"],
                "backend_binding_schema_version": INPUT_BINDING_SCHEMA_VERSION,
                "backend_binding_semantic_sha256": binding_semantic_sha256_v4_1(backend),
                "backend_binding": backend,
                "latest_pointer_raw_evidence": _raw_evidence(latest_pointer_raw),
                "components_raw_evidence": components_evidence,
                "component_symbols": symbols,
                "serving_historical_semantics": {
                    "derived_serving_root": serving["absolute_root"],
                    "symbol_count": serving["symbol_count"],
                    "was_scanned": False,
                },
                "immutable_reopen_descriptor": {
                    "snapshot_manifest": backend["snapshot_manifest"],
                    "pit_generation_manifest": backend["pit_generation"]["manifest"],
                    "pit_membership": backend["pit_generation"]["membership"],
                    "table_inventory": backend["table"],
                },
                "historical_validation_reads_current_pointer": False,
                "historical_validation_reads_current_components": False,
            }
        )
    )


def _stable_source_file(
    path: Path,
    label: str,
    *,
    require_single_link: bool = True,
) -> tuple[bytes, tuple[int, ...]]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise _error(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise _error(f"{label} must be a regular non-symlink file")
    if before.st_uid != os.getuid() or (
        require_single_link and before.st_nlink != 1
    ) or before.st_nlink < 1:
        raise _error(f"{label} owner/hard-link contract failed")
    signature = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        opened_signature = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened_signature != signature:
            raise _error(f"{label} changed while opening")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        after_signature = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if after_signature != signature:
            raise _error(f"{label} changed while reading")
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            raise _error(f"{label} length mismatch")
        return raw, signature
    finally:
        os.close(descriptor)


def _project_root_from_recorded_pointer(source: Mapping[str, Any]) -> Path:
    pointer = Path(
        source["backend_binding"]["latest_pointer"]["absolute_path"]
    )
    if tuple(pointer.parts[-4:]) != ("data", "parquet", "cn", "_latest.json"):
        raise _error("recorded latest pointer path is not the myQuant CN control")
    return pointer.parents[3]


def _assert_owned_nofollow_chain(
    target: Path,
    *,
    boundary: Path,
    include_target: bool,
    label: str,
) -> None:
    try:
        relative = target.relative_to(boundary)
    except ValueError as exc:
        raise _error(f"{label} escapes the recorded project root") from exc
    current = boundary
    parts = relative.parts if include_target else relative.parts[:-1]
    for part in ("", *parts):
        if part:
            current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise _error(f"{label} directory chain is missing: {current}") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
        ):
            raise _error(f"{label} directory chain is unsafe: {current}")


def _reopen_recorded_file(
    record: Mapping[str, Any],
    *,
    boundary: Path,
    label: str,
) -> dict[str, Any]:
    normalized = _validate_binding_record(record, label)
    path = Path(normalized["absolute_path"])
    _assert_owned_nofollow_chain(
        path,
        boundary=boundary,
        include_target=False,
        label=label,
    )
    first_raw, first_signature = _stable_source_file(path, label)
    second_raw, second_signature = _stable_source_file(path, label)
    if first_raw != second_raw or first_signature != second_signature:
        raise _error(f"{label} changed across stable reopen passes")
    if (
        len(first_raw) != normalized["size_bytes"]
        or hashlib.sha256(first_raw).hexdigest() != normalized["sha256"]
    ):
        raise _error(f"{label} recorded size/SHA mismatch")
    return {
        "absolute_path": str(path),
        "size_bytes": len(first_raw),
        "byte_sha256": hashlib.sha256(first_raw).hexdigest(),
        "stable_reopen_passes": 2,
    }


def _table_inventory_pass(
    root: Path,
) -> tuple[list[dict[str, Any]], tuple[tuple[str, tuple[int, ...]], ...]]:
    inventory: list[dict[str, Any]] = []
    identities: list[tuple[str, tuple[int, ...]]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        relative_directory = directory.relative_to(root)
        directory_name = (
            "." if not relative_directory.parts else relative_directory.as_posix()
        )
        metadata = os.lstat(directory)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
        ):
            raise _error(f"historical table directory is unsafe: {directory}")
        identities.append(
            (
                f"dir:{directory_name}",
                (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_uid,
                    metadata.st_gid,
                    metadata.st_nlink,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                ),
            )
        )
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise _error(f"historical table directory is unreadable: {directory}") from exc
        for entry in entries:
            path = directory / entry.name
            relative = path.relative_to(root)
            item_metadata = os.lstat(path)
            if stat.S_ISLNK(item_metadata.st_mode) or item_metadata.st_uid != os.getuid():
                raise _error(f"historical table entry is unsafe: {path}")
            if stat.S_ISDIR(item_metadata.st_mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(item_metadata.st_mode):
                raise _error(f"historical table entry must be regular: {path}")
            raw, signature = _stable_source_file(
                path,
                f"historical table file {relative.as_posix()}",
                require_single_link=False,
            )
            identities.append((f"file:{relative.as_posix()}", signature))
            inventory.append(
                {
                    "relative_path": relative.as_posix(),
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "hard_link_count": int(item_metadata.st_nlink),
                    "dataset_member": bool(
                        path.suffix == ".parquet"
                        and all(
                            not part.startswith((".", "_"))
                            for part in relative.parts
                        )
                    ),
                }
            )
    inventory.sort(key=lambda row: row["relative_path"])
    identities.sort(key=lambda row: row[0])
    return inventory, tuple(identities)


def revalidate_recorded_immutable_source_v4_2(
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen only the recorded immutable source files and table inventory.

    This function never reads the current pointer, components, or serving tree.
    Their recorded paths may exist or drift without affecting historical
    validation.
    """

    source = validate_historical_strict_full_a_source_binding_v4_2(
        source_binding
    )
    boundary = _project_root_from_recorded_pointer(source)
    _assert_owned_nofollow_chain(
        boundary,
        boundary=boundary,
        include_target=True,
        label="recorded project root",
    )
    descriptor = source["immutable_reopen_descriptor"]
    manifest = _reopen_recorded_file(
        descriptor["snapshot_manifest"],
        boundary=boundary,
        label="recorded snapshot manifest",
    )
    pit_manifest = _reopen_recorded_file(
        descriptor["pit_generation_manifest"],
        boundary=boundary,
        label="recorded PIT generation manifest",
    )
    pit_membership = _reopen_recorded_file(
        descriptor["pit_membership"],
        boundary=boundary,
        label="recorded PIT membership",
    )
    table = descriptor["table_inventory"]
    table_root = Path(table["absolute_root"])
    _assert_owned_nofollow_chain(
        table_root,
        boundary=boundary,
        include_target=True,
        label="recorded immutable table",
    )
    first_inventory, first_identities = _table_inventory_pass(table_root)
    second_inventory, second_identities = _table_inventory_pass(table_root)
    if (
        first_inventory != second_inventory
        or first_identities != second_identities
    ):
        raise _error("recorded immutable table changed across stable passes")
    if first_inventory != table["parquet_inventory"]:
        raise _error("recorded immutable table inventory mismatch")
    inventory_sha = hashlib.sha256(
        prereg.canonical_json_bytes_v4_2(first_inventory)
    ).hexdigest()
    if (
        inventory_sha != table["inventory_sha256"]
        or len(first_inventory) != table["regular_file_count"]
        or sum(row["dataset_member"] for row in first_inventory)
        != table["parquet_file_count"]
    ):
        raise _error("recorded immutable table inventory summary mismatch")
    return {
        "accepted": True,
        "validation_scope": "RECORDED_IMMUTABLE_REOPEN_ONLY",
        "strict_source_binding_semantic_sha256": source[
            "artifact_semantic_sha256"
        ],
        "snapshot_manifest": manifest,
        "pit_generation_manifest": pit_manifest,
        "pit_membership": pit_membership,
        "table_inventory": {
            "absolute_root": str(table_root),
            "regular_file_count": len(first_inventory),
            "parquet_file_count": sum(
                row["dataset_member"] for row in first_inventory
            ),
            "inventory_sha256": inventory_sha,
            "stable_reopen_passes": 2,
        },
        "current_pointer_read": False,
        "current_components_read": False,
        "serving_tree_read": False,
    }


def validate_code_binding_set_v4_2(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "path_count",
            "ordered_bindings",
            "artifact_semantic_sha256",
        },
        "code binding set",
    )
    if (
        payload["schema_version"] != CODE_BINDING_SET_SCHEMA_VERSION
        or payload["protocol_version"] != "v4"
    ):
        raise _error("code binding set schema/protocol mismatch")
    rows = payload["ordered_bindings"]
    if not isinstance(rows, list) or len(rows) != len(CODE_BINDING_PATHS_V4_2):
        raise _error("code binding set must contain the exact fixed inventory")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(rows):
        row = _exact(
            item,
            {"order", "relative_path", "size_bytes", "byte_sha256"},
            f"code binding[{index}]",
        )
        expected_path = CODE_BINDING_PATHS_V4_2[index]
        if row["order"] != index + 1 or row["relative_path"] != expected_path:
            raise _error("code binding set path/order mismatch")
        normalized.append(
            {
                "order": index + 1,
                "relative_path": expected_path,
                "size_bytes": _positive_int(row["size_bytes"], "code size_bytes"),
                "byte_sha256": _sha256(row["byte_sha256"], "code byte SHA"),
            }
        )
    if payload["path_count"] != len(normalized):
        raise _error("code path_count mismatch")
    _validate_self(payload, "code binding set")
    return copy.deepcopy(payload)


def _repository_root(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if type(raw) is not str:
        raise _error("repository_root must be an absolute path")
    return Path(_absolute_path(raw, "repository_root"))


def build_code_binding_set_v4_2(*, repository_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = _repository_root(repository_root)
    _assert_owned_nofollow_chain(
        root,
        boundary=root,
        include_target=True,
        label="code repository root",
    )
    rows: list[dict[str, Any]] = []
    for index, relative in enumerate(CODE_BINDING_PATHS_V4_2, start=1):
        path = root / relative
        _assert_owned_nofollow_chain(
            path,
            boundary=root,
            include_target=False,
            label=f"code binding {relative}",
        )
        raw, _signature = _stable_source_file(path, f"code binding {relative}")
        rows.append(
            {
                "order": index,
                "relative_path": relative,
                "size_bytes": len(raw),
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return validate_code_binding_set_v4_2(
        _seal(
            {
                "schema_version": CODE_BINDING_SET_SCHEMA_VERSION,
                "protocol_version": "v4",
                "path_count": len(rows),
                "ordered_bindings": rows,
            }
        )
    )


def revalidate_code_binding_set_v4_2(
    *, repository_root: str | os.PathLike[str], value: Mapping[str, Any]
) -> dict[str, Any]:
    expected = validate_code_binding_set_v4_2(value)
    live = build_code_binding_set_v4_2(repository_root=repository_root)
    if prereg.canonical_file_bytes_v4_2(live) != prereg.canonical_file_bytes_v4_2(expected):
        raise _error("code binding inputs drifted")
    return expected


def _validate_leaf_sources(
    *,
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    candidate_selection_spec: Mapping[str, Any],
    strict_full_a_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    aquant = prereg.validate_aquant_receipt_v4_2(aquant_receipt)
    myquant = prereg.validate_myquant_receipt_v4_2(myquant_receipt)
    operators = prereg.validate_operator_semantics_v4_2(operator_semantics)
    comparison = prereg.validate_comparison_catalog_receipt_v4_2(comparison_catalog_receipt)
    selection = prereg.validate_selection_spec_v4_2(
        candidate_selection_spec,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
    )
    source = validate_strict_full_a_source_binding_v4_2(strict_full_a_source_binding)
    code = validate_code_binding_set_v4_2(code_binding_set)
    return aquant, myquant, operators, comparison, selection, source, code


def deterministic_cycle_id_v4_2(*, cutoff: str, snapshot_id: str) -> str:
    normalized_cutoff = _iso_date(cutoff, "cutoff")
    normalized_snapshot = _snapshot_id(snapshot_id)
    return f"cn_full_a_v4_2_{normalized_cutoff.replace('-', '')}_{normalized_snapshot}"


def validate_cycle_root_v4_2(
    value: Mapping[str, Any],
    *,
    aquant_receipt: Mapping[str, Any] | None = None,
    myquant_receipt: Mapping[str, Any] | None = None,
    operator_semantics: Mapping[str, Any] | None = None,
    comparison_catalog_receipt: Mapping[str, Any] | None = None,
    candidate_selection_spec: Mapping[str, Any] | None = None,
    strict_full_a_source_binding: Mapping[str, Any] | None = None,
    code_binding_set: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "candidate_preregistration_schema_version",
            "market",
            "universe",
            "cutoff",
            "snapshot_id",
            "cycle_id",
            "ordered_identity_bindings",
            "cycle_root_sha256",
            "artifact_semantic_sha256",
        },
        "v4.2 cycle root",
    )
    if (
        payload["schema_version"] != CYCLE_ROOT_SCHEMA_VERSION
        or payload["protocol_version"] != "v4"
        or payload["candidate_preregistration_schema_version"] != prereg.SCHEMA_VERSION
        or payload["market"] != "CN"
        or payload["universe"] != "full_a"
    ):
        raise _error("cycle root fixed identity mismatch")
    cutoff = _iso_date(payload["cutoff"], "cycle root cutoff")
    snapshot_id = _snapshot_id(payload["snapshot_id"])
    cycle_id = deterministic_cycle_id_v4_2(cutoff=cutoff, snapshot_id=snapshot_id)
    if payload["cycle_id"] != cycle_id:
        raise _error("cycle_id is not the deterministic v4.2 identity")
    names = (
        "aquant_receipt",
        "myquant_receipt",
        "operator_semantics",
        "comparison_catalog_receipt",
        "candidate_selection_spec",
        "strict_full_a_source_binding",
        "code_binding_set",
    )
    identities = payload["ordered_identity_bindings"]
    if not isinstance(identities, list) or len(identities) != len(names):
        raise _error("cycle root identity inventory mismatch")
    normalized_identities: list[dict[str, Any]] = []
    for index, (item, name) in enumerate(zip(identities, names, strict=True), start=1):
        row = _exact(
            item,
            {"order", "name", "artifact_semantic_sha256"},
            f"cycle root identity[{index}]",
        )
        if row["order"] != index or row["name"] != name:
            raise _error("cycle root identity order/name mismatch")
        normalized_identities.append(
            {
                "order": index,
                "name": name,
                "artifact_semantic_sha256": _sha256(
                    row["artifact_semantic_sha256"], "cycle identity SHA"
                ),
            }
        )
    base = {
        "schema_version": CYCLE_ROOT_SCHEMA_VERSION,
        "protocol_version": "v4",
        "candidate_preregistration_schema_version": prereg.SCHEMA_VERSION,
        "market": "CN",
        "universe": "full_a",
        "cutoff": cutoff,
        "snapshot_id": snapshot_id,
        "cycle_id": cycle_id,
        "ordered_identity_bindings": normalized_identities,
    }
    expected_root = prereg.semantic_sha256_v4_2(base)
    if _sha256(payload["cycle_root_sha256"], "cycle_root_sha256") != expected_root:
        raise _error("cycle_root_sha256 mismatch")
    if all(
        item is not None
        for item in (
            aquant_receipt,
            myquant_receipt,
            operator_semantics,
            comparison_catalog_receipt,
            candidate_selection_spec,
            strict_full_a_source_binding,
            code_binding_set,
        )
    ):
        leaves = _validate_leaf_sources(
            aquant_receipt=aquant_receipt,  # type: ignore[arg-type]
            myquant_receipt=myquant_receipt,  # type: ignore[arg-type]
            operator_semantics=operator_semantics,  # type: ignore[arg-type]
            comparison_catalog_receipt=comparison_catalog_receipt,  # type: ignore[arg-type]
            candidate_selection_spec=candidate_selection_spec,  # type: ignore[arg-type]
            strict_full_a_source_binding=strict_full_a_source_binding,  # type: ignore[arg-type]
            code_binding_set=code_binding_set,  # type: ignore[arg-type]
        )
        expected_identities = [
            {
                "order": index,
                "name": name,
                "artifact_semantic_sha256": leaf["artifact_semantic_sha256"],
            }
            for index, (name, leaf) in enumerate(zip(names, leaves, strict=True), start=1)
        ]
        if normalized_identities != expected_identities:
            raise _error("cycle root cross-artifact identity mismatch")
        source = leaves[5]
        if cutoff != source["cutoff"] or snapshot_id != source["snapshot_id"]:
            raise _error("cycle root source cutoff/snapshot mismatch")
    _validate_self(payload, "cycle root")
    return copy.deepcopy(payload)


def build_cycle_root_v4_2(
    *,
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    candidate_selection_spec: Mapping[str, Any],
    strict_full_a_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
) -> dict[str, Any]:
    leaves = _validate_leaf_sources(
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
        candidate_selection_spec=candidate_selection_spec,
        strict_full_a_source_binding=strict_full_a_source_binding,
        code_binding_set=code_binding_set,
    )
    source = leaves[5]
    names = (
        "aquant_receipt",
        "myquant_receipt",
        "operator_semantics",
        "comparison_catalog_receipt",
        "candidate_selection_spec",
        "strict_full_a_source_binding",
        "code_binding_set",
    )
    cycle_id = deterministic_cycle_id_v4_2(
        cutoff=source["cutoff"], snapshot_id=source["snapshot_id"]
    )
    identities = [
        {
            "order": index,
            "name": name,
            "artifact_semantic_sha256": leaf["artifact_semantic_sha256"],
        }
        for index, (name, leaf) in enumerate(zip(names, leaves, strict=True), start=1)
    ]
    base = {
        "schema_version": CYCLE_ROOT_SCHEMA_VERSION,
        "protocol_version": "v4",
        "candidate_preregistration_schema_version": prereg.SCHEMA_VERSION,
        "market": "CN",
        "universe": "full_a",
        "cutoff": source["cutoff"],
        "snapshot_id": source["snapshot_id"],
        "cycle_id": cycle_id,
        "ordered_identity_bindings": identities,
    }
    return validate_cycle_root_v4_2(
        _seal({**base, "cycle_root_sha256": prereg.semantic_sha256_v4_2(base)}),
        aquant_receipt=leaves[0],
        myquant_receipt=leaves[1],
        operator_semantics=leaves[2],
        comparison_catalog_receipt=leaves[3],
        candidate_selection_spec=leaves[4],
        strict_full_a_source_binding=leaves[5],
        code_binding_set=leaves[6],
    )


def _definition_builder() -> Any:
    return getattr(prereg, "build_definition_identity_collision_audit_v4_2")


def _definition_validator() -> Any:
    return getattr(prereg, "validate_definition_identity_collision_audit_v4_2")


def build_candidate_preregistration_bundle_artifacts_v4_2(
    *,
    aquant_receipt: Mapping[str, Any],
    myquant_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    candidate_selection_spec: Mapping[str, Any],
    strict_full_a_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Rebuild the exact deterministic fourteen-artifact graph."""

    aquant, myquant, operators, comparison, selection, source, code = _validate_leaf_sources(
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
        candidate_selection_spec=candidate_selection_spec,
        strict_full_a_source_binding=strict_full_a_source_binding,
        code_binding_set=code_binding_set,
    )
    cycle_root = build_cycle_root_v4_2(
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
        candidate_selection_spec=selection,
        strict_full_a_source_binding=source,
        code_binding_set=code,
    )
    envelope = prereg.build_future_source_envelope_v4_2(
        cycle_id=cycle_root["cycle_id"],
        analysis_start=source["analysis_start"],
        cutoff=source["cutoff"],
        snapshot_id=source["snapshot_id"],
        snapshot_date=(
            source["snapshot_id"][:4]
            + "-"
            + source["snapshot_id"][4:6]
            + "-"
            + source["snapshot_id"][6:8]
        ),
        full_a_scope_count=source["expected_scope_count"],
        full_a_scope_sha256=source["full_a_scope_sha256"],
        serving_inventory_count=source["serving_inventory_count"],
        strict_source_binding_semantic_sha256=source["artifact_semantic_sha256"],
        selection_spec=selection,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
        code_binding_set_semantic_sha256=code["artifact_semantic_sha256"],
    )
    collision = _definition_builder()(
        selection_spec=selection,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
    )
    predecessor = build_genesis_cycle_state_v4_1(
        cycle_id=cycle_root["cycle_id"],
        cycle_root_sha256=cycle_root["cycle_root_sha256"],
        source_chain_node_sha256=envelope["artifact_semantic_sha256"],
    )
    predecessor_byte = cycle_state_byte_sha256_v4_1(predecessor)
    orchestration = prereg.build_preregistration_discovery_cycle_v4_2(
        predecessor_state=predecessor,
        predecessor_byte_sha256=predecessor_byte,
        expected_predecessor_byte_sha256=predecessor_byte,
        expected_predecessor_semantic_sha256=predecessor["state_semantic_sha256"],
        future_source_envelope=envelope,
        selection_spec=selection,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
        definition_identity_collision_audit=collision,
        code_binding_set_semantic_sha256=code["artifact_semantic_sha256"],
        strict_source_binding_semantic_sha256=source[
            "artifact_semantic_sha256"
        ],
        full_a_scope_sha256=source["full_a_scope_sha256"],
        full_a_scope_count=source["expected_scope_count"],
        serving_inventory_count=source["serving_inventory_count"],
    )
    orchestration = prereg.validate_preregistration_discovery_cycle_v4_2(
        orchestration,
        predecessor_state=predecessor,
        predecessor_byte_sha256=predecessor_byte,
        expected_predecessor_byte_sha256=predecessor_byte,
        expected_predecessor_semantic_sha256=predecessor[
            "state_semantic_sha256"
        ],
        future_source_envelope=envelope,
        selection_spec=selection,
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
        definition_identity_collision_audit=collision,
        code_binding_set_semantic_sha256=code["artifact_semantic_sha256"],
        strict_source_binding_semantic_sha256=source[
            "artifact_semantic_sha256"
        ],
        full_a_scope_sha256=source["full_a_scope_sha256"],
        full_a_scope_count=source["expected_scope_count"],
        serving_inventory_count=source["serving_inventory_count"],
    )
    return {
        AQUANT_IDEA_SOURCE_RECEIPT_FILENAME_V4_2: aquant,
        MYQUANT_ALPHA158_SOURCE_RECEIPT_FILENAME_V4_2: myquant,
        OPERATOR_SEMANTICS_FILENAME_V4_2: operators,
        COMPARISON_CATALOG_RECEIPT_FILENAME_V4_2: comparison,
        CANDIDATE_SELECTION_SPEC_FILENAME_V4_2: selection,
        STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2: source,
        CODE_BINDING_SET_FILENAME_V4_2: code,
        FUTURE_SOURCE_ENVELOPE_FILENAME_V4_2: envelope,
        CYCLE_ROOT_FILENAME_V4_2: cycle_root,
        DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_2: collision,
        PRECOMMITTED_STATE_FILENAME_V4_2: predecessor,
        DISCOVERY_SOURCE_NODE_FILENAME_V4_2: copy.deepcopy(orchestration["source_node"]),
        DISCOVERY_STATE_FILENAME_V4_2: copy.deepcopy(orchestration["discovery_state"]),
        PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_2: orchestration,
    }


def _canonical_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return prereg.canonical_file_bytes_v4_2(left) == prereg.canonical_file_bytes_v4_2(right)


def _validate_report(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "filename",
            "run_id",
            "cycle_id",
            "cycle_root_sha256",
            "publication_evidence_scope",
            "intended_destination",
            "deterministic_identity",
            "required_commit_primitive",
            "commit_success_claimed",
            "no_clobber_success_claimed",
            "fsync_success_claimed",
            "durability_success_claimed",
            "state_contract",
            "artifact_bindings",
            "side_effects",
            "artifact_semantic_sha256",
        },
        "candidate preregistration readback report",
    )
    if (
        payload["schema_version"] != READBACK_REPORT_SCHEMA_VERSION
        or payload["protocol_version"] != "v4"
        or payload["filename"] != READBACK_REPORT_FILENAME_V4_2
    ):
        raise _error("readback report schema/protocol/filename mismatch")
    if payload["run_id"] != payload["cycle_id"]:
        raise _error("readback run_id must equal deterministic cycle_id")
    if payload["publication_evidence_scope"] != "PRECOMMIT_INTENT_ONLY":
        raise _error("bundle report may claim precommit intent only")
    if payload["intended_destination"] != {
        "root_suffix": list(ROOT_SUFFIX_V4_2),
        "directory_name": payload["cycle_id"],
    }:
        raise _error("intended destination mismatch")
    if payload["deterministic_identity"] != {
        "market": "CN",
        "universe": "full_a",
        "protocol_version": "v4",
        "cycle_id": payload["cycle_id"],
    }:
        raise _error("deterministic identity mismatch")
    if payload["required_commit_primitive"] != "renameatx_np(RENAME_EXCL)":
        raise _error("exclusive Darwin commit primitive is required")
    for field in (
        "commit_success_claimed",
        "no_clobber_success_claimed",
        "fsync_success_claimed",
        "durability_success_claimed",
    ):
        if payload[field] is not False:
            raise _error(f"{field} must remain false inside staged bytes")
    if payload["state_contract"] != {
        "precommitted_persisted": True,
        "precommitted_role": "INTRA_BUNDLE_LINEAGE_ONLY",
        "discovery_persisted": True,
        "sole_final_current_state": DISCOVERY,
        "external_pointer_mutation": False,
    }:
        raise _error("readback state contract mismatch")
    bindings = payload["artifact_bindings"]
    binding_filenames = [
        row.get("filename") for row in bindings if isinstance(row, Mapping)
    ] if isinstance(bindings, list) else []
    if binding_filenames != list(INPUT_FILENAMES_V4_2):
        raise _error("readback artifact binding inventory mismatch")
    for index, item in enumerate(bindings):
        row = _exact(
            item,
            {"filename", "byte_sha256", "size_bytes", "mode", "uid", "nlink"},
            f"readback binding[{index}]",
        )
        _sha256(row["byte_sha256"], "readback byte SHA")
        _positive_int(row["size_bytes"], "readback size")
        if (
            row["mode"] != 0o600
            or type(row["uid"]) is not int
            or row["uid"] < 0
            or row["nlink"] != 1
        ):
            raise _error("readback owner/private binding mismatch")
    if payload["side_effects"] != prereg.SIDE_EFFECT_FLAGS:
        raise _error("readback side effects must remain exact false")
    _sha256(payload["cycle_root_sha256"], "readback cycle root")
    _validate_self(payload, "readback report")
    return copy.deepcopy(payload)


def _validate_artifact(filename: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if filename == AQUANT_IDEA_SOURCE_RECEIPT_FILENAME_V4_2:
        return prereg.validate_aquant_receipt_v4_2(value)
    if filename == MYQUANT_ALPHA158_SOURCE_RECEIPT_FILENAME_V4_2:
        return prereg.validate_myquant_receipt_v4_2(value)
    if filename == OPERATOR_SEMANTICS_FILENAME_V4_2:
        return prereg.validate_operator_semantics_v4_2(value)
    if filename == COMPARISON_CATALOG_RECEIPT_FILENAME_V4_2:
        return prereg.validate_comparison_catalog_receipt_v4_2(value)
    if filename == STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2:
        return validate_strict_full_a_source_binding_v4_2(value)
    if filename == CODE_BINDING_SET_FILENAME_V4_2:
        return validate_code_binding_set_v4_2(value)
    if filename == CYCLE_ROOT_FILENAME_V4_2:
        return validate_cycle_root_v4_2(value)
    if filename in (PRECOMMITTED_STATE_FILENAME_V4_2, DISCOVERY_STATE_FILENAME_V4_2):
        return validate_cycle_state_v4_1(value)
    if filename == READBACK_REPORT_FILENAME_V4_2:
        return _validate_report(value)
    if filename in INPUT_FILENAMES_V4_2:
        return _validate_self(value, filename)
    raise _error(f"unknown v4.2 bundle artifact: {filename}")


def _build_readback_report(
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(artifacts) != INPUT_FILENAMES_V4_2:
        raise _error("readback builder input inventory/order mismatch")
    cycle_root = validate_cycle_root_v4_2(artifacts[CYCLE_ROOT_FILENAME_V4_2])
    if run_id != cycle_root["cycle_id"]:
        raise _error("publication directory must equal deterministic cycle_id")
    return _validate_report(
        _seal(
            {
                "schema_version": READBACK_REPORT_SCHEMA_VERSION,
                "protocol_version": "v4",
                "filename": READBACK_REPORT_FILENAME_V4_2,
                "run_id": run_id,
                "cycle_id": cycle_root["cycle_id"],
                "cycle_root_sha256": cycle_root["cycle_root_sha256"],
                "publication_evidence_scope": "PRECOMMIT_INTENT_ONLY",
                "intended_destination": {
                    "root_suffix": list(ROOT_SUFFIX_V4_2),
                    "directory_name": cycle_root["cycle_id"],
                },
                "deterministic_identity": {
                    "market": "CN",
                    "universe": "full_a",
                    "protocol_version": "v4",
                    "cycle_id": cycle_root["cycle_id"],
                },
                "required_commit_primitive": "renameatx_np(RENAME_EXCL)",
                "commit_success_claimed": False,
                "no_clobber_success_claimed": False,
                "fsync_success_claimed": False,
                "durability_success_claimed": False,
                "state_contract": {
                    "precommitted_persisted": True,
                    "precommitted_role": "INTRA_BUNDLE_LINEAGE_ONLY",
                    "discovery_persisted": True,
                    "sole_final_current_state": DISCOVERY,
                    "external_pointer_mutation": False,
                },
                "artifact_bindings": [copy.deepcopy(dict(row)) for row in artifact_bindings],
                "side_effects": copy.deepcopy(prereg.SIDE_EFFECT_FLAGS),
            }
        )
    )


def validate_candidate_preregistration_bundle_artifacts_v4_2(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate the full graph by deterministic reconstruction."""

    expected_names = {*INPUT_FILENAMES_V4_2, READBACK_REPORT_FILENAME_V4_2}
    if set(values) != expected_names:
        raise _error("complete bundle inventory mismatch")
    normalized_inputs = validate_candidate_preregistration_bundle_inputs_v4_2(
        {filename: values[filename] for filename in INPUT_FILENAMES_V4_2}
    )
    normalized = {
        **normalized_inputs,
        READBACK_REPORT_FILENAME_V4_2: _validate_report(
            values[READBACK_REPORT_FILENAME_V4_2]
        ),
    }
    report = normalized[READBACK_REPORT_FILENAME_V4_2]
    normalized_cycle_root = normalized[CYCLE_ROOT_FILENAME_V4_2]
    if (
        report["cycle_id"] != normalized_cycle_root["cycle_id"]
        or report["cycle_root_sha256"]
        != normalized_cycle_root["cycle_root_sha256"]
    ):
        raise _error("readback report cycle identity mismatch")
    binding_by_name = {row["filename"]: row for row in report["artifact_bindings"]}
    for filename in INPUT_FILENAMES_V4_2:
        raw = prereg.canonical_file_bytes_v4_2(normalized[filename])
        if (
            binding_by_name[filename]["byte_sha256"]
            != hashlib.sha256(raw).hexdigest()
            or binding_by_name[filename]["size_bytes"] != len(raw)
        ):
            raise _error(f"readback byte binding mismatch: {filename}")
    return normalized


def validate_candidate_preregistration_bundle_inputs_v4_2(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate the exact fourteen inputs before a report can be generated."""

    if set(values) != set(INPUT_FILENAMES_V4_2):
        raise _error("bundle input inventory mismatch")
    normalized = {
        filename: _validate_artifact(filename, values[filename])
        for filename in INPUT_FILENAMES_V4_2
    }
    rebuilt = build_candidate_preregistration_bundle_artifacts_v4_2(
        aquant_receipt=normalized[AQUANT_IDEA_SOURCE_RECEIPT_FILENAME_V4_2],
        myquant_receipt=normalized[MYQUANT_ALPHA158_SOURCE_RECEIPT_FILENAME_V4_2],
        operator_semantics=normalized[OPERATOR_SEMANTICS_FILENAME_V4_2],
        comparison_catalog_receipt=normalized[COMPARISON_CATALOG_RECEIPT_FILENAME_V4_2],
        candidate_selection_spec=normalized[CANDIDATE_SELECTION_SPEC_FILENAME_V4_2],
        strict_full_a_source_binding=normalized[STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2],
        code_binding_set=normalized[CODE_BINDING_SET_FILENAME_V4_2],
    )
    for filename in INPUT_FILENAMES_V4_2:
        if not _canonical_equal(normalized[filename], rebuilt[filename]):
            raise _error(f"cross-artifact graph mismatch: {filename}")
    return normalized


def candidate_preregistration_bundle_contract_v4_2() -> PrivateBundleContract:
    return PrivateBundleContract(
        root_suffix=ROOT_SUFFIX_V4_2,
        input_filenames=INPUT_FILENAMES_V4_2,
        readback_report_filename=READBACK_REPORT_FILENAME_V4_2,
        canonicalize=prereg.canonical_file_bytes_v4_2,
        validate_artifact=_validate_artifact,
        validate_complete=validate_candidate_preregistration_bundle_artifacts_v4_2,
        build_readback_report=_build_readback_report,
    )


def publish_candidate_preregistration_bundle_v4_2(
    *,
    private_root: str | os.PathLike[str],
    artifacts: Mapping[str, Mapping[str, Any]],
    revalidate_inputs: Any,
    _test_fault_hook: Any = None,
    _test_race_hook: Any = None,
) -> dict[str, Any]:
    cycle_root = validate_cycle_root_v4_2(artifacts[CYCLE_ROOT_FILENAME_V4_2])
    return publish_private_bundle(
        private_root=private_root,
        run_id=cycle_root["cycle_id"],
        artifacts=artifacts,
        contract=candidate_preregistration_bundle_contract_v4_2(),
        revalidate_inputs=revalidate_inputs,
        _test_fault_hook=_test_fault_hook,
        _test_race_hook=_test_race_hook,
    )


def readback_candidate_preregistration_bundle_files_v4_2(
    bundle_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read and cross-validate only the owner-private bundle files."""

    return readback_private_bundle(
        bundle_path,
        contract=candidate_preregistration_bundle_contract_v4_2(),
    )


def readback_candidate_preregistration_bundle_v4_2(
    bundle_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read a bundle and hard-reopen its exact immutable historical sources."""

    bundle = readback_candidate_preregistration_bundle_files_v4_2(bundle_path)
    source = bundle["artifacts"][STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2]
    return {
        **bundle,
        "immutable_source_readback": revalidate_recorded_immutable_source_v4_2(
            source
        ),
    }


__all__ = [
    "AQUANT_IDEA_SOURCE_RECEIPT_FILENAME_V4_2",
    "CANDIDATE_SELECTION_SPEC_FILENAME_V4_2",
    "CODE_BINDING_PATHS_V4_2",
    "CODE_BINDING_SET_FILENAME_V4_2",
    "COMPARISON_CATALOG_RECEIPT_FILENAME_V4_2",
    "CYCLE_ROOT_FILENAME_V4_2",
    "DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_2",
    "DISCOVERY_SOURCE_NODE_FILENAME_V4_2",
    "DISCOVERY_STATE_FILENAME_V4_2",
    "FUTURE_SOURCE_ENVELOPE_FILENAME_V4_2",
    "FactorGovernanceCandidatePreregistrationBundleV4_2Error",
    "INPUT_FILENAMES_V4_2",
    "MYQUANT_ALPHA158_SOURCE_RECEIPT_FILENAME_V4_2",
    "OPERATOR_SEMANTICS_FILENAME_V4_2",
    "PRECOMMITTED_STATE_FILENAME_V4_2",
    "PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_2",
    "READBACK_REPORT_FILENAME_V4_2",
    "ROOT_SUFFIX_V4_2",
    "STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2",
    "build_candidate_preregistration_bundle_artifacts_v4_2",
    "build_code_binding_set_v4_2",
    "build_cycle_root_v4_2",
    "build_strict_full_a_source_binding_v4_2",
    "candidate_preregistration_bundle_contract_v4_2",
    "deterministic_cycle_id_v4_2",
    "publish_candidate_preregistration_bundle_v4_2",
    "readback_candidate_preregistration_bundle_v4_2",
    "readback_candidate_preregistration_bundle_files_v4_2",
    "revalidate_recorded_immutable_source_v4_2",
    "revalidate_code_binding_set_v4_2",
    "validate_candidate_preregistration_bundle_artifacts_v4_2",
    "validate_candidate_preregistration_bundle_inputs_v4_2",
    "validate_code_binding_set_v4_2",
    "validate_cycle_root_v4_2",
    "validate_historical_strict_full_a_source_binding_v4_2",
    "validate_strict_full_a_source_binding_v4_2",
]
