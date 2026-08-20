"""Narrow canonical source and recomputation closure for Factor production.

This module deliberately stops before any pointer, CAS, System, Mainline,
Investment, portfolio, Strategy Record, broker, or trading authority.  It
proves only that the two approved price/volume Factors can be recomputed from
one strict Market/PIT/calendar closure.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Final, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_investor.contracts import (
    ContractError,
    artifact_byte_sha256,
    canonical_json_bytes,
    get_contract,
    seal_artifact,
    validate_artifact,
)
from quant_investor.market.exchange_calendar_closure import (
    validate_exchange_calendar_compilation,
)
from quant_investor.market.tushare_calendar_authority import (
    validate_published_trusted_provider_calendar_capture_root,
    validate_trusted_provider_calendar_capture_execution,
    validate_trusted_provider_calendar_capture_success,
    validate_trusted_provider_calendar_capture_transaction,
    validate_trusted_provider_calendar_compilation,
)
from quant_investor.system.release_install import validate_release_install_evidence
from quant_investor.system.release_install import verify_running_release_install_input
from quant_investor.market.tushare_calendar_authority import validate_calendar_authority_policy
from quant_investor.factors.governance.bootstrap_selection import (
    build_market_pit_selection,
    validate_market_pit_selection,
)
from quant_investor.system.components import validate_installed_component_manifest
from quant_investor.system.store import validate_object_ref

from .bootstrap import (
    BLEND_W75_CONTROL,
    BLEND_W80,
    CANONICAL_PARQUET,
    LOW_DOLLAR_VOLUME,
    _set_rows,
    bootstrap_factor_definitions,
    compute_bootstrap_signals,
    validate_bootstrap_factor_set,
)
from .bootstrap_evidence import (
    _DECISION_DOCUMENT,
    _READER_CONTRACT,
    build_bootstrap_exception_evidence,
    validate_bootstrap_exception_evidence,
)
from .errors import FactorGovernanceError
from .implementations import installed_implementation_rows, installed_semantic_row
from .legacy_zero_call import scan_release_legacy_zero_call
from .receipt import _build_factor_validation_receipt, validate_factor_validation_receipt
from .source import role_schema

FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND: Final = "factor.production_source_closure"
FACTOR_PRODUCTION_RECOMPUTATION_KIND: Final = "factor.production_recomputation_evidence"
FACTOR_LEGACY_ZERO_CALL_CERTIFICATE_KIND: Final = "factor.production_legacy_zero_call_certificate"
FACTOR_PRODUCTION_MARKET_INPUT_KIND: Final = "factor.production_market_input"
FACTOR_PRODUCTION_CALENDAR_CUSTODY_KIND: Final = (
    "factor.production_calendar_capture_custody_attestation"
)
FACTOR_PRODUCTION_GENERATION_KIND: Final = "factor.production_generation"
FACTOR_PRODUCTION_SCOPE: Final = "FACTOR_PRODUCTION"
FUNDAMENTAL_NOT_USED: Final = "NOT_USED_BY_ACTIVE_FACTOR_SET"
FUNDAMENTAL_ADVISORY: Final = "ADVISORY"
PRE_CAS_CURRENT: Final = "PRE_CAS_CURRENT"
HISTORICAL_RECOVERY: Final = "HISTORICAL_RECOVERY"
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_DATE_RE: Final = re.compile(r"^[0-9]{8}$")
_TIMESTAMP_RE: Final = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_SYMBOL_RE: Final = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
_MAX_SOURCE_BYTES: Final = 512 * 1024 * 1024
_MAX_SOURCE_ROWS: Final = 10_000_000
_MAX_SOURCE_CELLS: Final = 100_000_000
_BATCH_ROWS: Final = 2_048
_SOURCE_DESCRIPTOR_FIELDS: Final = frozenset(
    {
        "source_object_ref",
        "source_root_id",
        "relative_path",
        "media_type",
        "source_format",
        "byte_sha256",
        "size",
        "stat_identity",
    }
)
_SOURCE_STAT_FIELDS: Final = frozenset(
    {
        "st_ctime_ns",
        "st_dev",
        "st_gid",
        "st_ino",
        "st_mode",
        "st_mtime_ns",
        "st_nlink",
        "st_size",
        "st_uid",
    }
)
_FACTOR_SOURCE_ROLES: Final = (
    "exchange_calendar",
    "market_history",
    "pit_universe",
)
_EXPECTED_SOURCE_MEDIA_TYPE: Final = "application/vnd.apache.parquet"
_EXPECTED_SOURCE_FORMAT: Final = "PARQUET"
_MARKET_PIT_SELECTION_KIND: Final = "factor.production_market_pit_selection"
_CALENDAR_POLICY_KIND: Final = "system.calendar_authority_policy"
_CALENDAR_COMPILATION_KINDS: Final = frozenset(
    {
        "system.exchange_calendar_compilation",
        "system.trusted_provider_calendar_compilation",
    }
)
_SOURCE_BUNDLE_KIND: Final = "system.source_bundle"
_SOURCE_OBJECT_KIND: Final = "system.source_object"
_RELEASE_KIND: Final = "system.release"
_BOOTSTRAP_SOURCE_BUNDLE_ROLES: Final = {
    "decision_source": "bootstrap_decision",
    "exchange_calendar": "calendar",
    "implementation": "implementation_tree_manifest",
    "market": "market",
    "pit_universe": "pit",
    "recomputation": "recomputation",
    "source_generation": "source_generation",
}
_BOOTSTRAP_POLICY_SOURCE_ROLES: Final = ("code", *_BOOTSTRAP_SOURCE_BUNDLE_ROLES)
_MARKET_INPUT_FIELDS: Final = frozenset(
    {
        "factor_market_input_id",
        "state",
        "activation_scope",
        "as_of",
        "market_pit_selection_ref",
        "market_pointer_source_ref",
        "market_snapshot_manifest_source_ref",
        "market_scope_source_ref",
        "market_history_source_ref",
        "market_pointer_sha256",
        "market_snapshot_manifest_sha256",
        "market_history_sha256",
        "market_snapshot_id",
        "market_coverage_sha256",
        "market_expected_scope_sha256",
        "pit_generation_id",
        "pit_membership_sha256",
        "producer_module_path",
        "producer_module_sha256",
    }
)
_MARKET_INPUT_MODULE_PATH: Final = "quant_investor/factors/governance/production_authority.py"

ArtifactResolver = Callable[[Mapping[str, Any]], Mapping[str, Any] | bytes]
SourceResolver = Callable[[Mapping[str, Any], int], tuple[Mapping[str, Any], bytes]]
_SOURCE_CLOSURE_FIELDS: Final = frozenset(
    {
        "factor_production_source_closure_id",
        "state",
        "activation_scope",
        "admission_route",
        "producer_identity",
        "deployed_release_ref",
        "release_install_evidence_ref",
        "release_install_input_source_ref",
        "release_install_verification",
        "market_pit_selection_ref",
        "market_scope_source_ref",
        "calendar_authority_policy_ref",
        "calendar_compilation_ref",
        "calendar_capture_custody_attestation_ref",
        "factor_source_bundle_ref",
        "factor_policy_ref",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
        "factor_implementation_refs",
        "legacy_zero_call_ref",
        "market_input_ref",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    }
)
_RECOMPUTATION_FIELDS: Final = frozenset(
    {
        "factor_production_recomputation_id",
        "state",
        "activation_scope",
        "admission_route",
        "producer_identity",
        "as_of",
        "source_closure_ref",
        "deployed_release_ref",
        "factor_active_set_ref",
        "low_signal_sha256",
        "w80_signal_sha256",
        "signal_statistics",
        "signal_values",
        "active_factor_rows",
        "control_rows",
        "exact_replay_sha256",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
    }
)
_CALENDAR_CUSTODY_FIELDS: Final = frozenset(
    {
        "calendar_capture_custody_attestation_id",
        "state",
        "activation_scope",
        "capture_root_name",
        "deployed_release_ref",
        "capture_transaction_ref",
        "capture_execution_ref",
        "capture_success_ref",
        "published_root_device",
        "published_root_inode",
        "published_leaf_manifest",
        "published_leaf_manifest_sha256",
        "verified_at",
    }
)
_FACTOR_GENERATION_FIELDS: Final = frozenset(
    {
        "factor_production_generation_id",
        "state",
        "activation_scope",
        "admission_route",
        "producer_identity",
        "as_of",
        "deployed_release_ref",
        "release_install_evidence_ref",
        "release_install_input_source_ref",
        "release_install_verification",
        "source_closure_ref",
        "recomputation_evidence_ref",
        "market_pit_selection_ref",
        "market_scope_source_ref",
        "calendar_compilation_ref",
        "calendar_capture_custody_attestation_ref",
        "factor_source_bundle_ref",
        "market_input_ref",
        "factor_policy_ref",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
        "factor_implementation_refs",
        "legacy_zero_call_ref",
        "low_signal_sha256",
        "w80_signal_sha256",
        "signal_statistics",
        "signal_values",
        "active_factor_rows",
        "control_rows",
        "exact_replay_sha256",
        "fundamental_dependency_state",
        "fundamental_freshness_policy",
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    }
)
_LEGACY_ZERO_CALL_FIELDS: Final = frozenset(
    {
        "factor_legacy_zero_call_id",
        "state",
        "activation_scope",
        "final_commit",
        "final_tree",
        "resolver_inventory_ref",
        "active_legacy_import_count",
        "active_legacy_call_count",
        "active_legacy_path_hash_count",
        "legacy_entrypoint_count",
        "verification_module_path",
        "verification_module_sha256",
        "verification_command",
        "stdout_sha256",
        "stderr_sha256",
        "verified_at",
    }
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise FactorGovernanceError(f"{label} is not lowercase SHA-256")
    return value


def _git_oid(value: Any, *, label: str) -> str:
    if type(value) is not str or _GIT_OID_RE.fullmatch(value) is None:
        raise FactorGovernanceError(f"{label} is not a Git object ID")
    return value


def _as_of(value: Any) -> str:
    if type(value) is not str:
        raise FactorGovernanceError("Factor production as_of is invalid")
    compact = (
        value.replace("-", "") if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) else value
    )
    if _DATE_RE.fullmatch(compact) is None:
        raise FactorGovernanceError("Factor production as_of is invalid")
    try:
        parsed = datetime.strptime(compact, "%Y%m%d")
    except ValueError as exc:
        raise FactorGovernanceError("Factor production as_of is invalid") from exc
    if parsed.strftime("%Y%m%d") != compact:
        raise FactorGovernanceError("Factor production as_of is invalid")
    return compact


def _timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str or _TIMESTAMP_RE.fullmatch(value) is None:
        raise FactorGovernanceError(f"{label} is not canonical UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise FactorGovernanceError(f"{label} is not canonical UTC seconds") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise FactorGovernanceError(f"{label} is not canonical UTC seconds")
    return value


def _contract_sha(kind: str) -> str:
    """Resolve lazily while the contract owner integrates the authority lane."""

    return get_contract(kind).contract_sha256


def _artifact_ref(document: Mapping[str, Any]) -> dict[str, str]:
    try:
        artifact = validate_artifact(document)
    except ContractError as exc:
        raise FactorGovernanceError("Factor production artifact is invalid") from exc
    return {
        "kind": artifact["kind"],
        "contract_sha256": artifact["contract_sha256"],
        "artifact_id": artifact["artifact_id"],
        "semantic_sha256": artifact["semantic_sha256"],
        "byte_sha256": artifact_byte_sha256(artifact),
    }


def _artifact_ref_value(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != {
        "kind",
        "contract_sha256",
        "artifact_id",
        "semantic_sha256",
        "byte_sha256",
    }:
        raise FactorGovernanceError(f"{label} is not an exact artifact ref")
    return {
        "kind": str(value["kind"]),
        "contract_sha256": _sha(value["contract_sha256"], label=f"{label}.contract"),
        "artifact_id": str(value["artifact_id"]),
        "semantic_sha256": _sha(value["semantic_sha256"], label=f"{label}.semantic"),
        "byte_sha256": _sha(value["byte_sha256"], label=f"{label}.bytes"),
    }


def _refs(values: Sequence[Mapping[str, Any]], *, label: str) -> list[dict[str, str]]:
    rows = [
        validate_object_ref(value, label=f"{label}[{index}]") for index, value in enumerate(values)
    ]
    keys = [
        (
            row["kind"],
            row["contract_sha256"],
            row["artifact_id"],
            row["semantic_sha256"],
            row["byte_sha256"],
        )
        for row in rows
    ]
    if not rows or keys != sorted(keys) or len(keys) != len(set(keys)):
        raise FactorGovernanceError(f"{label} must be nonempty sorted unique refs")
    return rows


def _require_no_authority(payload: Mapping[str, Any]) -> None:
    for field in (
        "system_authority",
        "mainline_authority",
        "investment_authority",
        "portfolio_authority",
        "strategy_record_authority",
        "broker_authority",
    ):
        if payload.get(field) != "NONE":
            raise FactorGovernanceError(f"Factor production {field} must remain NONE")


def _validate_release_install_verification(value: Any) -> dict[str, Any]:
    fields = {
        "state",
        "release_ref",
        "source_archive_sha256",
        "wheel_sha256",
        "code_tree_sha256",
        "installed_code_manifest_sha256",
        "contract_catalog_sha256",
        "import_origin",
    }
    if type(value) is not dict or set(value) != fields or value["state"] != "PASS":
        raise FactorGovernanceError("Factor release-install verification fields differ")
    result = dict(value)
    result["release_ref"] = validate_object_ref(result["release_ref"], label="release_ref")
    for field in (
        "source_archive_sha256",
        "wheel_sha256",
        "code_tree_sha256",
        "installed_code_manifest_sha256",
        "contract_catalog_sha256",
    ):
        result[field] = _sha(result[field], label=f"release_install_verification.{field}")
    if type(result["import_origin"]) is not str or not result["import_origin"]:
        raise FactorGovernanceError("Factor release-install import origin differs")
    return result


def _source_object_artifact(
    document: Mapping[str, Any] | bytes,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        artifact = validate_artifact(document, expected_kind=_SOURCE_OBJECT_KIND)
    except ContractError as exc:
        raise FactorGovernanceError(f"{label} source object contract failed") from exc
    payload = artifact["payload"]
    if (
        type(payload) is not dict
        or type(payload.get("relative_path")) is not str
        or type(payload.get("byte_sha256")) is not str
    ):
        raise FactorGovernanceError(f"{label} source object payload differs")
    _sha(payload["byte_sha256"], label=f"{label}.byte_sha256")
    return artifact


def _canonical_json_mapping(raw: Any, *, label: str) -> dict[str, Any]:
    """Decode exact sealed JSON bytes without rewriting their on-disk form.

    Canonical market pointers/manifests are often intentionally pretty-printed
    operational JSON.  Their original bytes remain SHA-bound in the source
    object; this helper creates only an in-memory canonical semantic projection
    after rejecting duplicate keys, non-finite constants, and non-JSON values.
    """

    if type(raw) is not bytes:
        raise FactorGovernanceError(f"{label} raw bytes differ")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FactorGovernanceError(f"{label} has duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise FactorGovernanceError(f"{label} has a non-finite JSON constant: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
        canonical_json_bytes(value)
    except (UnicodeError, ValueError, TypeError, ContractError) as exc:
        raise FactorGovernanceError(f"{label} is not strict JSON") from exc
    if type(value) is not dict:
        raise FactorGovernanceError(f"{label} is not a JSON object")
    return dict(value)


def _market_binding(
    *,
    selection: Mapping[str, Any],
    market_pointer: Mapping[str, Any],
    market_snapshot_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    selection_payload = selection["payload"]
    pointer = dict(market_pointer)
    manifest = dict(market_snapshot_manifest)
    coverage = pointer.get("coverage")
    manifest_coverage = manifest.get("coverage")
    if (
        type(coverage) is not dict
        or canonical_json_bytes(coverage) != canonical_json_bytes(manifest_coverage)
        or pointer.get("status") != "OK"
        or manifest.get("status") != "OK"
        or pointer.get("blockers") != []
        or manifest.get("blockers") != []
        or pointer.get("snapshot_id") != manifest.get("snapshot_id")
    ):
        raise FactorGovernanceError("Factor Market pointer/snapshot closure differs")
    as_of = _as_of(selection_payload["as_of"])
    if (
        _as_of(pointer.get("latest_complete_trade_date")) != as_of
        or _as_of(manifest.get("latest_complete_trade_date")) != as_of
        or _as_of(coverage.get("latest_complete_trade_date")) != as_of
        or _as_of(coverage.get("coverage_trade_date")) != as_of
        or _as_of(coverage.get("upsert_target_trade_date")) != as_of
        or coverage.get("coverage_schema_version") != "cn-full-a-coverage.v4"
        or coverage.get("expected_scope_sha256")
        != selection_payload["market_expected_scope_sha256"]
        or _sha256(canonical_json_bytes(coverage)) != selection_payload["market_coverage_sha256"]
        or coverage.get("pit_generation_id") != selection_payload["pit_generation_id"]
        or coverage.get("pit_generation_manifest_sha256")
        != selection_payload["pit_generation_manifest_sha256"]
        or coverage.get("pit_membership_sha256") != selection_payload["pit_membership_sha256"]
    ):
        raise FactorGovernanceError("Factor Market coverage/PIT binding differs")
    if type(pointer.get("snapshot_id")) is not str or not pointer["snapshot_id"]:
        raise FactorGovernanceError("Factor Market snapshot identity differs")
    return {
        "as_of": as_of,
        "market_snapshot_id": pointer["snapshot_id"],
        "market_coverage_sha256": selection_payload["market_coverage_sha256"],
        "market_expected_scope_sha256": selection_payload["market_expected_scope_sha256"],
        "pit_generation_id": selection_payload["pit_generation_id"],
        "pit_membership_sha256": selection_payload["pit_membership_sha256"],
    }


def _production_module_sha256() -> str:
    return _sha256(Path(__file__).resolve(strict=True).read_bytes())


def build_factor_production_market_input(
    *,
    market_pit_selection: Mapping[str, Any] | bytes,
    market_pointer_source: Mapping[str, Any] | bytes,
    market_snapshot_manifest_source: Mapping[str, Any] | bytes,
    market_scope_source: Mapping[str, Any] | bytes,
    market_history_source: Mapping[str, Any] | bytes,
    market_pointer_raw: bytes,
    market_snapshot_manifest_raw: bytes,
    created_at: str,
) -> dict[str, Any]:
    """Seal the non-authorizing Market snapshot-to-Factor-table bridge."""

    selection = validate_market_pit_selection(market_pit_selection)
    pointer_source = _source_object_artifact(market_pointer_source, label="market pointer")
    manifest_source = _source_object_artifact(
        market_snapshot_manifest_source, label="market snapshot manifest"
    )
    scope_source = _source_object_artifact(market_scope_source, label="market scope")
    history_source = _source_object_artifact(market_history_source, label="market history")
    pointer_payload = pointer_source["payload"]
    manifest_payload = manifest_source["payload"]
    history_payload = history_source["payload"]
    if (
        _sha256(market_pointer_raw) != pointer_payload["byte_sha256"]
        or _sha256(market_snapshot_manifest_raw) != manifest_payload["byte_sha256"]
        or history_payload.get("source_format") != _EXPECTED_SOURCE_FORMAT
        or history_payload.get("media_type") != _EXPECTED_SOURCE_MEDIA_TYPE
    ):
        raise FactorGovernanceError("Factor Market source raw binding differs")
    binding = _market_binding(
        selection=selection,
        market_pointer=_canonical_json_mapping(market_pointer_raw, label="market pointer"),
        market_snapshot_manifest=_canonical_json_mapping(
            market_snapshot_manifest_raw, label="market snapshot manifest"
        ),
    )
    body = {
        "state": "VERIFIED",
        "activation_scope": FACTOR_PRODUCTION_SCOPE,
        "as_of": binding["as_of"],
        "market_pit_selection_ref": _artifact_ref(selection),
        "market_pointer_source_ref": _artifact_ref(pointer_source),
        "market_snapshot_manifest_source_ref": _artifact_ref(manifest_source),
        "market_scope_source_ref": _artifact_ref(scope_source),
        "market_history_source_ref": _artifact_ref(history_source),
        "market_pointer_sha256": pointer_payload["byte_sha256"],
        "market_snapshot_manifest_sha256": manifest_payload["byte_sha256"],
        "market_history_sha256": history_payload["byte_sha256"],
        "market_snapshot_id": binding["market_snapshot_id"],
        "market_coverage_sha256": binding["market_coverage_sha256"],
        "market_expected_scope_sha256": binding["market_expected_scope_sha256"],
        "pit_generation_id": binding["pit_generation_id"],
        "pit_membership_sha256": binding["pit_membership_sha256"],
        "producer_module_path": _MARKET_INPUT_MODULE_PATH,
        "producer_module_sha256": _production_module_sha256(),
    }
    identity = "factor-market-input-" + _sha256(canonical_json_bytes(body))
    artifact = seal_artifact(
        FACTOR_PRODUCTION_MARKET_INPUT_KIND,
        {"factor_market_input_id": identity, **body},
        created_at=created_at,
        contract_sha256=_contract_sha(FACTOR_PRODUCTION_MARKET_INPUT_KIND),
    )
    return validate_factor_production_market_input(artifact)


def validate_factor_production_market_input(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    try:
        artifact = validate_artifact(
            document,
            expected_kind=FACTOR_PRODUCTION_MARKET_INPUT_KIND,
            expected_contract_sha256=_contract_sha(FACTOR_PRODUCTION_MARKET_INPUT_KIND),
        )
    except ContractError as exc:
        raise FactorGovernanceError("Factor Market input contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != _MARKET_INPUT_FIELDS:
        raise FactorGovernanceError("Factor Market input fields differ")
    if (
        payload["state"] != "VERIFIED"
        or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE
        or payload["producer_module_path"] != _MARKET_INPUT_MODULE_PATH
        or payload["producer_module_sha256"] != _production_module_sha256()
    ):
        raise FactorGovernanceError("Factor Market input producer identity differs")
    _as_of(payload["as_of"])
    for field in (
        "market_pointer_sha256",
        "market_snapshot_manifest_sha256",
        "market_history_sha256",
        "market_coverage_sha256",
        "market_expected_scope_sha256",
        "pit_membership_sha256",
    ):
        _sha(payload[field], label=f"market_input.{field}")
    for field in (
        "market_pit_selection_ref",
        "market_pointer_source_ref",
        "market_snapshot_manifest_source_ref",
        "market_scope_source_ref",
        "market_history_source_ref",
    ):
        validate_object_ref(payload[field], label=f"market_input.{field}")
    if (
        payload["market_pit_selection_ref"]["kind"] != _MARKET_PIT_SELECTION_KIND
        or payload["market_pointer_source_ref"]["kind"] != _SOURCE_OBJECT_KIND
        or payload["market_snapshot_manifest_source_ref"]["kind"] != _SOURCE_OBJECT_KIND
        or payload["market_scope_source_ref"]["kind"] != _SOURCE_OBJECT_KIND
        or payload["market_history_source_ref"]["kind"] != _SOURCE_OBJECT_KIND
        or type(payload["market_snapshot_id"]) is not str
        or not payload["market_snapshot_id"]
        or type(payload["pit_generation_id"]) is not str
        or not payload["pit_generation_id"]
    ):
        raise FactorGovernanceError("Factor Market input refs differ")
    body = dict(payload)
    identity = body.pop("factor_market_input_id")
    if identity != "factor-market-input-" + _sha256(canonical_json_bytes(body)):
        raise FactorGovernanceError("Factor Market input identity differs")
    return artifact


def _capture_file_ref(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != {"relative_path", "byte_sha256"}:
        raise FactorGovernanceError(f"{label} is not an exact capture file ref")
    relative = value["relative_path"]
    if (
        type(relative) is not str
        or not relative
        or relative.startswith("/")
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise FactorGovernanceError(f"{label} relative path differs")
    return {
        "relative_path": relative,
        "byte_sha256": _sha(value["byte_sha256"], label=f"{label}.byte_sha256"),
    }


def build_factor_calendar_capture_custody_attestation(
    *,
    capture_parent: str | os.PathLike[str],
    capture_execution: Mapping[str, Any] | bytes,
    capture_execution_file_ref: Mapping[str, Any],
    capture_success: Mapping[str, Any] | bytes,
    capture_success_file_ref: Mapping[str, Any],
    deployed_release_ref: Mapping[str, Any],
    verified_at: str,
) -> dict[str, Any]:
    """Seal proof that the original published Calendar root passed custody replay.

    The filesystem validator runs *before* any leaf is copied into Factor
    custody.  The resulting attestation binds the original root identity and
    every exact transaction/execution/success/release leaf byte.
    """

    execution = validate_artifact(
        capture_execution,
        expected_kind="system.trusted_provider_calendar_capture_execution",
    )
    success = validate_artifact(
        capture_success,
        expected_kind="system.trusted_provider_calendar_capture_success",
    )
    execution_file_ref = _capture_file_ref(
        capture_execution_file_ref, label="capture_execution_file_ref"
    )
    success_file_ref = _capture_file_ref(capture_success_file_ref, label="capture_success_file_ref")
    leaves = validate_published_trusted_provider_calendar_capture_root(
        capture_parent=capture_parent,
        capture_execution=execution,
        capture_execution_file_ref=execution_file_ref,
        capture_success=success,
        capture_success_file_ref=success_file_ref,
    )
    execution_payload = execution["payload"]
    success_payload = success["payload"]
    release_ref = validate_object_ref(deployed_release_ref, label="deployed_release_ref")
    if execution_payload.get("deployed_release_ref") != release_ref:
        raise FactorGovernanceError("Calendar capture deployed release differs")
    root_name = execution_payload.get("capture_root_name")
    if (
        type(root_name) is not str
        or not root_name
        or success_payload.get("capture_root_name") != root_name
    ):
        raise FactorGovernanceError("Calendar capture root identity differs")
    transaction_raw = leaves.get("capture-transaction.json")
    if type(transaction_raw) is not bytes:
        raise FactorGovernanceError("Calendar capture transaction bytes are absent")
    transaction = validate_artifact(
        transaction_raw,
        expected_kind="system.trusted_provider_calendar_capture_transaction",
    )
    leaf_manifest: list[dict[str, Any]] = [
        {
            "relative_path": f"{root_name}/{leaf}",
            "byte_sha256": _sha256(raw),
            "size": len(raw),
        }
        for leaf, raw in sorted(leaves.items())
    ]
    if any(row["size"] <= 0 for row in leaf_manifest):
        raise FactorGovernanceError("Calendar capture contains an empty custody leaf")
    body = {
        "state": "VERIFIED",
        "activation_scope": FACTOR_PRODUCTION_SCOPE,
        "capture_root_name": root_name,
        "deployed_release_ref": release_ref,
        "capture_transaction_ref": _artifact_ref(transaction),
        "capture_execution_ref": _artifact_ref(execution),
        "capture_success_ref": _artifact_ref(success),
        "published_root_device": success_payload.get("published_root_device"),
        "published_root_inode": success_payload.get("published_root_inode"),
        "published_leaf_manifest": leaf_manifest,
        "published_leaf_manifest_sha256": _sha256(canonical_json_bytes(leaf_manifest)),
        "verified_at": _timestamp(verified_at, label="verified_at"),
    }
    if (
        type(body["published_root_device"]) is not int
        or body["published_root_device"] < 0
        or type(body["published_root_inode"]) is not int
        or body["published_root_inode"] <= 0
    ):
        raise FactorGovernanceError("Calendar capture published root identity differs")
    identity = "factor-calendar-custody-" + _sha256(canonical_json_bytes(body))
    artifact = seal_artifact(
        FACTOR_PRODUCTION_CALENDAR_CUSTODY_KIND,
        {"calendar_capture_custody_attestation_id": identity, **body},
        created_at=verified_at,
        contract_sha256=_contract_sha(FACTOR_PRODUCTION_CALENDAR_CUSTODY_KIND),
    )
    return validate_factor_calendar_capture_custody_attestation(artifact)


def validate_factor_calendar_capture_custody_attestation(  # noqa: C901
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    try:
        artifact = validate_artifact(
            document,
            expected_kind=FACTOR_PRODUCTION_CALENDAR_CUSTODY_KIND,
            expected_contract_sha256=_contract_sha(FACTOR_PRODUCTION_CALENDAR_CUSTODY_KIND),
        )
    except ContractError as exc:
        raise FactorGovernanceError("Calendar capture custody attestation contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != _CALENDAR_CUSTODY_FIELDS:
        raise FactorGovernanceError("Calendar capture custody attestation fields differ")
    if payload["state"] != "VERIFIED" or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE:
        raise FactorGovernanceError("Calendar capture custody attestation state differs")
    validate_object_ref(payload["deployed_release_ref"], label="deployed_release_ref")
    expected_kinds = {
        "capture_transaction_ref": "system.trusted_provider_calendar_capture_transaction",
        "capture_execution_ref": "system.trusted_provider_calendar_capture_execution",
        "capture_success_ref": "system.trusted_provider_calendar_capture_success",
    }
    for field, kind in expected_kinds.items():
        ref = _artifact_ref_value(payload[field], label=field)
        if ref["kind"] != kind:
            raise FactorGovernanceError(f"{field} kind differs")
    root_name = payload["capture_root_name"]
    rows = payload["published_leaf_manifest"]
    if type(root_name) is not str or not root_name or type(rows) is not list or not rows:
        raise FactorGovernanceError("Calendar capture custody manifest differs")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if type(row) is not dict or set(row) != {"relative_path", "byte_sha256", "size"}:
            raise FactorGovernanceError("Calendar capture custody leaf differs")
        ref = _capture_file_ref(
            {"relative_path": row["relative_path"], "byte_sha256": row["byte_sha256"]},
            label="published_leaf_manifest",
        )
        if not ref["relative_path"].startswith(root_name + "/"):
            raise FactorGovernanceError("Calendar capture custody leaf root differs")
        if type(row["size"]) is not int or row["size"] <= 0:
            raise FactorGovernanceError("Calendar capture custody leaf size differs")
        normalized.append({**ref, "size": row["size"]})
    if normalized != sorted(normalized, key=lambda row: row["relative_path"]):
        raise FactorGovernanceError("Calendar capture custody manifest order differs")
    if len({row["relative_path"] for row in normalized}) != len(normalized):
        raise FactorGovernanceError("Calendar capture custody manifest duplicates a leaf")
    if _sha256(canonical_json_bytes(normalized)) != _sha(
        payload["published_leaf_manifest_sha256"], label="published_leaf_manifest_sha256"
    ):
        raise FactorGovernanceError("Calendar capture custody manifest SHA differs")
    if (
        type(payload["published_root_device"]) is not int
        or payload["published_root_device"] < 0
        or type(payload["published_root_inode"]) is not int
        or payload["published_root_inode"] <= 0
    ):
        raise FactorGovernanceError("Calendar capture custody root identity differs")
    _timestamp(payload["verified_at"], label="verified_at")
    body = dict(payload)
    identity = body.pop("calendar_capture_custody_attestation_id")
    if identity != "factor-calendar-custody-" + _sha256(canonical_json_bytes(body)):
        raise FactorGovernanceError("Calendar capture custody identity differs")
    return artifact


def _same_ref(left: Mapping[str, Any], right: Mapping[str, Any], *, label: str) -> None:
    if validate_object_ref(left, label=f"{label}.left") != validate_object_ref(
        right, label=f"{label}.right"
    ):
        raise FactorGovernanceError(f"{label} differs")


def _resolve_exact_artifact(
    value: Mapping[str, Any],
    *,
    artifact_resolver: ArtifactResolver,
    label: str,
    expected_kinds: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Resolve one ref and require all five immutable identity fields to agree."""

    ref = validate_object_ref(value, label=label)
    try:
        artifact = validate_artifact(artifact_resolver(ref))
    except (ContractError, TypeError, ValueError) as exc:
        raise FactorGovernanceError(f"{label} cannot be resolved as a sealed artifact") from exc
    if _artifact_ref(artifact) != ref:
        raise FactorGovernanceError(f"{label} resolved artifact identity differs")
    if expected_kinds is not None and artifact["kind"] not in expected_kinds:
        raise FactorGovernanceError(f"{label} artifact kind differs")
    return artifact


def _source_stat_identity(
    value: Any,
    *,
    descriptor: Mapping[str, Any],
    raw: bytes,
    label: str,
) -> dict[str, int]:
    if type(value) is not dict or set(value) != _SOURCE_STAT_FIELDS:
        raise FactorGovernanceError(f"{label} source stat identity differs")
    identity: dict[str, int] = {}
    for field in _SOURCE_STAT_FIELDS:
        item = value[field]
        if type(item) is not int:
            raise FactorGovernanceError(f"{label} source stat identity differs")
        identity[field] = item
    if (
        not stat.S_ISREG(identity["st_mode"])
        or identity["st_uid"] != os.geteuid()
        or identity["st_nlink"] != 1
        or identity["st_size"] != len(raw)
    ):
        raise FactorGovernanceError(f"{label} source storage identity differs")
    mode = stat.S_IMODE(identity["st_mode"])
    if mode & 0o077 or mode & 0o100 or not mode & 0o400:
        raise FactorGovernanceError(f"{label} source storage mode differs")
    if descriptor["size"] != len(raw):
        raise FactorGovernanceError(f"{label} source descriptor size differs")
    return identity


def _source_descriptor(
    value: Any,
    *,
    source_ref: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    raw: bytes,
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _SOURCE_DESCRIPTOR_FIELDS:
        raise FactorGovernanceError(f"{label} source descriptor fields differ")
    ref = validate_object_ref(value["source_object_ref"], label=f"{label}.source_object_ref")
    _same_ref(ref, source_ref, label=f"{label}.source_object_ref")
    for field in ("source_root_id", "relative_path", "media_type", "source_format"):
        if value[field] != source_payload.get(field):
            raise FactorGovernanceError(f"{label} source descriptor differs")
    if value["byte_sha256"] != source_payload.get("byte_sha256"):
        raise FactorGovernanceError(f"{label} source byte SHA differs")
    _sha(value["byte_sha256"], label=f"{label}.byte_sha256")
    if type(value["size"]) is not int or value["size"] <= 0 or value["size"] > _MAX_SOURCE_BYTES:
        raise FactorGovernanceError(f"{label} source size differs")
    if _sha256(raw) != value["byte_sha256"]:
        raise FactorGovernanceError(f"{label} source raw SHA differs")
    _source_stat_identity(value["stat_identity"], descriptor=value, raw=raw, label=label)
    return dict(value)


def _read_source_object_twice(
    source_ref: Mapping[str, Any],
    *,
    source_payload: Mapping[str, Any],
    source_resolver: SourceResolver,
    label: str,
) -> bytes:
    """Read exact source bytes twice; a protected resolver supplies NOFOLLOW custody."""

    try:
        first_descriptor, first_raw = source_resolver(source_ref, _MAX_SOURCE_BYTES)
        second_descriptor, second_raw = source_resolver(source_ref, _MAX_SOURCE_BYTES)
    except (OSError, ValueError, TypeError) as exc:
        raise FactorGovernanceError(f"{label} source read failed") from exc
    if type(first_raw) is not bytes or type(second_raw) is not bytes:
        raise FactorGovernanceError(f"{label} source raw bytes differ")
    first = _source_descriptor(
        first_descriptor,
        source_ref=source_ref,
        source_payload=source_payload,
        raw=first_raw,
        label=f"{label}.first",
    )
    second = _source_descriptor(
        second_descriptor,
        source_ref=source_ref,
        source_payload=source_payload,
        raw=second_raw,
        label=f"{label}.second",
    )
    if first != second or first_raw != second_raw:
        raise FactorGovernanceError(f"{label} source changed between readbacks")
    return first_raw


def system_store_source_resolver(store: Any) -> SourceResolver:
    """Return the read-only source-custody adapter used by Factor authority.

    The adapter calls only source-object inspection and byte-read methods.  It
    never asks ``SystemStore`` for an active pointer, generation, marker, or
    any authority status.  Both inspections and the byte read are required so
    an inode, mode, hard-link, or content replacement cannot be hidden between
    the two Factor verifier reads.
    """

    def resolve(
        source_ref: Mapping[str, Any], maximum_bytes: int
    ) -> tuple[Mapping[str, Any], bytes]:
        try:
            before = store.inspect_source_object(
                source_ref, full_hash=True, maximum_bytes=maximum_bytes
            )
            _payload, raw = store.read_source_object_bytes(source_ref, maximum_bytes=maximum_bytes)
            after = store.inspect_source_object(
                source_ref, full_hash=True, maximum_bytes=maximum_bytes
            )
        except Exception as exc:  # Source storage errors are deliberately fail-closed here.
            raise FactorGovernanceError("Factor source custody read failed") from exc
        if before != after or before.get("byte_sha256") != _sha256(raw):
            raise FactorGovernanceError("Factor source custody changed during read")
        return before, raw

    return resolve


def build_factor_production_source_closure(
    *,
    deployed_release_ref: Mapping[str, Any],
    release_install_evidence_ref: Mapping[str, Any],
    release_install_input_source_ref: Mapping[str, Any],
    release_install_verification: Mapping[str, Any],
    market_pit_selection_ref: Mapping[str, Any],
    market_scope_source_ref: Mapping[str, Any],
    calendar_authority_policy_ref: Mapping[str, Any],
    calendar_compilation_ref: Mapping[str, Any],
    calendar_capture_custody_attestation_ref: Mapping[str, Any],
    factor_source_bundle_ref: Mapping[str, Any],
    factor_policy_ref: Mapping[str, Any],
    factor_active_set_ref: Mapping[str, Any],
    factor_validation_attestation_ref: Mapping[str, Any],
    factor_implementation_refs: Sequence[Mapping[str, Any]],
    legacy_zero_call_ref: Mapping[str, Any],
    market_input_ref: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    """Seal the narrow non-System authority closure for LOW/W80 production."""

    body = {
        "state": "VERIFIED",
        "activation_scope": FACTOR_PRODUCTION_SCOPE,
        "admission_route": "BOOTSTRAP_EXCEPTION",
        "producer_identity": "NOT_CLAIMED",
        "deployed_release_ref": validate_object_ref(deployed_release_ref),
        "release_install_evidence_ref": validate_object_ref(release_install_evidence_ref),
        "release_install_input_source_ref": validate_object_ref(release_install_input_source_ref),
        "release_install_verification": _validate_release_install_verification(
            release_install_verification
        ),
        "market_pit_selection_ref": validate_object_ref(market_pit_selection_ref),
        "market_scope_source_ref": validate_object_ref(market_scope_source_ref),
        "calendar_authority_policy_ref": validate_object_ref(calendar_authority_policy_ref),
        "calendar_compilation_ref": validate_object_ref(calendar_compilation_ref),
        "calendar_capture_custody_attestation_ref": validate_object_ref(
            calendar_capture_custody_attestation_ref
        ),
        "factor_source_bundle_ref": validate_object_ref(factor_source_bundle_ref),
        "factor_policy_ref": validate_object_ref(factor_policy_ref),
        "factor_active_set_ref": validate_object_ref(factor_active_set_ref),
        "factor_validation_attestation_ref": validate_object_ref(factor_validation_attestation_ref),
        "factor_implementation_refs": _refs(
            factor_implementation_refs, label="factor_implementation_refs"
        ),
        "legacy_zero_call_ref": validate_object_ref(legacy_zero_call_ref),
        "market_input_ref": validate_object_ref(market_input_ref),
        "fundamental_dependency_state": FUNDAMENTAL_NOT_USED,
        "fundamental_freshness_policy": FUNDAMENTAL_ADVISORY,
        "system_authority": "NONE",
        "mainline_authority": "NONE",
        "investment_authority": "NONE",
        "portfolio_authority": "NONE",
        "strategy_record_authority": "NONE",
        "broker_authority": "NONE",
    }
    identity = "factor-production-source-" + _sha256(canonical_json_bytes(body))
    artifact = seal_artifact(
        FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND,
        {"factor_production_source_closure_id": identity, **body},
        created_at=created_at,
        contract_sha256=_contract_sha(FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND),
    )
    return validate_factor_production_source_closure(artifact)


def validate_factor_production_source_closure(  # noqa: C901
    document: Mapping[str, Any] | bytes,
    *,
    artifact_resolver: ArtifactResolver | None = None,
    source_resolver: SourceResolver | None = None,
) -> dict[str, Any]:
    """Validate the sealed closure and optionally replay its three raw sources.

    A structural-only call is suitable while creating the immutable leaf.  A
    production Factor authority must provide *both* resolvers; that branch
    resolves all artifact refs, verifies the direct source topology and reads
    Calendar/Market/PIT exact bytes twice before re-running LOW and W80.
    """

    if (artifact_resolver is None) != (source_resolver is None):
        raise FactorGovernanceError("Factor production deep resolver pair is incomplete")
    try:
        artifact = validate_artifact(
            document,
            expected_kind=FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND,
            expected_contract_sha256=_contract_sha(FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND),
        )
    except ContractError as exc:
        raise FactorGovernanceError("Factor production source closure contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != _SOURCE_CLOSURE_FIELDS:
        raise FactorGovernanceError("Factor production source closure fields differ")
    if (
        payload["state"] != "VERIFIED"
        or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE
        or payload["admission_route"] != "BOOTSTRAP_EXCEPTION"
        or payload["producer_identity"] != "NOT_CLAIMED"
        or payload["fundamental_dependency_state"] != FUNDAMENTAL_NOT_USED
        or payload["fundamental_freshness_policy"] != FUNDAMENTAL_ADVISORY
    ):
        raise FactorGovernanceError("Factor production source closure policy differs")
    _require_no_authority(payload)
    for field in (
        "deployed_release_ref",
        "release_install_evidence_ref",
        "release_install_input_source_ref",
        "market_pit_selection_ref",
        "market_scope_source_ref",
        "calendar_authority_policy_ref",
        "calendar_compilation_ref",
        "calendar_capture_custody_attestation_ref",
        "factor_source_bundle_ref",
        "factor_policy_ref",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
        "legacy_zero_call_ref",
        "market_input_ref",
    ):
        validate_object_ref(payload[field], label=field)
    if payload["legacy_zero_call_ref"]["kind"] != FACTOR_LEGACY_ZERO_CALL_CERTIFICATE_KIND:
        raise FactorGovernanceError("Factor production legacy-zero-call ref differs")
    if payload["market_input_ref"]["kind"] != FACTOR_PRODUCTION_MARKET_INPUT_KIND:
        raise FactorGovernanceError("Factor production Market input ref differs")
    if payload["release_install_evidence_ref"]["kind"] != "system.release_install_evidence":
        raise FactorGovernanceError("Factor production release-install evidence ref differs")
    if payload["release_install_input_source_ref"]["kind"] != _SOURCE_OBJECT_KIND:
        raise FactorGovernanceError("Factor production release-install input source ref differs")
    _validate_release_install_verification(payload["release_install_verification"])
    if payload["market_scope_source_ref"]["kind"] != _SOURCE_OBJECT_KIND:
        raise FactorGovernanceError("Factor production Market scope source ref differs")
    if (
        payload["calendar_capture_custody_attestation_ref"]["kind"]
        != FACTOR_PRODUCTION_CALENDAR_CUSTODY_KIND
    ):
        raise FactorGovernanceError("Factor Calendar custody attestation ref differs")
    _refs(payload["factor_implementation_refs"], label="factor_implementation_refs")
    body = dict(payload)
    identity = body.pop("factor_production_source_closure_id")
    if identity != "factor-production-source-" + _sha256(canonical_json_bytes(body)):
        raise FactorGovernanceError("Factor production source closure identity differs")
    if artifact_resolver is not None and source_resolver is not None:
        _deep_replay_source_closure(
            artifact,
            artifact_resolver=artifact_resolver,
            source_resolver=source_resolver,
        )
    return artifact


def _read_stable_file_bytes(path: Path, *, expected_sha256: str) -> bytes:  # noqa: C901
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FactorGovernanceError("strict Factor production source is unavailable") from exc
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or mode & 0o077
        or mode & 0o100
        or not mode & 0o400
        or metadata.st_size <= 0
        or metadata.st_size > _MAX_SOURCE_BYTES
    ):
        raise FactorGovernanceError("strict Factor production source storage is invalid")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise FactorGovernanceError("strict Factor production source cannot be opened") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as exc:
        raise FactorGovernanceError("strict Factor production source changed during read") from exc

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
        )

    if (
        identity(metadata) != identity(before)
        or identity(before) != identity(after)
        or identity(after) != identity(current)
    ):
        raise FactorGovernanceError("strict Factor production source changed during hash")
    if digest.hexdigest() != _sha(expected_sha256, label="strict source SHA"):
        raise FactorGovernanceError("strict Factor production source SHA differs")
    return b"".join(chunks)


def _read_parquet_table(
    parquet: pq.ParquetFile,
    *,
    role: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    try:
        if parquet.schema_arrow != role_schema(role):
            raise FactorGovernanceError("strict Factor production Parquet schema differs")
        metadata = parquet.metadata
        if metadata.num_rows <= 0 or metadata.num_rows > _MAX_SOURCE_ROWS:
            raise FactorGovernanceError("strict Factor production source row bound differs")
        frames: list[pd.DataFrame] = []
        observed_rows = 0
        for batch in parquet.iter_batches(batch_size=_BATCH_ROWS, columns=list(columns)):
            observed_rows += batch.num_rows
            if observed_rows > _MAX_SOURCE_ROWS or observed_rows * len(columns) > _MAX_SOURCE_CELLS:
                raise FactorGovernanceError("strict Factor production source cell bound differs")
            frames.append(batch.to_pandas())
    except Exception as exc:
        if isinstance(exc, FactorGovernanceError):
            raise
        raise FactorGovernanceError("strict Factor production Parquet input is unreadable") from exc
    if observed_rows != metadata.num_rows or not frames:
        raise FactorGovernanceError("strict Factor production source row count differs")
    return pd.concat(frames, ignore_index=True)


def _read_table(
    path: Path,
    *,
    role: str,
    columns: Sequence[str],
    expected_sha256: str,
) -> pd.DataFrame:
    return _read_table_raw(
        _read_stable_file_bytes(path, expected_sha256=expected_sha256),
        role=role,
        columns=columns,
        expected_sha256=expected_sha256,
    )


def _read_table_raw(
    raw: bytes,
    *,
    role: str,
    columns: Sequence[str],
    expected_sha256: str,
) -> pd.DataFrame:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_SOURCE_BYTES:
        raise FactorGovernanceError("strict Factor production raw source size differs")
    if _sha256(raw) != _sha(expected_sha256, label="strict source SHA"):
        raise FactorGovernanceError("strict Factor production raw source SHA differs")
    try:
        return _read_parquet_table(
            pq.ParquetFile(pa.BufferReader(raw)),
            role=role,
            columns=columns,
        )
    except Exception as exc:
        if isinstance(exc, FactorGovernanceError):
            raise
        raise FactorGovernanceError(
            "strict Factor production raw Parquet input is unreadable"
        ) from exc


def _canonical_sessions(calendar: pd.DataFrame, *, as_of: str) -> list[date]:
    expected = {"ordinal", "open_session", "opens_at_utc", "closes_at_utc"}
    if set(calendar.columns) != expected:
        raise FactorGovernanceError("strict Factor calendar schema differs")
    sessions = list(calendar["open_session"])
    if any(not isinstance(value, date) or isinstance(value, datetime) for value in sessions):
        raise FactorGovernanceError("strict Factor calendar session type differs")
    ordinals = list(calendar["ordinal"])
    if any(type(value) is not int for value in ordinals) or ordinals != list(range(len(calendar))):
        raise FactorGovernanceError("strict Factor calendar ordinal sequence differs")
    opens = list(calendar["opens_at_utc"])
    closes = list(calendar["closes_at_utc"])
    for session, opens_at, closes_at in zip(sessions, opens, closes, strict=True):
        open_stamp = pd.Timestamp(opens_at)
        close_stamp = pd.Timestamp(closes_at)
        if (
            open_stamp.tzinfo is None
            or close_stamp.tzinfo is None
            or open_stamp.utcoffset() != pd.Timedelta(0)
            or close_stamp.utcoffset() != pd.Timedelta(0)
            or open_stamp >= close_stamp
            or open_stamp.date() != session
            or close_stamp.date() != session
            or open_stamp.hour != 1
            or open_stamp.minute != 30
            or open_stamp.second != 0
            or close_stamp.hour != 7
            or close_stamp.minute != 0
            or close_stamp.second != 0
        ):
            raise FactorGovernanceError("strict Factor calendar session times differ")
    ordered = sorted(sessions)
    if sessions != ordered:
        raise FactorGovernanceError("strict Factor calendar session order differs")
    cutoff = datetime.strptime(as_of, "%Y%m%d").date()
    selected = [value for value in ordered if value <= cutoff]
    if len(selected) < 91 or not selected or selected[-1] != cutoff:
        raise FactorGovernanceError("strict Factor calendar does not close the 90-session window")
    if len(selected) != len(set(selected)):
        raise FactorGovernanceError("strict Factor calendar sessions are duplicated")
    return selected


def _eligible_pit(pit: pd.DataFrame, *, signal_session: date) -> list[str]:
    expected = {"signal_session", "symbol", "industry", "total_mv", "tradable"}
    if set(pit.columns) != expected:
        raise FactorGovernanceError("strict Factor PIT schema differs")
    rows = pit.loc[pit["signal_session"].eq(signal_session)]
    if rows.empty or rows["symbol"].duplicated().any():
        raise FactorGovernanceError("strict Factor PIT cohort is invalid")
    symbols = sorted(
        [str(value) for value in rows.loc[rows["tradable"].eq(True), "symbol"]],
        key=lambda value: value.encode("utf-8"),
    )
    if not symbols or any(_SYMBOL_RE.fullmatch(value) is None for value in symbols):
        raise FactorGovernanceError("strict Factor PIT eligible cohort is invalid")
    return symbols


def _market_frames(
    market: pd.DataFrame,
    *,
    sessions: Sequence[date],
    eligible_symbols: Sequence[str],
) -> dict[str, pd.DataFrame]:
    expected = {"trade_date", "symbol", "adj_close", "amount", "vol"}
    if set(market.columns) != expected:
        raise FactorGovernanceError("strict Factor Market schema differs")
    rows = market.loc[
        market["trade_date"].isin(set(sessions)) & market["symbol"].isin(set(eligible_symbols))
    ].copy()
    if rows.empty or rows.duplicated(subset=["trade_date", "symbol"]).any():
        raise FactorGovernanceError("strict Factor Market rows are invalid")
    frames: dict[str, pd.DataFrame] = {}
    for symbol in eligible_symbols:
        frame = rows.loc[rows["symbol"].eq(symbol)].drop(columns=["symbol"])
        frame = frame.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
        if len(frame) != len(sessions):
            raise FactorGovernanceError("strict Factor Market history does not match PIT cohort")
        frames[symbol] = frame
    return frames


def _signal_projection(
    series: pd.Series, *, factor_id: str
) -> tuple[dict[str, str], dict[str, Any]]:
    values: dict[str, str] = {}
    finite: list[float] = []
    for symbol, value in series.sort_index().items():
        if type(symbol) is not str or _SYMBOL_RE.fullmatch(symbol) is None:
            raise FactorGovernanceError("Factor signal symbol is invalid")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise FactorGovernanceError("Factor signal is nonfinite")
        values[symbol] = numeric.hex()
        finite.append(numeric)
    if not finite or len(set(finite)) <= 1:
        raise FactorGovernanceError("Factor signal is empty or constant")
    return values, {
        "factor_id": factor_id,
        "signal_symbol_set_sha256": _sha256(canonical_json_bytes(sorted(values))),
        "source_symbol_count": len(values),
        "finite_count": len(finite),
        "distinct_finite_count": len(set(finite)),
        "coverage_numerator": len(values),
        "coverage_denominator": len(values),
        "coverage_rate": "1.000000000000",
        "signal_sha256": _sha256(canonical_json_bytes(values)),
        "implementation_sha256": installed_semantic_row(factor_id)["code_sha256"],
    }


def _expected_factor_policy_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    definitions = bootstrap_factor_definitions()
    active_rows, control_rows = _set_rows(definitions)
    expected_active = [
        row
        for row in sorted(definitions, key=lambda value: value["factor_id"].encode("utf-8"))
        if row["factor_id"] in {LOW_DOLLAR_VOLUME, BLEND_W80}
    ]
    expected_active_rows = [
        {
            "factor_id": row["factor_id"],
            "spec_id": row["spec_id"],
            "direction": row["direction"],
            "required_source_roles": list(row["required_source_roles"]),
            "weight": row["bootstrap_weight"],
            "role": row["role"],
            "selectable": row["selectable"],
        }
        for row in expected_active
    ]
    expected_control = [
        {
            "factor_id": BLEND_W75_CONTROL,
            "spec_id": next(
                row["spec_id"] for row in definitions if row["factor_id"] == BLEND_W75_CONTROL
            ),
            "direction": "HIGHER_IS_BETTER",
            "required_source_roles": ["EXCHANGE_CALENDAR", "MARKET", "PIT_MEMBERSHIP"],
            "weight": "0.000000000000",
            "role": "CONTROL_ONLY",
            "selectable": False,
        }
    ]
    if active_rows != expected_active_rows or control_rows != expected_control:
        raise FactorGovernanceError("installed Factor policy rows differ")
    return expected_active_rows, expected_control


def _validate_signal_statistics(value: Any) -> list[dict[str, Any]]:
    required = {
        "factor_id",
        "signal_symbol_set_sha256",
        "source_symbol_count",
        "finite_count",
        "distinct_finite_count",
        "coverage_numerator",
        "coverage_denominator",
        "coverage_rate",
        "signal_sha256",
        "implementation_sha256",
    }
    if type(value) is not list or len(value) != 2:
        raise FactorGovernanceError("Factor signal statistics are incomplete")
    expected_ids = [LOW_DOLLAR_VOLUME, BLEND_W80]
    rows: list[dict[str, Any]] = []
    for factor_id, row in zip(expected_ids, value, strict=True):
        if type(row) is not dict or set(row) != required or row["factor_id"] != factor_id:
            raise FactorGovernanceError("Factor signal statistics policy differs")
        for field in (
            "signal_symbol_set_sha256",
            "signal_sha256",
            "implementation_sha256",
        ):
            _sha(row[field], label=f"signal_statistics.{factor_id}.{field}")
        counts = (
            "source_symbol_count",
            "finite_count",
            "distinct_finite_count",
            "coverage_numerator",
            "coverage_denominator",
        )
        if any(type(row[field]) is not int or row[field] <= 0 for field in counts):
            raise FactorGovernanceError("Factor signal statistics counts differ")
        if (
            row["distinct_finite_count"] <= 1
            or row["distinct_finite_count"] > row["finite_count"]
            or row["finite_count"] != row["source_symbol_count"]
            or row["coverage_numerator"] != row["source_symbol_count"]
            or row["coverage_denominator"] != row["source_symbol_count"]
            or row["coverage_rate"] != "1.000000000000"
            or row["implementation_sha256"] != installed_semantic_row(factor_id)["code_sha256"]
        ):
            raise FactorGovernanceError("Factor signal statistics closure differs")
        rows.append(dict(row))
    return rows


def _recompute_loaded_sources(
    *,
    calendar: pd.DataFrame,
    pit: pd.DataFrame,
    market: pd.DataFrame,
    as_of: str,
    exchange_calendar_sha256: str,
    pit_universe_sha256: str,
    market_history_sha256: str,
) -> dict[str, Any]:
    cutoff = _as_of(as_of)
    full_sessions = _canonical_sessions(calendar, as_of=cutoff)
    sessions = full_sessions[-91:]
    if len(sessions) != 91:
        raise FactorGovernanceError("strict Factor calendar lacks the 91-session signal window")
    eligible = _eligible_pit(pit, signal_session=sessions[-1])
    frames = _market_frames(market, sessions=sessions, eligible_symbols=eligible)
    signals = compute_bootstrap_signals(frames, source_format=CANONICAL_PARQUET)
    low_values, low_statistics = _signal_projection(
        signals[LOW_DOLLAR_VOLUME], factor_id=LOW_DOLLAR_VOLUME
    )
    w80_values, w80_statistics = _signal_projection(signals[BLEND_W80], factor_id=BLEND_W80)
    active_rows, control_rows = _expected_factor_policy_rows()
    replay = {
        "as_of": cutoff,
        "eligible_symbols": eligible,
        "low": low_values,
        "w80": w80_values,
        "source_format": CANONICAL_PARQUET,
        "source_sha256s": {
            "exchange_calendar": _sha(exchange_calendar_sha256, label="calendar SHA"),
            "pit_universe": _sha(pit_universe_sha256, label="PIT SHA"),
            "market_history": _sha(market_history_sha256, label="Market SHA"),
        },
    }
    return {
        "as_of": cutoff,
        "active_factor_rows": active_rows,
        "control_rows": control_rows,
        "low_signal_sha256": low_statistics["signal_sha256"],
        "w80_signal_sha256": w80_statistics["signal_sha256"],
        "signal_values": {
            LOW_DOLLAR_VOLUME: low_values,
            BLEND_W80: w80_values,
        },
        "signal_statistics": [low_statistics, w80_statistics],
        "exact_replay_sha256": _sha256(canonical_json_bytes(replay)),
        "fundamental_dependency_state": FUNDAMENTAL_NOT_USED,
        "fundamental_freshness_policy": FUNDAMENTAL_ADVISORY,
        "fundamental_veto_effect": "NOT_APPLICABLE_UNLESS_SUPPLIED",
    }


def recompute_factor_production_signals(
    *,
    exchange_calendar_path: str | Path,
    pit_universe_path: str | Path,
    market_history_path: str | Path,
    exchange_calendar_sha256: str,
    pit_universe_sha256: str,
    market_history_sha256: str,
    as_of: str,
) -> dict[str, Any]:
    """Real strict-Parquet LOW/W80 recomputation with W75 permanently inert."""
    calendar = _read_table(
        Path(exchange_calendar_path),
        role="exchange_calendar",
        columns=["ordinal", "open_session", "opens_at_utc", "closes_at_utc"],
        expected_sha256=exchange_calendar_sha256,
    )
    pit = _read_table(
        Path(pit_universe_path),
        role="pit_universe",
        columns=["signal_session", "symbol", "industry", "total_mv", "tradable"],
        expected_sha256=pit_universe_sha256,
    )
    market = _read_table(
        Path(market_history_path),
        role="market_history",
        columns=["trade_date", "symbol", "adj_close", "amount", "vol"],
        expected_sha256=market_history_sha256,
    )
    return _recompute_loaded_sources(
        calendar=calendar,
        pit=pit,
        market=market,
        as_of=as_of,
        exchange_calendar_sha256=exchange_calendar_sha256,
        pit_universe_sha256=pit_universe_sha256,
        market_history_sha256=market_history_sha256,
    )


def _selection_as_of(
    selection: Mapping[str, Any],
    *,
    pit_source_payload: Mapping[str, Any],
) -> str:
    """Validate the Factor-relevant immutable Market-bound PIT projection."""

    try:
        validated = validate_market_pit_selection(selection)
    except Exception as exc:
        raise FactorGovernanceError("Market-bound PIT selection typed replay failed") from exc
    payload = validated.get("payload")
    required = {
        "market_pit_selection_id",
        "state",
        "selection_mode",
        "as_of",
        "market_pointer_file_ref",
        "market_snapshot_manifest_file_ref",
        "market_snapshot_id",
        "market_coverage_sha256",
        "market_expected_scope_sha256",
        "market_bound_pit_pointer_file_ref",
        "pit_generation_id",
        "pit_generation_manifest_file_ref",
        "pit_membership_file_ref",
        "pit_generation_manifest_sha256",
        "pit_membership_sha256",
        "observed_current_pit_pointer_file_ref",
        "observed_current_pit_pointer_sha256",
        "observed_current_pit_generation_id",
        "pinned_as_of_disclosure",
        "user_authorization_basis",
        "selection_module_path",
        "selection_module_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise FactorGovernanceError("Market-bound PIT selection fields differ")
    if (
        payload["state"] != "SEALED"
        or payload["selection_mode"] != "MARKET_COVERAGE_BOUND"
        or payload["pinned_as_of_disclosure"] != "MARKET_COVERAGE_BOUND_PIT_NOT_GLOBAL_CURRENT"
        or payload["user_authorization_basis"] != "USER_AUTHORIZED_BOOTSTRAP_EXCEPTION"
    ):
        raise FactorGovernanceError("Market-bound PIT selection policy differs")
    as_of = _as_of(payload["as_of"])
    for field in (
        "market_coverage_sha256",
        "market_expected_scope_sha256",
        "pit_generation_manifest_sha256",
        "pit_membership_sha256",
        "observed_current_pit_pointer_sha256",
        "selection_module_sha256",
    ):
        _sha(payload[field], label=f"market_pit_selection.{field}")
    for field in (
        "market_pointer_file_ref",
        "market_snapshot_manifest_file_ref",
        "market_bound_pit_pointer_file_ref",
        "pit_generation_manifest_file_ref",
        "pit_membership_file_ref",
        "observed_current_pit_pointer_file_ref",
    ):
        value = payload[field]
        if type(value) is not dict or set(value) != {"relative_path", "byte_sha256"}:
            raise FactorGovernanceError(f"Market-bound PIT selection {field} differs")
        if type(value["relative_path"]) is not str or not value["relative_path"]:
            raise FactorGovernanceError(f"Market-bound PIT selection {field} path differs")
        _sha(value["byte_sha256"], label=f"market_pit_selection.{field}.byte_sha256")
    if (
        payload["pit_membership_sha256"] != pit_source_payload["byte_sha256"]
        or payload["pit_membership_file_ref"]["byte_sha256"] != pit_source_payload["byte_sha256"]
        or payload["pit_membership_file_ref"]["relative_path"]
        != pit_source_payload["relative_path"]
    ):
        raise FactorGovernanceError("Market-bound PIT selection does not bind strict PIT bytes")
    return as_of


def _factor_source_topology(  # noqa: C901
    source_bundle_ref: Mapping[str, Any],
    *,
    artifact_resolver: ArtifactResolver,
    source_resolver: SourceResolver,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Resolve Factor input leaves plus the complete calendar capture subtree.

    The top-level source bundle stays limited to Calendar, Market, and PIT.
    Calendar is a nested immutable bundle because semantic official/Tushare
    replay needs its runtime bytes *and* every raw/capture leaf. PIT is also
    nested so the Market-bound selection can re-read its bound pointer,
    manifest, membership, and observed-current disclosure. Market is nested
    to preserve its exact full-A scope alongside the strict Factor table.
    """

    def flatten_capture_bundle(
        bundle_ref: Mapping[str, Any],
        *,
        label: str,
        role_path: tuple[str, ...] = (),
        ancestors: frozenset[str] = frozenset(),
    ) -> list[dict[str, Any]]:
        bundle = _resolve_exact_artifact(
            bundle_ref,
            artifact_resolver=artifact_resolver,
            label=label,
            expected_kinds=frozenset({_SOURCE_BUNDLE_KIND}),
        )
        bundle_sha = artifact_byte_sha256(bundle)
        if bundle_sha in ancestors:
            raise FactorGovernanceError("Factor calendar source bundle is cyclic")
        bundle_payload = bundle["payload"]
        rows = bundle_payload.get("sources") if type(bundle_payload) is dict else None
        if bundle_payload.get("state") != "IMMUTABLE" or type(rows) is not list or not rows:
            raise FactorGovernanceError("Factor calendar source bundle differs")
        raw_roles = [row.get("role") if type(row) is dict else None for row in rows]
        if any(type(role) is not str or not role for role in raw_roles):
            raise FactorGovernanceError("Factor calendar source roles differ")
        roles = [cast(str, role) for role in raw_roles]
        if roles != sorted(roles, key=lambda value: value.encode("utf-8")) or len(roles) != len(
            set(roles)
        ):
            raise FactorGovernanceError("Factor calendar source roles differ")
        leaves: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if type(row) is not dict or set(row) != {"role", "source_ref"}:
                raise FactorGovernanceError("Factor calendar source row differs")
            ref = validate_object_ref(row["source_ref"], label=f"{label}[{index}]")
            if ref["kind"] == _SOURCE_OBJECT_KIND:
                leaf = _resolved_source_leaf(
                    ref,
                    artifact_resolver=artifact_resolver,
                    source_resolver=source_resolver,
                    label=f"{label}.{row['role']}",
                )
                leaf["role_path"] = (*role_path, row["role"])
                leaves.append(leaf)
            elif ref["kind"] == _SOURCE_BUNDLE_KIND:
                leaves.extend(
                    flatten_capture_bundle(
                        ref,
                        label=f"{label}.{row['role']}",
                        role_path=(*role_path, row["role"]),
                        ancestors=ancestors | {bundle_sha},
                    )
                )
            else:
                raise FactorGovernanceError("Factor calendar source kind differs")
        keys = [
            (
                row["payload"].get("relative_path"),
                row["payload"].get("byte_sha256"),
            )
            for row in leaves
        ]
        if len(keys) != len(set(keys)):
            raise FactorGovernanceError("Factor source bundle leaves are duplicated")
        return leaves

    bundle = _resolve_exact_artifact(
        source_bundle_ref,
        artifact_resolver=artifact_resolver,
        label="factor_source_bundle_ref",
        expected_kinds=frozenset({_SOURCE_BUNDLE_KIND}),
    )
    payload = bundle["payload"]
    rows = payload.get("sources") if type(payload) is dict else None
    expected_roles = list(_FACTOR_SOURCE_ROLES)
    if (
        payload.get("state") != "IMMUTABLE"
        or type(rows) is not list
        or [row.get("role") if type(row) is dict else None for row in rows] != expected_roles
    ):
        raise FactorGovernanceError("Factor source bundle topology differs")
    branch_leaves: dict[str, list[dict[str, Any]]] = {}
    for role, row in zip(expected_roles, rows, strict=True):
        if type(row) is not dict or set(row) != {"role", "source_ref"}:
            raise FactorGovernanceError("Factor source bundle row differs")
        source_ref = validate_object_ref(row["source_ref"], label=f"factor source {role}")
        if role in _FACTOR_SOURCE_ROLES:
            if source_ref["kind"] != _SOURCE_BUNDLE_KIND:
                raise FactorGovernanceError(f"Factor {role} capture bundle is absent")
            branch_leaves[role] = flatten_capture_bundle(
                source_ref, label=f"factor {role} capture bundle"
            )
            continue
    if set(branch_leaves) != {
        "exchange_calendar",
        "market_history",
        "pit_universe",
    }:
        raise FactorGovernanceError("Factor source input topology differs")
    return {}, branch_leaves


def _leaf_for_file_ref(
    leaves: Sequence[Mapping[str, Any]],
    file_ref: Any,
    *,
    label: str,
    source_format: str | None = None,
) -> dict[str, Any]:
    if type(file_ref) is not dict or set(file_ref) != {"relative_path", "byte_sha256"}:
        raise FactorGovernanceError(f"{label} file ref differs")
    if type(file_ref["relative_path"]) is not str:
        raise FactorGovernanceError(f"{label} file path differs")
    _sha(file_ref["byte_sha256"], label=f"{label}.byte_sha256")
    matches = [
        dict(leaf)
        for leaf in leaves
        if leaf["payload"].get("relative_path") == file_ref["relative_path"]
        and leaf["payload"].get("byte_sha256") == file_ref["byte_sha256"]
    ]
    if len(matches) != 1:
        raise FactorGovernanceError(f"{label} source object is absent or ambiguous")
    leaf = matches[0]
    if source_format is not None and leaf["payload"].get("source_format") != source_format:
        raise FactorGovernanceError(f"{label} source format differs")
    return leaf


def _leaf_for_terminal_role(
    leaves: Sequence[Mapping[str, Any]],
    role: str,
    *,
    label: str,
    source_format: str,
) -> dict[str, Any]:
    matches = [
        dict(leaf)
        for leaf in leaves
        if leaf.get("role_path")
        and leaf["role_path"][-1] == role
        and leaf["payload"].get("source_format") == source_format
    ]
    if len(matches) != 1:
        raise FactorGovernanceError(f"{label} source leaf is absent or ambiguous")
    return matches[0]


def _canonical_pit_symbols(raw: bytes) -> list[str]:
    """Read only canonical PIT symbols to bind the Factor projection cohort."""

    if type(raw) is not bytes or not raw or len(raw) > _MAX_SOURCE_BYTES:
        raise FactorGovernanceError("canonical PIT membership bytes differ")
    try:
        parquet = pq.ParquetFile(pa.BufferReader(raw))
        if "symbol" not in parquet.schema_arrow.names:
            raise FactorGovernanceError("canonical PIT membership lacks symbols")
        symbols: list[str] = []
        for batch in parquet.iter_batches(batch_size=_BATCH_ROWS, columns=["symbol"]):
            for value in batch.column(0).to_pylist():
                if type(value) is not str or _SYMBOL_RE.fullmatch(value) is None:
                    raise FactorGovernanceError("canonical PIT symbol is invalid")
                symbols.append(value)
    except Exception as exc:
        if isinstance(exc, FactorGovernanceError):
            raise
        raise FactorGovernanceError("canonical PIT membership is unreadable") from exc
    if not symbols or len(symbols) != len(set(symbols)):
        raise FactorGovernanceError("canonical PIT symbols are incomplete")
    return sorted(symbols, key=lambda value: value.encode("utf-8"))


def _market_scope_sha256(symbols: Sequence[str]) -> str:
    if not symbols or list(symbols) != sorted(
        set(symbols), key=lambda value: value.encode("utf-8")
    ):
        raise FactorGovernanceError("Market expected scope symbols are not canonical")
    return _sha256("\n".join(symbols).encode("utf-8"))


def _require_market_scope_sha256(symbols: Sequence[str], expected_sha256: Any) -> None:
    if _market_scope_sha256(symbols) != _sha(expected_sha256, label="Market expected scope SHA"):
        raise FactorGovernanceError("Market expected scope does not match canonical PIT cohort")


def _market_scope_symbols(raw: bytes) -> list[str]:
    scope = _canonical_json_mapping(raw, label="Market scope")
    symbols = scope.get("full_a")
    if type(symbols) is not list or not symbols:
        raise FactorGovernanceError("Market scope full_a symbols are absent")
    normalized = [value for value in symbols if type(value) is str and _SYMBOL_RE.fullmatch(value)]
    if normalized != symbols or normalized != sorted(
        set(normalized), key=lambda value: value.encode("utf-8")
    ):
        raise FactorGovernanceError("Market scope full_a symbols are not canonical")
    stats = scope.get("stats")
    if type(stats) is not dict or stats.get("full_a") != len(normalized):
        raise FactorGovernanceError("Market scope full_a count differs")
    return list(normalized)


def _require_market_pit_scope_relation(
    *,
    market_scope_symbols: Sequence[str],
    canonical_pit_symbols: Sequence[str],
    factor_projection_symbols: Sequence[str],
) -> None:
    scope = list(market_scope_symbols)
    if not set(scope) <= set(canonical_pit_symbols):
        raise FactorGovernanceError("Market expected scope is outside canonical PIT membership")
    if list(factor_projection_symbols) != scope:
        raise FactorGovernanceError(
            "Factor PIT projection cohort differs from Market expected scope"
        )


def _deep_replay_market_pit_selection(
    selection: Mapping[str, Any],
    *,
    market_leaves: Sequence[Mapping[str, Any]],
    pit_leaves: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Rebuild the selection from its sealed raw pointer/manifest inputs."""

    validated = validate_market_pit_selection(selection)
    payload = validated["payload"]
    selected = {
        "market_pointer": _leaf_for_file_ref(
            market_leaves,
            payload["market_pointer_file_ref"],
            label="Market pointer",
            source_format="JSON",
        ),
        "market_snapshot_manifest": _leaf_for_file_ref(
            market_leaves,
            payload["market_snapshot_manifest_file_ref"],
            label="Market snapshot manifest",
            source_format="JSON",
        ),
        "market_bound_pit_pointer": _leaf_for_file_ref(
            pit_leaves,
            payload["market_bound_pit_pointer_file_ref"],
            label="Market-bound PIT pointer",
            source_format="JSON",
        ),
        "pit_generation_manifest": _leaf_for_file_ref(
            pit_leaves,
            payload["pit_generation_manifest_file_ref"],
            label="PIT generation manifest",
            source_format="JSON",
        ),
        "pit_membership": _leaf_for_file_ref(
            pit_leaves,
            payload["pit_membership_file_ref"],
            label="PIT membership",
            source_format="PARQUET",
        ),
        "observed_current_pit_pointer": _leaf_for_file_ref(
            pit_leaves,
            payload["observed_current_pit_pointer_file_ref"],
            label="observed current PIT pointer",
            source_format="JSON",
        ),
    }
    rebuilt = build_market_pit_selection(
        as_of=payload["as_of"],
        market_pointer_file_ref=payload["market_pointer_file_ref"],
        market_snapshot_manifest_file_ref=payload["market_snapshot_manifest_file_ref"],
        market_bound_pit_pointer_file_ref=payload["market_bound_pit_pointer_file_ref"],
        pit_generation_manifest_file_ref=payload["pit_generation_manifest_file_ref"],
        pit_membership_file_ref=payload["pit_membership_file_ref"],
        observed_current_pit_pointer_file_ref=payload["observed_current_pit_pointer_file_ref"],
        market_pointer=_canonical_json_mapping(
            selected["market_pointer"]["raw"], label="Market pointer"
        ),
        market_snapshot_manifest=_canonical_json_mapping(
            selected["market_snapshot_manifest"]["raw"], label="Market snapshot manifest"
        ),
        market_bound_pit_pointer=_canonical_json_mapping(
            selected["market_bound_pit_pointer"]["raw"], label="Market-bound PIT pointer"
        ),
        pit_generation_manifest=_canonical_json_mapping(
            selected["pit_generation_manifest"]["raw"], label="PIT generation manifest"
        ),
        observed_current_pit_pointer=_canonical_json_mapping(
            selected["observed_current_pit_pointer"]["raw"],
            label="observed current PIT pointer",
        ),
        created_at=validated["created_at"],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(validated):
        raise FactorGovernanceError("Market-bound PIT selection raw replay differs")
    return validated, selected


def _deep_replay_market_input(
    market_input: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    selected_files: Mapping[str, Mapping[str, Any]],
    market_leaf: Mapping[str, Any],
    market_scope_leaf: Mapping[str, Any],
) -> dict[str, Any]:
    input_artifact = validate_factor_production_market_input(market_input)
    payload = input_artifact["payload"]
    _same_ref(
        payload["market_pit_selection_ref"],
        _artifact_ref(selection),
        label="Factor Market input selection ref",
    )
    expected_sources = {
        "market_pointer_source_ref": selected_files["market_pointer"],
        "market_snapshot_manifest_source_ref": selected_files["market_snapshot_manifest"],
        "market_history_source_ref": market_leaf,
    }
    for field, leaf in expected_sources.items():
        _same_ref(payload[field], leaf["ref"], label=f"Factor Market input {field}")
        if payload[field.replace("_source_ref", "_sha256")] != leaf["payload"]["byte_sha256"]:
            raise FactorGovernanceError(f"Factor Market input {field} raw SHA differs")
    _same_ref(
        payload["market_scope_source_ref"],
        market_scope_leaf["ref"],
        label="Factor Market input market_scope_source_ref",
    )
    selection_payload = selection["payload"]
    binding = _market_binding(
        selection=selection,
        market_pointer=_canonical_json_mapping(
            selected_files["market_pointer"]["raw"], label="Market pointer"
        ),
        market_snapshot_manifest=_canonical_json_mapping(
            selected_files["market_snapshot_manifest"]["raw"], label="Market snapshot manifest"
        ),
    )
    for field in (
        "as_of",
        "market_snapshot_id",
        "market_coverage_sha256",
        "market_expected_scope_sha256",
        "pit_generation_id",
        "pit_membership_sha256",
    ):
        if payload[field] != binding[field]:
            raise FactorGovernanceError(f"Factor Market input {field} binding differs")
    if payload["pit_membership_sha256"] != selection_payload["pit_membership_sha256"]:
        raise FactorGovernanceError("Factor Market input PIT membership differs")
    return input_artifact


def _resolved_source_leaf(
    source_ref: Mapping[str, Any],
    *,
    artifact_resolver: ArtifactResolver,
    source_resolver: SourceResolver,
    label: str,
) -> dict[str, Any]:
    source = _resolve_exact_artifact(
        source_ref,
        artifact_resolver=artifact_resolver,
        label=label,
        expected_kinds=frozenset({_SOURCE_OBJECT_KIND}),
    )
    source_payload = source["payload"]
    if type(source_payload) is not dict:
        raise FactorGovernanceError(f"{label} source payload differs")
    return {
        "ref": validate_object_ref(source_ref, label=label),
        "artifact": source,
        "payload": source_payload,
        "raw": _read_source_object_twice(
            source_ref,
            source_payload=source_payload,
            source_resolver=source_resolver,
            label=label,
        ),
    }


def _bootstrap_policy_source_refs(payload: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Normalize the one release plus seven Bootstrap source references."""

    rows = payload.get("source_refs")
    if type(rows) is not list or len(rows) != len(_BOOTSTRAP_POLICY_SOURCE_ROLES):
        raise FactorGovernanceError("Bootstrap policy source refs are incomplete")
    result: dict[str, dict[str, str]] = {}
    observed_roles: list[str] = []
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {"role", "ref"} or type(row["role"]) is not str:
            raise FactorGovernanceError("Bootstrap policy source ref row differs")
        role = row["role"]
        if role in result:
            raise FactorGovernanceError("Bootstrap policy source roles are duplicated")
        expected_kind = _RELEASE_KIND if role == "code" else _SOURCE_BUNDLE_KIND
        if role not in _BOOTSTRAP_POLICY_SOURCE_ROLES:
            raise FactorGovernanceError("Bootstrap policy source role differs")
        reference = validate_object_ref(row["ref"], label=f"Bootstrap source_refs[{index}]")
        if reference["kind"] != expected_kind:
            raise FactorGovernanceError("Bootstrap policy source ref kind differs")
        observed_roles.append(role)
        result[role] = reference
    if tuple(observed_roles) != _BOOTSTRAP_POLICY_SOURCE_ROLES:
        raise FactorGovernanceError("Bootstrap policy source ref order differs")
    if len({_ref_sort_key(reference) for reference in result.values()}) != len(result):
        raise FactorGovernanceError("Bootstrap policy source refs are duplicated")
    return result


def _resolve_bootstrap_bundle_leaf(
    reference: Mapping[str, Any],
    *,
    outer_role: str,
    inner_role: str,
    artifact_resolver: ArtifactResolver,
    source_resolver: SourceResolver,
) -> dict[str, Any]:
    """Resolve one exact Bootstrap bundle and its sole source object."""

    bundle = _resolve_exact_artifact(
        reference,
        artifact_resolver=artifact_resolver,
        label=f"Bootstrap {outer_role} source bundle",
        expected_kinds=frozenset({_SOURCE_BUNDLE_KIND}),
    )
    bundle_payload = bundle["payload"]
    rows = bundle_payload.get("sources") if type(bundle_payload) is dict else None
    if (
        bundle_payload.get("state") != "IMMUTABLE"
        or type(rows) is not list
        or len(rows) != 1
        or type(rows[0]) is not dict
        or set(rows[0]) != {"role", "source_ref"}
        or rows[0]["role"] != inner_role
    ):
        raise FactorGovernanceError(f"Bootstrap {outer_role} source bundle differs")
    source_ref = validate_object_ref(
        rows[0]["source_ref"], label=f"Bootstrap {outer_role} source object ref"
    )
    if source_ref["kind"] != _SOURCE_OBJECT_KIND:
        raise FactorGovernanceError(f"Bootstrap {outer_role} source object kind differs")
    return _resolved_source_leaf(
        source_ref,
        artifact_resolver=artifact_resolver,
        source_resolver=source_resolver,
        label=f"Bootstrap {outer_role} source object",
    )


def _require_shared_source_root(
    leaves: Mapping[str, Mapping[str, Any]],
) -> str:
    """Reject any Bootstrap leaf that does not belong to one preparation root."""

    roots: set[str] = set()
    for label, leaf in leaves.items():
        payload = leaf.get("payload")
        root_id = payload.get("source_root_id") if type(payload) is dict else None
        if type(root_id) is not str or not root_id:
            raise FactorGovernanceError(f"{label} source root identity differs")
        roots.add(root_id)
    if len(roots) != 1:
        raise FactorGovernanceError("Bootstrap source objects span multiple source roots")
    return next(iter(roots))


def _require_exact_generated_json(
    leaf: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    expected_source_object_id: str,
    expected_relative_path: str,
    label: str,
) -> None:
    payload = leaf.get("payload")
    raw = leaf.get("raw")
    if (
        type(payload) is not dict
        or payload.get("source_object_id") != expected_source_object_id
        or payload.get("relative_path") != expected_relative_path
        or payload.get("source_format") != "JSON"
        or payload.get("media_type") != "application/json"
        or type(raw) is not bytes
        or raw != canonical_json_bytes(dict(expected))
    ):
        raise FactorGovernanceError(f"Bootstrap {label} generated bytes differ")


def _validate_deep_bootstrap_receipt_closure(  # noqa: C901
    *,
    payload: Mapping[str, Any],
    policy: Mapping[str, Any],
    active: Mapping[str, Any],
    attestation: Mapping[str, Any],
    implementation_component_refs: Mapping[str, Mapping[str, Any]],
    artifact_resolver: ArtifactResolver,
    source_resolver: SourceResolver,
    primary_source_leaves: Mapping[str, Mapping[str, Any]],
    recomputation: Mapping[str, Any],
) -> None:
    """Deeply replay the intrinsic receipt's eight Bootstrap evidence roots.

    The Bootstrap policy and receipt have separate sealed envelopes. Both
    must name the same release and the same seven one-object source bundles;
    each source object is then read through the sole caller-provided custody
    resolver before its generated content is accepted.
    """

    policy_refs = _bootstrap_policy_source_refs(policy["payload"])
    _same_ref(
        policy_refs["code"],
        payload["deployed_release_ref"],
        label="Bootstrap policy deployed release ref",
    )
    resolved_release = _resolve_exact_artifact(
        policy_refs["code"],
        artifact_resolver=artifact_resolver,
        label="Bootstrap policy release",
        expected_kinds=frozenset({_RELEASE_KIND}),
    )
    if _artifact_ref(resolved_release) != policy_refs["code"]:
        raise FactorGovernanceError("Bootstrap policy release identity differs")

    receipt_refs = attestation["payload"].get("evidence_refs")
    if type(receipt_refs) is not list:
        raise FactorGovernanceError("Bootstrap intrinsic receipt evidence refs are absent")
    normalized_receipt_refs = [
        validate_object_ref(value, label=f"Bootstrap intrinsic evidence_refs[{index}]")
        for index, value in enumerate(receipt_refs)
    ]
    expected_receipt_refs = sorted(policy_refs.values(), key=_ref_sort_key)
    if normalized_receipt_refs != expected_receipt_refs:
        raise FactorGovernanceError("Bootstrap intrinsic receipt evidence refs differ")
    resolved_evidence: list[dict[str, Any]] = []
    for index, reference in enumerate(normalized_receipt_refs):
        expected_kind = _RELEASE_KIND if reference == policy_refs["code"] else _SOURCE_BUNDLE_KIND
        resolved_evidence.append(
            _resolve_exact_artifact(
                reference,
                artifact_resolver=artifact_resolver,
                label=f"Bootstrap intrinsic evidence_refs[{index}]",
                expected_kinds=frozenset({expected_kind}),
            )
        )

    bootstrap_leaves = {
        role: _resolve_bootstrap_bundle_leaf(
            policy_refs[role],
            outer_role=role,
            inner_role=inner_role,
            artifact_resolver=artifact_resolver,
            source_resolver=source_resolver,
        )
        for role, inner_role in _BOOTSTRAP_SOURCE_BUNDLE_ROLES.items()
    }
    source_root_inputs = {
        **bootstrap_leaves,
        **primary_source_leaves,
    }
    _require_shared_source_root(source_root_inputs)

    for role, primary_role in (
        ("exchange_calendar", "exchange_calendar"),
        ("market", "market"),
        ("pit_universe", "pit_universe"),
    ):
        _same_ref(
            bootstrap_leaves[role]["ref"],
            primary_source_leaves[primary_role]["ref"],
            label=f"Bootstrap {role} source object ref",
        )

    decision_leaf = bootstrap_leaves["decision_source"]
    _require_exact_generated_json(
        decision_leaf,
        expected=_DECISION_DOCUMENT,
        expected_source_object_id="factor-bootstrap-decision",
        expected_relative_path="operations/unified_cutover/bootstrap-decision.json",
        label="decision",
    )
    if _sha256(decision_leaf["raw"]) != policy["payload"].get("decision_source_sha256"):
        raise FactorGovernanceError("Bootstrap decision source SHA differs")

    expected_implementation_rows = installed_implementation_rows(
        implementation_component_refs=implementation_component_refs
    )
    _require_exact_generated_json(
        bootstrap_leaves["implementation"],
        expected={
            "domain": "myquant-bootstrap-implementation-tree-manifest",
            "implementation_rows": expected_implementation_rows,
        },
        expected_source_object_id="factor-bootstrap-implementation",
        expected_relative_path="bootstrap/implementation-tree.json",
        label="implementation tree",
    )
    resolved_evidence_by_ref = {
        _ref_sort_key(_artifact_ref(document)): document for document in resolved_evidence
    }
    source_artifacts = {
        role: resolved_evidence_by_ref[_ref_sort_key(reference)]
        for role, reference in policy_refs.items()
    }
    rebuilt_policy = build_bootstrap_exception_evidence(
        decision_source_bytes=decision_leaf["raw"],
        source_artifacts=source_artifacts,
        implementation_source_sha256=bootstrap_leaves["implementation"]["payload"]["byte_sha256"],
        created_at=policy["created_at"],
    )
    if canonical_json_bytes(rebuilt_policy) != canonical_json_bytes(policy):
        raise FactorGovernanceError("Bootstrap policy exact replay differs")
    rebuilt_receipt = _build_factor_validation_receipt(
        policy=policy,
        active_set=active,
        evidence_artifacts=resolved_evidence,
        trusted_at=attestation["created_at"],
    )
    if canonical_json_bytes(rebuilt_receipt) != canonical_json_bytes(attestation):
        raise FactorGovernanceError("Bootstrap intrinsic receipt exact replay differs")

    expected_recomputation = {
        "authority": "NON_AUTHORIZING",
        "domain": "myquant-bootstrap-recomputation",
        "result": "EXACT_MATCH",
        "recomputation": dict(recomputation),
        "source_sha256s": {
            "exchange_calendar": primary_source_leaves["exchange_calendar"]["payload"][
                "byte_sha256"
            ],
            "market_history": primary_source_leaves["market"]["payload"]["byte_sha256"],
            "pit_universe": primary_source_leaves["pit_universe"]["payload"]["byte_sha256"],
        },
    }
    _require_exact_generated_json(
        bootstrap_leaves["recomputation"],
        expected=expected_recomputation,
        expected_source_object_id="factor-bootstrap-recomputation",
        expected_relative_path="bootstrap/recomputation.json",
        label="recomputation",
    )

    source_generation_rows = [
        {
            "role": role,
            "source_ref": primary_source_leaves[role]["ref"],
            "source_byte_sha256": primary_source_leaves[role]["payload"]["byte_sha256"],
        }
        for role in ("exchange_calendar", "market", "pit_universe")
    ]
    source_generation_rows.sort(key=lambda row: row["role"])
    source_generation_body = {
        "authority": "NON_AUTHORIZING",
        "domain": "myquant-bootstrap-source-generation",
        "reader_contract": dict(_READER_CONTRACT),
        "source_rows": source_generation_rows,
    }
    _require_exact_generated_json(
        bootstrap_leaves["source_generation"],
        expected={
            **source_generation_body,
            "generation_sha256": _sha256(canonical_json_bytes(source_generation_body)),
        },
        expected_source_object_id="factor-bootstrap-source-generation",
        expected_relative_path="bootstrap/source-generation.json",
        label="source generation",
    )


def _validate_deep_factor_policy(
    payload: Mapping[str, Any],
    *,
    artifact_resolver: ArtifactResolver,
    source_resolver: SourceResolver,
    expected_code_manifest_sha256: str,
    primary_source_leaves: Mapping[str, Mapping[str, Any]],
    recomputation: Mapping[str, Any],
) -> None:
    policy = _resolve_exact_artifact(
        payload["factor_policy_ref"],
        artifact_resolver=artifact_resolver,
        label="factor_policy_ref",
        expected_kinds=frozenset({"factor.bootstrap_exception_evidence"}),
    )
    active = _resolve_exact_artifact(
        payload["factor_active_set_ref"],
        artifact_resolver=artifact_resolver,
        label="factor_active_set_ref",
        expected_kinds=frozenset({"factor.bootstrap_set"}),
    )
    attestation = _resolve_exact_artifact(
        payload["factor_validation_attestation_ref"],
        artifact_resolver=artifact_resolver,
        label="factor_validation_attestation_ref",
        expected_kinds=frozenset({"factor.validation_receipt"}),
    )
    try:
        validate_bootstrap_exception_evidence(policy)
        active_validated = validate_bootstrap_factor_set(active)
        attestation_validated = validate_factor_validation_receipt(attestation)
    except (ContractError, FactorGovernanceError) as exc:
        raise FactorGovernanceError("Factor policy/active-set/attestation replay failed") from exc
    _same_ref(
        active_validated["payload"]["bootstrap_exception_evidence_ref"],
        payload["factor_policy_ref"],
        label="Factor active set policy ref",
    )
    _same_ref(
        attestation_validated["payload"]["policy_ref"],
        payload["factor_policy_ref"],
        label="Factor validation policy ref",
    )
    _same_ref(
        attestation_validated["payload"]["active_set_ref"],
        payload["factor_active_set_ref"],
        label="Factor validation active-set ref",
    )
    expected_implementation_ids = {
        installed_semantic_row(factor_id)["implementation_id"]
        for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80)
    }
    implementation_ids: set[str] = set()
    implementation_component_refs: dict[str, dict[str, str]] = {}
    for index, ref in enumerate(payload["factor_implementation_refs"]):
        implementation = _resolve_exact_artifact(
            ref,
            artifact_resolver=artifact_resolver,
            label=f"factor_implementation_refs[{index}]",
            expected_kinds=frozenset({"system.installed_component_manifest"}),
        )
        try:
            implementation_payload = validate_installed_component_manifest(implementation)[
                "payload"
            ]
        except Exception as exc:
            raise FactorGovernanceError("Factor implementation semantic replay failed") from exc
        if (
            implementation_payload.get("component_role") != "SOURCE_IMPLEMENTATION"
            or implementation_payload.get("release_manifest_ref") != payload["deployed_release_ref"]
            or implementation_payload.get("installed_code_manifest_sha256")
            != expected_code_manifest_sha256
        ):
            raise FactorGovernanceError("Factor implementation release closure differs")
        implementation_id = str(implementation_payload.get("component_id"))
        implementation_ids.add(implementation_id)
        factor_ids = [
            factor_id
            for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80)
            if installed_semantic_row(factor_id)["implementation_id"] == implementation_id
        ]
        if len(factor_ids) != 1 or factor_ids[0] in implementation_component_refs:
            raise FactorGovernanceError("Factor implementation identities differ")
        implementation_component_refs[factor_ids[0]] = _artifact_ref(implementation)
    if implementation_ids != expected_implementation_ids or set(implementation_component_refs) != {
        LOW_DOLLAR_VOLUME,
        BLEND_W80,
    }:
        raise FactorGovernanceError("Factor implementation identities differ")
    _validate_deep_bootstrap_receipt_closure(
        payload=payload,
        policy=policy,
        active=active_validated,
        attestation=attestation_validated,
        implementation_component_refs=implementation_component_refs,
        artifact_resolver=artifact_resolver,
        source_resolver=source_resolver,
        primary_source_leaves=primary_source_leaves,
        recomputation=recomputation,
    )


def _ref_sort_key(ref: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    normalized = validate_object_ref(ref, label="Factor source ref")
    return (
        normalized["kind"],
        normalized["contract_sha256"],
        normalized["artifact_id"],
        normalized["semantic_sha256"],
        normalized["byte_sha256"],
    )


def _exact_unique_refs(values: Sequence[Mapping[str, Any]], *, label: str) -> list[dict[str, str]]:
    refs = [
        validate_object_ref(value, label=f"{label}[{index}]") for index, value in enumerate(values)
    ]
    keys = [_ref_sort_key(value) for value in refs]
    if len(keys) != len(set(keys)):
        raise FactorGovernanceError(f"{label} contains duplicate refs")
    return [dict(value) for value in sorted(refs, key=_ref_sort_key)]


def _validate_deep_calendar(  # noqa: C901
    payload: Mapping[str, Any],
    *,
    artifact_resolver: ArtifactResolver,
    calendar_leaves: Sequence[Mapping[str, Any]],
    as_of: str,
    pit_symbols: Sequence[str],
    market_session_dates: Sequence[str],
) -> dict[str, Any]:
    """Re-run the native official/Tushare calendar compilation verifier."""

    policy = _resolve_exact_artifact(
        payload["calendar_authority_policy_ref"],
        artifact_resolver=artifact_resolver,
        label="calendar_authority_policy_ref",
        expected_kinds=frozenset({_CALENDAR_POLICY_KIND}),
    )
    compilation = _resolve_exact_artifact(
        payload["calendar_compilation_ref"],
        artifact_resolver=artifact_resolver,
        label="calendar_compilation_ref",
        expected_kinds=_CALENDAR_COMPILATION_KINDS,
    )
    policy_payload = policy["payload"]
    compilation_payload = compilation["payload"]
    if (
        compilation_payload.get("policy_ref") != payload["calendar_authority_policy_ref"]
        or policy_payload.get("expected_compilation_kind") != compilation["kind"]
        or _as_of(str(compilation_payload.get("cutoff_date", "")).replace("-", "")) != as_of
    ):
        raise FactorGovernanceError("Factor calendar policy/compilation closure differs")
    exchanges = sorted({symbol.rsplit(".", 1)[1] for symbol in pit_symbols})
    expected_exchange_ids = {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}
    if not set(exchanges) <= set(expected_exchange_ids):
        raise FactorGovernanceError("Factor PIT exchange identity differs")
    expected = [expected_exchange_ids[value] for value in exchanges]
    compiled = compilation_payload.get("pit_exchange_ids")
    if compiled is None:
        compiled = compilation_payload.get("projection_source_exchange_ids")
    if compilation["kind"] == "system.trusted_provider_calendar_compilation":
        if compiled != ["BSE", "SSE", "SZSE"] or not set(expected) <= set(compiled):
            raise FactorGovernanceError("Factor calendar/PIT exchange scope differs")
    elif compiled != expected:
        raise FactorGovernanceError("Factor calendar/PIT exchange scope differs")
    try:
        validate_calendar_authority_policy(
            policy,
            pit_exchange_ids=(
                compiled
                if compilation["kind"] == "system.trusted_provider_calendar_compilation"
                else expected
            ),
        )
    except Exception as exc:
        raise FactorGovernanceError("Factor calendar authority policy replay failed") from exc
    if not market_session_dates or list(market_session_dates) != sorted(set(market_session_dates)):
        raise FactorGovernanceError("Factor Market session projection differs")

    def raw_resolver(file_ref: Mapping[str, Any]) -> bytes:
        return _leaf_for_file_ref(
            calendar_leaves,
            file_ref,
            label="calendar capture raw",
        )["raw"]

    try:
        if compilation["kind"] == "system.trusted_provider_calendar_compilation":
            capability = _resolve_exact_artifact(
                compilation_payload["provider_capability_ref"],
                artifact_resolver=artifact_resolver,
                label="trusted calendar capability",
                expected_kinds=frozenset({"system.trusted_provider_calendar_capability"}),
            )
            captures = [
                _resolve_exact_artifact(
                    ref,
                    artifact_resolver=artifact_resolver,
                    label=f"trusted calendar capture[{index}]",
                    expected_kinds=frozenset({"system.trusted_provider_calendar_capture"}),
                )
                for index, ref in enumerate(compilation_payload["provider_capture_refs"])
            ]
            docs_raw = raw_resolver(capability["payload"]["docs_raw_file_ref"])
            replayed = validate_trusted_provider_calendar_compilation(
                compilation,
                policy=policy,
                capability=capability,
                capture_documents=captures,
                docs_raw=docs_raw,
                raw_resolver=raw_resolver,
                expected_release_ref=payload["deployed_release_ref"],
                pit_exchange_ids=compiled,
                market_session_dates=market_session_dates,
            )
        else:
            captures = [
                _resolve_exact_artifact(
                    ref,
                    artifact_resolver=artifact_resolver,
                    label=f"official calendar capture[{index}]",
                    expected_kinds=frozenset({"system.exchange_calendar_capture"}),
                )
                for index, ref in enumerate(compilation_payload["source_capture_refs"])
            ]
            admissions = [
                _resolve_exact_artifact(
                    ref,
                    artifact_resolver=artifact_resolver,
                    label=f"official calendar admission[{index}]",
                    expected_kinds=frozenset({"system.exchange_calendar_decoder_admission"}),
                )
                for index, ref in enumerate(compilation_payload["decoder_admission_refs"])
            ]
            indexes = [
                _resolve_exact_artifact(
                    ref,
                    artifact_resolver=artifact_resolver,
                    label=f"official calendar index closure[{index}]",
                    expected_kinds=frozenset({"system.exchange_calendar_index_closure"}),
                )
                for index, ref in enumerate(compilation_payload["index_closure_refs"])
            ]
            replayed = validate_exchange_calendar_compilation(
                compilation,
                pit_exchange_ids=expected,
                market_session_dates=market_session_dates,
                capture_documents=captures,
                admission_documents=admissions,
                index_closure_documents=indexes,
                raw_resolver=raw_resolver,
                expected_release_ref=payload["deployed_release_ref"],
                expected_policy_ref=payload["calendar_authority_policy_ref"],
            )
    except Exception as exc:
        raise FactorGovernanceError("Factor calendar compilation raw replay failed") from exc
    if canonical_json_bytes(replayed) != canonical_json_bytes(compilation):
        raise FactorGovernanceError("Factor calendar compilation replay identity differs")
    coverage_start = compilation_payload.get("coverage_start_date")
    if type(coverage_start) is not str or coverage_start > "2024-01-01":
        raise FactorGovernanceError("Factor calendar coverage start differs")
    return _leaf_for_file_ref(
        calendar_leaves,
        compilation_payload["calendar_parquet_file_ref"],
        label="compiled strict calendar parquet",
        source_format="PARQUET",
    )


def _validate_calendar_custody_chain(  # noqa: C901
    *,
    source_payload: Mapping[str, Any],
    custody: Mapping[str, Any],
    calendar_leaves: Sequence[Mapping[str, Any]],
    artifact_resolver: ArtifactResolver,
) -> None:
    """Cross-bind copied Calendar JSON bytes to transaction/execution/success."""

    custody_payload = custody["payload"]
    resolved: dict[str, dict[str, Any]] = {}
    expected = {
        "transaction": (
            "capture_transaction_ref",
            "system.trusted_provider_calendar_capture_transaction",
            "capture-transaction.json",
        ),
        "execution": (
            "capture_execution_ref",
            "system.trusted_provider_calendar_capture_execution",
            "capture-execution.json",
        ),
        "success": (
            "capture_success_ref",
            "system.trusted_provider_calendar_capture_success",
            "capture-success.json",
        ),
    }
    manifest_by_name = {
        str(row["relative_path"]).rsplit("/", 1)[-1]: row
        for row in custody_payload["published_leaf_manifest"]
    }
    for label, (field, kind, leaf_name) in expected.items():
        artifact = _resolve_exact_artifact(
            custody_payload[field],
            artifact_resolver=artifact_resolver,
            label=f"Calendar custody {field}",
            expected_kinds=frozenset({kind}),
        )
        row = manifest_by_name.get(leaf_name)
        if row is None:
            raise FactorGovernanceError(f"Calendar custody {leaf_name} is absent")
        leaf = _leaf_for_file_ref(
            calendar_leaves,
            {"relative_path": row["relative_path"], "byte_sha256": row["byte_sha256"]},
            label=f"Calendar custody {leaf_name}",
            source_format="JSON",
        )
        if canonical_json_bytes(artifact) != leaf["raw"] or len(leaf["raw"]) != row["size"]:
            raise FactorGovernanceError(f"Calendar custody {leaf_name} bytes differ")
        resolved[label] = artifact
    transaction_payload = resolved["transaction"]["payload"]
    execution_payload = resolved["execution"]["payload"]
    success_payload = resolved["success"]["payload"]
    transaction_manifest_ref = {
        "relative_path": manifest_by_name["capture-transaction.json"]["relative_path"],
        "byte_sha256": manifest_by_name["capture-transaction.json"]["byte_sha256"],
    }
    execution_manifest_ref = {
        "relative_path": manifest_by_name["capture-execution.json"]["relative_path"],
        "byte_sha256": manifest_by_name["capture-execution.json"]["byte_sha256"],
    }
    published_manifest_refs = sorted(
        [
            {"relative_path": row["relative_path"], "byte_sha256": row["byte_sha256"]}
            for name, row in manifest_by_name.items()
            if name != "capture-success.json"
        ],
        key=lambda row: row["relative_path"],
    )
    if (
        execution_payload["capture_transaction_file_ref"] != transaction_manifest_ref
        or success_payload["capture_transaction_file_ref"] != transaction_manifest_ref
        or success_payload["capture_execution_file_ref"] != execution_manifest_ref
        or success_payload["published_leaf_file_refs"] != published_manifest_refs
    ):
        raise FactorGovernanceError("Calendar custody manifest authority refs differ")
    input_leaf = _leaf_for_file_ref(
        calendar_leaves,
        execution_payload["release_install_input_file_ref"],
        label="Calendar release-install input",
        source_format="JSON",
    )
    _same_ref(
        input_leaf["ref"],
        source_payload["release_install_input_source_ref"],
        label="Calendar release-install input source ref",
    )
    validate_trusted_provider_calendar_capture_transaction(
        resolved["transaction"],
        documentation_raw_file_ref=transaction_payload["documentation_raw_file_ref"],
        capability_file_ref=transaction_payload["capability_file_ref"],
        policy_file_ref=transaction_payload["policy_file_ref"],
        provider_raw_file_refs=transaction_payload["provider_raw_file_refs"],
        provider_capture_file_refs=transaction_payload["provider_capture_file_refs"],
    )
    validate_trusted_provider_calendar_capture_execution(
        resolved["execution"],
        release_install_input_raw=input_leaf["raw"],
        documentation_raw_file_ref=execution_payload["documentation_raw_file_ref"],
        capability_file_ref=execution_payload["capability_file_ref"],
        policy_file_ref=execution_payload["policy_file_ref"],
        provider_raw_file_refs=execution_payload["provider_raw_file_refs"],
        provider_capture_file_refs=execution_payload["provider_capture_file_refs"],
        capture_transaction_file_ref=execution_payload["capture_transaction_file_ref"],
        historical=True,
    )
    validate_trusted_provider_calendar_capture_success(
        resolved["success"],
        capture_transaction_file_ref=success_payload["capture_transaction_file_ref"],
        capture_execution_file_ref=success_payload["capture_execution_file_ref"],
        published_leaf_file_refs=success_payload["published_leaf_file_refs"],
    )
    if (
        execution_payload["deployed_release_ref"] != source_payload["deployed_release_ref"]
        or execution_payload["release_install_evidence_ref"]
        != source_payload["release_install_evidence_ref"]
        or success_payload["capture_transaction_file_ref"]
        != execution_payload["capture_transaction_file_ref"]
    ):
        raise FactorGovernanceError("Calendar custody internal release/linkage differs")


def _deep_replay_source_closure(  # noqa: C901
    artifact: Mapping[str, Any],
    *,
    artifact_resolver: ArtifactResolver,
    source_resolver: SourceResolver,
    validation_mode: str = HISTORICAL_RECOVERY,
    current_release_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve and re-run the Factor closure without reading System authority."""

    payload = artifact["payload"]
    release = _resolve_exact_artifact(
        payload["deployed_release_ref"],
        artifact_resolver=artifact_resolver,
        label="deployed_release_ref",
        expected_kinds=frozenset({_RELEASE_KIND}),
    )
    if release["payload"].get("state") != "OPERATIONAL":
        raise FactorGovernanceError("Factor deployed release is not OPERATIONAL")
    install = _resolve_exact_artifact(
        payload["release_install_evidence_ref"],
        artifact_resolver=artifact_resolver,
        label="release_install_evidence_ref",
        expected_kinds=frozenset({"system.release_install_evidence"}),
    )
    try:
        install = validate_release_install_evidence(install)
    except Exception as exc:
        raise FactorGovernanceError("Factor release-install evidence replay failed") from exc
    install_payload = install["payload"]
    release_payload = release["payload"]
    verification = _validate_release_install_verification(payload["release_install_verification"])
    _same_ref(
        install_payload["release_ref"],
        payload["deployed_release_ref"],
        label="release-install deployed release ref",
    )
    if (
        install_payload.get("installed_code_manifest_sha256")
        != release_payload.get("code_manifest_sha256")
        or install_payload.get("git_code_manifest_sha256")
        != release_payload.get("code_manifest_sha256")
        or (install_payload.get("wheel") or {}).get("byte_sha256")
        != release_payload.get("wheel_sha256")
        or release_payload.get("code_sha256") != install_payload.get("code_tree_sha256")
        or verification["release_ref"] != payload["deployed_release_ref"]
        or verification["source_archive_sha256"] != install_payload["source_archive"]["byte_sha256"]
        or verification["wheel_sha256"] != install_payload["wheel"]["byte_sha256"]
        or verification["code_tree_sha256"] != install_payload["code_tree_sha256"]
        or verification["installed_code_manifest_sha256"]
        != install_payload["installed_code_manifest_sha256"]
        or verification["contract_catalog_sha256"] != install_payload["contract_catalog_sha256"]
        or verification["import_origin"] != install_payload["import_origin"]
    ):
        raise FactorGovernanceError("Factor release/install semantic identity differs")
    custody = _resolve_exact_artifact(
        payload["calendar_capture_custody_attestation_ref"],
        artifact_resolver=artifact_resolver,
        label="calendar_capture_custody_attestation_ref",
        expected_kinds=frozenset({FACTOR_PRODUCTION_CALENDAR_CUSTODY_KIND}),
    )
    custody = validate_factor_calendar_capture_custody_attestation(custody)
    _same_ref(
        custody["payload"]["deployed_release_ref"],
        payload["deployed_release_ref"],
        label="Calendar custody deployed release ref",
    )
    selection = _resolve_exact_artifact(
        payload["market_pit_selection_ref"],
        artifact_resolver=artifact_resolver,
        label="market_pit_selection_ref",
        expected_kinds=frozenset({_MARKET_PIT_SELECTION_KIND}),
    )
    legacy = _resolve_exact_artifact(
        payload["legacy_zero_call_ref"],
        artifact_resolver=artifact_resolver,
        label="legacy_zero_call_ref",
        expected_kinds=frozenset({FACTOR_LEGACY_ZERO_CALL_CERTIFICATE_KIND}),
    )
    legacy_payload = validate_factor_legacy_zero_call_certificate(legacy)["payload"]
    if (
        legacy_payload["final_commit"] != install_payload["final_commit"]
        or legacy_payload["final_tree"] != install_payload["final_tree"]
    ):
        raise FactorGovernanceError("Factor legacy certificate release source differs")
    market_input = _resolve_exact_artifact(
        payload["market_input_ref"],
        artifact_resolver=artifact_resolver,
        label="market_input_ref",
        expected_kinds=frozenset({FACTOR_PRODUCTION_MARKET_INPUT_KIND}),
    )
    _inputs, branches = _factor_source_topology(
        payload["factor_source_bundle_ref"],
        artifact_resolver=artifact_resolver,
        source_resolver=source_resolver,
    )
    if validation_mode not in {PRE_CAS_CURRENT, HISTORICAL_RECOVERY}:
        raise FactorGovernanceError("Factor release validation mode differs")
    if (validation_mode == PRE_CAS_CURRENT) != (current_release_root is not None):
        raise FactorGovernanceError("Factor current release root boundary differs")
    release_input_leaf = _resolved_source_leaf(
        payload["release_install_input_source_ref"],
        artifact_resolver=artifact_resolver,
        source_resolver=source_resolver,
        label="Factor release-install input source",
    )
    release_input = _canonical_json_mapping(
        release_input_leaf["raw"], label="Factor release-install input"
    )
    if (
        set(release_input) != {"release_install_evidence", "deployed_release"}
        or _artifact_ref(
            validate_release_install_evidence(release_input["release_install_evidence"])
        )
        != payload["release_install_evidence_ref"]
        or _artifact_ref(
            validate_artifact(release_input["deployed_release"], expected_kind=_RELEASE_KIND)
        )
        != payload["deployed_release_ref"]
    ):
        raise FactorGovernanceError("Factor release-install input artifact closure differs")
    if validation_mode == PRE_CAS_CURRENT:
        if current_release_root is None:
            raise FactorGovernanceError("Factor current release root is absent")
        try:
            fresh_verification = verify_running_release_install_input(
                release_input_leaf["raw"], repository_root=current_release_root
            )
        except Exception as exc:
            raise FactorGovernanceError(
                "Factor current release-install verification failed"
            ) from exc
        if _validate_release_install_verification(fresh_verification) != verification:
            raise FactorGovernanceError("Factor current release-install verification drifted")
    market_input_payload = market_input["payload"]
    pointer_leaf = _resolved_source_leaf(
        market_input_payload["market_pointer_source_ref"],
        artifact_resolver=artifact_resolver,
        source_resolver=source_resolver,
        label="Factor Market pointer source",
    )
    manifest_leaf = _resolved_source_leaf(
        market_input_payload["market_snapshot_manifest_source_ref"],
        artifact_resolver=artifact_resolver,
        source_resolver=source_resolver,
        label="Factor Market manifest source",
    )
    if (
        pointer_leaf["payload"].get("source_format") != "JSON"
        or manifest_leaf["payload"].get("source_format") != "JSON"
    ):
        raise FactorGovernanceError("Factor Market pointer/manifest source format differs")
    selection, selected_files = _deep_replay_market_pit_selection(
        selection,
        market_leaves=[pointer_leaf, manifest_leaf],
        pit_leaves=branches["pit_universe"],
    )
    market_leaf = _leaf_for_terminal_role(
        branches["market_history"],
        "factor-market-history",
        label="Factor Market projection",
        source_format="PARQUET",
    )
    scope_leaf = _leaf_for_terminal_role(
        branches["market_history"],
        "market-scope",
        label="Market expected scope",
        source_format="JSON",
    )
    _same_ref(
        scope_leaf["ref"],
        payload["market_scope_source_ref"],
        label="Market scope source ref",
    )
    scope_symbols = _market_scope_symbols(scope_leaf["raw"])
    _require_market_scope_sha256(
        scope_symbols,
        selection["payload"]["market_expected_scope_sha256"],
    )
    market_pointer_payload = _canonical_json_mapping(pointer_leaf["raw"], label="Market pointer")
    coverage = market_pointer_payload.get("coverage")
    if (
        type(coverage) is not dict
        or coverage.get("complete") is not True
        or coverage.get("coverage_ratio") != 1.0
        or coverage.get("blocking_incomplete_count") != 0
        or coverage.get("categories_checked") != ["full_a"]
        or coverage.get("classification_sets_disjoint") is not True
        or coverage.get("true_missing_symbols") != []
        or coverage.get("expected_scope_count") != len(scope_symbols)
        or coverage.get("coverage_complete_count") != len(scope_symbols)
    ):
        raise FactorGovernanceError("Market expected scope coverage is not clean full-A closure")
    absent = coverage.get("non_blocking_absent_symbols", [])
    if (
        type(absent) is not list
        or any(type(value) is not str or _SYMBOL_RE.fullmatch(value) is None for value in absent)
        or absent != sorted(set(absent), key=lambda value: value.encode("utf-8"))
        or not set(absent) <= set(scope_symbols)
    ):
        raise FactorGovernanceError("Market non-blocking absent symbols differ")
    _deep_replay_market_input(
        market_input,
        selection=selection,
        selected_files=selected_files,
        market_leaf=market_leaf,
        market_scope_leaf=scope_leaf,
    )
    canonical_pit_leaf = selected_files["pit_membership"]
    pit_leaf = _leaf_for_terminal_role(
        branches["pit_universe"],
        "factor-pit-universe",
        label="Factor PIT projection",
        source_format="PARQUET",
    )
    as_of = _as_of(selection["payload"]["as_of"])
    pit = _read_table_raw(
        pit_leaf["raw"],
        role="pit_universe",
        columns=["signal_session", "symbol", "industry", "total_mv", "tradable"],
        expected_sha256=pit_leaf["payload"]["byte_sha256"],
    )
    projected_symbols = sorted(
        [str(value) for value in pit["symbol"]], key=lambda value: value.encode("utf-8")
    )
    canonical_symbols = _canonical_pit_symbols(canonical_pit_leaf["raw"])
    _require_market_pit_scope_relation(
        market_scope_symbols=scope_symbols,
        canonical_pit_symbols=canonical_symbols,
        factor_projection_symbols=projected_symbols,
    )
    tradable_symbols = sorted(
        [str(row["symbol"]) for row in pit.to_dict(orient="records") if row["tradable"] is True],
        key=lambda value: value.encode("utf-8"),
    )
    expected_tradable = sorted(
        set(scope_symbols) - set(absent), key=lambda value: value.encode("utf-8")
    )
    if tradable_symbols != expected_tradable:
        raise FactorGovernanceError(
            "Factor PIT tradable cohort differs from Market observed cohort"
        )
    market = _read_table_raw(
        market_leaf["raw"],
        role="market_history",
        columns=["trade_date", "symbol", "adj_close", "amount", "vol"],
        expected_sha256=market_leaf["payload"]["byte_sha256"],
    )
    cutoff = datetime.strptime(as_of, "%Y%m%d").date()
    if any(
        not isinstance(value, date) or isinstance(value, datetime) for value in market["trade_date"]
    ):
        raise FactorGovernanceError("Factor Market session type differs")
    market_sessions = sorted({value for value in market["trade_date"] if value <= cutoff})
    if (
        not market_sessions
        or market_sessions[-1] != cutoff
        or any(value > cutoff for value in market["trade_date"])
    ):
        raise FactorGovernanceError("Factor Market session cutoff differs")
    cutoff_symbols = sorted(
        set(str(value) for value in market.loc[market["trade_date"].eq(cutoff), "symbol"]),
        key=lambda value: value.encode("utf-8"),
    )
    if cutoff_symbols != expected_tradable:
        raise FactorGovernanceError(
            "Factor Market cutoff cohort differs from Factor PIT tradable cohort"
        )
    preliminary_eligible = _eligible_pit(pit, signal_session=cutoff)
    calendar_leaf = _validate_deep_calendar(
        payload,
        artifact_resolver=artifact_resolver,
        calendar_leaves=branches["exchange_calendar"],
        as_of=as_of,
        pit_symbols=preliminary_eligible,
        market_session_dates=[value.isoformat() for value in market_sessions],
    )
    capture_manifest = custody["payload"]["published_leaf_manifest"]
    expected_capture_rows = {
        (row["relative_path"], row["byte_sha256"], row["size"]) for row in capture_manifest
    }
    observed_capture_rows = {
        (
            str(leaf["payload"].get("relative_path", "")),
            leaf["payload"]["byte_sha256"],
            len(leaf["raw"]),
        )
        for leaf in branches["exchange_calendar"]
        if str(leaf["payload"].get("relative_path", ""))
        in {row[0] for row in expected_capture_rows}
    }
    if observed_capture_rows != expected_capture_rows:
        raise FactorGovernanceError(
            "Calendar copied source custody differs from ingress attestation"
        )
    _validate_calendar_custody_chain(
        source_payload=payload,
        custody=custody,
        calendar_leaves=branches["exchange_calendar"],
        artifact_resolver=artifact_resolver,
    )
    calendar = _read_table_raw(
        calendar_leaf["raw"],
        role="exchange_calendar",
        columns=["ordinal", "open_session", "opens_at_utc", "closes_at_utc"],
        expected_sha256=calendar_leaf["payload"]["byte_sha256"],
    )
    all_sessions = _canonical_sessions(calendar, as_of=as_of)
    if len(all_sessions) < 391:
        raise FactorGovernanceError("Factor calendar has fewer than 391 open sessions")
    replay = _recompute_loaded_sources(
        calendar=calendar,
        pit=pit,
        market=market,
        as_of=as_of,
        exchange_calendar_sha256=calendar_leaf["payload"]["byte_sha256"],
        pit_universe_sha256=pit_leaf["payload"]["byte_sha256"],
        market_history_sha256=market_leaf["payload"]["byte_sha256"],
    )
    _validate_deep_factor_policy(
        payload,
        artifact_resolver=artifact_resolver,
        source_resolver=source_resolver,
        expected_code_manifest_sha256=install_payload["installed_code_manifest_sha256"],
        primary_source_leaves={
            "release_install_input": release_input_leaf,
            "market_scope": scope_leaf,
            "exchange_calendar": calendar_leaf,
            "market": market_leaf,
            "pit_universe": pit_leaf,
        },
        recomputation=replay,
    )
    return replay


def _validate_recomputation_values(  # noqa: C901
    values: Mapping[str, Any],
) -> dict[str, Any]:
    active_rows, control_rows = _expected_factor_policy_rows()
    statistics = _validate_signal_statistics(values.get("signal_statistics"))
    if (
        values.get("active_factor_rows") != active_rows
        or values.get("control_rows") != control_rows
    ):
        raise FactorGovernanceError("Factor active/control policy rows differ")
    by_factor = {row["factor_id"]: row for row in statistics}
    signal_values = values.get("signal_values")
    if type(signal_values) is not dict or set(signal_values) != {
        LOW_DOLLAR_VOLUME,
        BLEND_W80,
    }:
        raise FactorGovernanceError("Factor immutable signal values differ")
    normalized_signals: dict[str, dict[str, str]] = {}
    for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80):
        projection = signal_values[factor_id]
        if type(projection) is not dict or not projection:
            raise FactorGovernanceError("Factor immutable signal projection is empty")
        ordered: dict[str, str] = {}
        for symbol, hexadecimal in sorted(
            projection.items(), key=lambda row: str(row[0]).encode("utf-8")
        ):
            if (
                type(symbol) is not str
                or _SYMBOL_RE.fullmatch(symbol) is None
                or type(hexadecimal) is not str
            ):
                raise FactorGovernanceError("Factor immutable signal projection differs")
            try:
                numeric = float.fromhex(hexadecimal)
            except ValueError as exc:
                raise FactorGovernanceError("Factor immutable signal encoding differs") from exc
            if not math.isfinite(numeric) or numeric.hex() != hexadecimal:
                raise FactorGovernanceError("Factor immutable signal encoding differs")
            ordered[symbol] = hexadecimal
        if list(projection) != list(ordered):
            raise FactorGovernanceError("Factor immutable signal order differs")
        if _sha256(canonical_json_bytes(ordered)) != by_factor[factor_id]["signal_sha256"]:
            raise FactorGovernanceError("Factor immutable signal SHA differs")
        if (
            _sha256(canonical_json_bytes(list(ordered)))
            != by_factor[factor_id]["signal_symbol_set_sha256"]
        ):
            raise FactorGovernanceError("Factor immutable signal symbol set differs")
        normalized_signals[factor_id] = ordered
    if (
        _sha(values.get("low_signal_sha256"), label="low signal SHA")
        != by_factor[LOW_DOLLAR_VOLUME]["signal_sha256"]
        or _sha(values.get("w80_signal_sha256"), label="w80 signal SHA")
        != by_factor[BLEND_W80]["signal_sha256"]
        or values.get("fundamental_dependency_state") != FUNDAMENTAL_NOT_USED
        or values.get("fundamental_freshness_policy") != FUNDAMENTAL_ADVISORY
    ):
        raise FactorGovernanceError("Factor recomputation source policy differs")
    replay_sha = _sha(values.get("exact_replay_sha256"), label="replay SHA")
    return {
        "as_of": _as_of(values.get("as_of")),
        "low_signal_sha256": by_factor[LOW_DOLLAR_VOLUME]["signal_sha256"],
        "w80_signal_sha256": by_factor[BLEND_W80]["signal_sha256"],
        "signal_statistics": statistics,
        "signal_values": normalized_signals,
        "active_factor_rows": active_rows,
        "control_rows": control_rows,
        "exact_replay_sha256": replay_sha,
    }


def build_factor_production_recomputation_evidence(
    *,
    source_closure: Mapping[str, Any],
    deployed_release_ref: Mapping[str, Any],
    factor_active_set_ref: Mapping[str, Any],
    recomputation: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    source = validate_factor_production_source_closure(source_closure)
    source_payload = source["payload"]
    _same_ref(
        deployed_release_ref,
        source_payload["deployed_release_ref"],
        label="Factor recomputation deployed release ref",
    )
    _same_ref(
        factor_active_set_ref,
        source_payload["factor_active_set_ref"],
        label="Factor recomputation active-set ref",
    )
    values = _validate_recomputation_values(recomputation)
    body = {
        "state": "VERIFIED",
        "activation_scope": FACTOR_PRODUCTION_SCOPE,
        "admission_route": "BOOTSTRAP_EXCEPTION",
        "producer_identity": "NOT_CLAIMED",
        "as_of": values["as_of"],
        "source_closure_ref": _artifact_ref(source),
        "deployed_release_ref": validate_object_ref(deployed_release_ref),
        "factor_active_set_ref": validate_object_ref(factor_active_set_ref),
        "low_signal_sha256": values["low_signal_sha256"],
        "w80_signal_sha256": values["w80_signal_sha256"],
        "signal_statistics": values["signal_statistics"],
        "signal_values": values["signal_values"],
        "active_factor_rows": values["active_factor_rows"],
        "control_rows": values["control_rows"],
        "exact_replay_sha256": values["exact_replay_sha256"],
        "fundamental_dependency_state": FUNDAMENTAL_NOT_USED,
        "fundamental_freshness_policy": FUNDAMENTAL_ADVISORY,
    }
    identity = "factor-production-recompute-" + _sha256(canonical_json_bytes(body))
    artifact = seal_artifact(
        FACTOR_PRODUCTION_RECOMPUTATION_KIND,
        {"factor_production_recomputation_id": identity, **body},
        created_at=created_at,
        contract_sha256=_contract_sha(FACTOR_PRODUCTION_RECOMPUTATION_KIND),
    )
    return validate_factor_production_recomputation_evidence(artifact)


def validate_factor_production_recomputation_evidence(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    try:
        artifact = validate_artifact(
            document,
            expected_kind=FACTOR_PRODUCTION_RECOMPUTATION_KIND,
            expected_contract_sha256=_contract_sha(FACTOR_PRODUCTION_RECOMPUTATION_KIND),
        )
    except ContractError as exc:
        raise FactorGovernanceError("Factor production recomputation contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != _RECOMPUTATION_FIELDS:
        raise FactorGovernanceError("Factor production recomputation fields differ")
    if (
        payload["state"] != "VERIFIED"
        or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE
        or payload["admission_route"] != "BOOTSTRAP_EXCEPTION"
        or payload["producer_identity"] != "NOT_CLAIMED"
        or payload["fundamental_dependency_state"] != FUNDAMENTAL_NOT_USED
        or payload["fundamental_freshness_policy"] != FUNDAMENTAL_ADVISORY
    ):
        raise FactorGovernanceError("Factor production recomputation policy differs")
    # Source closure is an exact artifact ref, not a live System object ref.
    source_ref = _artifact_ref_value(payload["source_closure_ref"], label="source_closure_ref")
    if source_ref["kind"] != FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND:
        raise FactorGovernanceError("Factor recomputation source closure ref differs")
    for field in ("deployed_release_ref", "factor_active_set_ref"):
        validate_object_ref(payload[field], label=field)
    _validate_recomputation_values(payload)
    body = dict(payload)
    identity = body.pop("factor_production_recomputation_id")
    if identity != "factor-production-recompute-" + _sha256(canonical_json_bytes(body)):
        raise FactorGovernanceError("Factor recomputation identity differs")
    return artifact


def replay_factor_production_recomputation_evidence(
    document: Mapping[str, Any] | bytes,
    *,
    artifact_resolver: ArtifactResolver,
    source_resolver: SourceResolver,
    validation_mode: str = HISTORICAL_RECOVERY,
    current_release_root: str | Path | None = None,
) -> dict[str, Any]:
    """Deep-replay Factor evidence from immutable custody without System authority.

    This is the authority-time entrypoint.  It resolves the Factor closure and
    source leaves only; it neither reads a System pointer nor grants any
    System/Mainline/Investment/portfolio/trading authority.
    """

    artifact = validate_factor_production_recomputation_evidence(document)
    payload = artifact["payload"]
    source = _resolve_exact_artifact(
        payload["source_closure_ref"],
        artifact_resolver=artifact_resolver,
        label="recomputation source_closure_ref",
        expected_kinds=frozenset({FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND}),
    )
    validate_factor_production_source_closure(source)
    source_payload = source["payload"]
    for field in ("deployed_release_ref", "factor_active_set_ref"):
        _same_ref(payload[field], source_payload[field], label=f"recomputation {field}")
    replay = _deep_replay_source_closure(
        source,
        artifact_resolver=artifact_resolver,
        source_resolver=source_resolver,
        validation_mode=validation_mode,
        current_release_root=current_release_root,
    )
    expected = _validate_recomputation_values(replay)
    for field, value in expected.items():
        if payload[field] != value:
            raise FactorGovernanceError(f"Factor recomputation deep replay {field} differs")
    return artifact


def build_factor_production_generation(
    *,
    source_closure: Mapping[str, Any] | bytes,
    recomputation_evidence: Mapping[str, Any] | bytes,
    created_at: str,
) -> dict[str, Any]:
    """Build the immutable Factor-only generation after source recomputation.

    This generation is deliberately downstream of both evidence leaves and
    contains no System generation, pointer, marker, Mainline, Investment,
    portfolio, Strategy Record, broker, order, or trading authority.
    """

    source = validate_factor_production_source_closure(source_closure)
    recomputation = validate_factor_production_recomputation_evidence(recomputation_evidence)
    source_payload = source["payload"]
    recomputation_payload = recomputation["payload"]
    _same_ref(
        recomputation_payload["source_closure_ref"],
        _artifact_ref(source),
        label="Factor generation recomputation source closure ref",
    )
    for field in ("deployed_release_ref", "factor_active_set_ref"):
        _same_ref(
            recomputation_payload[field],
            source_payload[field],
            label=f"Factor generation {field}",
        )
    body = {
        "state": "OPERATIONAL",
        "activation_scope": FACTOR_PRODUCTION_SCOPE,
        "admission_route": "BOOTSTRAP_EXCEPTION",
        "producer_identity": "NOT_CLAIMED",
        "as_of": recomputation_payload["as_of"],
        "deployed_release_ref": source_payload["deployed_release_ref"],
        "release_install_evidence_ref": source_payload["release_install_evidence_ref"],
        "release_install_input_source_ref": source_payload["release_install_input_source_ref"],
        "release_install_verification": source_payload["release_install_verification"],
        "source_closure_ref": _artifact_ref(source),
        "recomputation_evidence_ref": _artifact_ref(recomputation),
        "market_pit_selection_ref": source_payload["market_pit_selection_ref"],
        "market_scope_source_ref": source_payload["market_scope_source_ref"],
        "calendar_compilation_ref": source_payload["calendar_compilation_ref"],
        "calendar_capture_custody_attestation_ref": source_payload[
            "calendar_capture_custody_attestation_ref"
        ],
        "factor_source_bundle_ref": source_payload["factor_source_bundle_ref"],
        "market_input_ref": source_payload["market_input_ref"],
        "factor_policy_ref": source_payload["factor_policy_ref"],
        "factor_active_set_ref": source_payload["factor_active_set_ref"],
        "factor_validation_attestation_ref": source_payload["factor_validation_attestation_ref"],
        "factor_implementation_refs": source_payload["factor_implementation_refs"],
        "legacy_zero_call_ref": source_payload["legacy_zero_call_ref"],
        "low_signal_sha256": recomputation_payload["low_signal_sha256"],
        "w80_signal_sha256": recomputation_payload["w80_signal_sha256"],
        "signal_statistics": recomputation_payload["signal_statistics"],
        "signal_values": recomputation_payload["signal_values"],
        "active_factor_rows": recomputation_payload["active_factor_rows"],
        "control_rows": recomputation_payload["control_rows"],
        "exact_replay_sha256": recomputation_payload["exact_replay_sha256"],
        "fundamental_dependency_state": FUNDAMENTAL_NOT_USED,
        "fundamental_freshness_policy": FUNDAMENTAL_ADVISORY,
        "system_authority": "NONE",
        "mainline_authority": "NONE",
        "investment_authority": "NONE",
        "portfolio_authority": "NONE",
        "strategy_record_authority": "NONE",
        "broker_authority": "NONE",
    }
    identity = "factor-production-generation-" + _sha256(canonical_json_bytes(body))
    artifact = seal_artifact(
        FACTOR_PRODUCTION_GENERATION_KIND,
        {"factor_production_generation_id": identity, **body},
        created_at=created_at,
        contract_sha256=_contract_sha(FACTOR_PRODUCTION_GENERATION_KIND),
    )
    return validate_factor_production_generation(artifact)


def validate_factor_production_generation(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    try:
        artifact = validate_artifact(
            document,
            expected_kind=FACTOR_PRODUCTION_GENERATION_KIND,
            expected_contract_sha256=_contract_sha(FACTOR_PRODUCTION_GENERATION_KIND),
        )
    except ContractError as exc:
        raise FactorGovernanceError("Factor production generation contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != _FACTOR_GENERATION_FIELDS:
        raise FactorGovernanceError("Factor production generation fields differ")
    if (
        payload["state"] != "OPERATIONAL"
        or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE
        or payload["admission_route"] != "BOOTSTRAP_EXCEPTION"
        or payload["producer_identity"] != "NOT_CLAIMED"
        or payload["fundamental_dependency_state"] != FUNDAMENTAL_NOT_USED
        or payload["fundamental_freshness_policy"] != FUNDAMENTAL_ADVISORY
    ):
        raise FactorGovernanceError("Factor production generation policy differs")
    _as_of(payload["as_of"])
    _require_no_authority(payload)
    expected_kinds = {
        "deployed_release_ref": _RELEASE_KIND,
        "release_install_evidence_ref": "system.release_install_evidence",
        "release_install_input_source_ref": _SOURCE_OBJECT_KIND,
        "source_closure_ref": FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND,
        "recomputation_evidence_ref": FACTOR_PRODUCTION_RECOMPUTATION_KIND,
        "market_pit_selection_ref": _MARKET_PIT_SELECTION_KIND,
        "market_scope_source_ref": _SOURCE_OBJECT_KIND,
        "calendar_capture_custody_attestation_ref": FACTOR_PRODUCTION_CALENDAR_CUSTODY_KIND,
        "factor_source_bundle_ref": _SOURCE_BUNDLE_KIND,
        "market_input_ref": FACTOR_PRODUCTION_MARKET_INPUT_KIND,
        "legacy_zero_call_ref": FACTOR_LEGACY_ZERO_CALL_CERTIFICATE_KIND,
    }
    for field, kind in expected_kinds.items():
        ref = validate_object_ref(payload[field], label=field)
        if ref["kind"] != kind:
            raise FactorGovernanceError(f"Factor production generation {field} kind differs")
    validate_object_ref(payload["calendar_compilation_ref"], label="calendar_compilation_ref")
    _validate_release_install_verification(payload["release_install_verification"])
    validate_object_ref(payload["factor_policy_ref"], label="factor_policy_ref")
    validate_object_ref(payload["factor_active_set_ref"], label="factor_active_set_ref")
    validate_object_ref(
        payload["factor_validation_attestation_ref"],
        label="factor_validation_attestation_ref",
    )
    _refs(payload["factor_implementation_refs"], label="factor_implementation_refs")
    values = _validate_recomputation_values(payload)
    for field in (
        "low_signal_sha256",
        "w80_signal_sha256",
        "signal_statistics",
        "signal_values",
        "active_factor_rows",
        "control_rows",
        "exact_replay_sha256",
    ):
        if payload[field] != values[field]:
            raise FactorGovernanceError(f"Factor production generation {field} differs")
    body = dict(payload)
    identity = body.pop("factor_production_generation_id")
    if identity != "factor-production-generation-" + _sha256(canonical_json_bytes(body)):
        raise FactorGovernanceError("Factor production generation identity differs")
    return artifact


def replay_factor_production_generation(
    document: Mapping[str, Any] | bytes,
    *,
    artifact_resolver: ArtifactResolver,
    source_resolver: SourceResolver,
    validation_mode: str,
    current_release_root: str | Path | None = None,
) -> dict[str, Any]:
    """Deep replay the Factor generation before any authority transition."""

    generation = validate_factor_production_generation(document)
    payload = generation["payload"]
    source = _resolve_exact_artifact(
        payload["source_closure_ref"],
        artifact_resolver=artifact_resolver,
        label="generation.source_closure_ref",
        expected_kinds=frozenset({FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND}),
    )
    recomputation = _resolve_exact_artifact(
        payload["recomputation_evidence_ref"],
        artifact_resolver=artifact_resolver,
        label="generation.recomputation_evidence_ref",
        expected_kinds=frozenset({FACTOR_PRODUCTION_RECOMPUTATION_KIND}),
    )
    replay_factor_production_recomputation_evidence(
        recomputation,
        artifact_resolver=artifact_resolver,
        source_resolver=source_resolver,
        validation_mode=validation_mode,
        current_release_root=current_release_root,
    )
    legacy = _resolve_exact_artifact(
        payload["legacy_zero_call_ref"],
        artifact_resolver=artifact_resolver,
        label="generation.legacy_zero_call_ref",
        expected_kinds=frozenset({FACTOR_LEGACY_ZERO_CALL_CERTIFICATE_KIND}),
    )
    validate_factor_legacy_zero_call_certificate(
        legacy,
        repository_root=(current_release_root if validation_mode == PRE_CAS_CURRENT else None),
    )
    source_payload = source["payload"]
    recomputation_payload = recomputation["payload"]
    direct_source_fields = (
        "deployed_release_ref",
        "release_install_evidence_ref",
        "release_install_input_source_ref",
        "release_install_verification",
        "market_pit_selection_ref",
        "market_scope_source_ref",
        "calendar_compilation_ref",
        "calendar_capture_custody_attestation_ref",
        "factor_source_bundle_ref",
        "market_input_ref",
        "factor_policy_ref",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
        "factor_implementation_refs",
        "legacy_zero_call_ref",
    )
    for field in direct_source_fields:
        if payload[field] != source_payload[field]:
            raise FactorGovernanceError(f"Factor generation source binding {field} differs")
    for field in (
        "as_of",
        "low_signal_sha256",
        "w80_signal_sha256",
        "signal_statistics",
        "signal_values",
        "active_factor_rows",
        "control_rows",
        "exact_replay_sha256",
    ):
        if payload[field] != recomputation_payload[field]:
            raise FactorGovernanceError(f"Factor generation recomputation {field} differs")
    return generation


def build_factor_legacy_zero_call_certificate(
    *,
    final_commit: str,
    final_tree: str,
    resolver_inventory_ref: Mapping[str, Any],
    verification_module_path: str,
    verification_module_sha256: str,
    verification_command: str,
    stdout_sha256: str,
    stderr_sha256: str,
    verified_at: str,
) -> dict[str, Any]:
    """Seal the narrow legacy-zero-call proof needed by Factor authority."""

    body = {
        "state": "VERIFIED",
        "activation_scope": FACTOR_PRODUCTION_SCOPE,
        "final_commit": _git_oid(final_commit, label="final_commit"),
        "final_tree": _git_oid(final_tree, label="final_tree"),
        "resolver_inventory_ref": validate_object_ref(resolver_inventory_ref),
        "active_legacy_import_count": 0,
        "active_legacy_call_count": 0,
        "active_legacy_path_hash_count": 0,
        "legacy_entrypoint_count": 0,
        "verification_module_path": verification_module_path,
        "verification_module_sha256": _sha(
            verification_module_sha256, label="verification_module_sha256"
        ),
        "verification_command": verification_command,
        "stdout_sha256": _sha(stdout_sha256, label="stdout_sha256"),
        "stderr_sha256": _sha(stderr_sha256, label="stderr_sha256"),
        "verified_at": _timestamp(verified_at, label="verified_at"),
    }
    identity = "factor-production-legacy-zero-call-" + _sha256(canonical_json_bytes(body))
    artifact = seal_artifact(
        FACTOR_LEGACY_ZERO_CALL_CERTIFICATE_KIND,
        {"factor_legacy_zero_call_id": identity, **body},
        created_at=verified_at,
        contract_sha256=_contract_sha(FACTOR_LEGACY_ZERO_CALL_CERTIFICATE_KIND),
    )
    return validate_factor_legacy_zero_call_certificate(artifact)


def _legacy_scan_result(
    result: Any,
    *,
    final_commit: str,
    final_tree: str,
) -> dict[str, Any]:
    required = {
        "final_commit",
        "final_tree",
        "resolver_inventory_ref",
        "active_legacy_import_count",
        "active_legacy_call_count",
        "active_legacy_path_hash_count",
        "legacy_entrypoint_count",
        "verification_module_path",
        "verification_module_sha256",
        "verification_command",
        "stdout",
        "stderr",
    }
    if type(result) is not dict or set(result) != required:
        raise FactorGovernanceError("Factor legacy scan result fields differ")
    if result["final_commit"] != final_commit or result["final_tree"] != final_tree:
        raise FactorGovernanceError("Factor legacy scan provenance differs")
    validate_object_ref(result["resolver_inventory_ref"], label="legacy scan resolver inventory")
    for field in (
        "active_legacy_import_count",
        "active_legacy_call_count",
        "active_legacy_path_hash_count",
        "legacy_entrypoint_count",
    ):
        if result[field] != 0:
            raise FactorGovernanceError("Factor legacy scan is nonzero")
    if (
        type(result["verification_module_path"]) is not str
        or not result["verification_module_path"].startswith("quant_investor/factors/")
        or type(result["verification_command"]) is not str
        or not result["verification_command"].strip()
        or "\n" in result["verification_command"]
        or type(result["stdout"]) is not bytes
        or type(result["stderr"]) is not bytes
    ):
        raise FactorGovernanceError("Factor legacy scan text differs")
    _sha(result["verification_module_sha256"], label="legacy scan module SHA")
    return dict(result)


def build_factor_legacy_zero_call_certificate_for_release(
    *,
    repository_root: str | Path,
    final_commit: str,
    final_tree: str,
    resolver_inventory_ref: Mapping[str, Any],
    verified_at: str,
) -> dict[str, Any]:
    """Run the fixed code-owned scanner and seal its immutable receipt."""

    commit = _git_oid(final_commit, label="final_commit")
    tree = _git_oid(final_tree, label="final_tree")
    try:
        result = _legacy_scan_result(
            scan_release_legacy_zero_call(
                repository_root=repository_root,
                final_commit=commit,
                final_tree=tree,
                resolver_inventory_ref=resolver_inventory_ref,
            ),
            final_commit=commit,
            final_tree=tree,
        )
    except FactorGovernanceError:
        raise
    except Exception as exc:
        raise FactorGovernanceError("Factor legacy scan failed") from exc
    return build_factor_legacy_zero_call_certificate(
        final_commit=commit,
        final_tree=tree,
        resolver_inventory_ref=result["resolver_inventory_ref"],
        verification_module_path=result["verification_module_path"],
        verification_module_sha256=result["verification_module_sha256"],
        verification_command=result["verification_command"],
        stdout_sha256=_sha256(result["stdout"]),
        stderr_sha256=_sha256(result["stderr"]),
        verified_at=verified_at,
    )


def validate_factor_legacy_zero_call_certificate(  # noqa: C901
    document: Mapping[str, Any] | bytes,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    try:
        artifact = validate_artifact(
            document,
            expected_kind=FACTOR_LEGACY_ZERO_CALL_CERTIFICATE_KIND,
            expected_contract_sha256=_contract_sha(FACTOR_LEGACY_ZERO_CALL_CERTIFICATE_KIND),
        )
    except ContractError as exc:
        raise FactorGovernanceError("Factor legacy-zero-call certificate contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != _LEGACY_ZERO_CALL_FIELDS:
        raise FactorGovernanceError("Factor legacy-zero-call certificate fields differ")
    if payload["state"] != "VERIFIED" or payload["activation_scope"] != FACTOR_PRODUCTION_SCOPE:
        raise FactorGovernanceError("Factor legacy-zero-call certificate state differs")
    for field in ("final_commit", "final_tree"):
        _git_oid(payload[field], label=field)
    for field in ("verification_module_sha256", "stdout_sha256", "stderr_sha256"):
        _sha(payload[field], label=field)
    if (
        type(payload["verification_module_path"]) is not str
        or not payload["verification_module_path"].startswith("quant_investor/factors/")
        or type(payload["verification_command"]) is not str
        or not payload["verification_command"].strip()
        or "\n" in payload["verification_command"]
    ):
        raise FactorGovernanceError("Factor legacy-zero-call certificate text differs")
    _timestamp(payload["verified_at"], label="verified_at")
    validate_object_ref(payload["resolver_inventory_ref"])
    for field in (
        "active_legacy_import_count",
        "active_legacy_call_count",
        "active_legacy_path_hash_count",
        "legacy_entrypoint_count",
    ):
        if payload[field] != 0:
            raise FactorGovernanceError("Factor legacy-zero-call certificate is nonzero")
    body = dict(payload)
    identity = body.pop("factor_legacy_zero_call_id")
    if identity != "factor-production-legacy-zero-call-" + _sha256(canonical_json_bytes(body)):
        raise FactorGovernanceError("Factor legacy-zero-call certificate identity differs")
    if repository_root is not None:
        try:
            observed = _legacy_scan_result(
                scan_release_legacy_zero_call(
                    repository_root=repository_root,
                    final_commit=payload["final_commit"],
                    final_tree=payload["final_tree"],
                    resolver_inventory_ref=payload["resolver_inventory_ref"],
                ),
                final_commit=payload["final_commit"],
                final_tree=payload["final_tree"],
            )
        except FactorGovernanceError:
            raise
        except Exception as exc:
            raise FactorGovernanceError("Factor legacy scan replay failed") from exc
        for field in (
            "resolver_inventory_ref",
            "active_legacy_import_count",
            "active_legacy_call_count",
            "active_legacy_path_hash_count",
            "legacy_entrypoint_count",
            "verification_module_path",
            "verification_module_sha256",
            "verification_command",
        ):
            if observed[field] != payload[field]:
                raise FactorGovernanceError(f"Factor legacy scan replay {field} differs")
        if (
            _sha256(observed["stdout"]) != payload["stdout_sha256"]
            or _sha256(observed["stderr"]) != payload["stderr_sha256"]
        ):
            raise FactorGovernanceError("Factor legacy scan replay output differs")
    return artifact


__all__ = [
    "FACTOR_PRODUCTION_RECOMPUTATION_KIND",
    "FACTOR_PRODUCTION_SCOPE",
    "FACTOR_PRODUCTION_SOURCE_CLOSURE_KIND",
    "FACTOR_PRODUCTION_MARKET_INPUT_KIND",
    "FACTOR_LEGACY_ZERO_CALL_CERTIFICATE_KIND",
    "FUNDAMENTAL_ADVISORY",
    "FUNDAMENTAL_NOT_USED",
    "build_factor_production_recomputation_evidence",
    "build_factor_production_source_closure",
    "build_factor_production_market_input",
    "build_factor_legacy_zero_call_certificate",
    "build_factor_legacy_zero_call_certificate_for_release",
    "recompute_factor_production_signals",
    "replay_factor_production_recomputation_evidence",
    "system_store_source_resolver",
    "validate_factor_production_recomputation_evidence",
    "validate_factor_production_source_closure",
    "validate_factor_production_market_input",
    "validate_factor_legacy_zero_call_certificate",
]
