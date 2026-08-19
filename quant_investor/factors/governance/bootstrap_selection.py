"""Factor-owned immutable Market-coverage-bound PIT source selection.

This small module is deliberately independent of bootstrap preparation,
assembly, activation, pointers, and provider calls.  It seals only the source
decision that lets a Factor-only authority use the PIT generation bound by an
exact Market snapshot rather than a mutable global PIT "current" pointer.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    get_contract,
    seal_artifact,
    validate_artifact,
)

from .errors import FactorGovernanceError as SelectionContractError

MARKET_PIT_SELECTION_KIND: Final = "factor.production_market_pit_selection"
MARKET_PIT_SELECTION_STATE: Final = "SEALED"
MARKET_PIT_SELECTION_MODE: Final = "MARKET_COVERAGE_BOUND"
MARKET_PIT_SELECTION_FIELDS: Final = frozenset(
    {
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
)

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_DATE_RE: Final = re.compile(r"^[0-9]{8}$")
_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_FILE_REF_FIELDS: Final = frozenset({"relative_path", "byte_sha256"})
_SELECTION_MODULE_PATH: Final = "quant_investor/factors/governance/bootstrap_selection.py"
_PINNED_AS_OF_DISCLOSURE: Final = "MARKET_COVERAGE_BOUND_PIT_NOT_GLOBAL_CURRENT"
_USER_AUTHORIZATION_BASIS: Final = "USER_AUTHORIZED_BOOTSTRAP_EXCEPTION"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SelectionContractError(f"{label} is not lowercase SHA-256")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise SelectionContractError(f"{label} is not a canonical identifier")
    return value


def _date(value: Any, *, label: str) -> str:
    if type(value) is not str or _DATE_RE.fullmatch(value) is None:
        raise SelectionContractError(f"{label} is not YYYYMMDD")
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise SelectionContractError(f"{label} is not YYYYMMDD") from exc
    if parsed.strftime("%Y%m%d") != value:
        raise SelectionContractError(f"{label} is not YYYYMMDD")
    return value


def _file_ref(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != _FILE_REF_FIELDS:
        raise SelectionContractError(f"{label} fields are not exact")
    relative = value.get("relative_path")
    if type(relative) is not str:
        raise SelectionContractError(f"{label}.relative_path is invalid")
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or str(path) != relative
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SelectionContractError(f"{label}.relative_path is invalid")
    return {
        "relative_path": relative,
        "byte_sha256": _sha(value.get("byte_sha256"), label=f"{label}.byte_sha256"),
    }


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SelectionContractError(f"{label} is not an object")
    return dict(value)


def _selection_contract_sha256() -> str:
    return get_contract(MARKET_PIT_SELECTION_KIND).contract_sha256


def _selection_module_sha256() -> str:
    return _sha256(Path(__file__).resolve(strict=True).read_bytes())


def _coverage(value: Any) -> dict[str, Any]:
    coverage = _mapping(value, label="market coverage")
    required = {
        "coverage_schema_version",
        "latest_complete_trade_date",
        "coverage_trade_date",
        "upsert_target_trade_date",
        "expected_scope_sha256",
        "pit_generation_id",
        "pit_generation_manifest_sha256",
        "pit_membership_sha256",
    }
    if not required <= set(coverage):
        raise SelectionContractError("market coverage lacks a Market/PIT binding")
    if coverage.get("coverage_schema_version") != "cn-full-a-coverage.v4":
        raise SelectionContractError("market coverage schema is not strict v4")
    return coverage


def _validate_source_documents(
    *,
    as_of: str,
    market_pointer: Mapping[str, Any],
    market_snapshot_manifest: Mapping[str, Any],
    market_bound_pit_pointer: Mapping[str, Any],
    pit_generation_manifest: Mapping[str, Any],
    pit_membership_sha256: str,
) -> tuple[str, str, str, str]:
    pointer = _mapping(market_pointer, label="market pointer")
    manifest = _mapping(market_snapshot_manifest, label="market snapshot manifest")
    pit_pointer = _mapping(market_bound_pit_pointer, label="market-bound PIT pointer")
    pit_manifest = _mapping(pit_generation_manifest, label="PIT generation manifest")
    coverage = _coverage(pointer.get("coverage"))
    manifest_coverage = _coverage(manifest.get("coverage"))
    if canonical_json_bytes(coverage) != canonical_json_bytes(manifest_coverage):
        raise SelectionContractError("market pointer/snapshot coverage differs")
    if (
        pointer.get("status") != "OK"
        or manifest.get("status") != "OK"
        or pointer.get("blockers") != []
        or manifest.get("blockers") != []
        or pointer.get("snapshot_id") != manifest.get("snapshot_id")
    ):
        raise SelectionContractError("market pointer/snapshot state differs")
    snapshot_id = _identifier(pointer.get("snapshot_id"), label="market snapshot_id")
    if (
        _date(pointer.get("latest_complete_trade_date"), label="market pointer cutoff") != as_of
        or _date(manifest.get("latest_complete_trade_date"), label="market manifest cutoff")
        != as_of
        or _date(coverage.get("latest_complete_trade_date"), label="market coverage cutoff")
        != as_of
        or _date(coverage.get("coverage_trade_date"), label="market coverage date") != as_of
        or _date(coverage.get("upsert_target_trade_date"), label="market coverage target") != as_of
    ):
        raise SelectionContractError("requested as_of differs from strict Market cutoff")
    expected_scope = _sha(coverage.get("expected_scope_sha256"), label="market expected scope")
    pit_generation_id = _identifier(
        coverage.get("pit_generation_id"), label="coverage PIT generation"
    )
    pit_manifest_sha = _sha(
        coverage.get("pit_generation_manifest_sha256"), label="coverage PIT manifest SHA"
    )
    pit_membership_sha = _sha(pit_membership_sha256, label="PIT membership SHA")
    if (
        _sha(coverage.get("pit_membership_sha256"), label="coverage PIT membership SHA")
        != pit_membership_sha
    ):
        raise SelectionContractError("market coverage/PIT membership SHA differs")
    if (
        pit_pointer.get("discovery_schema_version") != "cn_pit_universe_latest.v1"
        or pit_pointer.get("generation_id") != pit_generation_id
        or pit_pointer.get("generation_manifest_sha256") != pit_manifest_sha
        or pit_pointer.get("canonical_sha256") != pit_membership_sha
        or pit_manifest.get("generation_id") != pit_generation_id
        or pit_manifest.get("canonical_sha256") != pit_membership_sha
    ):
        raise SelectionContractError("market-bound PIT discovery binding differs")
    return snapshot_id, _sha256(canonical_json_bytes(coverage)), expected_scope, pit_generation_id


def build_market_pit_selection(
    *,
    as_of: str,
    market_pointer_file_ref: Mapping[str, Any],
    market_snapshot_manifest_file_ref: Mapping[str, Any],
    market_bound_pit_pointer_file_ref: Mapping[str, Any],
    pit_generation_manifest_file_ref: Mapping[str, Any],
    pit_membership_file_ref: Mapping[str, Any],
    observed_current_pit_pointer_file_ref: Mapping[str, Any],
    market_pointer: Mapping[str, Any],
    market_snapshot_manifest: Mapping[str, Any],
    market_bound_pit_pointer: Mapping[str, Any],
    pit_generation_manifest: Mapping[str, Any],
    observed_current_pit_pointer: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    """Seal the exact Market-coverage-bound, non-current PIT decision."""

    normalized_as_of = _date(as_of, label="as_of")
    refs = {
        "market_pointer_file_ref": _file_ref(
            market_pointer_file_ref, label="market_pointer_file_ref"
        ),
        "market_snapshot_manifest_file_ref": _file_ref(
            market_snapshot_manifest_file_ref, label="market_snapshot_manifest_file_ref"
        ),
        "market_bound_pit_pointer_file_ref": _file_ref(
            market_bound_pit_pointer_file_ref, label="market_bound_pit_pointer_file_ref"
        ),
        "pit_generation_manifest_file_ref": _file_ref(
            pit_generation_manifest_file_ref, label="pit_generation_manifest_file_ref"
        ),
        "pit_membership_file_ref": _file_ref(
            pit_membership_file_ref, label="pit_membership_file_ref"
        ),
        "observed_current_pit_pointer_file_ref": _file_ref(
            observed_current_pit_pointer_file_ref,
            label="observed_current_pit_pointer_file_ref",
        ),
    }
    if len({row["relative_path"] for row in refs.values()}) != len(refs):
        raise SelectionContractError("Market/PIT selection source paths are not unique")
    snapshot_id, coverage_sha, scope_sha, pit_generation_id = _validate_source_documents(
        as_of=normalized_as_of,
        market_pointer=market_pointer,
        market_snapshot_manifest=market_snapshot_manifest,
        market_bound_pit_pointer=market_bound_pit_pointer,
        pit_generation_manifest=pit_generation_manifest,
        pit_membership_sha256=refs["pit_membership_file_ref"]["byte_sha256"],
    )
    observed_pointer = _mapping(observed_current_pit_pointer, label="observed current PIT pointer")
    observed_generation = _identifier(
        observed_pointer.get("generation_id"), label="observed current PIT generation"
    )
    if observed_pointer.get("discovery_schema_version") != "cn_pit_universe_latest.v1":
        raise SelectionContractError("observed current PIT pointer schema differs")
    body = {
        "state": MARKET_PIT_SELECTION_STATE,
        "selection_mode": MARKET_PIT_SELECTION_MODE,
        "as_of": normalized_as_of,
        **refs,
        "market_snapshot_id": snapshot_id,
        "market_coverage_sha256": coverage_sha,
        "market_expected_scope_sha256": scope_sha,
        "pit_generation_id": pit_generation_id,
        "pit_generation_manifest_sha256": refs["pit_generation_manifest_file_ref"]["byte_sha256"],
        "pit_membership_sha256": refs["pit_membership_file_ref"]["byte_sha256"],
        "observed_current_pit_pointer_sha256": refs["observed_current_pit_pointer_file_ref"][
            "byte_sha256"
        ],
        "observed_current_pit_generation_id": observed_generation,
        "pinned_as_of_disclosure": _PINNED_AS_OF_DISCLOSURE,
        "user_authorization_basis": _USER_AUTHORIZATION_BASIS,
        "selection_module_path": _SELECTION_MODULE_PATH,
        "selection_module_sha256": _selection_module_sha256(),
    }
    identity = "factor-production-market-pit-" + _sha256(canonical_json_bytes(body))
    artifact = seal_artifact(
        MARKET_PIT_SELECTION_KIND,
        {"market_pit_selection_id": identity, **body},
        created_at=created_at,
        contract_sha256=_selection_contract_sha256(),
    )
    return validate_market_pit_selection(artifact)


def validate_market_pit_selection(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    """Validate the immutable selection's self-binding and explicit disclosure."""

    try:
        artifact = validate_artifact(
            document,
            expected_kind=MARKET_PIT_SELECTION_KIND,
            expected_contract_sha256=_selection_contract_sha256(),
        )
    except ContractError as exc:
        raise SelectionContractError("Market/PIT selection contract failed") from exc
    payload = artifact["payload"]
    if set(payload) != MARKET_PIT_SELECTION_FIELDS:
        raise SelectionContractError("Market/PIT selection fields are not exact")
    if (
        payload["state"] != MARKET_PIT_SELECTION_STATE
        or payload["selection_mode"] != MARKET_PIT_SELECTION_MODE
    ):
        raise SelectionContractError("Market/PIT selection state is invalid")
    _date(payload["as_of"], label="Market/PIT selection as_of")
    refs = {
        field: _file_ref(payload[field], label=field)
        for field in (
            "market_pointer_file_ref",
            "market_snapshot_manifest_file_ref",
            "market_bound_pit_pointer_file_ref",
            "pit_generation_manifest_file_ref",
            "pit_membership_file_ref",
            "observed_current_pit_pointer_file_ref",
        )
    }
    if len({row["relative_path"] for row in refs.values()}) != len(refs):
        raise SelectionContractError("Market/PIT selection source paths are not unique")
    _identifier(payload["market_snapshot_id"], label="market_snapshot_id")
    _sha(payload["market_coverage_sha256"], label="market_coverage_sha256")
    _sha(payload["market_expected_scope_sha256"], label="market_expected_scope_sha256")
    _identifier(payload["pit_generation_id"], label="pit_generation_id")
    _identifier(
        payload["observed_current_pit_generation_id"],
        label="observed_current_pit_generation_id",
    )
    if (
        _sha(payload["pit_generation_manifest_sha256"], label="pit manifest SHA")
        != refs["pit_generation_manifest_file_ref"]["byte_sha256"]
        or _sha(payload["pit_membership_sha256"], label="pit membership SHA")
        != refs["pit_membership_file_ref"]["byte_sha256"]
        or _sha(
            payload["observed_current_pit_pointer_sha256"],
            label="observed current PIT pointer SHA",
        )
        != refs["observed_current_pit_pointer_file_ref"]["byte_sha256"]
        or payload["pinned_as_of_disclosure"] != _PINNED_AS_OF_DISCLOSURE
        or payload["user_authorization_basis"] != _USER_AUTHORIZATION_BASIS
        or payload["selection_module_path"] != _SELECTION_MODULE_PATH
        or _sha(payload["selection_module_sha256"], label="selection module SHA")
        != _selection_module_sha256()
    ):
        raise SelectionContractError("Market/PIT selection self-binding differs")
    body = dict(payload)
    identity = body.pop("market_pit_selection_id")
    if identity != "factor-production-market-pit-" + _sha256(canonical_json_bytes(body)):
        raise SelectionContractError("Market/PIT selection identity differs")
    return artifact


__all__ = [
    "MARKET_PIT_SELECTION_FIELDS",
    "MARKET_PIT_SELECTION_KIND",
    "MARKET_PIT_SELECTION_MODE",
    "build_market_pit_selection",
    "validate_market_pit_selection",
]
