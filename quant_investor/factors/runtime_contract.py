"""Fail-closed production runtime and activation contracts for Quant factors.

The helpers in this module are offline and read-only.  They validate exact
registry-bound contracts and an operator-supplied activation receipt; they do
not create receipts or mutate the factor registry.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.factors.governance import FactorRecord
from quant_investor.factors.registry_store import load_registry_snapshot_strict


RUNTIME_CONTRACT_SCHEMA_VERSION = "factor-production-runtime-contract.v1"
ACTIVATION_RECEIPT_SCHEMA_VERSION = "quant-production-activation-receipt.v1"
RUNTIME_CONTRACT_METADATA_KEY = "production_factor_runtime_contracts"
MIN_PRODUCTION_CROSS_SECTION = 20
PRODUCTION_DATA_SEMANTICS = "strict-parquet-cn-daily-adjusted.v1"
PRICE_VOLUME_IMPLEMENTATION_VERSION = "price-volume-runtime.v1"

_RUNTIME_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "factor_name",
        "factor_version",
        "implementation_id",
        "implementation_version",
        "implementation_code_sha256",
        "required_columns",
        "data_semantics",
        "lookback_rows",
        "gate2_min_coverage_rate",
        "min_cross_section",
        "factor_definition_sha256",
        "factor_record_sha256",
        "factor_evidence_path",
        "factor_evidence_sha256",
    }
)
_ACTIVATION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "activation_id",
        "approved_by",
        "issued_at",
        "kill_switch_value",
        "registry_path",
        "registry_sha256",
        "production_factor_set_sha256",
        "production_runtime_contracts_sha256",
        "implementation_code_sha256s",
        "factor_governance_protocol_version",
        "factor_governance_protocol_hash",
        "receipt_sha256",
    }
)
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_WINDOW_PATTERN = re.compile(r"^(?P<prefix>[a-z0-9_]+)_(?P<window>[1-9][0-9]*)d$")
_SMOOTH_PATTERN = re.compile(
    r"^pv_volume_stability_smooth_(?P<base>[1-9][0-9]*)d_"
    r"(?P<smooth>[1-9][0-9]*)d$"
)
_GROWTH_PATTERN = re.compile(
    r"^pv_dollar_volume_growth_(?P<short>[1-9][0-9]*)d_"
    r"(?P<long>[1-9][0-9]*)d$"
)
_BLEND_PATTERN = re.compile(
    r"^pv_blend_volstab19x2_mom90_amihud5_w(?P<weight>[1-9][0-9]?)$"
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and _HEX_64.fullmatch(value) is not None


def _file_sha256_readback(path: Path) -> tuple[str, str | None]:
    try:
        first = path.read_bytes()
        second = path.read_bytes()
    except OSError as exc:
        return "", str(exc)
    if first != second:
        return "", "file_changed_during_readback"
    return hashlib.sha256(first).hexdigest(), None


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def factor_record_payload_sha256(record: FactorRecord) -> str:
    """Hash the exact normalized record payload used by runtime admission."""

    return _canonical_sha256(record.to_dict())


def factor_definition_sha256(record: FactorRecord) -> str:
    """Hash the immutable implementation-facing definition of one factor."""

    definition_metadata = {
        key: record.metadata.get(key)
        for key in (
            "expression",
            "params",
            "primitive_lineage",
            "primitive_contributions",
            "dominant_primitives",
            "factor_family",
            "dominant_primitive_cluster",
        )
        if key in record.metadata
    }
    return _canonical_sha256(
        {
            "name": record.name,
            "version": record.version,
            "category": record.category,
            "implementation": record.implementation,
            "direction": record.direction,
            "horizon_days": record.horizon_days,
            "definition_metadata": definition_metadata,
        }
    )


def _price_volume_spec(name: str) -> dict[str, Any]:
    if _BLEND_PATTERN.fullmatch(name):
        return {
            "required_columns": ["trade_date", "adj_close", "vol", "amount"],
            "lookback_rows": 91,
        }
    match = _SMOOTH_PATTERN.fullmatch(name)
    if match:
        return {
            "required_columns": ["trade_date", "vol"],
            "lookback_rows": int(match.group("base")) + int(match.group("smooth")),
        }
    match = _GROWTH_PATTERN.fullmatch(name)
    if match:
        return {
            "required_columns": ["trade_date", "amount"],
            "lookback_rows": max(int(match.group("short")), int(match.group("long"))),
        }
    match = _WINDOW_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"production implementation is not allowlisted: price_volume:{name}")
    prefix = match.group("prefix")
    window = int(match.group("window"))
    if prefix in {
        "pv_momentum",
        "pv_short_reversal",
        "pv_volatility_penalty",
        "pv_downside_volatility",
        "pv_price_efficiency",
    }:
        return {
            "required_columns": ["trade_date", "adj_close"],
            "lookback_rows": window + 1,
        }
    if prefix == "pv_volume_stability":
        return {
            "required_columns": ["trade_date", "vol"],
            "lookback_rows": window,
        }
    if prefix in {"pv_low_dollar_volume", "pv_high_dollar_volume"}:
        return {
            "required_columns": ["trade_date", "amount"],
            "lookback_rows": window,
        }
    if prefix == "pv_amihud_illiquidity":
        return {
            "required_columns": ["trade_date", "adj_close", "amount"],
            "lookback_rows": window + 1,
        }
    raise ValueError(f"production implementation is not allowlisted: price_volume:{name}")


def production_implementation_spec(implementation_id: str) -> dict[str, Any]:
    """Resolve one exact implementation through the production allowlist."""

    value = str(implementation_id or "").strip()
    if not value.startswith("price_volume:"):
        raise ValueError(f"production implementation is not allowlisted: {value}")
    name = value.split(":", 1)[1]
    spec = _price_volume_spec(name)
    return {
        "implementation_id": value,
        "implementation_version": PRICE_VOLUME_IMPLEMENTATION_VERSION,
        "implementation_code_path": Path(__file__).with_name("price_volume.py"),
        "required_columns": list(spec["required_columns"]),
        "data_semantics": PRODUCTION_DATA_SEMANTICS,
        "lookback_rows": int(spec["lookback_rows"]),
    }


def implementation_code_sha256(implementation_id: str) -> str:
    spec = production_implementation_spec(implementation_id)
    digest, error = _file_sha256_readback(spec["implementation_code_path"])
    if error:
        raise ValueError(f"implementation code readback failed: {error}")
    return digest


def production_runtime_contracts_sha256(contracts: Mapping[str, Any]) -> str:
    return _canonical_sha256(dict(contracts))


def activation_receipt_payload_sha256(receipt: Mapping[str, Any]) -> str:
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    return _canonical_sha256(body)


def _contract_blockers(
    record: FactorRecord,
    raw_contract: Any,
    *,
    expected_record_sha256: str,
) -> tuple[list[str], dict[str, Any] | None]:
    prefix = f"factor_runtime_contract:{record.name}"
    blockers: list[str] = []
    if not isinstance(raw_contract, Mapping):
        return [f"{prefix}:not_an_object"], None
    contract = dict(raw_contract)
    missing = sorted(_RUNTIME_CONTRACT_FIELDS - set(contract))
    unknown = sorted(set(contract) - _RUNTIME_CONTRACT_FIELDS)
    if missing:
        blockers.append(f"{prefix}:missing_fields:{','.join(missing)}")
    if unknown:
        blockers.append(f"{prefix}:unknown_fields:{','.join(unknown)}")
    if blockers:
        return blockers, None

    if contract["schema_version"] != RUNTIME_CONTRACT_SCHEMA_VERSION:
        blockers.append(f"{prefix}:schema_version_mismatch")
    if contract["factor_name"] != record.name:
        blockers.append(f"{prefix}:factor_name_mismatch")
    if contract["factor_version"] != record.version:
        blockers.append(f"{prefix}:factor_version_mismatch")
    if contract["implementation_id"] != record.implementation:
        blockers.append(f"{prefix}:implementation_id_mismatch")
    try:
        spec = production_implementation_spec(record.implementation)
    except ValueError:
        blockers.append(f"{prefix}:implementation_not_allowlisted")
        spec = None
    if spec is not None:
        factor_name = record.implementation.split(":", 1)[1]
        if factor_name != record.name:
            blockers.append(f"{prefix}:implementation_factor_name_mismatch")
        if contract["implementation_version"] != spec["implementation_version"]:
            blockers.append(f"{prefix}:implementation_version_mismatch")
        if contract["required_columns"] != spec["required_columns"]:
            blockers.append(f"{prefix}:required_columns_mismatch")
        if contract["data_semantics"] != spec["data_semantics"]:
            blockers.append(f"{prefix}:data_semantics_mismatch")
        lookback = contract["lookback_rows"]
        if (
            isinstance(lookback, bool)
            or not isinstance(lookback, int)
            or lookback <= 0
            or lookback != spec["lookback_rows"]
        ):
            blockers.append(f"{prefix}:lookback_rows_mismatch")
        expected_code_sha, code_error = _file_sha256_readback(
            spec["implementation_code_path"]
        )
        if code_error or contract["implementation_code_sha256"] != expected_code_sha:
            blockers.append(f"{prefix}:implementation_code_sha256_mismatch")

    required_columns = contract["required_columns"]
    if (
        not isinstance(required_columns, list)
        or not required_columns
        or any(not isinstance(item, str) or not item for item in required_columns)
        or len(required_columns) != len(set(required_columns))
    ):
        blockers.append(f"{prefix}:required_columns_invalid")

    minimum_coverage = contract["gate2_min_coverage_rate"]
    if (
        isinstance(minimum_coverage, bool)
        or not isinstance(minimum_coverage, Real)
        or not math.isfinite(float(minimum_coverage))
        or not 0.60 <= float(minimum_coverage) <= 1.0
    ):
        blockers.append(f"{prefix}:gate2_min_coverage_rate_invalid")
    gate_two = record.gate_map().get(2)
    evidence_coverage = (
        gate_two.metrics.get("coverage_rate") if gate_two is not None else None
    )
    if (
        isinstance(evidence_coverage, bool)
        or not isinstance(evidence_coverage, Real)
        or not math.isfinite(float(evidence_coverage))
        or (
            isinstance(minimum_coverage, Real)
            and not isinstance(minimum_coverage, bool)
            and math.isfinite(float(minimum_coverage))
            and float(minimum_coverage) > float(evidence_coverage) + 1e-12
        )
    ):
        blockers.append(f"{prefix}:gate2_coverage_evidence_invalid")

    min_cross_section = contract["min_cross_section"]
    if (
        isinstance(min_cross_section, bool)
        or not isinstance(min_cross_section, int)
        or min_cross_section < MIN_PRODUCTION_CROSS_SECTION
    ):
        blockers.append(f"{prefix}:min_cross_section_invalid")
    if contract["factor_definition_sha256"] != factor_definition_sha256(record):
        blockers.append(f"{prefix}:factor_definition_sha256_mismatch")
    current_record_sha256 = factor_record_payload_sha256(record)
    if (
        not _is_hash(expected_record_sha256)
        or expected_record_sha256 != current_record_sha256
        or contract["factor_record_sha256"] != current_record_sha256
    ):
        blockers.append(f"{prefix}:factor_record_sha256_mismatch")
    evidence_path_value = contract["factor_evidence_path"]
    if not isinstance(evidence_path_value, str) or not evidence_path_value.strip():
        blockers.append(f"{prefix}:factor_evidence_path_invalid")
    else:
        evidence_path = Path(evidence_path_value).expanduser()
        evidence_sha, evidence_error = _file_sha256_readback(evidence_path)
        if evidence_error or contract["factor_evidence_sha256"] != evidence_sha:
            blockers.append(f"{prefix}:factor_evidence_sha256_mismatch")
    if not _is_hash(contract["factor_evidence_sha256"]):
        blockers.append(f"{prefix}:factor_evidence_sha256_invalid")
    return blockers, contract if not blockers else None


def validate_production_runtime_contracts(
    records: Sequence[FactorRecord],
    registry_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact production set and every readback-bound contract."""

    names = [record.name for record in records]
    if len(names) != len(set(names)):
        return {
            "status": "governance_blocked",
            "contracts": {},
            "contracts_sha256": "",
            "implementation_code_sha256s": {},
            "blockers": ["production_runtime_factor_names_not_unique"],
        }
    raw_contracts = registry_metadata.get(RUNTIME_CONTRACT_METADATA_KEY)
    if not isinstance(raw_contracts, Mapping):
        return {
            "status": "governance_blocked",
            "contracts": {},
            "contracts_sha256": "",
            "implementation_code_sha256s": {},
            "blockers": ["production_runtime_contracts_missing"],
        }
    registry_path_value = registry_metadata.get("path")
    if not isinstance(registry_path_value, str) or not registry_path_value.strip():
        return {
            "status": "governance_blocked",
            "contracts": {},
            "contracts_sha256": "",
            "implementation_code_sha256s": {},
            "blockers": ["production_runtime_registry_path_missing"],
        }
    if registry_metadata.get("strict_loader") is not True:
        return {
            "status": "governance_blocked",
            "contracts": {},
            "contracts_sha256": "",
            "implementation_code_sha256s": {},
            "blockers": ["production_runtime_registry_not_strictly_loaded"],
        }
    try:
        snapshot = load_registry_snapshot_strict(registry_path_value)
        claimed_path = Path(registry_path_value).expanduser().resolve()
        snapshot_path = snapshot.path.expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return {
            "status": "governance_blocked",
            "contracts": {},
            "contracts_sha256": "",
            "implementation_code_sha256s": {},
            "blockers": ["production_runtime_registry_snapshot_unreadable"],
        }
    if claimed_path != snapshot_path:
        return {
            "status": "governance_blocked",
            "contracts": {},
            "contracts_sha256": "",
            "implementation_code_sha256s": {},
            "blockers": ["production_runtime_registry_snapshot_path_mismatch"],
        }
    claimed_registry_sha = registry_metadata.get("registry_sha256")
    if (
        not _is_hash(claimed_registry_sha)
        or claimed_registry_sha != snapshot.registry_sha256
    ):
        return {
            "status": "governance_blocked",
            "contracts": {},
            "contracts_sha256": "",
            "implementation_code_sha256s": {},
            "blockers": ["production_runtime_registry_readback_sha256_mismatch"],
        }

    snapshot_record_sha256s = dict(snapshot.record_sha256s)
    record_sha256s = registry_metadata.get("record_sha256s")
    if (
        not isinstance(record_sha256s, Mapping)
        or dict(record_sha256s) != snapshot_record_sha256s
    ):
        return {
            "status": "governance_blocked",
            "contracts": {},
            "contracts_sha256": "",
            "implementation_code_sha256s": {},
            "blockers": [
                "production_runtime_registry_snapshot_record_sha256s_mismatch"
            ],
        }

    snapshot_contracts = snapshot.metadata_payload.get(RUNTIME_CONTRACT_METADATA_KEY)
    if (
        not isinstance(raw_contracts, Mapping)
        or not isinstance(snapshot_contracts, Mapping)
        or dict(raw_contracts) != dict(snapshot_contracts)
    ):
        return {
            "status": "governance_blocked",
            "contracts": {},
            "contracts_sha256": "",
            "implementation_code_sha256s": {},
            "blockers": ["production_runtime_registry_snapshot_contracts_mismatch"],
        }
    contract_names = {str(name) for name in raw_contracts}
    if contract_names != set(names):
        return {
            "status": "governance_blocked",
            "contracts": {},
            "contracts_sha256": "",
            "implementation_code_sha256s": {},
            "blockers": ["production_runtime_contract_factor_set_mismatch"],
        }
    snapshot_selectable = {
        record.name: record for record in snapshot.registry.selectable_factors()
    }
    if set(snapshot_selectable) != set(names):
        return {
            "status": "governance_blocked",
            "contracts": {},
            "contracts_sha256": "",
            "implementation_code_sha256s": {},
            "blockers": [
                "production_runtime_registry_snapshot_factor_set_mismatch"
            ],
        }
    blockers: list[str] = []
    contracts: dict[str, dict[str, Any]] = {}
    code_hashes: dict[str, str] = {}
    for record in records:
        snapshot_record = snapshot_selectable.get(record.name)
        snapshot_record_sha = snapshot_record_sha256s.get(record.name)
        if (
            snapshot_record is None
            or snapshot_record.to_dict() != record.to_dict()
            or snapshot_record_sha != factor_record_payload_sha256(record)
        ):
            blockers.append(
                f"factor_runtime_contract:{record.name}:"
                "registry_snapshot_factor_record_sha256_mismatch"
            )
            continue
        row_blockers, contract = _contract_blockers(
            record,
            raw_contracts.get(record.name),
            expected_record_sha256=str(snapshot_record_sha or ""),
        )
        blockers.extend(row_blockers)
        if contract is not None:
            contracts[record.name] = contract
            code_hashes[record.name] = str(contract["implementation_code_sha256"])
    blockers = list(dict.fromkeys(blockers))
    return {
        "status": "ready" if not blockers else "governance_blocked",
        "contracts": contracts if not blockers else {},
        "contracts_sha256": (
            production_runtime_contracts_sha256(contracts) if not blockers else ""
        ),
        "implementation_code_sha256s": code_hashes if not blockers else {},
        "blockers": blockers,
    }


def validate_quant_production_activation(
    registry_metadata: Mapping[str, Any],
    production_manifest: Mapping[str, Any],
    runtime_contracts_sha256: str,
    *,
    implementation_code_sha256s: Mapping[str, str] | None = None,
    protocol_version: str,
    protocol_hash_value: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate kill switch and an exact registry/contract-bound receipt."""

    env = os.environ if environ is None else environ
    raw_switch = env.get("QUANT_PRODUCTION_KILL_SWITCH")
    if raw_switch is None:
        return {
            "status": "governance_blocked",
            "blockers": ["quant_production_kill_switch_missing"],
        }
    if raw_switch == "true":
        return {
            "status": "governance_blocked",
            "blockers": ["quant_production_kill_switch_active"],
        }
    if raw_switch != "false":
        blocker = (
            "quant_production_kill_switch_empty"
            if raw_switch == ""
            else "quant_production_kill_switch_invalid"
        )
        return {"status": "governance_blocked", "blockers": [blocker]}

    receipt_value = env.get("QUANT_PRODUCTION_ACTIVATION_RECEIPT")
    if not receipt_value:
        return {
            "status": "governance_blocked",
            "blockers": ["quant_production_activation_receipt_missing"],
        }
    if registry_metadata.get("strict_loader") is not True:
        return {
            "status": "governance_blocked",
            "blockers": ["quant_production_registry_not_strictly_loaded"],
        }
    registry_path_value = registry_metadata.get("path")
    registry_sha = registry_metadata.get("registry_sha256")
    if not isinstance(registry_path_value, str) or not _is_hash(registry_sha):
        return {
            "status": "governance_blocked",
            "blockers": ["quant_production_registry_readback_identity_missing"],
        }
    registry_path = Path(registry_path_value).expanduser().resolve()
    observed_registry_sha, registry_error = _file_sha256_readback(registry_path)
    if registry_error or observed_registry_sha != registry_sha:
        return {
            "status": "governance_blocked",
            "blockers": ["quant_production_registry_readback_mismatch"],
        }

    receipt_path = Path(receipt_value).expanduser()
    try:
        if stat.S_IMODE(receipt_path.stat().st_mode) != 0o600:
            return {
                "status": "governance_blocked",
                "blockers": ["quant_production_activation_receipt_permissions_unsafe"],
            }
        first = receipt_path.read_bytes()
        second = receipt_path.read_bytes()
    except OSError:
        return {
            "status": "governance_blocked",
            "blockers": ["quant_production_activation_receipt_unreadable"],
        }
    if first != second:
        return {
            "status": "governance_blocked",
            "blockers": ["quant_production_activation_receipt_readback_mismatch"],
        }
    observed_receipt_sha = hashlib.sha256(first).hexdigest()
    expected_receipt_sha = env.get(
        "QUANT_PRODUCTION_ACTIVATION_RECEIPT_SHA256"
    )
    if not _is_hash(expected_receipt_sha):
        return {
            "status": "governance_blocked",
            "blockers": ["quant_production_activation_receipt_expected_sha256_missing"],
        }
    if expected_receipt_sha != observed_receipt_sha:
        return {
            "status": "governance_blocked",
            "blockers": ["quant_production_activation_receipt_exact_bytes_mismatch"],
        }
    try:
        payload = json.loads(
            first.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {
            "status": "governance_blocked",
            "blockers": ["quant_production_activation_receipt_malformed"],
        }
    if not isinstance(payload, Mapping) or set(payload) != _ACTIVATION_RECEIPT_FIELDS:
        return {
            "status": "governance_blocked",
            "blockers": ["quant_production_activation_receipt_schema_invalid"],
        }
    receipt = dict(payload)
    expected_code_hashes = dict(implementation_code_sha256s or {})
    expected = {
        "schema_version": ACTIVATION_RECEIPT_SCHEMA_VERSION,
        "status": "authorized",
        "kill_switch_value": "false",
        "registry_path": str(registry_path),
        "registry_sha256": registry_sha,
        "production_factor_set_sha256": production_manifest.get(
            "production_factor_set_sha256"
        ),
        "production_runtime_contracts_sha256": runtime_contracts_sha256,
        "implementation_code_sha256s": expected_code_hashes,
        "factor_governance_protocol_version": protocol_version,
        "factor_governance_protocol_hash": protocol_hash_value,
    }
    blockers: list[str] = []
    for field, value in expected.items():
        if receipt.get(field) != value:
            blockers.append(f"quant_production_activation_receipt_{field}_mismatch")
    if not isinstance(receipt.get("issued_at"), str) or not receipt["issued_at"].strip():
        blockers.append("quant_production_activation_receipt_issued_at_invalid")
    for field in ("activation_id", "approved_by"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            blockers.append(f"quant_production_activation_receipt_{field}_invalid")
    if receipt.get("receipt_sha256") != activation_receipt_payload_sha256(receipt):
        blockers.append("quant_production_activation_receipt_hash_mismatch")
    blockers = list(dict.fromkeys(blockers))
    return {
        "status": "ready" if not blockers else "governance_blocked",
        "receipt_path": str(receipt_path.resolve()),
        "receipt_file_sha256": observed_receipt_sha,
        "blockers": blockers,
    }


__all__ = [
    "ACTIVATION_RECEIPT_SCHEMA_VERSION",
    "MIN_PRODUCTION_CROSS_SECTION",
    "PRODUCTION_DATA_SEMANTICS",
    "RUNTIME_CONTRACT_METADATA_KEY",
    "RUNTIME_CONTRACT_SCHEMA_VERSION",
    "activation_receipt_payload_sha256",
    "factor_definition_sha256",
    "factor_record_payload_sha256",
    "implementation_code_sha256",
    "production_implementation_spec",
    "production_runtime_contracts_sha256",
    "validate_production_runtime_contracts",
    "validate_quant_production_activation",
]
