"""Immutable non-authorizing observations of sealed Factor production signals."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import PurePosixPath
import re
from typing import Any, Final, Mapping

from quant_investor.contracts import (
    ContractError,
    canonical_json_bytes,
    get_contract,
    seal_artifact,
    validate_artifact,
)

from .governance.errors import FactorGovernanceError
from .production_authority import (
    FACTOR_PRODUCTION_OBSERVATIONS_ROOT,
    FactorProductionStore,
)

FACTOR_PRODUCTION_OBSERVATION_KIND: Final = "factor.production_observation"
PLANNED_HORIZONS: Final = (1, 5, 20, 60)
RETURN_FORMULA: Final = "close(T+h)/close(T)-1"
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_DATE_RE: Final = re.compile(r"^[0-9]{8}$")
_STAMP_RE: Final = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_FACTOR_IDS: Final = {
    "LOW": "pv_low_dollar_volume_5d",
    "W80": "pv_blend_volstab19x2_mom90_amihud5_w80",
}
_NO_AUTHORITY_FIELDS: Final = (
    "system_authority",
    "mainline_authority",
    "investment_authority",
    "portfolio_authority",
    "strategy_record_authority",
    "broker_authority",
)


def _fail(detail: str) -> FactorGovernanceError:
    return FactorGovernanceError(detail)


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _fail(f"{label} is not lowercase SHA-256")
    return value


def _stamp(value: Any, *, label: str) -> str:
    if type(value) is not str or _STAMP_RE.fullmatch(value) is None:
        raise _fail(f"{label} is not canonical UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise _fail(f"{label} is not canonical UTC seconds") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise _fail(f"{label} is not canonical UTC seconds")
    return value


def _ref(value: Any, *, label: str) -> dict[str, str]:
    fields = {"kind", "contract_sha256", "artifact_id", "semantic_sha256", "byte_sha256"}
    if type(value) is not dict or set(value) != fields:
        raise _fail(f"{label} is not an exact artifact ref")
    result = {field: str(value[field]) for field in fields}
    for field in ("contract_sha256", "semantic_sha256", "byte_sha256"):
        _sha(result[field], label=f"{label}.{field}")
    if not result["kind"] or not result["artifact_id"]:
        raise _fail(f"{label} text differs")
    return result


def _observation_path(signal_date: str, factor_alias: str) -> PurePosixPath:
    return (
        FACTOR_PRODUCTION_OBSERVATIONS_ROOT
        / signal_date[:4]
        / signal_date[4:6]
        / signal_date[6:8]
        / f"{factor_alias}.json"
    )


def build_factor_production_observation(
    *, inputs: Mapping[str, Any], factor_row: Mapping[str, Any], registered_at: str
) -> dict[str, Any]:
    """Seal one immutable OPEN observation from an already verified active head."""

    stamp = _stamp(registered_at, label="registered_at")
    signal_date = inputs.get("signal_date")
    if type(signal_date) is not str or _DATE_RE.fullmatch(signal_date) is None:
        raise _fail("Factor production observation signal_date differs")
    if stamp[:10].replace("-", "") < signal_date:
        raise _fail("Factor production observation precedes its signal date")
    alias = factor_row.get("factor_alias")
    factor_id = factor_row.get("factor_id")
    if alias not in _FACTOR_IDS or factor_id != _FACTOR_IDS[alias]:
        raise _fail("Factor production observation factor identity differs")
    generation_id = inputs.get("factor_generation_id")
    if type(generation_id) is not str or not generation_id.startswith(
        "factor-production-generation-"
    ):
        raise _fail("Factor production observation generation identity differs")
    body = {
        "state": "OPEN",
        "authority": "NON_AUTHORIZING",
        "factor_id": factor_id,
        "factor_alias": alias,
        "signal_date": signal_date,
        "registered_at": stamp,
        "planned_horizons": list(PLANNED_HORIZONS),
        "return_formula": RETURN_FORMULA,
        "factor_generation_id": generation_id,
        "factor_generation_sha256": _sha(
            inputs.get("factor_generation_sha256"), label="factor_generation_sha256"
        ),
        "factor_pointer_sha256": _sha(
            inputs.get("factor_pointer_sha256"), label="factor_pointer_sha256"
        ),
        "signal_sha256": _sha(factor_row.get("signal_sha256"), label="signal_sha256"),
        "signal_symbol_set_sha256": _sha(
            factor_row.get("signal_symbol_set_sha256"), label="signal_symbol_set_sha256"
        ),
        "symbol_count": factor_row.get("symbol_count"),
        "market_pointer_sha256": _sha(
            inputs.get("market_pointer_sha256"), label="market_pointer_sha256"
        ),
        "market_manifest_sha256": _sha(
            inputs.get("market_manifest_sha256"), label="market_manifest_sha256"
        ),
        "pit_pointer_sha256": _sha(inputs.get("pit_pointer_sha256"), label="pit_pointer_sha256"),
        "pit_manifest_sha256": _sha(inputs.get("pit_manifest_sha256"), label="pit_manifest_sha256"),
        "pit_membership_sha256": _sha(
            inputs.get("pit_membership_sha256"), label="pit_membership_sha256"
        ),
        "calendar_compilation_ref": _ref(
            inputs.get("calendar_compilation_ref"), label="calendar_compilation_ref"
        ),
        "calendar_capture_custody_attestation_ref": _ref(
            inputs.get("calendar_capture_custody_attestation_ref"),
            label="calendar_capture_custody_attestation_ref",
        ),
        **{field: "NONE" for field in _NO_AUTHORITY_FIELDS},
    }
    if type(body["symbol_count"]) is not int or body["symbol_count"] <= 0:
        raise _fail("Factor production observation symbol_count differs")
    identity = (
        "factor-production-observation-" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    )
    return validate_factor_production_observation(
        seal_artifact(
            FACTOR_PRODUCTION_OBSERVATION_KIND,
            {"factor_production_observation_id": identity, **body},
            created_at=stamp,
            contract_sha256=get_contract(FACTOR_PRODUCTION_OBSERVATION_KIND).contract_sha256,
        )
    )


def validate_factor_production_observation(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate one immutable production observation and its no-authority policy."""

    try:
        artifact = validate_artifact(
            document,
            expected_kind=FACTOR_PRODUCTION_OBSERVATION_KIND,
            expected_contract_sha256=get_contract(
                FACTOR_PRODUCTION_OBSERVATION_KIND
            ).contract_sha256,
        )
    except ContractError as exc:
        raise _fail("Factor production observation contract failed") from exc
    payload = artifact["payload"]
    expected_fields = set(get_contract(FACTOR_PRODUCTION_OBSERVATION_KIND).required_payload_fields)
    if set(payload) != expected_fields:
        raise _fail("Factor production observation fields differ")
    if (
        payload["state"] != "OPEN"
        or payload["authority"] != "NON_AUTHORIZING"
        or payload["planned_horizons"] != list(PLANNED_HORIZONS)
        or payload["return_formula"] != RETURN_FORMULA
        or any(payload[field] != "NONE" for field in _NO_AUTHORITY_FIELDS)
    ):
        raise _fail("Factor production observation policy differs")
    signal_date = payload["signal_date"]
    if type(signal_date) is not str or _DATE_RE.fullmatch(signal_date) is None:
        raise _fail("Factor production observation date differs")
    stamp = _stamp(payload["registered_at"], label="registered_at")
    if artifact["created_at"] != stamp or stamp[:10].replace("-", "") < signal_date:
        raise _fail("Factor production observation time binding differs")
    alias = payload["factor_alias"]
    if alias not in _FACTOR_IDS or payload["factor_id"] != _FACTOR_IDS[alias]:
        raise _fail("Factor production observation factor identity differs")
    if (
        type(payload["symbol_count"]) is not int
        or payload["symbol_count"] <= 0
        or type(payload["factor_generation_id"]) is not str
        or not payload["factor_generation_id"].startswith("factor-production-generation-")
    ):
        raise _fail("Factor production observation production binding differs")
    for field in (
        "factor_generation_sha256",
        "factor_pointer_sha256",
        "signal_sha256",
        "signal_symbol_set_sha256",
        "market_pointer_sha256",
        "market_manifest_sha256",
        "pit_pointer_sha256",
        "pit_manifest_sha256",
        "pit_membership_sha256",
    ):
        _sha(payload[field], label=field)
    _ref(payload["calendar_compilation_ref"], label="calendar_compilation_ref")
    _ref(
        payload["calendar_capture_custody_attestation_ref"],
        label="calendar_capture_custody_attestation_ref",
    )
    body = dict(payload)
    identity = body.pop("factor_production_observation_id")
    expected_identity = (
        "factor-production-observation-" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    )
    if identity != expected_identity:
        raise _fail("Factor production observation identity differs")
    return artifact


def _matches_inputs(
    observation: Mapping[str, Any], inputs: Mapping[str, Any], factor_row: Mapping[str, Any]
) -> bool:
    payload = observation["payload"]
    expected = {
        "factor_id": factor_row["factor_id"],
        "factor_alias": factor_row["factor_alias"],
        "signal_date": inputs["signal_date"],
        "factor_generation_id": inputs["factor_generation_id"],
        "factor_generation_sha256": inputs["factor_generation_sha256"],
        "factor_pointer_sha256": inputs["factor_pointer_sha256"],
        "signal_sha256": factor_row["signal_sha256"],
        "signal_symbol_set_sha256": factor_row["signal_symbol_set_sha256"],
        "symbol_count": factor_row["symbol_count"],
        "market_pointer_sha256": inputs["market_pointer_sha256"],
        "market_manifest_sha256": inputs["market_manifest_sha256"],
        "pit_pointer_sha256": inputs["pit_pointer_sha256"],
        "pit_manifest_sha256": inputs["pit_manifest_sha256"],
        "pit_membership_sha256": inputs["pit_membership_sha256"],
        "calendar_compilation_ref": inputs["calendar_compilation_ref"],
        "calendar_capture_custody_attestation_ref": inputs[
            "calendar_capture_custody_attestation_ref"
        ],
    }
    return all(payload.get(field) == value for field, value in expected.items())


def _register_factor_production_observations(
    workspace_root: str, *, registered_at: str
) -> dict[str, Any]:
    store = FactorProductionStore(workspace_root)
    with store._active_lock():
        inputs = store.read_active_observation_inputs()
        rows = list(inputs["factor_rows"])
        existing: dict[str, dict[str, Any]] = {}
        for row in rows:
            path = _observation_path(inputs["signal_date"], row["factor_alias"])
            stored = store.read_optional(path)
            if stored is None:
                continue
            observation = validate_factor_production_observation(stored.data)
            if not _matches_inputs(observation, inputs, row):
                raise _fail("Factor production observation immutable path conflicts")
            existing[row["factor_alias"]] = observation
        if existing:
            stamp = next(iter(existing.values()))["payload"]["registered_at"]
        else:
            stamp = _stamp(registered_at, label="registered_at")
        observations = []
        created_count = 0
        for row in rows:
            alias = row["factor_alias"]
            path = _observation_path(inputs["signal_date"], alias)
            selected_observation = existing.get(alias)
            if selected_observation is None:
                selected_observation = build_factor_production_observation(
                    inputs=inputs, factor_row=row, registered_at=stamp
                )
                stored = store.write_exact_once(path, canonical_json_bytes(selected_observation))
                selected_observation = validate_factor_production_observation(stored.data)
                created_count += 1
            observations.append(
                {
                    "factor_alias": alias,
                    "factor_id": row["factor_id"],
                    "observation_id": selected_observation["payload"][
                        "factor_production_observation_id"
                    ],
                    "observation_path": str(path),
                    "observation_sha256": hashlib.sha256(
                        canonical_json_bytes(selected_observation)
                    ).hexdigest(),
                    "state": "OPEN",
                }
            )
    return {
        "command_status": "REGISTERED" if created_count else "NO_ACTION",
        "authority_domain": "FACTOR_PRODUCTION_OBSERVATION_ONLY",
        "signal_date": inputs["signal_date"],
        "factor_generation_id": inputs["factor_generation_id"],
        "factor_generation_sha256": inputs["factor_generation_sha256"],
        "factor_pointer_sha256": inputs["factor_pointer_sha256"],
        "planned_horizons": list(PLANNED_HORIZONS),
        "created_count": created_count,
        "observations": observations,
        "prospective_admission_state": "NOT_CLAIMED",
        "outcome_state": "WAITING_FOR_FUTURE_SESSIONS",
        "system_authority": "NONE",
        "mainline_authority": "NONE",
        "investment_authority": "NONE",
        "portfolio_authority": "NONE",
        "broker_authority": "NONE",
        "order_authority": "NONE",
        "trade_authority": "NONE",
    }


def register_factor_production_observations(workspace_root: str) -> dict[str, Any]:
    """Register LOW/W80 observations atomically with code-owned current time."""

    return _register_factor_production_observations(
        workspace_root,
        registered_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


__all__ = [
    "FACTOR_PRODUCTION_OBSERVATION_KIND",
    "PLANNED_HORIZONS",
    "build_factor_production_observation",
    "register_factor_production_observations",
    "validate_factor_production_observation",
]
