"""Typed PIT generation catalog builders for admitted V17 v4 sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import Any, Final, NoReturn

from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_bytes,
    canonical_resource_bytes,
    load_canonical_artifact,
    seal_semantic,
    validate_artifact,
)

from .pit_admission import AdmittedPitClosure, REQUIRED_ROLES
from .security_directory import SourceAdmissionError
from .source_storage import (
    PIT_CATALOG_LOCK,
    PIT_CATALOG_POINTER,
    PIT_CATALOG_ROOT,
    SourceStore,
)

CATALOG_VERSION: Final = "myquant.v17.v4.pit-generation-catalog.v1"
POINTER_VERSION: Final = "myquant.v17.v4.pit-catalog-pointer.v1"
_NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "mainline_authority": False,
    "order": False,
    "production": False,
    "research_only": True,
    "trade": False,
}
_REF_KEYS: Final = {
    "artifact_id",
    "artifact_version",
    "byte_sha256",
    "cutoff",
    "relative_path",
    "semantic_sha256",
    "strategy_id",
}


@dataclass(frozen=True)
class PublishedPitCatalog:
    catalog_path: str
    catalog_byte_sha256: str
    pointer_path: str
    pointer_byte_sha256: str


def _blocked() -> NoReturn:
    raise SourceAdmissionError() from None


def _ordered_refs(
    values: Mapping[str, Mapping[str, Any]],
    *,
    strategy_id: str,
    cutoff: str,
    kind: str,
) -> dict[str, dict[str, Any]]:
    if type(values) is not dict or set(values) != set(REQUIRED_ROLES):
        _blocked()
    result: dict[str, dict[str, Any]] = {}
    for role in REQUIRED_ROLES:
        value = values[role]
        if type(value) is not dict or set(value) != _REF_KEYS:
            _blocked()
        expected_version = (
            f"myquant.v17.v4.dataset.{role}.v1"
            if kind == "dataset"
            else f"myquant.v17.v4.expected-keys.{role}.v1"
        )
        if (
            value["artifact_version"] != expected_version
            or value["strategy_id"] != strategy_id
            or value["cutoff"] > cutoff
        ):
            _blocked()
        result[role] = dict(value)
    return result


def build_pit_generation_catalog(
    admitted: AdmittedPitClosure,
    *,
    catalog_id: str,
    generation_id: str,
    strategy_id: str,
    dataset_refs: Mapping[str, Mapping[str, Any]],
    expected_key_inventory_refs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and immediately validate one immutable catalog artifact."""

    if not isinstance(admitted, AdmittedPitClosure):
        _blocked()
    data_refs = _ordered_refs(
        dataset_refs,
        strategy_id=strategy_id,
        cutoff=admitted.decision_cutoff,
        kind="dataset",
    )
    key_refs = _ordered_refs(
        expected_key_inventory_refs,
        strategy_id=strategy_id,
        cutoff=admitted.decision_cutoff,
        kind="expected-keys",
    )
    source_closure = {
        "admission_closure_sha256": admitted.closure_sha256,
        "dataset_refs": data_refs,
        "expected_key_inventory_refs": key_refs,
    }
    payload = seal_semantic(
        {
            "version": CATALOG_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "catalog_id": catalog_id,
            "generation_id": generation_id,
            "strategy_id": strategy_id,
            "history_start": admitted.history_start,
            "decision_session": admitted.decision_session,
            "cutoff": admitted.decision_cutoff,
            "dataset_refs": data_refs,
            "expected_key_inventory_refs": key_refs,
            "dataset_summaries": [
                dataset.as_dict() for dataset in admitted.datasets
            ],
            "admission_closure_sha256": admitted.closure_sha256,
            "source_closure_sha256": hashlib.sha256(
                canonical_bytes(source_closure)
            ).hexdigest(),
            "authority": dict(_NO_AUTHORITY),
        }
    )
    validate_artifact(payload)
    return payload


def build_pit_catalog_pointer(
    *,
    pointer_id: str,
    strategy_id: str,
    cutoff: str,
    updated_at: str,
    catalog_ref: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a typed v4-only pointer; this function performs no write."""

    if type(catalog_ref) is not dict or set(catalog_ref) != _REF_KEYS:
        _blocked()
    if (
        catalog_ref["artifact_version"] != CATALOG_VERSION
        or catalog_ref["strategy_id"] != strategy_id
        or catalog_ref["cutoff"] != cutoff
    ):
        _blocked()
    payload = seal_semantic(
        {
            "version": POINTER_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "pointer_id": pointer_id,
            "strategy_id": strategy_id,
            "cutoff": cutoff,
            "updated_at": updated_at,
            "state": "PIT_CATALOG_ACTIVE",
            "catalog_ref": dict(catalog_ref),
            "authority": dict(_NO_AUTHORITY),
        }
    )
    validate_artifact(payload)
    return payload


def publish_pit_generation_catalog(
    store: SourceStore,
    *,
    catalog: Mapping[str, Any],
    expected_pointer_sha256: str,
    updated_at: str,
) -> PublishedPitCatalog:
    """Read back every source ref before catalog publication and pointer CAS."""

    if not isinstance(store, SourceStore) or type(catalog) is not dict:
        _blocked()
    validated = validate_artifact(catalog)
    if validated.version != CATALOG_VERSION:
        _blocked()
    for field in ("dataset_refs", "expected_key_inventory_refs"):
        refs = catalog[field]
        for role in REQUIRED_ROLES:
            reference = refs[role]
            store.verify_sha256(
                reference["relative_path"],
                reference["byte_sha256"],
            )
    catalog_raw = canonical_resource_bytes(catalog)
    catalog_path = (
        PIT_CATALOG_ROOT
        / "generations"
        / f"{catalog['generation_id']}.json"
    )
    catalog_write = store.write_exact_once(catalog_path, catalog_raw)
    catalog_ref = {
        "artifact_id": catalog["catalog_id"],
        "artifact_version": catalog["version"],
        "byte_sha256": catalog_write.byte_sha256,
        "cutoff": catalog["cutoff"],
        "relative_path": str(catalog_path),
        "semantic_sha256": catalog["semantic_sha256"],
        "strategy_id": catalog["strategy_id"],
    }
    pointer = build_pit_catalog_pointer(
        pointer_id=f"pit-pointer-{catalog['generation_id']}",
        strategy_id=catalog["strategy_id"],
        cutoff=catalog["cutoff"],
        updated_at=updated_at,
        catalog_ref=catalog_ref,
    )
    pointer_raw = canonical_resource_bytes(pointer)
    with store.locked(PIT_CATALOG_LOCK):
        pointer_write = store.replace_cas(
            PIT_CATALOG_POINTER,
            expected_pointer_sha256,
            pointer_raw,
        )
        readback = store.read(
            PIT_CATALOG_POINTER,
            pointer_write.byte_sha256,
        )
        load_canonical_artifact(
            readback,
            expected_version=POINTER_VERSION,
            label="V17 v4 PIT catalog pointer",
        )
    return PublishedPitCatalog(
        catalog_path=str(catalog_path),
        catalog_byte_sha256=catalog_write.byte_sha256,
        pointer_path=str(PIT_CATALOG_POINTER),
        pointer_byte_sha256=pointer_write.byte_sha256,
    )


__all__ = [
    "CATALOG_VERSION",
    "POINTER_VERSION",
    "PublishedPitCatalog",
    "build_pit_catalog_pointer",
    "build_pit_generation_catalog",
    "publish_pit_generation_catalog",
]
