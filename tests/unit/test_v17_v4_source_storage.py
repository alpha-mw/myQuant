from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from typing import Any

import pytest

from quant_investor.v17_v4_contract import load_canonical_artifact
from quant_investor.v17_v4_runtime.pit_admission import REQUIRED_ROLES
from quant_investor.v17_v4_runtime.pit_catalog import (
    POINTER_VERSION,
    build_pit_generation_catalog,
    publish_pit_generation_catalog,
)
from quant_investor.v17_v4_runtime.source_storage import (
    EMPTY_SHA256,
    PIT_CATALOG_POINTER,
    SOURCE_ROOT,
    SourceCASMismatch,
    SourceStorageSecurityError,
    SourceStore,
)
from tests.unit.test_v17_v4_pit_admission import CUTOFF, _admitted


def _stored_refs(
    store: SourceStore,
    *,
    corrupt_role: str | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = {
        "dataset": {},
        "expected-keys": {},
    }
    for kind in result:
        for role in REQUIRED_ROLES:
            raw = f"{kind}:{role}\n".encode()
            relative_path = (
                f"data/private/v17_v4_sources/{role}/"
                f"{kind}-{role}.bin"
            )
            store.write_exact_once(relative_path, raw)
            digest = hashlib.sha256(raw).hexdigest()
            if role == corrupt_role and kind == "dataset":
                digest = "f" * 64
            result[kind][role] = {
                "artifact_id": f"{kind}-{role}",
                "artifact_version": (
                    f"myquant.v17.v4.{kind}.{role}.v1"
                ),
                "byte_sha256": digest,
                "cutoff": CUTOFF,
                "relative_path": relative_path,
                "semantic_sha256": hashlib.sha256(
                    f"semantic:{kind}:{role}".encode()
                ).hexdigest(),
                "strategy_id": "quant-first",
            }
    return result["dataset"], result["expected-keys"]


def _catalog(
    store: SourceStore,
    *,
    corrupt_role: str | None = None,
) -> dict[str, Any]:
    dataset_refs, key_refs = _stored_refs(
        store,
        corrupt_role=corrupt_role,
    )
    return build_pit_generation_catalog(
        _admitted(),
        catalog_id="pit-catalog-1",
        generation_id="pit-generation-1",
        strategy_id="quant-first",
        dataset_refs=dataset_refs,
        expected_key_inventory_refs=key_refs,
    )


def test_source_store_publishes_exact_catalog_then_cas_pointer(
    tmp_path: Path,
) -> None:
    store = SourceStore(tmp_path)
    store.initialize()
    catalog = _catalog(store)
    result = publish_pit_generation_catalog(
        store,
        catalog=catalog,
        expected_pointer_sha256=EMPTY_SHA256,
        updated_at=CUTOFF,
    )

    pointer_raw = store.read(
        PIT_CATALOG_POINTER,
        result.pointer_byte_sha256,
    )
    pointer = load_canonical_artifact(
        pointer_raw,
        expected_version=POINTER_VERSION,
    )
    assert pointer.payload["catalog_ref"]["byte_sha256"] == (
        result.catalog_byte_sha256
    )
    assert stat.S_IMODE(
        (tmp_path / SOURCE_ROOT).stat().st_mode
    ) == 0o700
    assert stat.S_IMODE(
        (tmp_path / result.catalog_path).stat().st_mode
    ) == 0o600
    assert not (
        tmp_path / "results/research_runtime_control"
    ).exists()


def test_missing_source_readback_writes_no_catalog_or_pointer(
    tmp_path: Path,
) -> None:
    store = SourceStore(tmp_path)
    store.initialize()
    catalog = _catalog(store, corrupt_role="market_bars")
    with pytest.raises(SourceCASMismatch):
        publish_pit_generation_catalog(
            store,
            catalog=catalog,
            expected_pointer_sha256=EMPTY_SHA256,
            updated_at=CUTOFF,
        )
    assert store.read_optional(PIT_CATALOG_POINTER) is None
    assert store.read_optional(
        "data/private/v17_v4_sources/pit_catalog/generations/"
        "pit-generation-1.json"
    ) is None


def test_pointer_cas_mismatch_never_replaces_existing_pointer(
    tmp_path: Path,
) -> None:
    store = SourceStore(tmp_path)
    store.initialize()
    catalog = _catalog(store)
    first = publish_pit_generation_catalog(
        store,
        catalog=catalog,
        expected_pointer_sha256=EMPTY_SHA256,
        updated_at=CUTOFF,
    )
    before = store.read(PIT_CATALOG_POINTER)
    with pytest.raises(SourceCASMismatch):
        publish_pit_generation_catalog(
            store,
            catalog=catalog,
            expected_pointer_sha256=EMPTY_SHA256,
            updated_at=CUTOFF,
        )
    assert store.read(PIT_CATALOG_POINTER) == before
    assert hashlib.sha256(before).hexdigest() == first.pointer_byte_sha256


def test_source_store_rejects_escape_symlink_hardlink_and_casefold_alias(
    tmp_path: Path,
) -> None:
    store = SourceStore(tmp_path)
    store.initialize()
    with pytest.raises(SourceStorageSecurityError):
        store.write_exact_once(
            "data/private/v17_v4_sources/../escape",
            b"x",
        )

    governed = tmp_path / SOURCE_ROOT
    symlink = governed / "symlink.bin"
    symlink.symlink_to(tmp_path / "outside.bin")
    with pytest.raises(SourceStorageSecurityError):
        store.read(str(SOURCE_ROOT / "symlink.bin"))

    original = SOURCE_ROOT / "original.bin"
    store.write_exact_once(original, b"original")
    os.link(
        tmp_path / original,
        governed / "hardlink.bin",
    )
    with pytest.raises(SourceStorageSecurityError):
        store.read(original)

    (governed / "Alias").mkdir(mode=0o700)
    with pytest.raises(SourceStorageSecurityError):
        store.write_exact_once(SOURCE_ROOT / "alias" / "x.bin", b"x")


def test_large_source_sha_readback_is_streamed_beyond_json_read_limit(
    tmp_path: Path,
) -> None:
    store = SourceStore(
        tmp_path,
        max_read_bytes=4,
        max_hash_bytes=64,
    )
    store.initialize()
    path = tmp_path / SOURCE_ROOT / "large.parquet"
    raw = b"PAR1" + b"x" * 24 + b"PAR1"
    path.write_bytes(raw)
    path.chmod(0o600)
    expected = hashlib.sha256(raw).hexdigest()

    verified = store.verify_sha256(SOURCE_ROOT / "large.parquet", expected)
    assert verified.size == len(raw)
    assert verified.byte_sha256 == expected
    with pytest.raises(SourceStorageSecurityError):
        store.read(SOURCE_ROOT / "large.parquet")
