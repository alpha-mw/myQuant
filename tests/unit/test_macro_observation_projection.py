from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest

from quant_investor.macro import store as observation_store
from quant_investor.macro.contracts import MacroObservation, canonical_hash
from quant_investor.macro.store import (
    MacroObservationStoreError,
    load_observations,
    pointer_sha256,
    publish_observation_projection,
    publish_observations,
)


def _row(
    name: str,
    period: str,
    *,
    value: float,
    local: bool = False,
    vintage: str = "initial",
    source_record_id: str = "",
) -> dict[str, object]:
    available_at = f"{period}T15:00:00+00:00"
    source_system = "local_strict_parquet" if local else "nbs_official"
    source_url = (
        f"local://strict-parquet/cn/macro/{name}.parquet"
        if local
        else f"https://www.stats.gov.cn/fixture/{name}.html"
    )
    return {
        "indicator_id": (
            "market.breadth" if local else "cn.pmi_manufacturing"
        ),
        "dimension_type": "market_confirmation" if local else "national",
        "industry_chain": "",
        "period_end": period,
        "release_at": available_at,
        "available_at": available_at,
        "vintage_id": vintage,
        "value": value,
        "unit": "%" if local else "index",
        "frequency": "daily" if local else "monthly",
        "source_system": source_system,
        "source_record_id": source_record_id or f"record-{name}-{period}",
        "source_url": source_url,
        "fetched_at": available_at,
        "quality_status": "pass",
    }


def _content_hash(row: Mapping[str, object]) -> str:
    return MacroObservation.from_mapping(row).content_hash


def _evidence_bundle(
    rows: Iterable[Mapping[str, object]],
    *,
    body_by_hash: Mapping[str, bytes] | None = None,
    parser_id: str = "projection-fixture.v1",
) -> tuple[
    dict[str, bytes],
    dict[str, dict[str, Any]],
    dict[str, list[str]],
]:
    bodies: dict[str, bytes] = {}
    metadata: dict[str, dict[str, Any]] = {}
    mapping: dict[str, list[str]] = {}
    for row in rows:
        content_hash = _content_hash(row)
        body = (
            body_by_hash[content_hash]
            if body_by_hash is not None
            else f"<html>{content_hash}</html>".encode()
        )
        digest = hashlib.sha256(body).hexdigest()
        bodies[digest] = body
        metadata[digest] = {
            "extension": ".html",
            "parser_id": parser_id,
        }
        mapping[content_hash] = [digest]
    return bodies, metadata, mapping


def _publish_parent(
    root: Path,
    rows: list[dict[str, object]],
    *,
    body_by_hash: Mapping[str, bytes] | None = None,
    run_id: str = "parent",
) -> dict[str, Any]:
    bodies, metadata, mapping = _evidence_bundle(
        rows,
        body_by_hash=body_by_hash,
    )
    return publish_observations(
        rows,
        root=root,
        run_id=run_id,
        expected_pointer_sha256="",
        metadata={"stage": "parent"},
        evidence_bytes=bodies,
        evidence_metadata=metadata,
        observation_evidence=mapping,
    )


def _project(
    root: Path,
    incoming: list[dict[str, object]],
    *,
    expected_pointer_sha256: str,
    retained: Iterable[str],
    removed: Iterable[str],
    run_id: str = "child",
    metadata: Mapping[str, Any] | None = None,
    body_by_hash: Mapping[str, bytes] | None = None,
    parser_id: str = "projection-fixture.v1",
    precommit_validator=None,
) -> dict[str, Any]:
    bodies, evidence_metadata, mapping = _evidence_bundle(
        incoming,
        body_by_hash=body_by_hash,
        parser_id=parser_id,
    )
    return publish_observation_projection(
        incoming,
        root=root,
        run_id=run_id,
        expected_pointer_sha256=expected_pointer_sha256,
        retained_parent_content_hashes=retained,
        removed_parent_content_hashes=removed,
        metadata=metadata or {"stage": "child"},
        evidence_bytes=bodies,
        evidence_metadata=evidence_metadata,
        observation_evidence=mapping,
        precommit_validator=precommit_validator,
    )


def _pointer_and_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer = json.loads((root / "_latest.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (root / pointer["manifest_path"]).read_text(encoding="utf-8")
    )
    return pointer, manifest


def _generation_snapshot(path: Path) -> dict[str, tuple[int, bytes | None]]:
    return {
        str(item.relative_to(path)): (
            stat.S_IMODE(os.lstat(item).st_mode),
            item.read_bytes() if item.is_file() else None,
        )
        for item in sorted(path.rglob("*"))
    }


def _three_row_parent() -> tuple[
    list[dict[str, object]],
    dict[str, bytes],
]:
    old_a = _row("old-a", "2026-01-31", value=49.8)
    old_b = _row("old-b", "2026-02-28", value=50.0)
    local = _row(
        "local-breadth",
        "2026-03-31",
        value=0.63,
        local=True,
    )
    bodies = {
        _content_hash(old_a): b"<html>old-a</html>",
        _content_hash(old_b): b"<html>old-b</html>",
        _content_hash(local): b"local-parquet-evidence",
    }
    return [old_a, old_b, local], bodies


def test_projection_selectively_keeps_local_row_and_exact_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    parent_rows, parent_bodies = _three_row_parent()
    _publish_parent(root, parent_rows, body_by_hash=parent_bodies)
    parent_sha = pointer_sha256(root)
    parent_pointer, parent_manifest = _pointer_and_manifest(root)
    parent_generation = root / "_generations" / parent_pointer["generation_id"]
    parent_snapshot = _generation_snapshot(parent_generation)
    local_hash = _content_hash(parent_rows[2])
    removed_hashes = {_content_hash(row) for row in parent_rows[:2]}
    old_digests = {
        digest
        for content_hash in removed_hashes
        for digest in parent_manifest["observation_evidence"][content_hash]
    }
    local_digest = parent_manifest["observation_evidence"][local_hash][0]
    new_a = _row("new-a", "2026-04-30", value=50.2)
    new_b = _row("new-b", "2026-05-31", value=50.4)
    shared_body = b"<html>one-reviewed-release-page</html>"
    incoming_bodies = {
        _content_hash(new_a): shared_body,
        _content_hash(new_b): shared_body,
    }

    result = _project(
        root,
        [new_a, new_b],
        expected_pointer_sha256=parent_sha,
        retained=[local_hash],
        removed=removed_hashes,
        body_by_hash=incoming_bodies,
    )
    rows, loaded = load_observations(root)
    manifest = loaded["generation_manifest"]
    child_digests = {item["sha256"] for item in manifest["evidence_files"]}
    shared_digest = hashlib.sha256(shared_body).hexdigest()

    assert result["promoted"] is True
    assert {row["content_hash"] for row in rows} == {
        local_hash,
        _content_hash(new_a),
        _content_hash(new_b),
    }
    assert child_digests == {local_digest, shared_digest}
    assert child_digests.isdisjoint(old_digests)
    assert manifest["added_content_hashes"] == sorted(
        [_content_hash(new_a), _content_hash(new_b)]
    )
    assert manifest["removed_content_hashes"] == sorted(removed_hashes)
    assert len(manifest["evidence_files"]) == 2
    assert _generation_snapshot(parent_generation) == parent_snapshot


def test_projection_child_permissions_hashes_and_readback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    parent = _row("parent", "2026-01-31", value=49.8)
    incoming = _row("incoming", "2026-02-28", value=50.1)
    _publish_parent(root, [parent])
    result = _project(
        root,
        [incoming],
        expected_pointer_sha256=pointer_sha256(root),
        retained=[],
        removed=[_content_hash(parent)],
    )
    rows, loaded = load_observations(root)
    pointer, manifest = _pointer_and_manifest(root)
    generation = root / "_generations" / pointer["generation_id"]

    assert [row["content_hash"] for row in rows] == [_content_hash(incoming)]
    assert loaded["generation_manifest"] == manifest
    assert manifest["schema_version"] == "macro-observation-generation.v2"
    assert manifest["content_set_hash"] == canonical_hash(
        {"hashes": [_content_hash(incoming)]}
    )
    assert manifest["evidence_set_sha256"] == canonical_hash(
        {"evidence_files": manifest["evidence_files"]}
    )
    assert result["generation_manifest"] == manifest
    assert stat.S_IMODE(os.lstat(generation).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(generation / "evidence").st_mode) == 0o700
    assert (
        stat.S_IMODE(os.lstat(generation / "evidence" / "raw").st_mode)
        == 0o700
    )
    for file_path in [
        generation / "observations.parquet",
        generation / "manifest.json",
        *(generation / "evidence" / "raw").iterdir(),
    ]:
        assert stat.S_IMODE(os.lstat(file_path).st_mode) == 0o600
    assert stat.S_IMODE(os.lstat(root / "_latest.json").st_mode) == 0o600


def test_projection_rejects_stale_cas_and_unknown_retained_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    parent = _row("parent", "2026-01-31", value=49.8)
    incoming = _row("incoming", "2026-02-28", value=50.1)
    _publish_parent(root, [parent])
    before = (root / "_latest.json").read_bytes()

    with pytest.raises(
        MacroObservationStoreError,
        match="pointer_cas_mismatch",
    ):
        _project(
            root,
            [incoming],
            expected_pointer_sha256="0" * 64,
            retained=[],
            removed=[_content_hash(parent)],
        )
    with pytest.raises(
        MacroObservationStoreError,
        match="retained_hash_unknown",
    ):
        _project(
            root,
            [incoming],
            expected_pointer_sha256=pointer_sha256(root),
            retained=["f" * 64],
            removed=[_content_hash(parent)],
        )

    assert (root / "_latest.json").read_bytes() == before
    assert not (root / "_generations" / "child").exists()


def test_projection_rejects_legacy_v1_parent(tmp_path: Path) -> None:
    root = tmp_path / "observations"
    parent = _row("parent", "2026-01-31", value=49.8)
    incoming = _row("incoming", "2026-02-28", value=50.1)
    publish_observations([parent], root=root, run_id="legacy")
    before = (root / "_latest.json").read_bytes()

    with pytest.raises(
        MacroObservationStoreError,
        match="parent_v2_required",
    ):
        _project(
            root,
            [incoming],
            expected_pointer_sha256=pointer_sha256(root),
            retained=[],
            removed=[_content_hash(parent)],
        )

    assert (root / "_latest.json").read_bytes() == before
    assert not (root / "_generations" / "child").exists()


def test_projection_missing_incoming_mapping_is_atomic(tmp_path: Path) -> None:
    root = tmp_path / "observations"
    parent = _row("parent", "2026-01-31", value=49.8)
    incoming = _row("incoming", "2026-02-28", value=50.1)
    _publish_parent(root, [parent])
    before = (root / "_latest.json").read_bytes()
    bodies, metadata, _mapping = _evidence_bundle([incoming])

    with pytest.raises(
        MacroObservationStoreError,
        match="observation_set_mismatch",
    ):
        publish_observation_projection(
            [incoming],
            root=root,
            run_id="child",
            expected_pointer_sha256=pointer_sha256(root),
            retained_parent_content_hashes=[],
            removed_parent_content_hashes=[_content_hash(parent)],
            metadata={"stage": "child"},
            evidence_bytes=bodies,
            evidence_metadata=metadata,
            observation_evidence={},
        )

    assert (root / "_latest.json").read_bytes() == before
    assert not (root / "_generations" / "child").exists()


def test_projection_rejects_parent_evidence_tamper_and_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    parent = _row("parent", "2026-01-31", value=49.8)
    incoming = _row("incoming", "2026-02-28", value=50.1)
    _publish_parent(root, [parent])
    pointer, manifest = _pointer_and_manifest(root)
    evidence_path = (
        root
        / Path(pointer["table_path"]).parent
        / manifest["evidence_files"][0]["path"]
    )
    backup = root / "safe-evidence.html"
    backup.write_bytes(evidence_path.read_bytes())
    os.chmod(backup, 0o600)
    evidence_path.unlink()
    evidence_path.symlink_to(backup)
    before = (root / "_latest.json").read_bytes()

    with pytest.raises(
        MacroObservationStoreError,
        match="evidence_file_unsafe",
    ):
        _project(
            root,
            [incoming],
            expected_pointer_sha256=hashlib.sha256(before).hexdigest(),
            retained=[],
            removed=[_content_hash(parent)],
        )

    assert (root / "_latest.json").read_bytes() == before
    assert not (root / "_generations" / "child").exists()


def test_projection_rejects_digest_metadata_conflict_with_removed_parent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    parent = _row("parent", "2026-01-31", value=49.8)
    incoming = _row("incoming", "2026-02-28", value=50.1)
    shared_body = b"<html>same-digest</html>"
    _publish_parent(
        root,
        [parent],
        body_by_hash={_content_hash(parent): shared_body},
    )
    before = (root / "_latest.json").read_bytes()

    with pytest.raises(
        MacroObservationStoreError,
        match="evidence_metadata_conflict",
    ):
        _project(
            root,
            [incoming],
            expected_pointer_sha256=pointer_sha256(root),
            retained=[],
            removed=[_content_hash(parent)],
            body_by_hash={_content_hash(incoming): shared_body},
            parser_id="conflicting-parser.v2",
        )

    assert (root / "_latest.json").read_bytes() == before


def test_projection_rejects_source_record_evidence_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    parent = _row(
        "shared-record",
        "2026-01-31",
        value=49.8,
        source_record_id="fixed-source-record",
    )
    incoming = _row(
        "shared-record",
        "2026-01-31",
        value=50.1,
        vintage="revision-1",
        source_record_id="fixed-source-record",
    )
    _publish_parent(root, [parent])
    before = pointer_sha256(root)

    with pytest.raises(
        MacroObservationStoreError,
        match="official_source_record_evidence_drift",
    ):
        _project(
            root,
            [incoming],
            expected_pointer_sha256=before,
            retained=[],
            removed=[_content_hash(parent)],
        )

    assert pointer_sha256(root) == before


def test_projection_precommit_failure_rolls_back_child(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    parent = _row("parent", "2026-01-31", value=49.8)
    incoming = _row("incoming", "2026-02-28", value=50.1)
    _publish_parent(root, [parent])
    before = (root / "_latest.json").read_bytes()

    def fail_precommit(rows, manifest) -> None:
        assert [row["content_hash"] for row in rows] == [
            _content_hash(incoming)
        ]
        assert manifest["removed_content_hashes"] == [_content_hash(parent)]
        raise RuntimeError("injected_projection_precommit_failure")

    with pytest.raises(
        RuntimeError,
        match="injected_projection_precommit_failure",
    ):
        _project(
            root,
            [incoming],
            expected_pointer_sha256=pointer_sha256(root),
            retained=[],
            removed=[_content_hash(parent)],
            precommit_validator=fail_precommit,
        )

    assert (root / "_latest.json").read_bytes() == before
    assert not (root / "_generations" / "child").exists()
    assert not any(
        item.name.startswith(".child.")
        for item in (root / "_generations").iterdir()
    )


def test_projection_exact_retry_validates_metadata_and_returns_no_update(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    parent = _row("parent", "2026-01-31", value=49.8)
    incoming = _row("incoming", "2026-02-28", value=50.1)
    _publish_parent(root, [parent])
    parent_sha = pointer_sha256(root)
    call = {
        "expected_pointer_sha256": parent_sha,
        "retained": [],
        "removed": [_content_hash(parent)],
        "run_id": "child",
        "metadata": {"stage": "child", "scope": "exact"},
    }

    first = _project(root, [incoming], **call)
    child_pointer = (root / "_latest.json").read_bytes()
    retry = _project(root, [incoming], **call)

    assert first["promoted"] is True
    assert retry["promoted"] is False
    assert retry["reason"] == "exact_projection_exists"
    assert (root / "_latest.json").read_bytes() == child_pointer
    with pytest.raises(
        MacroObservationStoreError,
        match="projection_retry_mismatch",
    ):
        _project(
            root,
            [incoming],
            **{**call, "metadata": {"stage": "child", "scope": "changed"}},
        )
    assert (root / "_latest.json").read_bytes() == child_pointer


def test_projection_metadata_change_creates_a_new_exact_child(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    parent = _row("parent", "2026-01-31", value=49.8)
    _publish_parent(root, [parent])
    parent_sha = pointer_sha256(root)
    parent_hash = _content_hash(parent)

    result = _project(
        root,
        [],
        expected_pointer_sha256=parent_sha,
        retained=[parent_hash],
        removed=[],
        metadata={"stage": "metadata-only-child"},
    )
    rows, loaded = load_observations(root)

    assert result["promoted"] is True
    assert pointer_sha256(root) != parent_sha
    assert [row["content_hash"] for row in rows] == [parent_hash]
    assert loaded["metadata"] == {"stage": "metadata-only-child"}
    assert loaded["generation_manifest"]["added_content_hashes"] == []
    assert loaded["generation_manifest"]["removed_content_hashes"] == []


def test_projection_rejects_duplicate_rows_and_inexact_parent_partition(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    parent_rows, parent_bodies = _three_row_parent()
    incoming = _row("incoming", "2026-04-30", value=50.1)
    _publish_parent(root, parent_rows, body_by_hash=parent_bodies)
    before = pointer_sha256(root)
    bodies, metadata, mapping = _evidence_bundle([incoming])

    with pytest.raises(MacroObservationStoreError, match="duplicate_row"):
        publish_observation_projection(
            [incoming, incoming],
            root=root,
            run_id="child",
            expected_pointer_sha256=before,
            retained_parent_content_hashes=[],
            removed_parent_content_hashes=[
                _content_hash(row) for row in parent_rows
            ],
            evidence_bytes=bodies,
            evidence_metadata=metadata,
            observation_evidence=mapping,
        )
    with pytest.raises(
        MacroObservationStoreError,
        match="parent_partition_mismatch",
    ):
        _project(
            root,
            [incoming],
            expected_pointer_sha256=before,
            retained=[_content_hash(parent_rows[2])],
            removed=[_content_hash(parent_rows[0])],
        )
    assert pointer_sha256(root) == before


def test_projection_pointer_drift_is_not_clobbered_and_staging_is_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "observations"
    parent = _row("parent", "2026-01-31", value=49.8)
    incoming = _row("incoming", "2026-02-28", value=50.1)
    _publish_parent(root, [parent])
    expected = pointer_sha256(root)
    pointer_path = root / "_latest.json"
    drifted_bytes = b""

    def inject_pointer_drift(_validator, _rows, _manifest) -> None:
        nonlocal drifted_bytes
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        pointer["concurrent_drift_marker"] = True
        drifted_bytes = (
            json.dumps(
                pointer,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode()
        pointer_path.write_bytes(drifted_bytes)
        os.chmod(pointer_path, 0o600)

    monkeypatch.setattr(
        observation_store,
        "_run_precommit_validator",
        inject_pointer_drift,
    )
    with pytest.raises(
        MacroObservationStoreError,
        match="pointer_changed_before_switch",
    ):
        _project(
            root,
            [incoming],
            expected_pointer_sha256=expected,
            retained=[],
            removed=[_content_hash(parent)],
        )

    assert pointer_path.read_bytes() == drifted_bytes
    assert not any(
        item.name.startswith(".child.")
        for item in (root / "_generations").iterdir()
    )
