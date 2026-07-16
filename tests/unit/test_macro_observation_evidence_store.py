from __future__ import annotations

import calendar
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import pytest

from quant_investor.macro.contracts import MacroObservation, canonical_hash
from quant_investor.macro import store as observation_store
from quant_investor.macro.store import (
    MacroObservationStoreError,
    load_observations,
    pointer_sha256,
    publish_observations,
)


def _row(
    *,
    period: str = "2026-04-30",
    available: str = "2026-05-01T01:30:00+00:00",
    value: float = 50.4,
    vintage: str = "initial",
    source_record_id: str = "t20260501_fixture",
) -> dict[str, object]:
    return {
        "indicator_id": "cn.pmi_manufacturing",
        "dimension_type": "national",
        "industry_chain": "",
        "period_end": period,
        "release_at": available,
        "available_at": available,
        "vintage_id": vintage,
        "value": value,
        "unit": "index",
        "frequency": "monthly",
        "source_system": "nbs_official",
        "source_record_id": source_record_id,
        "source_url": "https://www.stats.gov.cn/fixture/pmi.html",
        "fetched_at": available,
        "quality_status": "pass",
    }


def _evidence(
    row: dict[str, object],
    body: bytes = b"<html>official fixture</html>",
    *,
    extension: str = ".html",
) -> tuple[
    dict[str, bytes],
    dict[str, dict[str, Any]],
    dict[str, list[str]],
]:
    content_hash = MacroObservation.from_mapping(row).content_hash
    digest = hashlib.sha256(body).hexdigest()
    return (
        {digest: body},
        {
            digest: {
                "extension": extension,
                "parser_id": "fixture-parser.v1",
                "source_record_id": row["source_record_id"],
                "source_system": row["source_system"],
                "source_url": row["source_url"],
            }
        },
        {content_hash: [digest]},
    )


def _publish_with_evidence(
    root: Path,
    row: dict[str, object],
    *,
    run_id: str,
    body: bytes = b"<html>official fixture</html>",
) -> dict[str, Any]:
    bodies, metadata, mapping = _evidence(row, body)
    return publish_observations(
        [row],
        root=root,
        run_id=run_id,
        evidence_bytes=bodies,
        evidence_metadata=metadata,
        observation_evidence=mapping,
    )


def _pointer_and_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer = json.loads((root / "_latest.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (root / pointer["manifest_path"]).read_text(encoding="utf-8")
    )
    return pointer, manifest


def _rewrite_manifest(
    root: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    pointer, manifest = _pointer_and_manifest(root)
    mutate(manifest)
    manifest_path = root / pointer["manifest_path"]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.chmod(manifest_path, 0o600)
    pointer["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    pointer_path = root / "_latest.json"
    pointer_path.write_text(
        json.dumps(pointer, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.chmod(pointer_path, 0o600)


def test_evidence_sidecar_v2_round_trip_keeps_pointer_v1(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    row = _row()

    result = _publish_with_evidence(root, row, run_id="g1")
    rows, loaded = load_observations(root)
    pointer, manifest = _pointer_and_manifest(root)

    content_hash = MacroObservation.from_mapping(row).content_hash
    digest = next(iter(manifest["observation_evidence"][content_hash]))
    evidence_path = root / pointer["table_path"]
    evidence_path = (
        evidence_path.parent / manifest["evidence_files"][0]["path"]
    )
    assert result["promoted"] is True
    assert pointer["schema_version"] == "macro-observation-pointer.v1"
    assert manifest["schema_version"] == "macro-observation-generation.v2"
    assert manifest["evidence_file_count"] == 1
    assert manifest["evidence_set_sha256"] == canonical_hash(
        {"evidence_files": manifest["evidence_files"]}
    )
    assert manifest["observation_evidence"] == {content_hash: [digest]}
    assert evidence_path.name == f"{digest}.html"
    assert evidence_path.read_bytes() == b"<html>official fixture</html>"
    assert os.stat(evidence_path).st_mode & 0o777 == 0o600
    assert rows[0]["content_hash"] == content_hash
    assert loaded["generation_manifest"]["evidence_set_sha256"] == manifest[
        "evidence_set_sha256"
    ]
    before = pointer_sha256(root)
    duplicate = _publish_with_evidence(root, row, run_id="unused")
    assert duplicate["promoted"] is False
    assert duplicate["reason"] == "all_rows_exist"
    assert pointer_sha256(root) == before
    assert not (root / "_generations" / "unused").exists()
    duplicate_without_evidence = publish_observations(
        [row],
        root=root,
        run_id="unused-without-evidence",
    )
    assert duplicate_without_evidence["promoted"] is False
    assert duplicate_without_evidence["reason"] == "all_rows_exist"
    assert pointer_sha256(root) == before


def test_legacy_v1_generation_remains_loadable(tmp_path: Path) -> None:
    root = tmp_path / "observations"
    publish_observations([_row()], root=root, run_id="g1")

    rows, loaded = load_observations(root)

    assert len(rows) == 1
    assert loaded["generation_manifest"]["schema_version"] == (
        "macro-observation-generation.v1"
    )


def test_legacy_v1_chain_can_append_without_evidence(tmp_path: Path) -> None:
    root = tmp_path / "observations"
    publish_observations([_row()], root=root, run_id="g1")
    second = _row(
        period="2026-05-31",
        available="2026-05-31T01:30:00+00:00",
        source_record_id="t20260531_fixture",
    )

    promoted = publish_observations([second], root=root, run_id="g2")
    rows, loaded = load_observations(root)

    assert promoted["promoted"] is True
    assert len(rows) == 2
    assert loaded["generation_manifest"]["schema_version"] == (
        "macro-observation-generation.v1"
    )


def test_empty_root_publishes_and_loads_39_rows_with_13_evidence_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    rows: list[dict[str, object]] = []
    for offset in range(39):
        year = 2023 + offset // 12
        month = offset % 12 + 1
        last_day = calendar.monthrange(year, month)[1]
        period = f"{year:04d}-{month:02d}-{last_day:02d}"
        rows.append(
            _row(
                period=period,
                available=f"{period}T23:59:00+00:00",
                value=40.0 + offset / 10,
                source_record_id=f"official-record-{offset:02d}",
            )
        )
    bodies: dict[str, bytes] = {}
    metadata: dict[str, dict[str, Any]] = {}
    evidence_digests: list[str] = []
    for index in range(13):
        body = f"<html>official-page-{index:02d}</html>".encode()
        digest = hashlib.sha256(body).hexdigest()
        bodies[digest] = body
        metadata[digest] = {"extension": ".html"}
        evidence_digests.append(digest)
    mapping = {
        MacroObservation.from_mapping(row).content_hash: [
            evidence_digests[index % 13]
        ]
        for index, row in enumerate(rows)
    }

    promoted = publish_observations(
        rows,
        root=root,
        run_id="official-39",
        expected_pointer_sha256="",
        evidence_bytes=bodies,
        evidence_metadata=metadata,
        observation_evidence=mapping,
    )
    loaded_rows, loaded = load_observations(root)
    manifest = loaded["generation_manifest"]

    assert promoted["promoted"] is True
    assert len(loaded_rows) == 39
    assert manifest["evidence_file_count"] == 13
    assert len(manifest["observation_evidence"]) == 39
    assert {
        item["sha256"] for item in manifest["evidence_files"]
    } == set(bodies)

    with pytest.raises(
        MacroObservationStoreError,
        match="pointer_cas_mismatch",
    ):
        publish_observations(
            [
                _row(
                    period="2026-04-30",
                    available="2026-04-30T23:59:00+00:00",
                    source_record_id="official-record-39",
                )
            ],
            root=root,
            run_id="stale-cas",
            expected_pointer_sha256="",
        )


def test_existing_v1_row_can_be_enriched_without_row_duplication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    row = _row()
    publish_observations([row], root=root, run_id="g1")

    promoted = _publish_with_evidence(root, row, run_id="g2")
    rows, loaded = load_observations(root)

    assert promoted["promoted"] is True
    assert promoted["row_count"] == 1
    assert len(rows) == 1
    assert loaded["generation_manifest"]["added_content_hashes"] == []
    assert loaded["generation_manifest"]["schema_version"] == (
        "macro-observation-generation.v2"
    )


def test_v2_append_preserves_existing_evidence_bytes_and_mapping(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    first = _row()
    _publish_with_evidence(root, first, run_id="g1")
    _, first_manifest = _pointer_and_manifest(root)
    second = _row(
        period="2026-05-31",
        available="2026-05-31T01:30:00+00:00",
        source_record_id="t20260531_fixture",
    )

    _publish_with_evidence(
        root,
        second,
        run_id="g2",
        body=b"<html>second official fixture</html>",
    )
    rows, loaded = load_observations(root)
    manifest = loaded["generation_manifest"]

    assert len(rows) == 2
    assert manifest["schema_version"] == "macro-observation-generation.v2"
    assert manifest["evidence_file_count"] == 2
    assert set(first_manifest["observation_evidence"]).issubset(
        manifest["observation_evidence"]
    )
    assert set(
        item["sha256"] for item in first_manifest["evidence_files"]
    ).issubset(item["sha256"] for item in manifest["evidence_files"])


def test_v2_new_row_without_incoming_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    first = _row(source_record_id="fixed-record")
    _publish_with_evidence(root, first, run_id="g1", body=b"official-v1")
    before = (root / "_latest.json").read_bytes()
    revised = _row(
        source_record_id="fixed-record",
        vintage="revision-without-evidence",
        value=50.5,
    )

    with pytest.raises(
        MacroObservationStoreError,
        match="v2_new_rows_require_evidence",
    ):
        publish_observations([revised], root=root, run_id="g2")

    assert (root / "_latest.json").read_bytes() == before
    assert not (root / "_generations" / "g2").exists()


def test_same_official_record_with_different_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    first = _row(source_record_id="fixed-record")
    _publish_with_evidence(root, first, run_id="g1", body=b"official-v1")
    before = pointer_sha256(root)
    revised = _row(
        source_record_id="fixed-record",
        vintage="revision-1",
        value=50.5,
    )
    bodies, metadata, mapping = _evidence(revised, b"official-v2")

    with pytest.raises(
        MacroObservationStoreError,
        match="official_source_record_evidence_drift",
    ):
        publish_observations(
            [revised],
            root=root,
            run_id="g2",
            evidence_bytes=bodies,
            evidence_metadata=metadata,
            observation_evidence=mapping,
        )

    assert pointer_sha256(root) == before
    assert not (root / "_generations" / "g2").exists()


def test_same_official_record_with_same_evidence_allows_new_vintage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    body = b"same-official-entity"
    first = _row(source_record_id="fixed-record")
    _publish_with_evidence(root, first, run_id="g1", body=body)
    revised = _row(
        source_record_id="fixed-record",
        vintage="parser-correction",
        value=50.5,
    )

    promoted = _publish_with_evidence(
        root,
        revised,
        run_id="g2",
        body=body,
    )

    assert promoted["promoted"] is True
    rows, _ = load_observations(root)
    assert len(rows) == 2


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda root, path: path.write_bytes(path.read_bytes() + b"tamper"),
            "evidence_size_mismatch|evidence_hash_mismatch",
        ),
        (
            lambda root, path: os.chmod(path, 0o644),
            "evidence_permissions_unsafe",
        ),
        (
            lambda root, path: _rewrite_manifest(
                root,
                lambda manifest: manifest.__setitem__(
                    "evidence_set_sha256", "0" * 64
                ),
            ),
            "evidence_set_hash_mismatch",
        ),
        (
            lambda root, path: _rewrite_manifest(
                root,
                lambda manifest: manifest["evidence_files"][0].__setitem__(
                    "size_bytes",
                    manifest["evidence_files"][0]["size_bytes"] + 1,
                ),
            ),
            "evidence_size_mismatch",
        ),
        (
            lambda root, path: _rewrite_manifest(
                root,
                lambda manifest: manifest["evidence_files"][0][
                    "metadata"
                ].__setitem__("parser_id", "tampered"),
            ),
            "evidence_metadata_hash_mismatch",
        ),
        (
            lambda root, path: _rewrite_manifest(
                root,
                lambda manifest: manifest["evidence_files"][0].__setitem__(
                    "path", "../raw/escape.html"
                ),
            ),
            "evidence_path_unsafe",
        ),
    ],
)
def test_v2_loader_fails_closed_on_evidence_tampering(
    tmp_path: Path,
    mutation: Callable[[Path, Path], None],
    error: str,
) -> None:
    root = tmp_path / "observations"
    _publish_with_evidence(root, _row(), run_id="g1")
    pointer, manifest = _pointer_and_manifest(root)
    evidence_path = (
        root
        / Path(pointer["table_path"]).parent
        / manifest["evidence_files"][0]["path"]
    )

    mutation(root, evidence_path)

    with pytest.raises(MacroObservationStoreError, match=error):
        load_observations(root)


def test_v2_loader_rejects_evidence_file_symlink(tmp_path: Path) -> None:
    root = tmp_path / "observations"
    _publish_with_evidence(root, _row(), run_id="g1")
    pointer, manifest = _pointer_and_manifest(root)
    evidence_path = (
        root
        / Path(pointer["table_path"]).parent
        / manifest["evidence_files"][0]["path"]
    )
    backup = evidence_path.parents[2] / "backup.html"
    backup.write_bytes(evidence_path.read_bytes())
    os.chmod(backup, 0o600)
    evidence_path.unlink()
    evidence_path.symlink_to(backup)

    with pytest.raises(
        MacroObservationStoreError,
        match="evidence_file_unsafe",
    ):
        load_observations(root)


def test_evidence_inputs_require_exact_hashes_and_incoming_mapping(
    tmp_path: Path,
) -> None:
    root = tmp_path / "observations"
    row = _row()
    bodies, metadata, mapping = _evidence(row)
    digest = next(iter(bodies))

    with pytest.raises(MacroObservationStoreError, match="hash_mismatch"):
        publish_observations(
            [row],
            root=root,
            run_id="bad-hash",
            evidence_bytes={"0" * 64: bodies[digest]},
            evidence_metadata={"0" * 64: metadata[digest]},
            observation_evidence={
                next(iter(mapping)): ["0" * 64]
            },
        )
    with pytest.raises(
        MacroObservationStoreError,
        match="observation_set_mismatch",
    ):
        publish_observations(
            [row],
            root=root,
            run_id="bad-observation-map",
            evidence_bytes=bodies,
            evidence_metadata=metadata,
            observation_evidence={"0" * 64: [digest]},
        )
    bad_metadata = {digest: {**metadata[digest], "extension": ".txt"}}
    with pytest.raises(MacroObservationStoreError, match="extension_invalid"):
        publish_observations(
            [row],
            root=root,
            run_id="bad-extension",
            evidence_bytes=bodies,
            evidence_metadata=bad_metadata,
            observation_evidence=mapping,
        )


def test_post_write_evidence_readback_failure_rolls_back_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "observations"
    publish_observations([_row()], root=root, run_id="g1")
    before = pointer_sha256(root)
    second = _row(
        period="2026-05-31",
        available="2026-05-31T01:30:00+00:00",
        source_record_id="t20260531_fixture",
    )
    bodies, metadata, mapping = _evidence(second)
    original = observation_store._validate_generation_evidence

    def fail_after_rename(
        generation: Path,
        manifest: dict[str, Any],
        *,
        include_bytes: bool = False,
    ):
        if generation.name == "g2":
            raise MacroObservationStoreError("injected_final_readback_failure")
        return original(
            generation,
            manifest,
            include_bytes=include_bytes,
        )

    monkeypatch.setattr(
        observation_store,
        "_validate_generation_evidence",
        fail_after_rename,
    )
    with pytest.raises(
        MacroObservationStoreError,
        match="injected_final_readback_failure",
    ):
        publish_observations(
            [second],
            root=root,
            run_id="g2",
            evidence_bytes=bodies,
            evidence_metadata=metadata,
            observation_evidence=mapping,
        )

    assert pointer_sha256(root) == before
    assert not (root / "_generations" / "g2").exists()


def test_pointer_post_replace_fsync_failure_restores_exact_previous_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "observations"
    publish_observations([_row()], root=root, run_id="g1")
    pointer_path = root / "_latest.json"
    before = pointer_path.read_bytes()
    before_sha = pointer_sha256(root)
    second = _row(
        period="2026-05-31",
        available="2026-05-31T01:30:00+00:00",
        source_record_id="t20260531_fixture",
    )
    original = observation_store._fsync_directory
    failures = 0

    def fail_once_after_pointer_replace(path: Path) -> None:
        nonlocal failures
        if path == root and failures == 0:
            failures += 1
            raise OSError("injected_pointer_directory_fsync_failure")
        original(path)

    monkeypatch.setattr(
        observation_store,
        "_fsync_directory",
        fail_once_after_pointer_replace,
    )
    with pytest.raises(
        OSError,
        match="injected_pointer_directory_fsync_failure",
    ):
        publish_observations([second], root=root, run_id="g2")

    assert failures == 1
    assert pointer_path.read_bytes() == before
    assert pointer_sha256(root) == before_sha
    assert not (root / "_generations" / "g2").exists()
    rows, loaded = load_observations(root)
    assert len(rows) == 1
    assert loaded["generation_id"] == "g1"


def test_first_pointer_post_replace_fsync_failure_restores_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "observations"
    original = observation_store._fsync_directory
    failures = 0

    def fail_once_after_pointer_replace(path: Path) -> None:
        nonlocal failures
        if path == root and failures == 0:
            failures += 1
            raise OSError("injected_first_pointer_fsync_failure")
        original(path)

    monkeypatch.setattr(
        observation_store,
        "_fsync_directory",
        fail_once_after_pointer_replace,
    )
    with pytest.raises(
        OSError,
        match="injected_first_pointer_fsync_failure",
    ):
        publish_observations([_row()], root=root, run_id="g1")

    assert failures == 1
    assert not (root / "_latest.json").exists()
    assert not (root / "_generations" / "g1").exists()
