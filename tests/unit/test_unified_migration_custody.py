from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

import pytest

from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.migration.custody import (
    build_authority_archive_plan,
    copy_authority_archive,
)
from quant_investor.migration.errors import (
    ARCHIVE_COPY_CONFLICT,
    ARCHIVE_COPY_FORBIDDEN,
    PERMANENT_MARKER_PRESENT,
    UNIFIED_ACTIVE_PRESENT,
    UnifiedCutoverError,
)
from quant_investor.migration.migration import (
    EMPTY_SHA256,
    build_permanent_marker_payload,
    build_pre_cas_migration_receipt,
    validate_permanent_marker,
    validate_pre_cas_migration_receipt,
    write_pre_cas_migration_receipt,
)
from quant_investor.migration.resolver import INVENTORY_KIND
from quant_investor.migration.rules import (
    BASELINE_CUSTODY_FACTS_SHA256,
    load_rules,
)

from test_unified_migration_helpers import make_test_workspace, sha


CREATED_AT = "2026-08-14T00:00:00Z"


def _workspace(root: Path) -> Path:
    rules_path, _tracked = make_test_workspace(
        root,
        source_files={"src/main.py": b"VALUE = 1\n"},
        entrypoint_seeds=[{"kind": "module", "value": "src.main"}],
    )
    return rules_path


def _source_row(root: Path, relative: str, classification: str, raw: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "relative_path": relative,
        "origin": "RUNTIME" if not relative.endswith(".py") else "TRACKED",
        "file_kind": "JSON" if relative.endswith(".json") else "BYTES",
        "byte_sha256": sha(raw),
        "bytes": len(raw),
        "classification": classification,
        "classification_reason": "TEST_EXACT_CLASSIFICATION",
    }


def _inventory(root: Path) -> dict[str, object]:
    rows = [
        _source_row(
            root,
            "authority/_active.json",
            "ACTIVE_AUTHORITY",
            canonical_json_bytes({"authority": True}),
        ),
        _source_row(root, "caller.py", "ACTIVE_CALLER", b"VALUE = 1\n"),
        _source_row(
            root,
            "shadow/source.json",
            "NON_AUTHORITY_SHADOW",
            canonical_json_bytes({"shadow_only": True}),
        ),
        _source_row(
            root,
            "strategy/source.json",
            "INDEPENDENT_SOURCE",
            canonical_json_bytes({"authority": False}),
        ),
        _source_row(root, "legacy.py", "LEGACY_INACTIVE", b"OLD = True\n"),
        _source_row(root, "custody/archive.bin", "CUSTODY_ONLY", b"historical"),
    ]
    rows.sort(key=lambda row: str(row["relative_path"]))
    zero_ref = {"relative_path": "input.json", "byte_sha256": "0" * 64, "bytes": 0}
    payload = {
        "inventory_id": "inventory-test",
        "status": "COMPLETE",
        "rules_ref": zero_ref,
        "dynamic_import_allowlist_ref": zero_ref,
        "legacy_seed_manifest_ref": zero_ref,
        "legacy_custody_scope_ref": zero_ref,
        "replacement_test_map_ref": zero_ref,
        "bootstrap_decision_ref": zero_ref,
        "tracked_roots": [],
        "runtime_roots": [],
        "files": rows,
        "edges": [],
        "summary": {"blocker_codes": []},
    }
    return seal_artifact(INVENTORY_KIND, payload, created_at=CREATED_AT)


def _generation_manifest() -> dict[str, object]:
    payload = {
        "assembly_id": "assembly-test",
        "generation_state": "SYSTEM_SUSPENDED",
        "contract_catalog_sha256": "1" * 64,
        "release_manifest_ref": {},
        "source_refs": [],
        "factor_policy_ref": None,
        "factor_evidence_refs": [],
        "factor_active_set_ref": None,
        "factor_source_object_refs": [],
        "factor_validation_attestation_ref": None,
        "mainline_ref": None,
        "research_refs": [],
        "migration_receipt_ref": None,
        "migration_marker_ref": None,
        "skill_tree_sha256": "2" * 64,
        "automation_semantic_sha256": "3" * 64,
        "readiness_matrix_ref": {},
        "emergency_controller_sha256": None,
    }
    return seal_artifact("system.generation_manifest", payload, created_at=CREATED_AT)


def _active_pointer(manifest: dict[str, object]) -> dict[str, str]:
    manifest_raw = canonical_json_bytes(manifest)
    return {
        "activated_at": CREATED_AT,
        "generation_id": str(manifest["semantic_sha256"]),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "os_actor": "uid:test",
        "previous_pointer_sha256": EMPTY_SHA256,
    }


def test_archive_plan_copies_only_current_authority_and_seals_mode_0444(
    tmp_path: Path,
) -> None:
    rules_path = _workspace(tmp_path)
    inventory = _inventory(tmp_path)
    plan = build_authority_archive_plan(
        tmp_path,
        inventory,
        cutover_id="cutover-test",
        created_at=CREATED_AT,
        rules_path=rules_path,
    )
    entries = plan["payload"]["entries"]
    assert [row["source_relative_path"] for row in entries] == [
        "authority/_active.json"
    ]
    assert plan["payload"]["summary"]["non_authority_copy_count"] == 0

    with pytest.raises(UnifiedCutoverError) as forbidden:
        copy_authority_archive(tmp_path, plan)
    assert forbidden.value.code == ARCHIVE_COPY_FORBIDDEN

    first = copy_authority_archive(tmp_path, plan, allow_copy=True)
    assert first["copied_file_count"] == 1
    archived = tmp_path / entries[0]["archive_relative_path"]
    assert archived.read_bytes() == (tmp_path / "authority/_active.json").read_bytes()
    assert stat.S_IMODE(os.lstat(archived).st_mode) == 0o444
    assert not (archived.parent.parent / "shadow/source.json").exists()

    second = copy_authority_archive(tmp_path, plan, allow_copy=True)
    assert second["copied_file_count"] == 0
    assert second["already_present_file_count"] == 1
    archived.chmod(0o644)
    with pytest.raises(UnifiedCutoverError) as conflict:
        copy_authority_archive(tmp_path, plan, allow_copy=True)
    assert conflict.value.code == ARCHIVE_COPY_CONFLICT


def test_pre_cas_receipt_is_idempotent_preserves_discrepancy_and_refuses_replay(
    tmp_path: Path,
) -> None:
    rules_path = _workspace(tmp_path)
    inventory = _inventory(tmp_path)
    plan = build_authority_archive_plan(
        tmp_path,
        inventory,
        cutover_id="cutover-test",
        created_at=CREATED_AT,
        rules_path=rules_path,
    )
    manifest = _generation_manifest()
    pointer = _active_pointer(manifest)
    first = build_pre_cas_migration_receipt(
        tmp_path,
        inventory,
        plan,
        pointer,
        manifest,
        cutover_id="cutover-test",
        created_at=CREATED_AT,
        rules_path=rules_path,
    )
    second = build_pre_cas_migration_receipt(
        tmp_path,
        inventory,
        plan,
        pointer,
        manifest,
        cutover_id="cutover-test",
        created_at=CREATED_AT,
        rules_path=rules_path,
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    receipt = validate_pre_cas_migration_receipt(first)
    payload = receipt["payload"]
    assert payload["cas_performed"] is False
    assert payload["write_performed"] is False
    assert payload["expected_active_pointer_sha256"] == EMPTY_SHA256
    facts = payload["summary"]["baseline_custody_facts"]
    assert facts == load_rules(tmp_path, rules_path).rules.baseline_custody_facts
    assert sha(canonical_json_bytes(facts)) == BASELINE_CUSTODY_FACTS_SHA256

    rows = {row["classification"]: row for row in payload["source_to_target"]}
    authority = rows["ACTIVE_AUTHORITY"]
    assert authority["object_relative_path"] == (
        "results/system/objects/authority/"
        f"{authority['source_byte_sha256']}.json"
    )
    assert authority["pointer_history_relative_path"] == (
        f"results/system/pointer_history/{authority['source_byte_sha256']}.json"
    )
    assert rows["NON_AUTHORITY_SHADOW"]["action"] == "RECORD_ONLY"
    assert rows["NON_AUTHORITY_SHADOW"]["archive_relative_path"] is None
    assert rows["INDEPENDENT_SOURCE"]["action"] == "PRESERVE_INDEPENDENT"
    assert rows["INDEPENDENT_SOURCE"]["object_relative_path"] is None

    output = tmp_path / "receipt.json"
    assert write_pre_cas_migration_receipt(output, receipt) is True
    assert write_pre_cas_migration_receipt(output, receipt) is False

    marker = build_permanent_marker_payload(
        receipt,
        pointer,
        manifest,
        completed_at="2026-08-14T00:00:01Z",
    )
    validated_marker = validate_permanent_marker(marker)
    assert validated_marker["payload"]["migration_replay_refused"] is True
    assert validated_marker["payload"]["legacy_replay_refused"] is True
    assert validated_marker["payload"]["permanent_marker_path"] == (
        "results/system/_migration_complete.json"
    )

    active_path = tmp_path / "results/system/_active.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_bytes(canonical_json_bytes(pointer))
    with pytest.raises(UnifiedCutoverError) as active:
        build_pre_cas_migration_receipt(
            tmp_path,
            inventory,
            plan,
            pointer,
            manifest,
            cutover_id="cutover-test",
            created_at=CREATED_AT,
            rules_path=rules_path,
        )
    assert active.value.code == UNIFIED_ACTIVE_PRESENT
    active_path.unlink()

    marker_path = tmp_path / "results/system/_migration_complete.json"
    marker_path.write_bytes(canonical_json_bytes(marker))
    with pytest.raises(UnifiedCutoverError) as permanent:
        build_pre_cas_migration_receipt(
            tmp_path,
            inventory,
            plan,
            pointer,
            manifest,
            cutover_id="cutover-test",
            created_at=CREATED_AT,
            rules_path=rules_path,
        )
    assert permanent.value.code == PERMANENT_MARKER_PRESENT
