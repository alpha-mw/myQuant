from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import quant_investor.system.validation as validation_module

from quant_investor.contracts import (
    canonical_json_bytes,
    get_contract,
    seal_artifact,
)
from quant_investor.system import (
    BOOTSTRAP_VALIDATION_PROFILE,
    COMPONENT_REGISTRY_SHA256,
    EMERGENCY_CONTROLLER_PATH,
    SystemContractError,
    SystemNotFound,
    SystemSecurityError,
    SystemStore,
    build_emergency_controller,
    build_suspended_generation,
    installed_code_manifest_sha256,
)
from quant_investor.system.components import validation_profile

CREATED_AT = "2026-08-14T00:00:00Z"
MEDIA = {
    "JSON": "application/json",
    "PARQUET": "application/vnd.apache.parquet",
    "PYTHON": "text/x-python",
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _roots(tmp_path: Path) -> tuple[Path, Path, SystemStore]:
    workspace = tmp_path / "workspace"
    source = tmp_path / "canonical-source"
    workspace.mkdir(mode=0o700)
    source.mkdir(mode=0o700)
    return (
        workspace,
        source,
        SystemStore(
            workspace,
            source_root=source,
            source_root_id="canonical-test-source",
        ),
    )


def _write_source(source_root: Path, relative: str, raw: bytes) -> None:
    path = source_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)


def _put_source(
    store: SystemStore,
    relative: str,
    *,
    source_format: str,
) -> dict[str, str]:
    return store.put_source_file(
        relative,
        source_object_id=f"source-{relative.replace('/', '-')}",
        media_type=MEDIA[source_format],
        source_format=source_format,
        created_at=CREATED_AT,
    )


def _put_bundle(
    store: SystemStore,
    bundle_id: str,
    rows: list[tuple[str, dict[str, str]]],
) -> dict[str, str]:
    artifact = seal_artifact(
        "system.source_bundle",
        {
            "source_bundle_id": bundle_id,
            "state": "IMMUTABLE",
            "sources": [
                {"role": role, "source_ref": ref}
                for role, ref in sorted(rows, key=lambda row: row[0])
            ],
        },
        created_at=CREATED_AT,
    )
    return store.put_object(artifact)


def _put_generic(store: SystemStore, kind: str, artifact_id: str) -> dict[str, str]:
    definition = get_contract(kind)
    payload: dict[str, Any] = {field: None for field in definition.required_payload_fields}
    payload[definition.identity_field] = artifact_id
    return store.put_object(seal_artifact(kind, payload, created_at=CREATED_AT))


def _protected_factor_validation(
    store: SystemStore,
    monkeypatch: pytest.MonkeyPatch,
    *,
    release_ref: dict[str, str],
    policy_ref: dict[str, str],
    evidence_refs: list[dict[str, str]],
    active_set_ref: dict[str, str],
    source_object_refs: list[dict[str, str]],
) -> dict[str, Any]:
    validator_ref = store.put_object(
        seal_artifact(
            "factor.validator_manifest",
            {
                "validator_manifest_id": "security-validator-manifest",
                "release_manifest_ref": release_ref,
                "contextual_validator_component_ref": release_ref,
                "source_decoder_component_ref": release_ref,
                "implementation_rows": [],
                "validated_contracts": [],
                "authority": "NON_AUTHORIZING",
            },
            created_at=CREATED_AT,
        )
    )
    receipt_ref = store.put_object(
        seal_artifact(
            "factor.validation_receipt",
            {
                "validation_receipt_id": "security-validation-receipt",
                "policy_ref": policy_ref,
                "evidence_refs": evidence_refs,
                "active_set_ref": active_set_ref,
                "validated": True,
                "authority": "NON_AUTHORIZING",
            },
            created_at=CREATED_AT,
        )
    )
    request_result = store.build_validation_run_request(
        release_manifest_ref=release_ref,
        factor_validator_manifest_ref=validator_ref,
        intrinsic_receipt_ref=receipt_ref,
    )
    request_payload = request_result["validation_request"]["payload"]
    stat_rows: list[dict[str, Any]] = []
    total_bytes = 0
    for source_ref in source_object_refs:
        inspected = store.inspect_source_object(
            source_ref,
            full_hash=True,
            maximum_bytes=1024 * 1024,
        )
        stat_identity = inspected["stat_identity"]
        stat_rows.append(
            {
                "source_binding_sha256": hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "domain": "myquant-source-binding",
                            "source_root_id": inspected["source_root_id"],
                            "relative_path": inspected["relative_path"],
                        }
                    )
                ).hexdigest(),
                "source_object_ref": source_ref,
                "stat_identity": stat_identity,
                "stat_identity_sha256": hashlib.sha256(
                    canonical_json_bytes(stat_identity)
                ).hexdigest(),
            }
        )
        total_bytes += inspected["size"]
    plan = {
        "domain": "myquant-validation-run-plan",
        "validation_namespace_id": request_payload["validation_namespace_id"],
        "validation_profile_id": BOOTSTRAP_VALIDATION_PROFILE,
        "validation_lane": "BOOTSTRAP",
        "component_registry_sha256": COMPONENT_REGISTRY_SHA256,
        "release_manifest_ref": release_ref,
        "installed_code_manifest_sha256": installed_code_manifest_sha256(),
        "factor_validator_manifest_ref": validator_ref,
        "contextual_validator_component_ref": release_ref,
        "source_decoder_component_ref": release_ref,
        "implementation_component_refs": [],
        "intrinsic_receipt_ref": receipt_ref,
        "policy_ref": policy_ref,
        "evidence_refs": evidence_refs,
        "active_set_ref": active_set_ref,
        "candidate_state_ref": None,
        "candidate_state_pointer_sha256": "EMPTY",
        "source_attestation_refs": [],
        "source_object_refs": source_object_refs,
        "source_stat_rows": stat_rows,
        "source_stat_tree_sha256": hashlib.sha256(canonical_json_bytes(stat_rows)).hexdigest(),
        "factor_source_total_bytes": total_bytes,
        "maximum_total_factor_source_bytes": 2 * 1024**3,
        "custody_record_refs": [],
        "custody_head_ref": None,
        "custody_tree_sha256": hashlib.sha256(canonical_json_bytes([])).hexdigest(),
        "compiled_contracts": [],
    }
    derived = {
        "profile": validation_profile(BOOTSTRAP_VALIDATION_PROFILE),
        "receipt": {},
        "candidate": None,
        "factor_manifest": store.get_object(validator_ref),
        "release": store.get_object(release_ref),
        "plan": plan,
        "plan_sha256": hashlib.sha256(canonical_json_bytes(plan)).hexdigest(),
    }
    context_payload = {
        "contextual_result_id": "security-contextual-result",
        "validation_namespace_id": plan["validation_namespace_id"],
        "lane": plan["validation_lane"],
        "intrinsic_receipt_ref": receipt_ref,
        "policy_ref": policy_ref,
        "evidence_refs": evidence_refs,
        "active_set_ref": active_set_ref,
        "composite_state_ref": None,
        "factor_validator_manifest_ref": validator_ref,
        "contextual_validator_component_ref": release_ref,
        "source_decoder_component_ref": release_ref,
        "implementation_component_refs": [],
        "source_attestation_refs": [],
        "source_object_refs": source_object_refs,
        "custody_record_refs": [],
        "custody_tree_sha256": plan["custody_tree_sha256"],
        "custody_head_ref": None,
        "validated": True,
        "blockers": [],
        "authority": "NON_AUTHORIZING",
    }

    def derive_plan(
        target: SystemStore,
        payload: dict[str, Any],
        *,
        full_source_hash: bool,
    ) -> dict[str, Any]:
        del target, payload, full_source_hash
        return copy.deepcopy(derived)

    def invoke_callback(
        target: SystemStore,
        *,
        profile: dict[str, Any],
        validation_request: dict[str, Any],
        trusted_at: str,
    ) -> dict[str, Any]:
        del target, profile, validation_request, trusted_at
        return copy.deepcopy(context_payload)

    monkeypatch.setattr(validation_module, "_derive_plan", derive_plan)
    monkeypatch.setattr(validation_module, "_invoke_callback", invoke_callback)
    monkeypatch.setattr(validation_module, "_utc_now", lambda: CREATED_AT)
    return store.run_validation(request_result["validation_request_ref"])


def _operational_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SystemStore, dict[str, Any], dict[str, Path]]:
    _, source_root, store = _roots(tmp_path)
    source_rows = {
        "calendar.json": ("JSON", b'{"sessions":["2026-08-14"]}'),
        "fundamental-manifest.json": ("JSON", b'{"state":"immutable"}'),
        "fundamental.parquet": ("PARQUET", b"PAR1fundamental"),
        "market-manifest.json": ("JSON", b'{"state":"immutable"}'),
        "market.parquet": ("PARQUET", b"PAR1market"),
        "pit.parquet": ("PARQUET", b"PAR1pit"),
    }
    refs: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    for relative, (source_format, raw) in source_rows.items():
        _write_source(source_root, relative, raw)
        refs[relative] = _put_source(store, relative, source_format=source_format)
        paths[relative] = source_root / relative

    fundamental = _put_bundle(
        store,
        "fundamental-generation",
        [
            ("manifest", refs["fundamental-manifest.json"]),
            ("table", refs["fundamental.parquet"]),
        ],
    )
    market = _put_bundle(
        store,
        "market-snapshot",
        [
            ("manifest", refs["market-manifest.json"]),
            ("table", refs["market.parquet"]),
        ],
    )
    top_sources = _put_bundle(
        store,
        "operational-sources",
        [
            ("exchange_calendar", refs["calendar.json"]),
            ("fundamental_generation", fundamental),
            ("market_snapshot", market),
            ("pit_membership", refs["pit.parquet"]),
        ],
    )
    release = store.put_object(
        seal_artifact(
            "system.release",
            {
                "release_id": "operational-release",
                "state": "OPERATIONAL",
                "code_sha256": _sha("code"),
                "wheel_sha256": _sha("wheel"),
                "code_manifest_sha256": installed_code_manifest_sha256(),
            },
            created_at=CREATED_AT,
        )
    )
    factor_policy = _put_generic(store, "factor.preregistration", "policy")
    factor_evidence = _put_generic(store, "factor.canonical_replay_evidence", "evidence")
    factor_active = _put_generic(store, "factor.admitted_set", "active-set")
    validation = _protected_factor_validation(
        store,
        monkeypatch,
        release_ref=release,
        policy_ref=factor_policy,
        evidence_refs=[factor_evidence],
        active_set_ref=factor_active,
        source_object_refs=[refs["pit.parquet"]],
    )
    validation_receipt = validation["validation_attestation"]["payload"]["intrinsic_receipt_ref"]
    factor_status = store.put_object(
        seal_artifact(
            "factor.status",
            {
                "status_id": "factor-status",
                "active": {
                    "state": "ACTIVE",
                    "lane": "BOOTSTRAP",
                    "admission_route": "NONE",
                    "producer_identity": "NONE",
                    "factor_set_ref": factor_active,
                    "factor_ids": [],
                    "validation_receipt_ref": validation_receipt,
                    "contextual_result_ref": validation["contextual_result_ref"],
                    "validation_attestation_ref": validation["validation_attestation_ref"],
                },
                "observed": {},
                "readiness": "BLOCKED",
                "blockers": ["FACTOR_NOT_READY"],
                "activation_mutation_authorized": False,
            },
            created_at=CREATED_AT,
        )
    )
    migration = _put_generic(store, "system.migration.receipt", "migration")
    readiness = store.put_object(
        seal_artifact(
            "intelligence_readiness",
            {
                "readiness_id": "blocked-readiness",
                "factor_state": "BLOCKED",
                "factor_status_ref": factor_status,
                "admission_route": "NONE",
                "producer_identity": "NONE",
                "mainline_state": "UNINITIALIZED",
                "mainline_candidate_ref": None,
                "investment_state": "BLOCKED",
                "blockers": ["FACTOR_NOT_READY"],
            },
            created_at=CREATED_AT,
        )
    )
    suspended = build_suspended_generation(
        store,
        blockers=["EMERGENCY_TARGET"],
        created_at=CREATED_AT,
    )
    controller = build_emergency_controller(
        store,
        suspended_generation_id=suspended["generation_id"],
    )
    generation = store.assemble_generation(
        generation_state="OPERATIONAL",
        release_manifest_ref=release,
        source_refs=[top_sources],
        factor_source_object_refs=[refs["pit.parquet"]],
        factor_policy_ref=factor_policy,
        factor_evidence_refs=[factor_evidence],
        factor_active_set_ref=factor_active,
        factor_validation_attestation_ref=validation["validation_attestation_ref"],
        mainline_ref=None,
        research_refs=[],
        migration_receipt_ref=migration,
        migration_marker_ref=None,
        skill_tree_sha256=_sha("skills"),
        automation_semantic_sha256=_sha("automation"),
        readiness_matrix_ref=readiness,
        emergency_controller_sha256=controller["byte_sha256"],
        created_at=CREATED_AT,
    )
    return store, generation, paths


@pytest.mark.parametrize("action", ["missing", "tampered"])  # type: ignore[untyped-decorator]
def test_operational_verification_reopens_bound_source_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    store, generation, paths = _operational_generation(tmp_path, monkeypatch)
    target = paths["market.parquet"]
    expected: type[Exception]
    if action == "missing":
        target.unlink()
        expected = SystemNotFound
    else:
        target.write_bytes(b"PAR1tampered")
        target.chmod(0o600)
        expected = SystemContractError

    with pytest.raises(expected):
        store.verify_generation(generation["generation_id"])


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "action", ["missing", "unsafe_mode", "tampered"]
)
def test_operational_verification_reopens_bound_emergency_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    store, generation, _ = _operational_generation(tmp_path, monkeypatch)
    controller = store.workspace_root / str(EMERGENCY_CONTROLLER_PATH)
    expected: type[Exception]
    if action == "missing":
        controller.unlink()
        expected = SystemNotFound
    elif action == "unsafe_mode":
        controller.chmod(0o700)
        expected = SystemSecurityError
    else:
        controller.chmod(0o600)
        controller.write_bytes(controller.read_bytes() + b"\n")
        controller.chmod(0o500)
        expected = SystemContractError

    with pytest.raises(expected):
        store.verify_generation(generation["generation_id"])


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation",
    [
        "policy_ref",
        "evidence_refs",
        "validated",
        "authority",
        "evidence_order",
        "status_active_set_ref",
        "status_receipt_kind",
    ],
)
def test_operational_assembly_rejects_validation_receipt_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    store, generation, _ = _operational_generation(tmp_path, monkeypatch)
    manifest_payload = generation["manifest"]["payload"]
    receipt_payload = copy.deepcopy(generation["factor_validation_receipt"]["payload"])
    generation_evidence_refs = manifest_payload["factor_evidence_refs"]
    if mutation == "policy_ref":
        receipt_payload["policy_ref"] = manifest_payload["factor_evidence_refs"][0]
    elif mutation == "evidence_refs":
        receipt_payload["evidence_refs"] = [manifest_payload["factor_policy_ref"]]
    elif mutation == "validated":
        receipt_payload["validated"] = False
    elif mutation == "authority":
        receipt_payload["authority"] = "AUTHORIZING"
    elif mutation == "evidence_order":
        second_evidence = _put_generic(
            store,
            "factor.canonical_replay_evidence",
            "second-evidence",
        )
        generation_evidence_refs = sorted(
            [*generation_evidence_refs, second_evidence],
            key=lambda ref: (
                ref["kind"],
                ref["contract_sha256"],
                ref["artifact_id"],
                ref["semantic_sha256"],
                ref["byte_sha256"],
            ),
        )
        receipt_payload["evidence_refs"] = list(reversed(generation_evidence_refs))
    if mutation.startswith("status_"):
        receipt_ref = generation["factor_validation_receipt_ref"]
    else:
        receipt_ref = store.put_object(
            seal_artifact(
                "factor.validation_receipt",
                receipt_payload,
                created_at=CREATED_AT,
            )
        )

    status_payload = copy.deepcopy(generation["factor_status"]["payload"])
    status_payload["active"]["validation_receipt_ref"] = receipt_ref
    if mutation == "status_active_set_ref":
        status_payload["active"]["factor_set_ref"] = manifest_payload["factor_policy_ref"]
    elif mutation == "status_receipt_kind":
        status_payload["active"]["validation_receipt_ref"] = manifest_payload[
            "factor_evidence_refs"
        ][0]
    status_ref = store.put_object(
        seal_artifact("factor.status", status_payload, created_at=CREATED_AT)
    )
    readiness_payload = copy.deepcopy(generation["readiness"]["payload"])
    readiness_payload["factor_status_ref"] = status_ref
    readiness_ref = store.put_object(
        seal_artifact("intelligence_readiness", readiness_payload, created_at=CREATED_AT)
    )
    before = sorted(
        path.name for path in (store.workspace_root / "results/system/generations").iterdir()
    )

    with pytest.raises(SystemContractError):
        store.assemble_generation(
            generation_state=manifest_payload["generation_state"],
            release_manifest_ref=manifest_payload["release_manifest_ref"],
            source_refs=manifest_payload["source_refs"],
            factor_source_object_refs=manifest_payload["factor_source_object_refs"],
            factor_policy_ref=manifest_payload["factor_policy_ref"],
            factor_evidence_refs=generation_evidence_refs,
            factor_active_set_ref=manifest_payload["factor_active_set_ref"],
            factor_validation_attestation_ref=manifest_payload["factor_validation_attestation_ref"],
            mainline_ref=manifest_payload["mainline_ref"],
            research_refs=manifest_payload["research_refs"],
            migration_receipt_ref=manifest_payload["migration_receipt_ref"],
            migration_marker_ref=manifest_payload["migration_marker_ref"],
            skill_tree_sha256=manifest_payload["skill_tree_sha256"],
            automation_semantic_sha256=manifest_payload["automation_semantic_sha256"],
            readiness_matrix_ref=readiness_ref,
            emergency_controller_sha256=manifest_payload["emergency_controller_sha256"],
            created_at=CREATED_AT,
        )

    after = sorted(
        path.name for path in (store.workspace_root / "results/system/generations").iterdir()
    )
    assert after == before


def test_plain_sealed_context_and_attestation_cannot_publish_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, generation, _ = _operational_generation(tmp_path, monkeypatch)
    manifest_payload = generation["manifest"]["payload"]
    context_payload = copy.deepcopy(generation["factor_contextual_result"]["payload"])
    context_payload["contextual_result_id"] = "uncustodied-context"
    context_ref = store.put_object(
        seal_artifact(
            "factor.contextual_validation_result",
            context_payload,
            created_at=CREATED_AT,
        )
    )
    attestation_payload = copy.deepcopy(generation["factor_validation_attestation"]["payload"])
    attestation_payload["attestation_id"] = "uncustodied-attestation"
    attestation_payload["contextual_result_ref"] = context_ref
    attestation_ref = store.put_object(
        seal_artifact(
            "system.validation_attestation",
            attestation_payload,
            created_at=CREATED_AT,
        )
    )
    status_payload = copy.deepcopy(generation["factor_status"]["payload"])
    status_payload["active"]["contextual_result_ref"] = context_ref
    status_payload["active"]["validation_attestation_ref"] = attestation_ref
    status_ref = store.put_object(
        seal_artifact("factor.status", status_payload, created_at=CREATED_AT)
    )
    readiness_payload = copy.deepcopy(generation["readiness"]["payload"])
    readiness_payload["factor_status_ref"] = status_ref
    readiness_ref = store.put_object(
        seal_artifact("intelligence_readiness", readiness_payload, created_at=CREATED_AT)
    )
    generations = store.workspace_root / "results/system/generations"
    before = sorted(path.name for path in generations.iterdir())

    with pytest.raises(SystemContractError):
        store.assemble_generation(
            generation_state=manifest_payload["generation_state"],
            release_manifest_ref=manifest_payload["release_manifest_ref"],
            source_refs=manifest_payload["source_refs"],
            factor_source_object_refs=manifest_payload["factor_source_object_refs"],
            factor_policy_ref=manifest_payload["factor_policy_ref"],
            factor_evidence_refs=manifest_payload["factor_evidence_refs"],
            factor_active_set_ref=manifest_payload["factor_active_set_ref"],
            factor_validation_attestation_ref=attestation_ref,
            mainline_ref=manifest_payload["mainline_ref"],
            research_refs=manifest_payload["research_refs"],
            migration_receipt_ref=manifest_payload["migration_receipt_ref"],
            migration_marker_ref=manifest_payload["migration_marker_ref"],
            skill_tree_sha256=manifest_payload["skill_tree_sha256"],
            automation_semantic_sha256=manifest_payload["automation_semantic_sha256"],
            readiness_matrix_ref=readiness_ref,
            emergency_controller_sha256=manifest_payload["emergency_controller_sha256"],
            created_at=CREATED_AT,
        )

    assert sorted(path.name for path in generations.iterdir()) == before


def test_factor_source_projection_drift_cannot_publish_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, generation, _ = _operational_generation(tmp_path, monkeypatch)
    manifest_payload = generation["manifest"]["payload"]
    operational_source = generation["sources"][0]
    calendar_ref = next(
        row["source_ref"]
        for row in operational_source["payload"]["sources"]
        if row["role"] == "exchange_calendar"
    )
    generations = store.workspace_root / "results/system/generations"
    before = sorted(path.name for path in generations.iterdir())

    with pytest.raises(SystemContractError):
        store.assemble_generation(
            generation_state=manifest_payload["generation_state"],
            release_manifest_ref=manifest_payload["release_manifest_ref"],
            source_refs=manifest_payload["source_refs"],
            factor_source_object_refs=[calendar_ref],
            factor_policy_ref=manifest_payload["factor_policy_ref"],
            factor_evidence_refs=manifest_payload["factor_evidence_refs"],
            factor_active_set_ref=manifest_payload["factor_active_set_ref"],
            factor_validation_attestation_ref=manifest_payload["factor_validation_attestation_ref"],
            mainline_ref=manifest_payload["mainline_ref"],
            research_refs=manifest_payload["research_refs"],
            migration_receipt_ref=manifest_payload["migration_receipt_ref"],
            migration_marker_ref=manifest_payload["migration_marker_ref"],
            skill_tree_sha256=manifest_payload["skill_tree_sha256"],
            automation_semantic_sha256=manifest_payload["automation_semantic_sha256"],
            readiness_matrix_ref=manifest_payload["readiness_matrix_ref"],
            emergency_controller_sha256=manifest_payload["emergency_controller_sha256"],
            created_at=CREATED_AT,
        )

    assert sorted(path.name for path in generations.iterdir()) == before


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("generation_state", "wrong_kind"),
    [
        ("OPERATIONAL", "system.readiness"),
        ("SYSTEM_SUSPENDED", "intelligence_readiness"),
    ],
)
def test_generation_readiness_kind_is_role_specific(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_state: str,
    wrong_kind: str,
) -> None:
    store, operational, _ = _operational_generation(tmp_path, monkeypatch)
    if generation_state == "OPERATIONAL":
        source = operational
        manifest_payload = source["manifest"]["payload"]
        kwargs = {
            "generation_state": generation_state,
            "release_manifest_ref": manifest_payload["release_manifest_ref"],
            "source_refs": manifest_payload["source_refs"],
            "factor_source_object_refs": manifest_payload["factor_source_object_refs"],
            "factor_policy_ref": manifest_payload["factor_policy_ref"],
            "factor_evidence_refs": manifest_payload["factor_evidence_refs"],
            "factor_active_set_ref": manifest_payload["factor_active_set_ref"],
            "factor_validation_attestation_ref": manifest_payload[
                "factor_validation_attestation_ref"
            ],
            "mainline_ref": manifest_payload["mainline_ref"],
            "research_refs": manifest_payload["research_refs"],
            "migration_receipt_ref": manifest_payload["migration_receipt_ref"],
            "migration_marker_ref": manifest_payload["migration_marker_ref"],
            "skill_tree_sha256": manifest_payload["skill_tree_sha256"],
            "automation_semantic_sha256": manifest_payload["automation_semantic_sha256"],
            "emergency_controller_sha256": manifest_payload["emergency_controller_sha256"],
            "created_at": CREATED_AT,
        }
    else:
        source = build_suspended_generation(
            store,
            blockers=["ROLE_TEST"],
            created_at=CREATED_AT,
        )
        manifest_payload = source["manifest"]["payload"]
        kwargs = {
            "generation_state": generation_state,
            "release_manifest_ref": manifest_payload["release_manifest_ref"],
            "source_refs": [],
            "factor_source_object_refs": [],
            "factor_policy_ref": None,
            "factor_evidence_refs": [],
            "factor_active_set_ref": None,
            "factor_validation_attestation_ref": None,
            "mainline_ref": None,
            "research_refs": [],
            "migration_receipt_ref": None,
            "migration_marker_ref": None,
            "skill_tree_sha256": manifest_payload["skill_tree_sha256"],
            "automation_semantic_sha256": manifest_payload["automation_semantic_sha256"],
            "emergency_controller_sha256": None,
            "created_at": CREATED_AT,
        }
    wrong_readiness = store.put_object(
        seal_artifact(
            wrong_kind,
            copy.deepcopy(source["readiness"]["payload"]),
            created_at=CREATED_AT,
        )
    )
    before = sorted(
        path.name for path in (store.workspace_root / "results/system/generations").iterdir()
    )

    with pytest.raises(SystemContractError):
        store.assemble_generation(
            **kwargs,
            readiness_matrix_ref=wrong_readiness,
        )

    after = sorted(
        path.name for path in (store.workspace_root / "results/system/generations").iterdir()
    )
    assert after == before


def test_source_descriptor_streams_beyond_artifact_read_bound_without_copy(
    tmp_path: Path,
) -> None:
    workspace, source_root, store = _roots(tmp_path)
    large = source_root / "large.parquet"
    with large.open("wb") as handle:
        handle.truncate(65 * 1024 * 1024)
    large.chmod(0o600)

    ref = _put_source(store, "large.parquet", source_format="PARQUET")
    descriptor = store.get_object(ref)
    store._verify_source_object(descriptor)

    expected = hashlib.sha256()
    for _ in range(65):
        expected.update(b"\0" * (1024 * 1024))
    assert descriptor["payload"]["byte_sha256"] == expected.hexdigest()
    assert not (workspace / "results/system/objects/system.raw").exists()


def test_source_read_rejects_symlink_unsafe_mode_and_path_traversal(
    tmp_path: Path,
) -> None:
    _, source_root, store = _roots(tmp_path)
    _write_source(source_root, "real.json", b'{"ok":true}')
    (source_root / "link.json").symlink_to(source_root / "real.json")

    with pytest.raises(SystemSecurityError):
        _put_source(store, "link.json", source_format="JSON")

    (source_root / "real.json").chmod(0o666)
    with pytest.raises(SystemSecurityError):
        _put_source(store, "real.json", source_format="JSON")

    with pytest.raises(SystemSecurityError):
        store.put_source_file(
            "../escape.json",
            source_object_id="escape",
            media_type="application/json",
            source_format="JSON",
            created_at=CREATED_AT,
        )


def test_source_media_format_and_expected_hash_are_semantically_bound(
    tmp_path: Path,
) -> None:
    _, source_root, store = _roots(tmp_path)
    _write_source(source_root, "data.json", b'{"ok":true}')

    with pytest.raises(SystemContractError):
        store.put_source_file(
            "data.json",
            source_object_id="bad-media",
            media_type="text/plain",
            source_format="JSON",
            created_at=CREATED_AT,
        )
    with pytest.raises(SystemContractError):
        store.put_source_file(
            "data.json",
            source_object_id="bad-hash",
            media_type="application/json",
            source_format="JSON",
            expected_byte_sha256="0" * 64,
            created_at=CREATED_AT,
        )


def test_object_reads_reject_symlink_mode_hardlink_and_traversal(
    tmp_path: Path,
) -> None:
    workspace, _, store = _roots(tmp_path)
    artifact = seal_artifact(
        "system.release",
        {
            "release_id": "secure",
            "state": "TEST",
            "code_sha256": _sha("x"),
            "wheel_sha256": _sha("wheel"),
            "code_manifest_sha256": _sha("code-manifest"),
        },
        created_at=CREATED_AT,
    )
    ref = store.put_object(artifact)
    path = workspace / "results/system/objects/system.release" / f"{ref['byte_sha256']}.json"
    original = path.read_bytes()

    path.chmod(0o644)
    with pytest.raises(SystemSecurityError):
        store.get_object(ref)
    path.chmod(0o600)

    alias = path.with_suffix(".alias")
    os.link(path, alias)
    with pytest.raises(SystemSecurityError):
        store.get_object(ref)
    alias.unlink()

    target = path.with_suffix(".target")
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(SystemSecurityError):
        store.get_object(ref)
    path.unlink()
    target.rename(path)
    path.chmod(0o600)
    assert path.read_bytes() == original

    with pytest.raises(SystemContractError):
        store.get_object(ref["byte_sha256"], kind="../../escape")


def test_noncanonical_and_duplicate_object_bytes_are_rejected(tmp_path: Path) -> None:
    workspace, _, store = _roots(tmp_path)
    artifact = seal_artifact(
        "system.release",
        {
            "release_id": "noncanonical",
            "state": "TEST",
            "code_sha256": _sha("x"),
            "wheel_sha256": _sha("wheel"),
            "code_manifest_sha256": _sha("code-manifest"),
        },
        created_at=CREATED_AT,
    )
    real_ref = store.put_object(artifact)
    raw = canonical_json_bytes(artifact) + b"\n"
    byte_sha = hashlib.sha256(raw).hexdigest()
    path = workspace / "results/system/objects/system.release" / f"{byte_sha}.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    forged_ref = {
        **real_ref,
        "byte_sha256": byte_sha,
    }
    with pytest.raises(SystemContractError):
        store.get_object(forged_ref)

    duplicate = raw.replace(
        b'"kind":"system.release"',
        b'"kind":"system.release","kind":"system.release"',
    ).rstrip(b"\n")
    duplicate_sha = hashlib.sha256(duplicate).hexdigest()
    duplicate_path = path.with_name(f"{duplicate_sha}.json")
    duplicate_path.write_bytes(duplicate)
    duplicate_path.chmod(0o600)
    duplicate_ref = {**forged_ref, "byte_sha256": duplicate_sha}
    with pytest.raises(SystemContractError):
        store.get_object(duplicate_ref)


def test_contract_catalog_object_is_kind_scoped_and_tamper_detected(
    tmp_path: Path,
) -> None:
    workspace, _, store = _roots(tmp_path)
    catalog_sha = store.put_contract_catalog()
    path = workspace / "results/system/objects/system.contract_catalog" / f"{catalog_sha}.json"
    assert path.is_file()
    path.write_bytes(path.read_bytes() + b"\n")
    path.chmod(0o600)
    with pytest.raises(SystemContractError):
        store.read_contract_catalog(catalog_sha)
