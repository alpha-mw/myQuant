# flake8: noqa: E501
from __future__ import annotations

import base64
import fcntl
import hashlib
import importlib.util
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from quant_investor.market import fundamental_successor_promotion as promotion
from quant_investor.market.fundamental_limits import (
    FUNDAMENTAL_GENERIC_JSON_MAX_BYTES,
    FUNDAMENTAL_PREDECESSOR_MANIFEST_MAX_BYTES,
    FUNDAMENTAL_PREDECESSOR_MANIFEST_ROLE,
    FundamentalSizePolicyViolation,
    validate_fundamental_json_size_policy,
)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_ref(path: Path) -> dict[str, Any]:
    payload = _read(path)
    return {
        "path": str(path),
        "sha256": _sha(payload),
        "size": len(payload),
    }


def test_stable_json_read_enforces_role_specific_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "large-for-local-limit.json"
    payload = _json_bytes({"value": "bounded"})
    _write(path, payload)

    oversized = promotion._FileIdentity(
        path=path,
        sha256=_sha(payload),
        size=FUNDAMENTAL_GENERIC_JSON_MAX_BYTES + 1,
        signature=(0,),
    )
    monkeypatch.setattr(
        promotion,
        "_stable_file_hash",
        lambda *_args, **_kwargs: oversized,
    )
    with pytest.raises(
        promotion.SuccessorPromotionError,
        match="SUPPORT_REFERENCE_SIZE_POLICY_VIOLATION",
    ):
        promotion._stable_small_bytes(
            path,
            label="fixture manifest",
            semantic_role="successor_manifest",
            maximum_bytes=FUNDAMENTAL_GENERIC_JSON_MAX_BYTES,
        )

    monkeypatch.undo()
    readback, identity = promotion._stable_small_bytes(
        path,
        label="fixture manifest",
        semantic_role="successor_manifest",
        maximum_bytes=FUNDAMENTAL_GENERIC_JSON_MAX_BYTES,
    )
    assert readback == payload
    assert identity.sha256 == _sha(payload)


def test_fundamental_json_size_contract_has_exact_boundaries() -> None:
    assert FUNDAMENTAL_GENERIC_JSON_MAX_BYTES == 64 * 1024 * 1024
    assert FUNDAMENTAL_PREDECESSOR_MANIFEST_MAX_BYTES == 128 * 1024 * 1024
    validate_fundamental_json_size_policy(
        semantic_role=FUNDAMENTAL_PREDECESSOR_MANIFEST_ROLE,
        maximum_bytes=FUNDAMENTAL_PREDECESSOR_MANIFEST_MAX_BYTES,
        observed_bytes=FUNDAMENTAL_PREDECESSOR_MANIFEST_MAX_BYTES,
    )
    with pytest.raises(FundamentalSizePolicyViolation):
        validate_fundamental_json_size_policy(
            semantic_role=FUNDAMENTAL_PREDECESSOR_MANIFEST_ROLE,
            maximum_bytes=FUNDAMENTAL_PREDECESSOR_MANIFEST_MAX_BYTES,
            observed_bytes=FUNDAMENTAL_PREDECESSOR_MANIFEST_MAX_BYTES + 1,
        )
    with pytest.raises(ValueError, match="differs from semantic-role policy"):
        validate_fundamental_json_size_policy(
            semantic_role="support_manifest",
            maximum_bytes=FUNDAMENTAL_PREDECESSOR_MANIFEST_MAX_BYTES,
        )


def _pointer_binding(
    pointer_path: Path,
    *,
    refs: list[dict[str, Any]],
    generation_id: str = "",
    manifest_sha256: str = "",
    provenance_schema: str = "",
    cutoff: str = "",
    original_seam: str = "",
) -> dict[str, Any]:
    payload = _read(pointer_path)
    return {
        "live_pointer_path": str(pointer_path),
        "pointer_sha256": _sha(payload),
        "exact_pointer_bytes_b64": base64.b64encode(payload).decode("ascii"),
        "immutable_refs": refs,
        **({"generation_id": generation_id} if generation_id else {}),
        **({"manifest_sha256": manifest_sha256} if manifest_sha256 else {}),
        **(
            {"provenance_schema_version": provenance_schema}
            if provenance_schema
            else {}
        ),
        **({"effective_cutoff": cutoff} if cutoff else {}),
        **({"original_seam": original_seam} if original_seam else {}),
    }


def _write_parent_generation(
    canonical: Path,
    *,
    generation_id: str,
) -> tuple[Path, Path, dict[str, Path]]:
    generation = canonical / promotion.FUNDAMENTAL_GENERATIONS_DIRNAME / generation_id
    generation.mkdir(parents=True)
    tables: dict[str, Path] = {}
    table_manifest: dict[str, Any] = {}
    for index, table_name in enumerate(promotion.FUNDAMENTAL_TABLES):
        path = generation / f"{table_name}.parquet"
        payload = (f"parent:{table_name}:".encode("utf-8") * (index + 2))
        _write(path, payload)
        tables[table_name] = path
        table_manifest[table_name] = {
            "sha256": _sha(payload),
            "bytes": len(payload),
            "rows": index,
        }
    manifest = {
        "schema_version": "cn-fundamental-generation.v1",
        "status": "OK",
        "generation_id": generation_id,
        "tables": table_manifest,
        "metadata": {"gate2_passed": True},
    }
    manifest_path = generation / "manifest.json"
    _write(manifest_path, _json_bytes(manifest))
    relative = generation.relative_to(canonical)
    pointer = {
        "schema_version": "cn-fundamental-pointer.v1",
        "status": "OK",
        "generation_id": generation_id,
        "manifest_path": (relative / "manifest.json").as_posix(),
        "tables": {
            table_name: (relative / f"{table_name}.parquet").as_posix()
            for table_name in promotion.FUNDAMENTAL_TABLES
        },
        "metadata": {"gate2_passed": True},
    }
    pointer_path = canonical / promotion.FUNDAMENTAL_POINTER_FILENAME
    _write(pointer_path, _json_bytes(pointer))
    return pointer_path, manifest_path, tables


def _write_bound_pointer(root: Path, filename: str, label: str) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    immutable = root / f"{label}_immutable.json"
    _write(immutable, _json_bytes({"kind": label, "immutable": True}))
    pointer = root / filename
    _write(
        pointer,
        _json_bytes(
            {
                "schema_version": f"test-{label}-pointer.v1",
                "immutable_path": str(immutable),
                "immutable_sha256": _sha(_read(immutable)),
            }
        ),
    )
    return pointer, immutable


def _fixture(
    tmp_path: Path,
    *,
    parent_schema: str = "cn-fundamental-primary-provenance.v2",
    parent_cutoff: str = "20260806",
    original_seam: str = "20260806",
    target_cutoff: str = "20260814",
) -> dict[str, Any]:
    canonical = tmp_path / "canonical"
    staging = tmp_path / "staging"
    market_root = tmp_path / "market"
    pit_root = tmp_path / "pit"
    journal_root = tmp_path / "journal"
    canonical.mkdir()
    staging.mkdir()
    journal_root.mkdir(mode=0o700)
    os.chmod(journal_root, 0o700)

    parent_generation_id = "parent_v2" if parent_schema.endswith(".v2") else "parent_v3"
    parent_pointer, parent_manifest, parent_tables = _write_parent_generation(
        canonical,
        generation_id=parent_generation_id,
    )
    predecessor_refs = [_file_ref(parent_manifest)] + [
        _file_ref(parent_tables[name]) for name in promotion.FUNDAMENTAL_TABLES
    ]
    predecessor = _pointer_binding(
        parent_pointer,
        refs=predecessor_refs,
        generation_id=parent_generation_id,
        manifest_sha256=_sha(_read(parent_manifest)),
        provenance_schema=parent_schema,
        cutoff=parent_cutoff,
        original_seam=(original_seam if parent_schema.endswith(".v3") else ""),
    )
    market_pointer, market_immutable = _write_bound_pointer(
        market_root,
        "_latest.json",
        "market",
    )
    pit_pointer, pit_immutable = _write_bound_pointer(
        pit_root,
        "whatever-current-pointer.json",
        "pit",
    )
    market_binding = _pointer_binding(
        market_pointer,
        refs=[_file_ref(market_immutable)],
    )
    pit_binding = _pointer_binding(
        pit_pointer,
        refs=[_file_ref(pit_immutable)],
    )

    generation_id = "successor_20260814"
    generation = staging / promotion.FUNDAMENTAL_GENERATIONS_DIRNAME / generation_id
    provider_root = generation / "provider_evidence"
    provider_root.mkdir(parents=True)
    provider_file = provider_root / "execution" / "receipt.json"
    _write(
        provider_file,
        _json_bytes(
            {
                "status": "passed",
                "requests_failed": 0,
                "requests_malformed": 0,
            }
        ),
    )
    table_manifest: dict[str, Any] = {}
    for index, table_name in enumerate(promotion.FUNDAMENTAL_TABLES):
        path = generation / f"{table_name}.parquet"
        multiplier = 2 * 1024 * 1024 if table_name == "fundamental_daily" else 100
        payload = (f"successor:{index}:".encode("utf-8") * multiplier)
        _write(path, payload)
        table_manifest[table_name] = {
            "sha256": _sha(payload),
            "bytes": len(payload),
            "rows": multiplier,
        }
    provenance_binding_sha256 = _sha(b"successor-provenance-binding")
    chain_ids = [parent_generation_id]
    provenance = {
        "schema_version": promotion.SUCCESSOR_PROVENANCE_SCHEMA,
        "status": promotion.SUCCESSOR_PROVENANCE_STATUS,
        "mixed_generation": True,
        "predecessor": predecessor,
        "target_bindings": {
            "market": market_binding,
            "pit": pit_binding,
        },
        "successor_chain": {
            "schema_version": "cn-fundamental-successor-chain.v1",
            "depth": 1 if parent_schema.endswith(".v2") else 2,
            "generation_ids": chain_ids,
            "original_seam": original_seam,
            "cumulative_chain_sha256": _sha(b"chain"),
        },
        "original_seam": original_seam,
        "immediate_parent_cutoff": parent_cutoff,
        "target_cutoff": target_cutoff,
        "provenance_binding_sha256": provenance_binding_sha256,
    }
    manifest = {
        "schema_version": "cn-fundamental-generation.v1",
        "status": "OK",
        "generation_id": generation_id,
        "tables": table_manifest,
        "metadata": {
            "gate2_passed": True,
            "mixed_generation": True,
            "legacy_direct_reader_provenance": "limited",
        },
        "primary_provenance": provenance,
    }
    manifest_path = generation / "manifest.json"
    _write(manifest_path, _json_bytes(manifest))
    relative = generation.relative_to(staging)
    pointer = {
        "schema_version": "cn-fundamental-pointer.v1",
        "status": "OK",
        "generation_id": generation_id,
        "manifest_path": (relative / "manifest.json").as_posix(),
        "tables": {
            table_name: (relative / f"{table_name}.parquet").as_posix()
            for table_name in promotion.FUNDAMENTAL_TABLES
        },
        "metadata": {
            "gate2_passed": True,
            "mixed_generation": True,
            "legacy_direct_reader_provenance": "limited",
        },
        "primary_provenance": provenance,
    }
    _write(staging / promotion.FUNDAMENTAL_POINTER_FILENAME, _json_bytes(pointer))

    def validator(root: Path) -> dict[str, Any]:
        pointer_path = root / promotion.FUNDAMENTAL_POINTER_FILENAME
        pointer_bytes = _read(pointer_path)
        parsed_pointer = json.loads(pointer_bytes)
        dynamic_manifest_path = root / parsed_pointer["manifest_path"]
        manifest_bytes = _read(dynamic_manifest_path)
        parsed_manifest = json.loads(manifest_bytes)
        dynamic_generation = dynamic_manifest_path.parent
        dynamic_provenance = parsed_manifest["primary_provenance"]
        table_sha = {
            name: _sha(_read(root / parsed_pointer["tables"][name]))
            for name in promotion.FUNDAMENTAL_TABLES
        }
        provider_files: dict[str, Any] = {}
        for path in sorted((dynamic_generation / "provider_evidence").rglob("*")):
            if path.is_file():
                relative_path = path.relative_to(dynamic_generation).as_posix()
                payload = _read(path)
                provider_files[relative_path] = {
                    "path": str(path),
                    "sha256": _sha(payload),
                    "size": len(payload),
                }
        result = {
            "generation_id": parsed_pointer["generation_id"],
            "pointer_sha256": _sha(pointer_bytes),
            "manifest_sha256": _sha(manifest_bytes),
            "table_sha256": table_sha,
            "provider_evidence_files": provider_files,
            "primary_provenance": dynamic_provenance,
            "predecessor": dynamic_provenance["predecessor"],
            "successor_chain": dynamic_provenance["successor_chain"],
            "original_seam": dynamic_provenance["original_seam"],
            "immediate_parent_cutoff": dynamic_provenance[
                "immediate_parent_cutoff"
            ],
            "target_cutoff": dynamic_provenance["target_cutoff"],
            "provenance_binding_sha256": dynamic_provenance[
                "provenance_binding_sha256"
            ],
        }
        if "market" in dynamic_provenance["target_bindings"]:
            result["market_binding"] = dynamic_provenance["target_bindings"][
                "market"
            ]
        if "pit" in dynamic_provenance["target_bindings"]:
            result["pit_binding"] = dynamic_provenance["target_bindings"]["pit"]
        return result

    return {
        "canonical": canonical,
        "staging": staging,
        "journal": journal_root,
        "market_pointer": market_pointer,
        "pit_pointer": pit_pointer,
        "parent_pointer_bytes": _read(parent_pointer),
        "parent_pointer_sha256": _sha(_read(parent_pointer)),
        "generation_id": generation_id,
        "validator": validator,
    }


def _promote(fixture: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return promotion.promote_successor_generation(
        staging_root=fixture["staging"],
        canonical_root=fixture["canonical"],
        expected_pointer_sha256=fixture["parent_pointer_sha256"],
        execute=True,
        journal_root=fixture["journal"],
        journal_run_id=kwargs.pop("journal_run_id", "run-1"),
        staging_validator=fixture["validator"],
        **kwargs,
    )


def _journal_phases(journal_root: Path, run_id: str = "run-1") -> list[str]:
    phases: list[str] = []
    for path in sorted((journal_root / run_id).glob("*.json")):
        phases.append(json.loads(_read(path))["phase"])
    return phases


def _use_sealed_source_binding(fixture: dict[str, Any]) -> None:
    pointer_path = fixture["staging"] / promotion.FUNDAMENTAL_POINTER_FILENAME
    pointer = json.loads(_read(pointer_path))
    manifest_path = fixture["staging"] / pointer["manifest_path"]
    manifest = json.loads(_read(manifest_path))
    provenance = manifest["primary_provenance"]
    predecessor = dict(provenance["predecessor"])
    market = dict(provenance["target_bindings"]["market"])
    pit = dict(provenance["target_bindings"]["pit"])

    def captured(binding: dict[str, Any]) -> dict[str, Any]:
        payload = base64.b64decode(binding["exact_pointer_bytes_b64"])
        return {
            "byte_length": len(payload),
            "bytes_base64": binding["exact_pointer_bytes_b64"],
            "sha256": binding["pointer_sha256"],
        }

    scope = fixture["market_pointer"].parent / "expected_scope.json"
    history = fixture["market_pointer"].parent / "history_audit.json"
    pit_membership = fixture["pit_pointer"].parent / "membership.parquet"
    _write(scope, _json_bytes({"scope": "sealed"}))
    _write(history, _json_bytes({"history": "sealed"}))
    _write(pit_membership, b"sealed pit membership")
    predecessor_refs = predecessor["immutable_refs"]
    market_refs = market["immutable_refs"]
    pit_refs = pit["immutable_refs"]
    source_manifest = {
        "captured_pointers": {
            "predecessor": captured(predecessor),
            "market": captured(market),
            "pit": captured(pit),
        },
        "immutable_refs": {
            "live_pointer_paths": {
                "predecessor": str(
                    fixture["canonical"] / promotion.FUNDAMENTAL_POINTER_FILENAME
                ),
                "market": str(fixture["market_pointer"]),
                "pit": str(fixture["pit_pointer"]),
            },
            "predecessor": {
                "manifest_path": predecessor_refs[0]["path"],
                "manifest_sha256": predecessor_refs[0]["sha256"],
                "table_sha256": {
                    name: predecessor_refs[index + 1]["sha256"]
                    for index, name in enumerate(promotion.FUNDAMENTAL_TABLES)
                },
                "cutoff": "20260806",
            },
            "market": {
                "manifest_path": market_refs[0]["path"],
                "manifest_sha256": market_refs[0]["sha256"],
                "target": "20260814",
            },
            "pit": {
                "manifest_path": pit_refs[0]["path"],
                "manifest_sha256": pit_refs[0]["sha256"],
                "membership_path": str(pit_membership),
                "membership_sha256": _sha(_read(pit_membership)),
                "target": "20260814",
            },
            "scope": {
                "path": str(scope),
                "sha256": _sha(_read(scope)),
            },
            "history_audit": {
                "path": str(history),
                "sha256": _sha(_read(history)),
            },
        },
    }
    generation_root = manifest_path.parent
    support_path = generation_root / "provider_evidence/source/provider_manifest.json"
    _write(support_path, _json_bytes(source_manifest))
    durable_predecessor = {
        key: value
        for key, value in predecessor.items()
        if key
        not in {
            "live_pointer_path",
            "exact_pointer_bytes_b64",
            "immutable_refs",
        }
    }
    provenance["predecessor"] = durable_predecessor
    provenance["target_bindings"] = {
        "market_pointer": {
            "path": str(fixture["market_pointer"]),
            "sha256": market["pointer_sha256"],
            "as_of": "20260814",
        },
        "pit_membership": {
            "path": str(pit_membership),
            "sha256": _sha(_read(pit_membership)),
            "as_of": "20260814",
        },
        "expected_scope": {
            "path": str(scope),
            "sha256": _sha(_read(scope)),
            "as_of": "20260814",
        },
    }
    provenance["history_state"] = "mixed"
    provenance["machine_states"] = {
        "mixed": True,
        "legacy_direct_reader_provenance": "limited",
        "binding_aware_research_ready": True,
        "homogeneous_history_ready": False,
    }
    provenance.pop("mixed_generation")
    provenance["permanent_support_refs"] = {
        "support_manifest": {
            "path": "source/provider_manifest.json",
            "sha256": _sha(_read(support_path)),
        }
    }
    manifest["primary_provenance"] = provenance
    pointer["primary_provenance"] = provenance
    _write(manifest_path, _json_bytes(manifest))
    _write(pointer_path, _json_bytes(pointer))


def test_preflight_and_execute_false_are_strictly_read_only(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    before = _read(
        fixture["canonical"] / promotion.FUNDAMENTAL_POINTER_FILENAME
    )

    preflight = promotion.preflight_successor_promotion(
        staging_root=fixture["staging"],
        canonical_root=fixture["canonical"],
        expected_pointer_sha256=fixture["parent_pointer_sha256"],
        staging_validator=fixture["validator"],
    )
    dry = promotion.promote_successor_generation(
        staging_root=fixture["staging"],
        canonical_root=fixture["canonical"],
        expected_pointer_sha256=fixture["parent_pointer_sha256"],
        execute=False,
        staging_validator=fixture["validator"],
    )

    assert preflight["status"] == "PREFLIGHT_OK"
    assert dry["status"] == "PREFLIGHT_OK"
    assert dry["promoted"] is False
    assert _read(
        fixture["canonical"] / promotion.FUNDAMENTAL_POINTER_FILENAME
    ) == before
    assert list(fixture["journal"].iterdir()) == []
    assert not (
        fixture["canonical"]
        / promotion.FUNDAMENTAL_GENERATIONS_DIRNAME
        / fixture["generation_id"]
    ).exists()


def test_sealed_source_manifest_adapter_rebuilds_durable_pointer_bindings(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _use_sealed_source_binding(fixture)
    base_validator = fixture["validator"]

    def capture_style_validator(root: Path) -> dict[str, Any]:
        result = base_validator(root)
        pointer = json.loads(_read(root / promotion.FUNDAMENTAL_POINTER_FILENAME))
        evidence_root = (root / pointer["manifest_path"]).parent / "provider_evidence"
        result["provider_evidence_files"] = {
            path.relative_to(evidence_root).as_posix(): _sha(_read(path))
            for path in sorted(evidence_root.rglob("*"))
            if path.is_file()
        }
        return result

    preflight = promotion.preflight_successor_promotion(
        staging_root=fixture["staging"],
        canonical_root=fixture["canonical"],
        expected_pointer_sha256=fixture["parent_pointer_sha256"],
        staging_validator=capture_style_validator,
    )
    assert preflight["status"] == "PREFLIGHT_OK"
    assert preflight["market_pointer_sha256"] == _sha(
        _read(fixture["market_pointer"])
    )
    assert preflight["pit_pointer_sha256"] == _sha(_read(fixture["pit_pointer"]))


def test_real_stage_capture_provider_paths_match_generation_fileset(
    tmp_path: Path,
) -> None:
    fixture_module_path = Path(__file__).with_name(
        "test_fundamental_incremental_successor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_fundamental_incremental_successor_fixture",
        fixture_module_path,
    )
    assert spec is not None and spec.loader is not None
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    real_stage_root = tmp_path / "real-stage"
    real_stage_root.mkdir()
    bundle, targets, support_files, provider = fixture_module._path_backed_case(
        real_stage_root
    )
    from quant_investor.market.fundamental_incremental import (
        stage_successor_generation,
    )

    capture = stage_successor_generation(
        bundle,
        staging_root=tmp_path / "real-stage-output",
        generation_id="staged_successor",
        provider_manifest=provider,
        target_bindings=targets,
        provider_evidence_files=support_files,
    )
    scanned = promotion._scan_generation(capture.staging_root)
    declared = promotion._normalize_declared_provider_files(
        capture.provider_evidence_files,
        generation_root=scanned["generation_root"],
    )

    assert set(declared) == set(scanned["provider_files"])
    assert "provider_evidence/provider_manifest.json" in declared
    assert all(path.startswith("provider_evidence/") for path in declared)
    assert {
        path: item["sha256"] for path, item in declared.items()
    } == {
        path: identity.sha256
        for path, identity in scanned["provider_files"].items()
    }


def test_success_uses_fixed_lock_order_and_durable_phase_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    observed: list[str] = []
    original = promotion._secure_timed_lock

    @contextmanager
    def traced(path: Path, *, deadline: float, poll_seconds: float = 0.05):
        observed.append(path.name)
        with original(path, deadline=deadline, poll_seconds=poll_seconds):
            yield

    monkeypatch.setattr(promotion, "_secure_timed_lock", traced)
    result = _promote(fixture)

    assert result["promoted"] is True
    assert observed == [
        promotion.MARKET_WRITER_LOCK_FILENAME,
        promotion.PIT_WRITER_LOCK_FILENAME,
        promotion.FUNDAMENTAL_PROMOTION_LOCK_FILENAME,
    ]
    assert _journal_phases(fixture["journal"]) == [
        "INTENT",
        "PRECAS_VALIDATED",
        "INSTALL_INTENT",
        "GENERATION_INSTALLED",
        "CAS_COMMITTED",
        "POSTCHECK_PASSED",
        "TERMINAL",
    ]
    for record in (fixture["journal"] / "run-1").glob("*.json"):
        assert stat_mode(record) == 0o600
    assert stat_mode(fixture["journal"]) == 0o700


def test_predecessor_manifest_limit_does_not_expand_candidate_postcheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    observed_limits: list[tuple[str, int]] = []
    original = promotion._validate_pointer_references

    def traced(*args: Any, **kwargs: Any) -> dict[str, Any]:
        observed_limits.append(
            (
                str(kwargs.get("manifest_semantic_role", "successor_manifest")),
                int(
                    kwargs.get(
                        "manifest_maximum_bytes",
                        FUNDAMENTAL_GENERIC_JSON_MAX_BYTES,
                    )
                ),
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(promotion, "_validate_pointer_references", traced)
    result = _promote(fixture)

    assert result["promoted"] is True
    assert (
        FUNDAMENTAL_PREDECESSOR_MANIFEST_ROLE,
        FUNDAMENTAL_PREDECESSOR_MANIFEST_MAX_BYTES,
    ) in observed_limits
    assert (
        "successor_manifest",
        FUNDAMENTAL_GENERIC_JSON_MAX_BYTES,
    ) in observed_limits
    assert observed_limits[-1] == (
        "successor_manifest",
        FUNDAMENTAL_GENERIC_JSON_MAX_BYTES,
    )


def stat_mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode & 0o777


def test_expected_sha_mismatch_fails_before_journal_write(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(
        promotion.SuccessorPromotionError,
        match="embedded predecessor SHA256",
    ):
        promotion.preflight_successor_promotion(
            staging_root=fixture["staging"],
            canonical_root=fixture["canonical"],
            expected_pointer_sha256="0" * 64,
            staging_validator=fixture["validator"],
        )
    assert list(fixture["journal"].iterdir()) == []


def test_execute_rejects_preexisting_empty_journal_run(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    run_root = fixture["journal"] / "run-1"
    run_root.mkdir(mode=0o700)
    os.chmod(run_root, 0o700)

    with pytest.raises(
        promotion.SuccessorPromotionError,
        match="journal run already exists",
    ):
        _promote(fixture)
    assert _read(
        fixture["canonical"] / promotion.FUNDAMENTAL_POINTER_FILENAME
    ) == fixture["parent_pointer_bytes"]
    assert list(run_root.iterdir()) == []


def test_durable_pointer_bindings_require_immutable_refs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    base_validator = fixture["validator"]

    def missing_refs(root: Path) -> dict[str, Any]:
        result = base_validator(root)
        result["market_binding"] = dict(result["market_binding"])
        result["market_binding"]["immutable_refs"] = []
        return result

    with pytest.raises(
        promotion.SuccessorPromotionError,
        match="immutable refs are missing",
    ):
        promotion.preflight_successor_promotion(
            staging_root=fixture["staging"],
            canonical_root=fixture["canonical"],
            expected_pointer_sha256=fixture["parent_pointer_sha256"],
            staging_validator=missing_refs,
        )


def test_market_pointer_drift_inside_locks_blocks_cas(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    changed = False

    def drift(_capture: dict[str, Any], phase: str) -> bool:
        nonlocal changed
        if phase == "preflight" and not changed:
            changed = True
            _write(fixture["market_pointer"], _json_bytes({"drift": True}))
        return True

    with pytest.raises(
        promotion.SuccessorPromotionError,
        match="live market pointer CAS mismatch",
    ):
        _promote(fixture, live_binding_validator=drift)
    assert _read(
        fixture["canonical"] / promotion.FUNDAMENTAL_POINTER_FILENAME
    ) == fixture["parent_pointer_bytes"]
    assert _journal_phases(fixture["journal"])[-1] == "TERMINAL"


def test_symlink_lock_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = tmp_path / "lock-target"
    _write(target, b"")
    lock_path = fixture["market_pointer"].parent / promotion.MARKET_WRITER_LOCK_FILENAME
    lock_path.symlink_to(target)
    with pytest.raises(promotion.SuccessorPromotionError, match="writer lock is unsafe"):
        _promote(fixture)
    assert _read(
        fixture["canonical"] / promotion.FUNDAMENTAL_POINTER_FILENAME
    ) == fixture["parent_pointer_bytes"]


def test_lock_timeout_is_bounded(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    lock_path = fixture["market_pointer"].parent / promotion.MARKET_WRITER_LOCK_FILENAME
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            promotion.SuccessorPromotionError,
            match="writer lock acquisition timed out",
        ):
            _promote(fixture, lock_timeout_seconds=0.0)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class _Crash(BaseException):
    pass


def test_crash_before_rename_leaves_no_generation_and_is_recoverable(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def crash(phase: str) -> None:
        if phase == "before_generation_rename":
            raise _Crash()

    with pytest.raises(_Crash):
        _promote(fixture, fault_injector=crash)
    final_root = (
        fixture["canonical"]
        / promotion.FUNDAMENTAL_GENERATIONS_DIRNAME
        / fixture["generation_id"]
    )
    assert not final_root.exists()
    assert _journal_phases(fixture["journal"])[-1] == "INSTALL_INTENT"
    recovery = promotion.recover_successor_promotion(
        canonical_root=fixture["canonical"],
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        execute=True,
    )
    assert recovery["status"] == "ABANDONED_BEFORE_INSTALL"


def test_crash_after_rename_is_classified_as_retained_orphan(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def crash(phase: str) -> None:
        if phase == "after_generation_rename":
            raise _Crash()

    with pytest.raises(_Crash):
        _promote(fixture, fault_injector=crash)
    final_root = (
        fixture["canonical"]
        / promotion.FUNDAMENTAL_GENERATIONS_DIRNAME
        / fixture["generation_id"]
    )
    assert final_root.exists()
    assert _journal_phases(fixture["journal"])[-1] == "INSTALL_INTENT"
    dry = promotion.recover_successor_promotion(
        canonical_root=fixture["canonical"],
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        execute=False,
    )
    assert dry["status"] == "ORPHAN_RETAINED"
    recovered = promotion.recover_successor_promotion(
        canonical_root=fixture["canonical"],
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        execute=True,
    )
    assert recovered["status"] == "ORPHAN_RETAINED"
    assert final_root.exists()
    assert _read(
        fixture["canonical"] / promotion.FUNDAMENTAL_POINTER_FILENAME
    ) == fixture["parent_pointer_bytes"]


def test_recovery_closes_crash_after_pointer_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    observed: list[str] = []
    original = promotion._secure_timed_lock

    @contextmanager
    def traced(path: Path, *, deadline: float, poll_seconds: float = 0.05):
        observed.append(path.name)
        with original(path, deadline=deadline, poll_seconds=poll_seconds):
            yield

    monkeypatch.setattr(promotion, "_secure_timed_lock", traced)

    def crash(phase: str) -> None:
        if phase == "after_pointer_write":
            raise _Crash()

    with pytest.raises(_Crash):
        _promote(fixture, fault_injector=crash)
    assert _journal_phases(fixture["journal"])[-1] == "GENERATION_INSTALLED"
    recovered = promotion.recover_successor_promotion(
        canonical_root=fixture["canonical"],
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        execute=True,
    )
    assert recovered["status"] == "SUCCESS"
    assert observed == [
        promotion.MARKET_WRITER_LOCK_FILENAME,
        promotion.PIT_WRITER_LOCK_FILENAME,
        promotion.FUNDAMENTAL_PROMOTION_LOCK_FILENAME,
        promotion.MARKET_WRITER_LOCK_FILENAME,
        promotion.PIT_WRITER_LOCK_FILENAME,
        promotion.FUNDAMENTAL_PROMOTION_LOCK_FILENAME,
    ]
    assert _journal_phases(fixture["journal"])[-3:] == [
        "CAS_COMMITTED",
        "POSTCHECK_PASSED",
        "TERMINAL",
    ]


def test_exception_after_pointer_write_leaves_recoverable_open_journal(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def interrupt(phase: str) -> None:
        if phase == "after_pointer_write":
            raise RuntimeError("journal write interruption")

    with pytest.raises(
        promotion.SuccessorPromotionError,
        match="recovery is required",
    ) as caught:
        _promote(fixture, fault_injector=interrupt)
    assert caught.value.status == "PROMOTION_UNCERTAIN"
    assert _journal_phases(fixture["journal"])[-1] == "GENERATION_INSTALLED"

    recovered = promotion.recover_successor_promotion(
        canonical_root=fixture["canonical"],
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        execute=True,
    )
    assert recovered["status"] == "SUCCESS"
    assert _journal_phases(fixture["journal"])[-3:] == [
        "CAS_COMMITTED",
        "POSTCHECK_PASSED",
        "TERMINAL",
    ]


def test_postcheck_failure_restores_exact_predecessor_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def reject(_capture: dict[str, Any], phase: str) -> bool:
        return phase != "locked_postcheck"

    with pytest.raises(
        promotion.SuccessorPromotionError,
        match="exact predecessor bytes restored",
    ):
        _promote(fixture, live_binding_validator=reject)
    assert _read(
        fixture["canonical"] / promotion.FUNDAMENTAL_POINTER_FILENAME
    ) == fixture["parent_pointer_bytes"]
    assert _journal_phases(fixture["journal"])[-2:] == [
        "ROLLBACK_COMMITTED",
        "TERMINAL",
    ]
    final_root = (
        fixture["canonical"]
        / promotion.FUNDAMENTAL_GENERATIONS_DIRNAME
        / fixture["generation_id"]
    )
    assert final_root.exists()


def test_third_party_pointer_drift_refuses_blind_rollback(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    third_party_bytes = _json_bytes({"third_party": True})

    def drift(_capture: dict[str, Any], phase: str) -> bool:
        if phase == "locked_postcheck":
            _write(
                fixture["canonical"] / promotion.FUNDAMENTAL_POINTER_FILENAME,
                third_party_bytes,
            )
            return False
        return True

    with pytest.raises(
        promotion.SuccessorPromotionError,
        match="rollback refused",
    ) as caught:
        _promote(fixture, live_binding_validator=drift)
    assert caught.value.status == "PROMOTION_UNCERTAIN"
    assert _read(
        fixture["canonical"] / promotion.FUNDAMENTAL_POINTER_FILENAME
    ) == third_party_bytes
    terminal = json.loads(
        _read(sorted((fixture["journal"] / "run-1").glob("*.json"))[-1])
    )
    assert terminal["details"]["status"] == "PROMOTION_UNCERTAIN"


def test_large_table_copy_never_calls_path_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)

    def forbidden(_path: Path) -> bytes:
        raise AssertionError("Path.read_bytes must not be used")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    result = _promote(fixture)
    assert result["promoted"] is True
    final_daily = (
        fixture["canonical"]
        / promotion.FUNDAMENTAL_GENERATIONS_DIRNAME
        / fixture["generation_id"]
        / "fundamental_daily.parquet"
    )
    assert final_daily.stat().st_size > 6 * 1024 * 1024


def test_historical_validation_survives_live_market_and_pit_advance(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _promote(fixture)
    _write(fixture["market_pointer"], _json_bytes({"new_market": True}))
    _write(fixture["pit_pointer"], _json_bytes({"new_pit": True}))

    historical = promotion.validate_successor_historical_evidence(
        staging_root=fixture["canonical"],
        staging_validator=fixture["validator"],
    )
    assert historical["status"] == "OK"
    assert historical["historical_only"] is True


def test_v3_to_v3_keeps_original_seam_and_cycle_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        parent_schema=promotion.SUCCESSOR_PROVENANCE_SCHEMA,
        parent_cutoff="20260814",
        original_seam="20260806",
        target_cutoff="20260815",
    )
    preflight = promotion.preflight_successor_promotion(
        staging_root=fixture["staging"],
        canonical_root=fixture["canonical"],
        expected_pointer_sha256=fixture["parent_pointer_sha256"],
        staging_validator=fixture["validator"],
    )
    assert preflight["original_seam"] == "20260806"
    assert preflight["immediate_parent_cutoff"] == "20260814"

    manifest_path = next(
        (fixture["staging"] / promotion.FUNDAMENTAL_GENERATIONS_DIRNAME).glob(
            "*/manifest.json"
        )
    )
    manifest = json.loads(_read(manifest_path))
    manifest["primary_provenance"]["successor_chain"]["generation_ids"] = [
        "parent_v3",
        "parent_v3",
    ]
    _write(manifest_path, _json_bytes(manifest))
    pointer_path = fixture["staging"] / promotion.FUNDAMENTAL_POINTER_FILENAME
    pointer = json.loads(_read(pointer_path))
    pointer["primary_provenance"] = manifest["primary_provenance"]
    _write(pointer_path, _json_bytes(pointer))
    with pytest.raises(promotion.SuccessorPromotionError, match="cycle"):
        promotion.preflight_successor_promotion(
            staging_root=fixture["staging"],
            canonical_root=fixture["canonical"],
            expected_pointer_sha256=fixture["parent_pointer_sha256"],
            staging_validator=fixture["validator"],
        )


def test_tampered_captured_pointer_bytes_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = next(
        (fixture["staging"] / promotion.FUNDAMENTAL_GENERATIONS_DIRNAME).glob(
            "*/manifest.json"
        )
    )
    manifest = json.loads(_read(manifest_path))
    manifest["primary_provenance"]["predecessor"][
        "exact_pointer_bytes_b64"
    ] = base64.b64encode(b"{}\n").decode("ascii")
    _write(manifest_path, _json_bytes(manifest))
    pointer_path = fixture["staging"] / promotion.FUNDAMENTAL_POINTER_FILENAME
    pointer = json.loads(_read(pointer_path))
    pointer["primary_provenance"] = manifest["primary_provenance"]
    _write(pointer_path, _json_bytes(pointer))
    with pytest.raises(promotion.SuccessorPromotionError, match="pointer SHA256 mismatch"):
        promotion.validate_successor_historical_evidence(
            staging_root=fixture["staging"],
            staging_validator=fixture["validator"],
        )
