from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.factors.governance import (
    FactorLifecycleState,
    FactorRecord,
    GATE_SPECS,
    GateResult,
)
from quant_investor.factors.registry_store import (
    METADATA_ABSENT,
    FactorRegistryConflictError,
    FactorRegistryMalformedError,
    FactorRegistryMissingError,
    FactorRegistryValidationError,
    apply_factor_record_patch,
    factor_record_sha256,
    load_registry_snapshot_strict,
    registry_file_sha256,
    rollback_factor_record_patch,
)


def _record(
    name: str,
    *,
    weight: float = 0.05,
    owner: str = "fixture",
) -> dict:
    return FactorRecord(
        name=name,
        version="v1",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        implementation=f"builtin:{name}",
        weight=weight,
        owner=owner,
        gate_results=[
            GateResult(
                gate_id=spec.gate_id,
                gate_key=spec.key,
                title=spec.title,
                passed=True,
            )
            for spec in GATE_SPECS
        ],
        metadata={"fixture": True},
    ).to_dict()


def _write_registry(path: Path, records: list[dict]) -> bytes:
    payload = {
        "schema_version": "mined-factor-registry.v1",
        "metadata": {"fixture": True},
        "factors": records,
    }
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _journal(tmp_path: Path, name: str) -> Path:
    return tmp_path / "journals" / f"{name}.json"


def test_strict_snapshot_reports_exact_registry_and_stable_record_hashes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    raw = _write_registry(path, [_record("alpha"), _record("beta")])

    snapshot = load_registry_snapshot_strict(path)

    assert snapshot.registry_sha256 == registry_file_sha256(path)
    assert snapshot.registry_sha256 == hashlib.sha256(raw).hexdigest()
    assert set(snapshot.record_payloads) == {"alpha", "beta"}
    assert snapshot.record_sha256s["alpha"] == factor_record_sha256(
        snapshot.record_payloads["alpha"]
    )
    reversed_alpha = dict(
        reversed(list(snapshot.record_payloads["alpha"].items()))
    )
    assert (
        factor_record_sha256(reversed_alpha)
        == snapshot.record_sha256s["alpha"]
    )


def test_strict_loader_fails_closed_for_missing_malformed_and_duplicate(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FactorRegistryMissingError):
        load_registry_snapshot_strict(missing)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad\n", encoding="utf-8")
    with pytest.raises(FactorRegistryMalformedError):
        load_registry_snapshot_strict(malformed)

    non_finite = tmp_path / "non_finite.json"
    non_finite.write_text(
        '{"schema_version":"v1","metadata":{"value":NaN},"factors":[]}',
        encoding="utf-8",
    )
    with pytest.raises(FactorRegistryMalformedError):
        load_registry_snapshot_strict(non_finite)

    duplicate = tmp_path / "duplicate.json"
    _write_registry(duplicate, [_record("alpha"), _record("alpha")])
    with pytest.raises(
        FactorRegistryValidationError,
        match="duplicate factor name",
    ):
        load_registry_snapshot_strict(duplicate)

    missing_write = tmp_path / "missing_write.json"
    with pytest.raises(FactorRegistryMissingError):
        apply_factor_record_patch(
            missing_write,
            {"alpha": _record("alpha")},
            expected_registry_sha256="0" * 64,
            expected_record_sha256s={"alpha": None},
            mutation_id="mutation-missing-registry",
            reason="missing registry fixture",
            journal_path=_journal(tmp_path, "missing-registry"),
            write=True,
        )
    assert not missing_write.with_name(
        f".{missing_write.name}.lock"
    ).exists()


def test_strict_loader_rejects_future_registry_schema(tmp_path: Path) -> None:
    path = tmp_path / "future_schema.json"
    _write_registry(path, [_record("alpha")])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "mined-factor-registry.v2"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        FactorRegistryValidationError,
        match="unsupported factor registry schema_version",
    ):
        load_registry_snapshot_strict(path)


def test_strict_loader_rejects_unknown_factor_record_field(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unknown_record_field.json"
    record = _record("alpha")
    record["future_extension"] = {"unsafe_to_rewrite": True}
    _write_registry(path, [record])

    with pytest.raises(
        FactorRegistryValidationError,
        match="unsupported fields.*future_extension",
    ):
        load_registry_snapshot_strict(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weight", True),
        ("weight", "0.05"),
        ("direction", True),
        ("direction", "1"),
        ("direction", 0),
        ("direction", 2),
        ("horizon_days", True),
        ("horizon_days", "5"),
        ("thematic", 1),
        ("narrow_coverage", "false"),
    ],
)
def test_strict_loader_rejects_coercible_record_field_types(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / f"invalid_{field}.json"
    record = _record("alpha")
    record[field] = value
    _write_registry(path, [record])

    with pytest.raises(FactorRegistryValidationError):
        load_registry_snapshot_strict(path)


@pytest.mark.parametrize("passed", [1, 0, "true", "false", None])
def test_strict_loader_rejects_non_boolean_gate_passed(
    tmp_path: Path,
    passed: object,
) -> None:
    path = tmp_path / "invalid_gate_passed.json"
    record = _record("alpha")
    record["gate_results"][0]["passed"] = passed
    _write_registry(path, [record])

    with pytest.raises(
        FactorRegistryValidationError,
        match="passed must be a boolean",
    ):
        load_registry_snapshot_strict(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_gate",
        "duplicate_gate",
        "mismatched_gate_key",
        "extra_gate",
        "unknown_gate_field",
    ],
)
def test_strict_loader_requires_exact_unique_gate_1_to_8_contract(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / f"{mutation}.json"
    record = _record("alpha")
    gates = record["gate_results"]
    if mutation == "missing_gate":
        gates.pop()
    elif mutation == "duplicate_gate":
        gates[-1] = copy.deepcopy(gates[0])
    elif mutation == "mismatched_gate_key":
        gates[0]["gate_key"] = "coverage_stability"
    elif mutation == "extra_gate":
        extra = copy.deepcopy(gates[-1])
        extra["gate_id"] = 9
        extra["gate_key"] = "gate_9"
        gates.append(extra)
    else:
        gates[0]["future_extension"] = True
    _write_registry(path, [record])

    with pytest.raises(FactorRegistryValidationError):
        load_registry_snapshot_strict(path)


def test_strict_loader_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    path = tmp_path / "unknown_top_level.json"
    _write_registry(path, [_record("alpha")])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["future_extension"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        FactorRegistryValidationError,
        match="unsupported top-level fields",
    ):
        load_registry_snapshot_strict(path)


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":"mined-factor-registry.v1",'
        '"schema_version":"mined-factor-registry.v1","metadata":{},"factors":[]}',
        '{"schema_version":"mined-factor-registry.v1","metadata":{},"factors":['
        '{"name":"alpha","name":"beta"}]}',
        '{"schema_version":"mined-factor-registry.v1","metadata":{},"factors":['
        '{"name":"alpha","state":"production_factor","gate_results":['
        '{"gate_id":1,"gate_id":2}]}]}',
    ],
)
def test_strict_loader_rejects_duplicate_json_keys(
    tmp_path: Path,
    raw: str,
) -> None:
    path = tmp_path / "duplicate_key.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(
        FactorRegistryMalformedError,
        match="duplicate JSON key",
    ):
        load_registry_snapshot_strict(path)


def test_patch_defaults_to_dry_run_and_emits_exact_inverse_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    original = _write_registry(path, [_record("alpha"), _record("beta")])
    snapshot = load_registry_snapshot_strict(path)
    updated_alpha = copy.deepcopy(snapshot.record_payloads["alpha"])
    updated_alpha["weight"] = 0.02

    manifest = apply_factor_record_patch(
        path,
        {"alpha": updated_alpha},
        expected_registry_sha256=snapshot.registry_sha256,
        expected_record_sha256s={"alpha": snapshot.record_sha256s["alpha"]},
        mutation_id="mutation-dry-run",
        reason="fixture dry run",
    )

    assert path.read_bytes() == original
    assert manifest["status"] == "dry_run"
    assert manifest["write_requested"] is False
    assert manifest["applied"] is False
    assert (
        manifest["changed_records"][0]["before_record"]
        == snapshot.record_payloads["alpha"]
    )
    assert manifest["changed_records"][0]["after_record"] == updated_alpha
    assert (
        manifest["inverse_patch"]["records"]["alpha"]
        == snapshot.record_payloads["alpha"]
    )
    assert not path.with_name(f".{path.name}.lock").exists()


def test_explicit_write_applies_atomic_patch_and_verifies_readback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from quant_investor.factors import registry_store

    path = tmp_path / "registry.json"
    _write_registry(path, [_record("alpha"), _record("beta")])
    path.chmod(0o640)
    snapshot = load_registry_snapshot_strict(path)
    updated_alpha = copy.deepcopy(snapshot.record_payloads["alpha"])
    updated_alpha["weight"] = 0.02
    fsync_calls: list[int] = []
    real_fsync = registry_store.os.fsync

    def track_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        real_fsync(file_descriptor)

    monkeypatch.setattr(registry_store.os, "fsync", track_fsync)

    manifest = apply_factor_record_patch(
        path,
        {"alpha": updated_alpha},
        expected_registry_sha256=snapshot.registry_sha256,
        expected_record_sha256s={"alpha": snapshot.record_sha256s["alpha"]},
        mutation_id="mutation-write",
        reason="fixture write",
        journal_path=_journal(tmp_path, "mutation-write"),
        write=True,
    )
    readback = load_registry_snapshot_strict(path)

    assert manifest["status"] == "applied"
    assert manifest["write_requested"] is True
    assert manifest["applied"] is True
    assert manifest["readback_registry_sha256"] == readback.registry_sha256
    assert manifest["after_registry_sha256"] == readback.registry_sha256
    assert readback.record_payloads["alpha"]["weight"] == 0.02
    assert readback.record_payloads["beta"] == snapshot.record_payloads["beta"]
    assert len(fsync_calls) >= 6
    assert path.stat().st_mode & 0o777 == 0o640
    journal = json.loads(
        Path(manifest["journal_path"]).read_text(encoding="utf-8")
    )
    assert journal == manifest


def test_whole_file_cas_rejects_a_stale_registry_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    _write_registry(path, [_record("alpha")])
    stale = load_registry_snapshot_strict(path)
    current_alpha = copy.deepcopy(stale.record_payloads["alpha"])
    current_alpha["owner"] = "concurrent"
    _write_registry(path, [current_alpha])
    requested = copy.deepcopy(stale.record_payloads["alpha"])
    requested["weight"] = 0.01

    with pytest.raises(
        FactorRegistryConflictError,
        match="registry CAS conflict",
    ):
        apply_factor_record_patch(
            path,
            {"alpha": requested},
            expected_registry_sha256=stale.registry_sha256,
            expected_record_sha256s={"alpha": stale.record_sha256s["alpha"]},
            mutation_id="mutation-stale-registry",
            reason="stale registry fixture",
            journal_path=_journal(tmp_path, "stale-registry"),
            write=True,
        )

    current = load_registry_snapshot_strict(path)
    assert current.record_payloads["alpha"]["owner"] == "concurrent"


def test_target_record_cas_rejects_wrong_hash_and_existing_add(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    original = _write_registry(path, [_record("alpha")])
    snapshot = load_registry_snapshot_strict(path)
    updated = copy.deepcopy(snapshot.record_payloads["alpha"])
    updated["weight"] = 0.01

    with pytest.raises(
        FactorRegistryConflictError,
        match="record CAS conflict",
    ):
        apply_factor_record_patch(
            path,
            {"alpha": updated},
            expected_registry_sha256=snapshot.registry_sha256,
            expected_record_sha256s={"alpha": "0" * 64},
            mutation_id="mutation-wrong-record",
            reason="wrong record hash fixture",
            journal_path=_journal(tmp_path, "wrong-record"),
            write=True,
        )
    with pytest.raises(FactorRegistryConflictError, match="expected absent"):
        apply_factor_record_patch(
            path,
            {"alpha": updated},
            expected_registry_sha256=snapshot.registry_sha256,
            expected_record_sha256s={"alpha": None},
            mutation_id="mutation-existing-add",
            reason="existing add fixture",
        )

    assert path.read_bytes() == original


def test_atomic_replace_failure_preserves_original_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from quant_investor.factors import registry_store

    path = tmp_path / "registry.json"
    original = _write_registry(path, [_record("alpha")])
    snapshot = load_registry_snapshot_strict(path)
    updated = copy.deepcopy(snapshot.record_payloads["alpha"])
    updated["weight"] = 0.01

    real_replace = registry_store.os.replace

    def fail_replace(source: str | Path, target: str | Path) -> None:
        if Path(target) == path:
            assert Path(source).parent == path.parent
            raise OSError("replace failed")
        real_replace(source, target)

    monkeypatch.setattr(registry_store.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        apply_factor_record_patch(
            path,
            {"alpha": updated},
            expected_registry_sha256=snapshot.registry_sha256,
            expected_record_sha256s={
                "alpha": snapshot.record_sha256s["alpha"]
            },
            mutation_id="mutation-replace-failure",
            reason="atomic failure fixture",
            journal_path=_journal(tmp_path, "replace-failure"),
            write=True,
        )

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_prepared_journal_before_registry_replace_cannot_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from quant_investor.factors import registry_store

    path = tmp_path / "registry.json"
    original = _write_registry(path, [_record("alpha")])
    snapshot = load_registry_snapshot_strict(path)
    updated = copy.deepcopy(snapshot.record_payloads["alpha"])
    updated["weight"] = 0.01
    journal_path = _journal(tmp_path, "prepared-before-replace")

    def crash_before_replace(*_args, **_kwargs) -> None:
        raise RuntimeError("fixture crash before registry replace")

    monkeypatch.setattr(
        registry_store,
        "_atomic_replace_registry",
        crash_before_replace,
    )
    with pytest.raises(RuntimeError, match="before registry replace"):
        apply_factor_record_patch(
            path,
            {"alpha": updated},
            expected_registry_sha256=snapshot.registry_sha256,
            expected_record_sha256s={
                "alpha": snapshot.record_sha256s["alpha"]
            },
            mutation_id="prepared-before-replace",
            reason="prepared journal before-state fixture",
            journal_path=journal_path,
            write=True,
        )

    prepared = json.loads(journal_path.read_text(encoding="utf-8"))
    assert prepared["status"] == "prepared"
    assert prepared["write_requested"] is True
    assert prepared["applied"] is False
    assert path.read_bytes() == original
    with pytest.raises(FactorRegistryConflictError, match="record CAS"):
        rollback_factor_record_patch(
            path,
            prepared,
            mutation_id="rollback-prepared-before-replace",
            journal_path=_journal(tmp_path, "rollback-prepared-before"),
            write=True,
        )


def test_prepared_journal_after_registry_replace_can_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from quant_investor.factors import registry_store

    path = tmp_path / "registry.json"
    _write_registry(path, [_record("alpha")])
    initial = load_registry_snapshot_strict(path)
    updated = copy.deepcopy(initial.record_payloads["alpha"])
    updated["weight"] = 0.01
    journal_path = _journal(tmp_path, "prepared-after-replace")
    real_journal_write = registry_store._atomic_write_mutation_journal

    def crash_on_applied_journal(
        target: Path,
        payload: dict,
        *,
        replace_existing: bool,
    ) -> None:
        if replace_existing:
            raise RuntimeError("fixture crash after registry replace")
        real_journal_write(
            target,
            payload,
            replace_existing=replace_existing,
        )

    monkeypatch.setattr(
        registry_store,
        "_atomic_write_mutation_journal",
        crash_on_applied_journal,
    )
    with pytest.raises(RuntimeError, match="after registry replace"):
        apply_factor_record_patch(
            path,
            {"alpha": updated},
            expected_registry_sha256=initial.registry_sha256,
            expected_record_sha256s={
                "alpha": initial.record_sha256s["alpha"]
            },
            mutation_id="prepared-after-replace",
            reason="prepared journal after-state fixture",
            journal_path=journal_path,
            write=True,
        )

    prepared = json.loads(journal_path.read_text(encoding="utf-8"))
    changed = load_registry_snapshot_strict(path)
    assert prepared["status"] == "prepared"
    assert changed.record_payloads["alpha"]["weight"] == 0.01

    monkeypatch.setattr(
        registry_store,
        "_atomic_write_mutation_journal",
        real_journal_write,
    )
    rollback = rollback_factor_record_patch(
        path,
        prepared,
        mutation_id="rollback-prepared-after-replace",
        journal_path=_journal(tmp_path, "rollback-prepared-after"),
        write=True,
    )
    restored = load_registry_snapshot_strict(path)
    assert rollback["status"] == "applied"
    assert restored.record_payloads == initial.record_payloads


def test_record_level_rollback_preserves_unrelated_later_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    _write_registry(path, [_record("alpha"), _record("beta")])
    initial = load_registry_snapshot_strict(path)
    changed_alpha = copy.deepcopy(initial.record_payloads["alpha"])
    changed_alpha["weight"] = 0.02
    alpha_manifest = apply_factor_record_patch(
        path,
        {"alpha": changed_alpha},
        expected_registry_sha256=initial.registry_sha256,
        expected_record_sha256s={"alpha": initial.record_sha256s["alpha"]},
        mutation_id="mutation-alpha",
        reason="change alpha",
        journal_path=_journal(tmp_path, "mutation-alpha"),
        write=True,
    )

    after_alpha = load_registry_snapshot_strict(path)
    changed_beta = copy.deepcopy(after_alpha.record_payloads["beta"])
    changed_beta["owner"] = "later-unrelated-change"
    apply_factor_record_patch(
        path,
        {"beta": changed_beta},
        expected_registry_sha256=after_alpha.registry_sha256,
        expected_record_sha256s={"beta": after_alpha.record_sha256s["beta"]},
        mutation_id="mutation-beta",
        reason="change beta later",
        journal_path=_journal(tmp_path, "mutation-beta"),
        write=True,
    )

    rollback = rollback_factor_record_patch(
        path,
        alpha_manifest,
        mutation_id="rollback-alpha",
        journal_path=_journal(tmp_path, "rollback-alpha"),
        write=True,
    )
    final = load_registry_snapshot_strict(path)

    assert rollback["status"] == "applied"
    assert rollback["rollback_of"] == "mutation-alpha"
    rollback_journal = json.loads(
        _journal(tmp_path, "rollback-alpha").read_text(encoding="utf-8")
    )
    assert rollback_journal["rollback_of"] == "mutation-alpha"
    assert rollback_journal == rollback
    assert final.record_payloads["alpha"] == initial.record_payloads["alpha"]
    assert final.record_payloads["beta"] == changed_beta

    with pytest.raises(FactorRegistryConflictError):
        rollback_factor_record_patch(
            path,
            alpha_manifest,
            mutation_id="rollback-alpha-again",
            journal_path=_journal(tmp_path, "rollback-alpha-again"),
            write=True,
        )


def test_inverse_patch_restores_deleted_record_and_removes_added_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    _write_registry(path, [_record("alpha"), _record("beta")])
    initial = load_registry_snapshot_strict(path)
    gamma = _record("gamma", weight=0.0)

    manifest = apply_factor_record_patch(
        path,
        {"alpha": None, "gamma": gamma},
        expected_registry_sha256=initial.registry_sha256,
        expected_record_sha256s={
            "alpha": initial.record_sha256s["alpha"],
            "gamma": None,
        },
        mutation_id="mutation-add-delete",
        reason="add and delete fixture",
        journal_path=_journal(tmp_path, "mutation-add-delete"),
        write=True,
    )
    changed = load_registry_snapshot_strict(path)
    assert set(changed.record_payloads) == {"beta", "gamma"}

    dry_run = rollback_factor_record_patch(
        path,
        manifest,
        mutation_id="rollback-add-delete-dry",
    )
    assert dry_run["status"] == "dry_run"
    dry_readback = load_registry_snapshot_strict(path)
    assert set(dry_readback.record_payloads) == {"beta", "gamma"}

    rollback_factor_record_patch(
        path,
        manifest,
        mutation_id="rollback-add-delete",
        journal_path=_journal(tmp_path, "rollback-add-delete"),
        write=True,
    )
    restored = load_registry_snapshot_strict(path)
    assert restored.record_payloads == initial.record_payloads


def test_metadata_cas_inverse_preserves_unrelated_later_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    _write_registry(path, [_record("alpha")])
    initial = load_registry_snapshot_strict(path)

    metadata_manifest = apply_factor_record_patch(
        path,
        {},
        expected_registry_sha256=initial.registry_sha256,
        expected_record_sha256s={},
        metadata_updates={
            "fixture": "changed",
            "health_run": {"status": "applied"},
        },
        expected_metadata_values={
            "fixture": True,
            "health_run": METADATA_ABSENT,
        },
        mutation_id="mutation-metadata",
        reason="metadata fixture",
        journal_path=_journal(tmp_path, "mutation-metadata"),
        write=True,
    )
    changed = load_registry_snapshot_strict(path)
    assert changed.metadata_payload["fixture"] == "changed"
    assert changed.metadata_payload["health_run"] == {"status": "applied"}
    assert metadata_manifest["changed_metadata_count"] == 2

    apply_factor_record_patch(
        path,
        {},
        expected_registry_sha256=changed.registry_sha256,
        expected_record_sha256s={},
        metadata_updates={"unrelated": 42},
        expected_metadata_values={"unrelated": METADATA_ABSENT},
        mutation_id="mutation-unrelated-metadata",
        reason="later unrelated metadata fixture",
        journal_path=_journal(tmp_path, "unrelated-metadata"),
        write=True,
    )

    rollback_factor_record_patch(
        path,
        metadata_manifest,
        mutation_id="rollback-metadata",
        journal_path=_journal(tmp_path, "rollback-metadata"),
        write=True,
    )
    restored = load_registry_snapshot_strict(path)
    assert restored.metadata_payload == {"fixture": True, "unrelated": 42}
    assert restored.record_payloads == initial.record_payloads

    with pytest.raises(FactorRegistryConflictError, match="metadata CAS"):
        apply_factor_record_patch(
            path,
            {},
            expected_registry_sha256=restored.registry_sha256,
            expected_record_sha256s={},
            metadata_updates={"fixture": False},
            expected_metadata_values={"fixture": "wrong"},
            mutation_id="mutation-metadata-conflict",
            reason="metadata conflict fixture",
        )


def test_patch_rejects_name_mismatch_noop_and_incomplete_preconditions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    _write_registry(path, [_record("alpha")])
    snapshot = load_registry_snapshot_strict(path)

    with pytest.raises(FactorRegistryValidationError, match="does not match"):
        apply_factor_record_patch(
            path,
            {"alpha": _record("beta")},
            expected_registry_sha256=snapshot.registry_sha256,
            expected_record_sha256s={
                "alpha": snapshot.record_sha256s["alpha"]
            },
            mutation_id="mutation-name-mismatch",
            reason="name mismatch fixture",
        )
    with pytest.raises(FactorRegistryValidationError, match="does not change"):
        apply_factor_record_patch(
            path,
            {"alpha": snapshot.record_payloads["alpha"]},
            expected_registry_sha256=snapshot.registry_sha256,
            expected_record_sha256s={
                "alpha": snapshot.record_sha256s["alpha"]
            },
            mutation_id="mutation-noop",
            reason="noop fixture",
        )
    with pytest.raises(FactorRegistryValidationError, match="exactly"):
        apply_factor_record_patch(
            path,
            {"alpha": None},
            expected_registry_sha256=snapshot.registry_sha256,
            expected_record_sha256s={},
            mutation_id="mutation-missing-precondition",
            reason="missing precondition fixture",
        )

    dry_manifest = apply_factor_record_patch(
        path,
        {"alpha": None},
        expected_registry_sha256=snapshot.registry_sha256,
        expected_record_sha256s={
            "alpha": snapshot.record_sha256s["alpha"]
        },
        mutation_id="mutation-not-applied",
        reason="dry manifest fixture",
    )
    with pytest.raises(
        FactorRegistryValidationError,
        match="applied or prepared write mutation",
    ):
        rollback_factor_record_patch(
            path,
            dry_manifest,
            mutation_id="rollback-not-applied",
        )
