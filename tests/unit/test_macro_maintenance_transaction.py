from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_investor.macro import maintenance_transaction as transaction
from quant_investor.macro import maintenance


def test_retrospective_targets_derive_from_frozen_parents(tmp_path: Path, monkeypatch) -> None:
    generation = tmp_path / "release-generation"
    generation.mkdir()
    (generation / "market_open_days.json").write_text(
        json.dumps(
            {
                "open_dates": [
                    "20260821",
                    "20260824",
                    "20260825",
                    "20260826",
                    "20260827",
                ]
            }
        )
    )
    monkeypatch.setattr(
        maintenance,
        "load_release_calendar",
        lambda **_kwargs: SimpleNamespace(
            identity=SimpleNamespace(generation_path=str(generation))
        ),
    )
    monkeypatch.setattr(maintenance, "observation_pointer_sha256", lambda _root: "o" * 64)
    monkeypatch.setattr(
        maintenance,
        "load_observations",
        lambda _root: (
            [],
            {"generation_manifest": {"metadata": {"local_target_trade_date": "20260821"}}},
        ),
    )

    assert maintenance._expected_retrospective_coverage_targets(
        release_root=tmp_path / "release",
        expected_release_pointer_sha256="r" * 64,
        observations_root=tmp_path / "observations",
        expected_observations_pointer_sha256="o" * 64,
        target_date="20260827",
    ) == ["20260824", "20260825", "20260826"]


def test_retrospective_targets_reject_non_open_final_target(tmp_path: Path, monkeypatch) -> None:
    generation = tmp_path / "release-generation"
    generation.mkdir()
    (generation / "market_open_days.json").write_text(
        json.dumps({"open_dates": ["20260821", "20260824", "20260825"]})
    )
    monkeypatch.setattr(
        maintenance,
        "load_release_calendar",
        lambda **_kwargs: SimpleNamespace(
            identity=SimpleNamespace(generation_path=str(generation))
        ),
    )
    monkeypatch.setattr(maintenance, "observation_pointer_sha256", lambda _root: "o" * 64)
    monkeypatch.setattr(
        maintenance,
        "load_observations",
        lambda _root: (
            [],
            {"generation_manifest": {"metadata": {"local_target_trade_date": "20260821"}}},
        ),
    )

    with pytest.raises(maintenance.MacroMaintenanceError, match="catch_up_window_invalid"):
        maintenance._expected_retrospective_coverage_targets(
            release_root=tmp_path / "release",
            expected_release_pointer_sha256="r" * 64,
            observations_root=tmp_path / "observations",
            expected_observations_pointer_sha256="o" * 64,
            target_date="20260827",
        )


def _raw(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _private(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def _store(root: Path, generation_id: str) -> tuple[bytes, str]:
    generations = _private(root / "_generations")
    generation = _private(generations / generation_id)
    manifest = _raw({"generation_id": generation_id, "status": "OK"})
    (generation / "manifest.json").write_bytes(manifest)
    os.chmod(generation / "manifest.json", 0o600)
    pointer = _raw({"generation_id": generation_id, "status": "OK"})
    (root / "_latest.json").write_bytes(pointer)
    os.chmod(root / "_latest.json", 0o600)
    return pointer, _sha(pointer)


def _fixture(tmp_path: Path, *, authority_mode: str = "canonical"):
    market = _private(tmp_path / "market")
    market_pointer = market / "_latest.json"
    market_raw = _raw({"generation_id": "market-active", "status": "OK"})
    market_pointer.write_bytes(market_raw)
    os.chmod(market_pointer, 0o600)
    pit = _private(tmp_path / "pit")
    pit_pointer = pit / "stock_basic_membership_latest.json"
    pit_raw = _raw({"generation_id": "pit-active", "status": "OK"})
    pit_pointer.write_bytes(pit_raw)
    os.chmod(pit_pointer, 0o600)
    release = _private(tmp_path / "release")
    observations = _private(tmp_path / "observations")
    release_old_raw, release_old = _store(release, "release-parent")
    observations_old_raw, observations_old = _store(observations, "observations-parent")
    del release_old_raw, observations_old_raw

    candidate_release = _private(tmp_path / "candidate-release")
    candidate_observations = _private(tmp_path / "candidate-observations")
    release_new_raw, release_new = _store(candidate_release, "release-child")
    observations_new_raw, observations_new = _store(candidate_observations, "observations-child")
    del release_new_raw, observations_new_raw

    prepared = _private(tmp_path / "prepared")
    sealed = transaction.seal_prepared_macro_transaction(
        prepared_root=prepared,
        release_candidate_root=candidate_release,
        observations_candidate_root=candidate_observations,
        release_canonical_root=release,
        observations_canonical_root=observations,
        expected_release_pointer_sha256=release_old,
        expected_observations_pointer_sha256=observations_old,
        market_pointer_path=market_pointer,
        expected_market_pointer_sha256=_sha(market_raw),
        pit_pointer_path=pit_pointer,
        expected_pit_pointer_sha256=_sha(pit_raw),
        authority_mode=authority_mode,
        target_date="20260819",
    )
    journal = _private(tmp_path / "journal")
    return {
        "release": release,
        "observations": observations,
        "market_pointer": market_pointer,
        "market_sha": _sha(market_raw),
        "pit_pointer": pit_pointer,
        "pit_sha": _sha(pit_raw),
        "release_old": release_old,
        "release_new": release_new,
        "observations_old": observations_old,
        "observations_new": observations_new,
        "prepared_path": sealed["prepared_path"],
        "prepared_sha": sealed["prepared_sha256"],
        "journal": journal,
    }


def _pointer_sha(root: Path) -> str:
    return _sha((root / "_latest.json").read_bytes())


def _authority_args(fixture: dict) -> dict:
    return {
        "market_pointer_path": fixture["market_pointer"],
        "expected_market_pointer_sha256": fixture["market_sha"],
        "pit_pointer_path": fixture["pit_pointer"],
        "expected_pit_pointer_sha256": fixture["pit_sha"],
    }


def _phases(journal: Path, run_id: str = "run-1") -> list[str]:
    return [
        json.loads(path.read_text())["phase"] for path in sorted((journal / run_id).glob("*.json"))
    ]


def _valid_postcheck(release, observations) -> None:
    assert _pointer_sha(release["canonical_root"]) == release["new_pointer_sha256"]
    assert _pointer_sha(observations["canonical_root"]) == observations["new_pointer_sha256"]


def test_commit_has_exact_phase_order_and_is_strictly_read_back(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(transaction, "_postcheck", _valid_postcheck)
    prepared_payload = json.loads(Path(fixture["prepared_path"]).read_text())
    assert prepared_payload["input_bindings"]["market_pointer_authority"] == {
        "path": str(fixture["market_pointer"]),
        "sha256": fixture["market_sha"],
    }
    assert prepared_payload["input_bindings"]["pit_pointer_authority"] == {
        "path": str(fixture["pit_pointer"]),
        "sha256": fixture["pit_sha"],
    }

    result = transaction.commit_prepared_macro_transaction(
        prepared_path=fixture["prepared_path"],
        expected_prepared_sha256=fixture["prepared_sha"],
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        **_authority_args(fixture),
    )

    assert result["status"] == "SUCCESS"
    assert _phases(fixture["journal"]) == list(transaction.PHASES)
    assert _pointer_sha(fixture["release"]) == fixture["release_new"]
    assert _pointer_sha(fixture["observations"]) == fixture["observations_new"]
    assert (fixture["release"] / "_generations" / "release-child").is_dir()
    assert (fixture["observations"] / "_generations" / "observations-child").is_dir()
    intent = json.loads(sorted((fixture["journal"] / "run-1").glob("*.json"))[0].read_text())
    assert intent["details"]["market_authority"]["pointer_path"] == str(fixture["market_pointer"])
    assert intent["details"]["market_authority"]["pointer_sha256"] == fixture["market_sha"]
    assert (
        base64.b64decode(intent["details"]["market_authority"]["pointer_bytes_b64"])
        == fixture["market_pointer"].read_bytes()
    )
    assert intent["details"]["pit_authority"]["pointer_sha256"] == fixture["pit_sha"]


def test_commit_acquires_market_pit_release_observations_lock_order(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(transaction, "_postcheck", _valid_postcheck)
    acquired: list[str] = []
    original = transaction._store_lock

    from contextlib import contextmanager

    @contextmanager
    def observed(root, filename, blocker):
        acquired.append(filename)
        with original(root, filename, blocker):
            yield

    monkeypatch.setattr(transaction, "_store_lock", observed)
    transaction.commit_prepared_macro_transaction(
        prepared_path=fixture["prepared_path"],
        expected_prepared_sha256=fixture["prepared_sha"],
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        **_authority_args(fixture),
    )
    assert acquired == [
        ".market_writer.lock",
        ".pit_writer.lock",
        ".release-calendar.lock",
        ".promotion.lock",
    ]


def test_commit_rejects_authority_argument_mismatch_before_journal(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    authority_args = _authority_args(fixture)
    authority_args["expected_market_pointer_sha256"] = "0" * 64
    with pytest.raises(
        transaction.MacroMaintenanceTransactionError,
        match="market_authority_argument_mismatch",
    ):
        transaction.commit_prepared_macro_transaction(
            prepared_path=fixture["prepared_path"],
            expected_prepared_sha256=fixture["prepared_sha"],
            journal_root=fixture["journal"],
            journal_run_id="run-1",
            **authority_args,
        )
    assert not (fixture["journal"] / "run-1").exists()


def test_candidate_shadow_authority_can_prepare_but_cannot_commit(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, authority_mode="candidate")
    with pytest.raises(
        transaction.MacroMaintenanceTransactionError,
        match="candidate_authority_not_executable",
    ):
        transaction.commit_prepared_macro_transaction(
            prepared_path=fixture["prepared_path"],
            expected_prepared_sha256=fixture["prepared_sha"],
            journal_root=fixture["journal"],
            journal_run_id="run-1",
            **_authority_args(fixture),
        )
    assert not (fixture["journal"] / "run-1").exists()


def test_authorities_are_revalidated_at_every_commit_boundary(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(transaction, "_postcheck", _valid_postcheck)
    checkpoints: list[str] = []
    original = transaction._revalidate_authorities

    def observed(authorities, *, checkpoint):
        checkpoints.append(checkpoint)
        return original(authorities, checkpoint=checkpoint)

    monkeypatch.setattr(transaction, "_revalidate_authorities", observed)
    transaction.commit_prepared_macro_transaction(
        prepared_path=fixture["prepared_path"],
        expected_prepared_sha256=fixture["prepared_sha"],
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        **_authority_args(fixture),
    )
    assert checkpoints == [
        "before_install",
        "after_install",
        "before_release_pointer_switch",
        "after_release_pointer_switch",
        "after_observations_pointer_switch",
        "before_postcheck",
        "after_postcheck",
    ]


def test_authority_drift_after_generation_install_blocks_both_macro_cas(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(transaction, "_postcheck", _valid_postcheck)

    with pytest.raises(RuntimeError, match="stop after install"):
        transaction.commit_prepared_macro_transaction(
            prepared_path=fixture["prepared_path"],
            expected_prepared_sha256=fixture["prepared_sha"],
            journal_root=fixture["journal"],
            journal_run_id="run-1",
            **_authority_args(fixture),
            failure_injector=lambda phase: (
                (_ for _ in ()).throw(RuntimeError("stop after install"))
                if phase == "BOTH_GENERATIONS_INSTALLED"
                else None
            ),
        )
    assert _pointer_sha(fixture["release"]) == fixture["release_old"]
    assert _pointer_sha(fixture["observations"]) == fixture["observations_old"]
    fixture["market_pointer"].write_bytes(
        _raw({"generation_id": "market-concurrent-drift", "status": "OK"})
    )

    classified = transaction.recover_macro_transaction(
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        **_authority_args(fixture),
    )
    assert classified["classification"] == "PROMOTION_UNCERTAIN"
    assert "market_authority_drift" in classified["blockers"][0]
    with pytest.raises(
        transaction.MacroMaintenanceTransactionError,
        match="not_deterministic",
    ):
        transaction.recover_macro_transaction(
            journal_root=fixture["journal"],
            journal_run_id="run-1",
            **_authority_args(fixture),
            execute_forward=True,
        )
    assert _pointer_sha(fixture["release"]) == fixture["release_old"]
    assert _pointer_sha(fixture["observations"]) == fixture["observations_old"]


@pytest.mark.parametrize(
    "phase",
    [
        "INTENT",
        "BOTH_GENERATIONS_PREPARED",
        "BOTH_GENERATIONS_INSTALLED",
        "RELEASE_POINTER_COMMITTED",
        "OBSERVATIONS_POINTER_COMMITTED",
        "POSTCHECK_PASSED",
    ],
)
def test_every_interruption_is_read_only_classified_and_forward_recoverable(
    tmp_path: Path, monkeypatch, phase: str
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(transaction, "_postcheck", _valid_postcheck)

    def interrupt(observed: str) -> None:
        if observed == phase:
            raise RuntimeError(f"interrupted:{phase}")

    with pytest.raises(RuntimeError, match="interrupted"):
        transaction.commit_prepared_macro_transaction(
            prepared_path=fixture["prepared_path"],
            expected_prepared_sha256=fixture["prepared_sha"],
            journal_root=fixture["journal"],
            journal_run_id="run-1",
            **_authority_args(fixture),
            failure_injector=interrupt,
        )

    before = (
        _pointer_sha(fixture["release"]),
        _pointer_sha(fixture["observations"]),
        tuple(_phases(fixture["journal"])),
    )
    classified = transaction.recover_macro_transaction(
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        **_authority_args(fixture),
    )
    after = (
        _pointer_sha(fixture["release"]),
        _pointer_sha(fixture["observations"]),
        tuple(_phases(fixture["journal"])),
    )
    assert classified["classification"] in {
        "CAN_EXECUTE_FORWARD",
        "CAN_FINALIZE",
    }
    assert before == after

    recovered = transaction.recover_macro_transaction(
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        **_authority_args(fixture),
        execute_forward=True,
    )
    assert recovered["status"] == "SUCCESS"
    assert _phases(fixture["journal"]) == list(transaction.PHASES)
    assert (
        transaction.recover_macro_transaction(
            journal_root=fixture["journal"],
            journal_run_id="run-1",
            **_authority_args(fixture),
        )["classification"]
        == "TERMINAL"
    )
    terminal_phases = _phases(fixture["journal"])
    assert (
        transaction.recover_macro_transaction(
            journal_root=fixture["journal"],
            journal_run_id="run-1",
            **_authority_args(fixture),
            execute_forward=True,
        )["classification"]
        == "TERMINAL"
    )
    assert _phases(fixture["journal"]) == terminal_phases


def test_recovery_restores_missing_journal_record_after_pointer_write(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(transaction, "_postcheck", _valid_postcheck)
    original = transaction._append_journal
    failed = False

    def fail_after_release_pointer(run, phase, **kwargs):
        nonlocal failed
        if phase == "RELEASE_POINTER_COMMITTED" and not failed:
            failed = True
            raise RuntimeError("journal interruption")
        return original(run, phase, **kwargs)

    monkeypatch.setattr(transaction, "_append_journal", fail_after_release_pointer)
    with pytest.raises(RuntimeError, match="journal interruption"):
        transaction.commit_prepared_macro_transaction(
            prepared_path=fixture["prepared_path"],
            expected_prepared_sha256=fixture["prepared_sha"],
            journal_root=fixture["journal"],
            journal_run_id="run-1",
            **_authority_args(fixture),
        )
    assert _pointer_sha(fixture["release"]) == fixture["release_new"]
    assert _pointer_sha(fixture["observations"]) == fixture["observations_old"]

    monkeypatch.setattr(transaction, "_append_journal", original)
    result = transaction.recover_macro_transaction(
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        **_authority_args(fixture),
        execute_forward=True,
    )
    assert result["status"] == "SUCCESS"
    assert _phases(fixture["journal"]) == list(transaction.PHASES)


def test_third_party_pointer_drift_is_uncertain_and_cannot_execute(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(transaction, "_postcheck", _valid_postcheck)
    with pytest.raises(RuntimeError):
        transaction.commit_prepared_macro_transaction(
            prepared_path=fixture["prepared_path"],
            expected_prepared_sha256=fixture["prepared_sha"],
            journal_root=fixture["journal"],
            journal_run_id="run-1",
            **_authority_args(fixture),
            failure_injector=lambda phase: (
                (_ for _ in ()).throw(RuntimeError("stop")) if phase == "INTENT" else None
            ),
        )
    (fixture["release"] / "_latest.json").write_bytes(
        _raw({"generation_id": "third-party", "status": "OK"})
    )
    classified = transaction.recover_macro_transaction(
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        **_authority_args(fixture),
    )
    assert classified["classification"] == "PROMOTION_UNCERTAIN"
    with pytest.raises(
        transaction.MacroMaintenanceTransactionError,
        match="not_deterministic",
    ) as caught:
        transaction.recover_macro_transaction(
            journal_root=fixture["journal"],
            journal_run_id="run-1",
            **_authority_args(fixture),
            execute_forward=True,
        )
    assert caught.value.status == "PROMOTION_UNCERTAIN"


def test_operator_rollback_requires_all_four_hashes_and_refuses_drift(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(transaction, "_postcheck", _valid_postcheck)
    transaction.commit_prepared_macro_transaction(
        prepared_path=fixture["prepared_path"],
        expected_prepared_sha256=fixture["prepared_sha"],
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        **_authority_args(fixture),
    )
    with pytest.raises(
        transaction.MacroMaintenanceTransactionError,
        match="identity_mismatch",
    ):
        transaction.rollback_macro_transaction(
            journal_root=fixture["journal"],
            journal_run_id="run-1",
            **_authority_args(fixture),
            old_release_pointer_sha256=fixture["release_old"],
            new_release_pointer_sha256="0" * 64,
            old_observations_pointer_sha256=fixture["observations_old"],
            new_observations_pointer_sha256=fixture["observations_new"],
        )

    rolled_back = transaction.rollback_macro_transaction(
        journal_root=fixture["journal"],
        journal_run_id="run-1",
        **_authority_args(fixture),
        old_release_pointer_sha256=fixture["release_old"],
        new_release_pointer_sha256=fixture["release_new"],
        old_observations_pointer_sha256=fixture["observations_old"],
        new_observations_pointer_sha256=fixture["observations_new"],
    )
    assert rolled_back["status"] == "ROLLED_BACK"
    assert _pointer_sha(fixture["release"]) == fixture["release_old"]
    assert _pointer_sha(fixture["observations"]) == fixture["observations_old"]


def test_seal_rejects_non_private_root_and_candidate_symlink(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "valid")
    assert fixture["prepared_path"]

    unsafe = _private(tmp_path / "unsafe")
    os.chmod(unsafe, 0o755)
    with pytest.raises(
        transaction.MacroMaintenanceTransactionError,
        match="not_private",
    ):
        transaction.seal_prepared_macro_transaction(
            prepared_root=unsafe,
            release_candidate_root=tmp_path,
            observations_candidate_root=tmp_path,
            release_canonical_root=fixture["release"],
            observations_canonical_root=fixture["observations"],
            expected_release_pointer_sha256=fixture["release_old"],
            expected_observations_pointer_sha256=fixture["observations_old"],
            market_pointer_path=fixture["market_pointer"],
            expected_market_pointer_sha256=fixture["market_sha"],
            pit_pointer_path=fixture["pit_pointer"],
            expected_pit_pointer_sha256=fixture["pit_sha"],
            authority_mode="canonical",
            target_date="20260819",
        )

    candidate_target = _private(tmp_path / "candidate-target")
    candidate_link = tmp_path / "candidate-link"
    candidate_link.symlink_to(candidate_target, target_is_directory=True)
    second_prepared = _private(tmp_path / "second-prepared")
    with pytest.raises(
        transaction.MacroMaintenanceTransactionError,
        match="candidate_unsafe",
    ):
        transaction.seal_prepared_macro_transaction(
            prepared_root=second_prepared,
            release_candidate_root=candidate_link,
            observations_candidate_root=candidate_target,
            release_canonical_root=fixture["release"],
            observations_canonical_root=fixture["observations"],
            expected_release_pointer_sha256=fixture["release_old"],
            expected_observations_pointer_sha256=fixture["observations_old"],
            market_pointer_path=fixture["market_pointer"],
            expected_market_pointer_sha256=fixture["market_sha"],
            pit_pointer_path=fixture["pit_pointer"],
            expected_pit_pointer_sha256=fixture["pit_sha"],
            authority_mode="canonical",
            target_date="20260819",
        )


def test_component_prepare_writes_only_private_candidate_roots(tmp_path: Path, monkeypatch) -> None:
    market_root = _private(tmp_path / "market")
    market_pointer = market_root / "_latest.json"
    market_pointer.write_bytes(_raw({"generation_id": "market-active"}))
    pit_root = _private(tmp_path / "pit")
    pit_pointer = pit_root / "stock_basic_membership_latest.json"
    pit_pointer.write_bytes(_raw({"generation_id": "pit-active"}))
    release = _private(tmp_path / "release")
    observations = _private(tmp_path / "observations")
    release_before, release_sha = _store(release, "release-parent")
    observations_before, observations_sha = _store(observations, "observations-parent")
    inputs = []
    for name in ("snapshot.json", "coverage.json", "scope.json"):
        path = tmp_path / name
        path.write_text(name)
        inputs.append((path, _sha(path.read_bytes())))
    private_runs = _private(tmp_path / "private-runs")

    def fake_legacy(**kwargs):
        release_candidate = Path(kwargs["release_root"])
        observations_candidate = Path(kwargs["observations_root"])
        assert release_candidate != release
        assert observations_candidate != observations
        _store(release_candidate, kwargs["release_run_id"])
        _store(observations_candidate, kwargs["observations_run_id"])
        return {"status": "OK", "provider_calls": [{"issuer": "fixture"}]}

    monkeypatch.setattr(maintenance, "run_cn_macro_maintenance", fake_legacy)
    result = maintenance.prepare_cn_macro_maintenance_transaction(
        market="CN",
        target_date="20260819",
        snapshot_manifest_path=inputs[0][0],
        expected_snapshot_manifest_sha256=inputs[0][1],
        coverage_manifest_path=inputs[1][0],
        expected_coverage_manifest_sha256=inputs[1][1],
        scope_artifact_path=inputs[2][0],
        expected_scope_artifact_sha256=inputs[2][1],
        release_root=release,
        expected_release_pointer_sha256=release_sha,
        observations_root=observations,
        expected_observations_pointer_sha256=observations_sha,
        market_pointer_path=market_pointer,
        expected_market_pointer_sha256=_sha(market_pointer.read_bytes()),
        pit_pointer_path=pit_pointer,
        expected_pit_pointer_sha256=_sha(pit_pointer.read_bytes()),
        authority_mode="canonical",
        release_run_id="release-child",
        observations_run_id="observations-child",
        private_run_root=private_runs,
        transaction_run_id="transaction-1",
        allow_live=True,
    )
    assert result["status"] == "PREPARED"
    assert result["canonical_writes"] == []
    assert (release / "_latest.json").read_bytes() == release_before
    assert (observations / "_latest.json").read_bytes() == observations_before
    assert not (release / "_generations" / "release-child").exists()
    assert not (observations / "_generations" / "observations-child").exists()
