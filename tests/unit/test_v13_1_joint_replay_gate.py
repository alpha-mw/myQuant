from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from quant_investor.governance import replay_v13_1 as replay_module
from quant_investor.governance.replay_v13_1 import (
    ACCEPTANCE_EVIDENCE_SCHEMA_VERSION,
    FREEZE_EXCEPTION_CYCLE_ID,
    REQUIRED_REPLAY_SCENARIOS,
    REQUIRED_SCENARIO_CHECKS,
    SCENARIO_EVIDENCE_SCHEMA_VERSION,
    SHADOW_EVIDENCE_SCHEMA_VERSION,
    _payload_sha256,
    _sha256,
    build_joint_replay_manifest,
    build_replay_split,
    build_threshold_seal,
    validate_threshold_seal,
    verify_joint_replay_manifest,
    write_manifest_atomic,
)


DATASET_SHA = "a" * 64
PROTOCOL_HASHES = {
    "theme_v2": "b" * 64,
    "factor_v2": "c" * 64,
    "dashboard_contract_v2": "d" * 64,
}
THRESHOLDS = {
    "attention": 0.6,
    "scenario_metrics": {
        "*": {"net_excess_return": {"min": 0.0}},
    },
}


def _load_script_module(name: str, filename: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _seal(*, validation_end_date: str = "2026-01-21") -> dict:
    payload = build_threshold_seal(
        thresholds=THRESHOLDS,
        dataset_sha256=DATASET_SHA,
        validation_end_date=validation_end_date,
    ).to_dict()
    payload["_artifact_readback_verified"] = True
    payload["_artifact_sha256"] = "e" * 64
    payload["_canonical_seal_ledger_verified"] = True
    payload["_seal_ledger_sha256"] = "6" * 64
    return payload


def _verified_payload(payload: dict, *, artifact_sha: str) -> dict:
    payload = dict(payload)
    payload["evidence_sha256"] = _payload_sha256(payload)
    payload["_artifact_readback_verified"] = True
    payload["_artifact_sha256"] = artifact_sha
    return payload


def _acceptance() -> dict:
    return _verified_payload(
        {
            "schema_version": ACCEPTANCE_EVIDENCE_SCHEMA_VERSION,
            "dataset_sha256": DATASET_SHA,
            "dashboard": {
                "p0_clear": True,
                "attribution_reconciled": True,
                "private_data_boundary_pass": True,
            },
            "theme": {
                "coverage_pass": True,
                "pit_pass": True,
                "forced_admission_removed": True,
                "forced_theme_count": 0,
                "rollback_pass": True,
            },
            "factor": {
                "targeted_transition_pass": True,
                "idempotent_readback_pass": True,
                "rollback_pass": True,
            },
        },
        artifact_sha="f" * 64,
    )


def _scenarios(dates: list[str]) -> dict:
    split = build_replay_split(dates)
    split_sha = _sha256(split.to_dict())
    dates_sha = _sha256(sorted(set(dates)))
    return {
        name: _verified_payload(
            {
                "schema_version": SCENARIO_EVIDENCE_SCHEMA_VERSION,
                "scenario": name,
                "dataset_sha256": DATASET_SHA,
                "split_sha256": split_sha,
                "trade_dates_sha256": dates_sha,
                "source_snapshot_sha256": "1" * 64,
                "protocol_hashes": PROTOCOL_HASHES,
                "checks": {key: True for key in REQUIRED_SCENARIO_CHECKS},
                "metric_checks": {"acceptance_thresholds": True},
                "metrics": {"net_excess_return": 0.01},
            },
            artifact_sha=(f"{index:x}" * 64)[:64],
        )
        for index, name in enumerate(REQUIRED_REPLAY_SCENARIOS, start=2)
    }


def _shadow_evidence(dates: list[str]) -> list[dict]:
    return [
        _verified_payload(
            {
                "schema_version": SHADOW_EVIDENCE_SCHEMA_VERSION,
                "trade_date": trade_date,
                "dataset_sha256": DATASET_SHA,
                "protocol_hash": PROTOCOL_HASHES["theme_v2"],
                "snapshot_sha256": "9" * 64,
                "pit_verified": True,
            },
            artifact_sha="8" * 64,
        )
        for trade_date in dates
    ]


def test_replay_split_is_chronological_and_deduplicates_same_day() -> None:
    dates = [f"2026-01-{day:02d}" for day in range(1, 11)] + ["2026-01-05"]
    split = build_replay_split(reversed(dates))

    assert len(split.train) == 6
    assert len(split.validation) == 2
    assert len(split.holdout) == 2
    assert split.train[-1] < split.validation[0] < split.holdout[0]


def test_open_holdout_rejects_threshold_or_dataset_drift() -> None:
    seal = build_threshold_seal(
        thresholds={"attention": 0.6},
        dataset_sha256=DATASET_SHA,
        validation_end_date="2026-01-08",
    )

    assert validate_threshold_seal(
        seal,
        current_thresholds={"attention": 0.6},
        dataset_sha256=DATASET_SHA,
        holdout_opened=True,
        expected_threshold_hash=seal.threshold_hash,
    ) == []
    wrong_cycle = seal.to_dict()
    wrong_cycle["freeze_exception_cycle_id"] = "other-cycle"
    assert "holdout_freeze_exception_cycle_mismatch" in validate_threshold_seal(
        wrong_cycle,
        current_thresholds={"attention": 0.6},
        dataset_sha256=DATASET_SHA,
        holdout_opened=True,
        expected_threshold_hash=seal.threshold_hash,
    )
    blockers = validate_threshold_seal(
        seal,
        current_thresholds={"attention": 0.61},
        dataset_sha256="b" * 64,
        holdout_opened=True,
        expected_threshold_hash=seal.threshold_hash,
    )
    assert "holdout_thresholds_changed_after_seal" in blockers
    assert "holdout_dataset_hash_changed_after_seal" in blockers


def test_threshold_seal_uses_one_canonical_path_and_verified_ledger(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seal_script = _load_script_module(
        "seal_v13_1_holdout_thresholds_test",
        "seal_v13_1_holdout_thresholds.py",
    )
    gate_script = _load_script_module(
        "run_v13_1_joint_replay_gate_test",
        "run_v13_1_joint_replay_gate.py",
    )
    seal_root = tmp_path / "private" / "replay" / "threshold_seals"
    seal_ledger = seal_root / "seal_ledger.jsonl"
    monkeypatch.setattr(seal_script, "CANONICAL_SEAL_ROOT", seal_root)
    monkeypatch.setattr(seal_script, "CANONICAL_SEAL_LEDGER", seal_ledger)
    monkeypatch.setattr(gate_script, "CANONICAL_SEAL_ROOT", seal_root)
    monkeypatch.setattr(gate_script, "CANONICAL_SEAL_LEDGER", seal_ledger)
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(json.dumps(THRESHOLDS), encoding="utf-8")

    assert seal_script.main(
        [
            "--thresholds-json",
            str(thresholds_path),
            "--dataset-sha256",
            DATASET_SHA,
            "--validation-end-date",
            "2026-01-21",
        ]
    ) == 0
    seal_path = seal_root / f"{DATASET_SHA}.json"
    cycle_lock = seal_root / ".freeze_exception_cycle.sealed.lock"
    seal_sha = hashlib.sha256(seal_path.read_bytes()).hexdigest()
    ledger_sha = hashlib.sha256(seal_ledger.read_bytes()).hexdigest()
    assert seal_path.stat().st_mode & 0o777 == 0o600
    assert seal_ledger.stat().st_mode & 0o777 == 0o600
    assert cycle_lock.stat().st_mode & 0o777 == 0o600
    assert json.loads(cycle_lock.read_text(encoding="utf-8"))[
        "dataset_sha256"
    ] == DATASET_SHA
    assert json.loads(seal_path.read_text(encoding="utf-8"))[
        "freeze_exception_cycle_id"
    ] == FREEZE_EXCEPTION_CYCLE_ID

    verified = gate_script._read_canonical_threshold_seal(
        dataset_sha256=DATASET_SHA,
        expected_seal_sha256=seal_sha,
        expected_ledger_sha256=ledger_sha,
    )
    assert verified["_canonical_seal_ledger_verified"] is True
    assert verified["_seal_ledger_sha256"] == ledger_sha
    with pytest.raises(FileExistsError):
        seal_script.main(
            [
                "--thresholds-json",
                str(thresholds_path),
                "--dataset-sha256",
                DATASET_SHA,
                "--validation-end-date",
                "2026-01-21",
            ]
        )
    with pytest.raises(FileExistsError, match="new dataset"):
        seal_script.main(
            [
                "--thresholds-json",
                str(thresholds_path),
                "--dataset-sha256",
                "b" * 64,
                "--validation-end-date",
                "2026-01-21",
            ]
        )


def test_private_manifest_writer_is_0600_hash_bound_and_readback_verified(
    tmp_path: Path,
) -> None:
    target = tmp_path / "private" / "joint_manifest.json"
    write_manifest_atomic(target, {"schema_version": "test.v1", "status": "blocked"})

    assert target.stat().st_mode & 0o777 == 0o600
    payload = json.loads(target.read_text(encoding="utf-8"))
    manifest_hash = payload.pop("manifest_sha256")
    assert manifest_hash == _sha256(payload)


def test_joint_gate_cli_requires_persisted_manifest_output() -> None:
    gate_script = _load_script_module(
        "run_v13_1_joint_replay_gate_parser_test",
        "run_v13_1_joint_replay_gate.py",
    )
    output_action = next(
        action
        for action in gate_script.build_parser()._actions
        if action.dest == "output"
    )

    assert output_action.required is True


def test_manifest_requires_verified_scenarios_and_shadow_artifacts() -> None:
    dates = [f"2026-01-{day:02d}" for day in range(1, 29)]
    scenarios = _scenarios(dates)
    scenarios.pop("theme_v2_formal_gate")
    seal = _seal()

    manifest = build_joint_replay_manifest(
        run_id="unit",
        trade_dates=dates,
        dataset_sha256=DATASET_SHA,
        protocol_hashes=PROTOCOL_HASHES,
        scenario_results=scenarios,
        theme_shadow_dates=_shadow_evidence(dates[:19]),
        threshold_seal=seal,
        current_thresholds=THRESHOLDS,
        acceptance=_acceptance(),
        holdout_opened=True,
        expected_threshold_hash=seal["threshold_hash"],
    )

    assert manifest["status"] == "blocked"
    assert manifest["theme_live_shadow"]["distinct_trade_day_count"] == 19
    assert any(
        item.startswith("theme_v2_formal_gate:scenario_evidence_schema_invalid")
        for item in manifest["blockers"]
    )
    assert manifest["activation"]["theme_formal"]["enabled"] is False
    assert manifest["activation"]["joint_path"]["enabled"] is False


def test_self_declared_passed_json_and_closed_holdout_cannot_activate() -> None:
    dates = [f"2026-01-{day:02d}" for day in range(1, 29)]
    seal = _seal()
    manifest = build_joint_replay_manifest(
        run_id="forged",
        trade_dates=dates,
        dataset_sha256=DATASET_SHA,
        protocol_hashes=PROTOCOL_HASHES,
        scenario_results={
            name: {"passed": True, "dataset_sha256": DATASET_SHA}
            for name in REQUIRED_REPLAY_SCENARIOS
        },
        theme_shadow_dates=dates[:20],
        threshold_seal=seal,
        current_thresholds=THRESHOLDS,
        acceptance={"dashboard": {}, "theme": {}, "factor": {}},
        holdout_opened=False,
    )

    assert manifest["status"] == "blocked"
    assert "holdout_not_opened" in manifest["blockers"]
    assert all(
        decision["enabled"] is False
        for decision in manifest["activation"].values()
    )


def test_metric_pass_is_recomputed_from_sealed_thresholds() -> None:
    dates = [f"2026-01-{day:02d}" for day in range(1, 29)]
    seal = _seal()
    scenarios = _scenarios(dates)
    forged = dict(scenarios["theme_v2_formal_gate"])
    forged.pop("evidence_sha256")
    forged["passed"] = True
    forged["metric_checks"] = {"acceptance_thresholds": True}
    forged["metrics"] = {"net_excess_return": -0.01}
    scenarios["theme_v2_formal_gate"] = _verified_payload(
        {
            key: value
            for key, value in forged.items()
            if not key.startswith("_artifact_")
        },
        artifact_sha="7" * 64,
    )

    manifest = build_joint_replay_manifest(
        run_id="metric-recompute",
        trade_dates=dates,
        dataset_sha256=DATASET_SHA,
        protocol_hashes=PROTOCOL_HASHES,
        scenario_results=scenarios,
        theme_shadow_dates=_shadow_evidence(dates[:20]),
        threshold_seal=seal,
        current_thresholds=THRESHOLDS,
        acceptance=_acceptance(),
        holdout_opened=True,
        expected_threshold_hash=seal["threshold_hash"],
    )

    assert manifest["status"] == "blocked"
    assert any(
        item
        == "theme_v2_formal_gate:scenario_metric_threshold_failed:net_excess_return"
        for item in manifest["blockers"]
    )
    assert manifest["activation"]["theme_formal"]["enabled"] is False


def test_verified_attestations_still_cannot_replace_real_joint_replay_producer() -> None:
    dates = [f"2026-01-{day:02d}" for day in range(1, 29)]
    seal = _seal()
    scenarios = _scenarios(dates)
    scenarios["industry_baseline"]["_artifact_path"] = "/private/source.json"
    shadow_evidence = _shadow_evidence(dates[:20])
    shadow_evidence[0]["_artifact_path"] = "/private/shadow.json"
    manifest = build_joint_replay_manifest(
        run_id="unit",
        trade_dates=dates,
        dataset_sha256=DATASET_SHA,
        protocol_hashes=PROTOCOL_HASHES,
        scenario_results=scenarios,
        theme_shadow_dates=shadow_evidence,
        threshold_seal=seal,
        current_thresholds=THRESHOLDS,
        acceptance=_acceptance(),
        holdout_opened=True,
        expected_threshold_hash=seal["threshold_hash"],
    )

    assert manifest["status"] == "blocked"
    assert "canonical_joint_replay_producer_not_implemented" in manifest[
        "blockers"
    ]
    assert manifest["activation"]["dashboard"]["enabled"] is True
    assert manifest["activation"]["theme_formal"]["enabled"] is False
    assert manifest["activation"]["factor_transitions"]["enabled"] is False
    assert manifest["activation"]["joint_path"]["enabled"] is False
    assert manifest["controls"]["registry_mutation"] is False
    assert "/private/" not in json.dumps(manifest)


def test_canonical_manifest_verifier_recomputes_complete_ready_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        replay_module,
        "CANONICAL_JOINT_REPLAY_PRODUCER_AVAILABLE",
        True,
    )
    dates = [f"2026-01-{day:02d}" for day in range(1, 29)]
    seal = _seal()
    manifest = build_joint_replay_manifest(
        run_id="verified-runtime",
        trade_dates=dates,
        dataset_sha256=DATASET_SHA,
        protocol_hashes=PROTOCOL_HASHES,
        scenario_results=_scenarios(dates),
        theme_shadow_dates=_shadow_evidence(dates[:20]),
        threshold_seal=seal,
        current_thresholds=THRESHOLDS,
        acceptance=_acceptance(),
        holdout_opened=True,
        expected_threshold_hash=seal["threshold_hash"],
        generated_at="2026-07-12T12:00:00+08:00",
    )
    assert manifest["status"] == "ready"
    target = tmp_path / "private" / "joint_manifest.json"
    write_manifest_atomic(target, manifest)
    artifact_sha = hashlib.sha256(target.read_bytes()).hexdigest()

    verified = verify_joint_replay_manifest(
        target,
        expected_artifact_sha256=artifact_sha,
        expected_theme_protocol_hash=PROTOCOL_HASHES["theme_v2"],
    )
    mismatch = verify_joint_replay_manifest(
        target,
        expected_artifact_sha256=artifact_sha,
        expected_theme_protocol_hash="f" * 64,
    )

    assert verified["ready"] is True
    assert verified["blockers"] == []
    assert verified["readback_verified"] is True
    assert mismatch["ready"] is False
    assert "joint_manifest_protocol_hash_mismatch" in mismatch["blockers"]


def test_canonical_manifest_verifier_rejects_subset_ready_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        replay_module,
        "CANONICAL_JOINT_REPLAY_PRODUCER_AVAILABLE",
        True,
    )
    target = tmp_path / "private" / "subset.json"
    write_manifest_atomic(
        target,
        {
            "schema_version": "myquant.joint_replay_gate.v1",
            "status": "ready",
            "blockers": [],
            "dataset_sha256": DATASET_SHA,
            "protocol_hashes": PROTOCOL_HASHES,
        },
    )
    artifact_sha = hashlib.sha256(target.read_bytes()).hexdigest()

    verified = verify_joint_replay_manifest(
        target,
        expected_artifact_sha256=artifact_sha,
        expected_theme_protocol_hash=PROTOCOL_HASHES["theme_v2"],
    )

    assert verified["ready"] is False
    assert "joint_manifest_scenario_set_invalid" in verified["blockers"]
    assert "joint_manifest_activation_recompute_mismatch" in verified["blockers"]
