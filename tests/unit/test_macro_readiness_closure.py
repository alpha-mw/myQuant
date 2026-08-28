from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.contracts import canonical_json_bytes
import quant_investor.macro.readiness_closure as closure


def _write(path: Path, value: dict | bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    for parent in [path.parent, *path.parent.parents]:
        if parent == parent.parent:
            break
        if parent.exists():
            parent.chmod(0o700)
        if parent.name.startswith("pytest-"):
            break
    raw = value if isinstance(value, bytes) else canonical_json_bytes(value)
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def _fixture(tmp_path: Path, monkeypatch) -> tuple[dict, str, str]:
    root = tmp_path
    target = "20260827"
    transaction_id = "macro-test-v1"
    transaction = root / closure.TRANSACTION_ROOT / transaction_id
    prepared_dir = transaction / "_prepared" / transaction_id / "prepared"
    journal_dir = transaction / "journals" / transaction_id
    market_pointer = {
        "snapshot_id": "snapshot-test",
        "latest_complete_trade_date": target,
    }
    pit_pointer = {"generation_id": f"pit-{target}-test"}
    release_pointer = {"generation_id": "release-test"}
    observations_pointer = {
        "generation_id": "observations-test",
        "metadata": {
            "as_of": target,
            "local_target_trade_date": target,
            "latest_local_trade_date": target,
        },
    }
    pointer_rows = {
        "market": (closure.MARKET_POINTER, market_pointer),
        "pit": (closure.PIT_POINTER, pit_pointer),
        "release": (closure.RELEASE_POINTER, release_pointer),
        "observations": (closure.OBSERVATIONS_POINTER, observations_pointer),
    }
    pointer_shas = {
        name: _write(root / relative, value) for name, (relative, value) in pointer_rows.items()
    }
    artifacts = prepared_dir / "artifacts"
    artifact_names = {
        "market": "market_authority_pointer.json",
        "pit": "pit_authority_pointer.json",
        "release": "release_new_pointer.json",
        "observations": "observations_new_pointer.json",
    }
    for name, filename in artifact_names.items():
        _write(artifacts / filename, pointer_rows[name][1])
    release_generation = root / closure.RELEASE_POINTER.parent / "_generations" / "release-test"
    observations_generation = (
        root / closure.OBSERVATIONS_POINTER.parent / "_generations" / "observations-test"
    )
    _write(release_generation / "manifest.json", {"status": "OK"})
    _write(
        release_generation / "market_open_days.json",
        {"market": "CN", "open_dates": [target]},
    )
    _write(observations_generation / "manifest.json", {"status": "OK"})
    source = root / "data/private/source.json"
    source_sha = _write(source, {"source": "sealed"})
    release_tree = "a" * 64
    observations_tree = "b" * 64
    prepared = {
        "schema_version": "cn-macro-dual-pointer-prepared.v1",
        "target_date": target,
        "prepared_at": "2026-08-27T12:19:59+00:00",
        "authority_mode": "canonical",
        "release": {
            "canonical_root": str(root / closure.RELEASE_POINTER.parent),
            "candidate_root": str(prepared_dir.parent / "release_candidate"),
            "generation_id": "release-test",
            "generation_tree_sha256": release_tree,
            "old_pointer_sha256": "c" * 64,
            "new_pointer_sha256": pointer_shas["release"],
            "old_pointer_artifact": "artifacts/release_old_pointer.json",
            "new_pointer_artifact": f"artifacts/{artifact_names['release']}",
        },
        "observations": {
            "canonical_root": str(root / closure.OBSERVATIONS_POINTER.parent),
            "candidate_root": str(prepared_dir.parent / "observations_candidate"),
            "generation_id": "observations-test",
            "generation_tree_sha256": observations_tree,
            "old_pointer_sha256": "d" * 64,
            "new_pointer_sha256": pointer_shas["observations"],
            "old_pointer_artifact": "artifacts/observations_old_pointer.json",
            "new_pointer_artifact": f"artifacts/{artifact_names['observations']}",
            "dependency_generations": [],
        },
        "input_bindings": {
            "source": {"path": str(source), "sha256": source_sha},
            "market_pointer_authority": {
                "path": str(root / closure.MARKET_POINTER),
                "sha256": pointer_shas["market"],
            },
            "pit_pointer_authority": {
                "path": str(root / closure.PIT_POINTER),
                "sha256": pointer_shas["pit"],
            },
        },
        "authorities": {
            "market": {
                "pointer_path": str(root / closure.MARKET_POINTER),
                "pointer_sha256": pointer_shas["market"],
                "pointer_artifact": f"artifacts/{artifact_names['market']}",
            },
            "pit": {
                "pointer_path": str(root / closure.PIT_POINTER),
                "pointer_sha256": pointer_shas["pit"],
                "pointer_artifact": f"artifacts/{artifact_names['pit']}",
            },
        },
    }
    prepared_path = prepared_dir / "prepared.json"
    prepared_sha = _write(prepared_path, prepared)
    recorded = datetime(2026, 8, 27, 12, 20, tzinfo=timezone.utc)
    terminal_path = ""
    terminal_sha = ""
    for index, phase in enumerate(closure.PHASES, start=1):
        document = {
            "schema_version": closure.JOURNAL_SCHEMA,
            "sequence": index,
            "phase": phase,
            "recorded_at": recorded.isoformat(),
            "prepared_path": str(prepared_path),
            "prepared_sha256": prepared_sha,
            "details": {"status": "SUCCESS"} if phase == "TERMINAL" else {},
        }
        path = journal_dir / f"{index:04d}-{phase.lower()}.json"
        terminal_sha = _write(path, document)
        terminal_path = path.relative_to(root).as_posix()

    def fake_tree(path):
        return release_tree if Path(path).name == "release-test" else observations_tree

    monkeypatch.setattr(closure, "generation_tree_sha256", fake_tree)
    monkeypatch.setattr(closure, "load_release_calendar", lambda **_kwargs: object())
    monkeypatch.setattr(
        closure,
        "load_observations",
        lambda _root: (
            [{"row": 1}],
            {
                "pointer_sha256": pointer_shas["observations"],
                "generation_id": "observations-test",
            },
        ),
    )
    return pointer_shas, terminal_path, terminal_sha


def test_macro_readiness_closure_is_deterministic_and_time_bound(
    tmp_path: Path, monkeypatch
) -> None:
    pointers, terminal_path, terminal_sha = _fixture(tmp_path, monkeypatch)
    built = closure.build_macro_readiness_closure(
        workspace_root=tmp_path,
        terminal_path=terminal_path,
        terminal_sha256=terminal_sha,
    )
    assert built["available_at"] == "2026-08-27T12:20:00.000000Z"
    assert built["veto_lifecycle"]["state"] == "NOT_PRESENT"
    first = closure.seal_macro_readiness_closure(
        workspace_root=tmp_path,
        terminal_path=terminal_path,
        terminal_sha256=terminal_sha,
    )
    second = closure.seal_macro_readiness_closure(
        workspace_root=tmp_path,
        terminal_path=terminal_path,
        terminal_sha256=terminal_sha,
    )
    assert first["status"] == "SEALED"
    assert second["status"] == "NO_ACTION"
    assert first["closure_path"] == second["closure_path"]
    assert first["closure_sha256"] == second["closure_sha256"]
    assert (
        closure.validate_macro_readiness_closure(
            workspace_root=tmp_path,
            closure=first["closure"],
        )
        == first["closure"]
    )
    current = closure.verify_current_macro_readiness_closure(
        workspace_root=tmp_path,
        closure=first["closure"],
        expected_target_date="20260827",
        decision_as_of="2026-08-27T15:00:00Z",
    )
    assert current["pointer_sha256"]["market"] == pointers["market"]


def test_macro_readiness_rejects_late_evidence_and_current_drift(
    tmp_path: Path, monkeypatch
) -> None:
    _pointers, terminal_path, terminal_sha = _fixture(tmp_path, monkeypatch)
    built = closure.build_macro_readiness_closure(
        workspace_root=tmp_path,
        terminal_path=terminal_path,
        terminal_sha256=terminal_sha,
    )
    with pytest.raises(
        closure.MacroReadinessClosureError,
        match="NOT_AVAILABLE_AT_DECISION",
    ):
        closure.verify_current_macro_readiness_closure(
            workspace_root=tmp_path,
            closure=built,
            expected_target_date="20260827",
            decision_as_of="2026-08-27T12:19:59Z",
        )
    _write(tmp_path / closure.MARKET_POINTER, {"snapshot_id": "drift"})
    assert (
        closure.validate_macro_readiness_closure(
            workspace_root=tmp_path,
            closure=built,
        )
        == built
    )
    with pytest.raises(closure.MacroReadinessClosureError, match="CURRENT_POINTER"):
        closure.verify_current_macro_readiness_closure(
            workspace_root=tmp_path,
            closure=built,
            expected_target_date="20260827",
            decision_as_of="2026-08-27T15:00:00Z",
        )


def test_macro_readiness_intrinsic_survives_new_live_veto_but_current_fails(
    tmp_path: Path, monkeypatch
) -> None:
    _pointers, terminal_path, terminal_sha = _fixture(tmp_path, monkeypatch)
    built = closure.build_macro_readiness_closure(
        workspace_root=tmp_path,
        terminal_path=terminal_path,
        terminal_sha256=terminal_sha,
    )
    _write(tmp_path / closure.VETO_PATH, {"blockers": ["NEW_VETO"]})
    assert (
        closure.validate_macro_readiness_closure(
            workspace_root=tmp_path,
            closure=built,
        )
        == built
    )
    with pytest.raises(closure.MacroReadinessClosureError, match="LIVE_VETO_PRESENT"):
        closure.verify_current_macro_readiness_closure(
            workspace_root=tmp_path,
            closure=built,
            expected_target_date="20260827",
            decision_as_of="2026-08-27T15:00:00Z",
        )


def test_macro_readiness_requires_exact_seven_record_journal(tmp_path: Path, monkeypatch) -> None:
    _pointers, terminal_path, terminal_sha = _fixture(tmp_path, monkeypatch)
    terminal = tmp_path / terminal_path
    extra = terminal.parent / "unexpected.txt"
    _write(extra, b"unexpected")
    with pytest.raises(closure.MacroReadinessClosureError, match="JOURNAL_SHAPE"):
        closure.build_macro_readiness_closure(
            workspace_root=tmp_path,
            terminal_path=terminal_path,
            terminal_sha256=terminal_sha,
        )
