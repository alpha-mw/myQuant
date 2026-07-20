from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import build_factor_v4_3_candidate_preregistration as subject


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path.joinpath(*subject.bundle_v4_3.ROOT_SUFFIX_V4_3)
    root.mkdir(parents=True)
    root.chmod(0o700)
    return root


def _stable(path: Path, content: bytes = b"stable\n") -> subject.StableFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return subject._stable_file(path, label=path.name)


def _entry(tmp_path: Path) -> subject.PublicationInputs:
    return subject.PublicationInputs(
        aquant_git_objects={"A_quant/source.py": b"source\n"},
        strict_source_binding={"artifact_semantic_sha256": "1" * 64},
        code_bindings=(
            {
                "relative_path": subject.CODE_BINDING_PATHS[0],
                "absolute_path": str(tmp_path / subject.CODE_BINDING_PATHS[0]),
                "byte_sha256": "2" * 64,
                "size_bytes": 1,
                "mode": 0o644,
                "uid": os.getuid(),
                "nlink": 1,
            },
        ),
        protected_bindings=(
            {
                "name": "registry",
                "absolute_path": str(tmp_path / "registry.json"),
                "byte_sha256": "3" * 64,
                "size_bytes": 1,
                "mode": 0o644,
                "uid": os.getuid(),
                "nlink": 1,
            },
        ),
        runtime_fingerprint={"fixed": True},
    )


def _published(root: Path, *, report_phase: str = "PRECOMMIT_INTENT_ONLY") -> dict[str, Any]:
    bundle = root / subject.FIXED_CYCLE_ID
    report_name = subject.bundle_v4_3.READBACK_REPORT_FILENAME_V4_3
    return {
        "accepted": True,
        "bundle_path": str(bundle),
        "publication_phase": "COMMITTED",
        "exclusive_rename_completed": True,
        "durability_commit_verified": True,
        "publication_authority": True,
        "artifact_descriptors": {
            report_name: {
                "absolute_path": str(bundle / report_name),
                "byte_sha256": "4" * 64,
                "size_bytes": 100,
                "mode": 0o600,
                "uid": os.getuid(),
                "nlink": 1,
            }
        },
        "readback_report": {
            "publication_phase": report_phase,
            "exclusive_rename_completed": False,
            "durability_commit_verified": False,
            "publication_authority": False,
            "side_effects": subject.prereg_v4_3.SIDE_EFFECT_FLAGS_V4_3,
            "artifact_semantic_sha256": "5" * 64,
        },
    }


def test_cli_surface_is_exact_and_has_no_override_or_side_effect_flags() -> None:
    parser = subject.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert tuple(subparsers.choices) == ("publish", "readback")
    publish = subparsers.choices["publish"]
    readback = subparsers.choices["readback"]
    assert [action.dest for action in publish._actions] == ["help"]
    assert [action.dest for action in readback._actions] == [
        "help",
        "bundle_path",
        "expected_readback_report_byte_sha256",
        "expected_readback_report_semantic_sha256",
    ]
    help_text = parser.format_help() + publish.format_help() + readback.format_help()
    assert all(token not in help_text for token in subject._FORBIDDEN_ARGUMENT_TOKENS)
    with pytest.raises(SystemExit):
        parser.parse_args(["publish", "--snapshot", "replacement"])


def test_real_bundle_public_entrypoints_have_the_required_signatures() -> None:
    publish = subject.bundle_v4_3.publish_candidate_preregistration_bundle_v4_3
    readback = subject.bundle_v4_3.readback_candidate_preregistration_bundle_v4_3
    assert callable(publish)
    assert callable(readback)
    assert tuple(inspect.signature(publish).parameters)[:8] == (
        "private_root",
        "repository_root",
        "preregistered_at",
        "aquant_git_objects",
        "strict_source_binding",
        "code_bindings",
        "protected_bindings",
        "revalidate_inputs",
    )
    assert tuple(inspect.signature(readback).parameters) == (
        "bundle_path",
        "expected_readback_report_byte_sha256",
        "expected_readback_report_semantic_sha256",
    )


def test_fixed_root_cycle_candidates_sources_and_strict_oracles() -> None:
    assert str(subject.PRODUCTION_PRIVATE_ROOT) == (
        "/Users/maxwell/mySpace/myQuant/reports/factor_governance/private/"
        "v4_3_candidate_preregistration"
    )
    assert subject.FIXED_CYCLE_ID == (
        "cn_full_a_v4_3_20260717_20260717T172132Z"
    )
    assert tuple(subject.prereg_v4_3.EXPECTED_CANDIDATES_V4_3) == (
        "event_guidance_revision_90d",
        "event_earnings_drift_60d",
        "fund_roe_delta_annual",
        "pv_small_float_cap",
        "value_book_to_price",
        "industry_relative_momentum_20d",
    )
    assert subject.prereg_v4_3.AQUANT_GIT_TOP == "/Users/maxwell/mySpace"
    assert subject.prereg_v4_3.AQUANT_COMMIT_V4_3 == (
        "4424dcecc384f614b0e9fd5e36cf094e9244bad5"
    )
    assert len(subject.prereg_v4_3.AQUANT_SOURCE_SPECS_V4_3) == 8
    assert all(
        row["git_tree_path"].startswith("A_quant/")
        for row in subject.prereg_v4_3.AQUANT_SOURCE_SPECS_V4_3
    )
    assert subject.SNAPSHOT_ID == "20260717T172132Z"
    assert subject.CUTOFF_DATE == "2026-07-17"
    assert subject.EXPECTED_FULL_A_COUNT == 5502
    assert subject.EXPECTED_CALENDAR_SEMANTIC_SHA256 == (
        "99be5e97027fa1837eb737bd6aa4d1adee57107a3592ed14c30858dc5be28f48"
    )


def test_v42_six_file_hard_lock_is_unchanged() -> None:
    expected = {
        "quant_investor/factors/governance_candidate_preregistration_v4_2.py": (
            "f05007568a955bfe02fc3f3bf7d7b6694259840deee4f4851473cf2a96bc90cc"
        ),
        "quant_investor/factors/governance_candidate_preregistration_bundle_v4_2.py": (
            "49b0c3dd5550494c4fd945234b884b45a25e716699fb6d5a4e5be100f9d40bfe"
        ),
        "scripts/build_factor_v4_2_candidate_preregistration.py": (
            "e9a7a03094bfd5d260e515a0c5dd6c3b2f0714aec4b9467d7b31c28252637e17"
        ),
        "tests/unit/test_factor_governance_candidate_preregistration_v4_2.py": (
            "759582cc2ab2a6e770dfc61d8f37a6a4afa31715eeda3783d9616385394977d4"
        ),
        "tests/unit/test_factor_governance_candidate_preregistration_bundle_v4_2.py": (
            "891249806cf581807b56f3c5dfd082932431b64ba1a92c993878d99adb365d40"
        ),
        "tests/unit/test_build_factor_v4_2_candidate_preregistration.py": (
            "854511a9ef9cbd62a6a54b1b8098b9626d95b346bca2d5949c1581ff065067f3"
        ),
    }
    assert {
        relative: hashlib.sha256((subject.PROJECT_ROOT / relative).read_bytes()).hexdigest()
        for relative in expected
    } == expected
    assert {
        row["relative_path"]: row["expected_byte_sha256"]
        for row in subject.bundle_v4_3.V4_2_LOCKED_FILES_V4_3
    } == expected


def test_private_root_rejects_missing_wrong_mode_owner_and_existing_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.joinpath(*subject.bundle_v4_3.ROOT_SUFFIX_V4_3)
    with pytest.raises(subject.FactorV4_3CandidatePreregistrationRunnerError, match="exist"):
        subject._validate_private_root_preflight(root)

    root = _private_root(tmp_path / "mode")
    root.chmod(0o755)
    with pytest.raises(subject.FactorV4_3CandidatePreregistrationRunnerError, match="0700"):
        subject._validate_private_root_preflight(root)
    root.chmod(0o700)
    real_uid = os.getuid()
    monkeypatch.setattr(subject.os, "getuid", lambda: real_uid + 1)
    with pytest.raises(subject.FactorV4_3CandidatePreregistrationRunnerError, match="owner"):
        subject._validate_private_root_preflight(root)
    monkeypatch.undo()

    root = _private_root(tmp_path / "existing")
    destination = root / subject.FIXED_CYCLE_ID
    destination.mkdir()
    before = tuple(root.iterdir())
    with pytest.raises(
        subject.FactorV4_3CandidatePreregistrationRunnerError,
        match="already exists",
    ):
        subject._validate_private_root_preflight(root)
    assert tuple(root.iterdir()) == before


def test_root_failure_happens_before_any_input_read_or_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.joinpath(*subject.bundle_v4_3.ROOT_SUFFIX_V4_3)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("root rejection must precede all source work")

    monkeypatch.setattr(subject, "_collect_publication_inputs", forbidden)
    monkeypatch.setattr(
        subject.bundle_v4_3,
        "publish_candidate_preregistration_bundle_v4_3",
        forbidden,
    )
    with pytest.raises(subject.FactorV4_3CandidatePreregistrationRunnerError, match="exist"):
        subject.run_publish(argparse.Namespace(command="publish"), private_root=root)
    assert not root.exists()


def test_code_and_protected_descriptor_shapes_orders_and_drift(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    for index, relative in enumerate(subject.CODE_BINDING_PATHS):
        _stable(repository / relative, f"code-{index}\n".encode())
    code = subject._build_code_bindings(repository)
    assert tuple(row["relative_path"] for row in code) == subject.CODE_BINDING_PATHS
    assert all(
        set(row)
        == {
            "relative_path",
            "absolute_path",
            "byte_sha256",
            "size_bytes",
            "mode",
            "uid",
            "nlink",
        }
        for row in code
    )

    specs: list[tuple[str, Path, str]] = []
    for index, name in enumerate(subject.bundle_v4_3.PROTECTED_BINDING_NAMES_V4_3):
        observed = _stable(tmp_path / "protected" / f"{name}.json", f"p-{index}\n".encode())
        specs.append((name, observed.path, observed.byte_sha256))
    protected = subject._build_protected_bindings(specs)
    assert tuple(row["name"] for row in protected) == tuple(
        subject.bundle_v4_3.PROTECTED_BINDING_NAMES_V4_3
    )
    assert all(
        set(row)
        == {
            "name",
            "absolute_path",
            "byte_sha256",
            "size_bytes",
            "mode",
            "uid",
            "nlink",
        }
        for row in protected
    )
    specs[2][1].write_bytes(b"drift\n")
    with pytest.raises(subject.FactorV4_3CandidatePreregistrationRunnerError, match="SHA"):
        subject._build_protected_bindings(specs)


def test_aquant_sources_are_read_only_through_pinned_git_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    real = subject._run_git

    def observed(arguments: list[str]) -> bytes:
        calls.append(tuple(arguments))
        return real(arguments)

    monkeypatch.setattr(subject, "_run_git", observed)
    monkeypatch.setattr(
        subject.prereg_v4_3,
        "build_aquant_source_set_receipt_v4_3",
        lambda **_kwargs: {"accepted": True},
    )
    objects = subject._read_aquant_git_objects()
    assert tuple(objects) == tuple(
        row["git_tree_path"]
        for row in subject.prereg_v4_3.AQUANT_SOURCE_SPECS_V4_3
    )
    assert calls[0][0:2] == ("rev-parse", "--verify")
    assert len([call for call in calls if call[0] == "ls-tree"]) == 8
    assert len([call for call in calls if call[0:2] == ("cat-file", "blob")]) == 8
    assert all(
        not any(token in call for token in ("status", "diff", "show", "log"))
        for call in calls
    )


def test_aquant_git_blob_drift_fails_before_pure_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_rows = iter(subject.prereg_v4_3.AQUANT_SOURCE_SPECS_V4_3)
    current: dict[str, Any] = {}

    def fake(arguments: list[str]) -> bytes:
        if arguments[0] == "rev-parse":
            return f"{subject.prereg_v4_3.AQUANT_COMMIT_V4_3}\n".encode()
        if arguments[0] == "ls-tree":
            row = next(source_rows)
            current.clear()
            current.update(row)
            return (
                f"{row['mode']} blob {row['blob_oid']}\t{row['git_tree_path']}\n"
            ).encode()
        assert arguments[0:2] == ["cat-file", "blob"]
        return b"drifted Git object\n"

    monkeypatch.setattr(subject, "_run_git", fake)
    monkeypatch.setattr(subject, "_validate_static_contract", lambda: None)
    with pytest.raises(subject.FactorV4_3CandidatePreregistrationRunnerError, match="blob SHA"):
        subject._read_aquant_git_objects()


def test_strict_source_builder_receives_only_the_fixed_explicit_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    backend = {
        "snapshot_id": subject.SNAPSHOT_ID,
        "cutoff_date": subject.CUTOFF_DATE,
        "table": {"inventory_sha256": subject.EXPECTED_TABLE_INVENTORY_SEMANTIC_SHA256},
        "components": {
            "count": subject.EXPECTED_FULL_A_COUNT,
            "newline_set_sha256": subject.EXPECTED_FULL_A_SEMANTIC_SHA256,
        },
    }
    bound = SimpleNamespace(binding=backend)

    def bind(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return bound

    strict = {
        "protocol_version": "v4",
        "snapshot_id": subject.SNAPSHOT_ID,
        "cutoff": subject.CUTOFF_DATE,
        "latest_available_trade_date": subject.LATEST_COMPLETE_TRADE_DATE,
        "latest_complete_trade_date": subject.LATEST_COMPLETE_TRADE_DATE,
        "expected_scope_count": subject.EXPECTED_FULL_A_COUNT,
        "full_a_scope_sha256": subject.EXPECTED_FULL_A_SEMANTIC_SHA256,
        "calendar_semantic_sha256": subject.EXPECTED_CALENDAR_SEMANTIC_SHA256,
        "table_inventory_semantic_sha256": (
            subject.EXPECTED_TABLE_INVENTORY_SEMANTIC_SHA256
        ),
        "serving_inventory_semantic_sha256": (
            subject.EXPECTED_SERVING_INVENTORY_SEMANTIC_SHA256
        ),
    }
    monkeypatch.setattr(
        subject.source_readback_v4_1,
        "bind_explicit_cutoff_inputs_v4_1",
        bind,
    )
    monkeypatch.setattr(
        subject.bundle_v4_3,
        "build_strict_full_a_source_binding_v4_3",
        lambda *, bound_inputs: strict if bound_inputs is bound else None,
    )
    assert subject._build_strict_source_binding() == strict
    assert captured["latest_pointer_path"] == subject.LATEST_POINTER_PATH
    assert captured["snapshot_manifest_path"] == subject.SNAPSHOT_MANIFEST_PATH
    assert captured["pit_membership_path"] == subject.PIT_MEMBERSHIP_PATH
    assert captured["table_root"] == subject.TABLE_ROOT
    assert captured["snapshot_id"] == subject.SNAPSHOT_ID
    assert captured["cutoff_date"] == subject.CUTOFF_DATE
    assert captured["expected_full_a_count"] == 5502


@pytest.mark.parametrize(
    "field",
    [
        "aquant_git_objects",
        "strict_source_binding",
        "code_bindings",
        "protected_bindings",
        "runtime_fingerprint",
    ],
)
def test_under_lock_callback_rejects_git_source_code_or_protected_drift(
    field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    first = _entry(tmp_path)
    replacements: dict[str, Any] = {
        "aquant_git_objects": {"A_quant/source.py": b"changed\n"},
        "strict_source_binding": {"artifact_semantic_sha256": "9" * 64},
        "code_bindings": ({**first.code_bindings[0], "byte_sha256": "9" * 64},),
        "protected_bindings": (
            {**first.protected_bindings[0], "byte_sha256": "9" * 64},
        ),
        "runtime_fingerprint": {"fixed": False},
    }
    changed = replace(first, **{field: replacements[field]})
    entries = iter((first, changed))
    monkeypatch.setattr(subject, "_validate_static_contract", lambda: None)
    monkeypatch.setattr(
        subject,
        "_collect_publication_inputs",
        lambda **_kwargs: next(entries),
    )

    def fake_publish(**kwargs: Any) -> dict[str, Any]:
        kwargs["revalidate_inputs"]()
        raise AssertionError("commit must not be reached after drift")

    monkeypatch.setattr(
        subject.bundle_v4_3,
        "publish_candidate_preregistration_bundle_v4_3",
        fake_publish,
    )
    with pytest.raises(subject.FactorV4_3CandidatePreregistrationRunnerError, match="changed"):
        subject.run_publish(
            argparse.Namespace(command="publish"),
            private_root=root,
            preregistered_at_factory=lambda: "2026-07-19T11:00:00+08:00",
        )
    assert not (root / subject.FIXED_CYCLE_ID).exists()


def test_publish_passes_exact_public_api_and_emits_committed_only_after_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    entry = _entry(tmp_path)
    monkeypatch.setattr(subject, "_validate_static_contract", lambda: None)
    monkeypatch.setattr(subject, "_collect_publication_inputs", lambda **_kwargs: entry)
    calls = 0

    def publish(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert set(kwargs) == {
            "private_root",
            "repository_root",
            "preregistered_at",
            "aquant_git_objects",
            "strict_source_binding",
            "code_bindings",
            "protected_bindings",
            "revalidate_inputs",
        }
        assert kwargs["private_root"] == root
        assert kwargs["aquant_git_objects"] == entry.aquant_git_objects
        kwargs["revalidate_inputs"]()
        return _published(root)

    monkeypatch.setattr(
        subject.bundle_v4_3,
        "publish_candidate_preregistration_bundle_v4_3",
        publish,
    )
    result = subject.run_publish(
        argparse.Namespace(command="publish"),
        private_root=root,
        preregistered_at_factory=lambda: "2026-07-19T11:00:00+08:00",
    )
    assert calls == 1
    assert result["accepted"] is True
    assert result["status"] == "COMMITTED"
    assert result["publication_phase"] == "COMMITTED"
    assert result["internal_readback_report_phase"] == "PRECOMMIT_INTENT_ONLY"
    assert result["readback_report_byte_sha256"] == "4" * 64
    assert result["readback_report_semantic_sha256"] == "5" * 64
    assert result["authority"] == subject.prereg_v4_3.AUTHORITY_FLAGS_V4_3
    assert result["side_effects"] == subject.prereg_v4_3.SIDE_EFFECT_FLAGS_V4_3


def test_publish_rejects_internal_report_that_claims_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    entry = _entry(tmp_path)
    monkeypatch.setattr(subject, "_validate_static_contract", lambda: None)
    monkeypatch.setattr(subject, "_collect_publication_inputs", lambda **_kwargs: entry)
    monkeypatch.setattr(
        subject.bundle_v4_3,
        "publish_candidate_preregistration_bundle_v4_3",
        lambda **_kwargs: _published(root, report_phase="COMMITTED"),
    )
    with pytest.raises(subject.FactorV4_3CandidatePreregistrationRunnerError, match="PRECOMMIT"):
        subject.run_publish(
            argparse.Namespace(command="publish"),
            private_root=root,
            preregistered_at_factory=lambda: "2026-07-19T11:00:00+08:00",
        )


def test_actual_tmp_root_publish_readback_hash_mismatch_and_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    lock_observations: list[tuple[str, ...]] = []
    real_observe_lock = (
        subject.bundle_v4_3._observe_v4_2_contract_lock_snapshot
    )

    def observe_lock(**kwargs: Any) -> Any:
        snapshot = real_observe_lock(**kwargs)
        lock_observations.append(
            tuple(
                row["byte_sha256"]
                for row in snapshot.artifact["ordered_locked_files"]
            )
        )
        return snapshot

    monkeypatch.setattr(
        subject.bundle_v4_3,
        "_observe_v4_2_contract_lock_snapshot",
        observe_lock,
    )
    published = subject.run_publish(
        argparse.Namespace(command="publish"),
        private_root=root,
        preregistered_at_factory=lambda: "2026-07-19T11:00:00+08:00",
    )
    assert published["status"] == "COMMITTED"
    assert len(lock_observations) == 3
    assert len(set(lock_observations)) == 1
    assert len(lock_observations[0]) == 6
    bundle_path = Path(published["bundle_path"])
    assert bundle_path == root / subject.FIXED_CYCLE_ID
    assert bundle_path.is_dir()

    monkeypatch.setattr(subject, "PRODUCTION_PRIVATE_ROOT", root)
    monkeypatch.setattr(subject, "_validate_static_contract", lambda: None)
    args = argparse.Namespace(
        bundle_path=str(bundle_path),
        expected_readback_report_byte_sha256=published[
            "readback_report_byte_sha256"
        ],
        expected_readback_report_semantic_sha256=published[
            "readback_report_semantic_sha256"
        ],
    )
    readback = subject.run_readback(args)
    assert readback["status"] == "READBACK_ACCEPTED"
    assert readback["publication_phase_claimed"] is False

    before = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    bad_args = argparse.Namespace(**vars(args))
    bad_args.expected_readback_report_byte_sha256 = "0" * 64
    with pytest.raises(
        subject.bundle_v4_3.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match="expected readback report byte SHA",
    ):
        subject.run_readback(bad_args)
    with pytest.raises(
        subject.FactorV4_3CandidatePreregistrationRunnerError,
        match="already exists",
    ):
        subject.run_publish(
            argparse.Namespace(command="publish"),
            private_root=root,
            preregistered_at_factory=lambda: "2026-07-19T11:00:01+08:00",
        )
    after = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_explicit_readback_calls_exact_hash_bound_bundle_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    bundle_path = root / subject.FIXED_CYCLE_ID
    monkeypatch.setattr(subject, "PRODUCTION_PRIVATE_ROOT", root)
    monkeypatch.setattr(subject, "_validate_static_contract", lambda: None)
    captured: dict[str, Any] = {}

    def readback(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "accepted": True,
            "expected_hashes_verified": True,
            "bundle_path": str(bundle_path),
            "readback_report_byte_sha256": "6" * 64,
            "readback_report_semantic_sha256": "7" * 64,
        }

    monkeypatch.setattr(
        subject.bundle_v4_3,
        "readback_candidate_preregistration_bundle_v4_3",
        readback,
    )
    args = argparse.Namespace(
        bundle_path=str(bundle_path),
        expected_readback_report_byte_sha256="6" * 64,
        expected_readback_report_semantic_sha256="7" * 64,
    )
    result = subject.run_readback(args)
    assert captured == {
        "bundle_path": bundle_path,
        "expected_readback_report_byte_sha256": "6" * 64,
        "expected_readback_report_semantic_sha256": "7" * 64,
    }
    assert result["status"] == "READBACK_ACCEPTED"
    assert result["publication_phase_claimed"] is False
    assert result["current_latest_or_fallback_discovery_used"] is False


def test_readback_hash_or_path_mismatch_fails_before_bundle_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    monkeypatch.setattr(subject, "PRODUCTION_PRIVATE_ROOT", root)
    monkeypatch.setattr(subject, "_validate_static_contract", lambda: None)

    def forbidden(**_kwargs: Any) -> Any:
        raise AssertionError("invalid explicit readback must not reach bundle")

    monkeypatch.setattr(
        subject.bundle_v4_3,
        "readback_candidate_preregistration_bundle_v4_3",
        forbidden,
    )
    with pytest.raises(subject.FactorV4_3CandidatePreregistrationRunnerError, match="exact fixed"):
        subject.run_readback(
            argparse.Namespace(
                bundle_path=str(root / "other-cycle"),
                expected_readback_report_byte_sha256="6" * 64,
                expected_readback_report_semantic_sha256="7" * 64,
            )
        )
    with pytest.raises(subject.FactorV4_3CandidatePreregistrationRunnerError, match="SHA-256"):
        subject.run_readback(
            argparse.Namespace(
                bundle_path=str(root / subject.FIXED_CYCLE_ID),
                expected_readback_report_byte_sha256="not-a-hash",
                expected_readback_report_semantic_sha256="7" * 64,
            )
        )


def test_runner_has_no_provider_llm_or_execution_imports() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(
        {
            "requests",
            "httpx",
            "tushare",
            "yfinance",
            "openai",
            "anthropic",
            "ccxt",
            "ib_insync",
        }
    )
