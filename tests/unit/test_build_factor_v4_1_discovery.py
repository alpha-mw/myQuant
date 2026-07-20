from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.build_factor_v4_1_discovery as runner


GIT = "/usr/bin/git"


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        [GIT, "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "LC_ALL": "C",
        },
    )
    return completed.stdout


def _git_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    executable_key: str | None = None,
) -> tuple[Path, SimpleNamespace, dict[str, bytes]]:
    repository = tmp_path / "git-parent"
    repository.mkdir(mode=0o700)
    subprocess.run(
        [GIT, "init", "-q", str(repository)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _git(repository, "config", "user.name", "Fixture")
    _git(repository, "config", "user.email", "fixture@example.invalid")

    payloads: dict[str, bytes] = {}
    provisional_specs: list[tuple[str, str]] = []
    for index, authoritative in enumerate(runner.AQUANT_SOURCE_SPECS):
        data = f"fixture-{authoritative.key}-{index}\n".encode("ascii")
        target = repository / authoritative.repository_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if authoritative.key == executable_key:
            target.chmod(0o755)
        payloads[authoritative.key] = data
        provisional_specs.append((authoritative.key, authoritative.repository_path))
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "fixture")
    commit = _git(repository, "rev-parse", "HEAD").decode().strip()

    specs: list[runner.AquantSourceSpec] = []
    for key, repository_path in provisional_specs:
        row = _git(repository, "ls-tree", commit, "--", repository_path)
        blob_oid = row.decode().split()[2]
        specs.append(
            runner.AquantSourceSpec(
                key=key,
                repository_path=repository_path,
                expected_blob_oid=blob_oid,
                expected_raw_sha256=hashlib.sha256(payloads[key]).hexdigest(),
            )
        )

    monkeypatch.setattr(runner, "EXPECTED_AQUANT_GIT_TOP_LEVEL", repository)
    monkeypatch.setattr(runner, "PINNED_AQUANT_COMMIT", commit)
    monkeypatch.setattr(runner, "AQUANT_SOURCE_SPECS", tuple(specs))
    values: dict[str, Any] = {
        "git_executable": GIT,
        "aquant_git_top_level": str(repository),
        "aquant_pinned_commit": commit,
    }
    for spec in specs:
        values[f"expected_aquant_{spec.key}_blob_oid"] = spec.expected_blob_oid
        values[f"expected_aquant_{spec.key}_sha256"] = spec.expected_raw_sha256
    return repository, SimpleNamespace(**values), payloads


def _dummy_cli(tmp_path: Path, *, private_root: str | None = None) -> list[str]:
    unused = str((tmp_path / "unused").resolve())
    sha = "1" * 64
    arguments = [
        "--predecessor-input-binding-path",
        unused,
        "--expected-predecessor-input-binding-sha256",
        sha,
        "--expected-predecessor-input-binding-semantic-sha256",
        sha,
        "--predecessor-design-source-path",
        unused,
        "--expected-predecessor-design-source-sha256",
        sha,
        "--expected-predecessor-design-source-semantic-sha256",
        sha,
        "--precommitted-state-path",
        unused,
        "--expected-precommitted-state-sha256",
        sha,
        "--expected-precommitted-state-semantic-sha256",
        sha,
        "--predecessor-source-node-path",
        unused,
        "--expected-predecessor-source-node-sha256",
        sha,
        "--expected-predecessor-source-node-semantic-sha256",
        sha,
        "--predecessor-readback-report-path",
        unused,
        "--expected-predecessor-readback-report-sha256",
        sha,
        "--expected-predecessor-readback-report-semantic-sha256",
        sha,
        "--base-ontology-path",
        unused,
        "--expected-base-ontology-sha256",
        sha,
        "--expected-base-ontology-semantic-sha256",
        sha,
        "--base-catalog-path",
        unused,
        "--expected-base-catalog-sha256",
        sha,
        "--expected-base-catalog-semantic-sha256",
        sha,
        "--local-evaluator-path",
        unused,
        "--expected-local-evaluator-sha256",
        sha,
        "--private-root",
        private_root or str((tmp_path / "private").resolve()),
        "--run-id",
        "fixture-discovery",
        "--cycle-id",
        "fixture-cycle",
        "--aquant-git-top-level",
        str(runner.EXPECTED_AQUANT_GIT_TOP_LEVEL),
        "--aquant-pinned-commit",
        runner.PINNED_AQUANT_COMMIT,
    ]
    for spec in runner.AQUANT_SOURCE_SPECS:
        flag = spec.key.replace("_", "-")
        arguments.extend(
            [
                f"--expected-aquant-{flag}-blob-oid",
                spec.expected_blob_oid,
                f"--expected-aquant-{flag}-sha256",
                spec.expected_raw_sha256,
            ]
        )
    return arguments


def _bound_fixture(tmp_path: Path) -> runner.BoundDiscoveryInputs:
    source_sha = "2" * 64
    state = runner.cycle_state.build_genesis_cycle_state_v4_1(
        cycle_id="fixture-cycle",
        cycle_root_sha256="3" * 64,
        source_chain_node_sha256=source_sha,
    )
    state_binding = runner.BoundJsonArtifact(
        absolute_path=str(tmp_path / "state.json"),
        raw_sha256=runner.cycle_state.byte_sha256(state),
        semantic_sha256=state["state_semantic_sha256"],
        value=state,
    )
    source_binding = runner.BoundJsonArtifact(
        absolute_path=str(tmp_path / "source.json"),
        raw_sha256="4" * 64,
        semantic_sha256=source_sha,
        value={"semantic_sha256": source_sha},
    )
    input_binding = runner.BoundJsonArtifact(
        absolute_path=str(tmp_path / "input.json"),
        raw_sha256="0" * 63 + "1",
        semantic_sha256="0" * 63 + "2",
        value={},
    )
    design_source = runner.BoundJsonArtifact(
        absolute_path=str(tmp_path / "design.json"),
        raw_sha256="0" * 63 + "3",
        semantic_sha256="0" * 63 + "4",
        value={},
    )
    readback_report = runner.BoundJsonArtifact(
        absolute_path=str(tmp_path / "readback.json"),
        raw_sha256="0" * 63 + "5",
        semantic_sha256="0" * 63 + "6",
        value={},
    )
    ontology = runner.BoundJsonArtifact(
        absolute_path=str(tmp_path / "ontology.json"),
        raw_sha256="5" * 64,
        semantic_sha256="6" * 64,
        value={"semantic_sha256": "6" * 64},
    )
    catalog = runner.BoundJsonArtifact(
        absolute_path=str(tmp_path / "catalog.json"),
        raw_sha256="7" * 64,
        semantic_sha256="8" * 64,
        value={"semantic_sha256": "8" * 64},
    )
    evaluator = runner.BoundFile(str(tmp_path / "eval.py"), "9" * 64, 10)
    generator = runner.BoundGitSource(
        key="generator",
        repository_path="A_quant/scripts/run_factor_batch_screen.py",
        blob_oid="a" * 40,
        raw_sha256="b" * 64,
        size_bytes=10,
        data=b"generator",
    )
    git = runner.BoundGitObjects(
        repository_top_level=str(tmp_path / "git"),
        git_dir=str(tmp_path / "git" / ".git"),
        object_dir=str(tmp_path / "git" / ".git" / "objects"),
        pinned_commit="c" * 40,
        sources=(generator,),
    )
    code = (runner.BoundFile(str(tmp_path / "runner.py"), "d" * 64, 1),)
    return runner.BoundDiscoveryInputs(
        predecessor_input_binding=input_binding,
        predecessor_design_source=design_source,
        predecessor_state=state_binding,
        predecessor_source_node=source_binding,
        predecessor_readback_report=readback_report,
        base_ontology=ontology,
        base_catalog=catalog,
        local_evaluator=evaluator,
        git_objects=git,
        code_bindings=code,
        stable_identity_sha256="e" * 64,
    )


def test_sanitized_git_environment_drops_all_inherited_git_controls() -> None:
    environment = runner.sanitized_git_environment(
        {
            "PATH": "/safe/path",
            "HOME": "/safe/home",
            "GIT_DIR": "/attacker/repo",
            "GIT_OBJECT_DIRECTORY": "/attacker/objects",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.sshCommand",
            "GIT_CONFIG_VALUE_0": "attacker",
        }
    )

    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["HOME"] == "/var/empty"
    assert environment["TMPDIR"] == "/private/tmp"
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_NO_LAZY_FETCH"] == "1"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["LC_ALL"] == "C"
    assert "GIT_DIR" not in environment
    assert "GIT_OBJECT_DIRECTORY" not in environment
    assert "GIT_CONFIG_COUNT" not in environment
    assert "GIT_CONFIG_KEY_0" not in environment
    assert "GIT_CONFIG_VALUE_0" not in environment


def test_cli_rejects_relative_paths_malformed_hashes_and_operational_options(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(_dummy_cli(tmp_path, private_root="relative/private"))
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="absolute path"):
        runner.run(args)
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="SHA-256"):
        runner._sha256("ABC", "fixture")
    with pytest.raises(SystemExit):
        runner.parse_args([*_dummy_cli(tmp_path), "--registry-path", "/tmp/registry"])


def test_strict_json_rejects_duplicate_keys_and_non_finite_constants() -> None:
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="duplicate"):
        runner._decode_json_object(b'{"a":1,"a":2}', "fixture")
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="non-finite"):
        runner._decode_json_object(b'{"a":NaN}', "fixture")


def test_secure_file_reader_rejects_symlink_hardlink_and_private_mode(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private.json"
    private.write_text("{}", encoding="utf-8")
    private.chmod(0o644)
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="0600"):
        runner._regular_file_bytes(str(private), "fixture", require_private=True)
    private.chmod(0o600)
    hardlink = tmp_path / "hardlink.json"
    os.link(private, hardlink)
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="hard-link"):
        runner._regular_file_bytes(str(private), "fixture", require_private=True)
    hardlink.unlink()
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(private)
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="non-symlink"):
        runner._regular_file_bytes(str(symlink), "fixture", require_private=True)


def test_predecessor_directory_requires_exact_private_inventory(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "predecessor"
    bundle.mkdir(mode=0o700)
    artifacts: dict[str, runner.BoundJsonArtifact] = {}
    for index, filename in enumerate(
        sorted(runner.PREDECESSOR_DIRECTORY_ENTRIES - {".lock"}), start=1
    ):
        path = bundle / filename
        path.write_bytes(b"{}\n")
        path.chmod(0o600)
        artifacts[filename] = runner.BoundJsonArtifact(
            absolute_path=str(path),
            raw_sha256=f"{index:064x}",
            semantic_sha256=f"{index + 10:064x}",
            value={},
        )
    lock = bundle / ".lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)

    runner._verify_predecessor_directory(artifacts)
    extra = bundle / "unexpected.json"
    extra.write_bytes(b"{}\n")
    extra.chmod(0o600)
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="entries mismatch"):
        runner._verify_predecessor_directory(artifacts)


def test_pinned_git_object_read_ignores_dirty_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, args, payloads = _git_fixture(tmp_path, monkeypatch)

    first = runner.bind_pinned_aquant_git_objects(args)  # type: ignore[arg-type]
    generator_path = next(
        spec.repository_path
        for spec in runner.AQUANT_SOURCE_SPECS
        if spec.key == "generator"
    )
    (repository / generator_path).write_text("dirty-worktree\n", encoding="utf-8")
    second = runner.bind_pinned_aquant_git_objects(args)  # type: ignore[arg-type]

    assert first == second
    assert {item.key: item.data for item in second.sources} == payloads


def test_git_alternates_replacements_and_worktree_config_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, args, _payloads = _git_fixture(tmp_path, monkeypatch)
    alternates = repository / ".git" / "objects" / "info" / "alternates"
    alternates.parent.mkdir(parents=True, exist_ok=True)
    alternates.write_text("/tmp/attacker\n", encoding="utf-8")
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="alternate"):
        runner.bind_pinned_aquant_git_objects(args)  # type: ignore[arg-type]
    alternates.unlink()

    replacement = repository / ".git" / "refs" / "replace" / args.aquant_pinned_commit
    replacement.parent.mkdir(parents=True, exist_ok=True)
    replacement.write_text(f"{args.aquant_pinned_commit}\n", encoding="ascii")
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="replacement"):
        runner.bind_pinned_aquant_git_objects(args)  # type: ignore[arg-type]
    replacement.unlink()

    worktree_config = repository / ".git" / "config.worktree"
    worktree_config.write_text("[core]\n\tbare = false\n", encoding="utf-8")
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="worktree config"):
        runner.bind_pinned_aquant_git_objects(args)  # type: ignore[arg-type]


def test_git_mode_blob_and_source_substitution_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository, args, _payloads = _git_fixture(
        tmp_path, monkeypatch, executable_key="generator"
    )
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="mode/blob/path"):
        runner.bind_pinned_aquant_git_objects(args)  # type: ignore[arg-type]

    tmp_path_two = tmp_path / "second"
    tmp_path_two.mkdir()
    _repository, args, _payloads = _git_fixture(tmp_path_two, monkeypatch)
    specs = list(runner.AQUANT_SOURCE_SPECS)
    specs[0] = replace(specs[0], expected_raw_sha256="f" * 64)
    monkeypatch.setattr(runner, "AQUANT_SOURCE_SPECS", tuple(specs))
    setattr(args, f"expected_aquant_{specs[0].key}_sha256", "f" * 64)
    with pytest.raises(
        runner.FactorV4_1DiscoveryRunnerError,
        match="raw SHA-256 mismatch",
    ):
        runner.bind_pinned_aquant_git_objects(args)  # type: ignore[arg-type]


def test_candidate_and_accounting_oracles_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [{"name": f"candidate_{index:03d}"} for index in range(100)]
    names_sha = runner._semantic_sha256([row["name"] for row in candidates])
    monkeypatch.setattr(runner, "EXPECTED_ORDERED_NAMES_SEMANTIC_SHA256", names_sha)
    runner._verify_candidate_oracle(candidates)
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="name oracle"):
        runner._verify_candidate_oracle([*candidates[:-1], {"name": "changed"}])

    compatible = [f"compatible_{index:02d}" for index in range(43)]
    aliases = [f"alias_{index:02d}" for index in range(6)]
    monkeypatch.setattr(
        runner,
        "EXPECTED_COMPATIBLE_NAMES_SEMANTIC_SHA256",
        runner._semantic_sha256(compatible),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_ALIAS_NAMES_SEMANTIC_SHA256",
        runner._semantic_sha256(aliases),
    )
    artifacts: dict[str, dict[str, Any]] = {
        "source_idea_audit.v4_1.json": {
            "total_idea_count": 100,
            "compatible_count": 43,
            "incompatible_count": 57,
            "new_candidate_count": 37,
            "structural_alias_count": 6,
            "compatible_ordered_names_semantic_sha256": runner._semantic_sha256(
                compatible
            ),
            "structural_alias_ordered_names_semantic_sha256": (
                runner._semantic_sha256(aliases)
            ),
        },
        "discovery_catalog.v4_1.json": {
            "member_count": 273,
            "selected_count": 267,
        },
        "structural_collision_audit.v4_1.json": {},
    }
    runner._validate_accounting_oracle(artifacts)
    artifacts["discovery_catalog.v4_1.json"]["selected_count"] = 266
    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="selected_count"):
        runner._validate_accounting_oracle(artifacts)

    assert runner.EXPECTED_AQUANT_ACCOUNTING == {
        "source_idea_count": 100,
        "compatible_count": 43,
        "incompatible_count": 57,
        "new_candidate_count": 37,
        "structural_alias_count": 6,
        "discovery_member_count": 273,
        "selected_count": 267,
        "unselected_count": 6,
    }


def test_stable_input_callback_detects_cas_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = _bound_fixture(tmp_path)
    changed = replace(initial, stable_identity_sha256="f" * 64)
    monkeypatch.setattr(runner, "_bind_all_inputs", lambda _args: changed)

    with pytest.raises(runner.FactorV4_1DiscoveryRunnerError, match="CAS changed"):
        runner._make_revalidator(argparse.Namespace(), initial)()


def test_runner_builds_only_discovery_and_uses_monkeypatched_publisher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = runner.parse_args(_dummy_cli(tmp_path))
    bound = _bound_fixture(tmp_path)
    args.expected_precommitted_state_sha256 = bound.predecessor_state.raw_sha256
    args.expected_precommitted_state_semantic_sha256 = (
        bound.predecessor_state.semantic_sha256
    )
    monkeypatch.setattr(runner, "_bind_all_inputs", lambda _args: bound)
    candidates = [{"name": f"candidate_{index:03d}"} for index in range(100)]
    monkeypatch.setattr(
        runner.discovery,
        "extract_aquant_candidates_from_source",
        lambda _source: candidates,
    )
    monkeypatch.setattr(runner, "_verify_candidate_oracle", lambda _rows: None)
    monkeypatch.setattr(
        runner.discovery,
        "build_aquant_source_receipt_v4_1",
        lambda **_kwargs: {"semantic_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        runner.discovery,
        "build_local_compatibility_contract_v4_1",
        lambda **_kwargs: {"semantic_sha256": "2" * 64},
    )
    first_six = {
        "aquant_source_receipt.v4_1.json": {"semantic_sha256": "1" * 64},
        "source_idea_audit.v4_1.json": {},
        "local_compatibility_contract.v4_1.json": {
            "semantic_sha256": "2" * 64
        },
        "discovery_catalog.v4_1.json": {},
        "structural_collision_audit.v4_1.json": {},
        "discovery_source_node.v4_1.json": {"semantic_sha256": "4" * 64},
    }
    monkeypatch.setattr(
        runner.discovery,
        "build_source_idea_audit_v4_1",
        lambda **_kwargs: first_six["source_idea_audit.v4_1.json"],
    )
    monkeypatch.setattr(
        runner.discovery,
        "build_discovery_catalog_v4_1",
        lambda **_kwargs: first_six["discovery_catalog.v4_1.json"],
    )
    monkeypatch.setattr(
        runner.discovery,
        "build_structural_collision_audit_v4_1",
        lambda **_kwargs: first_six["structural_collision_audit.v4_1.json"],
    )
    monkeypatch.setattr(
        runner.discovery,
        "build_discovery_source_node_v4_1",
        lambda **_kwargs: first_six["discovery_source_node.v4_1.json"],
    )
    monkeypatch.setattr(
        runner.discovery,
        "build_discovery_cycle_state_v4_1",
        lambda **_kwargs: runner.cycle_state.build_next_cycle_state_v4_1(
            predecessor=bound.predecessor_state.value,
            predecessor_byte_sha256=bound.predecessor_state.raw_sha256,
            expected_predecessor_byte_sha256=bound.predecessor_state.raw_sha256,
            expected_predecessor_semantic_sha256=(
                bound.predecessor_state.semantic_sha256
            ),
            cycle_id="fixture-cycle",
            cycle_root_sha256=bound.predecessor_state.value[
                "cycle_root_sha256"
            ],
            next_state=runner.cycle_state.DISCOVERY,
            source_chain_node_sha256="4" * 64,
        ),
    )
    validated_filenames: list[str] = []

    def fake_validate(filename: str, value: dict[str, Any]) -> dict[str, Any]:
        validated_filenames.append(filename)
        return value

    monkeypatch.setattr(
        runner.discovery,
        "validate_discovery_artifact_v4_1",
        fake_validate,
    )
    monkeypatch.setattr(runner, "_validate_accounting_oracle", lambda _artifacts: None)
    observed: dict[str, Any] = {}

    def fake_publish(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        assert set(kwargs["artifacts"]) == {
            *first_six,
            "cycle_state.discovery.v4_1.json",
        }
        state = kwargs["artifacts"]["cycle_state.discovery.v4_1.json"]
        assert state["state"] == "DISCOVERY"
        assert state["holdout_unsealed"] is False
        assert state["source_chain_node_sha256"] == "4" * 64
        kwargs["revalidate_inputs"]()
        return {
            "readiness": "EXPLORATORY_DISCOVERY",
            "qualification": False,
            "bundle_path": str(tmp_path / "not-written"),
            "side_effects": dict(runner.FIXED_SIDE_EFFECTS),
        }

    monkeypatch.setattr(
        runner.publication, "publish_discovery_bundle_v4_1", fake_publish
    )

    result = runner.run(args)

    assert result["readiness"] == "EXPLORATORY_DISCOVERY"
    assert result["qualification"] is False
    assert result["formal_admission_authority"] is False
    assert result["production_apply_enabled"] is False
    assert result["holdout"] == "sealed_not_appended"
    assert result["blockers"] == list(runner.FIXED_BLOCKERS)
    assert result["statuses"] == runner.FIXED_NOT_RUN_STATUSES
    assert all(value is False for value in result["side_effects"].values())
    assert validated_filenames == list(
        runner.discovery.PRE_READBACK_ARTIFACT_FILENAMES
    )
    assert "readback_context" not in observed
    assert not (tmp_path / "not-written").exists()


def test_main_fails_closed_without_emitting_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        runner,
        "run",
        lambda _args: (_ for _ in ()).throw(
            runner.FactorV4_1DiscoveryRunnerError("stable CAS mismatch")
        ),
    )

    assert runner.main(_dummy_cli(tmp_path)) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload == {
        "error": "stable CAS mismatch",
        "qualification": False,
        "readiness": "BLOCKED_FAIL_CLOSED",
    }
