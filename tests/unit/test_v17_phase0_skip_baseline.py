from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
from types import ModuleType
from typing import TypedDict

import pytest

from quant_investor.v17_v2_contract.schema_validation import (
    preflight_packaged_schema,
    validate_instance_against_schema,
)


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = _load_script("v17_phase0_skip_baseline")


def _parent_runtime_binding() -> dict[str, object]:
    prefix = "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/" "Python.framework/Versions/3.13"
    paths = [
        f"{prefix}/lib/python313.zip",
        f"{prefix}/lib/python3.13",
        f"{prefix}/lib/python3.13/lib-dynload",
    ]
    return {
        "executable": True,
        "flags": {
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "isolated": 1,
            "no_site": 1,
            "no_user_site": 1,
            "safe_path": True,
        },
        "implementation": "cpython",
        "lexical_executable": subject.BASE_PYTHON_PATH,
        "mode": "0755",
        "pyvenv_cfg": {"path": f"{prefix}/pyvenv.cfg", "state": "ABSENT"},
        "resolved_executable": subject.BASE_PYTHON_PATH,
        "sha256": subject.BASE_PYTHON_SHA256,
        "size_bytes": subject.BASE_PYTHON_SIZE,
        "sys_base_prefix": prefix,
        "sys_base_prefix_realpath": prefix,
        "sys_path": paths,
        "sys_path_sha256": subject._sha256(subject._canonical_bytes(paths)),
        "sys_prefix": prefix,
        "sys_prefix_realpath": prefix,
        "version": subject.BASE_PYTHON_VERSION,
        "version_info": list(subject.BASE_PYTHON_VERSION_INFO),
    }


def _pip_status() -> dict[str, object]:
    return {
        "child_environment_policy": dict(subject.PIP_CHILD_ENVIRONMENT_POLICY),
        "loaded_modules": [],
        "observation_scope": subject.PIP_OBSERVATION_SCOPE,
        "pip_spec": {
            "origin": None,
            "search_locations": [],
            "visible": False,
        },
        "site_sys_path_entries": [],
    }


def _run(argv: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": "/var/empty",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": "/private/tmp",
        },
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _private_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True)
        path.chmod(0o700)
    return path


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-q"], cwd=repo)
    _run(["git", "config", "user.email", "phase0@example.invalid"], cwd=repo)
    _run(["git", "config", "user.name", "Phase Zero"], cwd=repo)
    (repo / "README.md").write_text("phase zero\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "quant-investor"\nversion = "17.0.0"\n',
        encoding="utf-8",
    )
    (repo / "requirements.txt").write_text("pytest==9.0.2\n", encoding="utf-8")
    package = repo / "quant_investor"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "17.0.0"\n', encoding="utf-8")
    tests = repo / "tests" / "unit"
    tests.mkdir(parents=True)
    (tests / "test_optional.py").write_text(
        "def test_optional():\n    pass\n",
        encoding="utf-8",
    )
    producer = repo / subject.PRODUCER_PATH
    producer.parent.mkdir(parents=True)
    shutil.copyfile(Path(subject.__file__), producer)
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-q", "-m", "base"], cwd=repo)
    return repo


def _pytest_output(*, skipped: int = 42, extra: str = "") -> bytes:
    return (
        "=========================== short test summary info ============================\n"
        f"SKIPPED [{skipped}] tests/unit/test_optional.py:1: optional dependency unavailable\n"
        f"{extra}"
        f"9 passed, {skipped} skipped in 0.21s\n"
    ).encode("utf-8")


def _main_suite_binding(path: Path, *, fill: str) -> dict[str, object]:
    return {
        "gid": os.getgid(),
        "mode": "0644",
        "path": str(path),
        "sha256": fill * 64,
        "size_bytes": 123,
        "st_dev": 1,
        "st_ino": 2,
        "st_nlink": 1,
        "uid": os.getuid(),
    }


def _policy_bindings(repo_root: Path) -> dict[str, dict[str, object]]:
    return {
        "policy_binding": _main_suite_binding(
            repo_root / subject.MAIN_SUITE_POLICY_PATH,
            fill="5",
        ),
        "policy_manifest_binding": _main_suite_binding(
            repo_root / subject.MAIN_SUITE_PACKAGE_MANIFEST_PATH,
            fill="6",
        ),
        "policy_schema_binding": _main_suite_binding(
            repo_root / subject.MAIN_SUITE_POLICY_SCHEMA_PATH,
            fill="7",
        ),
    }


def _main_suite_contract_inputs(
    repo_root: Path,
) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
    runtime = _private_dir(repo_root.parent / "main-suite-runtime")
    home = _private_dir(runtime / "home")
    tmp = _private_dir(runtime / "tmp")
    cache = _private_dir(runtime / "cache")
    black = _private_dir(cache / "black")
    mypy = _private_dir(cache / "mypy")
    pycache = _private_dir(cache / "pycache")
    environment = {
        "BLACK_CACHE_DIR": str(black),
        "HOME": str(home),
        "MYPY_CACHE_DIR": str(mypy),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPYCACHEPREFIX": str(pycache),
        "TMPDIR": str(tmp),
        "XDG_CACHE_HOME": str(cache),
    }
    policy = {
        "main_runtime": {"lexical_python": "/bound/python"},
        "module_policy": {
            "candidate_content_binding": "OUTER_SOURCE_STATE",
            "candidate_module_source_paths": ["quant_investor/__init__.py"],
        },
        "pytest_environment": {
            "allowed_keys": sorted(environment),
            "dynamic_path_keys": [
                "BLACK_CACHE_DIR",
                "HOME",
                "MYPY_CACHE_DIR",
                "PYTHONPYCACHEPREFIX",
                "TMPDIR",
                "XDG_CACHE_HOME",
            ],
            "forbidden": [],
            "path_topology": subject.MAIN_SUITE_PATH_TOPOLOGY,
            "required": {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        },
        "wrapper_binding": {"path": "/bound/wrapper"},
    }
    pycache_binding = subject._empty_private_directory_binding(
        pycache,
        label="test main-suite pycache",
    )
    return policy, dict(sorted(environment.items())), pycache_binding


def _main_suite_result(
    *,
    repo_root: Path,
    challenge_sha256: str,
    stdout: bytes,
) -> dict[str, object]:
    stderr = b""
    attestation = b"three-bound-attestation-frames"
    policy, environment, pycache_binding = _main_suite_contract_inputs(repo_root)
    bytecode_policy = {
        "dont_write_bytecode": True,
        "pycache_prefix": environment["PYTHONPYCACHEPREFIX"],
    }
    frames = [
        {
            "payload": {
                "challenge_binding_sha256": challenge_sha256,
                "environment": environment,
                "frame": "pre_import",
                "pid": 101,
                "ppid": 100,
                "runtime": {"bytecode_policy": bytecode_policy},
            },
            "payload_sha256": "1" * 64,
            "payload_size_bytes": 1,
            "phase": 1,
        },
        {
            "payload": {
                "challenge_binding_sha256": challenge_sha256,
                "frame": "pre_collection",
                "pid": 101,
                "ppid": 100,
                "runtime": {"bytecode_policy": bytecode_policy},
            },
            "payload_sha256": "2" * 64,
            "payload_size_bytes": 1,
            "phase": 2,
        },
        {
            "payload": {
                "challenge_binding_sha256": challenge_sha256,
                "final_loaded_modules": {},
                "frame": "terminal_complete",
                "pid": 101,
                "ppid": 100,
                "pytest_exit_code": 0,
            },
            "payload_sha256": "3" * 64,
            "payload_size_bytes": 1,
            "phase": 3,
        },
    ]
    tail = (
        struct.pack(">Q", len(stdout))
        + stdout
        + struct.pack(">Q", len(stderr))
        + stderr
        + struct.pack(">Q", len(attestation))
        + attestation
    )
    receipt = subject._seal(
        {
            "accepted": True,
            "attestations": frames,
            "authority": False,
            "challenge_binding": {
                "kind": "SKIP_SOURCE_STATE",
                "sha256": challenge_sha256,
            },
            "claims": {
                "exit_code": 0,
                "final_audit_completed": True,
                "final_audit_enforced": True,
                "kernel_egress_attested": False,
                "network_unreachability_proven": False,
                "offline_policy_enforced": True,
                "signal": None,
            },
            "command": {
                "argv": [
                    "/bound/python",
                    "-I",
                    "-S",
                    "-B",
                    "-X",
                    f"pycache_prefix={environment['PYTHONPYCACHEPREFIX']}",
                    "/bound/wrapper",
                    str(repo_root / subject.MAIN_SUITE_POLICY_PATH),
                    _policy_bindings(repo_root)["policy_binding"]["sha256"],
                    "--",
                    *subject.MAIN_SUITE_PYTEST_ARGS,
                ],
                "cwd": str(repo_root),
                "environment": environment,
            },
            "external_after": {
                "pycache_prefix": pycache_binding,
                "snapshot_sha256": "4" * 64,
            },
            "external_before": {
                "pycache_prefix": pycache_binding,
                "snapshot_sha256": "4" * 64,
            },
            "failure_codes": [],
            "failures": [],
            "finalization": {
                "cleanup": {"attempted": True, "status": "PASSED"},
                "external_after": {
                    "attempted": True,
                    "equal": True,
                    "status": "PASSED",
                },
            },
            "framing": subject.MAIN_SUITE_FRAMING,
            "limitations": list(subject.NORMATIVE_LIMITATIONS),
            "outcome": "PASSED",
            **_policy_bindings(repo_root),
            "protocol_version": subject.PROTOCOL_VERSION,
            "schema_id": subject.MAIN_SUITE_RECEIPT_SCHEMA_ID,
            "streams": {
                "attestation": {
                    "offset_bytes": 8 + len(stdout) + 8 + len(stderr) + 8,
                    "sha256": hashlib.sha256(attestation).hexdigest(),
                    "size_bytes": len(attestation),
                },
                "stderr": {
                    "offset_bytes": 8 + len(stdout) + 8,
                    "sha256": hashlib.sha256(stderr).hexdigest(),
                    "size_bytes": len(stderr),
                },
                "stdout": {
                    "offset_bytes": 8,
                    "sha256": hashlib.sha256(stdout).hexdigest(),
                    "size_bytes": len(stdout),
                },
                "tail_sha256": hashlib.sha256(tail).hexdigest(),
                "tail_size_bytes": len(tail),
            },
            "timing": {"phase1_elapsed_ms": 1, "phase2_elapsed_ms": 2},
            "version": subject.MAIN_SUITE_RECEIPT_VERSION,
        }
    )
    return {
        "attestation": attestation,
        "raw": (
            subject.MAIN_SUITE_RECEIPT_PREFIX + subject._canonical_bytes(receipt) + b"\n" + tail
        ),
        "receipt": receipt,
        "stderr": stderr,
        "stdout": stdout,
    }


def _replace_main_suite_receipt(
    result: dict[str, object],
    receipt: dict[str, object],
) -> dict[str, object]:
    candidate = dict(receipt)
    candidate.pop("semantic_sha256", None)
    sealed = subject._seal(candidate)
    raw = result["raw"]
    assert type(raw) is bytes
    tail = raw.split(b"\n", 1)[1]
    changed = dict(result)
    changed["receipt"] = sealed
    changed["raw"] = (
        subject.MAIN_SUITE_RECEIPT_PREFIX + subject._canonical_bytes(sealed) + b"\n" + tail
    )
    return changed


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path, dict[str, str]]] = []

    def __call__(
        self,
        argv: list[str],
        cwd: Path,
        environment: dict[str, str],
    ) -> tuple[int, bytes, bytes]:
        self.calls.append((list(argv), cwd, dict(environment)))
        if len(self.calls) == 1:
            native = Path(environment["UV_PROJECT_ENVIRONMENT"])
            (native / "bin").mkdir(parents=True)
            (native / "bin" / "python").symlink_to(subject.BASE_PYTHON_PATH)
            metadata = (
                native
                / "lib"
                / "python3.13"
                / "site-packages"
                / "pytest-9.0.2.dist-info"
                / "METADATA"
            )
            metadata.parent.mkdir(parents=True)
            metadata.write_text(
                "Metadata-Version: 2.4\nName: pytest\nVersion: 9.0.2\n",
                encoding="utf-8",
            )
            return 0, b"sync complete\n", b""
        raise AssertionError("only the locked sync may use the command runner")


class _FakeMainSuiteRunner:
    def __init__(self, *, stdout: bytes | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.stdout = stdout or _pytest_output()

    def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return _main_suite_result(
            repo_root=kwargs["repo_root"],  # type: ignore[arg-type]
            challenge_sha256=kwargs["challenge_binding_sha256"],  # type: ignore[arg-type]
            stdout=self.stdout,
        )


class _Case(TypedDict):
    bundle: Path
    output: Path
    repo: Path
    main_runner: _FakeMainSuiteRunner
    runner: _FakeRunner
    work: Path


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Case:
    repo = _repo(tmp_path)
    authority = tmp_path / "authority"
    authority.mkdir()
    monkeypatch.setattr(subject, "AUTHORITY_REPO_ROOT", authority)
    monkeypatch.setattr(subject, "_validate_runtime_invocation", lambda: None)
    monkeypatch.setattr(subject, "_parent_runtime_binding", _parent_runtime_binding)
    monkeypatch.setattr(subject, "_sample_pip_status", _pip_status)
    monkeypatch.setattr(subject, "_checked_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subject, "_main_suite_policy_bindings", _policy_bindings)
    monkeypatch.setattr(
        subject,
        "_load_main_suite_module",
        lambda repo_root: (object(), {"path": subject.MAIN_SUITE_HARNESS_PATH}),
    )
    monkeypatch.setattr(
        subject,
        "_main_suite_policy",
        lambda repo_root, _harness: (
            _main_suite_contract_inputs(repo_root)[0],
            _policy_bindings(repo_root),
        ),
    )
    bundle = _private_dir(tmp_path / "baseline")
    work = _private_dir(tmp_path / "work")
    output = bundle / "skip-baseline.json"
    return {
        "bundle": bundle,
        "output": output,
        "repo": repo,
        "main_runner": _FakeMainSuiteRunner(),
        "runner": _FakeRunner(),
        "work": work,
    }


def _candidate_policy(paths: list[str]) -> dict[str, object]:
    return {
        "module_policy": {
            "candidate_content_binding": "OUTER_SOURCE_STATE",
            "candidate_module_source_paths": paths,
        }
    }


def test_candidate_sources_accept_tracked_untracked_and_ignored_package(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text(
        "quant_investor/generated.py\nscripts/ignored.py\n",
        encoding="utf-8",
    )
    _run(["git", "add", ".gitignore"], cwd=repo)
    _run(["git", "commit", "-q", "-m", "ignore generated sources"], cwd=repo)
    (repo / "candidate_untracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "quant_investor" / "generated.py").write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )
    snapshot = subject._git_snapshot(repo)
    package = subject._sample_package_source_superset(repo)

    subject._validate_candidate_source_membership(
        repo_root=repo,
        policy=_candidate_policy(
            [
                "candidate_untracked.py",
                "quant_investor/__init__.py",
                "quant_investor/generated.py",
            ]
        ),
        source_state=subject._public_source_state(snapshot),
        package_binding=package,
    )


@pytest.mark.parametrize(
    ("relative", "kind"),
    [
        ("missing.py", "missing"),
        ("scripts/ignored.py", "ignored"),
        ("ignored_link.py", "symlink"),
    ],
)
def test_candidate_sources_reject_unsealed_or_nonconcrete_paths(
    tmp_path: Path,
    relative: str,
    kind: str,
) -> None:
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text(
        "scripts/ignored.py\nignored_link.py\n",
        encoding="utf-8",
    )
    _run(["git", "add", ".gitignore"], cwd=repo)
    _run(["git", "commit", "-q", "-m", "ignore rejected sources"], cwd=repo)
    if kind == "ignored":
        path = repo / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("VALUE = 1\n", encoding="utf-8")
    elif kind == "symlink":
        (repo / relative).symlink_to(repo / "README.md")
    snapshot = subject._git_snapshot(repo)
    package = subject._sample_package_source_superset(repo)

    with pytest.raises(subject.SkipBaselineError, match="candidate source"):
        subject._validate_candidate_source_membership(
            repo_root=repo,
            policy=_candidate_policy([relative]),
            source_state=subject._public_source_state(snapshot),
            package_binding=package,
        )


def test_builds_closed_canonical_frozen_baseline_exact_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    report = subject.build_skip_baseline(
        repo_root=case["repo"],
        bundle_root=case["bundle"],
        work_root=case["work"],
        output_json=case["output"],
        command_runner=case["runner"],
        main_suite_runner=case["main_runner"],
    )

    assert set(report) == subject.ROOT_KEYS
    assert report["status"] == "FROZEN"
    assert report["accepted"] is True
    assert report["authority"] is False
    assert report["parent_runtime_binding"] == _parent_runtime_binding()
    assert report["pip_status_before"] == report["pip_status_after"] == _pip_status()
    assert report["expected_skip_count"] == report["observed_skip_count"] == 42
    assert report["entries"] == [
        {
            "count": 42,
            "line": 1,
            "path": "tests/unit/test_optional.py",
            "reason": "optional dependency unavailable",
        }
    ]
    assert set(report["claims"]) == subject.CLAIMS_KEYS
    assert report["claims"]["skipped"] == 42
    assert report["claims"]["failed"] == report["claims"]["errors"] == 0
    assert [command["ordinal"] for command in report["commands"]] == [1, 2]
    assert report["commands"][0]["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert report["commands"][1]["environment"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert len(case["runner"].calls) == 1
    assert len(case["main_runner"].calls) == 1
    assert report["challenge_binding"] == {
        "kind": "SKIP_SOURCE_STATE",
        "sha256": report["source_state"]["source_state_sha256"],
    }
    assert report["main_suite_receipt"]["challenge_binding"] == report["challenge_binding"]
    expected_main = _main_suite_result(
        repo_root=case["repo"],
        challenge_sha256=report["challenge_binding"]["sha256"],
        stdout=_pytest_output(),
    )
    expected_raw = expected_main["raw"]
    assert type(expected_raw) is bytes
    assert report["main_suite_raw_binding"] == {
        "sha256": hashlib.sha256(expected_raw).hexdigest(),
        "size_bytes": len(expected_raw),
    }

    raw = case["output"].read_bytes()
    assert raw == subject._canonical_resource_bytes(report)
    assert stat.S_IMODE(case["output"].stat().st_mode) == 0o600
    assert report["semantic_sha256"] == subject._semantic_sha256(report)
    subject.validate_skip_baseline(report, repo_root=case["repo"])

    schema = json.loads(
        (
            Path(__file__).parents[2]
            / "scripts"
            / "schemas"
            / "v17_phase0_skip_baseline.v2.schema.json"
        ).read_text(encoding="utf-8")
    )
    preflight_packaged_schema(schema)
    validate_instance_against_schema(report, schema)
    macos_report = copy.deepcopy(report)
    macos_report["commands"][1]["environment"]["__CF_USER_TEXT_ENCODING"] = "0x1F5:0x0:0x0"
    validate_instance_against_schema(macos_report, schema)

    with pytest.raises(subject.SkipBaselineError, match="already exists"):
        subject.write_skip_baseline_exact_once(
            case["output"],
            report,
            bundle_root=case["bundle"],
        )
    assert case["output"].read_bytes() == raw


def test_skip_main_suite_schema_precedes_exact_semantic_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    challenge_sha256 = "a" * 64
    result = _main_suite_result(
        repo_root=repo,
        challenge_sha256=challenge_sha256,
        stdout=_pytest_output(),
    )
    order: list[str] = []

    def checked_schema(value: object, **_kwargs: object) -> None:
        assert value == result["receipt"]
        order.append("schema")

    monkeypatch.setattr(subject, "_checked_schema", checked_schema)
    policy, environment, pycache_binding = _main_suite_contract_inputs(repo)
    assert (
        subject._validate_main_suite_contract_result(
            result,
            repo_root=repo,
            policy=policy,
            policy_bindings=_policy_bindings(repo),
            expected_environment=environment,
            expected_pycache_binding=pycache_binding,
            challenge_binding_kind="SKIP_SOURCE_STATE",
            challenge_binding_sha256=challenge_sha256,
        )
        == result
    )
    assert order == ["schema"]

    receipt = json.loads(json.dumps(result["receipt"]))
    receipt["policy_schema_binding"]["sha256"] = "0" * 64
    tampered = _replace_main_suite_receipt(result, receipt)
    order.clear()

    def checked_tampered(value: object, **_kwargs: object) -> None:
        assert value == tampered["receipt"]
        order.append("schema")

    monkeypatch.setattr(subject, "_checked_schema", checked_tampered)
    with pytest.raises(subject.SkipBaselineError, match="policy receipt binding"):
        subject._validate_main_suite_contract_result(
            tampered,
            repo_root=repo,
            policy=policy,
            policy_bindings=_policy_bindings(repo),
            expected_environment=environment,
            expected_pycache_binding=pycache_binding,
            challenge_binding_kind="SKIP_SOURCE_STATE",
            challenge_binding_sha256=challenge_sha256,
        )
    assert order == ["schema"]


@pytest.mark.parametrize(
    ("pytest_code", "output", "message"),
    [
        (1, _pytest_output(extra="1 failed, "), "does not satisfy"),
        (0, _pytest_output(skipped=41), "does not satisfy"),
        (
            0,
            (
                "SKIPPED [42] tests/unit/test_optional.py:1: optional dependency unavailable\n"
                "9 passed, 42 skipped, 1 xpassed in 0.21s\n"
            ).encode(),
            "does not satisfy",
        ),
    ],
)
def test_refuses_failed_or_nonexact_pytest_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pytest_code: int,
    output: bytes,
    message: str,
) -> None:
    case = _case(tmp_path, monkeypatch)
    runner = _FakeRunner()
    main_runner = _FakeMainSuiteRunner(stdout=output)
    with pytest.raises(subject.SkipBaselineError, match=message):
        subject.build_skip_baseline(
            repo_root=case["repo"],
            bundle_root=case["bundle"],
            work_root=case["work"],
            output_json=case["output"],
            command_runner=runner,
            main_suite_runner=main_runner,
        )
    assert not case["output"].exists()


def test_refuses_nonempty_fresh_root_before_any_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    (case["bundle"] / "old").write_text("not resumable\n", encoding="utf-8")
    with pytest.raises(subject.SkipBaselineError, match="fresh and empty"):
        subject.build_skip_baseline(
            repo_root=case["repo"],
            bundle_root=case["bundle"],
            work_root=case["work"],
            output_json=case["output"],
            command_runner=case["runner"],
            main_suite_runner=case["main_runner"],
        )
    assert case["runner"].calls == []


def test_refuses_nonprivate_or_symlink_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    case["work"].chmod(0o755)
    with pytest.raises(subject.SkipBaselineError, match="mode 0700"):
        subject.build_skip_baseline(
            repo_root=case["repo"],
            bundle_root=case["bundle"],
            work_root=case["work"],
            output_json=case["output"],
            command_runner=case["runner"],
            main_suite_runner=case["main_runner"],
        )

    case["work"].chmod(0o700)
    link = tmp_path / "work-link"
    link.symlink_to(case["work"], target_is_directory=True)
    with pytest.raises(subject.SkipBaselineError, match="symlink"):
        subject.build_skip_baseline(
            repo_root=case["repo"],
            bundle_root=case["bundle"],
            work_root=link,
            output_json=case["output"],
            command_runner=case["runner"],
            main_suite_runner=case["main_runner"],
        )


def test_semantic_and_transcript_tampering_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    report = subject.build_skip_baseline(
        repo_root=case["repo"],
        bundle_root=case["bundle"],
        work_root=case["work"],
        output_json=case["output"],
        command_runner=case["runner"],
        main_suite_runner=case["main_runner"],
    )
    tampered = json.loads(json.dumps(report))
    tampered["commands"][1]["stdout"]["bytes_base64"] = ""
    tampered["semantic_sha256"] = subject._semantic_sha256(tampered)
    with pytest.raises(subject.SkipBaselineError, match="binding mismatch"):
        subject.validate_skip_baseline(tampered)
