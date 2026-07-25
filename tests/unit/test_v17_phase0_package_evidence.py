from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
from types import ModuleType
from typing import Any, Mapping, cast

import pytest


def _load_subject() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "v17_phase0_package_evidence.py"
    spec = importlib.util.spec_from_file_location("v17_phase0_package_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = _load_subject()
BASE_COMMIT = "a" * 40
SOURCE_BINDING = {
    "base_commit": BASE_COMMIT,
    "binary_diff_sha256": "b" * 64,
    "porcelain_sha256": "c" * 64,
    "source_state_sha256": "d" * 64,
    "untracked_inventory_sha256": "e" * 64,
}
SDIST_RAW = b"sdist bytes\n"
WHEEL_RAW = b"wheel bytes\n"
WRAPPER_RAW = b"#!/tmp/install/bin/python\n"


def _physical_superset() -> dict[str, object]:
    files = {
        "README.md": b"readme\n",
        "pyproject.toml": b"[project]\nname='quant-investor'\nversion='17.0.0'\n",
        "quant_investor/__init__.py": b'"""package"""\n',
        "quant_investor/v17_v2_contract/resources/main_suite_runtime_policy.v1.json": (
            b'{"authority":false}\n'
        ),
        "quant_investor/v17_v2_contract/schemas/main_suite_runtime_policy.v1.schema.json": (
            b'{"type":"object"}\n'
        ),
        "requirements.txt": b"pytest==9.0.2\n",
    }
    rows: list[dict[str, object]] = [
        {
            "kind": "file",
            "mode": 0o644,
            "path": path,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        for path, raw in files.items()
    ]
    rows.append(
        {
            "kind": "directory",
            "mode": 0o755,
            "path": "quant_investor",
            "sha256": None,
            "size_bytes": 0,
        }
    )
    rows.sort(key=lambda row: str(row["path"]))
    return {
        "row_count": len(rows),
        "rows": rows,
        "sha256": hashlib.sha256(subject._canonical_bytes(rows)).hexdigest(),
    }


def _hatch_namespace() -> dict[str, object]:
    physical = _physical_superset()
    physical_rows = cast(list[dict[str, object]], physical["rows"])
    file_rows = {str(row["path"]): row for row in physical_rows if row["kind"] == "file"}
    rows: list[dict[str, object]] = []
    for target, paths in (
        ("sdist", sorted(file_rows)),
        ("wheel", sorted(path for path in file_rows if path.startswith("quant_investor/"))),
    ):
        for path in paths:
            source = file_rows[path]
            rows.append(
                {
                    "distribution_path": path,
                    "mode": source["mode"],
                    "sha256": source["sha256"],
                    "size_bytes": source["size_bytes"],
                    "source_path": path,
                    "target": target,
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["target"]),
            str(row["distribution_path"]),
            str(row["source_path"]),
        )
    )
    wheel_inventory = {
        path: {
            "sha256": file_rows[path]["sha256"],
            "size": file_rows[path]["size_bytes"],
        }
        for path in sorted(file_rows)
        if path.startswith("quant_investor/")
    }
    wheel_projection = hashlib.sha256(subject._canonical_bytes(wheel_inventory)).hexdigest()
    return {
        "row_count": len(rows),
        "rows": rows,
        "sha256": hashlib.sha256(subject._canonical_bytes(rows)).hexdigest(),
        "wheel_projection_sha256": wheel_projection,
    }


def _selector_payload(repo: Path, build_venv: Path) -> dict[str, object]:
    module_bindings = {
        name: {
            "path": str(
                (
                    repo / "quant_investor" / "v17_v2_contract" / "package_parity.py"
                    if name == "package_parity"
                    else build_venv / "site" / f"{name.replace('.', '_')}.py"
                )
            ),
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
            "size_bytes": len(name),
        }
        for name in (
            "hatchling.build",
            "hatchling.builders.sdist",
            "hatchling.builders.wheel",
            "package_parity",
        )
    }
    return {
        "hatch_source_namespace": _hatch_namespace(),
        "hatchling_version": "1.31.0",
        "package_source_superset": _physical_superset(),
        "selector_modules": module_bindings,
    }


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    path.chmod(0o700)
    return path


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _source_binding_file(path: Path, binding: Mapping[str, str] = SOURCE_BINDING) -> Path:
    path.write_bytes(subject._canonical_resource_bytes(dict(binding)))
    path.chmod(0o600)
    return path


def _repo(path: Path) -> Path:
    repo = path / "repo"
    repo.parent.mkdir(parents=True, exist_ok=True)
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "quant_investor" / "v17_v2_contract").mkdir(parents=True)
    return repo


def _parity_payload() -> dict[str, object]:
    inventory_sha256 = str(_hatch_namespace()["wheel_projection_sha256"])
    physical_rows = cast(list[dict[str, object]], _physical_superset()["rows"])
    package_file_count = sum(
        1
        for row in physical_rows
        if row["kind"] == "file" and str(row["path"]).startswith("quant_investor/")
    )
    return {
        "accepted": True,
        "installed_provenance": {
            "direct_url": {
                "archive_info_sha256": hashlib.sha256(WHEEL_RAW).hexdigest(),
                "editable": False,
                "present": True,
                "sha256": "1" * 64,
                "url": "file:///tmp/wheel/quant_investor-17.0.0-py3-none-any.whl",
            },
            "dist_info_file_sha256s": {
                "INSTALLER": "5" * 64,
                "METADATA": "6" * 64,
                "RECORD": "7" * 64,
                "REQUESTED": "b" * 64,
                "WHEEL": "8" * 64,
                "direct_url.json": "9" * 64,
                "entry_points.txt": "a" * 64,
                "licenses/LICENSE": "c" * 64,
            },
            "dist_info_path": "/tmp/env/site/quant_investor-17.0.0.dist-info",
            "environment_root": "/tmp/env",
            "installed_package_root": "/tmp/env/site/quant_investor",
            "metadata": {"name": "quant-investor", "version": "17.0.0"},
            "non_editable_verified": True,
            "record": {
                "dist_info_file_count": 8,
                "file_count": package_file_count + 8,
                "package_file_count": package_file_count,
                "record_sha256": "2" * 64,
            },
            "site_packages_root": "/tmp/env/site",
        },
        "package_file_count": package_file_count,
        "package_inventory": {
            "file_count": package_file_count,
            "sha256": inventory_sha256,
        },
        "package_inventory_sha256": inventory_sha256,
        "sdist_sha256": hashlib.sha256(SDIST_RAW).hexdigest(),
        "source_equals_sdist_equals_wheel_equals_installed": True,
        "wheel_provenance": {
            "dist_info_file_sha256s": {
                "METADATA": "6" * 64,
                "RECORD": "7" * 64,
                "WHEEL": "8" * 64,
                "entry_points.txt": "a" * 64,
                "licenses/LICENSE": "c" * 64,
            },
            "dist_info_root": "quant_investor-17.0.0.dist-info",
            "metadata": {"name": "quant-investor", "version": "17.0.0"},
            "record": {
                "file_count": package_file_count + 5,
                "record_sha256": "4" * 64,
            },
        },
        "wheel_sha256": hashlib.sha256(WHEEL_RAW).hexdigest(),
    }


class FakeRunner:
    def __init__(
        self,
        *,
        backend_inventory: list[dict[str, str]] | None = None,
        backend_packages: Mapping[str, str] | None = None,
        bad_backend: bool = False,
        fail_at: int | None = None,
        pip_version: str = subject.EXPECTED_PIP_VERSION,
        uv_output: str = subject.EXPECTED_UV_VERSION_OUTPUT,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.backend_packages = dict(
            subject.EXPECTED_BUILD_BACKEND_PACKAGES
            if backend_packages is None
            else backend_packages
        )
        self.backend_inventory = [
            dict(item)
            for item in (
                subject.EXPECTED_BUILD_BACKEND_INVENTORY
                if backend_inventory is None
                else backend_inventory
            )
        ]
        self.fail_at = fail_at
        self.bad_backend = bad_backend
        self.pip_version = pip_version
        self.uv_output = uv_output

    def __call__(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        shell: bool,
        check: bool,
        stdout: int,
        stderr: int,
    ) -> subprocess.CompletedProcess[bytes]:
        assert shell is False
        assert check is False
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.PIPE
        self.calls.append({"argv": argv, "cwd": cwd, "env": env})
        index = len(self.calls)
        if self.fail_at == index:
            return subprocess.CompletedProcess(argv, 9, b"", b"forced failure")
        out = self._stdout(argv)
        return subprocess.CompletedProcess(argv, 0, out, b"")

    def _stdout(self, argv: list[str]) -> bytes:
        if argv[1:6] == ["-I", "-S", "-B", "-c", subject._python_probe_code()]:
            raw = Path(argv[0]).read_bytes()
            payload: dict[str, object] = {
                "executable": argv[0],
                "implementation": "cpython",
                "pip_absence": {
                    "find_spec_present": False,
                    "loaded_modules": [],
                    "observation_scope": subject.BASE_RUNTIME_PIP_OBSERVATION_SCOPE,
                    "site_sys_path_entries": [],
                },
                "realpath": str(Path(argv[0]).resolve()),
                "runtime_flags": subject.EXPECTED_BASE_RUNTIME_FLAGS,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "version": "3.13.7",
                "version_info": [3, 13, 7],
            }
            return subject._canonical_bytes(payload)
        if argv[1:] == ["--version"]:
            return f"{self.uv_output}\n".encode()
        if argv[1:4] == ["-I", "-c", subject._backend_probe_code()]:
            root = Path(argv[0]).parents[1]
            backend_payload: dict[str, object] = {
                "backend_file": str(
                    root / "lib" / "python3.13" / "site-packages" / "hatchling" / "build.py"
                ),
                "backend_module": "wrong.build" if self.bad_backend else "hatchling.build",
                "hatchling_version": "1.31.0",
                "package_inventory": self.backend_inventory,
                "package_versions": self.backend_packages,
                "pip_absence": {
                    "distribution_names": [],
                    "find_spec_present": False,
                    "package_paths": [],
                    "wrapper_paths": [],
                },
                "unnamed_distribution_count": 0,
            }
            return subject._canonical_bytes(backend_payload)
        if argv[1:4] == ["-I", "-c", subject._hatch_selector_probe_code()]:
            return subject._canonical_bytes(
                _selector_payload(Path(argv[4]), Path(argv[0]).parents[1])
            )
        if argv[1:4] == ["-I", "-c", subject._ensurepip_bundle_probe_code()]:
            bundle_path = Path(argv[0]).parents[3] / "base-stdlib" / subject.EXPECTED_PIP_WHEEL_NAME
            payload = {
                "ensurepip_version": subject.EXPECTED_PIP_VERSION,
                "match_count": 1,
                "wheel": {
                    "is_symlink": False,
                    "mode": "0644",
                    "name": subject.EXPECTED_PIP_WHEEL_NAME,
                    "nlink": 1,
                    "path": str(bundle_path),
                    "realpath": str(bundle_path),
                    "sha256": subject.EXPECTED_PIP_WHEEL_SHA256,
                    "size_bytes": subject.EXPECTED_PIP_WHEEL_SIZE,
                    "stable": True,
                },
            }
            return subject._canonical_bytes(payload)
        if argv[1:4] == ["-I", "-c", subject._install_inventory_probe_code()]:
            return subject._canonical_bytes(self._install_inventory(argv[0], include_project=False))
        if argv[1:4] == ["-I", "-c", subject._installed_paths_probe_code()]:
            root = Path(argv[0]).parents[1]
            package = root / "lib" / "python3.13" / "site-packages" / "quant_investor"
            dist_info = package.parent / "quant_investor-17.0.0.dist-info"
            package.mkdir(parents=True, exist_ok=True)
            dist_info.mkdir(parents=True, exist_ok=True)
            payload = self._install_inventory(argv[0], include_project=True)
            return subject._canonical_bytes(payload)
        if argv[1:3] == ["venv", "--python"]:
            root = Path(argv[-1])
            _executable(root / "bin" / "python")
            return b"created venv"
        if argv[1:4] == ["pip", "install", "--python"]:
            return b"installed hatchling"
        if argv[1:3] == ["build", "--sdist"]:
            out_dir = Path(argv[argv.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "quant_investor-17.0.0.tar.gz").write_bytes(SDIST_RAW)
            return b"built sdist"
        if argv[1:3] == ["build", "--wheel"]:
            out_dir = Path(argv[argv.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "quant_investor-17.0.0-py3-none-any.whl").write_bytes(WHEEL_RAW)
            return b"built wheel"
        if argv[1:5] == ["-I", "-m", "ensurepip", "--upgrade"]:
            return b"ensurepip"
        if argv[1:5] == ["-I", "-m", "pip", "--version"]:
            root = Path(argv[0]).parents[1]
            return (
                f"pip {self.pip_version} from "
                f"{root}/lib/python3.13/site-packages/pip (python 3.13)\n"
            ).encode()
        if argv[1:5] == ["-I", "-m", "pip", "install"]:
            return b"pip installed wheel"
        if argv[1].endswith("package_parity.py"):
            return subject._canonical_bytes(_parity_payload())
        raise AssertionError(f"unexpected argv: {argv}")

    @staticmethod
    def _install_inventory(python: str, *, include_project: bool) -> dict[str, object]:
        root = Path(python).parents[1]
        site = root / "lib" / "python3.13" / "site-packages"
        wrappers = [
            {
                "is_symlink": False,
                "mode": "0700",
                "name": name,
                "path": str(root / "bin" / name),
                "sha256": hashlib.sha256(WRAPPER_RAW).hexdigest(),
                "size_bytes": len(WRAPPER_RAW),
            }
            for name in subject.EXPECTED_PIP_WRAPPERS
        ]
        inventory = [{"name": "pip", "version": subject.EXPECTED_PIP_VERSION}]
        payload: dict[str, object] = {
            "distribution_inventory": inventory,
            "pip_find_spec_present": True,
            "pip_package_paths": [
                str(site / "pip"),
                str(site / f"pip-{subject.EXPECTED_PIP_VERSION}.dist-info"),
            ],
            "pip_wrappers": wrappers,
            "plain_pip_absent": True,
            "site_packages_root": str(site),
        }
        if include_project:
            inventory.append({"name": "quant-investor", "version": "17.0.0"})
            payload["installed_dist_info"] = str(site / "quant_investor-17.0.0.dist-info")
            payload["installed_package_root"] = str(site / "quant_investor")
        return payload


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo = _repo(tmp_path)
    private = _private_dir(tmp_path / "private")
    cache = _private_dir(tmp_path / "uv-cache")
    base_python = _executable(tmp_path / "python" / "python3.13")
    uv = _executable(tmp_path / "uv-bin" / "uv")
    binding = _source_binding_file(private / "source_binding.json")
    session_path = private / "session.json"
    session_raw = subject._canonical_resource_bytes({"session": "synthetic"})
    session_path.write_bytes(session_raw)
    session_path.chmod(0o600)
    samples = [SOURCE_BINDING, SOURCE_BINDING]

    def sample(_repo: Path, _base_commit: str) -> dict[str, str]:
        return dict(samples.pop(0))

    monkeypatch.setattr(subject, "_sample_source_binding", sample)
    monkeypatch.setattr(subject, "AUTHORITY_REPO_ROOT", tmp_path / "authority")
    physical = _physical_superset()
    monkeypatch.setattr(
        subject,
        "_sample_physical_superset",
        lambda _repo: json.loads(json.dumps(physical)),
    )
    monkeypatch.setattr(subject, "_validate_checked_schema", lambda *_args, **_kwargs: {})
    protected = subject._sample_protected_roots(repo)
    cache_binding = subject._uv_cache_binding(cache)
    python_raw = base_python.read_bytes()
    uv_raw = uv.read_bytes()
    toolchain = {
        "base_python": {
            "executable": True,
            "implementation": "cpython",
            "lexical_path": str(base_python),
            "mode": "0700",
            "realpath": str(base_python),
            "sha256": hashlib.sha256(python_raw).hexdigest(),
            "size_bytes": len(python_raw),
            "version": subject.EXPECTED_PYTHON_VERSION,
            "version_info": list(subject.EXPECTED_PYTHON_VERSION_INFO),
        },
        "pip_scope": {
            "allowed_wrappers": list(subject.EXPECTED_PIP_WRAPPERS),
            "build_pip_absent": True,
            "bundled_wheel": {
                "name": subject.EXPECTED_PIP_WHEEL_NAME,
                "sha256": subject.EXPECTED_PIP_WHEEL_SHA256,
                "size_bytes": subject.EXPECTED_PIP_WHEEL_SIZE,
            },
            "ensurepip_argv_suffix": ["-I", "-m", "ensurepip", "--upgrade"],
            "environment_scope": "PACKAGE_INSTALL_ENV_ONLY",
            "native_pip_absent": True,
            "plain_pip_absent": True,
            "version": subject.EXPECTED_PIP_VERSION,
        },
        "uv": {
            "executable": True,
            "lexical_path": str(uv),
            "mode": "0700",
            "output": subject.EXPECTED_UV_VERSION_OUTPUT,
            "realpath": str(uv),
            "sha256": hashlib.sha256(uv_raw).hexdigest(),
            "size_bytes": len(uv_raw),
            "version": subject.EXPECTED_UV_VERSION,
        },
        "uv_cache": cache_binding,
    }
    session = {
        "limitations": list(subject.LIMITATIONS),
        "package_source_superset": {
            "row_count": physical["row_count"],
            "sha256": physical["sha256"],
        },
        "protected_roots": protected,
        "source_binding": dict(SOURCE_BINDING),
        "toolchain_binding": toolchain,
        "uv_cache_binding": cache_binding,
    }
    session_binding = {
        "path": str(session_path),
        "semantic_sha256": "f" * 64,
        "session_id": "phase0-test-session",
        "sha256": hashlib.sha256(session_raw).hexdigest(),
        "size_bytes": len(session_raw),
    }
    monkeypatch.setattr(
        subject,
        "_load_session_manifest",
        lambda *_args, **_kwargs: (
            dict(session),
            dict(session_binding),
            session_raw,
            session_path,
        ),
    )
    return {
        "base_python": base_python,
        "binding": binding,
        "cache": cache,
        "private": private,
        "repo": repo,
        "samples": samples,
        "session": session,
        "session_path": session_path,
        "uv": uv,
        "work_root": tmp_path / "work",
    }


def _build(case: dict[str, Any]) -> dict[str, Any]:
    return subject.build_package_evidence(
        repo_root=case["repo"],
        expected_base_commit=BASE_COMMIT,
        session_manifest=case["session_path"],
        expected_source_binding_json=case["binding"],
        base_python=case["base_python"],
        uv_bin=case["uv"],
        uv_cache=case["cache"],
        work_root=case["work_root"],
    )


def test_package_session_keys_match_session_v2_schema() -> None:
    schema = json.loads(
        (
            Path(__file__).parents[2] / "scripts" / "schemas" / "v17_phase0_session.v2.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert subject.SESSION_KEYS == set(schema["required"])


def test_positive_pipeline_captures_order_offline_flags_hashes_and_semantic_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    case["cache"].chmod(0o755)
    cache_binding = subject._uv_cache_binding(case["cache"])
    case["session"]["uv_cache_binding"] = cache_binding
    case["session"]["toolchain_binding"]["uv_cache"] = cache_binding
    runner = FakeRunner()
    monkeypatch.setattr(subject.subprocess, "run", runner)
    report = _build(case)

    assert report["accepted"] is True
    assert report["phase0_gate_roles"] == list(subject.EXPECTED_GATE_ROLES)
    assert report["source_binding"] == SOURCE_BINDING
    assert report["semantic_sha256"] == subject._semantic_sha256(report)
    provenance = report["build_install_provenance"]
    assert provenance["artifact_bindings"]["sdist"]["sha256"] == report["sdist_sha256"]
    assert provenance["artifact_bindings"]["wheel"]["sha256"] == report["wheel_sha256"]
    assert provenance["build_backend"]["backend_module"] == "hatchling.build"
    assert provenance["build_backend"]["hatchling_version"] == "1.31.0"
    assert (
        provenance["build_backend"]["package_versions"] == subject.EXPECTED_BUILD_BACKEND_PACKAGES
    )
    assert (
        provenance["build_backend"]["package_inventory"] == subject.EXPECTED_BUILD_BACKEND_INVENTORY
    )
    assert provenance["base_interpreter"]["version_info"][:2] == [3, 13]
    assert provenance["uv_runtime"]["version"] == subject.EXPECTED_UV_VERSION
    assert provenance["pip_runtime"]["version"] == subject.EXPECTED_PIP_VERSION
    assert provenance["source_binding_artifact"] == {
        "path": str(case["binding"]),
        "sha256": hashlib.sha256(case["binding"].read_bytes()).hexdigest(),
        "size_bytes": case["binding"].stat().st_size,
    }
    assert provenance["command_roles"] == list(subject.COMMAND_ROLES)
    assert [command["role"] for command in provenance["commands"]] == list(subject.COMMAND_ROLES)
    hatch_rows = provenance["hatch_source_namespace_session"]["after_rows"]
    wheel_paths = {row["distribution_path"] for row in hatch_rows if row["target"] == "wheel"}
    assert {
        "quant_investor/v17_v2_contract/resources/main_suite_runtime_policy.v1.json",
        "quant_investor/v17_v2_contract/schemas/main_suite_runtime_policy.v1.schema.json",
    } <= wheel_paths
    assert provenance["combined_output_sha256"] == subject._sha256(
        subject._canonical_bytes(provenance["commands"])
    )
    assert stat.S_IMODE(case["work_root"].lstat().st_mode) == 0o700

    argv_order = [call["argv"][1:3] for call in runner.calls]
    assert argv_order == [
        ["-I", "-S"],
        ["--version"],
        ["venv", "--python"],
        ["pip", "install"],
        ["-I", "-c"],
        ["-I", "-c"],
        ["build", "--sdist"],
        ["build", "--wheel"],
        ["venv", "--python"],
        ["-I", "-c"],
        ["-I", "-m"],
        ["-I", "-c"],
        ["-I", "-m"],
        ["-I", "-c"],
        ["-I", "-m"],
        ["-I", "-c"],
        ["-I", "-c"],
        [runner.calls[17]["argv"][1], "--source-package-root"],
        ["-I", "-c"],
    ]
    for call in runner.calls:
        argv = call["argv"]
        if argv[1] in {"venv", "pip", "build"}:
            assert "--offline" in argv
            assert call["env"]["UV_PYTHON_DOWNLOADS"] == "never"
            assert call["env"]["UV_CACHE_DIR"] == str(case["cache"])
    assert "--no-build-isolation" in runner.calls[6]["argv"]
    assert "--no-cache" in runner.calls[6]["argv"]
    assert "--no-index" not in runner.calls[3]["argv"]
    assert runner.calls[3]["argv"][-5:] == list(subject.BUILD_BACKEND_REQUIREMENTS)
    assert "--no-index" in runner.calls[6]["argv"]
    assert runner.calls[6]["argv"][runner.calls[6]["argv"].index("--python") + 1].endswith(
        "build-venv/bin/python"
    )
    assert "--no-deps" in runner.calls[14]["argv"]
    assert "--no-compile" in runner.calls[14]["argv"]
    assert runner.calls[17]["argv"][1].endswith("quant_investor/v17_v2_contract/package_parity.py")
    assert runner.calls[17]["argv"][-4:] == [
        "--expected-name",
        "quant-investor",
        "--expected-version",
        "17.0.0",
    ]
    for command, call in zip(provenance["commands"], runner.calls, strict=True):
        proof = command["sanitized_environment"]
        assert proof["base_environment"] == subject.SAFE_EXECUTION_ENVIRONMENT
        assert proof["effective_environment"] == dict(sorted(call["env"].items()))
        assert proof["overrides"] == command["env"]
        assert proof["host_environment"]["inherited_value_count"] == 0
        assert proof["host_environment"]["secret_values_recorded"] is False


def test_main_writes_canonical_output_exact_once_0600(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    runner = FakeRunner()
    monkeypatch.setattr(subject.subprocess, "run", runner)
    output = case["private"] / "package_parity.json"
    code = subject.main(
        [
            "--repo-root",
            str(case["repo"]),
            "--expected-base-commit",
            BASE_COMMIT,
            "--session-manifest",
            str(case["session_path"]),
            "--expected-source-binding-json",
            str(case["binding"]),
            "--base-python",
            str(case["base_python"]),
            "--uv-bin",
            str(case["uv"]),
            "--uv-cache",
            str(case["cache"]),
            "--work-root",
            str(case["work_root"]),
            "--output-json",
            str(output),
        ]
    )
    assert code == 0
    raw = output.read_bytes()
    assert stat.S_IMODE(output.lstat().st_mode) == 0o600
    assert raw == subject._canonical_resource_bytes(json.loads(raw))
    assert (
        subject.main(
            [
                "--repo-root",
                str(case["repo"]),
                "--expected-base-commit",
                BASE_COMMIT,
                "--session-manifest",
                str(case["session_path"]),
                "--expected-source-binding-json",
                str(case["binding"]),
                "--base-python",
                str(case["base_python"]),
                "--uv-bin",
                str(case["uv"]),
                "--uv-cache",
                str(case["cache"]),
                "--work-root",
                str(tmp_path / "second-work"),
                "--output-json",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_bytes() == raw


def test_exact_once_output_parent_swap_before_create_writes_nowhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    private = _private_dir(tmp_path / "private")
    moved = tmp_path / "private-moved"
    external = _private_dir(tmp_path / "external")
    output = private / "package.json"
    real_open = os.open
    swapped = False

    def racing_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if (
            not swapped
            and dir_fd is None
            and Path(path) == private
            and flags & getattr(os, "O_DIRECTORY", 0)
        ):
            private.rename(moved)
            private.symlink_to(external, target_is_directory=True)
            swapped = True
        return descriptor

    monkeypatch.setattr(subject.os, "open", racing_open)
    with pytest.raises(subject.PackageEvidenceError, match="parent changed before create"):
        subject._write_exact_once(output, b"sealed\n", repo_root=repo)
    assert not (external / output.name).exists()
    assert not (moved / output.name).exists()


def test_exact_once_output_parent_swap_at_create_cleans_owned_inode_and_writes_nowhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    private = _private_dir(tmp_path / "private")
    moved = tmp_path / "private-moved"
    external = _private_dir(tmp_path / "external")
    output = private / "package.json"
    real_open = os.open
    swapped = False

    def racing_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and dir_fd is not None and str(path) == output.name:
            private.rename(moved)
            private.symlink_to(external, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(subject.os, "open", racing_open)
    with pytest.raises(subject.PackageEvidenceError, match="parent changed"):
        subject._write_exact_once(output, b"sealed\n", repo_root=repo)
    assert not (external / output.name).exists()
    assert not (moved / output.name).exists()


def test_main_binds_output_parent_identity_across_long_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    output_parent = _private_dir(tmp_path / "output-private")
    moved = tmp_path / "output-private-moved"
    output = output_parent / "package.json"

    class LongBuildParentSwapRunner(FakeRunner):
        def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            completed = super().__call__(argv, **kwargs)
            if len(self.calls) == 1:
                output_parent.rename(moved)
                _private_dir(output_parent)
            return completed

    monkeypatch.setattr(subject.subprocess, "run", LongBuildParentSwapRunner())
    code = subject.main(
        [
            "--repo-root",
            str(case["repo"]),
            "--expected-base-commit",
            BASE_COMMIT,
            "--session-manifest",
            str(case["session_path"]),
            "--expected-source-binding-json",
            str(case["binding"]),
            "--base-python",
            str(case["base_python"]),
            "--uv-bin",
            str(case["uv"]),
            "--uv-cache",
            str(case["cache"]),
            "--work-root",
            str(case["work_root"]),
            "--output-json",
            str(output),
        ]
    )
    assert code == 2
    assert not output.exists()
    assert not (moved / output.name).exists()


def test_wrong_backend_version_fails_closed_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner(bad_backend=True))
    with pytest.raises(subject.PackageEvidenceError, match="build backend"):
        _build(case)


@pytest.mark.parametrize(
    "package_versions",
    [
        {
            key: value
            for key, value in subject.EXPECTED_BUILD_BACKEND_PACKAGES.items()
            if key != "pluggy"
        },
        {**subject.EXPECTED_BUILD_BACKEND_PACKAGES, "unexpected": "1.0"},
        {**subject.EXPECTED_BUILD_BACKEND_PACKAGES, "packaging": "26.1"},
    ],
    ids=["missing-package", "extra-package", "version-drift"],
)
def test_backend_inventory_must_match_every_locked_package_exactly(
    package_versions: Mapping[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        FakeRunner(backend_packages=package_versions),
    )
    with pytest.raises(subject.PackageEvidenceError, match="package inventory mismatch"):
        _build(case)


def test_backend_inventory_rejects_duplicate_or_unnamed_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    duplicate = [
        *subject.EXPECTED_BUILD_BACKEND_INVENTORY,
        {"name": "pluggy", "version": "1.6.0"},
    ]
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        FakeRunner(backend_inventory=duplicate),
    )
    with pytest.raises(subject.PackageEvidenceError, match="distribution inventory mismatch"):
        _build(case)

    second = _case(tmp_path / "unnamed", monkeypatch)

    class UnnamedDistributionRunner(FakeRunner):
        def _stdout(self, argv: list[str]) -> bytes:
            raw = super()._stdout(argv)
            if argv[1:4] != ["-I", "-c", subject._backend_probe_code()]:
                return raw
            payload = json.loads(raw)
            payload["unnamed_distribution_count"] = 1
            return subject._canonical_bytes(payload)

    monkeypatch.setattr(subject.subprocess, "run", UnnamedDistributionRunner())
    with pytest.raises(subject.PackageEvidenceError, match="distribution inventory mismatch"):
        _build(second)


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (FakeRunner(uv_output="uv 0.10.8 (old 2026-01-01)"), "uv version"),
        (FakeRunner(uv_output="uv 0.10.9 (different-build 2026-03-06)"), "uv version"),
        (FakeRunner(pip_version="25.1"), "pip version"),
    ],
    ids=["wrong-uv-version", "wrong-uv-build", "wrong-pip-version"],
)
def test_frozen_uv_and_measured_post_ensurepip_versions_fail_closed_on_drift(
    runner: FakeRunner,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(subject.subprocess, "run", runner)
    with pytest.raises(subject.PackageEvidenceError, match=message):
        _build(case)


def test_source_drift_before_and_after_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner())
    monkeypatch.setattr(
        subject,
        "_sample_source_binding",
        lambda _repo, _base: {**SOURCE_BINDING, "porcelain_sha256": "0" * 64},
    )
    with pytest.raises(subject.PackageEvidenceError, match="differs before"):
        _build(case)

    second = _case(tmp_path / "second", monkeypatch)
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner())
    samples = [SOURCE_BINDING, {**SOURCE_BINDING, "porcelain_sha256": "0" * 64}]
    monkeypatch.setattr(subject, "_sample_source_binding", lambda _repo, _base: samples.pop(0))
    with pytest.raises(subject.PackageEvidenceError, match="drifted"):
        _build(second)


def test_artifact_hash_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)

    class BadParityRunner(FakeRunner):
        def _stdout(self, argv: list[str]) -> bytes:
            if len(argv) > 1 and argv[1].endswith("package_parity.py"):
                payload = {**_parity_payload(), "wheel_sha256": "0" * 64}
                return subject._canonical_bytes(payload)
            return super()._stdout(argv)

    monkeypatch.setattr(subject.subprocess, "run", BadParityRunner())
    with pytest.raises(subject.PackageEvidenceError, match="artifact hash"):
        _build(case)


def test_fresh_work_root_required_and_no_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    case["work_root"].mkdir()
    case["work_root"].chmod(0o700)
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner())
    with pytest.raises(subject.PackageEvidenceError, match="fresh"):
        _build(case)


def test_command_failure_keeps_zero_final_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner(fail_at=5))
    output = case["private"] / "failed.json"
    assert (
        subject.main(
            [
                "--repo-root",
                str(case["repo"]),
                "--expected-base-commit",
                BASE_COMMIT,
                "--session-manifest",
                str(case["session_path"]),
                "--expected-source-binding-json",
                str(case["binding"]),
                "--base-python",
                str(case["base_python"]),
                "--uv-bin",
                str(case["uv"]),
                "--uv-cache",
                str(case["cache"]),
                "--work-root",
                str(case["work_root"]),
                "--output-json",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()
    assert case["work_root"].exists()


def test_invalid_source_binding_and_base_commit_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    _source_binding_file(case["binding"], {**SOURCE_BINDING, "base_commit": "f" * 40})
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner())
    with pytest.raises(subject.PackageEvidenceError, match="disagree"):
        _build(case)


def test_private_binding_and_output_must_be_outside_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    inside = case["repo"] / "binding.json"
    _source_binding_file(inside)
    case["binding"] = inside
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner())
    with pytest.raises(subject.PackageEvidenceError, match="outside the repository"):
        _build(case)

    second = _case(tmp_path / "second-private", monkeypatch)
    output = second["repo"] / "package.json"
    assert (
        subject.main(
            [
                "--repo-root",
                str(second["repo"]),
                "--expected-base-commit",
                BASE_COMMIT,
                "--session-manifest",
                str(second["session_path"]),
                "--expected-source-binding-json",
                str(second["binding"]),
                "--base-python",
                str(second["base_python"]),
                "--uv-bin",
                str(second["uv"]),
                "--uv-cache",
                str(second["cache"]),
                "--work-root",
                str(second["work_root"]),
                "--output-json",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_relative_cli_paths_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner())
    with pytest.raises(subject.PackageEvidenceError, match="repo root must be an absolute path"):
        subject.build_package_evidence(
            repo_root=Path("relative-repo"),
            expected_base_commit=BASE_COMMIT,
            session_manifest=case["session_path"],
            expected_source_binding_json=case["binding"],
            base_python=case["base_python"],
            uv_bin=case["uv"],
            uv_cache=case["cache"],
            work_root=case["work_root"],
        )
    with pytest.raises(
        subject.PackageEvidenceError, match="session manifest must be an absolute path"
    ):
        subject.build_package_evidence(
            repo_root=case["repo"],
            expected_base_commit=BASE_COMMIT,
            session_manifest=Path("session.json"),
            expected_source_binding_json=case["binding"],
            base_python=case["base_python"],
            uv_bin=case["uv"],
            uv_cache=case["cache"],
            work_root=case["work_root"],
        )
    with pytest.raises(
        subject.PackageEvidenceError, match="expected source binding must be an absolute path"
    ):
        subject.build_package_evidence(
            repo_root=case["repo"],
            expected_base_commit=BASE_COMMIT,
            session_manifest=case["session_path"],
            expected_source_binding_json=Path("binding.json"),
            base_python=case["base_python"],
            uv_bin=case["uv"],
            uv_cache=case["cache"],
            work_root=case["work_root"],
        )
    with pytest.raises(subject.PackageEvidenceError, match="base Python must be an absolute path"):
        subject.build_package_evidence(
            repo_root=case["repo"],
            expected_base_commit=BASE_COMMIT,
            session_manifest=case["session_path"],
            expected_source_binding_json=case["binding"],
            base_python=Path("python3.13"),
            uv_bin=case["uv"],
            uv_cache=case["cache"],
            work_root=case["work_root"],
        )
    with pytest.raises(subject.PackageEvidenceError, match="uv binary must be an absolute path"):
        subject.build_package_evidence(
            repo_root=case["repo"],
            expected_base_commit=BASE_COMMIT,
            session_manifest=case["session_path"],
            expected_source_binding_json=case["binding"],
            base_python=case["base_python"],
            uv_bin=Path("uv"),
            uv_cache=case["cache"],
            work_root=case["work_root"],
        )
    with pytest.raises(subject.PackageEvidenceError, match="uv cache must be an absolute path"):
        subject.build_package_evidence(
            repo_root=case["repo"],
            expected_base_commit=BASE_COMMIT,
            session_manifest=case["session_path"],
            expected_source_binding_json=case["binding"],
            base_python=case["base_python"],
            uv_bin=case["uv"],
            uv_cache=Path("uv-cache"),
            work_root=case["work_root"],
        )
    with pytest.raises(subject.PackageEvidenceError, match="work root must be an absolute path"):
        subject.build_package_evidence(
            repo_root=case["repo"],
            expected_base_commit=BASE_COMMIT,
            session_manifest=case["session_path"],
            expected_source_binding_json=case["binding"],
            base_python=case["base_python"],
            uv_bin=case["uv"],
            uv_cache=case["cache"],
            work_root=Path("work"),
        )
    with pytest.raises(subject.PackageEvidenceError, match="output JSON must be an absolute path"):
        subject._validate_output_target(Path("out.json"), repo_root=case["repo"])


def test_host_env_is_fully_scrubbed_and_receipt_proves_exact_effective_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    polluted = {
        "ALL_PROXY": "http://proxy.invalid",
        "CURL_CA_BUNDLE": "/bad/curl.pem",
        "HTTPS_PROXY": "http://proxy.invalid",
        "HTTP_PROXY": "http://proxy.invalid",
        "NO_PROXY": "*",
        "PIP_CONFIG_FILE": "/bad/pip.conf",
        "PIP_INDEX_URL": "https://example.invalid/simple",
        "PYTHONHOME": "/bad/pythonhome",
        "PYTHONPATH": "/bad/pythonpath",
        "REQUESTS_CA_BUNDLE": "/bad/requests.pem",
        "SSL_CERT_FILE": "/bad/cert.pem",
        "UV_INDEX_URL": "https://example.invalid/simple",
        "UV_PROJECT": "/bad/project",
        "VIRTUAL_ENV": "/bad/venv",
    }
    for key, value in polluted.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("MYQUANT_ALLOWED_HOST_ENV", "kept")
    runner = FakeRunner()
    monkeypatch.setattr(subject.subprocess, "run", runner)
    report = _build(case)
    assert report["accepted"] is True
    for call in runner.calls:
        env = call["env"]
        for key, value in polluted.items():
            assert env.get(key) != value
        assert "MYQUANT_ALLOWED_HOST_ENV" not in env
    receipts = report["build_install_provenance"]["commands"]
    for receipt, call in zip(receipts, runner.calls, strict=True):
        proof = receipt["sanitized_environment"]
        assert proof["effective_environment"] == dict(sorted(call["env"].items()))
        assert proof["overrides"] == receipt["env"]
        assert proof["host_environment"]["inherited_value_count"] == 0
        assert proof["host_environment"]["secret_values_recorded"] is False
        assert proof["host_environment"]["stripped_variable_name_count"] == len(os.environ)
        assert proof["host_environment"]["stripped_variable_names_sha256"] == subject._sha256(
            subject._canonical_bytes(sorted(os.environ))
        )


def test_command_environment_mutation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)

    class MutatingEnvironmentRunner(FakeRunner):
        def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            completed = super().__call__(argv, **kwargs)
            kwargs["env"]["MUTATED_AFTER_EXECUTION"] = "forbidden"
            return completed

    monkeypatch.setattr(subject.subprocess, "run", MutatingEnvironmentRunner())
    with pytest.raises(subject.PackageEvidenceError, match="environment changed"):
        _build(case)


def test_non_allowlisted_or_variable_secret_environment_values_are_never_recorded(
    tmp_path: Path,
) -> None:
    commands: list[dict[str, object]] = []
    with pytest.raises(subject.PackageEvidenceError, match="non-allowlisted"):
        subject._run_command(
            ["/usr/bin/true"],
            role="forbidden-environment",
            cwd=tmp_path,
            env_overrides={"AWS_SECRET_ACCESS_KEY": "must-not-be-recorded"},
            tool_version="test",
            commands=commands,
        )
    assert commands == []
    with pytest.raises(subject.PackageEvidenceError, match="fixed environment"):
        subject._run_command(
            ["/usr/bin/true"],
            role="forbidden-environment",
            cwd=tmp_path,
            env_overrides={"PIP_NO_INDEX": "secret-value"},
            tool_version="test",
            commands=commands,
        )
    assert commands == []


def test_source_binding_rejects_symlink_hardlink_and_non_private_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner())

    symlink_case = _case(tmp_path / "symlink", monkeypatch)
    target = symlink_case["binding"]
    link = symlink_case["private"] / "binding-link.json"
    link.symlink_to(target)
    symlink_case["binding"] = link
    with pytest.raises(subject.PackageEvidenceError, match="symlink indirection"):
        _build(symlink_case)

    hardlink_case = _case(tmp_path / "hardlink", monkeypatch)
    hardlink = hardlink_case["private"] / "binding-hardlink.json"
    os.link(hardlink_case["binding"], hardlink)
    hardlink_case["binding"] = hardlink
    with pytest.raises(subject.PackageEvidenceError, match="link count"):
        _build(hardlink_case)

    mode_case = _case(tmp_path / "mode", monkeypatch)
    mode_case["binding"].chmod(0o640)
    with pytest.raises(subject.PackageEvidenceError, match="permissions"):
        _build(mode_case)


def test_source_binding_replacement_during_no_follow_read_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    real_read = os.read
    replaced = False

    def racing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, size)
        if not replaced:
            replacement = case["private"] / "replacement.json"
            _source_binding_file(replacement)
            os.replace(replacement, case["binding"])
            replaced = True
        return chunk

    monkeypatch.setattr(subject.os, "read", racing_read)
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner())
    with pytest.raises(subject.PackageEvidenceError, match="changed during stable read"):
        _build(case)


def test_source_binding_parent_rename_to_same_inode_symlink_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    private = case["private"]
    moved = tmp_path / "private-moved"
    real_open = os.open
    swapped = False

    def racing_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if (
            not swapped
            and dir_fd is None
            and Path(path) == private
            and flags & getattr(os, "O_DIRECTORY", 0)
        ):
            private.rename(moved)
            private.symlink_to(moved, target_is_directory=True)
            swapped = True
        return descriptor

    monkeypatch.setattr(subject.os, "open", racing_open)
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner())
    with pytest.raises(subject.PackageEvidenceError, match="parent changed during stable read"):
        _build(case)


def test_generic_critical_file_binding_rejects_parent_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = _private_dir(tmp_path / "critical")
    artifact = private / "artifact.whl"
    artifact.write_bytes(b"artifact")
    artifact.chmod(0o600)
    moved = tmp_path / "critical-moved"
    real_open = os.open
    swapped = False

    def racing_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if (
            not swapped
            and dir_fd is None
            and Path(path) == private
            and flags & getattr(os, "O_DIRECTORY", 0)
        ):
            private.rename(moved)
            private.symlink_to(moved, target_is_directory=True)
            swapped = True
        return descriptor

    monkeypatch.setattr(subject.os, "open", racing_open)
    with pytest.raises(subject.PackageEvidenceError, match="parent changed during stable read"):
        subject._file_binding(artifact, label="critical artifact")


def test_critical_file_read_fails_closed_without_required_no_follow_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = _private_dir(tmp_path / "critical")
    artifact = private / "artifact.whl"
    artifact.write_bytes(b"artifact")
    artifact.chmod(0o600)
    monkeypatch.setattr(subject.os, "O_NOFOLLOW", 0)
    with pytest.raises(subject.PackageEvidenceError, match="O_NOFOLLOW"):
        subject._file_binding(artifact, label="critical artifact")


def test_source_binding_bytes_must_remain_unchanged_through_package_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)

    class SourceBindingDriftRunner(FakeRunner):
        def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            completed = super().__call__(argv, **kwargs)
            if len(self.calls) == 1:
                _source_binding_file(
                    case["binding"],
                    {**SOURCE_BINDING, "porcelain_sha256": "f" * 64},
                )
            return completed

    monkeypatch.setattr(subject.subprocess, "run", SourceBindingDriftRunner())
    with pytest.raises(subject.PackageEvidenceError, match="source binding drifted"):
        _build(case)


def test_wrong_python_version_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = _case(tmp_path, monkeypatch)

    class WrongPythonRunner(FakeRunner):
        def _stdout(self, argv: list[str]) -> bytes:
            if argv[1:6] == ["-I", "-S", "-B", "-c", subject._python_probe_code()]:
                payload = {
                    "executable": argv[0],
                    "implementation": "cpython",
                    "realpath": str(Path(argv[0]).resolve()),
                    "sha256": "1" * 64,
                    "version": "3.12.9",
                    "version_info": [3, 12, 9],
                }
                return subject._canonical_bytes(payload)
            return super()._stdout(argv)

    monkeypatch.setattr(subject.subprocess, "run", WrongPythonRunner())
    with pytest.raises(subject.PackageEvidenceError, match="CPython 3.13"):
        _build(case)


def test_forbids_shell_control_operators_in_command_capture(tmp_path: Path) -> None:
    commands: list[dict[str, object]] = []
    with pytest.raises(subject.PackageEvidenceError, match="shell control"):
        subject._run_command(
            ["git", "diff", "|"],
            role="bad_shell",
            cwd=tmp_path,
            env_overrides={},
            tool_version="git version fake",
            commands=commands,
        )


def test_v2_closed_schema_accepts_report_and_rejects_old_extra_and_bool_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_investor.v17_v2_contract.schema_validation import (
        SchemaValidationError,
        preflight_packaged_schema,
        validate_instance_against_schema,
    )

    real_checked_schema = subject._validate_checked_schema
    case = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner())
    report = _build(case)
    schema_path = (
        Path(__file__).parents[2]
        / "scripts"
        / "schemas"
        / "v17_phase0_package_evidence.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    preflight_packaged_schema(schema)
    validate_instance_against_schema(report, schema)
    real_checked_schema(
        report,
        repo_root=Path(__file__).parents[2],
        schema_relative_path=subject.PACKAGE_EVIDENCE_SCHEMA_PATH,
        schema_id=subject.PACKAGE_EVIDENCE_SCHEMA_ID,
        artifact_version=subject.PACKAGE_EVIDENCE_VERSION,
    )

    old = copy.deepcopy(report)
    old["version"] = "myquant.v17.v2.phase0-package-parity-evidence.v1"
    with pytest.raises(SchemaValidationError):
        validate_instance_against_schema(old, schema)
    extra = copy.deepcopy(report)
    extra["unexpected"] = True
    with pytest.raises(SchemaValidationError):
        validate_instance_against_schema(extra, schema)
    boolean_count = copy.deepcopy(report)
    boolean_count["build_install_provenance"]["command_count"] = True
    with pytest.raises(SchemaValidationError):
        validate_instance_against_schema(boolean_count, schema)


def test_session_toolchain_pip_bundle_and_wrapper_drift_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain_case = _case(tmp_path / "toolchain", monkeypatch)
    toolchain_case["session"]["toolchain_binding"]["base_python"]["version"] = "3.13.6"
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner())
    with pytest.raises(subject.PackageEvidenceError, match="toolchain differs"):
        _build(toolchain_case)

    bundle_case = _case(tmp_path / "bundle", monkeypatch)

    class BundleDriftRunner(FakeRunner):
        def _stdout(self, argv: list[str]) -> bytes:
            raw = super()._stdout(argv)
            if argv[1:4] == ["-I", "-c", subject._ensurepip_bundle_probe_code()]:
                payload = json.loads(raw)
                payload["wheel"]["sha256"] = "0" * 64
                return subject._canonical_bytes(payload)
            return raw

    monkeypatch.setattr(subject.subprocess, "run", BundleDriftRunner())
    with pytest.raises(subject.PackageEvidenceError, match="bundled pip wheel mismatch"):
        _build(bundle_case)

    wrapper_case = _case(tmp_path / "wrapper", monkeypatch)

    class WrapperDriftRunner(FakeRunner):
        def _stdout(self, argv: list[str]) -> bytes:
            raw = super()._stdout(argv)
            if argv[1:4] == ["-I", "-c", subject._install_inventory_probe_code()]:
                payload = json.loads(raw)
                payload["pip_wrappers"][0]["is_symlink"] = True
                return subject._canonical_bytes(payload)
            return raw

    monkeypatch.setattr(subject.subprocess, "run", WrapperDriftRunner())
    with pytest.raises(subject.PackageEvidenceError, match="wrapper binding is unsafe"):
        _build(wrapper_case)


def test_protected_superset_and_hatch_snapshot_drift_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected_case = _case(tmp_path / "protected", monkeypatch)
    real_sample_protected = subject._sample_protected_roots
    protected_samples = [
        protected_case["session"]["protected_roots"],
        [
            *protected_case["session"]["protected_roots"][:-1],
            {
                **protected_case["session"]["protected_roots"][-1],
                "path": str(protected_case["repo"] / "results" / "changed"),
            },
        ],
    ]
    monkeypatch.setattr(
        subject,
        "_sample_protected_roots",
        lambda _repo: copy.deepcopy(protected_samples.pop(0)),
    )
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner())
    with pytest.raises(subject.PackageEvidenceError, match="protected roots drifted"):
        _build(protected_case)
    monkeypatch.setattr(subject, "_sample_protected_roots", real_sample_protected)

    superset_case = _case(tmp_path / "superset", monkeypatch)
    physical_samples: list[dict[str, Any]] = [
        _physical_superset(),
        copy.deepcopy(_physical_superset()),
    ]
    physical_samples[-1]["rows"][0]["mode"] = 0o600
    physical_samples[-1]["sha256"] = subject._sha256(
        subject._canonical_bytes(physical_samples[-1]["rows"])
    )
    monkeypatch.setattr(
        subject,
        "_sample_physical_superset",
        lambda _repo: copy.deepcopy(physical_samples.pop(0)),
    )
    monkeypatch.setattr(subject.subprocess, "run", FakeRunner())
    with pytest.raises(subject.PackageEvidenceError, match="physical source superset drifted"):
        _build(superset_case)

    hatch_case = _case(tmp_path / "hatch", monkeypatch)

    class HatchDriftRunner(FakeRunner):
        def _stdout(self, argv: list[str]) -> bytes:
            raw = super()._stdout(argv)
            if (
                argv[1:4] == ["-I", "-c", subject._hatch_selector_probe_code()]
                and sum(
                    call["argv"][1:4] == ["-I", "-c", subject._hatch_selector_probe_code()]
                    for call in self.calls
                )
                == 3
            ):
                payload = json.loads(raw)
                payload["hatch_source_namespace"]["rows"][0]["mode"] = 0o600
                payload["hatch_source_namespace"]["sha256"] = subject._sha256(
                    subject._canonical_bytes(payload["hatch_source_namespace"]["rows"])
                )
                payload["package_source_superset"]["rows"][0]["mode"] = 0o600
                payload["package_source_superset"]["sha256"] = subject._sha256(
                    subject._canonical_bytes(payload["package_source_superset"]["rows"])
                )
                return subject._canonical_bytes(payload)
            return raw

    monkeypatch.setattr(subject.subprocess, "run", HatchDriftRunner())
    with pytest.raises(
        subject.PackageEvidenceError, match="selector or physical source rows drifted"
    ):
        _build(hatch_case)
