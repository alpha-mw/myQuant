from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import stat
import subprocess
from types import ModuleType

import pytest


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = _load_script("v17_phase0_diff_check")


def _run(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": "/private/tmp",
    }
    if env is not None:
        environment.update(env)
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    path.chmod(0o700)
    return path


def _registry_source() -> str:
    values = ",\n".join(
        f"        {value!r}" for value in sorted(subject.PHASE0_ALLOWED_PATTERN_REGISTRY)
    )
    return "PHASE0_ALLOWED_PATTERN_REGISTRY = frozenset(\n" "    {\n" f"{values}\n" "    }\n" ")\n"


def _main_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "main"
    repo.mkdir()
    _run(["git", "init", "-q"], cwd=repo)
    _run(["git", "config", "user.email", "phase0@example.invalid"], cwd=repo)
    _run(["git", "config", "user.name", "Phase Zero"], cwd=repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    index_source = repo / "scripts" / "v17_phase0_evidence_index.py"
    index_source.parent.mkdir()
    index_source.write_text(_registry_source(), encoding="utf-8")
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-q", "-m", "base"], cwd=repo)
    return repo


def _linked_repo(tmp_path: Path, *, bad_whitespace: bool = False) -> Path:
    main = _main_repo(tmp_path)
    linked = tmp_path / "linked"
    _run(["git", "worktree", "add", "-q", "-b", "phase0-linked", str(linked)], cwd=main)
    untracked = linked / "scripts" / "v17_phase0_diff_check.py"
    untracked.write_text("value = 1 \n" if bad_whitespace else "value = 1\n", encoding="utf-8")
    return linked


def _binding(repo: Path) -> dict[str, str]:
    return subject._source_binding_from_snapshot(subject._source_snapshot(repo))


def test_real_linked_worktree_uses_alternate_index_and_intent_to_add(
    tmp_path: Path,
) -> None:
    repo = _linked_repo(tmp_path)
    work = _private_dir(tmp_path / "diff-work")
    real_before = subject._real_index_snapshot(repo)

    result = subject.run_isolated_diff_check(
        repo_root=repo,
        work_root=work,
        expected_source_binding=_binding(repo),
    )

    assert result["argv"] == ["git", "diff", "--check"]
    assert result["cwd"] == str(repo)
    assert result["exit_code"] == 0
    assert result["signal"] is None
    assert result["stdout"] == result["stderr"] == b""
    assert result["environment"]["GIT_INDEX_FILE"] == str(work / subject.ALTERNATE_INDEX_NAME)
    assert result["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert subject._real_index_snapshot(repo) == real_before
    assert "/worktrees/" in real_before["path"]

    staged = _run(
        [
            "git",
            "ls-files",
            "--stage",
            "--",
            "scripts/v17_phase0_diff_check.py",
        ],
        cwd=repo,
        env=result["environment"],
    )
    assert " 0\tscripts/v17_phase0_diff_check.py" in staged
    assert len(staged.split()[1]) in {40, 64}
    object_files = [path for path in (work / "objects").rglob("*") if path.is_file()]
    assert len(object_files) <= 1


def test_nonempty_diff_output_is_returned_without_hiding_real_exit(
    tmp_path: Path,
) -> None:
    repo = _linked_repo(tmp_path, bad_whitespace=True)
    result = subject.run_isolated_diff_check(
        repo_root=repo,
        work_root=_private_dir(tmp_path / "diff-work"),
        expected_source_binding=_binding(repo),
    )
    assert result["exit_code"] != 0
    assert b"trailing whitespace" in result["stdout"]
    assert b"scripts/v17_phase0_diff_check.py" in result["stdout"]


def test_real_index_stat_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _linked_repo(tmp_path)
    real_index = Path(subject._real_index_snapshot(repo)["path"])
    original_execute = subject._execute
    drifted = False

    def drift_after_final(
        argv: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
    ) -> tuple[int, bytes, bytes]:
        nonlocal drifted
        result = original_execute(argv, cwd=cwd, environment=environment)
        if list(argv) == ["git", "diff", "--check"] and not drifted:
            observed = real_index.stat()
            os.utime(
                real_index,
                ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000),
            )
            drifted = True
        return result

    monkeypatch.setattr(subject, "_execute", drift_after_final)
    with pytest.raises(subject.DiffCheckError, match="index/refs/object identity"):
        subject.run_isolated_diff_check(
            repo_root=repo,
            work_root=_private_dir(tmp_path / "diff-work"),
            expected_source_binding=_binding(repo),
        )


def test_object_inventory_ignores_directory_times_but_binds_file_bytes(
    tmp_path: Path,
) -> None:
    objects = _private_dir(tmp_path / "objects")
    fanout = _private_dir(objects / "aa")
    object_file = fanout / "object"
    object_file.write_bytes(b"first")
    before = subject._directory_inventory(objects)

    for directory in (objects, fanout):
        observed = directory.stat()
        os.utime(
            directory,
            ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000),
        )
    observed_file = object_file.stat()
    os.utime(
        object_file,
        ns=(observed_file.st_atime_ns, observed_file.st_mtime_ns + 1_000_000),
    )
    assert subject._directory_inventory(objects) == before

    object_file.write_bytes(b"other")
    after = subject._directory_inventory(objects)
    assert after != before
    before_file = next(row for row in before if row["type"] == "file")
    after_file = next(row for row in after if row["type"] == "file")
    assert before_file["sha256"] != after_file["sha256"]


@pytest.mark.parametrize("kind", ["nonempty", "permission", "symlink"])
def test_fresh_owner_private_concrete_work_root_is_required(
    tmp_path: Path,
    kind: str,
) -> None:
    repo = _linked_repo(tmp_path)
    concrete = _private_dir(tmp_path / "concrete-work")
    work = concrete
    expected = "fresh and empty"
    if kind == "nonempty":
        (work / "old").write_text("not fresh\n", encoding="utf-8")
    elif kind == "permission":
        work.chmod(0o755)
        expected = "mode 0700"
    else:
        work = tmp_path / "work-link"
        work.symlink_to(concrete, target_is_directory=True)
        expected = "symlink"
    with pytest.raises(subject.DiffCheckError, match=expected):
        subject.run_isolated_diff_check(
            repo_root=repo,
            work_root=work,
            expected_source_binding=_binding(repo),
        )


def test_registry_or_source_binding_drift_is_rejected_before_git_mutation(
    tmp_path: Path,
) -> None:
    repo = _linked_repo(tmp_path)
    index_source = repo / "scripts" / "v17_phase0_evidence_index.py"
    index_source.write_text(
        "PHASE0_ALLOWED_PATTERN_REGISTRY = frozenset({'other'})\n",
        encoding="utf-8",
    )
    with pytest.raises(subject.DiffCheckError, match="registry drifted"):
        subject.run_isolated_diff_check(
            repo_root=repo,
            work_root=_private_dir(tmp_path / "diff-work"),
            expected_source_binding=_binding(repo),
        )

    _run(["git", "restore", "scripts/v17_phase0_evidence_index.py"], cwd=repo)
    stale = _binding(repo)
    stale["source_state_sha256"] = "0" * 64
    with pytest.raises(subject.DiffCheckError, match="does not match"):
        subject.run_isolated_diff_check(
            repo_root=repo,
            work_root=_private_dir(tmp_path / "second-work"),
            expected_source_binding=stale,
        )
