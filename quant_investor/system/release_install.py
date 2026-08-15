"""Deterministic release build, isolated install, and exact-origin verification.

The production release path is deliberately separate from generation assembly.
It proves that a wheel built from one frozen Git tree is the package imported by
the deployed interpreter.  It never writes the System pointer and has no
broker, order, portfolio, or Strategy Record authority.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Final
import zipfile

from quant_investor.contracts import (
    ContractError,
    artifact_byte_sha256,
    canonical_json_bytes,
    contract_catalog_sha256,
    parse_canonical_json_bytes,
    seal_artifact,
    validate_artifact,
)

from .errors import SystemContractError, SystemPreconditionError, SystemSecurityError
from .release import INSTALLED_CODE_MANIFEST_DOMAIN

RELEASE_INSTALL_EVIDENCE_KIND: Final = "system.release_install_evidence"
RELEASE_KIND: Final = "system.release"
DEPENDENCY_INSTALL_MODE: Final = "UV_LOCKED_NON_EDITABLE_WHEEL"
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE: Final = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_ARCHIVE_ROW_FIELDS: Final = frozenset({"path", "byte_sha256", "size"})
_INPUT_FIELDS: Final = frozenset({"release_install_evidence", "deployed_release"})
_IGNORED_PARTS: Final = frozenset({"__pycache__"})
_IGNORED_SUFFIXES: Final = frozenset({".pyc", ".pyo"})
_MAX_ARTIFACT_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_PROBE_BYTES: Final = 1024 * 1024


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not lowercase SHA-256")
    return value


def _git_oid(value: Any, *, label: str) -> str:
    if type(value) is not str or _GIT_OID_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not a canonical Git object id")
    return value


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise SystemContractError(f"{label} is not canonical text")
    return value


def _object_ref(artifact: Mapping[str, Any]) -> dict[str, str]:
    document = validate_artifact(dict(artifact))
    return {
        "kind": document["kind"],
        "contract_sha256": document["contract_sha256"],
        "artifact_id": document["artifact_id"],
        "semantic_sha256": document["semantic_sha256"],
        "byte_sha256": artifact_byte_sha256(document),
    }


def _validate_ref(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != {
        "kind",
        "contract_sha256",
        "artifact_id",
        "semantic_sha256",
        "byte_sha256",
    }:
        raise SystemContractError(f"{label} fields are not exact")
    result = {
        "kind": _text(value["kind"], label=f"{label}.kind"),
        "contract_sha256": _sha(value["contract_sha256"], label=f"{label}.contract_sha256"),
        "artifact_id": _text(value["artifact_id"], label=f"{label}.artifact_id"),
        "semantic_sha256": _sha(value["semantic_sha256"], label=f"{label}.semantic_sha256"),
        "byte_sha256": _sha(value["byte_sha256"], label=f"{label}.byte_sha256"),
    }
    return result


def _git(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemPreconditionError("release Git verification could not run") from exc
    if completed.returncode != 0:
        raise SystemPreconditionError("release Git verification failed")
    return completed.stdout


def _git_scalar(root: Path, *arguments: str) -> str:
    try:
        value = _git(root, *arguments).decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise SystemPreconditionError("release Git identity is not ASCII") from exc
    return _git_oid(value, label="release Git identity")


def _tree_rows(root: Path, commit: str) -> list[dict[str, str]]:
    raw = _git(root, "ls-tree", "-rz", "--full-tree", commit)
    rows: list[dict[str, str]] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        try:
            header, path_raw = entry.split(b"\t", 1)
            mode_raw, kind_raw, oid_raw = header.split(b" ", 2)
            path = path_raw.decode("utf-8")
            mode = mode_raw.decode("ascii")
            kind = kind_raw.decode("ascii")
            oid = oid_raw.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise SystemPreconditionError("release Git tree is malformed") from exc
        if kind == "blob":
            rows.append(
                {
                    "path": path,
                    "mode": mode,
                    "git_blob_oid": _git_oid(oid, label="release tree blob"),
                }
            )
    if rows != sorted(rows, key=lambda row: row["path"]):
        raise SystemPreconditionError("release Git tree order is unstable")
    return rows


def code_tree_sha256(repository_root: str | os.PathLike[str], commit: str) -> str:
    """Return the canonical full-tree inventory SHA for a frozen commit."""

    root = Path(repository_root).resolve(strict=True)
    frozen = _git_oid(commit, label="commit")
    if _git_scalar(root, "rev-parse", f"{frozen}^{{commit}}") != frozen:
        raise SystemPreconditionError("release commit object differs")
    return _sha256(canonical_json_bytes(_tree_rows(root, frozen)))


def _code_file(path: str) -> bool:
    parts = Path(path).parts
    return (
        bool(parts)
        and parts[0] == "quant_investor"
        and not any(part in _IGNORED_PARTS for part in parts)
        and Path(path).suffix not in _IGNORED_SUFFIXES
    )


def _manifest_sha(files: list[dict[str, Any]]) -> str:
    rows = sorted(files, key=lambda row: row["path"])
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise SystemPreconditionError("release code manifest is empty or duplicated")
    return _sha256(canonical_json_bytes({"domain": INSTALLED_CODE_MANIFEST_DOMAIN, "files": rows}))


def git_code_manifest_sha256(repository_root: str | os.PathLike[str], commit: str) -> str:
    """Hash exact package bytes from Git using the installed-manifest domain."""

    root = Path(repository_root).resolve(strict=True)
    frozen = _git_oid(commit, label="commit")
    files: list[dict[str, Any]] = []
    for row in _tree_rows(root, frozen):
        path = row["path"]
        if not _code_file(path):
            continue
        raw = _git(root, "cat-file", "blob", row["git_blob_oid"])
        files.append({"path": path, "byte_sha256": _sha256(raw), "size": len(raw)})
    return _manifest_sha(files)


def _regular_file(path: Path, *, label: str) -> tuple[bytes, os.stat_result]:
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) & 0o022
            or before.st_nlink != 1
            or before.st_size > _MAX_ARTIFACT_BYTES
        ):
            raise SystemSecurityError(f"{label} is not an owner-controlled regular file")
        raw = path.read_bytes()
        after = path.lstat()
    except SystemSecurityError:
        raise
    except OSError as exc:
        raise SystemSecurityError(f"{label} cannot be read") from exc
    if before != after or len(raw) != after.st_size:
        raise SystemSecurityError(f"{label} changed during readback")
    return raw, after


def _archive_row(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _ARCHIVE_ROW_FIELDS:
        raise SystemContractError(f"{label} fields are not exact")
    path = Path(_text(value["path"], label=f"{label}.path"))
    if not path.is_absolute():
        raise SystemContractError(f"{label}.path is not absolute")
    size = value["size"]
    if type(size) is not int or size <= 0 or size > _MAX_ARTIFACT_BYTES:
        raise SystemContractError(f"{label}.size is invalid")
    return {
        "path": str(path),
        "byte_sha256": _sha(value["byte_sha256"], label=f"{label}.byte_sha256"),
        "size": size,
    }


def build_release_install_evidence(
    *,
    final_commit: str,
    final_tree: str,
    code_tree_sha256_value: str,
    git_code_manifest_sha256_value: str,
    release_ref: Mapping[str, Any],
    source_archive: Mapping[str, Any],
    wheel: Mapping[str, Any],
    install_root: str | os.PathLike[str],
    python_executable: str | os.PathLike[str],
    python_executable_sha256: str,
    import_origin: str | os.PathLike[str],
    installed_code_manifest_sha256: str,
    contract_catalog_sha256_value: str,
    lockfile_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    """Seal observations; deep validation independently re-reads every binding."""

    root = Path(install_root)
    python = Path(python_executable)
    origin = Path(import_origin)
    for path, label in (
        (root, "install_root"),
        (python, "python_executable"),
        (origin, "import_origin"),
    ):
        if not path.is_absolute():
            raise SystemContractError(f"{label} is not absolute")
    body = {
        "state": "VALIDATED",
        "final_commit": _git_oid(final_commit, label="final_commit"),
        "final_tree": _git_oid(final_tree, label="final_tree"),
        "code_tree_sha256": _sha(code_tree_sha256_value, label="code_tree_sha256"),
        "git_code_manifest_sha256": _sha(
            git_code_manifest_sha256_value, label="git_code_manifest_sha256"
        ),
        "release_ref": _validate_ref(release_ref, label="release_ref"),
        "source_archive": _archive_row(source_archive, label="source_archive"),
        "wheel": _archive_row(wheel, label="wheel"),
        "install_root": str(root),
        "python_executable": str(python),
        "python_executable_sha256": _sha(
            python_executable_sha256, label="python_executable_sha256"
        ),
        "import_origin": str(origin),
        "installed_code_manifest_sha256": _sha(
            installed_code_manifest_sha256, label="installed_code_manifest_sha256"
        ),
        "contract_catalog_sha256": _sha(
            contract_catalog_sha256_value, label="contract_catalog_sha256"
        ),
        "lockfile_sha256": _sha(lockfile_sha256, label="lockfile_sha256"),
        "dependency_install_mode": DEPENDENCY_INSTALL_MODE,
        "editable_install": False,
        "source_tree_import": False,
    }
    identity = "release-install-" + _sha256(canonical_json_bytes(body))
    return seal_artifact(
        RELEASE_INSTALL_EVIDENCE_KIND,
        {**body, "release_install_id": identity},
        created_at=created_at,
    )


def validate_release_install_evidence(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    try:
        artifact = validate_artifact(document, expected_kind=RELEASE_INSTALL_EVIDENCE_KIND)
    except ContractError as exc:
        raise SystemContractError("release install evidence contract failed") from exc
    payload = artifact["payload"]
    if (
        payload["state"] != "VALIDATED"
        or payload["dependency_install_mode"] != DEPENDENCY_INSTALL_MODE
        or payload["editable_install"] is not False
        or payload["source_tree_import"] is not False
    ):
        raise SystemPreconditionError("release install evidence is not fail-closed")
    if payload["release_ref"]["kind"] != RELEASE_KIND:
        raise SystemContractError("release install evidence release kind is invalid")
    rebuilt = build_release_install_evidence(
        final_commit=payload["final_commit"],
        final_tree=payload["final_tree"],
        code_tree_sha256_value=payload["code_tree_sha256"],
        git_code_manifest_sha256_value=payload["git_code_manifest_sha256"],
        release_ref=payload["release_ref"],
        source_archive=payload["source_archive"],
        wheel=payload["wheel"],
        install_root=payload["install_root"],
        python_executable=payload["python_executable"],
        python_executable_sha256=payload["python_executable_sha256"],
        import_origin=payload["import_origin"],
        installed_code_manifest_sha256=payload["installed_code_manifest_sha256"],
        contract_catalog_sha256_value=payload["contract_catalog_sha256"],
        lockfile_sha256=payload["lockfile_sha256"],
        created_at=artifact["created_at"],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact):
        raise SystemContractError("release install evidence semantic replay differs")
    return artifact


def _wheel_code_manifest(path: Path) -> str:
    files: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith("/") or not _code_file(name):
                    continue
                raw = archive.read(name)
                files.append({"path": name, "byte_sha256": _sha256(raw), "size": len(raw)})
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise SystemPreconditionError("release wheel cannot be replayed") from exc
    return _manifest_sha(files)


def _sdist_code_manifest(path: Path) -> str:
    files: list[dict[str, Any]] = []
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            for member in sorted(archive.getmembers(), key=lambda row: row.name):
                if not member.isfile():
                    continue
                parts = Path(member.name).parts
                if len(parts) < 2:
                    continue
                relative = Path(*parts[1:]).as_posix()
                if not _code_file(relative):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise SystemPreconditionError("release source member is unreadable")
                raw = handle.read(_MAX_ARTIFACT_BYTES + 1)
                if len(raw) > _MAX_ARTIFACT_BYTES:
                    raise SystemPreconditionError("release source member exceeds byte bound")
                files.append({"path": relative, "byte_sha256": _sha256(raw), "size": len(raw)})
    except (OSError, tarfile.TarError) as exc:
        raise SystemPreconditionError("release source archive cannot be replayed") from exc
    return _manifest_sha(files)


def _verify_install_root(path: Path, repository_root: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SystemSecurityError("release install root is absent") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise SystemSecurityError("release install root is not owner-only")
    try:
        path.relative_to(repository_root)
    except ValueError:
        return
    raise SystemSecurityError("release install root is inside the source checkout")


def _probe_install(python: Path, install_root: Path, repository_root: Path) -> dict[str, Any]:
    probe = (
        "import json, pathlib, quant_investor; "
        "from quant_investor.contracts import contract_catalog_sha256; "
        "from quant_investor.system.release import installed_code_manifest_sha256; "
        "print(json.dumps({'import_origin':str(pathlib.Path(quant_investor.__file__).resolve()),"
        "'installed_code_manifest_sha256':installed_code_manifest_sha256(),"
        "'contract_catalog_sha256':contract_catalog_sha256()},sort_keys=True,separators=(',',':')))"
    )
    environment = {
        "HOME": str(install_root),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": "",
    }
    with tempfile.TemporaryDirectory(prefix="release-probe-") as directory:
        cwd = Path(directory).resolve(strict=True)
        if cwd == repository_root or repository_root in cwd.parents:
            raise SystemSecurityError("release probe cwd overlaps source checkout")
        try:
            completed = subprocess.run(
                [str(python), "-I", "-c", probe],
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SystemPreconditionError("isolated release import probe failed") from exc
    if completed.returncode != 0 or len(completed.stdout) > _MAX_PROBE_BYTES:
        raise SystemPreconditionError("isolated release import probe did not pass")
    try:
        result = json.loads(completed.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SystemPreconditionError("isolated release import probe is malformed") from exc
    if type(result) is not dict or set(result) != {
        "import_origin",
        "installed_code_manifest_sha256",
        "contract_catalog_sha256",
    }:
        raise SystemPreconditionError("isolated release import probe fields are not exact")
    return result


def _owner_directory(path: Path, *, create: bool, label: str) -> Path:
    if create and not path.exists():
        try:
            path.mkdir(parents=True, mode=0o700)
            path.chmod(0o700)
        except OSError as exc:
            raise SystemSecurityError(f"{label} cannot be created") from exc
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SystemSecurityError(f"{label} is absent") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise SystemSecurityError(f"{label} is not owner-only")
    return path.resolve(strict=True)


def _publish_exact(root: Path, raw: bytes, *, suffix: str) -> Path:
    digest = _sha256(raw)
    target = root / f"{digest}{suffix}"
    if target.exists():
        observed, _metadata = _regular_file(target, label="published release artifact")
        if observed != raw:
            raise SystemPreconditionError("published release artifact conflicts")
        return target
    temporary = root / f".{digest}.tmp-{os.getpid()}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SystemSecurityError("release artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, target, follow_symlinks=False)
    except FileExistsError as exc:
        raise SystemPreconditionError("published release artifact conflicts") from exc
    except OSError as exc:
        raise SystemSecurityError("release artifact publication failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    observed, _metadata = _regular_file(target, label="published release artifact")
    if observed != raw:
        raise SystemPreconditionError("published release artifact readback differs")
    return target


def _run_release_tool(
    argv: list[str], *, root: Path, environment: Mapping[str, str], timeout: int
) -> None:
    try:
        completed = subprocess.run(
            argv,
            cwd=root,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemPreconditionError("release build/install tool could not run") from exc
    if completed.returncode != 0:
        raise SystemPreconditionError("release build/install tool did not pass")


def prepare_operational_release(  # noqa: C901
    *,
    repository_root: str | os.PathLike[str],
    release_root: str | os.PathLike[str],
    final_commit: str,
    final_tree: str,
    created_at: str,
) -> dict[str, Any]:
    """Build, install, and seal one frozen operational release without network access."""

    root = Path(repository_root).resolve(strict=True)
    commit = _git_oid(final_commit, label="final_commit")
    tree = _git_oid(final_tree, label="final_tree")
    if (
        _git_scalar(root, "rev-parse", "HEAD^{commit}") != commit
        or _git_scalar(root, "rev-parse", "HEAD^{tree}") != tree
        or _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    ):
        raise SystemPreconditionError("release preparation checkout is not exact and clean")
    release_base = _owner_directory(Path(release_root), create=True, label="release root")
    if release_base == root or root in release_base.parents or release_base in root.parents:
        raise SystemSecurityError("release root overlaps the source checkout")
    artifact_root = _owner_directory(
        release_base / "artifacts", create=True, label="release artifact root"
    )
    install_base = _owner_directory(
        release_base / "installs", create=True, label="release install base"
    )
    uv_value = shutil.which("uv")
    if uv_value is None:
        raise SystemPreconditionError("release build tool is unavailable")
    uv = Path(uv_value).resolve(strict=True)
    source_epoch = _git(root, "show", "-s", "--format=%ct", commit).decode("ascii").strip()
    if not source_epoch.isdigit():
        raise SystemPreconditionError("release source epoch is invalid")
    environment = {
        "HOME": os.environ.get("HOME", str(release_base)),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": "",
        "SOURCE_DATE_EPOCH": source_epoch,
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(release_base / "uv-cache")),
    }
    with tempfile.TemporaryDirectory(prefix="release-build-", dir=release_base) as directory:
        build_root = Path(directory)
        build_output = build_root / "dist"
        build_output.mkdir(mode=0o700)
        _run_release_tool(
            [
                str(uv),
                "build",
                "--offline",
                "--no-build-isolation",
                "--no-create-gitignore",
                "--out-dir",
                str(build_output),
            ],
            root=root,
            environment=environment,
            timeout=900,
        )
        source_candidates = sorted(build_output.glob("*.tar.gz"))
        wheel_candidates = sorted(build_output.glob("*.whl"))
        if len(source_candidates) != 1 or len(wheel_candidates) != 1:
            raise SystemPreconditionError("release build outputs are not exact")
        source_raw = source_candidates[0].read_bytes()
        wheel_raw = wheel_candidates[0].read_bytes()
        source_path = _publish_exact(artifact_root, source_raw, suffix=".tar.gz")
        wheel_path = _publish_exact(artifact_root, wheel_raw, suffix=".whl")

    tree_sha = code_tree_sha256(root, commit)
    git_manifest_sha = git_code_manifest_sha256(root, commit)
    if (
        _sdist_code_manifest(source_path) != git_manifest_sha
        or _wheel_code_manifest(wheel_path) != git_manifest_sha
    ):
        raise SystemPreconditionError("built release archives differ from frozen package bytes")
    wheel_sha = _sha256(wheel_raw)
    install_root = install_base / f"{commit}-{wheel_sha}"
    if install_root.exists():
        _owner_directory(install_root, create=False, label="release install root")
    else:
        install_root.mkdir(mode=0o700)
        install_root.chmod(0o700)
        install_environment = {
            **environment,
            "UV_PROJECT_ENVIRONMENT": str(install_root),
            "VIRTUAL_ENV": str(install_root),
        }
        _run_release_tool(
            [
                str(uv),
                "venv",
                "--offline",
                "--no-project",
                "--python",
                sys.executable,
                str(install_root),
            ],
            root=root,
            environment=install_environment,
            timeout=300,
        )
        install_root.chmod(0o700)
        _run_release_tool(
            [
                str(uv),
                "sync",
                "--locked",
                "--no-install-project",
                "--no-install-workspace",
            ],
            root=root,
            environment=install_environment,
            timeout=1800,
        )
        python = install_root / "bin/python"
        _run_release_tool(
            [
                str(uv),
                "pip",
                "install",
                "--offline",
                "--no-index",
                "--no-deps",
                "--link-mode",
                "copy",
                "--python",
                str(python),
                str(wheel_path),
            ],
            root=root,
            environment=install_environment,
            timeout=600,
        )
    install_root = install_root.resolve(strict=True)
    python = install_root / "bin/python"
    python_resolved = python.resolve(strict=True)
    python_sha = _sha256(python_resolved.read_bytes())
    probe = _probe_install(python, install_root, root)
    if probe["installed_code_manifest_sha256"] != git_manifest_sha:
        raise SystemPreconditionError("prepared installed code differs from frozen Git package")
    if probe["contract_catalog_sha256"] != contract_catalog_sha256():
        raise SystemPreconditionError("prepared installed contract catalog differs")
    release_body = {
        "state": "OPERATIONAL",
        "code_sha256": tree_sha,
        "wheel_sha256": wheel_sha,
        "code_manifest_sha256": git_manifest_sha,
    }
    release_id = "unified-release-" + _sha256(canonical_json_bytes(release_body))
    release = seal_artifact(
        RELEASE_KIND,
        {**release_body, "release_id": release_id},
        created_at=created_at,
    )
    release_ref = _object_ref(release)
    lock_raw, _metadata = _regular_file(root / "uv.lock", label="release lockfile")
    evidence = build_release_install_evidence(
        final_commit=commit,
        final_tree=tree,
        code_tree_sha256_value=tree_sha,
        git_code_manifest_sha256_value=git_manifest_sha,
        release_ref=release_ref,
        source_archive={
            "path": str(source_path),
            "byte_sha256": _sha256(source_raw),
            "size": len(source_raw),
        },
        wheel={"path": str(wheel_path), "byte_sha256": wheel_sha, "size": len(wheel_raw)},
        install_root=str(install_root),
        python_executable=str(python),
        python_executable_sha256=python_sha,
        import_origin=probe["import_origin"],
        installed_code_manifest_sha256=probe["installed_code_manifest_sha256"],
        contract_catalog_sha256_value=probe["contract_catalog_sha256"],
        lockfile_sha256=_sha256(lock_raw),
        created_at=created_at,
    )
    exact_input = canonical_json_bytes(
        {"release_install_evidence": evidence, "deployed_release": release}
    )
    verification = verify_release_install_input(exact_input, repository_root=root)
    if verification["state"] != "PASS":
        raise SystemPreconditionError("prepared release verification did not pass")
    return {
        "release": release,
        "release_install_evidence": evidence,
        "verification": verification,
    }


def verify_release_install_input(  # noqa: C901
    raw: bytes,
    *,
    repository_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Deeply replay exact release/install evidence supplied on stdin."""

    try:
        value = parse_canonical_json_bytes(raw)
    except ContractError as exc:
        raise SystemContractError("release install gate input is not canonical") from exc
    if type(value) is not dict or set(value) != _INPUT_FIELDS:
        raise SystemContractError("release install gate input fields are not exact")
    evidence = validate_release_install_evidence(value["release_install_evidence"])
    try:
        release = validate_artifact(value["deployed_release"], expected_kind=RELEASE_KIND)
    except ContractError as exc:
        raise SystemContractError("deployed release contract failed") from exc
    payload = evidence["payload"]
    release_payload = release["payload"]
    if _object_ref(release) != payload["release_ref"]:
        raise SystemPreconditionError("release install evidence binds another release")

    root = Path(repository_root).resolve(strict=True)
    commit = payload["final_commit"]
    if _git_scalar(root, "rev-parse", "HEAD^{commit}") != commit:
        raise SystemPreconditionError("release install checkout is not frozen HEAD")
    if _git_scalar(root, "rev-parse", "HEAD^{tree}") != payload["final_tree"]:
        raise SystemPreconditionError("release install checkout tree differs")
    if _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise SystemPreconditionError("release install checkout is not clean")
    observed_tree_sha = code_tree_sha256(root, commit)
    observed_git_manifest = git_code_manifest_sha256(root, commit)
    if (
        observed_tree_sha != payload["code_tree_sha256"]
        or observed_git_manifest != payload["git_code_manifest_sha256"]
        or release_payload["code_sha256"] != observed_tree_sha
    ):
        raise SystemPreconditionError("release source tree identity differs")

    source_row = payload["source_archive"]
    wheel_row = payload["wheel"]
    source_path = Path(source_row["path"]).resolve(strict=True)
    wheel_path = Path(wheel_row["path"]).resolve(strict=True)
    source_raw, source_stat = _regular_file(source_path, label="source archive")
    wheel_raw, wheel_stat = _regular_file(wheel_path, label="wheel")
    if (
        len(source_raw) != source_row["size"]
        or source_stat.st_size != source_row["size"]
        or _sha256(source_raw) != source_row["byte_sha256"]
        or len(wheel_raw) != wheel_row["size"]
        or wheel_stat.st_size != wheel_row["size"]
        or _sha256(wheel_raw) != wheel_row["byte_sha256"]
        or release_payload["wheel_sha256"] != wheel_row["byte_sha256"]
    ):
        raise SystemPreconditionError("release archive exact bytes differ")
    if (
        _wheel_code_manifest(wheel_path) != observed_git_manifest
        or _sdist_code_manifest(source_path) != observed_git_manifest
    ):
        raise SystemPreconditionError("release archives do not reproduce frozen package bytes")

    lock_raw, _lock_stat = _regular_file(root / "uv.lock", label="release lockfile")
    if _sha256(lock_raw) != payload["lockfile_sha256"]:
        raise SystemPreconditionError("release dependency lock differs")
    install_root = Path(payload["install_root"]).resolve(strict=True)
    _verify_install_root(install_root, root)
    python = Path(payload["python_executable"])
    try:
        python_resolved = python.resolve(strict=True)
        python_raw = python_resolved.read_bytes()
    except OSError as exc:
        raise SystemSecurityError("release interpreter cannot be read") from exc
    if _sha256(python_raw) != payload["python_executable_sha256"]:
        raise SystemPreconditionError("release interpreter identity differs")
    probe = _probe_install(python, install_root, root)
    origin = Path(probe["import_origin"]).resolve(strict=True)
    if origin != Path(payload["import_origin"]).resolve(strict=True):
        raise SystemPreconditionError("installed import origin differs")
    if install_root not in origin.parents or root == origin or root in origin.parents:
        raise SystemPreconditionError("release import resolves to the source checkout")
    if origin.as_posix().endswith("/quant_investor/__init__.py") is False:
        raise SystemPreconditionError("release import origin is not the installed package")
    if (
        probe["installed_code_manifest_sha256"] != payload["installed_code_manifest_sha256"]
        or probe["installed_code_manifest_sha256"] != observed_git_manifest
        or release_payload["code_manifest_sha256"] != observed_git_manifest
        or probe["contract_catalog_sha256"] != payload["contract_catalog_sha256"]
        or probe["contract_catalog_sha256"] != contract_catalog_sha256()
    ):
        raise SystemPreconditionError("installed release semantic identity differs")
    return {
        "state": "PASS",
        "release_ref": payload["release_ref"],
        "source_archive_sha256": source_row["byte_sha256"],
        "wheel_sha256": wheel_row["byte_sha256"],
        "code_tree_sha256": observed_tree_sha,
        "installed_code_manifest_sha256": observed_git_manifest,
        "contract_catalog_sha256": probe["contract_catalog_sha256"],
        "import_origin": str(origin),
    }


def release_install_gate_main() -> int:
    """Fixed stdin-only entry point used by the final cutover gate runner."""

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(0, min(1024 * 1024, 64 * 1024 * 1024 + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > 64 * 1024 * 1024:
            break
    raw = b"".join(chunks)
    if len(raw) > 64 * 1024 * 1024:
        raise SystemContractError("release install gate input exceeds byte bound")
    result = verify_release_install_input(raw, repository_root=Path.cwd())
    os.write(1, canonical_json_bytes(result))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the fixed runner
    raise SystemExit(release_install_gate_main())


__all__ = [
    "DEPENDENCY_INSTALL_MODE",
    "RELEASE_INSTALL_EVIDENCE_KIND",
    "build_release_install_evidence",
    "code_tree_sha256",
    "git_code_manifest_sha256",
    "prepare_operational_release",
    "release_install_gate_main",
    "validate_release_install_evidence",
    "verify_release_install_input",
]
