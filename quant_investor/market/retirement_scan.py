"""Hash-bound, inventory-stable scanner for the retirement cutover."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from quant_investor.market.retirement import (
    ACTIVE_SCHEDULE_NAMES,
    REQUIRED_SCHEDULE_NAMES,
)

ALLOWLIST_SCHEMA = "retirement-scan-allowlist.v1"
RUNTIME_EVIDENCE_SCHEMA = "retirement-runtime-evidence.v1"
SCAN_REPORT_SCHEMA = "retirement-scan-report.v1"
MAX_RUNTIME_EVIDENCE_AGE = timedelta(minutes=15)

REQUIRED_RUNTIME_DOMAINS = (
    "automations",
    "scripts",
    "launchagents",
    "crontab",
    "processes",
    "open_fds",
    "dynamic_commands",
)

SKILL_LOGICAL_PATH = "external/skill/SKILL.md"
SCHEDULE_LOGICAL_PATHS = {
    name: f"external/schedules/{name}.toml" for name in ACTIVE_SCHEDULE_NAMES
}
REQUIRED_EXTERNAL_LOGICAL_PATHS = frozenset({SKILL_LOGICAL_PATH, *SCHEDULE_LOGICAL_PATHS.values()})

TEXT_TOKEN = re.compile(r"(?<![A-Za-z0-9])v16(?![0-9])", re.IGNORECASE)
PATH_TOKEN = re.compile(r"(^|[./_\-])v16($|[./_\-])", re.IGNORECASE)
VERSION_16 = re.compile(r"(?<![0-9])16\.0\.0(?![0-9])")
_BINARY_UTF16_V16 = re.compile(rb"(?i)(?:v\x001\x006\x00|\x00v\x001\x006)")
_HEX_DIGITS = frozenset("0123456789abcdef")

# These are exact repository-root-relative prefixes.  In particular, the
# active Python package ``quant_investor/data`` is not excluded.
EXCLUDED_ROOT_PREFIXES = frozenset(
    {
        ".claude/worktrees",
        ".git",
        ".omx",
        ".venv",
        ".venv-managed",
        ".uv-python",
        "venv",
        "node_modules",
        "frontend/node_modules",
        "build",
        "dist",
        "data",
        "reports",
        "results",
    }
)
EXCLUDED_LOGICAL_PATHS = frozenset({".agent/CONTINUITY.md"})
EXCLUDED_FILE_NAMES = frozenset({".DS_Store"})
EXCLUDED_CACHE_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)


class RetirementScanError(RuntimeError):
    """The scan is incomplete, unsafe, stale, or differs from its allowlist."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and text == text.lower() and set(text) <= _HEX_DIGITS


def _require_sha256(value: object, field: str) -> str:
    text = str(value).strip().lower()
    if not _is_sha256(text):
        raise RetirementScanError(f"{field} must be a lowercase SHA256")
    return text


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("semantic_sha256", None)
    return _sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RetirementScanError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_file_payload(path: Path) -> tuple[bytes, str, tuple[int, int, int, int, int, int]]:
    try:
        initial = os.lstat(path)
    except OSError as exc:
        raise RetirementScanError(f"scan input is missing or unreadable: {path}: {exc}") from exc
    initial_fingerprint = _stat_fingerprint(initial)
    if stat.S_ISLNK(initial.st_mode):
        try:
            payload = os.readlink(path).encode("utf-8")
            final = os.lstat(path)
        except OSError as exc:
            raise RetirementScanError(f"symlink changed while it was read: {path}: {exc}") from exc
        if _stat_fingerprint(final) != initial_fingerprint:
            raise RetirementScanError(f"symlink changed while it was read: {path}")
        return payload, "symlink", initial_fingerprint
    if not stat.S_ISREG(initial.st_mode):
        raise RetirementScanError(f"unsupported scan file type: {path}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RetirementScanError(f"scan file could not be opened safely: {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or (
            before.st_dev,
            before.st_ino,
        ) != (initial.st_dev, initial.st_ino):
            raise RetirementScanError(f"scan file identity changed before read: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        if _stat_fingerprint(before) != _stat_fingerprint(after):
            raise RetirementScanError(f"scan file changed while it was read: {path}")
    finally:
        os.close(fd)
    try:
        path_after = os.lstat(path)
    except OSError as exc:
        raise RetirementScanError(f"scan file disappeared after read: {path}: {exc}") from exc
    if _stat_fingerprint(path_after) != _stat_fingerprint(after):
        raise RetirementScanError(f"scan file path identity changed after read: {path}")
    return b"".join(chunks), "file", _stat_fingerprint(after)


def _read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload, kind, _ = _stable_file_payload(path)
    if kind != "file":
        raise RetirementScanError(f"JSON evidence cannot be a symlink: {path}")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RetirementScanError(f"non-finite JSON value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetirementScanError(f"invalid JSON evidence: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RetirementScanError(f"JSON evidence must be an object: {path}")
    return payload, value


def _valid_logical_path(value: object) -> str:
    text = str(value)
    parsed = PurePosixPath(text)
    if (
        not text
        or text.startswith("/")
        or text != parsed.as_posix()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise RetirementScanError(f"invalid logical scan path: {text!r}")
    return text


def _load_allowlist(path: Path) -> tuple[str, dict[tuple[str, str], dict[str, Any]]]:
    payload, value = _read_json(path)
    if set(value) != {"schema_version", "authority", "entries"}:
        raise RetirementScanError("retirement allowlist has an unexpected shape")
    if value.get("schema_version") != ALLOWLIST_SCHEMA:
        raise RetirementScanError("unexpected retirement allowlist schema")
    if value.get("authority") is not False:
        raise RetirementScanError("retirement allowlist must declare authority=false")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise RetirementScanError("retirement allowlist entries must be a list")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "detector",
            "exact_count",
            "reason",
            "expected_sha256",
        }:
            raise RetirementScanError("allowlist entry has an unexpected shape")
        logical_path = _valid_logical_path(entry["path"])
        detector = str(entry["detector"])
        count = entry["exact_count"]
        reason = str(entry["reason"]).strip()
        expected_sha = _require_sha256(entry["expected_sha256"], "allowlist expected_sha256")
        if logical_path == path.name or logical_path.endswith(f"/{path.name}"):
            raise RetirementScanError("allowlist cannot reference itself")
        if not detector or type(count) is not int or count <= 0:
            raise RetirementScanError("allowlist count/detector is invalid")
        if not reason:
            raise RetirementScanError("allowlist reason is invalid")
        key = (logical_path, detector)
        if key in index:
            raise RetirementScanError(f"duplicate allowlist entry: {key}")
        normalized = dict(entry)
        normalized["path"] = logical_path
        normalized["expected_sha256"] = expected_sha
        index[key] = normalized
    return _sha256(payload), index


def _is_excluded_directory(relative: str) -> bool:
    if relative in EXCLUDED_ROOT_PREFIXES:
        return True
    return any(relative.startswith(f"{prefix}/") for prefix in EXCLUDED_ROOT_PREFIXES)


def _inventory(
    root: Path,
    allowlist_path: Path,
) -> dict[str, tuple[Path, tuple[int, int, int, int, int, int]]]:
    inventory: dict[str, tuple[Path, tuple[int, int, int, int, int, int]]] = {}
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept_dirs: list[str] = []
        for name in sorted(dirnames):
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            try:
                candidate_stat = os.lstat(candidate)
            except OSError as exc:
                raise RetirementScanError(f"inventory path changed: {candidate}: {exc}") from exc
            if stat.S_ISLNK(candidate_stat.st_mode):
                inventory[relative] = (candidate, _stat_fingerprint(candidate_stat))
                continue
            if (
                _is_excluded_directory(relative)
                or name in EXCLUDED_CACHE_DIR_NAMES
                or name.endswith(".egg-info")
            ):
                continue
            kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            path = current_path / name
            if path.absolute() == allowlist_path.absolute():
                continue
            relative = path.relative_to(root).as_posix()
            if (
                _is_excluded_directory(relative)
                or relative in EXCLUDED_LOGICAL_PATHS
                or name in EXCLUDED_FILE_NAMES
            ):
                continue
            try:
                path_stat = os.lstat(path)
            except OSError as exc:
                raise RetirementScanError(f"inventory file changed: {path}: {exc}") from exc
            if relative in inventory:
                raise RetirementScanError(f"duplicate repository inventory path: {relative}")
            inventory[relative] = (path, _stat_fingerprint(path_stat))
    return inventory


def _python_findings(text: str) -> Counter[str]:
    result: Counter[str] = Counter()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return result
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result["python.import-v16"] += sum(
                1 for alias in node.names if TEXT_TOKEN.search(alias.name)
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if TEXT_TOKEN.search(module):
                result["python.import-v16"] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            result["python.string-v16"] += len(TEXT_TOKEN.findall(node.value))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            pieces: list[str] = []

            def collect(value: ast.AST) -> bool:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    pieces.append(value.value)
                    return True
                if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
                    return collect(value.left) and collect(value.right)
                return False

            if collect(node) and TEXT_TOKEN.search("".join(pieces)):
                result["python.constructed-v16"] += 1
    return +result


def _detect(logical_path: str, payload: bytes, kind: str) -> Counter[str]:
    result: Counter[str] = Counter()
    path_count = len(PATH_TOKEN.findall(logical_path))
    if path_count:
        result["path.v16"] = path_count
    if kind == "symlink":
        target = payload.decode("utf-8", errors="replace")
        count = len(TEXT_TOKEN.findall(target))
        if count:
            result["symlink.target-v16"] = count
        return result
    if b"\x00" in payload:
        count = len(re.findall(rb"(?i)v16", payload)) + len(_BINARY_UTF16_V16.findall(payload))
        if count:
            result["binary.v16"] = count
        return result
    text = payload.decode("utf-8", errors="replace")
    count = len(TEXT_TOKEN.findall(text))
    if count:
        result["text.v16-token"] = count
    version_count = len(VERSION_16.findall(text))
    if version_count:
        result["text.version-16.0.0"] = version_count
    if logical_path.endswith(".py"):
        result.update(_python_findings(text))
    return +result


def _parse_utc_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RetirementScanError(f"{field} must be an RFC3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RetirementScanError(f"{field} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise RetirementScanError(f"{field} must be UTC")
    return parsed.astimezone(timezone.utc)


def _normalized_bindings(
    *,
    cutover_id: str,
    repo_root: str | Path,
    repo_sha256: str,
    skill_sha256: str,
    schedule_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if not cutover_id or "/" in cutover_id or "\\" in cutover_id or ".." in cutover_id:
        raise RetirementScanError("cutover_id is not a safe identifier")
    root_candidate = Path(repo_root).absolute()
    if root_candidate.is_symlink() or not root_candidate.is_dir():
        raise RetirementScanError("repo root must be a real directory")
    root = root_candidate.resolve(strict=True)
    if not isinstance(schedule_sha256, Mapping) or set(schedule_sha256) != set(
        REQUIRED_SCHEDULE_NAMES
    ):
        raise RetirementScanError("schedule SHA map must contain the exact nine schedules")
    schedules = {
        name: _require_sha256(schedule_sha256[name], f"schedule_sha256[{name}]")
        for name in sorted(REQUIRED_SCHEDULE_NAMES)
    }
    return {
        "cutover_id": cutover_id,
        "repo_root": str(root),
        "repo_sha256": _require_sha256(repo_sha256, "repo_sha256"),
        "skill_sha256": _require_sha256(skill_sha256, "skill_sha256"),
        "schedule_sha256": schedules,
    }


def validate_runtime_evidence(
    path: str | Path,
    *,
    cutover_id: str,
    repo_root: str | Path,
    repo_sha256: str,
    skill_sha256: str,
    schedule_sha256: Mapping[str, str],
    now_utc: datetime | None = None,
) -> tuple[str, dict[str, Any]]:
    evidence_path = Path(path)
    payload, value = _read_json(evidence_path)
    expected_keys = {
        "schema_version",
        "cutover_id",
        "repo_root",
        "repo_sha256",
        "skill_sha256",
        "schedule_sha256",
        "generated_at",
        "expires_at",
        "domains",
        "authority",
        "semantic_sha256",
    }
    if set(value) != expected_keys:
        raise RetirementScanError("runtime evidence has an unexpected shape")
    if value.get("schema_version") != RUNTIME_EVIDENCE_SCHEMA:
        raise RetirementScanError("unexpected runtime-evidence schema")
    if value.get("authority") is not False:
        raise RetirementScanError("runtime evidence must declare authority=false")
    semantic_sha = _require_sha256(value.get("semantic_sha256"), "semantic_sha256")
    if semantic_sha != _semantic_sha256(value):
        raise RetirementScanError("runtime-evidence semantic SHA mismatch")
    bindings = _normalized_bindings(
        cutover_id=cutover_id,
        repo_root=repo_root,
        repo_sha256=repo_sha256,
        skill_sha256=skill_sha256,
        schedule_sha256=schedule_sha256,
    )
    actual_bindings = {key: value.get(key) for key in bindings}
    if actual_bindings != bindings:
        raise RetirementScanError("runtime evidence cutover/hash binding mismatch")

    generated_at = _parse_utc_timestamp(value.get("generated_at"), "generated_at")
    expires_at = _parse_utc_timestamp(value.get("expires_at"), "expires_at")
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if expires_at <= generated_at or expires_at - generated_at > MAX_RUNTIME_EVIDENCE_AGE:
        raise RetirementScanError("runtime evidence freshness window is invalid")
    if now < generated_at or now > expires_at:
        raise RetirementScanError("runtime evidence is stale or future-dated")

    domains = value.get("domains")
    if not isinstance(domains, dict) or set(domains) != set(REQUIRED_RUNTIME_DOMAINS):
        raise RetirementScanError("runtime-evidence domains are incomplete")
    raw_identities: set[tuple[int, int]] = set()
    for name in REQUIRED_RUNTIME_DOMAINS:
        domain = domains[name]
        if not isinstance(domain, dict) or set(domain) != {
            "status",
            "matches",
            "evidence_path",
            "evidence_sha256",
            "exit_code",
        }:
            raise RetirementScanError(f"runtime domain has invalid shape: {name}")
        if domain["status"] != "COMPLETE" or domain["exit_code"] != 0:
            raise RetirementScanError(f"runtime domain is incomplete: {name}")
        if domain["matches"] != []:
            raise RetirementScanError(f"runtime domain still has v16 matches: {name}")
        raw_path = Path(str(domain["evidence_path"]))
        if not raw_path.is_absolute() or str(raw_path) != str(raw_path.absolute()):
            raise RetirementScanError(f"runtime domain evidence path is not absolute: {name}")
        raw_payload, raw_kind, raw_fingerprint = _stable_file_payload(raw_path)
        if raw_kind != "file":
            raise RetirementScanError(f"runtime domain evidence cannot be a symlink: {name}")
        expected_raw_sha = _require_sha256(domain["evidence_sha256"], f"runtime domain SHA: {name}")
        if _sha256(raw_payload) != expected_raw_sha:
            raise RetirementScanError(f"runtime domain raw evidence SHA mismatch: {name}")
        raw_identity = (raw_fingerprint[0], raw_fingerprint[1])
        if raw_identity in raw_identities:
            raise RetirementScanError("runtime domains must bind distinct raw evidence files")
        raw_identities.add(raw_identity)
        if _detect(f"runtime/{name}", raw_payload, raw_kind):
            raise RetirementScanError(f"runtime domain raw evidence still contains v16: {name}")
    return _sha256(payload), value


def _required_external_hashes(
    *,
    skill_sha256: str,
    schedule_sha256: Mapping[str, str],
) -> dict[str, str]:
    result = {SKILL_LOGICAL_PATH: _require_sha256(skill_sha256, "skill_sha256")}
    result.update(
        {
            SCHEDULE_LOGICAL_PATHS[name]: _require_sha256(
                schedule_sha256[name], f"schedule_sha256[{name}]"
            )
            for name in ACTIVE_SCHEDULE_NAMES
        }
    )
    return result


def scan_retirement(
    *,
    repo_root: str | Path,
    allowlist_path: str | Path,
    runtime_evidence_path: str | Path,
    cutover_id: str,
    repo_sha256: str,
    skill_sha256: str,
    schedule_sha256: Mapping[str, str],
    external_files: Mapping[str, str | Path],
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    bindings = _normalized_bindings(
        cutover_id=cutover_id,
        repo_root=repo_root,
        repo_sha256=repo_sha256,
        skill_sha256=skill_sha256,
        schedule_sha256=schedule_sha256,
    )
    root = Path(bindings["repo_root"])
    allowlist_candidate = Path(allowlist_path).absolute()
    # Read before resolving so a symlink cannot be laundered into a trusted
    # regular file outside the repository inventory.
    allowlist_sha, allowed = _load_allowlist(allowlist_candidate)
    allowlist = allowlist_candidate.resolve(strict=True)
    if allowlist != allowlist_candidate:
        raise RetirementScanError("allowlist path must be canonical and cannot be a symlink")
    try:
        allowlist.relative_to(root)
    except ValueError as exc:
        raise RetirementScanError("allowlist must be a regular file inside repo_root") from exc
    runtime_sha, _ = validate_runtime_evidence(
        runtime_evidence_path,
        **bindings,
        now_utc=now_utc,
    )

    if not isinstance(external_files, Mapping) or set(external_files) != set(
        REQUIRED_EXTERNAL_LOGICAL_PATHS
    ):
        raise RetirementScanError(
            "external files must contain the exact installed skill and nine schedules"
        )
    expected_external_hashes = _required_external_hashes(
        skill_sha256=bindings["skill_sha256"],
        schedule_sha256=bindings["schedule_sha256"],
    )

    inventory_before = _inventory(root, allowlist)
    files: list[tuple[str, Path, tuple[int, int, int, int, int, int] | None, str | None]] = [
        (logical_path, path, fingerprint, None)
        for logical_path, (path, fingerprint) in sorted(inventory_before.items())
    ]
    repo_logical_paths = set(inventory_before)
    external_identities: set[tuple[int, int]] = set()
    external_readback: dict[str, str] = {}
    for logical_path in sorted(REQUIRED_EXTERNAL_LOGICAL_PATHS):
        _valid_logical_path(logical_path)
        if logical_path in repo_logical_paths:
            raise RetirementScanError(
                f"external logical path collides with repository: {logical_path}"
            )
        raw_path = Path(external_files[logical_path])
        payload, kind, fingerprint = _stable_file_payload(raw_path)
        if kind != "file":
            raise RetirementScanError(f"external evidence cannot be a symlink: {logical_path}")
        digest = _sha256(payload)
        if digest != expected_external_hashes[logical_path]:
            raise RetirementScanError(f"external evidence SHA mismatch: {logical_path}")
        identity = (fingerprint[0], fingerprint[1])
        if identity in external_identities:
            raise RetirementScanError("external skill/schedules must be distinct files")
        external_identities.add(identity)
        external_readback[logical_path] = digest
        files.append((logical_path, raw_path, fingerprint, digest))

    findings: dict[tuple[str, str], dict[str, Any]] = {}
    for logical_path, path, expected_fingerprint, expected_digest in files:
        payload, kind, fingerprint = _stable_file_payload(path)
        if fingerprint != expected_fingerprint:
            raise RetirementScanError(f"scan inventory identity drifted: {logical_path}")
        file_sha = _sha256(payload)
        if expected_digest is not None and file_sha != expected_digest:
            raise RetirementScanError(f"external evidence changed during scan: {logical_path}")
        for detector, count in _detect(logical_path, payload, kind).items():
            key = (logical_path, detector)
            if key in findings:
                raise RetirementScanError(f"duplicate scanner finding key: {key}")
            findings[key] = {
                "path": logical_path,
                "detector": detector,
                "exact_count": count,
                "file_sha256": file_sha,
            }

    inventory_after = _inventory(root, allowlist)
    if {key: fingerprint for key, (_, fingerprint) in inventory_before.items()} != {
        key: fingerprint for key, (_, fingerprint) in inventory_after.items()
    }:
        raise RetirementScanError("repository inventory changed during retirement scan")
    final_allowlist_sha, _ = _load_allowlist(allowlist)
    if final_allowlist_sha != allowlist_sha:
        raise RetirementScanError("allowlist changed during retirement scan")
    final_runtime_sha, _ = validate_runtime_evidence(
        runtime_evidence_path,
        **bindings,
        now_utc=now_utc,
    )
    if final_runtime_sha != runtime_sha:
        raise RetirementScanError("runtime evidence changed during retirement scan")
    for logical_path, external_raw_path in external_files.items():
        payload, kind, _ = _stable_file_payload(Path(external_raw_path))
        if kind != "file" or _sha256(payload) != external_readback[logical_path]:
            raise RetirementScanError(f"external evidence changed during scan: {logical_path}")

    unknown = sorted(key for key in findings if key not in allowed)
    stale = sorted(key for key in allowed if key not in findings)
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(findings) & set(allowed)):
        finding = findings[key]
        entry = allowed[key]
        if (
            finding["exact_count"] != entry["exact_count"]
            or finding["file_sha256"] != entry["expected_sha256"]
        ):
            mismatches.append(
                {
                    "path": key[0],
                    "detector": key[1],
                    "actual_count": finding["exact_count"],
                    "expected_count": entry["exact_count"],
                    "actual_sha256": finding["file_sha256"],
                    "expected_sha256": entry["expected_sha256"],
                }
            )

    inventory_payload = json.dumps(
        {
            logical_path: list(fingerprint)
            for logical_path, (_, fingerprint) in sorted(inventory_before.items())
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    report = {
        "schema_version": SCAN_REPORT_SCHEMA,
        "status": "CLEAN" if not unknown and not stale and not mismatches else "BLOCKED",
        **bindings,
        "allowlist_sha256": allowlist_sha,
        "runtime_evidence_sha256": runtime_sha,
        "repository_inventory_sha256": _sha256(inventory_payload),
        "external_file_sha256": dict(sorted(external_readback.items())),
        "scanned_file_count": len(files),
        "allowed_finding_count": len(findings) - len(unknown),
        "unknown": [findings[key] for key in unknown],
        "stale_allowlist_entries": [{"path": key[0], "detector": key[1]} for key in stale],
        "mismatches": mismatches,
        "authority": False,
    }
    report["semantic_sha256"] = _semantic_sha256(report)
    if report["status"] != "CLEAN":
        raise RetirementScanError(json.dumps(report, sort_keys=True))
    return report


__all__ = [
    "ALLOWLIST_SCHEMA",
    "REQUIRED_EXTERNAL_LOGICAL_PATHS",
    "REQUIRED_RUNTIME_DOMAINS",
    "RUNTIME_EVIDENCE_SCHEMA",
    "RetirementScanError",
    "SCHEDULE_LOGICAL_PATHS",
    "SKILL_LOGICAL_PATH",
    "scan_retirement",
    "validate_runtime_evidence",
]
