#!/usr/bin/env python3
"""Verify the exact pre-cutover dirty inventory before final build or CAS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Final, NoReturn


MANIFEST_RELATIVE_PATH: Final = Path(
    "operations/unified_cutover/pre-cutover-dirty-inventory.json"
)
DEFAULT_MAIN_CHECKOUT: Final = Path("/Users/maxwell/mySpace/myQuant")
MANIFEST_KIND: Final = "system.migration.pre_cutover_dirty_inventory"
VERIFICATION_KIND: Final = "system.migration.pre_cutover_dirty_inventory_verification"

UNCONFIRMED: Final = "UNCONFIRMED"
ABSORBED: Final = "ABSORBED_IN_INTEGRATION_COMMIT"
EXPLICITLY_DISPOSITIONED: Final = "EXPLICITLY_DISPOSITIONED"
DISPOSITIONS: Final = (UNCONFIRMED, ABSORBED, EXPLICITLY_DISPOSITIONED)

MANIFEST_FIELDS: Final = {
    "kind",
    "contract_sha256",
    "captured_at",
    "main_head",
    "entries",
    "inventory_sha256",
    "integration_commit",
    "user_confirmed_at",
    "user_confirmed_by",
}
ENTRY_FIELDS: Final = {
    "status",
    "path",
    "byte_sha256",
    "size",
    "user_disposition",
    "disposition_reason",
}
CAPTURE_ENTRY_FIELDS: Final = ("status", "path", "byte_sha256", "size")
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
STATUS_CHARS: Final = frozenset(" MADRCU?!")

BLOCKED_MANIFEST = "PRE_CUTOVER_DIRTY_INVENTORY_MANIFEST_INVALID"
BLOCKED_HEAD_DRIFT = "PRE_CUTOVER_DIRTY_INVENTORY_HEAD_DRIFT"
BLOCKED_INVENTORY_DRIFT = "PRE_CUTOVER_DIRTY_INVENTORY_DRIFT"
BLOCKED_UNCONFIRMED = "PRE_CUTOVER_DIRTY_INVENTORY_UNCONFIRMED"
BLOCKED_INCOMPLETE = "PRE_CUTOVER_DIRTY_INVENTORY_DISPOSITIONS_INCOMPLETE"
BLOCKED_INTEGRATION = "PRE_CUTOVER_INTEGRATION_COMMIT_INVALID"
BLOCKED_CHECKOUT_DIRTY = "PRE_CUTOVER_INTEGRATION_CHECKOUT_DIRTY"
BLOCKED_ABSORPTION = "PRE_CUTOVER_INTEGRATION_ABSORPTION_MISMATCH"


class PreCutoverPreflightError(ValueError):
    """Fail-closed preflight error with a stable blocker code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact sorted-key UTF-8 JSON bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _contract_sha256() -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "field_names": sorted(MANIFEST_FIELDS),
                "kind": MANIFEST_KIND,
                "strict_fields": True,
            }
        )
    )


MANIFEST_CONTRACT_SHA256: Final = _contract_sha256()


def _fail(code: str, detail: str) -> NoReturn:
    raise PreCutoverPreflightError(code, detail)


def _object_without_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(BLOCKED_MANIFEST, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    _fail(BLOCKED_MANIFEST, f"non-finite JSON value: {value}")


def _parse_canonical_json(raw: bytes) -> dict[str, Any]:
    try:
        document = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(BLOCKED_MANIFEST, f"manifest is not valid UTF-8 JSON: {exc}")
    if type(document) is not dict:
        _fail(BLOCKED_MANIFEST, "manifest root must be an object")
    if raw != canonical_json_bytes(document):
        _fail(BLOCKED_MANIFEST, "manifest is not exact compact canonical JSON")
    return document


def _canonical_relative_path(value: Any, *, label: str) -> str:
    if type(value) is not str:
        _fail(BLOCKED_MANIFEST, f"{label} must be text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail(BLOCKED_MANIFEST, f"{label} is not a canonical relative path")
    return value


def _canonical_utc(value: Any, *, label: str) -> str:
    if type(value) is not str:
        _fail(BLOCKED_MANIFEST, f"{label} must be canonical UTC text")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        _fail(BLOCKED_MANIFEST, f"{label} must use YYYY-MM-DDTHH:MM:SSZ")
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        _fail(BLOCKED_MANIFEST, f"{label} is not canonical UTC")
    return value


def _canonical_confirmation_text(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 160
        or any(ord(character) < 32 for character in value)
    ):
        _fail(BLOCKED_MANIFEST, "user_confirmed_by is not canonical text")
    return value


def _read_stable_regular(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        _fail(BLOCKED_INVENTORY_DRIFT, f"inventory path is unavailable: {path}: {exc}")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        _fail(BLOCKED_INVENTORY_DRIFT, f"inventory path is not a regular file: {path}")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        _fail(BLOCKED_INVENTORY_DRIFT, f"inventory path cannot be opened: {path}: {exc}")
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        _fail(BLOCKED_INVENTORY_DRIFT, f"inventory path changed after read: {path}: {exc}")
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    opened_identity = (
        opened.st_dev,
        opened.st_ino,
        opened.st_mode,
        opened.st_size,
        opened.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != opened_identity or before_identity != after_identity:
        _fail(BLOCKED_INVENTORY_DRIFT, f"inventory path changed while read: {path}")
    return b"".join(chunks)


def _git(checkout_root: Path, arguments: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=checkout_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(BLOCKED_INTEGRATION, f"git read failed for {checkout_root}: {exc}")
    return completed.stdout


def _git_head(checkout_root: Path) -> str:
    head = _git(checkout_root, ("rev-parse", "HEAD")).decode("ascii").strip()
    if COMMIT_RE.fullmatch(head) is None:
        _fail(BLOCKED_INTEGRATION, "checkout HEAD is not a canonical commit SHA")
    return head


def _dirty_rows(checkout_root: Path) -> list[dict[str, Any]]:
    raw = _git(
        checkout_root,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--no-renames",
        ),
    )
    rows: list[dict[str, Any]] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        if len(item) < 4 or item[2:3] != b" ":
            _fail(BLOCKED_INVENTORY_DRIFT, "git returned malformed porcelain output")
        try:
            status_text = item[:2].decode("ascii")
            relative = item[3:].decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            _fail(BLOCKED_INVENTORY_DRIFT, "git status contains non-canonical text")
        if (
            len(status_text) != 2
            or status_text == "  "
            or any(character not in STATUS_CHARS for character in status_text)
        ):
            _fail(BLOCKED_INVENTORY_DRIFT, f"unsupported git status: {status_text!r}")
        relative = _canonical_relative_path(relative, label="git status path")
        content = _read_stable_regular(checkout_root / relative)
        rows.append(
            {
                "status": status_text,
                "path": relative,
                "byte_sha256": sha256_bytes(content),
                "size": len(content),
            }
        )
    rows.sort(key=lambda row: row["path"].encode("utf-8"))
    if len({row["path"] for row in rows}) != len(rows):
        _fail(BLOCKED_INVENTORY_DRIFT, "git status contains duplicate paths")
    return rows


def _capture_preimage(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "captured_at": document["captured_at"],
        "main_head": document["main_head"],
        "entries": [
            {field: entry[field] for field in CAPTURE_ENTRY_FIELDS}
            for entry in document["entries"]
        ],
    }


def build_unconfirmed_inventory(
    checkout_root: Path,
    *,
    captured_at: str,
) -> dict[str, Any]:
    """Build an in-memory unconfirmed capture without writing any file."""

    captured_at = _canonical_utc(captured_at, label="captured_at")
    capture_rows = _dirty_rows(checkout_root.resolve())
    if not capture_rows:
        _fail(BLOCKED_INVENTORY_DRIFT, "pre-cutover inventory must not be empty")
    entries = [
        {
            **row,
            "user_disposition": UNCONFIRMED,
            "disposition_reason": None,
        }
        for row in capture_rows
    ]
    document: dict[str, Any] = {
        "kind": MANIFEST_KIND,
        "contract_sha256": MANIFEST_CONTRACT_SHA256,
        "captured_at": captured_at,
        "main_head": _git_head(checkout_root.resolve()),
        "entries": entries,
        "inventory_sha256": "",
        "integration_commit": None,
        "user_confirmed_at": None,
        "user_confirmed_by": None,
    }
    document["inventory_sha256"] = sha256_bytes(
        canonical_json_bytes(_capture_preimage(document))
    )
    return document


def validate_manifest_bytes(raw: bytes) -> dict[str, Any]:  # noqa: C901
    """Validate canonical bytes and the complete capture/disposition contract."""

    document = _parse_canonical_json(raw)
    if set(document) != MANIFEST_FIELDS:
        _fail(BLOCKED_MANIFEST, "manifest fields are not exact")
    if document["kind"] != MANIFEST_KIND:
        _fail(BLOCKED_MANIFEST, "manifest kind is invalid")
    if document["contract_sha256"] != MANIFEST_CONTRACT_SHA256:
        _fail(BLOCKED_MANIFEST, "manifest contract SHA is invalid")
    captured_at = _canonical_utc(document["captured_at"], label="captured_at")
    if type(document["main_head"]) is not str or COMMIT_RE.fullmatch(
        document["main_head"]
    ) is None:
        _fail(BLOCKED_MANIFEST, "main_head is not a canonical commit SHA")

    entries = document["entries"]
    if type(entries) is not list or not entries:
        _fail(BLOCKED_MANIFEST, "entries must be a non-empty list")
    paths: list[str] = []
    dispositions: list[str] = []
    for index, entry in enumerate(entries):
        if type(entry) is not dict or set(entry) != ENTRY_FIELDS:
            _fail(BLOCKED_MANIFEST, f"entry {index} fields are not exact")
        status_text = entry["status"]
        if (
            type(status_text) is not str
            or len(status_text) != 2
            or status_text == "  "
            or any(character not in STATUS_CHARS for character in status_text)
        ):
            _fail(BLOCKED_MANIFEST, f"entry {index} status is invalid")
        path = _canonical_relative_path(entry["path"], label=f"entry {index} path")
        if type(entry["byte_sha256"]) is not str or SHA256_RE.fullmatch(
            entry["byte_sha256"]
        ) is None:
            _fail(BLOCKED_MANIFEST, f"entry {index} byte SHA is invalid")
        if type(entry["size"]) is not int or entry["size"] < 0:
            _fail(BLOCKED_MANIFEST, f"entry {index} size is invalid")
        disposition = entry["user_disposition"]
        reason = entry["disposition_reason"]
        if disposition not in DISPOSITIONS:
            _fail(BLOCKED_MANIFEST, f"entry {index} disposition is invalid")
        if disposition == UNCONFIRMED:
            if reason is not None:
                _fail(BLOCKED_MANIFEST, "unconfirmed rows cannot have a disposition reason")
        elif (
            type(reason) is not str
            or not reason
            or reason != reason.strip()
            or len(reason) > 500
            or any(ord(character) < 32 for character in reason)
        ):
            _fail(BLOCKED_MANIFEST, "confirmed rows require a canonical reason")
        paths.append(path)
        dispositions.append(disposition)
    if paths != sorted(set(paths), key=lambda value: value.encode("utf-8")):
        _fail(BLOCKED_MANIFEST, "entry paths are not sorted and unique")

    if type(document["inventory_sha256"]) is not str or SHA256_RE.fullmatch(
        document["inventory_sha256"]
    ) is None:
        _fail(BLOCKED_MANIFEST, "inventory_sha256 is invalid")
    observed_inventory_sha = sha256_bytes(
        canonical_json_bytes(_capture_preimage(document))
    )
    if observed_inventory_sha != document["inventory_sha256"]:
        _fail(BLOCKED_MANIFEST, "inventory capture hash is invalid")

    has_unconfirmed = UNCONFIRMED in dispositions
    integration_commit = document["integration_commit"]
    confirmed_at = document["user_confirmed_at"]
    confirmed_by = document["user_confirmed_by"]
    if has_unconfirmed:
        if any(value is not None for value in (integration_commit, confirmed_at, confirmed_by)):
            _fail(
                BLOCKED_MANIFEST,
                "confirmation metadata must remain null while any row is unconfirmed",
            )
    else:
        if type(integration_commit) is not str or COMMIT_RE.fullmatch(
            integration_commit
        ) is None:
            _fail(BLOCKED_MANIFEST, "integration_commit is not a canonical commit SHA")
        confirmed_at = _canonical_utc(confirmed_at, label="user_confirmed_at")
        if confirmed_at <= captured_at:
            _fail(BLOCKED_MANIFEST, "user confirmation must be later than capture")
        _canonical_confirmation_text(confirmed_by)
    return document


def _read_manifest(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        _fail(BLOCKED_MANIFEST, f"manifest is unavailable: {exc}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail(BLOCKED_MANIFEST, "manifest must be a regular non-symlink file")
    return _read_stable_regular(path)


def _expected_capture_rows(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {field: entry[field] for field in CAPTURE_ENTRY_FIELDS}
        for entry in document["entries"]
    ]


def _verify_unconfirmed_capture(
    document: Mapping[str, Any], checkout_root: Path
) -> None:
    observed_head = _git_head(checkout_root)
    if observed_head != document["main_head"]:
        _fail(
            BLOCKED_HEAD_DRIFT,
            f"checkout HEAD drifted: expected {document['main_head']}, observed {observed_head}",
        )
    observed_rows = _dirty_rows(checkout_root)
    expected_rows = _expected_capture_rows(document)
    if observed_rows != expected_rows:
        _fail(BLOCKED_INVENTORY_DRIFT, "checkout status, path, size, or bytes drifted")


def _git_blob(checkout_root: Path, commit: str, relative_path: str) -> bytes:
    try:
        return _git(checkout_root, ("show", f"{commit}:{relative_path}"))
    except PreCutoverPreflightError:
        _fail(
            BLOCKED_ABSORPTION,
            f"absorbed path is absent from integration commit: {relative_path}",
        )


def _verify_integration_commit(
    document: Mapping[str, Any], checkout_root: Path
) -> dict[str, Any]:
    integration_commit = document["integration_commit"]
    observed_head = _git_head(checkout_root)
    if observed_head != integration_commit:
        _fail(
            BLOCKED_INTEGRATION,
            f"checkout HEAD is not the confirmed integration commit: {observed_head}",
        )
    if _dirty_rows(checkout_root):
        _fail(BLOCKED_CHECKOUT_DIRTY, "confirmed integration checkout is not clean")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", document["main_head"], integration_commit],
            cwd=checkout_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        _fail(BLOCKED_INTEGRATION, "integration commit is not a descendant of capture HEAD")

    absorbed_count = 0
    explicit_count = 0
    for entry in document["entries"]:
        if entry["user_disposition"] == ABSORBED:
            raw = _git_blob(checkout_root, integration_commit, entry["path"])
            if len(raw) != entry["size"] or sha256_bytes(raw) != entry["byte_sha256"]:
                _fail(
                    BLOCKED_ABSORPTION,
                    f"absorbed bytes do not match capture: {entry['path']}",
                )
            absorbed_count += 1
        elif entry["user_disposition"] == EXPLICITLY_DISPOSITIONED:
            explicit_count += 1
        else:
            _fail(BLOCKED_INCOMPLETE, "every captured row needs an explicit disposition")
    return {
        "absorbed_count": absorbed_count,
        "explicitly_dispositioned_count": explicit_count,
    }


def verify_pre_cutover_preflight(
    *,
    manifest_path: Path,
    checkout_root: Path,
) -> dict[str, Any]:
    """Verify the dirty-inventory gate without mutating either checkout."""

    document = validate_manifest_bytes(_read_manifest(manifest_path))
    checkout_root = checkout_root.resolve()
    dispositions = [entry["user_disposition"] for entry in document["entries"]]
    if UNCONFIRMED in dispositions:
        _verify_unconfirmed_capture(document, checkout_root)
        if any(disposition != UNCONFIRMED for disposition in dispositions):
            _fail(BLOCKED_INCOMPLETE, "captured rows are only partially dispositioned")
        _fail(BLOCKED_UNCONFIRMED, "all captured rows still require explicit user disposition")

    counts = _verify_integration_commit(document, checkout_root)
    return {
        "kind": VERIFICATION_KIND,
        "verified": True,
        "status": "GATE_SATISFIED",
        "capture_main_head": document["main_head"],
        "integration_commit": document["integration_commit"],
        "inventory_sha256": document["inventory_sha256"],
        "entry_count": len(document["entries"]),
        **counts,
        "dirty_inventory_gate_satisfied": True,
        "eligible_for_next_preflight_gate": True,
        "final_build_authorized": False,
        "cas_authorized": False,
        "external_write_performed": False,
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        result = verify_pre_cutover_preflight(
            manifest_path=repository_root / MANIFEST_RELATIVE_PATH,
            checkout_root=DEFAULT_MAIN_CHECKOUT,
        )
    except PreCutoverPreflightError as exc:
        blocked = {
            "kind": VERIFICATION_KIND,
            "verified": False,
            "status": "BLOCKED",
            "blocker_code": exc.code,
            "detail": str(exc),
            "dirty_inventory_gate_satisfied": False,
            "eligible_for_next_preflight_gate": False,
            "final_build_authorized": False,
            "cas_authorized": False,
            "external_write_performed": False,
        }
        print(canonical_json_bytes(blocked).decode("utf-8"), file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
