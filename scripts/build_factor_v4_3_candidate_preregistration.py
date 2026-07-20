#!/usr/bin/env python3
"""Publish or explicitly read back the fixed v4.3 preregistration bundle.

The command is deliberately definition-only and has no configurable source,
cycle, candidate, outcome, governance-mutation, provider, portfolio, or
execution surface.  ``publish`` reads the exact frozen A_quant Git blobs and
the exact immutable CN snapshot.  ``readback`` reopens only the one explicit
absolute bundle path supplied by the caller.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.factors import (  # noqa: E402
    governance_candidate_preregistration_bundle_v4_3 as bundle_v4_3,
)
from quant_investor.factors import (  # noqa: E402
    governance_candidate_preregistration_v4_3 as prereg_v4_3,
)
from quant_investor.factors import (  # noqa: E402
    governance_source_readback_v4_1 as source_readback_v4_1,
)


PRODUCTION_PRIVATE_ROOT = Path(
    "/Users/maxwell/mySpace/myQuant/reports/factor_governance/private/"
    "v4_3_candidate_preregistration"
)
FIXED_CYCLE_ID = "cn_full_a_v4_3_20260717_20260717T172132Z"
AQUANT_GIT_TOP = Path(prereg_v4_3.AQUANT_GIT_TOP)

SNAPSHOT_ID = "20260717T172132Z"
ANALYSIS_START = "2021-06-25"
CUTOFF_DATE = "2026-07-17"
LATEST_COMPLETE_TRADE_DATE = "20260717"
EXPECTED_FULL_A_COUNT = 5502
EXPECTED_SERVING_INVENTORY_COUNT = 5728
EXPECTED_FULL_A_SEMANTIC_SHA256 = (
    "41ad09c4c6f759714682ffce4420f6cbb9c2bc34827f443bb4f6965485e69721"
)
EXPECTED_CALENDAR_SEMANTIC_SHA256 = (
    "99be5e97027fa1837eb737bd6aa4d1adee57107a3592ed14c30858dc5be28f48"
)
EXPECTED_TABLE_INVENTORY_SEMANTIC_SHA256 = (
    "d3b281045dfa34af49371a2847877920a062ac077aeee8525d381fc4713a7330"
)
EXPECTED_SERVING_INVENTORY_SEMANTIC_SHA256 = (
    "fd15330350fff4e92684d7dfb6bf4b5077ba9e547aa3321f94db3b957ff4e7bc"
)

LATEST_POINTER_PATH = PROJECT_ROOT / "data" / "parquet" / "cn" / "_latest.json"
SNAPSHOT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "parquet" / "cn" / "_snapshots" / f"{SNAPSHOT_ID}.json"
)
COMPONENTS_PATH = PROJECT_ROOT / "data" / "cn_universe" / "cn_index_components.json"
PIT_GENERATION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "parquet"
    / "cn"
    / "reference"
    / "_generations"
    / "pit-20260717-5a3853ca2dd955e3"
)
PIT_GENERATION_MANIFEST_PATH = PIT_GENERATION_ROOT / "manifest.json"
PIT_MEMBERSHIP_PATH = PIT_GENERATION_ROOT / "stock_basic_membership.parquet"
TABLE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "parquet"
    / "cn"
    / "_snapshots"
    / SNAPSHOT_ID
    / "table"
    / "bars"
)
SERVING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "parquet"
    / "cn"
    / "_snapshots"
    / SNAPSHOT_ID
    / "serving"
    / "bars"
)

EXPECTED_LATEST_POINTER_SHA256 = (
    "551a16aef636630ab25f34ddd8b8a1ca343e993a529678d2222ee402f16ff285"
)
EXPECTED_SNAPSHOT_MANIFEST_SHA256 = (
    "11b0edbc69609d07fa6bcaba33936ffdc7d15ab3f44845a9c658583e89cf1f71"
)
EXPECTED_COMPONENTS_SHA256 = (
    "35b8f45b559dfe3c15459cf817d1fef74aca22df410d4f5b02426e65be618f60"
)
EXPECTED_PIT_GENERATION_MANIFEST_SHA256 = (
    "9c9c9ee1af849e669e50d0593e524f42573ae9d9f367f185414574abc79125b1"
)
EXPECTED_PIT_MEMBERSHIP_SHA256 = (
    "6a4d42edd9581c5cf8ba6a472e3b89bd1747eb1c1731f07cfd53efc949ebf4e1"
)

CODE_BINDING_PATHS: tuple[str, ...] = (
    "scripts/build_factor_v4_3_candidate_preregistration.py",
    "quant_investor/factors/governance_candidate_preregistration_v4_3.py",
    "quant_investor/factors/governance_candidate_preregistration_bundle_v4_3.py",
    "quant_investor/factors/governance_cycle_state_v4_1.py",
    "quant_investor/factors/governance_private_bundle_io.py",
    "quant_investor/factors/governance_source_readback_v4_1.py",
    "quant_investor/factors/governance_screening_v4.py",
    "quant_investor/codex_review/storage.py",
    "quant_investor/market/pit_universe.py",
    "quant_investor/factors/governance_source_v4_1.py",
)

PROTECTED_BINDING_SPECS: tuple[tuple[str, Path, str], ...] = (
    (
        "registry",
        PROJECT_ROOT / "quant_investor" / "factor_registry" / "mined_factors.json",
        "b8369dfef7d27156999e93e3a1a12020e072db0296532fee10b0335d8bddca2f",
    ),
    (
        "latest_pointer",
        LATEST_POINTER_PATH,
        EXPECTED_LATEST_POINTER_SHA256,
    ),
    (
        "catalog",
        PROJECT_ROOT / "data" / "parquet" / "cn" / "_catalog.json",
        "ffb1f42be3c53924d515a7b5ac27a3e4d85e0516030d7854fe1251c89377ddd8",
    ),
    (
        "fundamental_latest",
        PROJECT_ROOT / "data" / "parquet" / "cn" / "_fundamental_latest.json",
        "eeb5d2f584f5351f024f520125894843c53ff7dc9cfcdaa165c1002345e37bbd",
    ),
    (
        "latest_manifest",
        PROJECT_ROOT / "data" / "parquet" / "cn" / "latest_manifest.json",
        "a4a6df91cffe9495759e475f3fe5f6e227d82c3e3b0e3bc7912d056cb3f9adf5",
    ),
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OID_RE = re.compile(r"[0-9a-f]{40}")
_FORBIDDEN_ARGUMENT_TOKENS = (
    "--root",
    "--private-root",
    "--cutoff",
    "--snapshot",
    "--cycle",
    "--candidate",
    "--outcome",
    "--registry",
    "--proposal",
    "--replay",
    "--transaction",
    "--apply",
    "--provider",
    "--llm",
    "--portfolio",
    "--broker",
    "--order",
    "--trade",
    "--scan",
    "--latest",
    "--fallback",
)


class FactorV4_3CandidatePreregistrationRunnerError(ValueError):
    """Raised when the fixed v4.3 runner rejects an input fail closed."""


@dataclass(frozen=True)
class StableFile:
    """One stable, current-owner, single-link regular-file observation."""

    path: Path
    raw: bytes
    byte_sha256: str
    size_bytes: int
    mode: int
    uid: int
    nlink: int
    signature: tuple[int, ...]


@dataclass(frozen=True)
class PublicationInputs:
    """All fixed input bytes and descriptors supplied to the bundle layer."""

    aquant_git_objects: dict[str, bytes]
    strict_source_binding: dict[str, Any]
    code_bindings: tuple[dict[str, Any], ...]
    protected_bindings: tuple[dict[str, Any], ...]
    runtime_fingerprint: dict[str, Any]


def _error(message: str) -> FactorV4_3CandidatePreregistrationRunnerError:
    return FactorV4_3CandidatePreregistrationRunnerError(message)


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _error(f"{label} must be a lowercase SHA-256")
    return value


def _absolute(value: Any, label: str) -> Path:
    if type(value) is not str or not value.startswith("/") or "\x00" in value:
        raise _error(f"{label} must be an absolute normalized path")
    path = Path(value)
    if os.path.abspath(value) != value or any(
        part in {"", ".", ".."} for part in path.parts[1:]
    ):
        raise _error(f"{label} must be an absolute normalized path")
    return path


def _signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_uid),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _stable_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    max_bytes: int = 512 * 1024 * 1024,
) -> StableFile:
    """Read a regular file twice through one no-follow descriptor."""

    expected = (
        None
        if expected_sha256 is None
        else _sha256(expected_sha256, f"{label} expected SHA-256")
    )
    descriptor: int | None = None
    try:
        before = os.lstat(path)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > max_bytes
        ):
            raise _error(f"{label} is not a stable owned regular file: {path}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _signature(opened) != _signature(before):
            raise _error(f"{label} changed while opening: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            first = handle.read(max_bytes + 1)
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            second = handle.read(max_bytes + 1)
        after = os.fstat(descriptor)
        if (
            len(first) > max_bytes
            or first != second
            or _signature(after) != _signature(opened)
        ):
            raise _error(f"{label} changed across stable readback: {path}")
        digest = hashlib.sha256(first).hexdigest()
        if expected is not None and digest != expected:
            raise _error(f"{label} SHA-256 mismatch: {path}")
        return StableFile(
            path=path,
            raw=first,
            byte_sha256=digest,
            size_bytes=len(first),
            mode=stat.S_IMODE(after.st_mode),
            uid=int(after.st_uid),
            nlink=int(after.st_nlink),
            signature=_signature(after),
        )
    except FactorV4_3CandidatePreregistrationRunnerError:
        raise
    except OSError as exc:
        raise _error(f"{label} is unavailable: {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_static_contract() -> None:
    if str(PRODUCTION_PRIVATE_ROOT) != (
        "/Users/maxwell/mySpace/myQuant/reports/factor_governance/private/"
        "v4_3_candidate_preregistration"
    ):
        raise _error("fixed private root drifted")
    if tuple(bundle_v4_3.ROOT_SUFFIX_V4_3) != (
        "reports",
        "factor_governance",
        "private",
        "v4_3_candidate_preregistration",
    ):
        raise _error("bundle private-root contract drifted")
    if bundle_v4_3.CYCLE_ID_V4_3 != FIXED_CYCLE_ID:
        raise _error("bundle cycle identity drifted")
    if tuple(bundle_v4_3.CODE_BINDING_PATHS_V4_3) != CODE_BINDING_PATHS:
        raise _error("bundle code-binding order drifted")
    if tuple(bundle_v4_3.PROTECTED_BINDING_NAMES_V4_3) != tuple(
        name for name, _path, _digest in PROTECTED_BINDING_SPECS
    ):
        raise _error("bundle protected-binding order drifted")
    if prereg_v4_3.AQUANT_GIT_TOP != str(AQUANT_GIT_TOP):
        raise _error("pure A_quant Git-top oracle drifted")
    if prereg_v4_3.AQUANT_COMMIT_V4_3 != (
        "4424dcecc384f614b0e9fd5e36cf094e9244bad5"
    ):
        raise _error("pure A_quant commit oracle drifted")


def _validate_private_root_preflight(root: Path) -> None:
    """Reject a missing, aliased, non-private, or already-used fixed root."""

    if not root.is_absolute() or os.path.abspath(root) != str(root):
        raise _error("private root must be absolute and normalized")
    suffix = tuple(bundle_v4_3.ROOT_SUFFIX_V4_3)
    if tuple(root.parts[-len(suffix) :]) != suffix:
        raise _error("private root must be the exact v4.3 preregistration lane")
    try:
        metadata = os.lstat(root)
    except OSError as exc:
        raise _error(f"fixed private root must already exist: {root}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise _error("fixed private root must be a current-owner 0700 directory")
    destination = root / FIXED_CYCLE_ID
    try:
        os.lstat(destination)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise _error(f"cannot inspect deterministic destination: {destination}") from exc
    raise _error(f"deterministic cycle destination already exists: {destination}")


def _run_git(arguments: Sequence[str]) -> bytes:
    """Run one read-only Git-object command at the frozen repository top."""

    try:
        metadata = os.lstat(AQUANT_GIT_TOP)
    except OSError as exc:
        raise _error("fixed A_quant Git top is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise _error("fixed A_quant Git top must be a real directory")
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
    }
    try:
        completed = subprocess.run(
            ["git", "-C", str(AQUANT_GIT_TOP), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _error(f"pinned A_quant Git-object read failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise _error(f"pinned A_quant Git-object read failed: {detail}")
    return completed.stdout


def _read_aquant_git_objects() -> dict[str, bytes]:
    """Read all eight exact definition sources without touching the worktree."""

    _validate_static_contract()
    commit = prereg_v4_3.AQUANT_COMMIT_V4_3
    if _OID_RE.fullmatch(commit) is None:
        raise _error("pure A_quant commit oracle is not a lowercase Git OID")
    resolved = _run_git(["rev-parse", "--verify", f"{commit}^{{commit}}"])
    if resolved.decode("ascii").strip() != commit:
        raise _error("A_quant commit resolution mismatch")

    objects: dict[str, bytes] = {}
    specs = prereg_v4_3.AQUANT_SOURCE_SPECS_V4_3
    expected_order = list(range(1, len(specs) + 1))
    if [row.get("order") for row in specs] != expected_order:
        raise _error("pure A_quant source order is not contiguous")
    for row in specs:
        path = row.get("git_tree_path")
        blob_oid = row.get("blob_oid")
        raw_sha256 = row.get("raw_sha256")
        mode = row.get("mode")
        if (
            type(path) is not str
            or not path.startswith("A_quant/")
            or Path(path).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(path).parts)
            or type(blob_oid) is not str
            or _OID_RE.fullmatch(blob_oid) is None
            or type(raw_sha256) is not str
            or _SHA256_RE.fullmatch(raw_sha256) is None
            or mode != "100644"
            or path in objects
        ):
            raise _error("pure A_quant source binding is malformed")
        tree_row = _run_git(["ls-tree", commit, "--", path]).decode(
            "utf-8"
        ).strip()
        prefix, separator, tree_path = tree_row.partition("\t")
        if (
            not separator
            or tree_path != path
            or prefix.split() != [mode, "blob", blob_oid]
        ):
            raise _error(f"A_quant pinned tree entry mismatch: {path}")
        raw = _run_git(["cat-file", "blob", blob_oid])
        if hashlib.sha256(raw).hexdigest() != raw_sha256:
            raise _error(f"A_quant pinned blob SHA-256 mismatch: {path}")
        objects[path] = raw

    prereg_v4_3.validate_runtime_fingerprint_v4_3(
        prereg_v4_3.runtime_fingerprint_v4_3()
    )
    prereg_v4_3.build_aquant_source_set_receipt_v4_3(
        aquant_git_objects=objects,
        runtime_fingerprint=prereg_v4_3.runtime_fingerprint_v4_3(),
    )
    return objects


def _build_code_bindings(repository_root: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for relative in CODE_BINDING_PATHS:
        observed = _stable_file(
            repository_root / relative,
            label=f"code binding {relative}",
            max_bytes=16 * 1024 * 1024,
        )
        rows.append(
            {
                "relative_path": relative,
                "absolute_path": str(observed.path),
                "byte_sha256": observed.byte_sha256,
                "size_bytes": observed.size_bytes,
                "mode": observed.mode,
                "uid": observed.uid,
                "nlink": observed.nlink,
            }
        )
    return tuple(rows)


def _build_protected_bindings(
    specs: Sequence[tuple[str, Path, str]],
) -> tuple[dict[str, Any], ...]:
    if tuple(name for name, _path, _digest in specs) != tuple(
        bundle_v4_3.PROTECTED_BINDING_NAMES_V4_3
    ):
        raise _error("protected binding names/order differ from the fixed contract")
    rows: list[dict[str, Any]] = []
    for name, path, digest in specs:
        observed = _stable_file(
            path,
            label=f"protected binding {name}",
            expected_sha256=digest,
            max_bytes=64 * 1024 * 1024,
        )
        rows.append(
            {
                "name": name,
                "absolute_path": str(observed.path),
                "byte_sha256": observed.byte_sha256,
                "size_bytes": observed.size_bytes,
                "mode": observed.mode,
                "uid": observed.uid,
                "nlink": observed.nlink,
            }
        )
    return tuple(rows)


def _build_strict_source_binding() -> dict[str, Any]:
    bound = source_readback_v4_1.bind_explicit_cutoff_inputs_v4_1(
        latest_pointer_path=LATEST_POINTER_PATH,
        expected_latest_pointer_sha256=EXPECTED_LATEST_POINTER_SHA256,
        snapshot_manifest_path=SNAPSHOT_MANIFEST_PATH,
        expected_snapshot_manifest_sha256=EXPECTED_SNAPSHOT_MANIFEST_SHA256,
        components_path=COMPONENTS_PATH,
        expected_components_sha256=EXPECTED_COMPONENTS_SHA256,
        expected_full_a_semantic_sha256=EXPECTED_FULL_A_SEMANTIC_SHA256,
        pit_generation_manifest_path=PIT_GENERATION_MANIFEST_PATH,
        expected_pit_generation_manifest_sha256=(
            EXPECTED_PIT_GENERATION_MANIFEST_SHA256
        ),
        pit_membership_path=PIT_MEMBERSHIP_PATH,
        expected_pit_membership_sha256=EXPECTED_PIT_MEMBERSHIP_SHA256,
        table_root=TABLE_ROOT,
        snapshot_id=SNAPSHOT_ID,
        analysis_start=ANALYSIS_START,
        cutoff_date=CUTOFF_DATE,
        expected_full_a_count=EXPECTED_FULL_A_COUNT,
        expected_serving_inventory_count=EXPECTED_SERVING_INVENTORY_COUNT,
    )
    if (
        bound.binding["snapshot_id"] != SNAPSHOT_ID
        or bound.binding["cutoff_date"] != CUTOFF_DATE
        or bound.binding["table"]["inventory_sha256"]
        != EXPECTED_TABLE_INVENTORY_SEMANTIC_SHA256
        or bound.binding["components"]["count"] != EXPECTED_FULL_A_COUNT
        or bound.binding["components"]["newline_set_sha256"]
        != EXPECTED_FULL_A_SEMANTIC_SHA256
    ):
        raise _error("strict v4.1 backend binding differs from the fixed source oracle")
    strict = bundle_v4_3.build_strict_full_a_source_binding_v4_3(
        bound_inputs=bound
    )
    if (
        strict.get("protocol_version") != "v4"
        or strict.get("snapshot_id") != SNAPSHOT_ID
        or strict.get("cutoff") != CUTOFF_DATE
        or strict.get("latest_available_trade_date")
        != LATEST_COMPLETE_TRADE_DATE
        or strict.get("latest_complete_trade_date")
        != LATEST_COMPLETE_TRADE_DATE
        or strict.get("expected_scope_count") != EXPECTED_FULL_A_COUNT
        or strict.get("full_a_scope_sha256")
        != EXPECTED_FULL_A_SEMANTIC_SHA256
        or strict.get("calendar_semantic_sha256")
        != EXPECTED_CALENDAR_SEMANTIC_SHA256
        or strict.get("table_inventory_semantic_sha256")
        != EXPECTED_TABLE_INVENTORY_SEMANTIC_SHA256
        or strict.get("serving_inventory_semantic_sha256")
        != EXPECTED_SERVING_INVENTORY_SEMANTIC_SHA256
    ):
        raise _error("strict v4.3 source binding differs from the fixed source oracle")
    return copy.deepcopy(dict(strict))


def _collect_publication_inputs(
    *,
    repository_root: Path,
    protected_specs: Sequence[tuple[str, Path, str]],
) -> PublicationInputs:
    _validate_static_contract()
    runtime = prereg_v4_3.validate_runtime_fingerprint_v4_3(
        prereg_v4_3.runtime_fingerprint_v4_3()
    )
    return PublicationInputs(
        aquant_git_objects=_read_aquant_git_objects(),
        strict_source_binding=_build_strict_source_binding(),
        code_bindings=_build_code_bindings(repository_root),
        protected_bindings=_build_protected_bindings(protected_specs),
        runtime_fingerprint=runtime,
    )


def _current_preregistered_at() -> str:
    return (
        datetime.now(ZoneInfo("Asia/Shanghai"))
        .replace(microsecond=0)
        .isoformat(timespec="seconds")
    )


def _validated_report_descriptor(
    published: Mapping[str, Any],
    *,
    bundle_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptors = published.get("artifact_descriptors")
    if not isinstance(descriptors, Mapping):
        raise _error("bundle result artifact_descriptors must be a mapping")
    report_name = bundle_v4_3.READBACK_REPORT_FILENAME_V4_3
    descriptor = descriptors.get(report_name)
    if not isinstance(descriptor, Mapping):
        raise _error("bundle result is missing the readback report descriptor")
    fields = {
        "absolute_path",
        "byte_sha256",
        "size_bytes",
        "mode",
        "uid",
        "nlink",
    }
    if set(descriptor) != fields:
        raise _error("readback report descriptor fields mismatch")
    path = _absolute(descriptor["absolute_path"], "readback report path")
    if path != bundle_path / report_name:
        raise _error("readback report path differs from the fixed bundle path")
    _sha256(descriptor["byte_sha256"], "readback report byte SHA-256")
    if (
        type(descriptor["size_bytes"]) is not int
        or descriptor["size_bytes"] <= 0
        or descriptor["mode"] != 0o600
        or descriptor["uid"] != os.getuid()
        or descriptor["nlink"] != 1
    ):
        raise _error("readback report private descriptor contract failed")
    report = published.get("readback_report")
    if not isinstance(report, Mapping):
        raise _error("bundle result is missing the readback report")
    semantic = report.get("artifact_semantic_sha256")
    _sha256(semantic, "readback report semantic SHA-256")
    return dict(descriptor), dict(report)


def run_publish(
    _args: argparse.Namespace,
    *,
    private_root: Path = PRODUCTION_PRIVATE_ROOT,
    repository_root: Path = PROJECT_ROOT,
    protected_specs: Sequence[tuple[str, Path, str]] = PROTECTED_BINDING_SPECS,
    preregistered_at_factory: Callable[[], str] = _current_preregistered_at,
) -> dict[str, Any]:
    """Publish the one fixed bundle after exact before/under-lock readback."""

    _validate_static_contract()
    _validate_private_root_preflight(private_root)
    preregistered_at = preregistered_at_factory()
    entry = _collect_publication_inputs(
        repository_root=repository_root,
        protected_specs=protected_specs,
    )

    def revalidate_inputs() -> None:
        locked = _collect_publication_inputs(
            repository_root=repository_root,
            protected_specs=protected_specs,
        )
        if locked != entry:
            raise _error("publication inputs changed before commit")

    published = bundle_v4_3.publish_candidate_preregistration_bundle_v4_3(
        private_root=private_root,
        repository_root=repository_root,
        preregistered_at=preregistered_at,
        aquant_git_objects=entry.aquant_git_objects,
        strict_source_binding=entry.strict_source_binding,
        code_bindings=entry.code_bindings,
        protected_bindings=entry.protected_bindings,
        revalidate_inputs=revalidate_inputs,
    )
    expected_bundle = private_root / FIXED_CYCLE_ID
    bundle_path = _absolute(published.get("bundle_path"), "published bundle path")
    if bundle_path != expected_bundle:
        raise _error("published bundle path differs from the fixed cycle")
    if (
        published.get("accepted") is not True
        or published.get("publication_phase") != "COMMITTED"
        or published.get("exclusive_rename_completed") is not True
        or published.get("durability_commit_verified") is not True
        or published.get("publication_authority") is not True
    ):
        raise _error("bundle publisher did not prove a durable exclusive commit")
    descriptor, report = _validated_report_descriptor(
        published,
        bundle_path=bundle_path,
    )
    if (
        report.get("publication_phase") != "PRECOMMIT_INTENT_ONLY"
        or report.get("exclusive_rename_completed") is not False
        or report.get("durability_commit_verified") is not False
        or report.get("publication_authority") is not False
        or report.get("side_effects") != prereg_v4_3.SIDE_EFFECT_FLAGS_V4_3
    ):
        raise _error("in-bundle report must remain PRECOMMIT_INTENT_ONLY")
    return {
        "accepted": True,
        "status": "COMMITTED",
        "mode": "publish",
        "protocol_version": "v4",
        "evidence_contract_version": "v4.3",
        "cycle_id": FIXED_CYCLE_ID,
        "candidate_names": list(prereg_v4_3.EXPECTED_CANDIDATES_V4_3),
        "preregistered_at": preregistered_at,
        "bundle_path": str(bundle_path),
        "readback_report_path": descriptor["absolute_path"],
        "readback_report_byte_sha256": descriptor["byte_sha256"],
        "readback_report_semantic_sha256": report["artifact_semantic_sha256"],
        "publication_phase": "COMMITTED",
        "exclusive_rename_completed": True,
        "durability_commit_verified": True,
        "internal_readback_report_phase": "PRECOMMIT_INTENT_ONLY",
        "authority": copy.deepcopy(prereg_v4_3.AUTHORITY_FLAGS_V4_3),
        "side_effects": copy.deepcopy(prereg_v4_3.SIDE_EFFECT_FLAGS_V4_3),
    }


def run_readback(args: argparse.Namespace) -> dict[str, Any]:
    """Explicitly reopen only the fixed historical bundle and expected report."""

    _validate_static_contract()
    bundle_path = _absolute(args.bundle_path, "bundle path")
    if bundle_path != PRODUCTION_PRIVATE_ROOT / FIXED_CYCLE_ID:
        raise _error("bundle path must be the exact fixed v4.3 cycle")
    expected_byte = _sha256(
        args.expected_readback_report_byte_sha256,
        "expected readback report byte SHA-256",
    )
    expected_semantic = _sha256(
        args.expected_readback_report_semantic_sha256,
        "expected readback report semantic SHA-256",
    )
    result = bundle_v4_3.readback_candidate_preregistration_bundle_v4_3(
        bundle_path=bundle_path,
        expected_readback_report_byte_sha256=expected_byte,
        expected_readback_report_semantic_sha256=expected_semantic,
    )
    if (
        result.get("accepted") is not True
        or result.get("expected_hashes_verified") is not True
        or result.get("readback_report_byte_sha256") != expected_byte
        or result.get("readback_report_semantic_sha256") != expected_semantic
        or result.get("bundle_path") != str(bundle_path)
    ):
        raise _error("historical v4.3 bundle readback was not accepted")
    return {
        "accepted": True,
        "status": "READBACK_ACCEPTED",
        "mode": "readback",
        "protocol_version": "v4",
        "evidence_contract_version": "v4.3",
        "cycle_id": FIXED_CYCLE_ID,
        "candidate_names": list(prereg_v4_3.EXPECTED_CANDIDATES_V4_3),
        "bundle_path": str(bundle_path),
        "readback_report_byte_sha256": expected_byte,
        "readback_report_semantic_sha256": expected_semantic,
        "publication_phase_claimed": False,
        "current_latest_or_fallback_discovery_used": False,
        "authority": copy.deepcopy(prereg_v4_3.AUTHORITY_FLAGS_V4_3),
        "side_effects": copy.deepcopy(prereg_v4_3.SIDE_EFFECT_FLAGS_V4_3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("publish", help="publish the one fixed definition bundle")
    readback = commands.add_parser(
        "readback", help="reopen one explicit immutable fixed bundle"
    )
    readback.add_argument("--bundle-path", required=True)
    readback.add_argument("--expected-readback-report-byte-sha256", required=True)
    readback.add_argument(
        "--expected-readback-report-semantic-sha256", required=True
    )
    help_text = parser.format_help() + "".join(
        command.format_help() for command in commands.choices.values()
    )
    if any(token in help_text for token in _FORBIDDEN_ARGUMENT_TOKENS):
        raise _error("forbidden override or side-effect argument leaked into CLI")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_publish(args) if args.command == "publish" else run_readback(args)
    except Exception as exc:
        payload = {
            "accepted": False,
            "status": "REJECTED_FAIL_CLOSED",
            "detail": str(exc),
            "authority": copy.deepcopy(prereg_v4_3.AUTHORITY_FLAGS_V4_3),
            "side_effects": copy.deepcopy(prereg_v4_3.SIDE_EFFECT_FLAGS_V4_3),
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
