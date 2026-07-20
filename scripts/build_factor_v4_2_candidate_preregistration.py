#!/usr/bin/env python3
"""Publish one future, owner-private v4.2 candidate preregistration bundle.

The runner is deliberately offline and explicit-input only.  It binds one
strict-Parquet CN ``full_a`` snapshot, one pinned A_quant Git object, the
myQuant alpha158 source, one validated v4 comparison catalog, and the exact
code used to build the bundle.  It has no registry-write, proposal, replay,
transaction, provider, portfolio, broker, order, or trade surface.

``publish`` uses a deterministic cycle directory and the shared Darwin
``renameatx_np(RENAME_EXCL)`` publisher.  ``readback`` reopens a historical
bundle without consulting the current live pointer or protected controls; it
only revalidates the immutable snapshot/PIT/table material recorded inside the
bundle.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import date, datetime
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.factors import governance_candidate_preregistration_bundle_v4_2 as bundle_v4_2  # noqa: E402
from quant_investor.factors import governance_candidate_preregistration_v4_2 as prereg_v4_2  # noqa: E402
from quant_investor.factors import governance_private_bundle_io as private_io  # noqa: E402
from quant_investor.factors import governance_screening_v4 as screening_v4  # noqa: E402
from quant_investor.factors import governance_source_readback_v4_1 as source_readback_v4_1  # noqa: E402


PRODUCTION_PRIVATE_ROOT = Path(
    "/Users/maxwell/mySpace/myQuant/reports/factor_governance/private/"
    "v4_2_candidate_preregistration"
)
MYQUANT_ALPHA158_PATH = PROJECT_ROOT / "quant_investor" / "alpha158.py"
AQUANT_GIT_TREE_PATH = "scripts/run_factor_batch_screen.py"

PROTECTED_CONTROL_PATHS: tuple[tuple[str, Path], ...] = (
    (
        "registry",
        PROJECT_ROOT
        / "quant_investor"
        / "factor_registry"
        / "mined_factors.json",
    ),
    ("latest_pointer", PROJECT_ROOT / "data" / "parquet" / "cn" / "_latest.json"),
    ("catalog", PROJECT_ROOT / "data" / "parquet" / "cn" / "_catalog.json"),
    (
        "fundamental_latest",
        PROJECT_ROOT / "data" / "parquet" / "cn" / "_fundamental_latest.json",
    ),
    (
        "latest_manifest",
        PROJECT_ROOT / "data" / "parquet" / "cn" / "latest_manifest.json",
    ),
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OID_RE = re.compile(r"[0-9a-f]{40}")
_SNAPSHOT_ID_RE = re.compile(r"(\d{8})T(\d{6})Z")
_CODE_EXPECTATION_RE = re.compile(r"([^=]+)=([0-9a-f]{64})")
_FORBIDDEN_ARGUMENT_TOKENS = (
    "--private-root",
    "--cycle-id",
    "--run-id",
    "--registry-write",
    "--proposal",
    "--replay",
    "--transaction",
    "--apply",
    "--provider",
    "--portfolio",
    "--broker",
    "--order",
    "--trade",
)


class FactorV4_2CandidatePreregistrationRunnerError(ValueError):
    """Raised when explicit preregistration publication fails closed."""


@dataclass(frozen=True)
class StableFile:
    """One stable, non-symlink file observation."""

    path: Path
    raw: bytes
    byte_sha256: str
    signature: tuple[int, ...]


@dataclass(frozen=True)
class PublicationInputs:
    """The exact normalized publication inputs and built artifacts."""

    cycle_id: str
    artifacts: dict[str, dict[str, Any]]
    protected_controls: dict[str, StableFile]
    source_binding_semantic_sha256: str
    code_binding_set_semantic_sha256: str


def _error(message: str) -> FactorV4_2CandidatePreregistrationRunnerError:
    return FactorV4_2CandidatePreregistrationRunnerError(message)


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _error(f"{label} must be a lowercase SHA-256")
    return value


def _oid(value: Any, label: str) -> str:
    if type(value) is not str or _OID_RE.fullmatch(value) is None:
        raise _error(f"{label} must be a lowercase Git OID")
    return value


def _absolute(value: Any, label: str) -> Path:
    if type(value) is not str or not value.startswith("/") or "\x00" in value:
        raise _error(f"{label} must be an absolute normalized path")
    path = Path(value)
    if os.path.abspath(value) != value or any(part in {"", ".", ".."} for part in path.parts[1:]):
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
    """Read a current-UID regular file twice through a no-follow descriptor."""

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
            or before.st_size > max_bytes
        ):
            raise _error(f"{label} is not a safe owned regular file: {path}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _signature(opened) != _signature(before):
            raise _error(f"{label} changed while opening: {path}")
        first = b""
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            first = handle.read(max_bytes + 1)
        if len(first) > max_bytes:
            raise _error(f"{label} exceeds the maximum size: {path}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            second = handle.read(max_bytes + 1)
        after = os.fstat(descriptor)
        if first != second or _signature(after) != _signature(opened):
            raise _error(f"{label} changed across stable readback: {path}")
        digest = hashlib.sha256(first).hexdigest()
        if expected is not None and digest != expected:
            raise _error(f"{label} SHA-256 mismatch: {path}")
        return StableFile(
            path=path,
            raw=first,
            byte_sha256=digest,
            signature=_signature(after),
        )
    except FactorV4_2CandidatePreregistrationRunnerError:
        raise
    except OSError as exc:
        raise _error(f"{label} is unavailable: {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise _error(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise _error(f"{label} must be a JSON object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise _error(f"{label} must contain finite JSON values") from exc
    return copy.deepcopy(value)


def _validate_cycle_identity(
    *,
    snapshot_id: Any,
    analysis_start: Any,
    cutoff: Any,
) -> str:
    if type(snapshot_id) is not str:
        raise _error("snapshot_id must be a canonical UTC timestamp")
    match = _SNAPSHOT_ID_RE.fullmatch(snapshot_id)
    if match is None:
        raise _error("snapshot_id must use YYYYMMDDTHHMMSSZ")
    try:
        snapshot_timestamp = datetime.strptime(snapshot_id, "%Y%m%dT%H%M%SZ")
        snapshot_date = snapshot_timestamp.date()
        start_date = date.fromisoformat(str(analysis_start))
        cutoff_date = date.fromisoformat(str(cutoff))
    except ValueError as exc:
        raise _error("analysis_start/cutoff/snapshot_id date is invalid") from exc
    if str(start_date) != analysis_start or str(cutoff_date) != cutoff:
        raise _error("analysis_start and cutoff must use canonical YYYY-MM-DD")
    if start_date > cutoff_date:
        raise _error("analysis_start must not be after cutoff")
    if cutoff_date <= date(2026, 7, 17):
        raise _error("cutoff must be later than 2026-07-17")
    if snapshot_date < cutoff_date:
        raise _error("snapshot_id date must not be before cutoff")
    return f"cn_full_a_v4_2_{cutoff_date:%Y%m%d}_{snapshot_id}"


def _validate_private_root_preflight(root: Path, *, cycle_id: str) -> None:
    """Reject a missing, aliased, non-private, or already-used root."""

    if not root.is_absolute() or os.path.abspath(root) != str(root):
        raise _error("private root must be absolute and normalized")
    if tuple(root.parts[-len(bundle_v4_2.ROOT_SUFFIX_V4_2) :]) != tuple(
        bundle_v4_2.ROOT_SUFFIX_V4_2
    ):
        raise _error("private root must be the exact v4.2 preregistration lane")
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
    destination = root / cycle_id
    try:
        os.lstat(destination)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise _error(f"cannot inspect deterministic destination: {destination}") from exc
    else:
        raise _error(f"deterministic cycle destination already exists: {destination}")


def _parse_code_expectations(values: Sequence[str]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for value in values:
        if type(value) is not str:
            raise _error("expected code binding must be RELATIVE_PATH=SHA256")
        match = _CODE_EXPECTATION_RE.fullmatch(value)
        if match is None:
            raise _error("expected code binding must be RELATIVE_PATH=SHA256")
        relative, digest = match.groups()
        path = Path(relative)
        if (
            path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or relative in rows
        ):
            raise _error("expected code binding paths must be unique normalized relatives")
        rows[relative] = digest
    expected_paths = tuple(bundle_v4_2.CODE_BINDING_PATHS_V4_2)
    if set(rows) != set(expected_paths) or len(rows) != len(expected_paths):
        missing = sorted(set(expected_paths) - set(rows))
        extra = sorted(set(rows) - set(expected_paths))
        raise _error(
            "expected code binding inventory mismatch: "
            f"missing={','.join(missing) or '-'};extra={','.join(extra) or '-'}"
        )
    return {relative: rows[relative] for relative in expected_paths}


def _snapshot_code_bindings(
    *,
    repository_root: Path,
    expectations: Mapping[str, str],
) -> dict[str, StableFile]:
    result: dict[str, StableFile] = {}
    for relative in bundle_v4_2.CODE_BINDING_PATHS_V4_2:
        result[relative] = _stable_file(
            repository_root / relative,
            label=f"code binding {relative}",
            expected_sha256=expectations[relative],
            max_bytes=16 * 1024 * 1024,
        )
    return result


def _snapshot_protected_controls(
    paths: Sequence[tuple[str, Path]],
) -> dict[str, StableFile]:
    names = [name for name, _path in paths]
    if len(names) != 5 or len(set(names)) != 5:
        raise _error("protected control inventory must contain exactly five names")
    return {
        name: _stable_file(path, label=f"protected control {name}")
        for name, path in paths
    }


def _assert_snapshots_unchanged(
    snapshots: Mapping[str, StableFile],
    *,
    label: str,
) -> None:
    for name, expected in snapshots.items():
        current = _stable_file(
            expected.path,
            label=f"{label} {name}",
            expected_sha256=expected.byte_sha256,
            max_bytes=max(len(expected.raw), 1) + 1,
        )
        if current.signature != expected.signature or current.raw != expected.raw:
            raise _error(f"{label} changed before commit: {name}")


def _run_git(repository: Path, arguments: Sequence[str]) -> bytes:
    if not repository.is_absolute() or repository.is_symlink() or not repository.is_dir():
        raise _error("A_quant Git repository must be an absolute real directory")
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
    }
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _error(f"pinned A_quant Git object read failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise _error(f"pinned A_quant Git object read failed: {detail}")
    return completed.stdout


def _verify_pinned_aquant(args: argparse.Namespace) -> dict[str, Any]:
    repository = _absolute(args.aquant_git_repository, "A_quant Git repository")
    commit = _oid(args.aquant_commit, "A_quant commit")
    blob_oid = _oid(args.aquant_blob_oid, "A_quant blob OID")
    raw_sha = _sha256(args.expected_aquant_raw_sha256, "A_quant raw SHA-256")
    if (
        commit != prereg_v4_2.AQUANT_COMMIT
        or blob_oid != prereg_v4_2.AQUANT_BLOB_OID
        or raw_sha != prereg_v4_2.AQUANT_RAW_SHA256
        or args.aquant_mode != prereg_v4_2.AQUANT_MODE
        or args.aquant_source_path != AQUANT_GIT_TREE_PATH
        or f"{repository.name}/{args.aquant_source_path}" != prereg_v4_2.AQUANT_PATH
    ):
        raise _error("explicit A_quant pin differs from the frozen v4.2 source oracle")
    resolved_commit = _run_git(
        repository, ["rev-parse", "--verify", f"{commit}^{{commit}}"]
    ).decode("ascii").strip()
    if resolved_commit != commit:
        raise _error("A_quant commit resolution mismatch")
    tree_row = _run_git(
        repository, ["ls-tree", commit, "--", args.aquant_source_path]
    ).decode("utf-8").strip()
    prefix, separator, tree_path = tree_row.partition("\t")
    parts = prefix.split()
    if (
        not separator
        or tree_path != args.aquant_source_path
        or parts != [args.aquant_mode, "blob", blob_oid]
    ):
        raise _error("A_quant pinned tree entry mismatch")
    raw = _run_git(repository, ["cat-file", "blob", blob_oid])
    if hashlib.sha256(raw).hexdigest() != raw_sha:
        raise _error("A_quant pinned blob SHA-256 mismatch")
    return prereg_v4_2.build_aquant_receipt_v4_2()


def _bind_strict_source(args: argparse.Namespace) -> tuple[Any, bytes, bytes]:
    bound = source_readback_v4_1.bind_explicit_cutoff_inputs_v4_1(
        latest_pointer_path=args.latest_pointer_path,
        expected_latest_pointer_sha256=args.expected_latest_pointer_sha256,
        snapshot_manifest_path=args.snapshot_manifest_path,
        expected_snapshot_manifest_sha256=args.expected_snapshot_manifest_sha256,
        components_path=args.components_path,
        expected_components_sha256=args.expected_components_sha256,
        expected_full_a_semantic_sha256=args.expected_full_a_semantic_sha256,
        pit_generation_manifest_path=args.pit_generation_manifest_path,
        expected_pit_generation_manifest_sha256=(
            args.expected_pit_generation_manifest_sha256
        ),
        pit_membership_path=args.pit_membership_path,
        expected_pit_membership_sha256=args.expected_pit_membership_sha256,
        table_root=args.table_root,
        snapshot_id=args.snapshot_id,
        analysis_start=args.analysis_start,
        cutoff_date=args.cutoff,
        expected_full_a_count=args.expected_full_a_count,
        expected_serving_inventory_count=args.expected_serving_inventory_count,
    )
    expected_inventory = _sha256(
        args.expected_table_inventory_sha256,
        "expected table inventory SHA-256",
    )
    if bound.binding["table"]["inventory_sha256"] != expected_inventory:
        raise _error("strict table inventory SHA-256 mismatch")
    pointer = _stable_file(
        _absolute(args.latest_pointer_path, "latest pointer"),
        label="latest pointer",
        expected_sha256=args.expected_latest_pointer_sha256,
    )
    components = _stable_file(
        _absolute(args.components_path, "components"),
        label="components",
        expected_sha256=args.expected_components_sha256,
    )
    return bound, pointer.raw, components.raw


def _comparison_catalog(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    ontology_file = _stable_file(
        _absolute(args.comparison_ontology_path, "comparison ontology"),
        label="comparison ontology",
        expected_sha256=args.expected_comparison_ontology_sha256,
        max_bytes=64 * 1024 * 1024,
    )
    catalog_file = _stable_file(
        _absolute(args.comparison_catalog_path, "comparison catalog"),
        label="comparison catalog",
        expected_sha256=args.expected_comparison_catalog_sha256,
        max_bytes=128 * 1024 * 1024,
    )
    ontology = screening_v4.validate_primitive_ontology_v4(
        _strict_json_object(ontology_file.raw, "comparison ontology")
    )
    catalog = screening_v4.validate_candidate_catalog_v4(
        _strict_json_object(catalog_file.raw, "comparison catalog"),
        ontology=ontology,
    )
    receipt = prereg_v4_2.build_comparison_catalog_receipt_v4_2(
        catalog_id=args.comparison_catalog_id,
        catalog_byte_sha256=catalog_file.byte_sha256,
        catalog_semantic_sha256=catalog["semantic_sha256"],
        primitive_count=len(ontology["primitives"]),
        definition_identity_inventory=[
            {
                "name": row["name"],
                "definition_identity_sha256": row["definition_sha256"],
            }
            for row in catalog["candidates"]
        ],
    )
    return catalog, receipt


def _collect_publication_inputs(
    args: argparse.Namespace,
    *,
    repository_root: Path,
    protected_paths: Sequence[tuple[str, Path]],
) -> PublicationInputs:
    cycle_id = _validate_cycle_identity(
        snapshot_id=args.snapshot_id,
        analysis_start=args.analysis_start,
        cutoff=args.cutoff,
    )
    expected_code = _parse_code_expectations(args.expected_code_sha256)
    _snapshot_code_bindings(
        repository_root=repository_root,
        expectations=expected_code,
    )
    controls = _snapshot_protected_controls(protected_paths)

    bound, pointer_raw, components_raw = _bind_strict_source(args)
    strict_source = bundle_v4_2.build_strict_full_a_source_binding_v4_2(
        bound_inputs=bound,
        latest_pointer_raw=pointer_raw,
        components_raw=components_raw,
    )

    aquant_receipt = _verify_pinned_aquant(args)
    alpha158 = _stable_file(
        _absolute(args.myquant_alpha158_path, "myQuant alpha158 source"),
        label="myQuant alpha158 source",
        expected_sha256=args.expected_myquant_alpha158_sha256,
        max_bytes=16 * 1024 * 1024,
    )
    if (
        alpha158.path != MYQUANT_ALPHA158_PATH
        or alpha158.byte_sha256 != prereg_v4_2.MYQUANT_FULL_SHA256
    ):
        raise _error("explicit myQuant alpha158 source differs from the frozen oracle")
    myquant_receipt = prereg_v4_2.build_myquant_receipt_v4_2()
    operator_semantics = prereg_v4_2.build_operator_semantics_v4_2()
    _catalog, comparison_receipt = _comparison_catalog(args)
    selection = prereg_v4_2.build_selection_spec_v4_2(
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_receipt,
    )
    code_binding_set = bundle_v4_2.build_code_binding_set_v4_2(
        repository_root=repository_root
    )
    observed_code = {
        row["relative_path"]: row["byte_sha256"]
        for row in code_binding_set["ordered_bindings"]
    }
    if observed_code != expected_code:
        raise _error("built code binding set differs from explicit expected hashes")
    bundle_v4_2.revalidate_code_binding_set_v4_2(
        repository_root=repository_root,
        value=code_binding_set,
    )
    artifacts = bundle_v4_2.build_candidate_preregistration_bundle_artifacts_v4_2(
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_receipt,
        candidate_selection_spec=selection,
        strict_full_a_source_binding=strict_source,
        code_binding_set=code_binding_set,
    )
    normalized = bundle_v4_2.validate_candidate_preregistration_bundle_inputs_v4_2(
        artifacts
    )
    return PublicationInputs(
        cycle_id=cycle_id,
        artifacts={name: dict(value) for name, value in normalized.items()},
        protected_controls=controls,
        source_binding_semantic_sha256=strict_source["artifact_semantic_sha256"],
        code_binding_set_semantic_sha256=code_binding_set[
            "artifact_semantic_sha256"
        ],
    )


def _postcommit_control_diagnostics(
    entry: Mapping[str, StableFile],
) -> dict[str, Any]:
    """Return diagnostics only; never raise after a successful commit."""

    rows: list[dict[str, Any]] = []
    try:
        for name, expected in entry.items():
            try:
                current = _stable_file(
                    expected.path,
                    label=f"postcommit protected control {name}",
                )
            except Exception as exc:  # diagnostic-only after commit
                rows.append(
                    {
                        "name": name,
                        "before_sha256": expected.byte_sha256,
                        "after_sha256": None,
                        "unchanged": False,
                        "diagnostic": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "name": name,
                    "before_sha256": expected.byte_sha256,
                    "after_sha256": current.byte_sha256,
                    "unchanged": (
                        current.byte_sha256 == expected.byte_sha256
                        and current.signature == expected.signature
                    ),
                    "diagnostic": None,
                }
            )
    except Exception as exc:  # pragma: no cover - final defensive boundary
        return {"status": "DIAGNOSTIC_UNAVAILABLE", "rows": [], "detail": str(exc)}
    return {
        "status": (
            "UNCHANGED"
            if all(row["unchanged"] is True for row in rows)
            else "DRIFT_DIAGNOSTIC_ONLY"
        ),
        "rows": rows,
    }


def _postcommit_immutable_diagnostics(
    strict_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Best-effort external reopen after commit; this can never reject commit."""

    try:
        detail = bundle_v4_2.revalidate_recorded_immutable_source_v4_2(
            strict_source
        )
    except Exception as exc:
        return {
            "status": "DRIFT_DIAGNOSTIC_ONLY",
            "accepted": False,
            "detail": str(exc),
        }
    return {
        "status": "VERIFIED_DIAGNOSTIC_ONLY",
        "accepted": detail.get("accepted") is True,
        "detail": detail,
    }


def _validated_artifact_descriptors(
    value: Any,
    *,
    bundle_path: Path,
) -> dict[str, dict[str, Any]]:
    """Validate the exact mapping shape returned by shared private I/O."""

    if not isinstance(value, Mapping):
        raise _error("artifact_descriptors must be a filename-keyed mapping")
    expected_names = (
        *bundle_v4_2.INPUT_FILENAMES_V4_2,
        bundle_v4_2.READBACK_REPORT_FILENAME_V4_2,
    )
    if set(value) != set(expected_names) or any(type(name) is not str for name in value):
        raise _error("artifact_descriptors filename inventory mismatch")
    normalized: dict[str, dict[str, Any]] = {}
    fields = {
        "absolute_path",
        "byte_sha256",
        "size_bytes",
        "mode",
        "uid",
        "nlink",
    }
    for name in expected_names:
        item = value[name]
        if not isinstance(item, Mapping) or set(item) != fields:
            raise _error(f"artifact descriptor fields mismatch: {name}")
        absolute_path = _absolute(
            item["absolute_path"], f"artifact descriptor path {name}"
        )
        if absolute_path != bundle_path / name:
            raise _error(f"artifact descriptor path mismatch: {name}")
        byte_digest = _sha256(
            item["byte_sha256"], f"artifact descriptor byte SHA {name}"
        )
        if type(item["size_bytes"]) is not int or item["size_bytes"] <= 0:
            raise _error(f"artifact descriptor size mismatch: {name}")
        if (
            item["mode"] != 0o600
            or item["uid"] != os.getuid()
            or item["nlink"] != 1
        ):
            raise _error(f"artifact descriptor private-file contract failed: {name}")
        normalized[name] = {
            "absolute_path": str(absolute_path),
            "byte_sha256": byte_digest,
            "size_bytes": item["size_bytes"],
            "mode": 0o600,
            "uid": os.getuid(),
            "nlink": 1,
        }
    return normalized


def run_publish(
    args: argparse.Namespace,
    *,
    private_root: Path = PRODUCTION_PRIVATE_ROOT,
    repository_root: Path = PROJECT_ROOT,
    protected_paths: Sequence[tuple[str, Path]] = PROTECTED_CONTROL_PATHS,
    exclusive_rename_probe: Callable[[], None] | None = None,
    _test_race_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Preflight, publish once, independently reopen, then diagnose controls."""

    cycle_id = _validate_cycle_identity(
        snapshot_id=args.snapshot_id,
        analysis_start=args.analysis_start,
        cutoff=args.cutoff,
    )
    probe = exclusive_rename_probe or private_io._require_exclusive_rename_support
    probe()
    _validate_private_root_preflight(private_root, cycle_id=cycle_id)
    entry = _collect_publication_inputs(
        args,
        repository_root=repository_root,
        protected_paths=protected_paths,
    )
    if entry.cycle_id != cycle_id:
        raise _error("derived cycle identity changed across preflight")

    def revalidate_inputs() -> None:
        locked = _collect_publication_inputs(
            args,
            repository_root=repository_root,
            protected_paths=protected_paths,
        )
        if (
            locked.cycle_id != entry.cycle_id
            or locked.artifacts != entry.artifacts
            or locked.source_binding_semantic_sha256
            != entry.source_binding_semantic_sha256
            or locked.code_binding_set_semantic_sha256
            != entry.code_binding_set_semantic_sha256
        ):
            raise _error("publication inputs changed before commit")
        _assert_snapshots_unchanged(
            entry.protected_controls,
            label="protected control",
        )

    published = bundle_v4_2.publish_candidate_preregistration_bundle_v4_2(
        private_root=private_root,
        artifacts=entry.artifacts,
        revalidate_inputs=revalidate_inputs,
        _test_race_hook=_test_race_hook,
    )
    independent = bundle_v4_2.readback_candidate_preregistration_bundle_files_v4_2(
        published["bundle_path"]
    )
    if independent.get("accepted") is not True:
        raise _error("independent canonical bundle reopen was not accepted")
    report_filename = bundle_v4_2.READBACK_REPORT_FILENAME_V4_2
    descriptors = _validated_artifact_descriptors(
        independent["artifact_descriptors"],
        bundle_path=_absolute(independent["bundle_path"], "published bundle path"),
    )
    report = independent["readback_report"]
    diagnostics = _postcommit_control_diagnostics(entry.protected_controls)
    immutable_diagnostics = _postcommit_immutable_diagnostics(
        independent["artifacts"][
            bundle_v4_2.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2
        ]
    )
    return {
        "accepted": True,
        "mode": "publish",
        "protocol_version": "v4",
        "evidence_contract_version": "v4.2",
        "cycle_id": cycle_id,
        "bundle_path": independent["bundle_path"],
        "readback_report_path": descriptors[report_filename]["absolute_path"],
        "readback_report_byte_sha256": descriptors[report_filename]["byte_sha256"],
        "readback_report_semantic_sha256": report["artifact_semantic_sha256"],
        "source_binding_semantic_sha256": entry.source_binding_semantic_sha256,
        "code_binding_set_semantic_sha256": entry.code_binding_set_semantic_sha256,
        "publisher_return_accepted": published.get("accepted") is True,
        "independent_reopen_accepted": True,
        "exact_once_scope": "deterministic_cycle_directory_RENAME_EXCL_only",
        "external_maintenance_serialization_claimed": False,
        "protected_controls": diagnostics,
        "immutable_source_postcommit": immutable_diagnostics,
        "authority": copy.deepcopy(prereg_v4_2.AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(prereg_v4_2.SIDE_EFFECT_FLAGS),
    }


def run_readback(args: argparse.Namespace) -> dict[str, Any]:
    """Reopen one historical bundle without consulting current mutable state."""

    expected_byte = _sha256(
        args.expected_readback_report_byte_sha256,
        "expected readback report byte SHA-256",
    )
    expected_semantic = _sha256(
        args.expected_readback_report_semantic_sha256,
        "expected readback report semantic SHA-256",
    )
    bundle_path = _absolute(args.bundle_path, "bundle path")
    result = bundle_v4_2.readback_candidate_preregistration_bundle_files_v4_2(
        bundle_path
    )
    report_filename = bundle_v4_2.READBACK_REPORT_FILENAME_V4_2
    descriptors = _validated_artifact_descriptors(
        result["artifact_descriptors"],
        bundle_path=bundle_path,
    )
    report = result["readback_report"]
    if descriptors[report_filename]["byte_sha256"] != expected_byte:
        raise _error("historical readback report byte SHA-256 mismatch")
    if report["artifact_semantic_sha256"] != expected_semantic:
        raise _error("historical readback report semantic SHA-256 mismatch")
    strict_source = result["artifacts"][
        bundle_v4_2.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2
    ]
    immutable = bundle_v4_2.revalidate_recorded_immutable_source_v4_2(
        strict_source
    )
    if (
        immutable.get("accepted") is not True
        or immutable.get("current_pointer_read") is not False
        or immutable.get("current_components_read") is not False
        or immutable.get("serving_tree_read") is not False
    ):
        raise _error("historical immutable reopen scope was not exact")
    return {
        "accepted": True,
        "mode": "readback",
        "protocol_version": "v4",
        "evidence_contract_version": "v4.2",
        "bundle_path": str(bundle_path),
        "readback_report_byte_sha256": expected_byte,
        "readback_report_semantic_sha256": expected_semantic,
        "immutable_reopen": immutable,
        "current_latest_pointer_read": False,
        "current_components_read": False,
        "current_protected_controls_read": False,
        "authority": copy.deepcopy(prereg_v4_2.AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(prereg_v4_2.SIDE_EFFECT_FLAGS),
    }


def _add_publish_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--latest-pointer-path", required=True)
    parser.add_argument("--expected-latest-pointer-sha256", required=True)
    parser.add_argument("--snapshot-manifest-path", required=True)
    parser.add_argument("--expected-snapshot-manifest-sha256", required=True)
    parser.add_argument("--components-path", required=True)
    parser.add_argument("--expected-components-sha256", required=True)
    parser.add_argument("--expected-full-a-semantic-sha256", required=True)
    parser.add_argument("--pit-generation-manifest-path", required=True)
    parser.add_argument("--expected-pit-generation-manifest-sha256", required=True)
    parser.add_argument("--pit-membership-path", required=True)
    parser.add_argument("--expected-pit-membership-sha256", required=True)
    parser.add_argument("--table-root", required=True)
    parser.add_argument("--expected-table-inventory-sha256", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--analysis-start", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--expected-full-a-count", type=int, required=True)
    parser.add_argument("--expected-serving-inventory-count", type=int, required=True)
    parser.add_argument("--aquant-git-repository", required=True)
    parser.add_argument("--aquant-commit", required=True)
    parser.add_argument("--aquant-source-path", required=True)
    parser.add_argument("--aquant-blob-oid", required=True)
    parser.add_argument("--expected-aquant-raw-sha256", required=True)
    parser.add_argument("--aquant-mode", required=True)
    parser.add_argument("--myquant-alpha158-path", required=True)
    parser.add_argument("--expected-myquant-alpha158-sha256", required=True)
    parser.add_argument("--comparison-catalog-path", required=True)
    parser.add_argument("--expected-comparison-catalog-sha256", required=True)
    parser.add_argument("--comparison-ontology-path", required=True)
    parser.add_argument("--expected-comparison-ontology-sha256", required=True)
    parser.add_argument("--comparison-catalog-id", required=True)
    parser.add_argument(
        "--expected-code-sha256",
        action="append",
        default=[],
        metavar="RELATIVE_PATH=SHA256",
        help="repeat for the exact fixed v4.2 code-binding tuple",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish", help="publish one future exact cycle")
    _add_publish_arguments(publish)
    readback = commands.add_parser(
        "readback", help="historically reopen one immutable private bundle"
    )
    readback.add_argument("--bundle-path", required=True)
    readback.add_argument("--expected-readback-report-byte-sha256", required=True)
    readback.add_argument(
        "--expected-readback-report-semantic-sha256", required=True
    )
    help_text = parser.format_help() + publish.format_help() + readback.format_help()
    if any(token in help_text for token in _FORBIDDEN_ARGUMENT_TOKENS):
        raise _error("forbidden mutation/execution argument leaked into CLI surface")
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
            "side_effects": copy.deepcopy(prereg_v4_2.SIDE_EFFECT_FLAGS),
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
