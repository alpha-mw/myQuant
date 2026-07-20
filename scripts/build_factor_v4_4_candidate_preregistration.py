#!/usr/bin/env python3
"""Publish one future, owner-private v4.4 five-candidate preregistration.

The command is offline and explicit-input only.  It reconstructs the sealed
four-candidate v4.2 evidence graph with the unchanged public v4.2 builders,
embeds the exact accepted v4.3 prior-diagnostic bytes, and creates an
independent v4.4 ``PRECOMMITTED -> DISCOVERY`` cycle.  Historical diagnostic
statistics are provenance only and confer no health, admission, activation,
registry, proposal, execution, or new-risk authority.

Publication is exact-once under the shared owner-private bundle lock and uses
Darwin ``renameatx_np(RENAME_EXCL)``.  A historical readback never consults
the current pointer, current protected controls, or mutable diagnostic
sources; it may re-open only the immutable strict source recorded in the
embedded v4.2 graph.
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

from quant_investor.factors import (  # noqa: E402
    governance_candidate_preregistration_bundle_v4_2 as bundle_v4_2,
)
from quant_investor.factors import (  # noqa: E402
    governance_candidate_preregistration_bundle_v4_4 as bundle_v4_4,
)
from quant_investor.factors import (  # noqa: E402
    governance_candidate_preregistration_v4_2 as prereg_v4_2,
)
from quant_investor.factors import (  # noqa: E402
    governance_candidate_preregistration_v4_4 as prereg_v4_4,
)
from quant_investor.factors import (  # noqa: E402
    governance_prior_diagnostic_nomination_bundle_v4_3 as diagnostic_bundle_v4_3,
)
from quant_investor.factors import (  # noqa: E402
    governance_prior_diagnostic_nomination_v4_3 as diagnostic_v4_3,
)
from quant_investor.factors import governance_private_bundle_io as private_io  # noqa: E402
from quant_investor.factors import governance_screening_v4 as screening_v4  # noqa: E402
from quant_investor.factors import (  # noqa: E402
    governance_source_readback_v4_1 as source_readback_v4_1,
)


PRODUCTION_PRIVATE_ROOT = Path(
    "/Users/maxwell/mySpace/myQuant/reports/factor_governance/private/"
    "v4_4_candidate_preregistration"
)
FIXED_DIAGNOSTIC_BUNDLE_PATH = Path(
    "/Users/maxwell/mySpace/myQuant/reports/factor_governance/private/"
    "v4_3_prior_diagnostic_nomination/"
    "cn_full_a_v4_3_prior_nomination_20260717_20260717T172132Z"
)
MYQUANT_ALPHA158_PATH = PROJECT_ROOT / "quant_investor" / "alpha158.py"
AQUANT_GIT_TREE_PATH = "scripts/run_factor_batch_screen.py"

PROTECTED_CONTROL_PATHS: tuple[tuple[str, Path], ...] = (
    (
        "registry",
        PROJECT_ROOT / "quant_investor" / "factor_registry" / "mined_factors.json",
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
_SNAPSHOT_ID_RE = re.compile(r"\d{8}T\d{6}Z")
_NAMED_SHA_RE = re.compile(r"([^=]+)=([0-9a-f]{64})")
_FORBIDDEN_ARGUMENT_TOKENS = (
    "--private-root",
    "--cycle-id",
    "--run-id",
    "--candidate",
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


class FactorV4_4CandidatePreregistrationRunnerError(ValueError):
    """Raised when the v4.4 preregistration runner fails closed."""


@dataclass(frozen=True)
class StableFile:
    """One stable, owner-controlled, non-symlink regular-file observation."""

    path: Path
    raw: bytes
    byte_sha256: str
    size_bytes: int
    signature: tuple[int, ...]


@dataclass(frozen=True)
class PublicationInputs:
    """The complete twice-collected byte graph supplied to the publisher."""

    cycle_id: str
    artifacts: dict[str, dict[str, Any]]
    raw_input_bindings: tuple[dict[str, Any], ...]
    collected_raw_bytes: dict[str, bytes]
    protected_controls: dict[str, StableFile]
    code_bindings: dict[str, StableFile]
    source_binding_semantic_sha256: str
    code_binding_set_semantic_sha256: str
    diagnostic_current_mutable_sources_read: bool


def _error(message: str) -> FactorV4_4CandidatePreregistrationRunnerError:
    return FactorV4_4CandidatePreregistrationRunnerError(message)


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
        int(metadata.st_gid),
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
    """Read one current-owner, single-link regular file twice with no-follow."""

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
            raise _error(f"{label} is not a safe owned regular file: {path}")
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
            signature=_signature(after),
        )
    except FactorV4_4CandidatePreregistrationRunnerError:
        raise
    except OSError as exc:
        raise _error(f"{label} is unavailable: {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=pairs,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise _error(f"{label} is not strict finite UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise _error(f"{label} must be a JSON object")
    return copy.deepcopy(value)


def _validate_cycle_identity(
    *, snapshot_id: Any, analysis_start: Any, cutoff: Any
) -> str:
    """Reject stale identities before a platform probe, root read, or collection."""

    if type(analysis_start) is not str or type(cutoff) is not str:
        raise _error("analysis_start/cutoff must use canonical YYYY-MM-DD")
    try:
        start = date.fromisoformat(analysis_start)
        end = date.fromisoformat(cutoff)
    except ValueError as exc:
        raise _error("analysis_start/cutoff date is invalid") from exc
    if start.isoformat() != analysis_start or end.isoformat() != cutoff:
        raise _error("analysis_start/cutoff must use canonical YYYY-MM-DD")
    if start > end:
        raise _error("analysis_start must not be after cutoff")
    if end <= date(2026, 7, 19):
        raise _error("v4.4 cutoff must be strictly later than 2026-07-19")
    if type(snapshot_id) is not str or _SNAPSHOT_ID_RE.fullmatch(snapshot_id) is None:
        raise _error("snapshot_id must use canonical YYYYMMDDTHHMMSSZ")
    try:
        snapshot = datetime.strptime(snapshot_id, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise _error("snapshot_id must be a real UTC timestamp") from exc
    if snapshot.date() != end:
        raise _error("snapshot_id calendar date must exactly equal v4.4 cutoff")
    try:
        return prereg_v4_4.deterministic_cycle_id_v4_4(
            cutoff=cutoff, snapshot_id=snapshot_id
        )
    except Exception as exc:
        raise _error(f"v4.4 cycle identity is invalid: {exc}") from exc


def _validate_publication_at(value: Any, *, cutoff: str) -> str:
    if type(value) is not str:
        raise _error("publication_at must be an offset-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise _error("publication_at must be an offset-aware ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _error("publication_at must be offset-aware")
    if parsed.isoformat() != value:
        raise _error("publication_at must use canonical ISO format")
    if parsed.date() < date.fromisoformat(cutoff):
        raise _error("publication_at must not precede cutoff")
    return value


def _validate_private_root_preflight(root: Path, *, cycle_id: str) -> None:
    if not root.is_absolute() or os.path.abspath(root) != str(root):
        raise _error("private root must be absolute and normalized")
    suffix = tuple(bundle_v4_4.ROOT_SUFFIX_V4_4)
    if tuple(root.parts[-len(suffix) :]) != suffix:
        raise _error("private root must be the exact v4.4 preregistration lane")
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
        return
    except OSError as exc:
        raise _error(f"cannot inspect deterministic destination: {destination}") from exc
    raise _error(f"deterministic cycle destination already exists: {destination}")


def _parse_named_hashes(
    values: Sequence[str], *, expected_names: Sequence[str], label: str
) -> dict[str, str]:
    rows: dict[str, str] = {}
    for value in values:
        if type(value) is not str:
            raise _error(f"{label} must use NAME=SHA256")
        match = _NAMED_SHA_RE.fullmatch(value)
        if match is None:
            raise _error(f"{label} must use NAME=SHA256")
        name, digest = match.groups()
        path = Path(name)
        if (
            path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or name in rows
        ):
            raise _error(f"{label} names must be unique normalized relatives")
        rows[name] = digest
    if set(rows) != set(expected_names) or len(rows) != len(expected_names):
        missing = sorted(set(expected_names) - set(rows))
        extra = sorted(set(rows) - set(expected_names))
        raise _error(
            f"{label} inventory mismatch: "
            f"missing={','.join(missing) or '-'};extra={','.join(extra) or '-'}"
        )
    return {name: rows[name] for name in expected_names}


def _snapshot_code_bindings(
    *, repository_root: Path, expectations: Mapping[str, str]
) -> dict[str, StableFile]:
    return {
        relative: _stable_file(
            repository_root / relative,
            label=f"code binding {relative}",
            expected_sha256=expectations[relative],
            max_bytes=16 * 1024 * 1024,
        )
        for relative in prereg_v4_4.CODE_BINDING_PATHS_V4_4
    }


def _snapshot_protected_controls(
    *,
    paths: Sequence[tuple[str, Path]],
    expectations: Mapping[str, str],
) -> dict[str, StableFile]:
    names = tuple(name for name, _path in paths)
    expected_names = tuple(name for name, _path in PROTECTED_CONTROL_PATHS)
    if names != expected_names or len(set(names)) != 5:
        raise _error("protected control inventory/order must be the exact five")
    return {
        name: _stable_file(
            path,
            label=f"protected control {name}",
            expected_sha256=expectations[name],
            max_bytes=64 * 1024 * 1024,
        )
        for name, path in paths
    }


def _assert_snapshots_unchanged(
    snapshots: Mapping[str, StableFile], *, label: str
) -> None:
    for name, expected in snapshots.items():
        current = _stable_file(
            expected.path,
            label=f"{label} {name}",
            expected_sha256=expected.byte_sha256,
            max_bytes=max(expected.size_bytes, 1) + 1,
        )
        if current.signature != expected.signature or current.raw != expected.raw:
            raise _error(f"{label} changed before commit: {name}")


def _run_git(repository: Path, arguments: Sequence[str]) -> bytes:
    """Read a pinned Git object with every inherited ``GIT_*`` key removed."""

    if not repository.is_absolute() or repository.is_symlink() or not repository.is_dir():
        raise _error("A_quant Git repository must be an absolute real directory")
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
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
        raise _error("explicit A_quant pin differs from the frozen v4.2 oracle")
    resolved = _run_git(
        repository, ["rev-parse", "--verify", f"{commit}^{{commit}}"]
    ).decode("ascii").strip()
    if resolved != commit:
        raise _error("A_quant commit resolution mismatch")
    tree_row = _run_git(
        repository, ["ls-tree", commit, "--", args.aquant_source_path]
    ).decode("utf-8").strip()
    prefix, separator, tree_path = tree_row.partition("\t")
    if (
        not separator
        or tree_path != args.aquant_source_path
        or prefix.split() != [args.aquant_mode, "blob", blob_oid]
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


def _collect_v4_2_predecessor_graph(
    args: argparse.Namespace,
    *,
    repository_root: Path,
    expected_code: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    """Rebuild the exact v4.2 graph without invoking its unsafe old runner."""

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
    expected_v4_2 = {
        relative: expected_code[relative]
        for relative in bundle_v4_2.CODE_BINDING_PATHS_V4_2
    }
    if observed_code != expected_v4_2:
        raise _error("v4.2 code binding set differs from explicit v4.4 expectations")
    bundle_v4_2.revalidate_code_binding_set_v4_2(
        repository_root=repository_root,
        value=code_binding_set,
    )
    unprefixed = bundle_v4_2.build_candidate_preregistration_bundle_artifacts_v4_2(
        aquant_receipt=aquant_receipt,
        myquant_receipt=myquant_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_receipt,
        candidate_selection_spec=selection,
        strict_full_a_source_binding=strict_source,
        code_binding_set=code_binding_set,
    )
    normalized = bundle_v4_2.validate_candidate_preregistration_bundle_inputs_v4_2(
        unprefixed
    )
    prefixed = {
        prereg_v4_4.V4_2_PREDECESSOR_PREFIX + filename: copy.deepcopy(
            normalized[filename]
        )
        for filename in bundle_v4_2.INPUT_FILENAMES_V4_2
    }
    raw = {
        prereg_v4_4.V4_2_PREDECESSOR_PREFIX + filename: (
            prereg_v4_2.canonical_file_bytes_v4_2(normalized[filename])
        )
        for filename in bundle_v4_2.INPUT_FILENAMES_V4_2
    }
    if tuple(prefixed) != tuple(prereg_v4_4.V4_2_PREDECESSOR_FILENAMES):
        raise _error("embedded v4.2 predecessor filename order drifted")
    prereg_v4_4.validate_v4_2_predecessor_graph_v4_4(prefixed)
    return prefixed, raw


def _collect_prior_diagnostic_graph(
    *, bundle_path: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    """Read only the exact accepted historical diagnostic files and bytes."""

    if bundle_path != FIXED_DIAGNOSTIC_BUNDLE_PATH:
        # This alternate is reachable only through the Python test-injection
        # surface.  Shared I/O still enforces the fixed root suffix and graph.
        if tuple(bundle_path.parent.parts[-len(diagnostic_bundle_v4_3.ROOT_SUFFIX_V4_3) :]) != tuple(
            diagnostic_bundle_v4_3.ROOT_SUFFIX_V4_3
        ):
            raise _error("diagnostic test path must preserve the fixed root suffix")
    result = diagnostic_bundle_v4_3.readback_prior_diagnostic_nomination_bundle_v4_3(
        bundle_path
    )
    if result.get("accepted") is not True:
        raise _error("historical diagnostic bundle readback was not accepted")
    artifacts = result.get("artifacts")
    descriptors = result.get("artifact_descriptors")
    if not isinstance(artifacts, Mapping) or not isinstance(descriptors, Mapping):
        raise _error("historical diagnostic readback shape mismatch")
    expected_names = tuple(prereg_v4_4.PRIOR_DIAGNOSTIC_FILENAMES)
    if tuple(artifacts) != expected_names or set(descriptors) != set(expected_names):
        raise _error("historical diagnostic inventory mismatch")
    expected_rows = {
        row["filename"]: row
        for row in prereg_v4_4.EXPECTED_PRIOR_DIAGNOSTIC_BINDINGS
    }
    normalized: dict[str, dict[str, Any]] = {}
    raw: dict[str, bytes] = {}
    for filename in expected_names:
        expected = expected_rows[filename]
        descriptor = descriptors.get(filename)
        if not isinstance(descriptor, Mapping):
            raise _error(f"historical diagnostic descriptor missing: {filename}")
        path = _absolute(
            descriptor.get("absolute_path"),
            f"historical diagnostic descriptor {filename}",
        )
        if path != bundle_path / filename:
            raise _error(f"historical diagnostic descriptor path mismatch: {filename}")
        observed = _stable_file(
            path,
            label=f"historical diagnostic {filename}",
            expected_sha256=expected["byte_sha256"],
            max_bytes=max(int(expected["size_bytes"]), 1),
        )
        artifact = copy.deepcopy(dict(artifacts[filename]))
        if (
            observed.size_bytes != expected["size_bytes"]
            or artifact.get("artifact_semantic_sha256")
            != expected["semantic_sha256"]
            or diagnostic_v4_3.canonical_file_bytes_v4_3(artifact) != observed.raw
        ):
            raise _error(f"historical diagnostic raw/semantic binding mismatch: {filename}")
        normalized[filename] = artifact
        raw[filename] = observed.raw
    if normalized[expected_names[1]].get("run_id") != diagnostic_v4_3.RUN_ID:
        raise _error("historical diagnostic run_id mismatch")
    prereg_v4_4.validate_prior_diagnostic_graph_v4_4(normalized)
    return normalized, raw


def _build_code_binding_set(
    snapshots: Mapping[str, StableFile],
) -> dict[str, Any]:
    rows = [
        {
            "order": index,
            "relative_path": relative,
            "byte_sha256": snapshots[relative].byte_sha256,
            "size_bytes": snapshots[relative].size_bytes,
        }
        for index, relative in enumerate(
            prereg_v4_4.CODE_BINDING_PATHS_V4_4, start=1
        )
    ]
    return prereg_v4_4.build_code_binding_set_v4_4(ordered_bindings=rows)


def _raw_input_bindings(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    collected_raw_bytes: Mapping[str, bytes],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for filename in bundle_v4_4.INPUT_FILENAMES_V4_4:
        raw = collected_raw_bytes.get(filename)
        if raw is None:
            raw = prereg_v4_4.canonical_file_bytes_v4_4(artifacts[filename])
        rows.append(
            {
                "filename": filename,
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    return tuple(rows)


def _collect_publication_inputs(
    args: argparse.Namespace,
    *,
    repository_root: Path,
    protected_paths: Sequence[tuple[str, Path]],
    diagnostic_bundle_path: Path,
) -> PublicationInputs:
    cycle_id = _validate_cycle_identity(
        snapshot_id=args.snapshot_id,
        analysis_start=args.analysis_start,
        cutoff=args.cutoff,
    )
    publication_at = _validate_publication_at(args.publication_at, cutoff=args.cutoff)
    expected_code = _parse_named_hashes(
        args.expected_code_sha256,
        expected_names=prereg_v4_4.CODE_BINDING_PATHS_V4_4,
        label="expected code binding",
    )
    expected_controls = _parse_named_hashes(
        args.expected_protected_control_sha256,
        expected_names=tuple(name for name, _path in PROTECTED_CONTROL_PATHS),
        label="expected protected control",
    )
    code_snapshots = _snapshot_code_bindings(
        repository_root=repository_root,
        expectations=expected_code,
    )
    controls = _snapshot_protected_controls(
        paths=protected_paths,
        expectations=expected_controls,
    )
    predecessor, predecessor_raw = _collect_v4_2_predecessor_graph(
        args,
        repository_root=repository_root,
        expected_code=expected_code,
    )
    diagnostic, diagnostic_raw = _collect_prior_diagnostic_graph(
        bundle_path=diagnostic_bundle_path
    )
    code_binding_set = _build_code_binding_set(code_snapshots)
    collected_raw = {**predecessor_raw, **diagnostic_raw}
    artifacts = bundle_v4_4.build_candidate_preregistration_bundle_artifacts_v4_4(
        v4_2_predecessor_artifacts=predecessor,
        prior_diagnostic_artifacts=diagnostic,
        code_binding_set=code_binding_set,
        publication_at=publication_at,
        collected_raw_bytes=collected_raw,
    )
    raw_bindings = _raw_input_bindings(
        artifacts=artifacts,
        collected_raw_bytes=collected_raw,
    )
    normalized = bundle_v4_4.validate_candidate_preregistration_bundle_inputs_v4_4(
        artifacts,
        raw_input_bindings=raw_bindings,
        collected_raw_bytes=collected_raw,
    )
    if normalized[bundle_v4_4.CYCLE_ROOT_FILENAME_V4_4]["cycle_id"] != cycle_id:
        raise _error("built v4.4 cycle identity differs from explicit identity")
    strict_name = (
        prereg_v4_4.V4_2_PREDECESSOR_PREFIX
        + bundle_v4_2.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2
    )
    return PublicationInputs(
        cycle_id=cycle_id,
        artifacts={name: copy.deepcopy(dict(value)) for name, value in normalized.items()},
        raw_input_bindings=tuple(copy.deepcopy(row) for row in raw_bindings),
        collected_raw_bytes=dict(collected_raw),
        protected_controls=controls,
        code_bindings=code_snapshots,
        source_binding_semantic_sha256=normalized[strict_name][
            "artifact_semantic_sha256"
        ],
        code_binding_set_semantic_sha256=normalized[
            bundle_v4_4.CODE_BINDING_SET_FILENAME_V4_4
        ]["artifact_semantic_sha256"],
        diagnostic_current_mutable_sources_read=False,
    )


def _validated_artifact_descriptors(
    value: Any, *, bundle_path: Path
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise _error("artifact_descriptors must be a filename-keyed mapping")
    expected_names = (
        *bundle_v4_4.INPUT_FILENAMES_V4_4,
        bundle_v4_4.READBACK_REPORT_FILENAME_V4_4,
    )
    if set(value) != set(expected_names) or any(type(name) is not str for name in value):
        raise _error("artifact_descriptors filename inventory mismatch")
    fields = {
        "absolute_path",
        "byte_sha256",
        "size_bytes",
        "mode",
        "uid",
        "nlink",
    }
    normalized: dict[str, dict[str, Any]] = {}
    for name in expected_names:
        item = value[name]
        if not isinstance(item, Mapping) or set(item) != fields:
            raise _error(f"artifact descriptor fields mismatch: {name}")
        path = _absolute(item["absolute_path"], f"artifact descriptor path {name}")
        if path != bundle_path / name:
            raise _error(f"artifact descriptor path mismatch: {name}")
        digest = _sha256(item["byte_sha256"], f"artifact byte SHA {name}")
        if (
            type(item["size_bytes"]) is not int
            or item["size_bytes"] <= 0
            or item["mode"] != 0o600
            or item["uid"] != os.getuid()
            or item["nlink"] != 1
        ):
            raise _error(f"artifact descriptor private-file contract failed: {name}")
        normalized[name] = {
            "absolute_path": str(path),
            "byte_sha256": digest,
            "size_bytes": item["size_bytes"],
            "mode": 0o600,
            "uid": os.getuid(),
            "nlink": 1,
        }
    return normalized


def _postcommit_control_diagnostics(
    entry: Mapping[str, StableFile],
) -> dict[str, Any]:
    """Return diagnostics only; never reject an already committed bundle."""

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
        "cross_system_atomicity_claimed": False,
    }


def _postcommit_immutable_diagnostics(
    strict_source: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        detail = bundle_v4_2.revalidate_recorded_immutable_source_v4_2(strict_source)
    except Exception as exc:
        return {
            "status": "DRIFT_DIAGNOSTIC_ONLY",
            "accepted": False,
            "detail": str(exc),
            "cross_system_atomicity_claimed": False,
        }
    return {
        "status": "VERIFIED_DIAGNOSTIC_ONLY",
        "accepted": detail.get("accepted") is True,
        "detail": detail,
        "cross_system_atomicity_claimed": False,
    }


def run_publish(
    args: argparse.Namespace,
    *,
    private_root: Path = PRODUCTION_PRIVATE_ROOT,
    repository_root: Path = PROJECT_ROOT,
    protected_paths: Sequence[tuple[str, Path]] = PROTECTED_CONTROL_PATHS,
    diagnostic_bundle_path: Path = FIXED_DIAGNOSTIC_BUNDLE_PATH,
    exclusive_rename_probe: Callable[[], None] | None = None,
    _test_fault_hook: Callable[..., None] | None = None,
    _test_race_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Validate, twice collect, exclusively publish, and independently reopen."""

    # These identity checks intentionally precede every platform/root/data
    # interaction.  A stale cutoff cannot disclose or touch publication state.
    cycle_id = _validate_cycle_identity(
        snapshot_id=args.snapshot_id,
        analysis_start=args.analysis_start,
        cutoff=args.cutoff,
    )
    publication_at = _validate_publication_at(args.publication_at, cutoff=args.cutoff)
    probe = exclusive_rename_probe or private_io._require_exclusive_rename_support
    probe()
    _validate_private_root_preflight(private_root, cycle_id=cycle_id)
    entry = _collect_publication_inputs(
        args,
        repository_root=repository_root,
        protected_paths=protected_paths,
        diagnostic_bundle_path=diagnostic_bundle_path,
    )
    if entry.cycle_id != cycle_id:
        raise _error("cycle identity changed across publication preflight")
    if entry.diagnostic_current_mutable_sources_read is not False:
        raise _error("historical diagnostic collection consulted mutable sources")

    def revalidate_inputs() -> None:
        locked = _collect_publication_inputs(
            args,
            repository_root=repository_root,
            protected_paths=protected_paths,
            diagnostic_bundle_path=diagnostic_bundle_path,
        )
        if locked != entry:
            raise _error("publication artifacts/raw bytes/descriptors changed before commit")
        _assert_snapshots_unchanged(entry.code_bindings, label="code binding")
        _assert_snapshots_unchanged(
            entry.protected_controls,
            label="protected control",
        )

    published = bundle_v4_4.publish_candidate_preregistration_bundle_v4_4(
        private_root=private_root,
        artifacts=entry.artifacts,
        raw_input_bindings=entry.raw_input_bindings,
        collected_raw_bytes=entry.collected_raw_bytes,
        revalidate_inputs=revalidate_inputs,
        _test_fault_hook=_test_fault_hook,
        _test_race_hook=_test_race_hook,
    )
    if published.get("accepted") is not True:
        raise _error("v4.4 private publisher did not accept the exclusive commit")
    independent = bundle_v4_4.readback_candidate_preregistration_bundle_files_v4_4(
        published["bundle_path"]
    )
    if independent.get("accepted") is not True:
        raise _error("independent canonical v4.4 bundle reopen was not accepted")
    bundle_path = _absolute(independent["bundle_path"], "published bundle path")
    if bundle_path != private_root / cycle_id:
        raise _error("published bundle path differs from deterministic cycle")
    descriptors = _validated_artifact_descriptors(
        independent["artifact_descriptors"],
        bundle_path=bundle_path,
    )
    report_name = bundle_v4_4.READBACK_REPORT_FILENAME_V4_4
    report = independent["readback_report"]
    if report.get("artifact_semantic_sha256") is None:
        raise _error("v4.4 readback report semantic identity is missing")
    strict_name = (
        prereg_v4_4.V4_2_PREDECESSOR_PREFIX
        + bundle_v4_2.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2
    )
    controls = _postcommit_control_diagnostics(entry.protected_controls)
    immutable = _postcommit_immutable_diagnostics(
        independent["artifacts"][strict_name]
    )
    return {
        "accepted": True,
        "mode": "publish",
        "protocol_version": "v4",
        "evidence_contract_version": "v4.4",
        "cycle_id": cycle_id,
        "publication_at": publication_at,
        "publication_time_authority": prereg_v4_4.PUBLICATION_TIME_AUTHORITY,
        "bundle_path": str(bundle_path),
        "readback_report_path": descriptors[report_name]["absolute_path"],
        "readback_report_byte_sha256": descriptors[report_name]["byte_sha256"],
        "readback_report_semantic_sha256": report["artifact_semantic_sha256"],
        "source_binding_semantic_sha256": entry.source_binding_semantic_sha256,
        "code_binding_set_semantic_sha256": entry.code_binding_set_semantic_sha256,
        "publisher_return_accepted": published.get("accepted") is True,
        "independent_reopen_accepted": True,
        "diagnostic_current_mutable_sources_read": False,
        "exact_once_scope": "deterministic_cycle_directory_RENAME_EXCL_only",
        "external_maintenance_serialization_claimed": False,
        "protected_controls": controls,
        "immutable_source_postcommit": immutable,
        "authority": copy.deepcopy(prereg_v4_4.AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(prereg_v4_4.SIDE_EFFECT_FLAGS),
    }


def run_readback(args: argparse.Namespace) -> dict[str, Any]:
    """Reopen an explicit historical bundle without current mutable sources."""

    expected_byte = _sha256(
        args.expected_readback_report_byte_sha256,
        "expected readback report byte SHA-256",
    )
    expected_semantic = _sha256(
        args.expected_readback_report_semantic_sha256,
        "expected readback report semantic SHA-256",
    )
    bundle_path = _absolute(args.bundle_path, "bundle path")
    result = bundle_v4_4.readback_candidate_preregistration_bundle_files_v4_4(
        bundle_path
    )
    if result.get("accepted") is not True:
        raise _error("historical v4.4 private bundle readback was not accepted")
    descriptors = _validated_artifact_descriptors(
        result["artifact_descriptors"],
        bundle_path=bundle_path,
    )
    report_name = bundle_v4_4.READBACK_REPORT_FILENAME_V4_4
    report = result["readback_report"]
    if descriptors[report_name]["byte_sha256"] != expected_byte:
        raise _error("historical readback report byte SHA-256 mismatch")
    if report.get("artifact_semantic_sha256") != expected_semantic:
        raise _error("historical readback report semantic SHA-256 mismatch")
    strict_name = (
        prereg_v4_4.V4_2_PREDECESSOR_PREFIX
        + bundle_v4_2.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2
    )
    immutable = bundle_v4_2.revalidate_recorded_immutable_source_v4_2(
        result["artifacts"][strict_name]
    )
    if (
        immutable.get("accepted") is not True
        or immutable.get("current_pointer_read") is not False
        or immutable.get("current_components_read") is not False
        or immutable.get("serving_tree_read") is not False
    ):
        raise _error("historical immutable source reopen scope was not exact")
    cycle_root = result["artifacts"][bundle_v4_4.CYCLE_ROOT_FILENAME_V4_4]
    return {
        "accepted": True,
        "mode": "readback",
        "protocol_version": "v4",
        "evidence_contract_version": "v4.4",
        "cycle_id": cycle_root["cycle_id"],
        "bundle_path": str(bundle_path),
        "readback_report_byte_sha256": expected_byte,
        "readback_report_semantic_sha256": expected_semantic,
        "immutable_reopen": immutable,
        "current_latest_pointer_read": False,
        "current_components_read": False,
        "current_protected_controls_read": False,
        "current_diagnostic_sources_read": False,
        "authority": copy.deepcopy(prereg_v4_4.AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(prereg_v4_4.SIDE_EFFECT_FLAGS),
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
    parser.add_argument("--publication-at", required=True)
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
        help="repeat for the exact ordered 15-file v4.4 code binding set",
    )
    parser.add_argument(
        "--expected-protected-control-sha256",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help="repeat for registry/latest_pointer/catalog/fundamental_latest/latest_manifest",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser(
        "publish", help="publish one future exact five-candidate cycle"
    )
    _add_publish_arguments(publish)
    readback = commands.add_parser(
        "readback", help="historically reopen one explicit immutable v4.4 bundle"
    )
    readback.add_argument("--bundle-path", required=True)
    readback.add_argument("--expected-readback-report-byte-sha256", required=True)
    readback.add_argument(
        "--expected-readback-report-semantic-sha256", required=True
    )
    help_text = parser.format_help() + publish.format_help() + readback.format_help()
    if any(token in help_text for token in _FORBIDDEN_ARGUMENT_TOKENS):
        raise _error("forbidden mutation/execution/identity argument leaked into CLI")
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
            "authority": copy.deepcopy(prereg_v4_4.AUTHORITY_FLAGS),
            "side_effects": copy.deepcopy(prereg_v4_4.SIDE_EFFECT_FLAGS),
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
