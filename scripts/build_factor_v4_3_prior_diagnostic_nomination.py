#!/usr/bin/env python3
"""Publish the sealed, owner-private v4.3 prior diagnostic nomination.

This command is deliberately offline, explicit-input only, and fixed to the
historical ``20260717T172132Z`` strict-Parquet snapshot.  It can publish only
the two-input/one-readback diagnostic bundle; it has no registry, proposal,
activation, provider, portfolio, broker, order, or trade surface.

Publication is exact-once.  The runner rejects a stale or different identity
before probing the platform or touching the private root, collects and
validates every input once, repeats the complete collection while holding the
shared private-bundle lock, commits with Darwin ``RENAME_EXCL``, and then
performs an independent canonical readback.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import csv
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
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
    governance_prior_diagnostic_nomination_bundle_v4_3 as bundle_v4_3,
)
from quant_investor.factors import governance_private_bundle_io as private_io  # noqa: E402


PRODUCTION_PRIVATE_ROOT = Path(
    "/Users/maxwell/mySpace/myQuant/reports/factor_governance/private/"
    "v4_3_prior_diagnostic_nomination"
)
FIXED_SNAPSHOT_ID = "20260717T172132Z"
FIXED_ANALYSIS_START = "2021-06-25"
FIXED_CUTOFF = "2026-07-17"
FIXED_RUN_ID = "cn_full_a_v4_3_prior_nomination_20260717_20260717T172132Z"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OID_RE = re.compile(r"[0-9a-f]{40}")
_CODE_EXPECTATION_RE = re.compile(r"([^=]+)=([0-9a-f]{64})")
_LEGACY_DEFINITION_IDENTITY_SHA256 = (
    "227d307ebd56ca81418e4fb8836c6aae0e41a528ff06ec2c705b5d264eab64fa"
)
_FORBIDDEN_ARGUMENT_TOKENS = (
    "--private-root",
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


class FactorV4_3PriorDiagnosticNominationRunnerError(ValueError):
    """Raised when the fixed diagnostic publication fails closed."""


@dataclass(frozen=True)
class StableFile:
    """One stable, current-owner, non-symlink regular-file observation."""

    path: Path
    raw: bytes
    byte_sha256: str
    signature: tuple[int, ...]


@dataclass(frozen=True)
class PublicationInputs:
    """The exact normalized inputs supplied to the private publisher."""

    run_id: str
    artifacts: dict[str, dict[str, Any]]


def _diagnostic() -> Any:
    """Delay the core import so the CLI remains importable during cutover."""

    return importlib.import_module(
        "quant_investor.factors.governance_prior_diagnostic_nomination_v4_3"
    )


def _error(message: str) -> FactorV4_3PriorDiagnosticNominationRunnerError:
    return FactorV4_3PriorDiagnosticNominationRunnerError(message)


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
    """Read one owner-controlled file twice through a no-follow descriptor."""

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
        return StableFile(path, first, digest, _signature(after))
    except FactorV4_3PriorDiagnosticNominationRunnerError:
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


def _validate_run_identity(
    *,
    snapshot_id: Any,
    analysis_start: Any,
    cutoff: Any,
) -> str:
    """Accept only the frozen historical diagnostic identity."""

    if type(snapshot_id) is not str:
        raise _error("snapshot_id must be the frozen canonical UTC timestamp")
    try:
        parsed_snapshot = datetime.strptime(snapshot_id, "%Y%m%dT%H%M%SZ")
        parsed_start = date.fromisoformat(str(analysis_start))
        parsed_cutoff = date.fromisoformat(str(cutoff))
    except ValueError as exc:
        raise _error("snapshot_id/analysis_start/cutoff identity is invalid") from exc
    if (
        parsed_snapshot.strftime("%Y%m%dT%H%M%SZ") != snapshot_id
        or parsed_start.isoformat() != analysis_start
        or parsed_cutoff.isoformat() != cutoff
    ):
        raise _error("snapshot_id/analysis_start/cutoff must be canonical")
    if (
        snapshot_id != FIXED_SNAPSHOT_ID
        or analysis_start != FIXED_ANALYSIS_START
        or cutoff != FIXED_CUTOFF
    ):
        raise _error("stale or different diagnostic identity is prohibited")
    return FIXED_RUN_ID


def _validate_definition_identity_expectation(value: Any) -> str:
    """Reject the superseded provisional identity before any platform probe."""

    expected = _sha256(value, "expected definition identity SHA-256")
    if expected == _LEGACY_DEFINITION_IDENTITY_SHA256:
        raise _error("superseded provisional definition identity is prohibited")
    diagnostic = _diagnostic()
    rebuilt = diagnostic.definition_identity_sha256_v4_3()
    if expected != rebuilt:
        raise _error("expected definition identity differs from the exact v4.3 identity")
    diagnostic.validate_definition_identity_payload_v4_3(
        diagnostic.build_definition_identity_payload_v4_3()
    )
    return rebuilt


def _validate_private_root_preflight(root: Path, *, run_id: str) -> None:
    """Require the exact existing owner-0700 lane and an absent destination."""

    if not root.is_absolute() or os.path.abspath(root) != str(root):
        raise _error("private root must be absolute and normalized")
    suffix = bundle_v4_3.ROOT_SUFFIX_V4_3_PRIOR_DIAGNOSTIC_NOMINATION
    if tuple(root.parts[-len(suffix) :]) != tuple(suffix):
        raise _error("private root must be the exact v4.3 diagnostic lane")
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
    destination = root / run_id
    try:
        os.lstat(destination)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise _error(f"cannot inspect deterministic destination: {destination}") from exc
    raise _error(f"deterministic run destination already exists: {destination}")


def _parse_code_expectations(values: Sequence[str]) -> dict[str, str]:
    diagnostic = _diagnostic()
    rows: dict[str, str] = {}
    for value in values:
        if type(value) is not str:
            raise _error("expected project code binding must be PATH=SHA256")
        match = _CODE_EXPECTATION_RE.fullmatch(value)
        if match is None:
            raise _error("expected project code binding must be PATH=SHA256")
        relative, digest = match.groups()
        path = Path(relative)
        if (
            path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or relative in rows
        ):
            raise _error("expected project code paths must be unique normalized relatives")
        rows[relative] = digest
    expected_paths = tuple(diagnostic.PROJECT_BINDING_PATHS)
    if set(rows) != set(expected_paths) or len(rows) != len(expected_paths):
        missing = sorted(set(expected_paths) - set(rows))
        extra = sorted(set(rows) - set(expected_paths))
        raise _error(
            "expected project code inventory mismatch: "
            f"missing={','.join(missing) or '-'};extra={','.join(extra) or '-'}"
        )
    return {path: rows[path] for path in expected_paths}


def _project_bindings(
    *,
    repository_root: Path,
    expected: Mapping[str, str],
) -> list[dict[str, Any]]:
    diagnostic = _diagnostic()
    result: list[dict[str, Any]] = []
    for relative in diagnostic.PROJECT_BINDING_PATHS:
        observed = _stable_file(
            repository_root / relative,
            label=f"project code binding {relative}",
            expected_sha256=expected[relative],
            max_bytes=32 * 1024 * 1024,
        )
        result.append(
            {
                "relative_path": relative,
                "byte_sha256": observed.byte_sha256,
                "size_bytes": len(observed.raw),
            }
        )
    return result


def _python_binding(expected_sha256: str) -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    observed = _stable_file(
        executable,
        label="resolved CPython executable",
        expected_sha256=expected_sha256,
        max_bytes=128 * 1024 * 1024,
    )
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "executable": str(executable),
        "executable_sha256": observed.byte_sha256,
    }


def _record_hash(value: str, *, label: str) -> str:
    algorithm, separator, encoded = value.partition("=")
    if algorithm != "sha256" or not separator or not encoded:
        raise _error(f"{label} must contain one sha256 RECORD digest")
    try:
        padding = "=" * ((4 - len(encoded) % 4) % 4)
        raw = base64.urlsafe_b64decode((encoded + padding).encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise _error(f"{label} contains an invalid RECORD digest") from exc
    if len(raw) != 32:
        raise _error(f"{label} RECORD digest must decode to SHA-256")
    return raw.hex()


def _distribution_binding(
    *,
    name: str,
    expected_version: str,
    expected_count: int,
    expected_inventory_sha256: str,
) -> dict[str, Any]:
    """Reconcile every package-prefix RECORD row with actual installed bytes."""

    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise _error(f"required runtime distribution is unavailable: {name}") from exc
    if distribution.version != expected_version:
        raise _error(f"runtime distribution version mismatch: {name}")
    distribution_root = Path(str(distribution.locate_file("")))
    metadata_path = Path(getattr(distribution, "_path", ""))
    if not metadata_path.is_absolute() or metadata_path.name.endswith(".dist-info") is False:
        raise _error(f"runtime distribution metadata path is unavailable: {name}")
    record_path = metadata_path / "RECORD"
    record = _stable_file(
        record_path,
        label=f"{name} RECORD",
        max_bytes=64 * 1024 * 1024,
    )
    try:
        rows = list(csv.reader(io.StringIO(record.raw.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise _error(f"{name} RECORD is not valid UTF-8 CSV") from exc
    prefix = f"{name}/"
    selected: list[tuple[str, str, str]] = []
    for index, row in enumerate(rows):
        if len(row) != 3:
            raise _error(f"{name} RECORD row[{index}] does not have three fields")
        if row[0].startswith(prefix):
            selected.append((row[0], row[1], row[2]))
    selected.sort(key=lambda item: item[0])
    if len(selected) != expected_count or len({row[0] for row in selected}) != len(selected):
        raise _error(f"{name} selected RECORD inventory count/uniqueness mismatch")

    inventory: list[dict[str, Any]] = []
    unhashed = 0
    hash_mismatch = 0
    size_mismatch = 0
    for relative, encoded_hash, encoded_size in selected:
        if (
            relative.startswith("/")
            or "\x00" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
        ):
            raise _error(f"{name} RECORD package path is unsafe")
        if not encoded_hash or not encoded_size:
            unhashed += 1
            continue
        expected_hash = _record_hash(encoded_hash, label=f"{name}:{relative}")
        try:
            expected_size = int(encoded_size)
        except ValueError as exc:
            raise _error(f"{name} RECORD size is not an integer: {relative}") from exc
        if expected_size < 0 or str(expected_size) != encoded_size:
            raise _error(f"{name} RECORD size is not canonical: {relative}")
        observed = _stable_file(
            distribution_root / relative,
            label=f"{name} installed file {relative}",
            max_bytes=512 * 1024 * 1024,
        )
        if observed.byte_sha256 != expected_hash:
            hash_mismatch += 1
        if len(observed.raw) != expected_size:
            size_mismatch += 1
        inventory.append(
            {
                "path": relative,
                "sha256": observed.byte_sha256,
                "size_bytes": len(observed.raw),
            }
        )
    if unhashed or hash_mismatch or size_mismatch or len(inventory) != expected_count:
        raise _error(f"{name} RECORD reconciliation failed")
    semantic_sha = _diagnostic().semantic_sha256_v4_3(inventory)
    if semantic_sha != expected_inventory_sha256:
        raise _error(f"{name} installed inventory semantic SHA-256 mismatch")
    return {
        "distribution": name,
        "version": distribution.version,
        "package_prefix": prefix,
        "record_path": str(record_path),
        "record_byte_sha256": record.byte_sha256,
        "record_selected_entry_count": len(inventory),
        "unhashed_selected_entry_count": unhashed,
        "hash_mismatch_count": hash_mismatch,
        "size_mismatch_count": size_mismatch,
        "file_inventory": inventory,
        "file_inventory_semantic_sha256": semantic_sha,
    }


def _runtime_distributions() -> list[dict[str, Any]]:
    return [
        _distribution_binding(
            name=name,
            expected_version=version,
            expected_count=count,
            expected_inventory_sha256=digest,
        )
        for name, version, count, digest in _diagnostic().EXPECTED_DISTRIBUTIONS
    ]


def _run_git(repository: Path, arguments: Sequence[str]) -> bytes:
    if not repository.is_absolute() or repository.is_symlink() or not repository.is_dir():
        raise _error("myQuant Git repository must be an absolute real directory")
    # Git gives several GIT_* variables precedence over ``-C``.  Inheriting
    # them would let a caller redirect object reads away from the repository
    # whose path is being attested, so discard the entire selector namespace
    # before installing the small, deterministic environment used here.
    environment = {
        **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
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
        raise _error(f"pinned myQuant Git object read failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise _error(f"pinned myQuant Git object read failed: {detail}")
    return completed.stdout


def _assignment_key(target: ast.AST) -> str | None:
    if not isinstance(target, ast.Subscript):
        return None
    slice_value = target.slice
    if isinstance(slice_value, ast.Constant) and type(slice_value.value) is str:
        return slice_value.value
    return None


def _verify_git_source(args: argparse.Namespace, *, repository_root: Path) -> None:
    diagnostic = _diagnostic()
    identity = diagnostic.validate_definition_identity_payload_v4_3(
        diagnostic.build_definition_identity_payload_v4_3()
    )
    expected = identity["source_binding"]
    repository = _absolute(args.myquant_git_repository, "myQuant Git repository")
    try:
        same_repository = repository.samefile(repository_root)
    except OSError as exc:
        raise _error("myQuant Git repository identity is unavailable") from exc
    if (
        not same_repository
        or args.myquant_commit != expected["commit"]
        or args.myquant_source_path != expected["path"]
        or args.myquant_source_mode != "100644"
        or args.myquant_source_blob_oid != expected["blob_oid"]
        or args.expected_myquant_source_sha256 != expected["file_sha256"]
    ):
        raise _error("explicit myQuant source pin differs from the v4.3 identity")
    if _OID_RE.fullmatch(args.myquant_commit) is None or _OID_RE.fullmatch(
        args.myquant_source_blob_oid
    ) is None:
        raise _error("myQuant commit/blob must be lowercase Git OIDs")
    resolved = _run_git(
        repository, ["rev-parse", "--verify", f"{args.myquant_commit}^{{commit}}"]
    ).decode("ascii").strip()
    if resolved != args.myquant_commit:
        raise _error("myQuant commit resolution mismatch")
    tree_row = _run_git(
        repository, ["ls-tree", args.myquant_commit, "--", args.myquant_source_path]
    ).decode("utf-8").strip()
    prefix, separator, tree_path = tree_row.partition("\t")
    if (
        not separator
        or tree_path != args.myquant_source_path
        or prefix.split()
        != [args.myquant_source_mode, "blob", args.myquant_source_blob_oid]
    ):
        raise _error("myQuant pinned tree entry mismatch")
    raw = _run_git(repository, ["cat-file", "blob", args.myquant_source_blob_oid])
    if hashlib.sha256(raw).hexdigest() != expected["file_sha256"]:
        raise _error("myQuant pinned source bytes mismatch")
    try:
        tree = ast.parse(raw.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise _error("myQuant pinned source is not parseable UTF-8 Python") from exc
    values = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(_assignment_key(target) == "VOL_OF_VOL_20D" for target in node.targets)
    ]
    if len(values) != 1:
        raise _error("myQuant source must contain one VOL_OF_VOL_20D assignment")
    dumped = ast.dump(values[0], annotate_fields=True, include_attributes=False)
    if hashlib.sha256(dumped.encode("utf-8")).hexdigest() != expected[
        "value_ast_sha256"
    ]:
        raise _error("myQuant VOL_OF_VOL_20D value AST mismatch")


def _validate_comparison_catalog(args: argparse.Namespace) -> None:
    catalog_file = _stable_file(
        _absolute(args.comparison_catalog_path, "comparison catalog"),
        label="comparison catalog",
        expected_sha256=args.expected_comparison_catalog_sha256,
        max_bytes=256 * 1024 * 1024,
    )
    catalog = _strict_json_object(catalog_file.raw, "comparison catalog")
    candidates = catalog.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise _error("comparison catalog candidate inventory is missing")
    target_name = "pv_low_vol_of_vol_20d"
    target_identity = _diagnostic().definition_identity_sha256_v4_3()
    for index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise _error(f"comparison catalog candidate[{index}] must be an object")
        if row.get("name") == target_name or target_identity in {
            row.get("definition_sha256"),
            row.get("definition_identity_sha256"),
        }:
            raise _error("nominated definition collides with the comparison inventory")


def _remask(value: Any, mask: Any) -> Any:
    import numpy as np

    return value.replace([np.inf, -np.inf], np.nan).where(mask)


def _reconstruct_scope(
    *,
    bound: Any,
    design_source: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    import pandas as pd

    from quant_investor.factors import governance_aquant_no_label_eval_v4_1 as evaluator
    from quant_investor.factors import governance_source_v4_1 as source

    normalized = source.validate_pit_records_v4_1(bound.pit_records)
    design = source.validate_design_source_node_v4_1(
        design_source,
        pit_records=normalized,
        expected_component_count=_diagnostic().COMPONENT_COUNT,
    )
    if design != dict(design_source):
        raise _error("sealed design source normalization drift")
    sessions = list(bound.calendar_sessions)
    symbols = [row["symbol"] for row in normalized]
    if (
        sessions != design["calendar_sessions"]
        or symbols != sorted(symbols)
        or len(sessions) != _diagnostic().SESSION_COUNT
        or len(symbols) != _diagnostic().SCOPE_COLUMN_COUNT
    ):
        raise _error("sealed PIT axes differ from the frozen 1227x5866 contract")
    session_values = np.asarray(sessions, dtype="U10")
    values = np.zeros((len(sessions), len(symbols)), dtype=bool)
    for column, row in enumerate(normalized):
        start = int(np.searchsorted(session_values, row["effective_from"], side="left"))
        end = (
            len(sessions)
            if row["effective_to"] is None
            else int(np.searchsorted(session_values, row["effective_to"], side="left"))
        )
        if start < end:
            values[start:end, column] = True
    cutoff_position = sessions.index(_diagnostic().CUTOFF_DATE)
    component_set = set(bound.component_symbols)
    values[cutoff_position, :] &= np.asarray(
        [symbol in component_set for symbol in symbols], dtype=bool
    )
    mask = pd.DataFrame(
        values,
        index=pd.DatetimeIndex(pd.to_datetime(sessions), name="trade_date"),
        columns=symbols,
        dtype=bool,
    )
    descriptor = evaluator.matrix_hash_descriptor_v4_1(mask.astype(float))
    expected = _diagnostic().SOURCE_BINDING_EXPECTED
    if (
        descriptor["shape"] != [
            _diagnostic().SESSION_COUNT,
            _diagnostic().SCOPE_COLUMN_COUNT,
        ]
        or descriptor["matrix_sha256"] != expected["eligibility_matrix_sha256"]
        or descriptor["date_axis_sha256"] != expected["date_axis_sha256"]
        or descriptor["symbol_axis_sha256"] != expected["symbol_axis_sha256"]
    ):
        raise _error("reconstructed PIT eligibility matrix binding mismatch")
    return mask, descriptor


def _load_adj_close(
    *,
    table_root: Path,
    inventory: Sequence[Mapping[str, Any]],
    mask: Any,
) -> Any:
    import pandas as pd
    import pyarrow.dataset as ds

    paths = [
        str(table_root / row["relative_path"])
        for row in inventory
        if row.get("dataset_member") is True
    ]
    if not paths:
        raise _error("strict table inventory has no dataset members")
    dataset = ds.dataset(paths, format="parquet")
    required = ("trade_date", "ts_code", "adj_close")
    if not set(required).issubset(set(dataset.schema.names)):
        raise _error("strict table is missing adj_close; fallback is prohibited")
    sessions = mask.index.strftime("%Y%m%d")
    table = dataset.to_table(
        columns=list(required),
        filter=(ds.field("trade_date") >= sessions[0])
        & (ds.field("trade_date") <= sessions[-1]),
    )
    raw = table.to_pandas()
    if raw.empty or list(raw.columns) != list(required):
        raise _error("strict adj_close projection is empty or reordered")
    if raw.duplicated(["trade_date", "ts_code"]).any():
        raise _error("strict adj_close projection contains duplicate rows")
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d")
    matrix = raw.pivot(
        index="trade_date", columns="ts_code", values="adj_close"
    ).reindex(index=mask.index, columns=mask.columns)
    return _remask(matrix.astype(float), mask)


def _monthly_dates(index: Any) -> list[Any]:
    import pandas as pd

    ordered = pd.DatetimeIndex(index)
    if not ordered.is_monotonic_increasing or ordered.has_duplicates:
        raise _error("sealed calendar must remain sorted and unique")
    diagnostic = _diagnostic()
    month_ends = pd.Series(ordered, index=ordered).groupby(ordered.to_period("M")).max()
    first = pd.Timestamp(ordered[diagnostic.WARMUP_SESSIONS])
    last = pd.Timestamp(ordered[-diagnostic.HORIZON_SESSIONS - 1])
    dates = [
        pd.Timestamp(value)
        for value in month_ends[(month_ends >= first) & (month_ends <= last)].tolist()
    ]
    rendered = tuple(value.strftime("%Y-%m-%d") for value in dates)
    if rendered != tuple(diagnostic.EXPECTED_MONTHLY_DATES):
        raise _error("formal closed-calendar month-end schedule mismatch")
    return dates


def _analysis_floor(adj_close: Any) -> str:
    import numpy as np
    import pandas as pd

    valid = adj_close.notna()
    dates = adj_close.index.values.astype("datetime64[ns]")
    first = np.asarray(
        [
            adj_close[column].first_valid_index().to_datetime64()
            if adj_close[column].first_valid_index() is not None
            else np.datetime64("NaT")
            for column in adj_close.columns
        ],
        dtype="datetime64[ns]",
    )
    last = np.asarray(
        [
            adj_close[column].last_valid_index().to_datetime64()
            if adj_close[column].last_valid_index() is not None
            else np.datetime64("NaT")
            for column in adj_close.columns
        ],
        dtype="datetime64[ns]",
    )
    observable = (
        (dates[:, None] >= first[None, :])
        & (dates[:, None] <= last[None, :])
        & ~np.isnat(first[None, :])
        & ~np.isnat(last[None, :])
    )
    counts = pd.Series(observable.sum(axis=1), index=adj_close.index, dtype=float)
    maximum = int(counts.max())
    minimum_cross_section = (
        max(20, int(maximum * 0.60))
        if maximum >= 20
        else max(1, int(np.ceil(maximum * 0.60)))
    )
    coverage = valid.sum(axis=1).div(counts.replace(0.0, np.nan))
    ready = coverage[(coverage >= 0.95) & (counts >= minimum_cross_section)]
    if ready.empty:
        raise _error("price-only analysis floor is unavailable")
    result = pd.Timestamp(ready.index[0]).strftime("%Y-%m-%d")
    if result != _diagnostic().ANALYSIS_FLOOR:
        raise _error("price-only analysis floor differs from the sealed contract")
    return result


def _signals(adj_close: Any, mask: Any) -> dict[str, Any]:
    price = _remask(adj_close, mask)
    returns = _remask(price.pct_change(periods=1, fill_method=None), mask)
    vol5 = _remask(
        returns.rolling(window=5, min_periods=5, center=False).std(ddof=1),
        mask,
    )
    vol_of_vol = _remask(
        vol5.rolling(window=20, min_periods=20, center=False).std(ddof=1),
        mask,
    )

    mom252 = _remask(price.pct_change(periods=252, fill_method=None), mask)
    skip21 = _remask(mom252.shift(periods=21), mask)

    mom60 = _remask(price.pct_change(periods=60, fill_method=None), mask)
    mean120 = _remask(
        mom60.rolling(window=120, min_periods=120, center=False).mean(),
        mask,
    )
    excess = _remask(mom60.sub(mean120), mask)
    return {
        "VOL_OF_VOL_20D": vol_of_vol,
        "MOM_12M_SKIP1M": skip21,
        "EXCESS_MOM_60D": excess,
    }


def _coverage_row(signal: Any, mask: Any, date_value: Any) -> dict[str, Any]:
    stamp = date_value
    finite = int(signal.loc[stamp].notna().sum())
    eligible = int(mask.loc[stamp].sum())
    count = _diagnostic().SCOPE_COLUMN_COUNT
    return {
        "date": stamp.strftime("%Y-%m-%d"),
        "finite_signal_count": finite,
        "eligible_signal_count": eligible,
        "scope_column_count": count,
        "coverage_rate": float(finite / count),
    }


def _evaluation_row(
    signal: Any,
    forward_return: Any,
    mask: Any,
    date_value: Any,
) -> dict[str, Any]:
    import pandas as pd

    base = _coverage_row(signal, mask, date_value)
    left = signal.loc[date_value]
    right = forward_return.loc[date_value]
    common = pd.concat([left.rename("signal"), right.rename("forward")], axis=1).dropna()
    common_count = int(len(common))
    reason: str | None = None
    rank_ic: float | None = None
    if common_count < _diagnostic().MIN_COMMON_SYMBOLS:
        reason = "COMMON_SYMBOL_COUNT_LT_20"
    elif common["signal"].nunique(dropna=True) <= 1:
        reason = "SIGNAL_NOT_UNIQUE"
    elif common["forward"].nunique(dropna=True) <= 1:
        reason = "FORWARD_NOT_UNIQUE"
    else:
        value = pd.Series(common["signal"], dtype="float64").corr(
            pd.Series(common["forward"], dtype="float64"),
            method="spearman",
            min_periods=20,
        )
        if pd.isna(value) or not math.isfinite(float(value)):
            reason = "RANK_IC_NONFINITE"
        else:
            rank_ic = float(value)
    return {
        **base,
        "common_symbol_count": common_count,
        "rank_ic": rank_ic,
        "exclusion_reason": reason,
    }


def _attempt_rows(
    *,
    source_name: str,
    signal: Any,
    forward_return: Any,
    mask: Any,
    monthly_dates: Sequence[Any],
) -> dict[str, Any]:
    maturity = [_coverage_row(signal, mask, date_value) for date_value in monthly_dates]
    qualifying = [
        row["date"]
        for row in maturity
        if row["coverage_rate"] >= _diagnostic().MATURITY_COVERAGE_THRESHOLD
    ]
    if not qualifying:
        raise _error(f"{source_name} never reaches the frozen maturity threshold")
    effective_start = qualifying[0]
    expected = next(
        row for row in _diagnostic().ATTEMPT_SPECS if row["source_name"] == source_name
    )
    if effective_start != expected["effective_start"]:
        raise _error(f"{source_name} effective maturity start mismatch")
    suffix = [date_value for date_value in monthly_dates if date_value.strftime("%Y-%m-%d") >= effective_start]
    evaluation = [
        _evaluation_row(signal, forward_return, mask, date_value)
        for date_value in suffix
    ]
    return _diagnostic().build_prior_diagnostic_attempt_v4_3(
        source_name=source_name,
        maturity_coverage_rows=maturity,
        evaluation_period_rows=evaluation,
    )


def _bind_sealed_source(
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], Any, dict[str, Any]]:
    from quant_investor.factors import governance_source_readback_v4_1 as readback

    diagnostic = _diagnostic()
    binding_file = _stable_file(
        _absolute(args.sealed_cutoff_binding_path, "sealed cutoff binding"),
        label="sealed cutoff binding",
        expected_sha256=args.expected_sealed_cutoff_binding_sha256,
        max_bytes=64 * 1024 * 1024,
    )
    design_file = _stable_file(
        _absolute(args.sealed_design_source_path, "sealed design source"),
        label="sealed design source",
        expected_sha256=args.expected_sealed_design_source_sha256,
        max_bytes=256 * 1024 * 1024,
    )
    expected_root = Path(diagnostic.SOURCE_BINDING_EXPECTED["cutoff_bundle_path"])
    if (
        binding_file.path != expected_root / readback.INPUT_BINDING_FILENAME
        or design_file.path != expected_root / readback.DESIGN_SOURCE_FILENAME
        or binding_file.byte_sha256
        != diagnostic.SOURCE_BINDING_EXPECTED["cutoff_input_binding_byte_sha256"]
        or design_file.byte_sha256
        != diagnostic.SOURCE_BINDING_EXPECTED["design_source_byte_sha256"]
    ):
        raise _error("sealed v4.1 source artifact path/hash mismatch")
    sealed_binding = _strict_json_object(binding_file.raw, "sealed cutoff binding")
    sealed_design = _strict_json_object(design_file.raw, "sealed design source")
    bound = readback.bind_explicit_cutoff_inputs_v4_1(
        latest_pointer_path=args.latest_pointer_path,
        expected_latest_pointer_sha256=args.expected_latest_pointer_sha256,
        snapshot_manifest_path=args.snapshot_manifest_path,
        expected_snapshot_manifest_sha256=args.expected_snapshot_manifest_sha256,
        components_path=args.components_path,
        expected_components_sha256=args.expected_components_sha256,
        expected_full_a_semantic_sha256=args.expected_full_a_semantic_sha256,
        pit_generation_manifest_path=args.pit_generation_manifest_path,
        expected_pit_generation_manifest_sha256=args.expected_pit_generation_manifest_sha256,
        pit_membership_path=args.pit_membership_path,
        expected_pit_membership_sha256=args.expected_pit_membership_sha256,
        table_root=args.table_root,
        snapshot_id=args.snapshot_id,
        analysis_start=args.analysis_start,
        cutoff_date=args.cutoff,
        expected_full_a_count=args.expected_full_a_count,
        expected_serving_inventory_count=args.expected_serving_inventory_count,
    )
    if bound.binding != sealed_binding:
        raise _error("reconstructed explicit cutoff binding differs from sealed bytes")
    if sealed_binding["table"]["inventory_sha256"] != _sha256(
        args.expected_table_inventory_sha256, "expected table inventory SHA-256"
    ):
        raise _error("strict table inventory SHA-256 mismatch")
    mask, mask_descriptor = _reconstruct_scope(
        bound=bound,
        design_source=sealed_design,
    )
    return bound, sealed_design, mask, mask_descriptor


def _collect_bound_diagnostic(
    args: argparse.Namespace,
    *,
    repository_root: Path,
) -> PublicationInputs:
    from quant_investor.factors import governance_aquant_no_label_eval_v4_1 as evaluator

    diagnostic = _diagnostic()
    if _validate_run_identity(
        snapshot_id=args.snapshot_id,
        analysis_start=args.analysis_start,
        cutoff=args.cutoff,
    ) != diagnostic.RUN_ID:
        raise _error("core and runner deterministic identities differ")
    _validate_definition_identity_expectation(
        args.expected_definition_identity_sha256
    )
    _verify_git_source(args, repository_root=repository_root)
    _validate_comparison_catalog(args)
    bound, sealed_design, mask, mask_descriptor = _bind_sealed_source(args)
    table_root = _absolute(args.table_root, "strict table root")
    inventory = bound.binding["table"]["parquet_inventory"]
    adj_close = _load_adj_close(
        table_root=table_root,
        inventory=inventory,
        mask=mask,
    )
    if _analysis_floor(adj_close) != diagnostic.ANALYSIS_FLOOR:
        raise _error("analysis floor drift")
    forward_return = _remask(
        adj_close.shift(-diagnostic.HORIZON_SESSIONS)
        .div(adj_close.shift(-1))
        .sub(1.0),
        mask,
    )
    monthly_dates = _monthly_dates(mask.index)
    signals = _signals(adj_close, mask)
    matrix_bindings = {
        "adj_close": evaluator.matrix_hash_descriptor_v4_1(adj_close)["matrix_sha256"],
        "forward_return": evaluator.matrix_hash_descriptor_v4_1(forward_return)["matrix_sha256"],
        **{
            name: evaluator.matrix_hash_descriptor_v4_1(signal)["matrix_sha256"]
            for name, signal in signals.items()
        },
    }
    if matrix_bindings != diagnostic.MATRIX_BINDINGS_EXPECTED:
        raise _error("recomputed matrix binding inventory mismatch")

    source_expected = diagnostic.SOURCE_BINDING_EXPECTED
    source_binding = {
        "cutoff_bundle_path": str(Path(args.sealed_cutoff_binding_path).parent),
        "cutoff_input_binding_byte_sha256": args.expected_sealed_cutoff_binding_sha256,
        "design_source_byte_sha256": args.expected_sealed_design_source_sha256,
        "design_source_semantic_sha256": sealed_design["semantic_sha256"],
        "snapshot_id": bound.binding["snapshot_id"],
        "cutoff_date": bound.binding["cutoff_date"],
        "analysis_floor": diagnostic.ANALYSIS_FLOOR,
        "calendar_semantic_sha256": bound.binding["calendar"]["semantic_sha256"],
        "table_inventory_sha256": bound.binding["table"]["inventory_sha256"],
        "pit_membership_sha256": bound.binding["pit_generation"]["membership"]["sha256"],
        "component_symbols_newline_sha256": bound.binding["components"]["newline_set_sha256"],
        "session_count": len(bound.calendar_sessions),
        "scope_column_count": mask.shape[1],
        "component_count": len(bound.component_symbols),
        "eligibility_matrix_sha256": mask_descriptor["matrix_sha256"],
        "date_axis_sha256": mask_descriptor["date_axis_sha256"],
        "symbol_axis_sha256": mask_descriptor["symbol_axis_sha256"],
    }
    if source_binding != source_expected:
        raise _error("reconstructed source binding differs from the exact v4.3 contract")
    expected_code = _parse_code_expectations(args.expected_project_code_sha256)
    runtime = diagnostic.build_prior_diagnostic_runtime_binding_v4_3(
        python=_python_binding(args.expected_python_executable_sha256),
        distributions=_runtime_distributions(),
        project_bindings=_project_bindings(
            repository_root=repository_root,
            expected=expected_code,
        ),
        source_binding=source_binding,
        matrix_bindings=matrix_bindings,
    )
    attempts = [
        _attempt_rows(
            source_name=spec["source_name"],
            signal=signals[spec["source_name"]],
            forward_return=forward_return,
            mask=mask,
            monthly_dates=monthly_dates,
        )
        for spec in diagnostic.ATTEMPT_SPECS
    ]
    nomination = diagnostic.build_prior_diagnostic_nomination_v4_3(
        attempts=attempts,
        runtime_binding_semantic_sha256=runtime["artifact_semantic_sha256"],
    )
    normalized = bundle_v4_3.validate_prior_diagnostic_nomination_bundle_inputs_v4_3(
        {
            bundle_v4_3.PRIOR_DIAGNOSTIC_RUNTIME_BINDING_FILENAME_V4_3: runtime,
            bundle_v4_3.PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3: nomination,
        }
    )
    return PublicationInputs(
        run_id=diagnostic.RUN_ID,
        artifacts={name: dict(value) for name, value in normalized.items()},
    )


def _collect_publication_inputs(
    args: argparse.Namespace,
    *,
    repository_root: Path,
) -> PublicationInputs:
    """Build both diagnostic artifacts from explicit, hash-bound inputs."""

    # The substantive collector is kept below the publication boundary so unit
    # tests can exercise real shared I/O with a synthetic, already-validated
    # artifact graph.  Production always enters through this exact function.
    return _collect_bound_diagnostic(args, repository_root=repository_root)


def _validated_artifact_descriptors(
    value: Any,
    *,
    bundle_path: Path,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise _error("artifact_descriptors must be a filename-keyed mapping")
    expected_names = (
        *bundle_v4_3.INPUT_FILENAMES_V4_3,
        bundle_v4_3.PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3,
    )
    if set(value) != set(expected_names) or any(
        type(name) is not str for name in value
    ):
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
        digest = _sha256(item["byte_sha256"], f"artifact descriptor SHA {name}")
        if type(item["size_bytes"]) is not int or item["size_bytes"] <= 0:
            raise _error(f"artifact descriptor size mismatch: {name}")
        if (
            item["mode"] != 0o600
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


def run_publish(
    args: argparse.Namespace,
    *,
    private_root: Path = PRODUCTION_PRIVATE_ROOT,
    repository_root: Path = PROJECT_ROOT,
    exclusive_rename_probe: Callable[[], None] | None = None,
    _test_race_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Publish once, independently reopen, and expose only diagnostic facts."""

    run_id = _validate_run_identity(
        snapshot_id=args.snapshot_id,
        analysis_start=args.analysis_start,
        cutoff=args.cutoff,
    )
    _validate_definition_identity_expectation(
        args.expected_definition_identity_sha256
    )
    probe = exclusive_rename_probe or private_io._require_exclusive_rename_support
    probe()
    _validate_private_root_preflight(private_root, run_id=run_id)
    initial = _collect_publication_inputs(args, repository_root=repository_root)
    if initial.run_id != run_id:
        raise _error("derived run identity changed across preflight")

    def revalidate_inputs() -> None:
        locked = _collect_publication_inputs(args, repository_root=repository_root)
        if locked != initial:
            raise _error("publication inputs changed before commit")

    published = bundle_v4_3.publish_prior_diagnostic_nomination_bundle_v4_3(
        private_root=private_root,
        artifacts=initial.artifacts,
        revalidate_inputs=revalidate_inputs,
        _test_race_hook=_test_race_hook,
    )
    independent = bundle_v4_3.readback_prior_diagnostic_nomination_bundle_v4_3(
        published["bundle_path"]
    )
    if independent.get("accepted") is not True:
        raise _error("independent canonical bundle reopen was not accepted")
    bundle_path = _absolute(independent["bundle_path"], "published bundle path")
    descriptors = _validated_artifact_descriptors(
        independent["artifact_descriptors"],
        bundle_path=bundle_path,
    )
    report_name = (
        bundle_v4_3.PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3
    )
    report = independent["readback_report"]
    nomination = independent["artifacts"][
        bundle_v4_3.PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3
    ]
    return {
        "accepted": True,
        "mode": "publish",
        "protocol_version": "v4",
        "evidence_contract_version": "v4.3",
        "run_id": run_id,
        "bundle_path": str(bundle_path),
        "readback_report_path": descriptors[report_name]["absolute_path"],
        "readback_report_byte_sha256": descriptors[report_name]["byte_sha256"],
        "readback_report_semantic_sha256": report["artifact_semantic_sha256"],
        "publisher_return_accepted": published.get("accepted") is True,
        "independent_reopen_accepted": True,
        "exact_once_scope": "deterministic_run_directory_RENAME_EXCL_only",
        "authority": copy.deepcopy(nomination["authority"]),
        "side_effects": copy.deepcopy(nomination["side_effects"]),
    }


def run_readback(args: argparse.Namespace) -> dict[str, Any]:
    """Reopen historical bundle bytes without consulting mutable sources."""

    expected_byte = _sha256(
        args.expected_readback_report_byte_sha256,
        "expected readback report byte SHA-256",
    )
    expected_semantic = _sha256(
        args.expected_readback_report_semantic_sha256,
        "expected readback report semantic SHA-256",
    )
    bundle_path = _absolute(args.bundle_path, "bundle path")
    result = bundle_v4_3.readback_prior_diagnostic_nomination_bundle_v4_3(
        bundle_path
    )
    descriptors = _validated_artifact_descriptors(
        result["artifact_descriptors"], bundle_path=bundle_path
    )
    report_name = (
        bundle_v4_3.PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3
    )
    report = result["readback_report"]
    if descriptors[report_name]["byte_sha256"] != expected_byte:
        raise _error("historical readback report byte SHA-256 mismatch")
    if report["artifact_semantic_sha256"] != expected_semantic:
        raise _error("historical readback report semantic SHA-256 mismatch")
    nomination = result["artifacts"][
        bundle_v4_3.PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3
    ]
    return {
        "accepted": True,
        "mode": "readback",
        "protocol_version": "v4",
        "evidence_contract_version": "v4.3",
        "run_id": nomination["run_id"],
        "bundle_path": str(bundle_path),
        "readback_report_byte_sha256": expected_byte,
        "readback_report_semantic_sha256": expected_semantic,
        "current_mutable_sources_read": False,
        "authority": copy.deepcopy(nomination["authority"]),
        "side_effects": copy.deepcopy(nomination["side_effects"]),
    }


def _add_publish_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--analysis-start", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--expected-definition-identity-sha256", required=True)
    parser.add_argument("--sealed-cutoff-binding-path", required=True)
    parser.add_argument("--expected-sealed-cutoff-binding-sha256", required=True)
    parser.add_argument("--sealed-design-source-path", required=True)
    parser.add_argument("--expected-sealed-design-source-sha256", required=True)
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
    parser.add_argument("--expected-full-a-count", type=int, required=True)
    parser.add_argument("--expected-serving-inventory-count", type=int, required=True)
    parser.add_argument("--comparison-catalog-path", required=True)
    parser.add_argument("--expected-comparison-catalog-sha256", required=True)
    parser.add_argument("--myquant-git-repository", required=True)
    parser.add_argument("--myquant-commit", required=True)
    parser.add_argument("--myquant-source-path", required=True)
    parser.add_argument("--myquant-source-mode", required=True)
    parser.add_argument("--myquant-source-blob-oid", required=True)
    parser.add_argument("--expected-myquant-source-sha256", required=True)
    parser.add_argument("--expected-python-executable-sha256", required=True)
    parser.add_argument(
        "--expected-project-code-sha256",
        action="append",
        default=[],
        metavar="RELATIVE_PATH=SHA256",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish", help="publish the fixed diagnostic")
    _add_publish_arguments(publish)
    readback = commands.add_parser(
        "readback", help="reopen one immutable diagnostic bundle"
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
        flags = getattr(_diagnostic(), "SIDE_EFFECT_FLAGS", {"repo_write": False})
        print(
            json.dumps(
                {
                    "accepted": False,
                    "status": "REJECTED_FAIL_CLOSED",
                    "detail": str(exc),
                    "side_effects": copy.deepcopy(flags),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
