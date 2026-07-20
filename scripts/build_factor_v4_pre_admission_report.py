#!/usr/bin/env python3
"""Freeze and screen the Factor v4 candidate catalog offline.

This runner is deliberately report-only.  It has no registry, activation,
WAL, receipt, apply, provider, or LLM interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.factors.aquant_expression import (  # noqa: E402
    build_aquant_expression_inputs,
)
from quant_investor.factors.governance_pre_admission_artifact_v4 import (  # noqa: E402
    PRE_ADMISSION_REPORT_FILENAME,
    build_factor_governance_pre_admission_report_v4,
    publish_factor_governance_pre_admission_report_v4,
    validate_factor_governance_pre_admission_report_v4,
)
from quant_investor.factors.governance_screening_v4 import (  # noqa: E402
    SOURCE_BINDING_FIELDS,
    build_candidate_catalog_v4,
    build_primitive_ontology_v4,
    build_screening_evidence_v4,
    canonical_json_bytes,
    canonical_semantic_sha256,
    validate_candidate_catalog_v4,
    validate_primitive_ontology_v4,
    validate_screening_evidence_v4,
)
from quant_investor.factors.pit_fundamentals import (  # noqa: E402
    DEFAULT_FUNDAMENTAL_MART_ROOT,
)
from scripts.mine_quant_branch_factors import (  # noqa: E402
    MiningCandidate,
    _formulaic_primitives,
    build_candidate_catalog,
    candidate_maturity_context,
    candidate_primitive_lineage,
    compute_candidate_signal,
    restrict_context_to_analysis_window,
)
from scripts.retest_aquant_alpha_mix_8gate import (  # noqa: E402
    RetestContext,
    build_context,
    candidate_metrics,
)


FIXED_WINDOWS = (5, 10, 15, 20, 25, 30, 40, 60, 90, 120)
FIXED_UNIVERSES = ("full_a",)
EXPECTED_CANDIDATE_COUNT = 230
ONTOLOGY_FILENAME = "primitive_ontology.v4.json"
SCREENING_FILENAME = "screening_evidence.v4.json"
RUN_CONFIG_FILENAME = "run_config.v4.json"
MARKET_DATA_INPUT_INVENTORY_FILENAME = "market_data_input_inventory.v1.json"
REGISTRY_PATH = PROJECT_ROOT / "quant_investor/factor_registry/mined_factors.json"
STATISTIC_CONTRACT = {
    "raw_p_method": "rank_ic_normal_erfc_two_sided.v1",
    "fdr_method": "benjamini_hochberg_by_ontology_family.v1",
    "q": 0.1,
}
SOURCE_BINDING_KEYS = set(SOURCE_BINDING_FIELDS)
_BUILTIN_LINEAGE = {
    "builtin_short_term_return_20d": ("close_return",),
    "builtin_volatility_penalty_60d": ("close_return",),
}
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FactorV4PreAdmissionRunnerError(ValueError):
    """Raised when the offline runner cannot prove an exact input boundary."""


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _validate_run_id(run_id: str) -> str:
    if type(run_id) is not str or run_id != run_id.strip():
        raise FactorV4PreAdmissionRunnerError("run_id must be one safe path segment")
    value = run_id
    if not _RUN_ID_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise FactorV4PreAdmissionRunnerError("run_id must be one safe path segment")
    return value


def _validate_sha256(value: str, *, label: str) -> str:
    if type(value) is not str or not _SHA256_PATTERN.fullmatch(value):
        raise FactorV4PreAdmissionRunnerError(f"{label} must be lowercase SHA-256")
    return value


def _ensure_private_directory(path: Path, *, create: bool) -> Path:
    target = _lexical_absolute(path)
    if create and not os.path.lexists(target):
        target.mkdir(parents=True, mode=0o700)
        target.chmod(0o700)
    try:
        metadata = os.lstat(target)
    except FileNotFoundError as exc:
        raise FactorV4PreAdmissionRunnerError(
            f"private directory missing: {target}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FactorV4PreAdmissionRunnerError(
            f"private directory must be a real directory: {target}"
        )
    if metadata.st_uid != os.getuid():
        raise FactorV4PreAdmissionRunnerError(
            f"private directory owner mismatch: {target}"
        )
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise FactorV4PreAdmissionRunnerError(
            f"private directory mode must be 0700: {target}"
        )
    return target


def _run_directory(private_root: str | Path, run_id: str, *, create: bool) -> Path:
    root = _ensure_private_directory(_lexical_absolute(private_root), create=create)
    target = root / _validate_run_id(run_id)
    return _ensure_private_directory(target, create=create)


def _file_signature(metadata: os.stat_result) -> tuple[int, ...]:
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


def _stable_file_bytes(path: Path, *, private: bool) -> bytes:
    target = _lexical_absolute(path)
    descriptor: int | None = None
    try:
        before = os.lstat(target)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise FactorV4PreAdmissionRunnerError(
                f"regular non-symlink file required: {target}"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags)
        opened = os.fstat(descriptor)
        if _file_signature(before) != _file_signature(opened):
            raise FactorV4PreAdmissionRunnerError(
                f"file changed while opening: {target}"
            )
        if opened.st_uid != os.getuid():
            raise FactorV4PreAdmissionRunnerError(f"file owner mismatch: {target}")
        if int(opened.st_nlink) != 1:
            raise FactorV4PreAdmissionRunnerError(
                f"file hard-link count must be one: {target}"
            )
        if private and stat.S_IMODE(opened.st_mode) != 0o600:
            raise FactorV4PreAdmissionRunnerError(
                f"private file mode must be 0600: {target}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
        if _file_signature(opened) != _file_signature(after):
            raise FactorV4PreAdmissionRunnerError(
                f"file changed while reading: {target}"
            )
        return raw
    except OSError as exc:
        raise FactorV4PreAdmissionRunnerError(
            f"file read failed: {target}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_json_object(path: Path, *, private: bool) -> tuple[dict[str, Any], bytes]:
    raw = _stable_file_bytes(path, private=private)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactorV4PreAdmissionRunnerError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise FactorV4PreAdmissionRunnerError(f"JSON object required: {path}")
    return dict(value), raw


def _read_private_canonical_json(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    target = _lexical_absolute(path)
    _ensure_private_directory(target.parent, create=False)
    value, raw = _read_json_object(target, private=True)
    if raw != canonical_json_bytes(value):
        raise FactorV4PreAdmissionRunnerError(
            f"private JSON is not exact canonical bytes: {target}"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != _validate_sha256(
        expected_sha256, label="expected catalog SHA-256"
    ):
        raise FactorV4PreAdmissionRunnerError("candidate catalog SHA-256 mismatch")
    return value, digest


def _write_private_exact_once(path: Path, value: Mapping[str, Any]) -> str:
    target = _lexical_absolute(path)
    _ensure_private_directory(target.parent, create=False)
    raw = canonical_json_bytes(dict(value))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(target, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(descriptor)
    except FileExistsError as exc:
        raise FactorV4PreAdmissionRunnerError(
            f"exact-once output already exists: {target}"
        ) from exc
    except OSError as exc:
        raise FactorV4PreAdmissionRunnerError(
            f"exact-once output write failed: {target}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    value_readback, digest = _read_private_canonical_json(target)
    if canonical_json_bytes(value_readback) != raw:
        raise FactorV4PreAdmissionRunnerError(
            f"exact-once output readback mismatch: {target}"
        )
    return digest


def _ontology_path(catalog_path: Path) -> Path:
    return _lexical_absolute(catalog_path).with_name(ONTOLOGY_FILENAME)


def _candidate_primitives(candidate: MiningCandidate) -> list[str]:
    lineage = candidate_primitive_lineage(candidate)
    primitives = [
        str(item).strip()
        for item in list(lineage.get("primitive_lineage", []) or [])
        if str(item).strip()
    ]
    if not primitives and candidate.name in _BUILTIN_LINEAGE:
        primitives = list(_BUILTIN_LINEAGE[candidate.name])
    if not primitives:
        raise FactorV4PreAdmissionRunnerError(
            f"candidate primitive lineage missing: {candidate.name}"
        )
    return sorted(set(primitives))


def _candidate_lookback(candidate: MiningCandidate) -> int:
    values: list[int] = []
    if candidate.window is not None:
        values.append(int(candidate.window))
    for key, raw in dict(candidate.params or {}).items():
        if str(key).endswith("_window"):
            try:
                values.append(int(raw))
            except (TypeError, ValueError):
                pass
        if str(key) in {"left", "right"}:
            values.extend(int(item) for item in re.findall(r"\d+", str(raw)))
    return max([value for value in values if value > 0], default=1)


def _candidate_input_fields(primitives: Sequence[str]) -> list[str]:
    fields: set[str] = set()
    for primitive in primitives:
        if primitive in {"price_momentum", "close_return", "downside_return"}:
            fields.add("adj_close")
        elif primitive == "volume":
            fields.add("volume")
        elif primitive == "traded_amount":
            fields.add("amount")
        elif primitive == "amihud_illiquidity":
            fields.update({"adj_close", "amount"})
        elif primitive.startswith("fin_") or primitive == "fcf_to_price":
            fields.add(primitive)
        else:
            raise FactorV4PreAdmissionRunnerError(
                f"candidate primitive input mapping missing: {primitive}"
            )
    return sorted(fields)


def _catalog_definition(candidate: MiningCandidate) -> dict[str, Any]:
    primitives = _candidate_primitives(candidate)
    params = dict(candidate.params or {})
    if "_runtime_family" in params:
        raise FactorV4PreAdmissionRunnerError(
            f"candidate params use reserved _runtime_family: {candidate.name}"
        )
    # compute_price_volume_signal dispatches on MiningCandidate.family.  Keep
    # that runtime-only discriminator in the frozen definition without using
    # the legacy free-form family for ontology/BH grouping.
    params["_runtime_family"] = candidate.family
    return {
        "name": candidate.name,
        "implementation": candidate.implementation,
        "expression": candidate.expression,
        "direction": 1.0,
        "params": params,
        "lookback": _candidate_lookback(candidate),
        "slot": f"primitive:{'+'.join(primitives)}",
        "input_fields": _candidate_input_fields(primitives),
        "primitive_ids": primitives,
    }


def _build_frozen_catalog() -> tuple[
    list[MiningCandidate], dict[str, Any], dict[str, Any]
]:
    candidates = build_candidate_catalog(FIXED_WINDOWS)
    names = [candidate.name for candidate in candidates]
    if len(candidates) != EXPECTED_CANDIDATE_COUNT or len(set(names)) != len(names):
        raise FactorV4PreAdmissionRunnerError(
            "fixed candidate catalog must contain exactly 230 unique candidates"
        )
    definitions = [_catalog_definition(candidate) for candidate in candidates]
    primitive_ids = sorted(
        {
            primitive
            for definition in definitions
            for primitive in definition["primitive_ids"]
        }
    )
    ontology = build_primitive_ontology_v4(
        [
            {"primitive_id": primitive_id, "family": primitive_id}
            for primitive_id in primitive_ids
        ]
    )
    validate_primitive_ontology_v4(ontology)
    catalog = build_candidate_catalog_v4(
        ontology=ontology,
        candidates=definitions,
    )
    validate_candidate_catalog_v4(catalog, ontology=ontology)
    if len(catalog["candidates"]) != EXPECTED_CANDIDATE_COUNT:
        raise FactorV4PreAdmissionRunnerError(
            "validated catalog candidate count is not exactly 230"
        )
    return candidates, ontology, catalog


def _prepare_freeze_directory(args: argparse.Namespace, catalog_path: Path) -> None:
    private_root = getattr(args, "private_root", "")
    run_id = getattr(args, "run_id", "")
    if type(private_root) is not str or not private_root or type(run_id) is not str or not run_id:
        raise FactorV4PreAdmissionRunnerError(
            "freeze-catalog requires explicit private-root and run-id"
        )
    run_dir = _run_directory(private_root, run_id, create=True)
    if _lexical_absolute(catalog_path).parent != run_dir:
        raise FactorV4PreAdmissionRunnerError(
            "catalog path must be directly under the explicit private run directory"
        )


def freeze_catalog(args: argparse.Namespace) -> dict[str, Any]:
    catalog_path = _lexical_absolute(args.catalog_path)
    if catalog_path.name == ONTOLOGY_FILENAME:
        raise FactorV4PreAdmissionRunnerError(
            "catalog path must not collide with the ontology sidecar"
        )
    _prepare_freeze_directory(args, catalog_path)
    ontology_path = _ontology_path(catalog_path)
    if os.path.lexists(catalog_path) or os.path.lexists(ontology_path):
        raise FactorV4PreAdmissionRunnerError(
            "freeze-catalog requires both exact-once outputs to be absent"
        )
    _candidates, ontology, catalog = _build_frozen_catalog()
    ontology_file_sha256 = _write_private_exact_once(ontology_path, ontology)
    catalog_file_sha256 = _write_private_exact_once(catalog_path, catalog)
    return {
        "mode": "freeze-catalog",
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "ontology_path": str(ontology_path),
        "ontology_sha256": ontology_file_sha256,
        "ontology_semantic_sha256": ontology["semantic_sha256"],
        "catalog_path": str(catalog_path),
        "catalog_sha256": catalog_file_sha256,
        "catalog_semantic_sha256": catalog["semantic_sha256"],
        "production_apply_enabled": False,
    }


def _read_and_validate_frozen_catalog(
    catalog_path: Path,
    expected_catalog_sha256: str,
) -> tuple[list[MiningCandidate], dict[str, Any], dict[str, Any], str]:
    ontology, _ontology_file_sha = _read_private_canonical_json(
        _ontology_path(catalog_path)
    )
    validate_primitive_ontology_v4(ontology)
    catalog, catalog_file_sha = _read_private_canonical_json(
        catalog_path,
        expected_sha256=expected_catalog_sha256,
    )
    validate_candidate_catalog_v4(catalog, ontology=ontology)
    candidates, current_ontology, current_catalog = _build_frozen_catalog()
    if canonical_json_bytes(ontology) != canonical_json_bytes(current_ontology):
        raise FactorV4PreAdmissionRunnerError(
            "frozen primitive ontology no longer matches current candidate code"
        )
    if canonical_json_bytes(catalog) != canonical_json_bytes(current_catalog):
        raise FactorV4PreAdmissionRunnerError(
            "frozen candidate definitions no longer match current candidate code"
        )
    return candidates, ontology, catalog, catalog_file_sha


def _restore_exposures(
    restricted: RetestContext,
    source: RetestContext,
) -> RetestContext:
    size_by_date = source.size_bucket_by_date
    if not size_by_date.empty:
        size_by_date = size_by_date.reindex(restricted.adj_close.index)
    return replace(
        restricted,
        sector_by_symbol=dict(source.sector_by_symbol),
        size_bucket_by_symbol=dict(source.size_bucket_by_symbol),
        size_bucket_by_date=size_by_date,
        exposure_metadata=dict(source.exposure_metadata),
    )


def _analysis_context(
    full_context: RetestContext,
    *,
    analysis_start_date: str,
    min_price_coverage: float,
) -> tuple[RetestContext, str]:
    restricted, resolved_start = restrict_context_to_analysis_window(
        full_context,
        analysis_start_date=analysis_start_date,
        min_price_coverage=min_price_coverage,
    )
    return _restore_exposures(restricted, full_context), resolved_start


def _maturity_context(
    full_context: RetestContext,
    signal: Any,
    *,
    base_start: str,
    min_signal_coverage: float,
) -> RetestContext:
    restricted, _effective_start = candidate_maturity_context(
        full_context,
        signal,
        base_start=base_start,
        min_signal_coverage=min_signal_coverage,
    )
    return _restore_exposures(restricted, full_context)


def _sha256_file(path: Path, *, private: bool = False) -> str:
    return hashlib.sha256(_stable_file_bytes(path, private=private)).hexdigest()


def _resolve_declared_path(
    raw_path: Any,
    *,
    anchor: Path,
    expected: Path | None = None,
) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise FactorV4PreAdmissionRunnerError("governed source path missing")
    declared = Path(text).expanduser()
    candidates = (
        [declared]
        if declared.is_absolute()
        else [PROJECT_ROOT / declared, anchor / declared]
    )
    resolved = [_lexical_absolute(candidate) for candidate in candidates]
    if expected is not None:
        expected_path = _lexical_absolute(expected)
        if expected_path not in resolved:
            raise FactorV4PreAdmissionRunnerError(
                "governed source path does not match its canonical location"
            )
        return expected_path
    existing = [candidate for candidate in resolved if candidate.exists()]
    if len(existing) != 1:
        raise FactorV4PreAdmissionRunnerError(
            "governed source path must resolve to exactly one file"
        )
    return existing[0]


def _owned_directory_signature(path: Path, *, label: str) -> tuple[int, ...]:
    target = _lexical_absolute(path)
    try:
        metadata = os.lstat(target)
    except OSError as exc:
        raise FactorV4PreAdmissionRunnerError(
            f"{label} missing or unreadable: {target}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FactorV4PreAdmissionRunnerError(
            f"{label} must be a real directory: {target}"
        )
    if metadata.st_uid != os.getuid():
        raise FactorV4PreAdmissionRunnerError(
            f"{label} owner mismatch: {target}"
        )
    return _file_signature(metadata)


def _assert_owned_directory_chain(
    path: Path,
    *,
    boundary: Path,
    label: str,
) -> None:
    target = _lexical_absolute(path)
    root = _lexical_absolute(boundary)
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise FactorV4PreAdmissionRunnerError(
            f"{label} escapes its governed boundary: {target}"
        ) from exc
    current = root
    _owned_directory_signature(current, label=label)
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise FactorV4PreAdmissionRunnerError(
                f"{label} contains an unsafe path segment"
            )
        current = current / part
        _owned_directory_signature(current, label=label)


def _resolve_exact_declared_path(
    raw_path: Any,
    *,
    anchor: Path,
    expected: Path,
    label: str,
) -> Path:
    if type(raw_path) is not str or raw_path != raw_path.strip() or not raw_path:
        raise FactorV4PreAdmissionRunnerError(f"{label} path missing or unsafe")
    declared = Path(raw_path)
    if ".." in declared.parts:
        raise FactorV4PreAdmissionRunnerError(f"{label} path contains parent traversal")
    expected_path = _lexical_absolute(expected)
    candidates = (
        [_lexical_absolute(declared)]
        if declared.is_absolute()
        else [
            _lexical_absolute(PROJECT_ROOT / declared),
            _lexical_absolute(anchor / declared),
        ]
    )
    if expected_path not in candidates:
        raise FactorV4PreAdmissionRunnerError(
            f"{label} path does not match its canonical location"
        )
    return expected_path


def _stable_file_sha256_size(
    path: Path,
    *,
    expected_signature: tuple[int, ...] | None = None,
    allow_multiple_hard_links: bool = False,
) -> tuple[str, int, int]:
    target = _lexical_absolute(path)
    descriptor: int | None = None
    try:
        before = os.lstat(target)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise FactorV4PreAdmissionRunnerError(
                f"regular non-symlink file required: {target}"
            )
        if before.st_uid != os.getuid():
            raise FactorV4PreAdmissionRunnerError(f"file owner mismatch: {target}")
        if int(before.st_nlink) < 1 or (
            not allow_multiple_hard_links and int(before.st_nlink) != 1
        ):
            raise FactorV4PreAdmissionRunnerError(
                f"file hard-link count is unsafe: {target}"
            )
        if expected_signature is not None and _file_signature(before) != (
            expected_signature
        ):
            raise FactorV4PreAdmissionRunnerError(
                f"file changed before hashing: {target}"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags)
        opened = os.fstat(descriptor)
        if _file_signature(before) != _file_signature(opened):
            raise FactorV4PreAdmissionRunnerError(
                f"file changed while opening: {target}"
            )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        if _file_signature(opened) != _file_signature(after):
            raise FactorV4PreAdmissionRunnerError(
                f"file changed while hashing: {target}"
            )
        return digest.hexdigest(), int(after.st_size), int(after.st_nlink)
    except OSError as exc:
        raise FactorV4PreAdmissionRunnerError(
            f"file hash failed: {target}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _market_tree_state(
    root: Path,
) -> tuple[
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[tuple[Path, str, tuple[int, ...]], ...],
]:
    governed_root = _lexical_absolute(root)
    directories: list[tuple[str, tuple[int, ...]]] = []
    files: list[tuple[str, tuple[int, ...]]] = []
    parquet_files: list[tuple[Path, str, tuple[int, ...]]] = []
    pending = [governed_root]
    while pending:
        directory = pending.pop()
        relative_directory = directory.relative_to(governed_root)
        relative_directory_text = (
            "." if not relative_directory.parts else relative_directory.as_posix()
        )
        directories.append(
            (
                relative_directory_text,
                _owned_directory_signature(
                    directory,
                    label="market data inventory directory",
                ),
            )
        )
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise FactorV4PreAdmissionRunnerError(
                f"market data inventory directory unreadable: {directory}"
            ) from exc
        for entry in entries:
            target = directory / entry.name
            relative = target.relative_to(governed_root)
            if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
                raise FactorV4PreAdmissionRunnerError(
                    f"unsafe market data inventory path: {target}"
                )
            try:
                metadata = os.lstat(target)
            except OSError as exc:
                raise FactorV4PreAdmissionRunnerError(
                    f"market data inventory entry unreadable: {target}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise FactorV4PreAdmissionRunnerError(
                    f"market data inventory symlink rejected: {target}"
                )
            if metadata.st_uid != os.getuid():
                raise FactorV4PreAdmissionRunnerError(
                    f"market data inventory owner mismatch: {target}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(target)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise FactorV4PreAdmissionRunnerError(
                    f"market data inventory special file rejected: {target}"
                )
            if int(metadata.st_nlink) < 1:
                raise FactorV4PreAdmissionRunnerError(
                    f"market data inventory invalid link count: {target}"
                )
            relative_text = relative.as_posix()
            signature = _file_signature(metadata)
            files.append((relative_text, signature))
            if target.name.endswith(".parquet"):
                parquet_files.append((target, relative_text, signature))
    return (
        tuple(sorted(directories)),
        tuple(sorted(files)),
        tuple(sorted(parquet_files, key=lambda item: item[1])),
    )


def _parquet_inventory(root: Path) -> list[dict[str, Any]]:
    before_directories, before_files, parquet_files = _market_tree_state(root)
    if not parquet_files:
        raise FactorV4PreAdmissionRunnerError(
            f"market data parquet inventory is empty: {_lexical_absolute(root)}"
        )
    inventory: list[dict[str, Any]] = []
    for path, relative_path, signature in parquet_files:
        digest, size_bytes, hard_link_count = _stable_file_sha256_size(
            path,
            expected_signature=signature,
            allow_multiple_hard_links=True,
        )
        inventory.append(
            {
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "sha256": digest,
                "hard_link_count": hard_link_count,
            }
        )
    after_directories, after_files, after_parquet_files = _market_tree_state(root)
    if (
        before_directories != after_directories
        or before_files != after_files
        or tuple((path, rel, sig) for path, rel, sig in parquet_files)
        != tuple((path, rel, sig) for path, rel, sig in after_parquet_files)
    ):
        raise FactorV4PreAdmissionRunnerError(
            "market data directory or file set changed during inventory"
        )
    return inventory


def _relative_binding_record(
    path: Path,
    *,
    market_root: Path,
    raw: bytes,
) -> dict[str, Any]:
    target = _lexical_absolute(path)
    try:
        relative_path = target.relative_to(_lexical_absolute(market_root)).as_posix()
    except ValueError as exc:
        raise FactorV4PreAdmissionRunnerError(
            f"market data binding escapes canonical root: {target}"
        ) from exc
    return {
        "relative_path": relative_path,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _market_data_input_inventory(data_root: Path) -> dict[str, Any]:
    canonical_data_root = _lexical_absolute(data_root)
    market_root = canonical_data_root / "parquet" / "cn"
    _assert_owned_directory_chain(
        market_root,
        boundary=canonical_data_root,
        label="canonical CN market root",
    )
    pointer_path = market_root / "_latest.json"
    pointer, pointer_raw = _read_json_object(pointer_path, private=False)
    snapshot_id = str(pointer.get("snapshot_id") or "")
    if (
        snapshot_id != snapshot_id.strip()
        or not snapshot_id
        or Path(snapshot_id).name != snapshot_id
        or snapshot_id in {".", ".."}
        or pointer.get("status") != "OK"
        or list(pointer.get("blockers", []) or [])
    ):
        raise FactorV4PreAdmissionRunnerError("strict Parquet pointer is not healthy")

    expected_manifest = market_root / "_snapshots" / f"{snapshot_id}.json"
    manifest_path = _resolve_exact_declared_path(
        pointer.get("manifest_path"),
        anchor=market_root,
        expected=expected_manifest,
        label="snapshot manifest",
    )
    _assert_owned_directory_chain(
        manifest_path.parent,
        boundary=market_root,
        label="snapshot manifest directory",
    )
    manifest, manifest_raw = _read_json_object(manifest_path, private=False)
    if (
        manifest.get("snapshot_id") != snapshot_id
        or manifest.get("market") != "CN"
        or manifest.get("status") != "OK"
        or manifest.get("readback_validated") is not True
        or list(manifest.get("blockers", []) or [])
    ):
        raise FactorV4PreAdmissionRunnerError(
            "strict Parquet snapshot manifest is not healthy"
        )
    _resolve_exact_declared_path(
        manifest.get("manifest_path"),
        anchor=market_root,
        expected=expected_manifest,
        label="snapshot manifest self-binding",
    )

    pointer_coverage = pointer.get("coverage")
    manifest_coverage = manifest.get("coverage")
    if (
        not isinstance(pointer_coverage, Mapping)
        or not isinstance(manifest_coverage, Mapping)
        or dict(pointer_coverage) != dict(manifest_coverage)
    ):
        raise FactorV4PreAdmissionRunnerError(
            "strict Parquet pointer and manifest coverage bindings differ"
        )
    coverage = dict(manifest_coverage)
    categories = list(coverage.get("categories_checked", []) or [])
    if (
        coverage.get("coverage_schema_version") != "cn-full-a-coverage.v4"
        or coverage.get("complete") is not True
        or "full_a" not in categories
    ):
        raise FactorV4PreAdmissionRunnerError("strict full-A coverage is not complete")
    expected_scope_sha = _validate_sha256(
        str(coverage.get("expected_scope_sha256") or ""),
        label="expected scope SHA-256",
    )
    expected_scope_count = coverage.get("expected_scope_count")
    if (
        isinstance(expected_scope_count, bool)
        or not isinstance(expected_scope_count, int)
        or expected_scope_count <= 0
    ):
        raise FactorV4PreAdmissionRunnerError(
            "strict full-A expected scope count is invalid"
        )

    snapshot_payload_root = market_root / "_snapshots" / snapshot_id
    expected_table_root = snapshot_payload_root / "table" / "bars"
    expected_serving_root = snapshot_payload_root / "serving" / "bars"
    resolved_roots: dict[str, Path] = {}
    for field, expected_root in (
        ("table_root", expected_table_root),
        ("derived_serving_root", expected_serving_root),
    ):
        if pointer.get(field) != manifest.get(field):
            raise FactorV4PreAdmissionRunnerError(
                f"strict Parquet pointer and manifest {field} differ"
            )
        resolved_roots[field] = _resolve_exact_declared_path(
            pointer.get(field),
            anchor=market_root,
            expected=expected_root,
            label=field,
        )
        _assert_owned_directory_chain(
            resolved_roots[field],
            boundary=market_root,
            label=field,
        )

    pit_generation_id = str(coverage.get("pit_generation_id") or "")
    if (
        pit_generation_id != pit_generation_id.strip()
        or not pit_generation_id
        or Path(pit_generation_id).name != pit_generation_id
        or pit_generation_id in {".", ".."}
    ):
        raise FactorV4PreAdmissionRunnerError("PIT generation id is unsafe")
    pit_generation_root = (
        market_root / "reference" / "_generations" / pit_generation_id
    )
    expected_pit_path = pit_generation_root / "stock_basic_membership.parquet"
    pit_path = _resolve_exact_declared_path(
        coverage.get("pit_membership_path"),
        anchor=manifest_path.parent,
        expected=expected_pit_path,
        label="PIT membership",
    )
    expected_pit_manifest = pit_generation_root / "manifest.json"
    pit_manifest_path = _resolve_exact_declared_path(
        coverage.get("pit_generation_manifest_path"),
        anchor=manifest_path.parent,
        expected=expected_pit_manifest,
        label="PIT generation manifest",
    )
    _assert_owned_directory_chain(
        pit_generation_root,
        boundary=market_root,
        label="PIT generation root",
    )
    pit_manifest, pit_manifest_raw = _read_json_object(
        pit_manifest_path,
        private=False,
    )
    expected_pit_manifest_sha = _validate_sha256(
        str(coverage.get("pit_generation_manifest_sha256") or ""),
        label="PIT generation manifest SHA-256",
    )
    if hashlib.sha256(pit_manifest_raw).hexdigest() != expected_pit_manifest_sha:
        raise FactorV4PreAdmissionRunnerError(
            "PIT generation manifest SHA-256 mismatch"
        )
    expected_pit_sha = _validate_sha256(
        str(coverage.get("pit_membership_sha256") or ""),
        label="PIT membership SHA-256",
    )
    _resolve_exact_declared_path(
        pit_manifest.get("canonical_path"),
        anchor=pit_generation_root,
        expected=expected_pit_path,
        label="PIT manifest canonical membership",
    )
    if (
        pit_manifest.get("generation_id") != pit_generation_id
        or pit_manifest.get("canonical_sha256") != expected_pit_sha
    ):
        raise FactorV4PreAdmissionRunnerError(
            "PIT generation manifest membership binding mismatch"
        )
    pit_raw = _stable_file_bytes(pit_path, private=False)
    if hashlib.sha256(pit_raw).hexdigest() != expected_pit_sha:
        raise FactorV4PreAdmissionRunnerError("PIT membership SHA-256 mismatch")

    table_inventory = _parquet_inventory(resolved_roots["table_root"])
    serving_inventory = _parquet_inventory(
        resolved_roots["derived_serving_root"]
    )
    return {
        "schema_version": "factor-v4-market-data-input-inventory.v1",
        "snapshot_id": snapshot_id,
        "latest_pointer": _relative_binding_record(
            pointer_path,
            market_root=market_root,
            raw=pointer_raw,
        ),
        "snapshot_manifest": _relative_binding_record(
            manifest_path,
            market_root=market_root,
            raw=manifest_raw,
        ),
        "pit_generation_id": pit_generation_id,
        "pit_generation_manifest": _relative_binding_record(
            pit_manifest_path,
            market_root=market_root,
            raw=pit_manifest_raw,
        ),
        "pit_membership": _relative_binding_record(
            pit_path,
            market_root=market_root,
            raw=pit_raw,
        ),
        "expected_scope": {
            "count": expected_scope_count,
            "sha256": expected_scope_sha,
        },
        "table_root": expected_table_root.relative_to(market_root).as_posix(),
        "serving_root": expected_serving_root.relative_to(market_root).as_posix(),
        "table_parquet_inventory": table_inventory,
        "serving_parquet_inventory": serving_inventory,
    }


def _fundamental_binding_sha256(fundamental_root: Path) -> str:
    root = _lexical_absolute(fundamental_root)
    pointer_path = root / "_fundamental_latest.json"
    pointer, _raw = _read_json_object(pointer_path, private=False)
    if (
        pointer.get("schema_version") != "cn-fundamental-pointer.v1"
        or pointer.get("status") != "OK"
    ):
        raise FactorV4PreAdmissionRunnerError("fundamental pointer is not healthy")
    manifest_path = _resolve_declared_path(
        pointer.get("manifest_path"),
        anchor=root,
    )
    _manifest, manifest_raw = _read_json_object(manifest_path, private=False)
    return hashlib.sha256(manifest_raw).hexdigest()


def _code_binding_sha256() -> str:
    import quant_investor.factors.aquant_expression as aquant_expression
    import quant_investor.factors.governance_pre_admission_artifact_v4 as preadmission
    import quant_investor.factors.governance_screening_v4 as screening
    import quant_investor.factors.pit_fundamentals as pit_fundamentals
    import quant_investor.factors.price_volume as price_volume
    import quant_investor.factors.runtime as factor_runtime
    import quant_investor.market.market_data_reader as market_data_reader
    import scripts.mine_quant_branch_factors as mining
    import scripts.retest_aquant_alpha_mix_8gate as retest

    # This map binds the runner, its pure contracts, and every directly used
    # signal/input/runtime source module.  Data artifacts have separate hashes.
    paths = [
        Path(__file__),
        Path(mining.__file__),
        Path(retest.__file__),
        Path(screening.__file__),
        Path(preadmission.__file__),
        Path(aquant_expression.__file__),
        Path(pit_fundamentals.__file__),
        Path(price_volume.__file__),
        Path(factor_runtime.__file__),
        Path(market_data_reader.__file__),
    ]
    return canonical_semantic_sha256(
        {
            str(path.resolve().relative_to(PROJECT_ROOT)): _sha256_file(path)
            for path in paths
        }
    )


def _static_source_bindings(
    *,
    data_root: Path,
    fundamental_root: Path,
    run_config_sha256: str,
    market_data_input: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    inventory = (
        dict(market_data_input)
        if market_data_input is not None
        else _market_data_input_inventory(data_root)
    )
    if inventory.get("schema_version") != (
        "factor-v4-market-data-input-inventory.v1"
    ):
        raise FactorV4PreAdmissionRunnerError(
            "market data input inventory schema mismatch"
        )
    latest_pointer = inventory.get("latest_pointer")
    snapshot_manifest = inventory.get("snapshot_manifest")
    pit_membership = inventory.get("pit_membership")
    if not all(
        isinstance(item, Mapping)
        for item in (latest_pointer, snapshot_manifest, pit_membership)
    ):
        raise FactorV4PreAdmissionRunnerError(
            "market data input inventory bindings missing"
        )
    return {
        "code_sha256": _code_binding_sha256(),
        "registry_file_sha256": _sha256_file(REGISTRY_PATH),
        "latest_pointer_sha256": _validate_sha256(
            str(latest_pointer.get("sha256") or ""),
            label="latest pointer SHA-256",
        ),
        "manifest_sha256": _validate_sha256(
            str(snapshot_manifest.get("sha256") or ""),
            label="snapshot manifest SHA-256",
        ),
        "market_data_input_sha256": canonical_semantic_sha256(inventory),
        "pit_sha256": _validate_sha256(
            str(pit_membership.get("sha256") or ""),
            label="PIT membership SHA-256",
        ),
        "fundamental_manifest_sha256": _fundamental_binding_sha256(
            fundamental_root
        ),
        "run_config_sha256": run_config_sha256,
    }


def _calendar_sha256(context: RetestContext) -> str:
    dates = [
        item.strftime("%Y-%m-%d")
        for item in sorted(set(context.adj_close.index))
    ]
    if not dates:
        raise FactorV4PreAdmissionRunnerError("screening calendar is empty")
    return canonical_semantic_sha256(
        {
            "schema_version": "factor-screening-open-session-calendar.v1",
            "open_session_dates": dates,
        }
    )


def _run_config(
    args: argparse.Namespace,
    *,
    catalog_path: Path,
    catalog_file_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "factor-v4-pre-admission-run-config.v1",
        "run_id": _validate_run_id(args.run_id),
        "private_root": str(_lexical_absolute(args.private_root)),
        "catalog_path": str(_lexical_absolute(catalog_path)),
        "catalog_file_sha256": catalog_file_sha256,
        "windows": list(FIXED_WINDOWS),
        "universes": list(FIXED_UNIVERSES),
        "data_root": str(_lexical_absolute(args.data_root)),
        "fundamental_mart_root": str(
            _lexical_absolute(args.fundamental_mart_root)
        ),
        "market_data_backend": "strict_parquet",
        "legacy_fundamental_fallback_allowed": False,
        "horizon_days": int(args.horizon_days),
        "warmup_days": int(args.warmup_days),
        "analysis_start_date": str(args.analysis_start_date),
        "min_analysis_price_coverage": float(args.min_analysis_price_coverage),
        "candidate_maturity_start": bool(args.candidate_maturity_start),
        "min_candidate_signal_coverage": float(
            args.min_candidate_signal_coverage
        ),
        "decision_cost_bps": float(args.decision_cost_bps),
        "incremental_sleeve_weight": float(args.incremental_sleeve_weight),
        "statistic_contract": dict(STATISTIC_CONTRACT),
        "provider_calls_enabled": False,
        "llm_calls_enabled": False,
        "production_apply_enabled": False,
    }


def _validate_numeric_run_config(args: argparse.Namespace) -> None:
    checks = {
        "horizon_days": float(args.horizon_days),
        "warmup_days": float(args.warmup_days),
        "min_analysis_price_coverage": float(args.min_analysis_price_coverage),
        "min_candidate_signal_coverage": float(args.min_candidate_signal_coverage),
        "decision_cost_bps": float(args.decision_cost_bps),
        "incremental_sleeve_weight": float(args.incremental_sleeve_weight),
    }
    if not all(math.isfinite(value) for value in checks.values()):
        raise FactorV4PreAdmissionRunnerError("run config values must be finite")
    if int(args.horizon_days) <= 0 or int(args.warmup_days) < 0:
        raise FactorV4PreAdmissionRunnerError("invalid horizon or warmup")
    if not 0.0 < float(args.min_analysis_price_coverage) <= 1.0:
        raise FactorV4PreAdmissionRunnerError("invalid analysis price coverage")
    if not 0.0 < float(args.min_candidate_signal_coverage) <= 1.0:
        raise FactorV4PreAdmissionRunnerError("invalid candidate signal coverage")
    if float(args.decision_cost_bps) < 0.0:
        raise FactorV4PreAdmissionRunnerError("decision cost cannot be negative")
    if not 0.0 <= float(args.incremental_sleeve_weight) <= 1.0:
        raise FactorV4PreAdmissionRunnerError("invalid incremental sleeve weight")


def _validate_strict_parquet_environment() -> None:
    backend = str(
        os.environ.get("MYQUANT_MARKET_DATA_BACKEND", "parquet")
    ).strip().lower()
    mode_policy = str(
        os.environ.get("MYQUANT_MARKET_DATA_MODE_POLICY", "strict")
    ).strip().lower()
    if backend != "parquet":
        raise FactorV4PreAdmissionRunnerError(
            "MYQUANT_MARKET_DATA_BACKEND must be parquet"
        )
    if mode_policy != "strict":
        raise FactorV4PreAdmissionRunnerError(
            "MYQUANT_MARKET_DATA_MODE_POLICY must be strict"
        )


def _candidate_evaluations(
    args: argparse.Namespace,
    candidates: Sequence[MiningCandidate],
) -> tuple[list[dict[str, Any]], RetestContext]:
    full_context = build_context(
        data_root=_lexical_absolute(args.data_root),
        universes=FIXED_UNIVERSES,
        horizon_days=int(args.horizon_days),
        warmup_days=int(args.warmup_days),
        fundamental_mart_root=_lexical_absolute(args.fundamental_mart_root),
    )
    analysis_context, resolved_start = _analysis_context(
        full_context,
        analysis_start_date=str(args.analysis_start_date),
        min_price_coverage=float(args.min_analysis_price_coverage),
    )
    try:
        expression_inputs = build_aquant_expression_inputs(
            full_context.frames,
            fundamental_mart_root=_lexical_absolute(args.fundamental_mart_root),
            allow_legacy_fundamental_fallback=False,
        )
    except Exception as exc:
        raise FactorV4PreAdmissionRunnerError(
            f"Fundamental expression input setup failed: {exc}"
        ) from exc
    try:
        formulaic_primitives = _formulaic_primitives(
            full_context,
            expression_inputs,
        )
    except Exception as exc:
        raise FactorV4PreAdmissionRunnerError(
            f"formulaic primitive setup failed: {exc}"
        ) from exc
    evaluations: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            signal = compute_candidate_signal(
                candidate,
                context=full_context,
                expression_inputs=expression_inputs,
                formulaic_primitives=formulaic_primitives,
            )
            metrics_context = analysis_context
            if args.candidate_maturity_start:
                metrics_context = _maturity_context(
                    full_context,
                    signal,
                    base_start=resolved_start,
                    min_signal_coverage=float(
                        args.min_candidate_signal_coverage
                    ),
                )
            metrics = candidate_metrics(
                signal=signal,
                context=metrics_context,
                decision_cost_bps=float(args.decision_cost_bps),
                incremental_sleeve=float(args.incremental_sleeve_weight),
            )
            raw_p_value = float(metrics["rank_ic_p_value"])
            if not math.isfinite(raw_p_value) or not 0.0 <= raw_p_value <= 1.0:
                raise FactorV4PreAdmissionRunnerError(
                    "candidate raw p-value is not finite in [0,1]"
                )
            evaluations.append(
                {
                    "name": candidate.name,
                    "evaluation_status": "evaluated",
                    "raw_p_value": raw_p_value,
                    "failure_reason": None,
                }
            )
        except Exception as exc:
            message = " ".join(str(exc).strip().split())
            evaluations.append(
                {
                    "name": candidate.name,
                    "evaluation_status": "compute_failed",
                    "raw_p_value": None,
                    "failure_reason": (
                        f"{type(exc).__name__}:{message}"
                        if message
                        else type(exc).__name__
                    ),
                }
            )
    return evaluations, full_context


def screen(args: argparse.Namespace) -> dict[str, Any]:
    _validate_numeric_run_config(args)
    catalog_path = _lexical_absolute(args.catalog_path)

    # This exact readback and current-definition comparison must finish before
    # build_context or any Fundamental/Parquet loader is called.
    candidates, ontology, catalog, catalog_file_sha = (
        _read_and_validate_frozen_catalog(
            catalog_path,
            args.expected_catalog_sha256,
        )
    )
    _validate_strict_parquet_environment()
    run_id = _validate_run_id(args.run_id)
    prospective_root = _lexical_absolute(args.private_root)
    if os.path.lexists(prospective_root):
        _ensure_private_directory(prospective_root, create=False)
    prospective_run_dir = prospective_root / run_id
    if os.path.lexists(prospective_run_dir):
        _ensure_private_directory(prospective_run_dir, create=False)
        for filename in (
            SCREENING_FILENAME,
            RUN_CONFIG_FILENAME,
            MARKET_DATA_INPUT_INVENTORY_FILENAME,
            PRE_ADMISSION_REPORT_FILENAME,
        ):
            if os.path.lexists(prospective_run_dir / filename):
                raise FactorV4PreAdmissionRunnerError(
                    f"screen exact-once output already exists: {filename}"
                )
    run_config = _run_config(
        args,
        catalog_path=catalog_path,
        catalog_file_sha256=catalog_file_sha,
    )
    run_config_sha = canonical_semantic_sha256(run_config)
    market_data_input_before = _market_data_input_inventory(
        _lexical_absolute(args.data_root)
    )
    static_before = _static_source_bindings(
        data_root=_lexical_absolute(args.data_root),
        fundamental_root=_lexical_absolute(args.fundamental_mart_root),
        run_config_sha256=run_config_sha,
        market_data_input=market_data_input_before,
    )
    evaluations, full_context = _candidate_evaluations(args, candidates)
    market_data_input_after = _market_data_input_inventory(
        _lexical_absolute(args.data_root)
    )
    static_after = _static_source_bindings(
        data_root=_lexical_absolute(args.data_root),
        fundamental_root=_lexical_absolute(args.fundamental_mart_root),
        run_config_sha256=run_config_sha,
        market_data_input=market_data_input_after,
    )
    if (
        market_data_input_before != market_data_input_after
        or static_before != static_after
    ):
        raise FactorV4PreAdmissionRunnerError(
            "screening source bindings changed during recomputation"
        )
    source_bindings = {
        **static_after,
        "calendar_sha256": _calendar_sha256(full_context),
    }
    if set(source_bindings) != SOURCE_BINDING_KEYS:
        raise FactorV4PreAdmissionRunnerError("screening source bindings incomplete")
    evidence = build_screening_evidence_v4(
        ontology=ontology,
        catalog=catalog,
        evaluations=evaluations,
        source_bindings=source_bindings,
        statistic_contract=STATISTIC_CONTRACT,
    )
    validate_screening_evidence_v4(
        evidence,
        ontology=ontology,
        catalog=catalog,
    )
    run_dir = _run_directory(args.private_root, run_id, create=True)
    screening_path = run_dir / SCREENING_FILENAME
    run_config_path = run_dir / RUN_CONFIG_FILENAME
    market_data_input_path = run_dir / MARKET_DATA_INPUT_INVENTORY_FILENAME
    if any(
        os.path.lexists(path)
        for path in (screening_path, run_config_path, market_data_input_path)
    ):
        raise FactorV4PreAdmissionRunnerError(
            "screen requires absent exact-once inventory, screening, and run-config outputs"
        )
    market_data_input_file_sha = _write_private_exact_once(
        market_data_input_path,
        market_data_input_after,
    )
    if market_data_input_file_sha != static_after["market_data_input_sha256"]:
        raise FactorV4PreAdmissionRunnerError(
            "market data input inventory write binding mismatch"
        )
    run_config_file_sha = _write_private_exact_once(run_config_path, run_config)
    screening_file_sha = _write_private_exact_once(screening_path, evidence)
    screening_readback, _readback_sha = _read_private_canonical_json(
        screening_path,
        expected_sha256=screening_file_sha,
    )
    validate_screening_evidence_v4(
        screening_readback,
        ontology=ontology,
        catalog=catalog,
    )

    rows = list(screening_readback["rows"])
    failed_count = sum(
        1 for row in rows if row["evaluation_status"] == "compute_failed"
    )
    screening_summary = {
        "schema_version": "factor-governance-screening-summary.v4",
        "evidence_class": "diagnostic_report_only",
        "screening_evidence_sha256": screening_readback["semantic_sha256"],
        "candidate_count": len(rows),
        "evaluated_count": len(rows) - failed_count,
        "bh_pass_count": sum(1 for row in rows if row["bh_pass"] is True),
        "compute_failed_count": failed_count,
    }
    screening_summary_sha = canonical_semantic_sha256(screening_summary)
    report = build_factor_governance_pre_admission_report_v4(
        run_id=run_id,
        screening_summary=screening_summary,
        screening_sha256=screening_summary_sha,
        codex_s1_status=None,
        codex_ic_status=None,
        replay_status=None,
    )
    validate_factor_governance_pre_admission_report_v4(report)
    published = publish_factor_governance_pre_admission_report_v4(
        private_root=_lexical_absolute(args.private_root),
        run_id=run_id,
        expected_report_sha256="empty",
        report=report,
    )
    return {
        "mode": "screen",
        "run_id": run_id,
        "run_directory": str(run_dir),
        "candidate_count": len(rows),
        "evaluated_count": len(rows) - failed_count,
        "compute_failed_count": failed_count,
        "bh_pass_count": screening_summary["bh_pass_count"],
        "catalog_path": str(catalog_path),
        "catalog_sha256": catalog_file_sha,
        "market_data_input_path": str(market_data_input_path),
        "market_data_input_sha256": market_data_input_file_sha,
        "run_config_path": str(run_config_path),
        "run_config_sha256": run_config_file_sha,
        "screening_path": str(screening_path),
        "screening_sha256": screening_file_sha,
        "screening_semantic_sha256": screening_readback["semantic_sha256"],
        "pre_admission_path": str(published["path"]),
        "pre_admission_sha256": str(published["sha256"]),
        "pre_admission_status": report["status"],
        "proposals": list(report["proposals"]),
        "production_apply_enabled": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    freeze_parser = subparsers.add_parser(
        "freeze-catalog",
        help="Freeze the exact 230-candidate catalog before any data load.",
    )
    freeze_parser.add_argument("--catalog-path", required=True)
    freeze_parser.add_argument("--private-root", required=True)
    freeze_parser.add_argument("--run-id", required=True)

    screen_parser = subparsers.add_parser(
        "screen",
        help="Recompute raw p-values and v4 BH evidence from a frozen catalog.",
    )
    screen_parser.add_argument("--catalog-path", required=True)
    screen_parser.add_argument("--expected-catalog-sha256", required=True)
    screen_parser.add_argument("--private-root", required=True)
    screen_parser.add_argument("--run-id", required=True)
    screen_parser.add_argument("--data-root", default="data")
    screen_parser.add_argument(
        "--fundamental-mart-root",
        default=str(DEFAULT_FUNDAMENTAL_MART_ROOT),
    )
    screen_parser.add_argument("--horizon-days", type=int, default=30)
    screen_parser.add_argument("--warmup-days", type=int, default=260)
    screen_parser.add_argument("--analysis-start-date", default="auto")
    screen_parser.add_argument(
        "--min-analysis-price-coverage", type=float, default=0.95
    )
    screen_parser.add_argument(
        "--candidate-maturity-start", action="store_true", default=True
    )
    screen_parser.add_argument(
        "--no-candidate-maturity-start",
        dest="candidate_maturity_start",
        action="store_false",
    )
    screen_parser.add_argument(
        "--min-candidate-signal-coverage", type=float, default=0.60
    )
    screen_parser.add_argument("--decision-cost-bps", type=float, default=1.0)
    screen_parser.add_argument(
        "--incremental-sleeve-weight", type=float, default=0.03
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = freeze_catalog(args) if args.mode == "freeze-catalog" else screen(args)
    except Exception as exc:
        print(f"factor_v4_pre_admission_error={exc}", file=sys.stderr)
        print("production_apply_enabled=false", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
