#!/usr/bin/env python3
"""Publish one private v4.1 classification-only formal catalog bundle.

All source, code, and protected-control identities are explicit.  The command
does not load market data, evaluate a signal, run statistics, create a proposal,
touch the registry, or expose replay, transaction, portfolio, broker, order, or
trade surfaces.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.factors import governance_discovery_readback_v4_1 as discovery_io  # noqa: E402
from quant_investor.factors import governance_discovery_v4_1 as discovery  # noqa: E402
from quant_investor.factors import governance_formal_catalog_adapter_v4_1 as adapter  # noqa: E402
from quant_investor.factors import governance_formal_catalog_bundle_v4_1 as bundle  # noqa: E402
from quant_investor.factors import governance_formal_catalog_materialization_v4_1 as materialization  # noqa: E402
from quant_investor.factors import governance_private_bundle_io as private_io  # noqa: E402
from quant_investor.factors import governance_screening_v4 as screening  # noqa: E402


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")
_MAX_BOUND_FILE_BYTES = 64 * 1024 * 1024

REQUIRED_CODE_PATHS = tuple(
    PROJECT_ROOT / suffix
    for suffix in (
        "quant_investor/factors/governance_formal_catalog_materialization_v4_1.py",
        "quant_investor/factors/governance_formal_catalog_adapter_v4_1.py",
        "quant_investor/factors/governance_private_bundle_io.py",
        "quant_investor/factors/governance_formal_catalog_bundle_v4_1.py",
        "quant_investor/factors/governance_discovery_v4_1.py",
        "quant_investor/factors/governance_discovery_readback_v4_1.py",
        "quant_investor/factors/governance_screening_v4.py",
        "quant_investor/factors/governance_cycle_state_v4_1.py",
        "quant_investor/factors/governance_source_v4_1.py",
        "quant_investor/factors/governance_source_readback_v4_1.py",
        "scripts/build_factor_v4_1_formal_catalog.py",
    )
)
REQUIRED_PROTECTED_PATHS = tuple(
    PROJECT_ROOT / suffix
    for suffix in (
        "quant_investor/factor_registry/mined_factors.json",
        "data/parquet/cn/_latest.json",
        "data/parquet/cn/_catalog.json",
        "data/parquet/cn/_fundamental_latest.json",
        "data/parquet/cn/latest_manifest.json",
    )
)

_DISCOVERY_SOURCE_BINDING_IDS = {
    discovery.AQUANT_SOURCE_RECEIPT_FILENAME: (
        f"discovery:{discovery.AQUANT_SOURCE_RECEIPT_FILENAME}"
    ),
    discovery.SOURCE_IDEA_AUDIT_FILENAME: "source_idea_audit",
    discovery.LOCAL_COMPATIBILITY_CONTRACT_FILENAME: (
        f"discovery:{discovery.LOCAL_COMPATIBILITY_CONTRACT_FILENAME}"
    ),
    discovery.DISCOVERY_CATALOG_FILENAME: "discovery_catalog",
    discovery.STRUCTURAL_COLLISION_AUDIT_FILENAME: (
        f"discovery:{discovery.STRUCTURAL_COLLISION_AUDIT_FILENAME}"
    ),
    discovery.DISCOVERY_SOURCE_NODE_FILENAME: (
        f"discovery:{discovery.DISCOVERY_SOURCE_NODE_FILENAME}"
    ),
    discovery.DISCOVERY_CYCLE_STATE_FILENAME: (
        f"discovery:{discovery.DISCOVERY_CYCLE_STATE_FILENAME}"
    ),
    discovery.DISCOVERY_READBACK_REPORT_FILENAME: (
        f"discovery:{discovery.DISCOVERY_READBACK_REPORT_FILENAME}"
    ),
}


class FactorV4_1FormalCatalogRunnerError(ValueError):
    """Raised when a formal-catalog research publication must fail closed."""


@dataclass(frozen=True)
class BoundInputs:
    base_ontology: dict[str, Any]
    base_catalog: dict[str, Any]
    discovery_values: dict[str, dict[str, Any]]
    discovery_bundle_path: str
    discovery_artifact_descriptors: dict[str, dict[str, Any]]
    source_bindings: list[dict[str, str]]
    code_bindings: list[dict[str, Any]]
    protected_bindings: dict[str, str]


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise FactorV4_1FormalCatalogRunnerError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise FactorV4_1FormalCatalogRunnerError(
            f"{label} must be an exact lowercase SHA-256"
        )
    return value


def _safe_id(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or _SAFE_ID_RE.fullmatch(value) is None
        or ".." in value
    ):
        raise FactorV4_1FormalCatalogRunnerError(
            f"{label} must be one safe non-empty path segment"
        )
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if type(value) is not str or not value.startswith("/") or "\x00" in value:
        raise FactorV4_1FormalCatalogRunnerError(
            f"{label} must be an absolute normalized path"
        )
    if os.path.abspath(value) != value or any(
        part in {"", ".", ".."} for part in value.split("/")[1:]
    ):
        raise FactorV4_1FormalCatalogRunnerError(
            f"{label} must not contain aliases or traversal"
        )
    return Path(value)


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_parent(path: Path) -> int:
    descriptor = os.open("/", _directory_flags())
    try:
        for component in path.parent.parts[1:]:
            child = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        opened = os.fstat(descriptor)
        current = os.lstat(path.parent)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (int(opened.st_dev), int(opened.st_ino))
            != (int(current.st_dev), int(current.st_ino))
        ):
            raise FactorV4_1FormalCatalogRunnerError(
                f"bound-file parent identity mismatch: {path.parent}"
            )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_once(parent_fd: int, filename: str) -> tuple[bytes, tuple[int, ...]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(filename, flags, dir_fd=parent_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FactorV4_1FormalCatalogRunnerError(
                f"bound file must be regular: {filename}"
            )
        if int(before.st_uid) != os.getuid() or int(before.st_nlink) != 1:
            raise FactorV4_1FormalCatalogRunnerError(
                f"bound file owner/link invariant failed: {filename}"
            )
        if int(before.st_size) <= 0 or int(before.st_size) > _MAX_BOUND_FILE_BYTES:
            raise FactorV4_1FormalCatalogRunnerError(
                f"bound file size is outside limits: {filename}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_BOUND_FILE_BYTES - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_BOUND_FILE_BYTES:
                raise FactorV4_1FormalCatalogRunnerError(
                    f"bound file exceeded size limit: {filename}"
                )
        after = os.fstat(descriptor)
        identity = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_mode),
            int(after.st_uid),
            int(after.st_gid),
            int(after.st_nlink),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
        )
        before_identity = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_mode),
            int(before.st_uid),
            int(before.st_gid),
            int(before.st_nlink),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
        )
        if identity != before_identity or total != int(after.st_size):
            raise FactorV4_1FormalCatalogRunnerError(
                f"bound file changed during read: {filename}"
            )
        return b"".join(chunks), identity
    finally:
        os.close(descriptor)


def _assert_absolute_binding_current(
    *,
    path: Path,
    parent_identity: tuple[int, int],
    leaf_identity: tuple[int, ...],
) -> None:
    """Re-prove that the absolute parent path and leaf still name the read objects."""

    current_parent_fd = _open_parent(path)
    try:
        opened_parent = os.fstat(current_parent_fd)
        opened_parent_identity = (
            int(opened_parent.st_dev),
            int(opened_parent.st_ino),
        )
        if opened_parent_identity != parent_identity:
            raise FactorV4_1FormalCatalogRunnerError(
                f"absolute bound-file parent identity changed: {path.parent}"
            )

        first_leaf = os.stat(
            path.name,
            dir_fd=current_parent_fd,
            follow_symlinks=False,
        )
        first_leaf_identity = (
            int(first_leaf.st_dev),
            int(first_leaf.st_ino),
            int(first_leaf.st_mode),
            int(first_leaf.st_uid),
            int(first_leaf.st_gid),
            int(first_leaf.st_nlink),
            int(first_leaf.st_size),
            int(first_leaf.st_mtime_ns),
            int(first_leaf.st_ctime_ns),
        )
        if not stat.S_ISREG(first_leaf.st_mode) or first_leaf_identity != leaf_identity:
            raise FactorV4_1FormalCatalogRunnerError(
                f"absolute bound-file leaf identity changed: {path}"
            )

        current_parent = os.lstat(path.parent)
        if (
            stat.S_ISLNK(current_parent.st_mode)
            or (int(current_parent.st_dev), int(current_parent.st_ino))
            != opened_parent_identity
        ):
            raise FactorV4_1FormalCatalogRunnerError(
                f"absolute bound-file parent path changed: {path.parent}"
            )
        second_leaf = os.stat(
            path.name,
            dir_fd=current_parent_fd,
            follow_symlinks=False,
        )
        second_leaf_identity = (
            int(second_leaf.st_dev),
            int(second_leaf.st_ino),
            int(second_leaf.st_mode),
            int(second_leaf.st_uid),
            int(second_leaf.st_gid),
            int(second_leaf.st_nlink),
            int(second_leaf.st_size),
            int(second_leaf.st_mtime_ns),
            int(second_leaf.st_ctime_ns),
        )
        final_parent = os.lstat(path.parent)
        if (
            second_leaf_identity != leaf_identity
            or stat.S_ISLNK(final_parent.st_mode)
            or (int(final_parent.st_dev), int(final_parent.st_ino))
            != opened_parent_identity
        ):
            raise FactorV4_1FormalCatalogRunnerError(
                f"absolute bound-file path changed during final identity proof: {path}"
            )
    finally:
        os.close(current_parent_fd)


def _stable_read_bound_file(path: Path, expected_sha256: str) -> bytes:
    expected = _sha(expected_sha256, f"expected SHA for {path}")
    parent_fd = _open_parent(path)
    try:
        opened_parent = os.fstat(parent_fd)
        parent_identity = (
            int(opened_parent.st_dev),
            int(opened_parent.st_ino),
        )
        first, first_identity = _read_once(parent_fd, path.name)
        second, second_identity = _read_once(parent_fd, path.name)
        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        current_identity = (
            int(current.st_dev),
            int(current.st_ino),
            int(current.st_mode),
            int(current.st_uid),
            int(current.st_gid),
            int(current.st_nlink),
            int(current.st_size),
            int(current.st_mtime_ns),
            int(current.st_ctime_ns),
        )
        if first != second or first_identity != second_identity or second_identity != current_identity:
            raise FactorV4_1FormalCatalogRunnerError(
                f"bound file changed across stable readback passes: {path}"
            )
        actual = hashlib.sha256(first).hexdigest()
        if actual != expected:
            raise FactorV4_1FormalCatalogRunnerError(
                f"bound file SHA mismatch: {path}"
            )
        _assert_absolute_binding_current(
            path=path,
            parent_identity=parent_identity,
            leaf_identity=second_identity,
        )
        return first
    finally:
        os.close(parent_fd)


def _parse_expected_bindings(
    raw_values: Sequence[str],
    *,
    expected_paths: Sequence[Path],
    label: str,
) -> dict[Path, str]:
    expected_set = set(expected_paths)
    parsed: dict[Path, str] = {}
    for index, raw in enumerate(raw_values):
        if type(raw) is not str or "=" not in raw:
            raise FactorV4_1FormalCatalogRunnerError(
                f"{label}[{index}] must be ABSOLUTE_PATH=SHA256"
            )
        path_raw, sha_raw = raw.rsplit("=", 1)
        path = _absolute_path(path_raw, f"{label}[{index}] path")
        if path not in expected_set or path in parsed:
            raise FactorV4_1FormalCatalogRunnerError(
                f"{label} path is unexpected or duplicated: {path}"
            )
        parsed[path] = _sha(sha_raw, f"{label}[{index}] SHA")
    if set(parsed) != expected_set:
        missing = sorted(str(path) for path in expected_set - set(parsed))
        extra = sorted(str(path) for path in set(parsed) - expected_set)
        raise FactorV4_1FormalCatalogRunnerError(
            f"{label} inventory mismatch: missing={missing};extra={extra}"
        )
    return parsed


def _bind_code(args: argparse.Namespace) -> list[dict[str, Any]]:
    expected = _parse_expected_bindings(
        args.code_binding,
        expected_paths=REQUIRED_CODE_PATHS,
        label="code_binding",
    )
    rows = []
    for path in sorted(expected, key=str):
        raw = _stable_read_bound_file(path, expected[path])
        rows.append(
            {
                "absolute_path": str(path),
                "raw_sha256": expected[path],
                "size_bytes": len(raw),
            }
        )
    expected_suffixes = set(materialization.REQUIRED_CODE_BINDING_SUFFIXES)
    actual_suffixes = {
        next(
            suffix
            for suffix in expected_suffixes
            if str(row["absolute_path"]).endswith(suffix)
        )
        for row in rows
    }
    if actual_suffixes != expected_suffixes:
        raise FactorV4_1FormalCatalogRunnerError(
            "code bindings do not match the materializer allowlist"
        )
    return rows


def _bind_protected(args: argparse.Namespace) -> dict[str, str]:
    expected = _parse_expected_bindings(
        args.protected_binding,
        expected_paths=REQUIRED_PROTECTED_PATHS,
        label="protected_binding",
    )
    for path in sorted(expected, key=str):
        _stable_read_bound_file(path, expected[path])
    return {str(path): expected[path] for path in sorted(expected, key=str)}


def _bind_base_json(
    *,
    path_value: str,
    expected_byte_sha256: str,
    expected_semantic_sha256: str,
    validator: Any,
    label: str,
) -> dict[str, Any]:
    path = _absolute_path(path_value, f"{label} path")
    readback = private_io.read_private_canonical_json(
        path,
        _sha(expected_byte_sha256, f"{label} byte SHA"),
        validator,
        canonicalizer=materialization.canonical_json_bytes_v4_1,
    )
    value = dict(readback["value"])
    expected_semantic = _sha(
        expected_semantic_sha256,
        f"{label} semantic SHA",
    )
    if value.get("semantic_sha256") != expected_semantic:
        raise FactorV4_1FormalCatalogRunnerError(
            f"{label} semantic SHA mismatch"
        )
    return {"value": value, "descriptor": dict(readback["descriptor"])}


def _bind_inputs(args: argparse.Namespace) -> BoundInputs:
    _safe_id(args.run_id, "run_id")
    cycle_id = _safe_id(args.cycle_id, "cycle_id")
    ontology_bound = _bind_base_json(
        path_value=args.base_ontology_path,
        expected_byte_sha256=args.expected_base_ontology_sha256,
        expected_semantic_sha256=args.expected_base_ontology_semantic_sha256,
        validator=screening.validate_primitive_ontology_v4,
        label="base ontology",
    )
    catalog_bound = _bind_base_json(
        path_value=args.base_catalog_path,
        expected_byte_sha256=args.expected_base_catalog_sha256,
        expected_semantic_sha256=args.expected_base_catalog_semantic_sha256,
        validator=lambda value: screening.validate_candidate_catalog_v4(
            value,
            ontology=ontology_bound["value"],
        ),
        label="base catalog",
    )
    discovery_path = _absolute_path(
        args.discovery_bundle_path,
        "discovery bundle path",
    )
    discovery_result = discovery_io.readback_discovery_bundle_values_v4_1(
        discovery_path,
        base_ontology=ontology_bound["value"],
        base_catalog=catalog_bound["value"],
    )
    if (
        discovery_result.get("accepted") is not True
        or discovery_result.get("readiness") != "EXPLORATORY_DISCOVERY"
        or discovery_result.get("qualification") is not False
        or discovery_result.get("formal_admission_authority") is not False
    ):
        raise FactorV4_1FormalCatalogRunnerError(
            "discovery bundle is not the exact accepted non-authoritative state"
        )
    values = {
        filename: dict(value)
        for filename, value in discovery_result["values"].items()
    }
    descriptors = {
        filename: dict(value)
        for filename, value in discovery_result[
            "artifact_descriptors"
        ].items()
    }
    report_value = values[discovery.DISCOVERY_READBACK_REPORT_FILENAME]
    report_descriptor = descriptors[discovery.DISCOVERY_READBACK_REPORT_FILENAME]
    if report_descriptor.get("byte_sha256") != _sha(
        args.expected_discovery_readback_report_sha256,
        "discovery readback report byte SHA",
    ):
        raise FactorV4_1FormalCatalogRunnerError(
            "discovery readback report byte SHA mismatch"
        )
    if report_value.get("report_semantic_sha256") != _sha(
        args.expected_discovery_readback_report_semantic_sha256,
        "discovery readback report semantic SHA",
    ):
        raise FactorV4_1FormalCatalogRunnerError(
            "discovery readback report semantic SHA mismatch"
        )
    if report_value.get("cycle_id") != cycle_id:
        raise FactorV4_1FormalCatalogRunnerError(
            "discovery bundle cycle_id mismatch"
        )

    source_bindings = [
        {
            "binding_id": "base_ontology",
            "byte_sha256": ontology_bound["descriptor"]["byte_sha256"],
            "semantic_sha256": ontology_bound["value"]["semantic_sha256"],
        },
        {
            "binding_id": "base_catalog",
            "byte_sha256": catalog_bound["descriptor"]["byte_sha256"],
            "semantic_sha256": catalog_bound["value"]["semantic_sha256"],
        },
    ]
    for filename in discovery.CANONICAL_ARTIFACT_FILENAMES:
        descriptor = descriptors[filename]
        source_bindings.append(
            {
                "binding_id": _DISCOVERY_SOURCE_BINDING_IDS[filename],
                "byte_sha256": descriptor["byte_sha256"],
                "semantic_sha256": descriptor["semantic_sha256"],
            }
        )
    source_bindings.sort(key=lambda row: row["binding_id"])
    if tuple(row["binding_id"] for row in source_bindings) != tuple(
        materialization.REQUIRED_SOURCE_BINDING_IDS
    ):
        raise FactorV4_1FormalCatalogRunnerError(
            "source bindings do not match the exact materializer inventory"
        )
    return BoundInputs(
        base_ontology=ontology_bound["value"],
        base_catalog=catalog_bound["value"],
        discovery_values=values,
        discovery_bundle_path=str(discovery_path),
        discovery_artifact_descriptors=descriptors,
        source_bindings=source_bindings,
        code_bindings=_bind_code(args),
        protected_bindings=_bind_protected(args),
    )


def _build_artifacts(bound: BoundInputs) -> dict[str, dict[str, Any]]:
    draft = materialization.build_formal_catalog_materialization_v4_1(
        discovery_values=bound.discovery_values,
        base_ontology=bound.base_ontology,
        base_catalog=bound.base_catalog,
        source_bindings=bound.source_bindings,
        code_bindings=bound.code_bindings,
        adapter_validation=None,
    )
    adapter_validation = adapter.build_formal_catalog_adapter_validation_v4_1(
        base_ontology=bound.base_ontology,
        base_catalog=bound.base_catalog,
        ontology=draft[materialization.FORMAL_ONTOLOGY_FILENAME],
        catalog=draft[materialization.FORMAL_CATALOG_FILENAME],
        mapping_proof=draft[materialization.PRIMITIVE_MAPPING_PROOF_FILENAME],
    )
    final = materialization.build_formal_catalog_materialization_v4_1(
        discovery_values=bound.discovery_values,
        base_ontology=bound.base_ontology,
        base_catalog=bound.base_catalog,
        source_bindings=bound.source_bindings,
        code_bindings=bound.code_bindings,
        adapter_validation=adapter_validation,
    )
    for filename in materialization.FORMAL_CATALOG_MATERIALIZATION_FILENAMES[:-1]:
        if materialization.canonical_json_bytes_v4_1(
            draft[filename]
        ) != materialization.canonical_json_bytes_v4_1(final[filename]):
            raise FactorV4_1FormalCatalogRunnerError(
                f"adapter binding changed a core materialization artifact: {filename}"
            )
    normalized = materialization.validate_formal_catalog_materialization_v4_1(
        final,
        discovery_values=bound.discovery_values,
        base_ontology=bound.base_ontology,
        base_catalog=bound.base_catalog,
        source_bindings=bound.source_bindings,
        code_bindings=bound.code_bindings,
        adapter_validation=adapter_validation,
    )
    normalized_adapter = adapter.validate_formal_catalog_adapter_validation_v4_1(
        adapter_validation,
        base_ontology=bound.base_ontology,
        base_catalog=bound.base_catalog,
        ontology=normalized[materialization.FORMAL_ONTOLOGY_FILENAME],
        catalog=normalized[materialization.FORMAL_CATALOG_FILENAME],
        mapping_proof=normalized[materialization.PRIMITIVE_MAPPING_PROOF_FILENAME],
    )
    return {
        **normalized,
        adapter.FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME: normalized_adapter,
    }


def _artifact_set_sha256(artifacts: Mapping[str, Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            [
                {
                    "filename": filename,
                    "byte_sha256": hashlib.sha256(
                        bundle.canonical_file_bytes_v4_1(artifacts[filename])
                    ).hexdigest(),
                }
                for filename in bundle.FORMAL_CATALOG_INPUT_FILENAMES
            ]
        )
    ).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    initial = _bind_inputs(args)
    artifacts = _build_artifacts(initial)
    artifact_set_sha256 = _artifact_set_sha256(artifacts)
    contract = bundle.build_formal_catalog_bundle_contract_v4_1(
        expected_artifacts=artifacts,
        discovery_values=initial.discovery_values,
        base_ontology=initial.base_ontology,
        base_catalog=initial.base_catalog,
        source_bindings=initial.source_bindings,
        code_bindings=initial.code_bindings,
        protected_bindings=initial.protected_bindings,
    )

    def revalidate_inputs() -> None:
        rebound = _bind_inputs(args)
        rebuilt = _build_artifacts(rebound)
        if _artifact_set_sha256(rebuilt) != artifact_set_sha256:
            raise FactorV4_1FormalCatalogRunnerError(
                "rebound formal artifacts differ before commit"
            )
        if _canonical_json_bytes(rebuilt) != _canonical_json_bytes(artifacts):
            raise FactorV4_1FormalCatalogRunnerError(
                "rebound formal artifact bytes differ before commit"
            )

    published = private_io.publish_private_bundle(
        private_root=_absolute_path(args.private_root, "private_root"),
        run_id=_safe_id(args.run_id, "run_id"),
        artifacts=artifacts,
        contract=contract,
        revalidate_inputs=revalidate_inputs,
    )
    after_protected = _bind_protected(args)
    if after_protected != initial.protected_bindings:
        raise FactorV4_1FormalCatalogRunnerError(
            "protected controls drifted after immutable private publication at "
            f"{published['bundle_path']}; bundle acceptance proves only the "
            "persisted build-and-precommit bindings, not postcommit stability"
        )
    report = dict(published["readback_report"])
    if (
        published.get("accepted") is not True
        or report.get("readiness") != bundle.READINESS
        or report.get("qualification") is not False
        or report.get("formal_admission_authority") is not False
        or report.get("production_apply_enabled") is not False
    ):
        raise FactorV4_1FormalCatalogRunnerError(
            "live formal bundle readback contradicted the research-only contract"
        )
    descriptors = {
        filename: dict(value)
        for filename, value in published["artifact_descriptors"].items()
    }
    return {
        "accepted": True,
        "readiness": report["readiness"],
        "lifecycle_state": report["lifecycle_state"],
        "qualification": False,
        "formal_admission_authority": False,
        "production_apply_enabled": False,
        "new_risk_authorized": False,
        "source_authenticity_recomputed_by_materializer": report[
            "source_authenticity_recomputed_by_materializer"
        ],
        "adapter_source_authenticity_recomputed": report[
            "adapter_source_authenticity_recomputed"
        ],
        "bundle_path": published["bundle_path"],
        "artifact_set_semantic_sha256": artifact_set_sha256,
        "artifact_descriptors": descriptors,
        "readback_report_semantic_sha256": report["report_semantic_sha256"],
        "source_accounting": report["source_accounting"],
        "catalog_accounting": report["catalog_accounting"],
        "ontology_accounting": report["ontology_accounting"],
        "measurement_status": report["measurement_status"],
        "blockers": report["blockers"],
        "side_effects": report["side_effects"],
        "discovery_bundle_path": initial.discovery_bundle_path,
        "discovery_readback_report_sha256": args.expected_discovery_readback_report_sha256,
        "base_ontology_path": args.base_ontology_path,
        "base_catalog_path": args.base_catalog_path,
        "protected_bindings_before": initial.protected_bindings,
        "protected_bindings_after": after_protected,
        "v4_replay_path": None,
        "v4_replay_sha256": None,
        "transaction_plan_path": None,
        "transaction_plan_sha256": None,
        "research_head_created": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one private, classification-only Factor v4.1 formal catalog"
        )
    )
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--discovery-bundle-path", required=True)
    parser.add_argument(
        "--expected-discovery-readback-report-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-discovery-readback-report-semantic-sha256",
        required=True,
    )
    parser.add_argument("--base-ontology-path", required=True)
    parser.add_argument("--expected-base-ontology-sha256", required=True)
    parser.add_argument(
        "--expected-base-ontology-semantic-sha256",
        required=True,
    )
    parser.add_argument("--base-catalog-path", required=True)
    parser.add_argument("--expected-base-catalog-sha256", required=True)
    parser.add_argument(
        "--expected-base-catalog-semantic-sha256",
        required=True,
    )
    parser.add_argument(
        "--code-binding",
        action="append",
        required=True,
        metavar="ABSOLUTE_PATH=SHA256",
    )
    parser.add_argument(
        "--protected-binding",
        action="append",
        required=True,
        metavar="ABSOLUTE_PATH=SHA256",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "readiness": "BLOCKED_FAIL_CLOSED",
                    "qualification": False,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
