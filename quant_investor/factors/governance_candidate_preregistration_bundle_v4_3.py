"""Owner-private exact-once bundle for Factor v4.3 preregistration.

The v4.3 evidence contract is independent from v4.2.  This module binds the
fixed 2026-07-17 strict-full-A source, exact code and comparison identities,
and the pure PRECOMMITTED -> DISCOVERY graph supplied by
``governance_candidate_preregistration_v4_3``.  Filesystem publication is a
single owner-private ``renameatx_np(RENAME_EXCL)`` transaction delegated to
``governance_private_bundle_io``.

The readback report written inside staging is intentionally incapable of
claiming publication success.  Only the live return from this module after the
exclusive rename, parent-directory fsync, and canonical reopen may report a
``COMMITTED`` publication phase.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from quant_investor.factors import governance_candidate_preregistration_v4_3 as prereg
from quant_investor.factors.governance_cycle_state_v4_1 import (
    DISCOVERY,
    PRECOMMITTED,
    build_genesis_cycle_state_v4_1,
    byte_sha256 as cycle_state_byte_sha256_v4_1,
    validate_cycle_state_v4_1,
)
from quant_investor.factors.governance_private_bundle_io import (
    PrivateBundleContract,
    publish_private_bundle,
    readback_private_bundle,
)
from quant_investor.factors.governance_source_readback_v4_1 import (
    BoundCutoffInputsV4_1,
    INPUT_BINDING_SCHEMA_VERSION,
    SOURCE_USE_PROHIBITED,
    binding_semantic_sha256_v4_1,
)


ROOT_SUFFIX_V4_3 = (
    "reports",
    "factor_governance",
    "private",
    "v4_3_candidate_preregistration",
)
CYCLE_ID_V4_3 = "cn_full_a_v4_3_20260717_20260717T172132Z"
SNAPSHOT_ID_V4_3 = "20260717T172132Z"
CUTOFF_V4_3 = "2026-07-17"
COMPACT_CUTOFF_V4_3 = "20260717"

AQUANT_IDEA_SOURCE_SET_RECEIPT_FILENAME_V4_3 = (
    "aquant_idea_source_set_receipt.v4_3.json"
)
OPERATOR_SEMANTICS_FILENAME_V4_3 = "operator_semantics.v4_3.json"
COMPARISON_CATALOG_RECEIPT_FILENAME_V4_3 = (
    "comparison_catalog_receipt.v4_3.json"
)
CANDIDATE_SELECTION_SPEC_FILENAME_V4_3 = "candidate_selection_spec.v4_3.json"
STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_3 = (
    "strict_full_a_source_binding.v4_3.json"
)
CODE_BINDING_SET_FILENAME_V4_3 = "code_binding_set.v4_3.json"
FUTURE_SOURCE_ENVELOPE_FILENAME_V4_3 = "future_source_envelope.v4_3.json"
CYCLE_ROOT_FILENAME_V4_3 = "cycle_root.v4_3.json"
DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_3 = (
    "definition_identity_collision_audit.v4_3.json"
)
PRECOMMITTED_STATE_FILENAME_V4_3 = "cycle_state.precommitted.v4_1.json"
DISCOVERY_SOURCE_NODE_FILENAME_V4_3 = "discovery_source_node.v4_3.json"
DISCOVERY_STATE_FILENAME_V4_3 = "cycle_state.discovery.v4_1.json"
PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_3 = (
    "prereg_discovery_orchestration.v4_3.json"
)
READBACK_REPORT_FILENAME_V4_3 = "candidate_preregistration_readback.v4_3.json"

INPUT_FILENAMES_V4_3 = (
    AQUANT_IDEA_SOURCE_SET_RECEIPT_FILENAME_V4_3,
    OPERATOR_SEMANTICS_FILENAME_V4_3,
    COMPARISON_CATALOG_RECEIPT_FILENAME_V4_3,
    CANDIDATE_SELECTION_SPEC_FILENAME_V4_3,
    STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_3,
    CODE_BINDING_SET_FILENAME_V4_3,
    FUTURE_SOURCE_ENVELOPE_FILENAME_V4_3,
    CYCLE_ROOT_FILENAME_V4_3,
    DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_3,
    PRECOMMITTED_STATE_FILENAME_V4_3,
    DISCOVERY_SOURCE_NODE_FILENAME_V4_3,
    DISCOVERY_STATE_FILENAME_V4_3,
    PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_3,
)

CODE_BINDING_PATHS_V4_3 = (
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

PROTECTED_BINDING_NAMES_V4_3 = (
    "registry",
    "latest_pointer",
    "catalog",
    "fundamental_latest",
    "latest_manifest",
)
PROTECTED_BINDING_RELATIVE_PATHS_V4_3 = (
    "quant_investor/factor_registry/mined_factors.json",
    "data/parquet/cn/_latest.json",
    "data/parquet/cn/_catalog.json",
    "data/parquet/cn/_fundamental_latest.json",
    "data/parquet/cn/latest_manifest.json",
)

REPOSITORY_ROOT_V4_3 = Path("/Users/maxwell/mySpace/myQuant")
V4_2_LOCKED_FILES_V4_3 = (
    {
        "order": 1,
        "lock_role": "pure_contract",
        "relative_path": (
            "quant_investor/factors/"
            "governance_candidate_preregistration_v4_2.py"
        ),
        "expected_byte_sha256": (
            "f05007568a955bfe02fc3f3bf7d7b6694259840deee4f4851473cf2a96bc90cc"
        ),
    },
    {
        "order": 2,
        "lock_role": "bundle_contract",
        "relative_path": (
            "quant_investor/factors/"
            "governance_candidate_preregistration_bundle_v4_2.py"
        ),
        "expected_byte_sha256": (
            "49b0c3dd5550494c4fd945234b884b45a25e716699fb6d5a4e5be100f9d40bfe"
        ),
    },
    {
        "order": 3,
        "lock_role": "publisher_cli",
        "relative_path": "scripts/build_factor_v4_2_candidate_preregistration.py",
        "expected_byte_sha256": (
            "e9a7a03094bfd5d260e515a0c5dd6c3b2f0714aec4b9467d7b31c28252637e17"
        ),
    },
    {
        "order": 4,
        "lock_role": "pure_contract_test",
        "relative_path": (
            "tests/unit/"
            "test_factor_governance_candidate_preregistration_v4_2.py"
        ),
        "expected_byte_sha256": (
            "759582cc2ab2a6e770dfc61d8f37a6a4afa31715eeda3783d9616385394977d4"
        ),
    },
    {
        "order": 5,
        "lock_role": "bundle_contract_test",
        "relative_path": (
            "tests/unit/"
            "test_factor_governance_candidate_preregistration_bundle_v4_2.py"
        ),
        "expected_byte_sha256": (
            "891249806cf581807b56f3c5dfd082932431b64ba1a92c993878d99adb365d40"
        ),
    },
    {
        "order": 6,
        "lock_role": "publisher_cli_test",
        "relative_path": (
            "tests/unit/test_build_factor_v4_2_candidate_preregistration.py"
        ),
        "expected_byte_sha256": (
            "854511a9ef9cbd62a6a54b1b8098b9626d95b346bca2d5949c1581ff065067f3"
        ),
    },
)

BASE230_CATALOG_PATH_V4_3 = Path(
    "/Users/maxwell/mySpace/myQuant/reports/factor_governance/private/"
    "v4_pre_admission/factor_v4_pre_admission_20260718_083224/"
    "candidate_catalog.v4.json"
)
BASE230_CATALOG_BYTE_SHA256_V4_3 = (
    "24860fbaa6482ecbffccb4bc41fc842475f76e308b4b232ef5bffe427a61efa4"
)
BASE230_CATALOG_SEMANTIC_SHA256_V4_3 = (
    "e427a71fd95be62aca85bc893a809d3c54cea965976cdcff9a0a4f1500b07c99"
)
FORMAL267_CATALOG_PATH_V4_3 = Path(
    "/Users/maxwell/mySpace/myQuant/reports/factor_governance/private/"
    "v4_1_formal_catalog/factor_v4_1_formal_catalog_20260718T191045Z/"
    "candidate_catalog.v4.json"
)
FORMAL267_CATALOG_BYTE_SHA256_V4_3 = (
    "09cb6ac73590a48e826845f608e4bd733e27c183b6abaa2079436ba5bb2169ee"
)
FORMAL267_CATALOG_SEMANTIC_SHA256_V4_3 = (
    "b4f2b2b80e1bfc69ea8be9228d9021afdbeee28540fc51c2e7ead100a219f75a"
)
V4_2_IDENTITY_SOURCE_PATH_V4_3 = (
    REPOSITORY_ROOT_V4_3 / V4_2_LOCKED_FILES_V4_3[0]["relative_path"]
)
V4_2_IDENTITY_SOURCE_BYTE_SHA256_V4_3 = V4_2_LOCKED_FILES_V4_3[0][
    "expected_byte_sha256"
]
V4_2_DEFINITION_IDENTITIES_V4_3 = (
    (
        "alpha_range_position_momentum_20d",
        "8e486283e2c36a4ecdfcd4059811afb4e42e75f53a6575f972ee17f2665a826f",
    ),
    (
        "pv_low_overnight_gap_20d",
        "a060bd0a52353b218bb963658073e20b1b9bc5cd598c7c4207263c7f45d7dd4e",
    ),
    (
        "pv_low_vol_ratio_10_60",
        "b8672e8996696c4f820f30cf6c4b97b2641cefe8b6e2ecd72ba1874685f87ac7",
    ),
    (
        "pv_price_volume_consistency_20d",
        "fe70f67577bc2bcd4d7bb4275d2b7aac3f4e2671ffd618cd9400d1f02145a41d",
    ),
)

LATEST_POINTER_BYTE_SHA256_V4_3 = (
    "551a16aef636630ab25f34ddd8b8a1ca343e993a529678d2222ee402f16ff285"
)
SNAPSHOT_MANIFEST_BYTE_SHA256_V4_3 = (
    "11b0edbc69609d07fa6bcaba33936ffdc7d15ab3f44845a9c658583e89cf1f71"
)
PIT_MEMBERSHIP_BYTE_SHA256_V4_3 = (
    "6a4d42edd9581c5cf8ba6a472e3b89bd1747eb1c1731f07cfd53efc949ebf4e1"
)
PIT_MANIFEST_BYTE_SHA256_V4_3 = (
    "9c9c9ee1af849e669e50d0593e524f42573ae9d9f367f185414574abc79125b1"
)
COMPONENTS_BYTE_SHA256_V4_3 = (
    "35b8f45b559dfe3c15459cf817d1fef74aca22df410d4f5b02426e65be618f60"
)
CALENDAR_SEMANTIC_SHA256_V4_3 = (
    "99be5e97027fa1837eb737bd6aa4d1adee57107a3592ed14c30858dc5be28f48"
)
TABLE_INVENTORY_SEMANTIC_SHA256_V4_3 = (
    "d3b281045dfa34af49371a2847877920a062ac077aeee8525d381fc4713a7330"
)
SERVING_INVENTORY_SEMANTIC_SHA256_V4_3 = (
    "fd15330350fff4e92684d7dfb6bf4b5077ba9e547aa3321f94db3b957ff4e7bc"
)
FULL_A_SCOPE_COUNT_V4_3 = 5502
FULL_A_SCOPE_SHA256_V4_3 = (
    "41ad09c4c6f759714682ffce4420f6cbb9c2bc34827f443bb4f6965485e69721"
)
SERVING_INVENTORY_COUNT_V4_3 = 5728

STRICT_SOURCE_SCHEMA_VERSION_V4_3 = (
    "factor-governance-strict-full-a-source-binding.v4.3"
)
CODE_BINDING_SET_SCHEMA_VERSION_V4_3 = "factor-governance-code-binding-set.v4.3"
V4_2_CONTRACT_LOCK_SCHEMA_VERSION_V4_3 = (
    "factor-governance-v4-2-six-file-contract-lock.v4.3"
)
CYCLE_ROOT_SCHEMA_VERSION_V4_3 = "factor-governance-cycle-root.v4.3"
READBACK_REPORT_SCHEMA_VERSION_V4_3 = (
    "factor-governance-candidate-preregistration-readback.v4.3"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CN_SYMBOL = re.compile(r"[0-9]{6}\.(?:SH|SZ|BJ)")
_DESCRIPTOR_FIELDS = frozenset(
    {"absolute_path", "byte_sha256", "size_bytes", "mode", "uid", "nlink"}
)


class FactorGovernanceCandidatePreregistrationBundleV4_3Error(ValueError):
    """Raised when a v4.3 bundle or its publication inputs fail closed."""


def _error(message: str) -> FactorGovernanceCandidatePreregistrationBundleV4_3Error:
    return FactorGovernanceCandidatePreregistrationBundleV4_3Error(message)


def _canonical_json(value: Any) -> bytes:
    return prereg.canonical_json_bytes_v4_3(value)


def _canonical_file(value: Any) -> bytes:
    return prereg.canonical_file_bytes_v4_3(value)


def _semantic(value: Any) -> str:
    return prereg.semantic_sha256_v4_3(value)


def _exact(value: Any, fields: set[str] | frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    payload = dict(value)
    if any(type(key) is not str for key in payload):
        raise _error(f"{label} field names must be strings")
    missing = sorted(set(fields) - set(payload))
    unknown = sorted(set(payload) - set(fields))
    if missing or unknown:
        raise _error(
            f"{label} fields invalid: missing={','.join(missing) or '-'};"
            f"unknown={','.join(unknown) or '-'}"
        )
    _canonical_json(payload)
    return payload


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise _error(f"{label} must be lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise _error(f"{label} must be a positive integer")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if type(value) is not str or not value.startswith("/") or "\x00" in value:
        raise _error(f"{label} must be an absolute normalized path")
    path = Path(value)
    if value == "/" or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise _error(f"{label} must be an absolute normalized path")
    if os.path.abspath(value) != value:
        raise _error(f"{label} must be an absolute normalized path")
    return path


def _self_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "artifact_semantic_sha256"}


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result["artifact_semantic_sha256"] = _semantic(_self_payload(result))
    return result


def _validate_self(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    supplied = _sha256(payload.get("artifact_semantic_sha256"), f"{label} self SHA")
    if supplied != _semantic(_self_payload(payload)):
        raise _error(f"{label} artifact_semantic_sha256 mismatch")
    return payload


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value}")

    def exact_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=exact_object,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise _error(f"{label} must be strict finite JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(f"{label} must be a JSON object")
    return value


def _signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _assert_owned_nofollow_chain(
    target: Path,
    *,
    boundary: Path,
    include_target: bool,
    label: str,
) -> None:
    try:
        relative = target.relative_to(boundary)
    except ValueError as exc:
        raise _error(f"{label} escapes its exact repository boundary") from exc
    current = boundary
    parts = relative.parts if include_target else relative.parts[:-1]
    for part in ("", *parts):
        if part:
            current /= part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise _error(f"{label} directory chain is missing: {current}") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or int(metadata.st_uid) != os.getuid()
        ):
            raise _error(f"{label} directory chain is unsafe: {current}")


@dataclass(frozen=True)
class _StableFile:
    path: Path
    raw: bytes
    descriptor: dict[str, Any]
    signature: tuple[int, ...]


def _stable_file(
    path: Path,
    *,
    boundary: Path,
    label: str,
    require_single_link: bool = True,
) -> _StableFile:
    if not path.is_absolute():
        raise _error(f"{label} path must be absolute")
    _assert_owned_nofollow_chain(path, boundary=boundary, include_target=False, label=label)
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise _error(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_uid) != os.getuid()
        or int(before.st_nlink) < 1
        or require_single_link and int(before.st_nlink) != 1
    ):
        raise _error(f"{label} owner/regular/non-symlink hard-link contract failed")
    identity = _signature(before)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise _error(f"{label} safe open failed: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if _signature(opened) != identity:
            raise _error(f"{label} changed while opening")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _signature(after) != identity:
            raise _error(f"{label} changed while reading")
        raw = b"".join(chunks)
        if len(raw) != int(after.st_size):
            raise _error(f"{label} read length mismatch")
    finally:
        os.close(descriptor)
    final = os.lstat(path)
    if _signature(final) != identity:
        raise _error(f"{label} path identity changed after reading")
    return _StableFile(
        path=path,
        raw=raw,
        descriptor={
            "absolute_path": str(path),
            "byte_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "mode": stat.S_IMODE(after.st_mode),
            "uid": int(after.st_uid),
            "nlink": int(after.st_nlink),
        },
        signature=identity,
    )


def _validated_descriptor(value: Any, *, expected_path: Path, label: str) -> dict[str, Any]:
    payload = _exact(value, _DESCRIPTOR_FIELDS, label)
    path = _absolute_path(payload["absolute_path"], f"{label}.absolute_path")
    if path != expected_path:
        raise _error(f"{label} absolute path mismatch")
    result = {
        "absolute_path": str(path),
        "byte_sha256": _sha256(payload["byte_sha256"], f"{label}.byte_sha256"),
        "size_bytes": _positive_int(payload["size_bytes"], f"{label}.size_bytes"),
        "mode": payload["mode"],
        "uid": payload["uid"],
        "nlink": payload["nlink"],
    }
    if (
        type(result["mode"]) is not int
        or result["mode"] <= 0
        or result["mode"] > 0o7777
        or result["uid"] != os.getuid()
        or result["nlink"] != 1
    ):
        raise _error(f"{label} owner/mode/nlink contract mismatch")
    return result


def _observe_descriptor(
    value: Mapping[str, Any], *, expected_path: Path, boundary: Path, label: str
) -> _StableFile:
    expected = _validated_descriptor(value, expected_path=expected_path, label=label)
    observed = _stable_file(expected_path, boundary=boundary, label=label)
    if observed.descriptor != expected:
        raise _error(f"{label} stable descriptor mismatch")
    return observed


@dataclass(frozen=True)
class _TreeSnapshot:
    summary: dict[str, Any]
    inventory: tuple[dict[str, Any], ...]
    identities: tuple[tuple[str, tuple[int, ...]], ...]


def _inventory_tree(root: Path, *, boundary: Path, label: str) -> _TreeSnapshot:
    _assert_owned_nofollow_chain(root, boundary=boundary, include_target=True, label=label)
    pending = [root]
    inventory: list[dict[str, Any]] = []
    identities: list[tuple[str, tuple[int, ...]]] = []
    while pending:
        directory = pending.pop()
        relative_directory = directory.relative_to(root)
        directory_name = "." if not relative_directory.parts else relative_directory.as_posix()
        metadata = os.lstat(directory)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or int(metadata.st_uid) != os.getuid()
        ):
            raise _error(f"{label} directory is unsafe: {directory}")
        identities.append((f"dir:{directory_name}", _signature(metadata)))
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise _error(f"{label} directory is unreadable: {directory}") from exc
        for entry in entries:
            path = directory / entry.name
            relative = path.relative_to(root)
            item_metadata = os.lstat(path)
            if stat.S_ISLNK(item_metadata.st_mode) or int(item_metadata.st_uid) != os.getuid():
                raise _error(f"{label} entry is unsafe: {path}")
            if stat.S_ISDIR(item_metadata.st_mode):
                pending.append(path)
                continue
            if not stat.S_ISREG(item_metadata.st_mode) or int(item_metadata.st_nlink) < 1:
                raise _error(f"{label} file must be owner regular: {path}")
            observed = _stable_file(
                path,
                boundary=boundary,
                label=f"{label} file",
                require_single_link=False,
            )
            identities.append((f"file:{relative.as_posix()}", observed.signature))
            inventory.append(
                {
                    "relative_path": relative.as_posix(),
                    "size_bytes": len(observed.raw),
                    "sha256": observed.descriptor["byte_sha256"],
                    "hard_link_count": observed.descriptor["nlink"],
                    "dataset_member": bool(
                        path.suffix == ".parquet"
                        and all(not part.startswith((".", "_")) for part in relative.parts)
                    ),
                }
            )
    inventory.sort(key=lambda row: row["relative_path"])
    identities.sort(key=lambda row: row[0])
    semantic = hashlib.sha256(_canonical_file(inventory)).hexdigest()
    return _TreeSnapshot(
        summary={
            "absolute_root": str(root),
            "regular_file_count": len(inventory),
            "parquet_file_count": sum(row["dataset_member"] for row in inventory),
            "inventory_semantic_sha256": semantic,
        },
        inventory=tuple(copy.deepcopy(inventory)),
        identities=tuple(identities),
    )


def _stable_tree(root: Path, *, boundary: Path, label: str) -> _TreeSnapshot:
    first = _inventory_tree(root, boundary=boundary, label=label)
    second = _inventory_tree(root, boundary=boundary, label=label)
    if first != second:
        raise _error(f"{label} changed across stable inventory passes")
    return first


def _binding_record(value: Any, label: str) -> dict[str, Any]:
    payload = _exact(value, {"absolute_path", "size_bytes", "sha256"}, label)
    return {
        "absolute_path": str(_absolute_path(payload["absolute_path"], f"{label}.absolute_path")),
        "size_bytes": _positive_int(payload["size_bytes"], f"{label}.size_bytes"),
        "sha256": _sha256(payload["sha256"], f"{label}.sha256"),
    }


def _validate_table_binding(value: Any) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "absolute_root",
            "regular_file_count",
            "parquet_file_count",
            "inventory_sha256",
            "parquet_inventory",
            "bound_symbol_inventory",
        },
        "backend table binding",
    )
    root = _absolute_path(payload["absolute_root"], "backend table absolute_root")
    rows = payload["parquet_inventory"]
    if not isinstance(rows, list) or not rows:
        raise _error("backend table inventory must be a non-empty list")
    normalized_rows: list[dict[str, Any]] = []
    previous: str | None = None
    for index, item in enumerate(rows):
        row = _exact(
            item,
            {"relative_path", "size_bytes", "sha256", "hard_link_count", "dataset_member"},
            f"backend table inventory[{index}]",
        )
        relative = row["relative_path"]
        if (
            type(relative) is not str
            or not relative
            or relative.startswith("/")
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
            or previous is not None and relative <= previous
        ):
            raise _error("backend table inventory paths must be safe, sorted, and unique")
        previous = relative
        if (
            type(row["hard_link_count"]) is not int
            or row["hard_link_count"] < 1
            or type(row["dataset_member"]) is not bool
        ):
            raise _error("backend table inventory hard-link/member contract mismatch")
        normalized_rows.append(
            {
                "relative_path": relative,
                "size_bytes": _positive_int(row["size_bytes"], "backend table size"),
                "sha256": _sha256(row["sha256"], "backend table SHA"),
                "hard_link_count": row["hard_link_count"],
                "dataset_member": row["dataset_member"],
            }
        )
    regular_count = _positive_int(payload["regular_file_count"], "table regular_file_count")
    parquet_count = _positive_int(payload["parquet_file_count"], "table parquet_file_count")
    if regular_count != len(normalized_rows) or parquet_count != sum(
        row["dataset_member"] for row in normalized_rows
    ):
        raise _error("backend table inventory counts mismatch")
    inventory_sha = hashlib.sha256(_canonical_file(normalized_rows)).hexdigest()
    if (
        _sha256(payload["inventory_sha256"], "backend table inventory SHA")
        != inventory_sha
        or inventory_sha != TABLE_INVENTORY_SEMANTIC_SHA256_V4_3
    ):
        raise _error("backend table inventory semantic SHA mismatch")
    symbol_inventory = _exact(
        payload["bound_symbol_inventory"],
        {"symbol_count", "symbols_newline_sha256", "noncanonical_symbol_count"},
        "backend bound symbol inventory",
    )
    if (
        type(symbol_inventory["symbol_count"]) is not int
        or symbol_inventory["symbol_count"] <= 0
        or symbol_inventory["noncanonical_symbol_count"] != 0
    ):
        raise _error("backend bound symbol inventory mismatch")
    _sha256(symbol_inventory["symbols_newline_sha256"], "bound symbols SHA")
    return {
        "absolute_root": str(root),
        "regular_file_count": regular_count,
        "parquet_file_count": parquet_count,
        "inventory_sha256": inventory_sha,
        "parquet_inventory": normalized_rows,
        "bound_symbol_inventory": copy.deepcopy(symbol_inventory),
    }


def _validate_backend_binding(value: Any) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "market",
            "snapshot_id",
            "cutoff_date",
            "latest_pointer",
            "snapshot_manifest",
            "components",
            "pit_generation",
            "table",
            "calendar",
            "eligibility_boundary",
            "readiness",
            "side_effects",
        },
        "v4.1 backend binding",
    )
    if (
        payload["schema_version"] != INPUT_BINDING_SCHEMA_VERSION
        or payload["market"] != "CN"
        or payload["snapshot_id"] != SNAPSHOT_ID_V4_3
        or payload["cutoff_date"] != CUTOFF_V4_3
    ):
        raise _error("backend binding fixed cutoff identity mismatch")
    latest = _binding_record(payload["latest_pointer"], "latest pointer binding")
    snapshot = _binding_record(payload["snapshot_manifest"], "snapshot manifest binding")
    components = _exact(
        payload["components"],
        {"absolute_path", "size_bytes", "sha256", "universe", "count", "newline_set_sha256"},
        "components binding",
    )
    component_record = _binding_record(
        {key: components[key] for key in ("absolute_path", "size_bytes", "sha256")},
        "components binding",
    )
    if (
        components["universe"] != "full_a"
        or components["count"] != FULL_A_SCOPE_COUNT_V4_3
        or components["newline_set_sha256"] != FULL_A_SCOPE_SHA256_V4_3
    ):
        raise _error("components fixed full-A identity mismatch")
    pit = _exact(
        payload["pit_generation"],
        {"generation_id", "manifest", "membership", "row_count", "historical_alias_table_evidence"},
        "PIT generation binding",
    )
    if pit["generation_id"] != "pit-20260717-5a3853ca2dd955e3":
        raise _error("PIT generation identity mismatch")
    pit_manifest = _binding_record(pit["manifest"], "PIT manifest binding")
    pit_membership = _binding_record(pit["membership"], "PIT membership binding")
    _positive_int(pit["row_count"], "PIT row_count")
    if not isinstance(pit["historical_alias_table_evidence"], list):
        raise _error("PIT historical alias evidence must be a list")
    table = _validate_table_binding(payload["table"])
    calendar = _exact(
        payload["calendar"],
        {"analysis_start", "cutoff_date", "open_session_count", "open_sessions", "semantic_sha256"},
        "backend calendar",
    )
    sessions = calendar["open_sessions"]
    if (
        type(calendar["analysis_start"]) is not str
        or calendar["cutoff_date"] != CUTOFF_V4_3
        or not isinstance(sessions, list)
        or not sessions
        or sessions != sorted(set(sessions))
        or sessions[0] != calendar["analysis_start"]
        or sessions[-1] != CUTOFF_V4_3
        or calendar["open_session_count"] != len(sessions)
    ):
        raise _error("backend calendar identity mismatch")
    _sha256(calendar["semantic_sha256"], "backend calendar semantic SHA")
    eligibility = _exact(
        payload["eligibility_boundary"],
        {"component_source", "pit_source", "bar_source", "serving_inventory"},
        "eligibility boundary",
    )
    serving = _exact(
        eligibility["serving_inventory"],
        {"absolute_root", "symbol_count", "use", "was_scanned"},
        "serving boundary",
    )
    serving_root = _absolute_path(serving["absolute_root"], "serving root")
    pointer_path = Path(latest["absolute_path"])
    if tuple(pointer_path.parts[-4:]) != ("data", "parquet", "cn", "_latest.json"):
        raise _error("latest pointer path is not the exact CN control")
    project_root = pointer_path.parents[3]
    cn_root = pointer_path.parent
    expected_snapshot = cn_root / "_snapshots" / f"{SNAPSHOT_ID_V4_3}.json"
    expected_table = cn_root / "_snapshots" / SNAPSHOT_ID_V4_3 / "table" / "bars"
    expected_serving = cn_root / "_snapshots" / SNAPSHOT_ID_V4_3 / "serving" / "bars"
    expected_components = project_root / "data" / "cn_universe" / "cn_index_components.json"
    expected_pit = cn_root / "reference" / "_generations" / str(pit["generation_id"])
    if (
        Path(snapshot["absolute_path"]) != expected_snapshot
        or Path(component_record["absolute_path"]) != expected_components
        or Path(pit_manifest["absolute_path"]) != expected_pit / "manifest.json"
        or Path(pit_membership["absolute_path"]) != expected_pit / "stock_basic_membership.parquet"
        or Path(table["absolute_root"]) != expected_table
        or serving_root != expected_serving
        or eligibility["component_source"] != str(expected_components)
        or eligibility["pit_source"] != str(expected_pit / "stock_basic_membership.parquet")
        or eligibility["bar_source"] != str(expected_table)
    ):
        raise _error("backend exact source paths mismatch")
    if (
        serving["symbol_count"] != SERVING_INVENTORY_COUNT_V4_3
        or serving["use"] != SOURCE_USE_PROHIBITED
        or serving["was_scanned"] is not False
    ):
        raise _error("backend serving eligibility boundary mismatch")
    if payload["readiness"] != "EXPLORATORY_INPUT_BOUND":
        raise _error("backend readiness mismatch")
    expected_effects = {
        "registry": False,
        "wal": False,
        "budget": False,
        "apply": False,
        "broker": False,
        "order": False,
        "trade": False,
        "network": False,
    }
    if payload["side_effects"] != expected_effects:
        raise _error("backend side effects mismatch")
    return {
        **copy.deepcopy(payload),
        "latest_pointer": latest,
        "snapshot_manifest": snapshot,
        "components": {
            **component_record,
            "universe": "full_a",
            "count": FULL_A_SCOPE_COUNT_V4_3,
            "newline_set_sha256": FULL_A_SCOPE_SHA256_V4_3,
        },
        "pit_generation": {
            **copy.deepcopy(pit),
            "manifest": pit_manifest,
            "membership": pit_membership,
        },
        "table": table,
    }


_SOURCE_BINDING_NAMES = (
    "latest_pointer",
    "snapshot_manifest",
    "components",
    "pit_generation_manifest",
    "pit_membership",
)


def _source_descriptor_rows(backend: Mapping[str, Any], *, boundary: Path) -> tuple[list[dict[str, Any]], tuple[_StableFile, ...]]:
    records = (
        backend["latest_pointer"],
        backend["snapshot_manifest"],
        backend["components"],
        backend["pit_generation"]["manifest"],
        backend["pit_generation"]["membership"],
    )
    expected_hashes = (
        LATEST_POINTER_BYTE_SHA256_V4_3,
        SNAPSHOT_MANIFEST_BYTE_SHA256_V4_3,
        COMPONENTS_BYTE_SHA256_V4_3,
        PIT_MANIFEST_BYTE_SHA256_V4_3,
        PIT_MEMBERSHIP_BYTE_SHA256_V4_3,
    )
    rows: list[dict[str, Any]] = []
    observations: list[_StableFile] = []
    for name, record, expected_sha in zip(_SOURCE_BINDING_NAMES, records, expected_hashes, strict=True):
        if record["sha256"] != expected_sha:
            raise _error(f"fixed source SHA mismatch: {name}")
        observed = _stable_file(Path(record["absolute_path"]), boundary=boundary, label=f"source {name}")
        if (
            observed.descriptor["byte_sha256"] != expected_sha
            or observed.descriptor["size_bytes"] != record["size_bytes"]
        ):
            raise _error(f"stable fixed source mismatch: {name}")
        rows.append({"name": name, **copy.deepcopy(observed.descriptor)})
        observations.append(observed)
    return rows, tuple(observations)


def validate_strict_full_a_source_binding_v4_3(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "market",
            "universe",
            "cycle_id",
            "snapshot_id",
            "analysis_start",
            "cutoff",
            "latest_available_trade_date",
            "latest_complete_trade_date",
            "expected_scope_count",
            "full_a_scope_sha256",
            "serving_inventory_count",
            "calendar_semantic_sha256",
            "table_inventory_semantic_sha256",
            "serving_inventory_semantic_sha256",
            "backend_binding_schema_version",
            "backend_binding_semantic_sha256",
            "backend_binding",
            "ordered_source_file_bindings",
            "table_inventory_binding",
            "serving_inventory_binding",
            "artifact_semantic_sha256",
        },
        "strict full-A source binding",
    )
    if (
        payload["schema_version"] != STRICT_SOURCE_SCHEMA_VERSION_V4_3
        or payload["protocol_version"] != "v4"
        or payload["market"] != "CN"
        or payload["universe"] != "full_a"
        or payload["cycle_id"] != CYCLE_ID_V4_3
        or payload["snapshot_id"] != SNAPSHOT_ID_V4_3
        or payload["cutoff"] != CUTOFF_V4_3
        or payload["latest_available_trade_date"] != COMPACT_CUTOFF_V4_3
        or payload["latest_complete_trade_date"] != COMPACT_CUTOFF_V4_3
        or payload["expected_scope_count"] != FULL_A_SCOPE_COUNT_V4_3
        or payload["full_a_scope_sha256"] != FULL_A_SCOPE_SHA256_V4_3
        or payload["serving_inventory_count"] != SERVING_INVENTORY_COUNT_V4_3
        or payload["calendar_semantic_sha256"] != CALENDAR_SEMANTIC_SHA256_V4_3
        or payload["table_inventory_semantic_sha256"] != TABLE_INVENTORY_SEMANTIC_SHA256_V4_3
        or payload["serving_inventory_semantic_sha256"] != SERVING_INVENTORY_SEMANTIC_SHA256_V4_3
        or payload["backend_binding_schema_version"] != INPUT_BINDING_SCHEMA_VERSION
    ):
        raise _error("strict source fixed identity mismatch")
    backend = _validate_backend_binding(payload["backend_binding"])
    if (
        payload["analysis_start"] != backend["calendar"]["analysis_start"]
        or payload["backend_binding_semantic_sha256"]
        != binding_semantic_sha256_v4_1(backend)
    ):
        raise _error("strict source backend semantic identity mismatch")
    rows = payload["ordered_source_file_bindings"]
    if not isinstance(rows, list) or len(rows) != len(_SOURCE_BINDING_NAMES):
        raise _error("strict source file binding inventory mismatch")
    expected_records = (
        backend["latest_pointer"],
        backend["snapshot_manifest"],
        backend["components"],
        backend["pit_generation"]["manifest"],
        backend["pit_generation"]["membership"],
    )
    for index, (item, name, record) in enumerate(zip(rows, _SOURCE_BINDING_NAMES, expected_records, strict=True)):
        row = _exact(item, {"name", *_DESCRIPTOR_FIELDS}, f"source binding[{index}]")
        if row["name"] != name:
            raise _error("strict source binding order/name mismatch")
        descriptor = _validated_descriptor(
            {key: row[key] for key in _DESCRIPTOR_FIELDS},
            expected_path=Path(record["absolute_path"]),
            label=f"source binding {name}",
        )
        if (
            descriptor["byte_sha256"] != record["sha256"]
            or descriptor["size_bytes"] != record["size_bytes"]
        ):
            raise _error(f"strict source descriptor mismatch: {name}")
    table = _exact(
        payload["table_inventory_binding"],
        {"absolute_root", "regular_file_count", "parquet_file_count", "inventory_semantic_sha256"},
        "table inventory binding",
    )
    serving = _exact(
        payload["serving_inventory_binding"],
        {"absolute_root", "regular_file_count", "parquet_file_count", "inventory_semantic_sha256"},
        "serving inventory binding",
    )
    if (
        table != {
            "absolute_root": backend["table"]["absolute_root"],
            "regular_file_count": backend["table"]["regular_file_count"],
            "parquet_file_count": backend["table"]["parquet_file_count"],
            "inventory_semantic_sha256": TABLE_INVENTORY_SEMANTIC_SHA256_V4_3,
        }
        or serving["absolute_root"]
        != backend["eligibility_boundary"]["serving_inventory"]["absolute_root"]
        or serving["regular_file_count"] != SERVING_INVENTORY_COUNT_V4_3
        or serving["parquet_file_count"] != SERVING_INVENTORY_COUNT_V4_3
        or serving["inventory_semantic_sha256"] != SERVING_INVENTORY_SEMANTIC_SHA256_V4_3
    ):
        raise _error("strict source table/serving inventory binding mismatch")
    _validate_self(payload, "strict full-A source binding")
    return copy.deepcopy(payload)


@dataclass(frozen=True)
class _StrictSourceSnapshot:
    files: tuple[_StableFile, ...]
    table: _TreeSnapshot
    serving: _TreeSnapshot


def revalidate_strict_full_a_source_binding_v4_3(
    value: Mapping[str, Any], *, repository_root: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    source = validate_strict_full_a_source_binding_v4_3(value)
    pointer = Path(source["backend_binding"]["latest_pointer"]["absolute_path"])
    boundary = pointer.parents[3]
    if repository_root is not None and _absolute_path(os.fspath(repository_root), "repository_root") != boundary:
        raise _error("strict source repository root mismatch")
    rows, files = _source_descriptor_rows(source["backend_binding"], boundary=boundary)
    if rows != source["ordered_source_file_bindings"]:
        raise _error("strict source file descriptors drifted")
    pointer_value = _strict_json_object(files[0].raw, "latest pointer")
    manifest_value = _strict_json_object(files[1].raw, "snapshot manifest")
    components_value = _strict_json_object(files[2].raw, "components")
    pit_manifest_value = _strict_json_object(files[3].raw, "PIT generation manifest")
    symbols = components_value.get("full_a")
    if (
        pointer_value.get("snapshot_id") != SNAPSHOT_ID_V4_3
        or pointer_value.get("status") != "OK"
        or pointer_value.get("blockers") != []
        or pointer_value.get("latest_available_trade_date") != COMPACT_CUTOFF_V4_3
        or pointer_value.get("latest_complete_trade_date") != COMPACT_CUTOFF_V4_3
        or manifest_value.get("snapshot_id") != SNAPSHOT_ID_V4_3
        or manifest_value.get("status") != "OK"
        or manifest_value.get("blockers") != []
        or manifest_value.get("latest_available_trade_date") != COMPACT_CUTOFF_V4_3
        or manifest_value.get("latest_complete_trade_date") != COMPACT_CUTOFF_V4_3
        or not isinstance(symbols, list)
        or len(symbols) != FULL_A_SCOPE_COUNT_V4_3
        or symbols != sorted(set(symbols))
        or any(type(item) is not str or _CN_SYMBOL.fullmatch(item) is None for item in symbols)
        or hashlib.sha256("\n".join(symbols).encode("ascii")).hexdigest()
        != FULL_A_SCOPE_SHA256_V4_3
        or pit_manifest_value.get("generation_id") != "pit-20260717-5a3853ca2dd955e3"
        or pit_manifest_value.get("canonical_sha256") != PIT_MEMBERSHIP_BYTE_SHA256_V4_3
    ):
        raise _error("strict source live semantic revalidation failed")
    table_root = Path(source["table_inventory_binding"]["absolute_root"])
    serving_root = Path(source["serving_inventory_binding"]["absolute_root"])
    table = _stable_tree(table_root, boundary=boundary, label="strict table inventory")
    serving = _stable_tree(serving_root, boundary=boundary, label="strict serving inventory")
    if (
        table.summary != source["table_inventory_binding"]
        or serving.summary != source["serving_inventory_binding"]
        or list(table.inventory) != source["backend_binding"]["table"]["parquet_inventory"]
    ):
        raise _error("strict table/serving source inventory drifted")
    return {
        "accepted": True,
        "strict_source_binding_semantic_sha256": source["artifact_semantic_sha256"],
        "snapshot": _StrictSourceSnapshot(files=files, table=table, serving=serving),
    }


def build_strict_full_a_source_binding_v4_3(
    *, bound_inputs: BoundCutoffInputsV4_1
) -> dict[str, Any]:
    if not isinstance(bound_inputs, BoundCutoffInputsV4_1):
        raise _error("bound_inputs must be BoundCutoffInputsV4_1")
    backend = _validate_backend_binding(bound_inputs.binding)
    if (
        tuple(bound_inputs.calendar_sessions) != tuple(backend["calendar"]["open_sessions"])
        or len(bound_inputs.component_symbols) != FULL_A_SCOPE_COUNT_V4_3
        or tuple(bound_inputs.component_symbols) != tuple(sorted(set(bound_inputs.component_symbols)))
        or hashlib.sha256("\n".join(bound_inputs.component_symbols).encode("ascii")).hexdigest()
        != FULL_A_SCOPE_SHA256_V4_3
        or len(bound_inputs.pit_records) != backend["pit_generation"]["row_count"]
    ):
        raise _error("bound input normalized data differs from fixed backend binding")
    pointer = Path(backend["latest_pointer"]["absolute_path"])
    boundary = pointer.parents[3]
    source_rows, _files = _source_descriptor_rows(backend, boundary=boundary)
    table = _stable_tree(Path(backend["table"]["absolute_root"]), boundary=boundary, label="strict table inventory")
    serving = _stable_tree(
        Path(backend["eligibility_boundary"]["serving_inventory"]["absolute_root"]),
        boundary=boundary,
        label="strict serving inventory",
    )
    if (
        table.summary["inventory_semantic_sha256"] != TABLE_INVENTORY_SEMANTIC_SHA256_V4_3
        or list(table.inventory) != backend["table"]["parquet_inventory"]
        or serving.summary["inventory_semantic_sha256"] != SERVING_INVENTORY_SEMANTIC_SHA256_V4_3
        or serving.summary["regular_file_count"] != SERVING_INVENTORY_COUNT_V4_3
        or serving.summary["parquet_file_count"] != SERVING_INVENTORY_COUNT_V4_3
    ):
        raise _error("fixed table/serving inventory mismatch")
    return validate_strict_full_a_source_binding_v4_3(
        _seal(
            {
                "schema_version": STRICT_SOURCE_SCHEMA_VERSION_V4_3,
                "protocol_version": "v4",
                "market": "CN",
                "universe": "full_a",
                "cycle_id": CYCLE_ID_V4_3,
                "snapshot_id": SNAPSHOT_ID_V4_3,
                "analysis_start": backend["calendar"]["analysis_start"],
                "cutoff": CUTOFF_V4_3,
                "latest_available_trade_date": COMPACT_CUTOFF_V4_3,
                "latest_complete_trade_date": COMPACT_CUTOFF_V4_3,
                "expected_scope_count": FULL_A_SCOPE_COUNT_V4_3,
                "full_a_scope_sha256": FULL_A_SCOPE_SHA256_V4_3,
                "serving_inventory_count": SERVING_INVENTORY_COUNT_V4_3,
                "calendar_semantic_sha256": CALENDAR_SEMANTIC_SHA256_V4_3,
                "table_inventory_semantic_sha256": TABLE_INVENTORY_SEMANTIC_SHA256_V4_3,
                "serving_inventory_semantic_sha256": SERVING_INVENTORY_SEMANTIC_SHA256_V4_3,
                "backend_binding_schema_version": INPUT_BINDING_SCHEMA_VERSION,
                "backend_binding_semantic_sha256": binding_semantic_sha256_v4_1(backend),
                "backend_binding": backend,
                "ordered_source_file_bindings": source_rows,
                "table_inventory_binding": table.summary,
                "serving_inventory_binding": serving.summary,
            }
        )
    )


def validate_code_binding_set_v4_3(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "path_count",
            "ordered_bindings",
            "artifact_semantic_sha256",
        },
        "code binding set",
    )
    if (
        payload["schema_version"] != CODE_BINDING_SET_SCHEMA_VERSION_V4_3
        or payload["protocol_version"] != "v4"
        or payload["path_count"] != len(CODE_BINDING_PATHS_V4_3)
    ):
        raise _error("code binding set fixed identity mismatch")
    rows = payload["ordered_bindings"]
    if not isinstance(rows, list) or len(rows) != len(CODE_BINDING_PATHS_V4_3):
        raise _error("code binding set inventory mismatch")
    for index, (item, relative) in enumerate(zip(rows, CODE_BINDING_PATHS_V4_3, strict=True), start=1):
        row = _exact(
            item,
            {"order", "relative_path", *_DESCRIPTOR_FIELDS},
            f"code binding[{index}]",
        )
        if row["order"] != index or row["relative_path"] != relative:
            raise _error("code binding path/order mismatch")
        absolute = _absolute_path(row["absolute_path"], "code binding absolute_path")
        if tuple(absolute.parts[-len(Path(relative).parts) :]) != Path(relative).parts:
            raise _error("code binding absolute/relative path mismatch")
        _validated_descriptor(
            {key: row[key] for key in _DESCRIPTOR_FIELDS},
            expected_path=absolute,
            label=f"code binding {relative}",
        )
    _validate_self(payload, "code binding set")
    return copy.deepcopy(payload)


def build_code_binding_set_v4_3(
    *,
    repository_root: str | os.PathLike[str],
    code_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = _absolute_path(os.fspath(repository_root), "repository_root")
    _assert_owned_nofollow_chain(root, boundary=root, include_target=True, label="repository root")
    if not isinstance(code_bindings, Sequence) or isinstance(code_bindings, (str, bytes, bytearray)):
        raise _error("code_bindings must be an exact ordered sequence")
    if len(code_bindings) != len(CODE_BINDING_PATHS_V4_3):
        raise _error("code_bindings must contain the exact ten paths")
    rows: list[dict[str, Any]] = []
    for index, (item, relative) in enumerate(zip(code_bindings, CODE_BINDING_PATHS_V4_3, strict=True), start=1):
        bound = _exact(item, {"relative_path", *_DESCRIPTOR_FIELDS}, f"code input {relative}")
        if bound["relative_path"] != relative:
            raise _error("code input relative path/order mismatch")
        descriptor = _observe_descriptor(
            {key: bound[key] for key in _DESCRIPTOR_FIELDS},
            expected_path=root / relative,
            boundary=root,
            label=f"code binding {relative}",
        ).descriptor
        rows.append({"order": index, "relative_path": relative, **descriptor})
    return validate_code_binding_set_v4_3(
        _seal(
            {
                "schema_version": CODE_BINDING_SET_SCHEMA_VERSION_V4_3,
                "protocol_version": "v4",
                "path_count": len(rows),
                "ordered_bindings": rows,
            }
        )
    )


def revalidate_code_binding_set_v4_3(
    *,
    repository_root: str | os.PathLike[str],
    code_bindings: Sequence[Mapping[str, Any]],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    expected = validate_code_binding_set_v4_3(value)
    live = build_code_binding_set_v4_3(
        repository_root=repository_root,
        code_bindings=code_bindings,
    )
    if _canonical_file(live) != _canonical_file(expected):
        raise _error("code binding set drifted")
    return expected


def _observe_protected_bindings(
    *,
    repository_root: Path,
    protected_bindings: Sequence[Mapping[str, Any]],
) -> tuple[_StableFile, ...]:
    if not isinstance(protected_bindings, Sequence) or isinstance(
        protected_bindings, (str, bytes, bytearray)
    ):
        raise _error("protected_bindings must be an exact ordered sequence")
    if len(protected_bindings) != len(PROTECTED_BINDING_NAMES_V4_3):
        raise _error("protected binding inventory must contain exactly five rows")
    observations: list[_StableFile] = []
    for index, (item, name, relative) in enumerate(
        zip(
            protected_bindings,
            PROTECTED_BINDING_NAMES_V4_3,
            PROTECTED_BINDING_RELATIVE_PATHS_V4_3,
            strict=True,
        )
    ):
        row = _exact(item, {"name", *_DESCRIPTOR_FIELDS}, f"protected binding[{index}]")
        if row["name"] != name:
            raise _error("protected binding name/order mismatch")
        observations.append(
            _observe_descriptor(
                {key: row[key] for key in _DESCRIPTOR_FIELDS},
                expected_path=repository_root / relative,
                boundary=repository_root,
                label=f"protected binding {name}",
            )
        )
    return tuple(observations)


def validate_protected_bindings_v4_3(
    *,
    repository_root: str | os.PathLike[str],
    protected_bindings: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    root = _absolute_path(os.fspath(repository_root), "repository_root")
    _assert_owned_nofollow_chain(root, boundary=root, include_target=True, label="repository root")
    return tuple(copy.deepcopy(item.descriptor) for item in _observe_protected_bindings(
        repository_root=root,
        protected_bindings=protected_bindings,
    ))


_V4_2_LOCK_SPEC_FIELDS_V4_3 = frozenset(
    {"order", "lock_role", "relative_path", "expected_byte_sha256"}
)
_V4_2_LOCK_ROW_FIELDS_V4_3 = frozenset(
    {*_V4_2_LOCK_SPEC_FIELDS_V4_3, *_DESCRIPTOR_FIELDS}
)


def _v4_2_locked_bytes_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    identities = [
        {
            "order": row["order"],
            "lock_role": row["lock_role"],
            "relative_path": row["relative_path"],
            "expected_byte_sha256": row["expected_byte_sha256"],
            "byte_sha256": row["byte_sha256"],
            "size_bytes": row["size_bytes"],
        }
        for row in rows
    ]
    return _semantic(
        {
            "domain": "v4_2-six-file-byte-identities.v4_3",
            "ordered_file_byte_identities": identities,
        }
    )


def _v4_2_locked_descriptors_semantic_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    return _semantic(
        {
            "domain": "v4_2-six-file-runtime-descriptors.v4_3",
            "ordered_locked_files": list(rows),
        }
    )


def validate_v4_2_contract_lock_v4_3(
    value: Mapping[str, Any],
    *,
    repository_root: str | os.PathLike[str] = REPOSITORY_ROOT_V4_3,
) -> dict[str, Any]:
    """Validate the embedded exact-six v4.2 runtime publication lock."""

    root = _absolute_path(os.fspath(repository_root), "repository_root")
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "lock_id",
            "locked_file_count",
            "ordered_locked_files",
            "locked_bytes_sha256",
            "locked_descriptors_semantic_sha256",
            "artifact_semantic_sha256",
        },
        "v4.2 contract lock",
    )
    if (
        payload["schema_version"] != V4_2_CONTRACT_LOCK_SCHEMA_VERSION_V4_3
        or payload["protocol_version"] != "v4"
        or payload["lock_id"] != "v4_2-exact-six-source-and-test-files"
        or payload["locked_file_count"] != len(V4_2_LOCKED_FILES_V4_3)
    ):
        raise _error("v4.2 contract lock fixed identity mismatch")
    rows = payload["ordered_locked_files"]
    if not isinstance(rows, list) or len(rows) != len(V4_2_LOCKED_FILES_V4_3):
        raise _error("v4.2 contract lock inventory mismatch")
    normalized_rows: list[dict[str, Any]] = []
    for index, (item, raw_spec) in enumerate(
        zip(rows, V4_2_LOCKED_FILES_V4_3, strict=True),
        start=1,
    ):
        spec = _exact(
            raw_spec,
            _V4_2_LOCK_SPEC_FIELDS_V4_3,
            f"v4.2 contract lock spec[{index}]",
        )
        relative = spec["relative_path"]
        if (
            type(spec["order"]) is not int
            or type(spec["lock_role"]) is not str
            or not spec["lock_role"]
            or type(relative) is not str
            or not relative
            or Path(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
        ):
            raise _error("v4.2 contract lock spec path/role/order mismatch")
        row = _exact(
            item,
            _V4_2_LOCK_ROW_FIELDS_V4_3,
            f"v4.2 contract lock row[{index}]",
        )
        expected_sha256 = _sha256(
            spec["expected_byte_sha256"],
            f"v4.2 contract lock spec[{index}] expected SHA",
        )
        if (
            spec["order"] != index
            or row["order"] != index
            or row["lock_role"] != spec["lock_role"]
            or row["relative_path"] != spec["relative_path"]
            or row["expected_byte_sha256"] != expected_sha256
        ):
            raise _error("v4.2 contract lock row order/identity mismatch")
        descriptor = _validated_descriptor(
            {key: row[key] for key in _DESCRIPTOR_FIELDS},
            expected_path=root / spec["relative_path"],
            label=f"v4.2 locked file {spec['lock_role']}",
        )
        if (
            descriptor["byte_sha256"] != expected_sha256
            or descriptor["mode"] != 0o644
        ):
            raise _error("v4.2 contract lock SHA/mode mismatch")
        normalized_rows.append({**copy.deepcopy(spec), **descriptor})
    if rows != normalized_rows:
        raise _error("v4.2 contract lock rows are not normalized")
    if payload["locked_bytes_sha256"] != _v4_2_locked_bytes_sha256(
        normalized_rows
    ):
        raise _error("v4.2 contract locked_bytes_sha256 mismatch")
    if payload[
        "locked_descriptors_semantic_sha256"
    ] != _v4_2_locked_descriptors_semantic_sha256(normalized_rows):
        raise _error("v4.2 contract descriptor semantic SHA mismatch")
    _validate_self(payload, "v4.2 contract lock")
    return copy.deepcopy(payload)


@dataclass(frozen=True)
class _V4_2ContractLockSnapshot:
    artifact: dict[str, Any]
    files: tuple[_StableFile, ...]


def _observe_v4_2_contract_lock_snapshot(
    *,
    repository_root: str | os.PathLike[str] = REPOSITORY_ROOT_V4_3,
) -> _V4_2ContractLockSnapshot:
    root = _absolute_path(os.fspath(repository_root), "repository_root")
    _assert_owned_nofollow_chain(
        root,
        boundary=root,
        include_target=True,
        label="repository root",
    )
    rows: list[dict[str, Any]] = []
    observations: list[_StableFile] = []
    for index, raw_spec in enumerate(V4_2_LOCKED_FILES_V4_3, start=1):
        spec = _exact(
            raw_spec,
            _V4_2_LOCK_SPEC_FIELDS_V4_3,
            f"v4.2 contract lock spec[{index}]",
        )
        if spec["order"] != index:
            raise _error("v4.2 contract lock spec order mismatch")
        relative = spec["relative_path"]
        if (
            type(spec["order"]) is not int
            or type(spec["lock_role"]) is not str
            or not spec["lock_role"]
            or type(relative) is not str
            or not relative
            or Path(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
        ):
            raise _error("v4.2 contract lock spec path/role/order mismatch")
        expected_sha256 = _sha256(
            spec["expected_byte_sha256"],
            f"v4.2 contract lock spec[{index}] expected SHA",
        )
        observed = _stable_file(
            root / spec["relative_path"],
            boundary=root,
            label=f"v4.2 locked file {spec['lock_role']}",
        )
        if observed.descriptor["byte_sha256"] != expected_sha256:
            raise _error(
                f"v4.2 locked file {spec['lock_role']} byte SHA mismatch"
            )
        if observed.descriptor["mode"] != 0o644:
            raise _error(f"v4.2 locked file {spec['lock_role']} mode mismatch")
        observations.append(observed)
        rows.append({**copy.deepcopy(spec), **observed.descriptor})
    artifact = validate_v4_2_contract_lock_v4_3(
        _seal(
            {
                "schema_version": V4_2_CONTRACT_LOCK_SCHEMA_VERSION_V4_3,
                "protocol_version": "v4",
                "lock_id": "v4_2-exact-six-source-and-test-files",
                "locked_file_count": len(rows),
                "ordered_locked_files": rows,
                "locked_bytes_sha256": _v4_2_locked_bytes_sha256(rows),
                "locked_descriptors_semantic_sha256": (
                    _v4_2_locked_descriptors_semantic_sha256(rows)
                ),
            }
        ),
        repository_root=root,
    )
    return _V4_2ContractLockSnapshot(
        artifact=artifact,
        files=tuple(observations),
    )


def build_v4_2_contract_lock_v4_3(
    *,
    repository_root: str | os.PathLike[str] = REPOSITORY_ROOT_V4_3,
) -> dict[str, Any]:
    """Observe and seal the exact six v4.2 files as runtime evidence."""

    return _observe_v4_2_contract_lock_snapshot(
        repository_root=repository_root
    ).artifact


def _catalog_identity_inventory(value: Mapping[str, Any], *, count: int, semantic_sha256: str, label: str) -> list[dict[str, str]]:
    if value.get("semantic_sha256") != semantic_sha256:
        raise _error(f"{label} semantic SHA field mismatch")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != count:
        raise _error(f"{label} candidate count mismatch")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise _error(f"{label} candidate[{index}] must be an object")
        name = candidate.get("name")
        identity = candidate.get("definition_sha256")
        if type(name) is not str or not name or name in seen:
            raise _error(f"{label} candidate names must be unique strings")
        seen.add(name)
        rows.append(
            {
                "name": name,
                "definition_identity_sha256": _sha256(identity, f"{label} definition SHA"),
            }
        )
    return sorted(rows, key=lambda row: row["name"])


def _unique_literal_assignment(module: ast.Module, name: str) -> Any:
    nodes = [
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(nodes) != 1:
        raise _error(f"v4.2 identity source must define {name} exactly once")
    try:
        return ast.literal_eval(nodes[0])
    except (ValueError, TypeError) as exc:
        raise _error(f"v4.2 identity source {name} must be a literal") from exc


def _v4_2_identity_inventory(raw: bytes) -> list[dict[str, str]]:
    try:
        module = ast.parse(raw.decode("utf-8"), filename=str(V4_2_IDENTITY_SOURCE_PATH_V4_3))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise _error("v4.2 identity source is not exact parseable Python") from exc
    names = _unique_literal_assignment(module, "EXPECTED_CANDIDATES")
    range_identity = _unique_literal_assignment(module, "AQUANT_RANGE_DEFINITION_SHA256")
    aliases = _unique_literal_assignment(module, "MYQUANT_ALIAS_ROWS")
    expected_names = tuple(name for name, _identity in V4_2_DEFINITION_IDENTITIES_V4_3)
    if names != expected_names or range_identity != V4_2_DEFINITION_IDENTITIES_V4_3[0][1]:
        raise _error("v4.2 identity source candidate/range constants mismatch")
    if not isinstance(aliases, tuple) or len(aliases) != 3:
        raise _error("v4.2 identity source alias rows mismatch")
    observed = {V4_2_DEFINITION_IDENTITIES_V4_3[0][0]: range_identity}
    for row in aliases:
        if not isinstance(row, dict) or set(row) != {
            "candidate",
            "source_factor",
            "direction",
            "source_ast_sha256",
            "bound_definition_sha256",
        }:
            raise _error("v4.2 identity source alias literal fields mismatch")
        name = row["candidate"]
        identity = row["bound_definition_sha256"]
        if name in observed:
            raise _error("v4.2 identity source candidate occurs multiple times")
        observed[name] = identity
    if tuple(observed.items()) != V4_2_DEFINITION_IDENTITIES_V4_3:
        raise _error("v4.2 identity source definition hashes mismatch")
    return [
        {"name": name, "definition_identity_sha256": identity}
        for name, identity in sorted(observed.items())
    ]


@dataclass(frozen=True)
class _ComparisonSnapshot:
    descriptor: dict[str, Any]
    files: tuple[_StableFile, ...]


def _fixed_comparison_snapshot() -> _ComparisonSnapshot:
    boundary = REPOSITORY_ROOT_V4_3
    base = _stable_file(BASE230_CATALOG_PATH_V4_3, boundary=boundary, label="base230 comparison catalog")
    formal = _stable_file(FORMAL267_CATALOG_PATH_V4_3, boundary=boundary, label="v4.1 comparison catalog")
    identity_source = _stable_file(
        V4_2_IDENTITY_SOURCE_PATH_V4_3,
        boundary=boundary,
        label="v4.2 identity source",
    )
    if (
        base.descriptor["byte_sha256"] != BASE230_CATALOG_BYTE_SHA256_V4_3
        or formal.descriptor["byte_sha256"] != FORMAL267_CATALOG_BYTE_SHA256_V4_3
        or identity_source.descriptor["byte_sha256"] != V4_2_IDENTITY_SOURCE_BYTE_SHA256_V4_3
    ):
        raise _error("fixed comparison source byte SHA mismatch")
    base_value = _strict_json_object(base.raw, "base230 comparison catalog")
    formal_value = _strict_json_object(formal.raw, "v4.1 comparison catalog")
    base_inventory = _catalog_identity_inventory(
        base_value,
        count=230,
        semantic_sha256=BASE230_CATALOG_SEMANTIC_SHA256_V4_3,
        label="base230 comparison catalog",
    )
    formal_inventory = _catalog_identity_inventory(
        formal_value,
        count=267,
        semantic_sha256=FORMAL267_CATALOG_SEMANTIC_SHA256_V4_3,
        label="v4.1 comparison catalog",
    )
    formal_by_name = {row["name"]: row["definition_identity_sha256"] for row in formal_inventory}
    if any(formal_by_name.get(row["name"]) != row["definition_identity_sha256"] for row in base_inventory):
        raise _error("base230 comparison identities are not preserved by v4.1")
    v42_inventory = _v4_2_identity_inventory(identity_source.raw)
    combined_by_name = dict(formal_by_name)
    for row in v42_inventory:
        if row["name"] in combined_by_name:
            raise _error("v4.2 identity source collides with v4.1 comparison names")
        combined_by_name[row["name"]] = row["definition_identity_sha256"]
    inventory = [
        {"name": name, "definition_identity_sha256": identity}
        for name, identity in sorted(combined_by_name.items())
    ]
    v42_semantic = _semantic({"definition_identity_inventory": v42_inventory})
    sources = [
        {
            "name": "base230",
            "byte_sha256": BASE230_CATALOG_BYTE_SHA256_V4_3,
            "semantic_sha256": BASE230_CATALOG_SEMANTIC_SHA256_V4_3,
            "candidate_count": 230,
        },
        {
            "name": "v4_1",
            "byte_sha256": FORMAL267_CATALOG_BYTE_SHA256_V4_3,
            "semantic_sha256": FORMAL267_CATALOG_SEMANTIC_SHA256_V4_3,
            "candidate_count": 267,
        },
        {
            "name": "v4_2",
            "byte_sha256": V4_2_IDENTITY_SOURCE_BYTE_SHA256_V4_3,
            "semantic_sha256": v42_semantic,
            "candidate_count": 4,
        },
    ]
    descriptor = {
        "catalog_id": "base230+v4_1_formal267+v4_2_identity_source.v4_3",
        "catalog_byte_sha256": _semantic(
            {"domain": "v4_3-comparison-source-bytes", "comparison_sources": sources}
        ),
        "catalog_semantic_sha256": _semantic(
            {"domain": "v4_3-comparison-identities", "definition_identity_inventory": inventory}
        ),
        "comparison_sources": sources,
        "definition_identity_inventory": inventory,
    }
    return _ComparisonSnapshot(
        descriptor=descriptor,
        files=(base, formal, identity_source),
    )


def build_comparison_catalog_receipt_v4_3() -> dict[str, Any]:
    snapshot = _fixed_comparison_snapshot()
    return prereg.build_comparison_catalog_receipt_v4_3(descriptor=snapshot.descriptor)


def _artifact_binding(name: str, artifact: Mapping[str, Any]) -> dict[str, str]:
    return prereg.build_artifact_binding_v4_3(name=name, artifact=artifact)


def _state_binding(name: str, artifact: Mapping[str, Any], *, state: str) -> dict[str, str]:
    normalized = validate_cycle_state_v4_1(artifact, expected_state=state)
    return prereg.validate_artifact_binding_v4_3(
        {
            "name": name,
            "byte_sha256": cycle_state_byte_sha256_v4_1(normalized),
            "semantic_sha256": normalized["state_semantic_sha256"],
        },
        expected_name=name,
    )


def _validate_leaf_sources(
    *,
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    candidate_selection_spec: Mapping[str, Any],
    strict_full_a_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    source = prereg.validate_aquant_source_set_receipt_v4_3(aquant_source_set_receipt)
    operator = prereg.validate_operator_semantics_v4_3(operator_semantics)
    comparison = prereg.validate_comparison_catalog_receipt_v4_3(comparison_catalog_receipt)
    selection = prereg.validate_selection_spec_v4_3(
        candidate_selection_spec,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
    )
    strict_source = validate_strict_full_a_source_binding_v4_3(strict_full_a_source_binding)
    code = validate_code_binding_set_v4_3(code_binding_set)
    if selection["publication"]["publication_date"] < CUTOFF_V4_3:
        raise _error("selection preregistration date must not precede fixed cycle cutoff")
    return source, operator, comparison, selection, strict_source, code


def validate_cycle_root_v4_3(
    value: Mapping[str, Any],
    *,
    aquant_source_set_receipt: Mapping[str, Any] | None = None,
    operator_semantics: Mapping[str, Any] | None = None,
    candidate_selection_spec: Mapping[str, Any] | None = None,
    strict_full_a_source_binding: Mapping[str, Any] | None = None,
    code_binding_set: Mapping[str, Any] | None = None,
    v4_2_contract_lock: Mapping[str, Any] | None = None,
    future_source_envelope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "candidate_preregistration_schema_version",
            "market",
            "universe",
            "cutoff",
            "snapshot_id",
            "cycle_id",
            "v4_2_contract_lock",
            "ordered_predecessor_bindings",
            "cycle_root_sha256",
            "artifact_semantic_sha256",
        },
        "v4.3 cycle root",
    )
    if (
        payload["schema_version"] != CYCLE_ROOT_SCHEMA_VERSION_V4_3
        or payload["protocol_version"] != "v4"
        or payload["candidate_preregistration_schema_version"] != prereg.SCHEMA_VERSION
        or payload["market"] != "CN"
        or payload["universe"] != "full_a"
        or payload["cutoff"] != CUTOFF_V4_3
        or payload["snapshot_id"] != SNAPSHOT_ID_V4_3
        or payload["cycle_id"] != CYCLE_ID_V4_3
    ):
        raise _error("cycle root fixed identity mismatch")
    bindings = payload["ordered_predecessor_bindings"]
    expected_names = (
        "selection_spec",
        "strict_source_binding",
        "code_binding_set",
        "future_source_envelope",
        "v4_2_contract_lock",
    )
    if not isinstance(bindings, list) or len(bindings) != len(expected_names):
        raise _error("cycle root predecessor inventory mismatch")
    normalized_bindings = [
        prereg.validate_artifact_binding_v4_3(item, expected_name=name)
        for item, name in zip(bindings, expected_names, strict=True)
    ]
    embedded_v4_2_contract_lock = validate_v4_2_contract_lock_v4_3(
        payload["v4_2_contract_lock"]
    )
    if normalized_bindings[-1] != _artifact_binding(
        "v4_2_contract_lock",
        embedded_v4_2_contract_lock,
    ):
        raise _error("cycle root v4.2 contract lock binding mismatch")
    if v4_2_contract_lock is not None:
        expected_v4_2_contract_lock = validate_v4_2_contract_lock_v4_3(
            v4_2_contract_lock
        )
        if not _canonical_equal(
            embedded_v4_2_contract_lock,
            expected_v4_2_contract_lock,
        ):
            raise _error("cycle root v4.2 contract lock mismatch")
    base = {
        "schema_version": CYCLE_ROOT_SCHEMA_VERSION_V4_3,
        "protocol_version": "v4",
        "candidate_preregistration_schema_version": prereg.SCHEMA_VERSION,
        "market": "CN",
        "universe": "full_a",
        "cutoff": CUTOFF_V4_3,
        "snapshot_id": SNAPSHOT_ID_V4_3,
        "cycle_id": CYCLE_ID_V4_3,
        "v4_2_contract_lock": embedded_v4_2_contract_lock,
        "ordered_predecessor_bindings": normalized_bindings,
    }
    if payload["cycle_root_sha256"] != _semantic(base):
        raise _error("cycle_root_sha256 mismatch")
    if all(
        item is not None
        for item in (
            aquant_source_set_receipt,
            operator_semantics,
            candidate_selection_spec,
            strict_full_a_source_binding,
            code_binding_set,
            future_source_envelope,
        )
    ):
        source = prereg.validate_aquant_source_set_receipt_v4_3(
            aquant_source_set_receipt  # type: ignore[arg-type]
        )
        operator = prereg.validate_operator_semantics_v4_3(
            operator_semantics  # type: ignore[arg-type]
        )
        selection = prereg.validate_selection_spec_v4_3(
            candidate_selection_spec,  # type: ignore[arg-type]
            aquant_source_set_receipt=source,
            operator_semantics=operator,
        )
        strict_source = validate_strict_full_a_source_binding_v4_3(
            strict_full_a_source_binding  # type: ignore[arg-type]
        )
        code = validate_code_binding_set_v4_3(code_binding_set)  # type: ignore[arg-type]
        expected = prereg.build_cycle_root_predecessor_bindings_v4_3(
            selection_spec=selection,
            aquant_source_set_receipt=source,
            operator_semantics=operator,
            strict_source_binding=_artifact_binding("strict_source_binding", strict_source),
            code_binding_set=_artifact_binding("code_binding_set", code),
            future_source_envelope=future_source_envelope,  # type: ignore[arg-type]
            full_a_scope_sha256=FULL_A_SCOPE_SHA256_V4_3,
            full_a_scope_count=FULL_A_SCOPE_COUNT_V4_3,
            serving_inventory_count=SERVING_INVENTORY_COUNT_V4_3,
        )
        expected.append(
            _artifact_binding(
                "v4_2_contract_lock",
                embedded_v4_2_contract_lock,
            )
        )
        if normalized_bindings != expected:
            raise _error("cycle root cross-artifact predecessor mismatch")
    _validate_self(payload, "cycle root")
    return copy.deepcopy(payload)


def build_cycle_root_v4_3(
    *,
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    candidate_selection_spec: Mapping[str, Any],
    strict_full_a_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
    future_source_envelope: Mapping[str, Any],
    v4_2_contract_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = prereg.validate_aquant_source_set_receipt_v4_3(aquant_source_set_receipt)
    operator = prereg.validate_operator_semantics_v4_3(operator_semantics)
    selection = prereg.validate_selection_spec_v4_3(
        candidate_selection_spec,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
    )
    strict_source = validate_strict_full_a_source_binding_v4_3(strict_full_a_source_binding)
    code = validate_code_binding_set_v4_3(code_binding_set)
    contract_lock = validate_v4_2_contract_lock_v4_3(
        v4_2_contract_lock
        if v4_2_contract_lock is not None
        else build_v4_2_contract_lock_v4_3()
    )
    bindings = prereg.build_cycle_root_predecessor_bindings_v4_3(
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        strict_source_binding=_artifact_binding("strict_source_binding", strict_source),
        code_binding_set=_artifact_binding("code_binding_set", code),
        future_source_envelope=future_source_envelope,
        full_a_scope_sha256=FULL_A_SCOPE_SHA256_V4_3,
        full_a_scope_count=FULL_A_SCOPE_COUNT_V4_3,
        serving_inventory_count=SERVING_INVENTORY_COUNT_V4_3,
    )
    bindings.append(_artifact_binding("v4_2_contract_lock", contract_lock))
    base = {
        "schema_version": CYCLE_ROOT_SCHEMA_VERSION_V4_3,
        "protocol_version": "v4",
        "candidate_preregistration_schema_version": prereg.SCHEMA_VERSION,
        "market": "CN",
        "universe": "full_a",
        "cutoff": CUTOFF_V4_3,
        "snapshot_id": SNAPSHOT_ID_V4_3,
        "cycle_id": CYCLE_ID_V4_3,
        "v4_2_contract_lock": contract_lock,
        "ordered_predecessor_bindings": bindings,
    }
    return validate_cycle_root_v4_3(
        _seal({**base, "cycle_root_sha256": _semantic(base)}),
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        candidate_selection_spec=selection,
        strict_full_a_source_binding=strict_source,
        code_binding_set=code,
        v4_2_contract_lock=contract_lock,
        future_source_envelope=future_source_envelope,
    )


def build_candidate_preregistration_bundle_artifacts_v4_3(
    *,
    aquant_source_set_receipt: Mapping[str, Any],
    operator_semantics: Mapping[str, Any],
    comparison_catalog_receipt: Mapping[str, Any],
    candidate_selection_spec: Mapping[str, Any],
    strict_full_a_source_binding: Mapping[str, Any],
    code_binding_set: Mapping[str, Any],
    v4_2_contract_lock: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Rebuild the exact deterministic thirteen-input v4.3 DAG."""

    source, operator, comparison, selection, strict_source, code = _validate_leaf_sources(
        aquant_source_set_receipt=aquant_source_set_receipt,
        operator_semantics=operator_semantics,
        comparison_catalog_receipt=comparison_catalog_receipt,
        candidate_selection_spec=candidate_selection_spec,
        strict_full_a_source_binding=strict_full_a_source_binding,
        code_binding_set=code_binding_set,
    )
    contract_lock = validate_v4_2_contract_lock_v4_3(
        v4_2_contract_lock
        if v4_2_contract_lock is not None
        else build_v4_2_contract_lock_v4_3()
    )
    strict_binding = _artifact_binding("strict_source_binding", strict_source)
    code_binding = _artifact_binding("code_binding_set", code)
    future = prereg.build_future_source_envelope_v4_3(
        cycle_id=CYCLE_ID_V4_3,
        analysis_start=strict_source["analysis_start"],
        cutoff=CUTOFF_V4_3,
        snapshot_id=SNAPSHOT_ID_V4_3,
        snapshot_date=CUTOFF_V4_3,
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        strict_source_binding=strict_binding,
        code_binding_set=code_binding,
        full_a_scope_sha256=FULL_A_SCOPE_SHA256_V4_3,
        full_a_scope_count=FULL_A_SCOPE_COUNT_V4_3,
        serving_inventory_count=SERVING_INVENTORY_COUNT_V4_3,
    )
    collision = prereg.build_definition_identity_collision_audit_v4_3(
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        comparison_catalog_receipt=comparison,
    )
    cycle_root = build_cycle_root_v4_3(
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        candidate_selection_spec=selection,
        strict_full_a_source_binding=strict_source,
        code_binding_set=code,
        future_source_envelope=future,
        v4_2_contract_lock=contract_lock,
    )
    cycle_root_binding = _artifact_binding("cycle_root", cycle_root)
    precommit_source_chain = prereg.build_precommit_source_chain_sha256_v4_3(
        future,
        collision,
    )
    predecessor = build_genesis_cycle_state_v4_1(
        cycle_id=CYCLE_ID_V4_3,
        cycle_root_sha256=cycle_root_binding["semantic_sha256"],
        source_chain_node_sha256=precommit_source_chain,
    )
    predecessor_byte = cycle_state_byte_sha256_v4_1(predecessor)
    orchestration = prereg.build_preregistration_discovery_cycle_v4_3(
        predecessor_state=predecessor,
        predecessor_byte_sha256=predecessor_byte,
        expected_predecessor_byte_sha256=predecessor_byte,
        expected_predecessor_semantic_sha256=predecessor["state_semantic_sha256"],
        future_source_envelope=future,
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        comparison_catalog_receipt=comparison,
        definition_identity_collision_audit=collision,
        cycle_root_binding=cycle_root_binding,
        strict_source_binding=strict_binding,
        code_binding_set=code_binding,
        full_a_scope_sha256=FULL_A_SCOPE_SHA256_V4_3,
        full_a_scope_count=FULL_A_SCOPE_COUNT_V4_3,
        serving_inventory_count=SERVING_INVENTORY_COUNT_V4_3,
    )
    orchestration = prereg.validate_preregistration_discovery_cycle_v4_3(
        orchestration,
        predecessor_state=predecessor,
        predecessor_byte_sha256=predecessor_byte,
        expected_predecessor_byte_sha256=predecessor_byte,
        expected_predecessor_semantic_sha256=predecessor["state_semantic_sha256"],
        future_source_envelope=future,
        selection_spec=selection,
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        comparison_catalog_receipt=comparison,
        definition_identity_collision_audit=collision,
        cycle_root_binding=cycle_root_binding,
        strict_source_binding=strict_binding,
        code_binding_set=code_binding,
        full_a_scope_sha256=FULL_A_SCOPE_SHA256_V4_3,
        full_a_scope_count=FULL_A_SCOPE_COUNT_V4_3,
        serving_inventory_count=SERVING_INVENTORY_COUNT_V4_3,
    )
    return {
        AQUANT_IDEA_SOURCE_SET_RECEIPT_FILENAME_V4_3: source,
        OPERATOR_SEMANTICS_FILENAME_V4_3: operator,
        COMPARISON_CATALOG_RECEIPT_FILENAME_V4_3: comparison,
        CANDIDATE_SELECTION_SPEC_FILENAME_V4_3: selection,
        STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_3: strict_source,
        CODE_BINDING_SET_FILENAME_V4_3: code,
        FUTURE_SOURCE_ENVELOPE_FILENAME_V4_3: future,
        CYCLE_ROOT_FILENAME_V4_3: cycle_root,
        DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_3: collision,
        PRECOMMITTED_STATE_FILENAME_V4_3: predecessor,
        DISCOVERY_SOURCE_NODE_FILENAME_V4_3: copy.deepcopy(orchestration["source_node"]),
        DISCOVERY_STATE_FILENAME_V4_3: copy.deepcopy(orchestration["discovery_state"]),
        PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_3: orchestration,
    }


def validate_candidate_preregistration_bundle_inputs_v4_3(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, Mapping) or set(values) != set(INPUT_FILENAMES_V4_3):
        raise _error("bundle input inventory mismatch")
    normalized = {
        filename: _validate_artifact(filename, values[filename])
        for filename in INPUT_FILENAMES_V4_3
    }
    rebuilt = build_candidate_preregistration_bundle_artifacts_v4_3(
        aquant_source_set_receipt=normalized[AQUANT_IDEA_SOURCE_SET_RECEIPT_FILENAME_V4_3],
        operator_semantics=normalized[OPERATOR_SEMANTICS_FILENAME_V4_3],
        comparison_catalog_receipt=normalized[COMPARISON_CATALOG_RECEIPT_FILENAME_V4_3],
        candidate_selection_spec=normalized[CANDIDATE_SELECTION_SPEC_FILENAME_V4_3],
        strict_full_a_source_binding=normalized[STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_3],
        code_binding_set=normalized[CODE_BINDING_SET_FILENAME_V4_3],
        v4_2_contract_lock=normalized[CYCLE_ROOT_FILENAME_V4_3][
            "v4_2_contract_lock"
        ],
    )
    for filename in INPUT_FILENAMES_V4_3:
        if not _canonical_equal(normalized[filename], rebuilt[filename]):
            raise _error(f"cross-artifact DAG mismatch: {filename}")
    return normalized


def validate_candidate_preregistration_bundle_artifacts_v4_3(
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected_names = {*INPUT_FILENAMES_V4_3, READBACK_REPORT_FILENAME_V4_3}
    if not isinstance(values, Mapping) or set(values) != expected_names:
        raise _error("complete bundle inventory mismatch")
    normalized_inputs = validate_candidate_preregistration_bundle_inputs_v4_3(
        {filename: values[filename] for filename in INPUT_FILENAMES_V4_3}
    )
    report = _validate_report(values[READBACK_REPORT_FILENAME_V4_3])
    bindings = {row["filename"]: row for row in report["artifact_bindings"]}
    for filename in INPUT_FILENAMES_V4_3:
        raw = _canonical_file(normalized_inputs[filename])
        binding = bindings[filename]
        if (
            binding["byte_sha256"] != hashlib.sha256(raw).hexdigest()
            or binding["semantic_sha256"]
            != _artifact_semantic_sha256(filename, normalized_inputs[filename])
            or binding["size_bytes"] != len(raw)
        ):
            raise _error(f"readback artifact byte/semantic binding mismatch: {filename}")
    return {
        **normalized_inputs,
        READBACK_REPORT_FILENAME_V4_3: report,
    }


def _artifact_semantic_sha256(filename: str, value: Mapping[str, Any]) -> str:
    if filename in (PRECOMMITTED_STATE_FILENAME_V4_3, DISCOVERY_STATE_FILENAME_V4_3):
        return _sha256(value.get("state_semantic_sha256"), f"{filename} semantic SHA")
    return _sha256(value.get("artifact_semantic_sha256"), f"{filename} semantic SHA")


def _validate_report(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "filename",
            "cycle_id",
            "publication_phase",
            "intended_destination",
            "exclusive_commit_primitive",
            "exclusive_rename_completed",
            "durability_commit_verified",
            "publication_authority",
            "state_contract",
            "artifact_bindings",
            "side_effects",
            "artifact_semantic_sha256",
        },
        "candidate preregistration readback report",
    )
    if (
        payload["schema_version"] != READBACK_REPORT_SCHEMA_VERSION_V4_3
        or payload["protocol_version"] != "v4"
        or payload["filename"] != READBACK_REPORT_FILENAME_V4_3
        or payload["cycle_id"] != CYCLE_ID_V4_3
        or payload["publication_phase"] != "PRECOMMIT_INTENT_ONLY"
    ):
        raise _error("readback report fixed identity/publication phase mismatch")
    if payload["intended_destination"] != {
        "root_suffix": list(ROOT_SUFFIX_V4_3),
        "directory_name": CYCLE_ID_V4_3,
    }:
        raise _error("readback intended destination mismatch")
    if payload["exclusive_commit_primitive"] != "renameatx_np(RENAME_EXCL)":
        raise _error("readback report exclusive commit primitive mismatch")
    if (
        payload["exclusive_rename_completed"] is not False
        or payload["durability_commit_verified"] is not False
        or payload["publication_authority"] is not False
    ):
        raise _error("staged readback report may not claim publication success")
    if payload["state_contract"] != {
        "precommitted_persisted": True,
        "precommitted_role": "INTRA_BUNDLE_LINEAGE_ONLY",
        "discovery_persisted": True,
        "sole_final_current_state": DISCOVERY,
        "external_pointer_mutation": False,
    }:
        raise _error("readback state contract mismatch")
    bindings = payload["artifact_bindings"]
    if not isinstance(bindings, list) or [
        item.get("filename") for item in bindings if isinstance(item, Mapping)
    ] != list(INPUT_FILENAMES_V4_3):
        raise _error("readback exact artifact inventory/order mismatch")
    for index, item in enumerate(bindings):
        row = _exact(
            item,
            {
                "filename",
                "byte_sha256",
                "semantic_sha256",
                "size_bytes",
                "mode",
                "uid",
                "nlink",
            },
            f"readback artifact binding[{index}]",
        )
        _sha256(row["byte_sha256"], "readback byte SHA")
        _sha256(row["semantic_sha256"], "readback semantic SHA")
        _positive_int(row["size_bytes"], "readback size")
        if row["mode"] != 0o600 or row["uid"] != os.getuid() or row["nlink"] != 1:
            raise _error("readback artifact owner/private binding mismatch")
    if payload["side_effects"] != prereg.SIDE_EFFECT_FLAGS:
        raise _error("readback report side effects must remain exact false")
    _validate_self(payload, "candidate preregistration readback report")
    return copy.deepcopy(payload)


def _build_readback_report(
    *,
    run_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if run_id != CYCLE_ID_V4_3 or tuple(artifacts) != INPUT_FILENAMES_V4_3:
        raise _error("readback builder deterministic inventory/run identity mismatch")
    if len(artifact_bindings) != len(INPUT_FILENAMES_V4_3):
        raise _error("readback builder artifact binding count mismatch")
    rows: list[dict[str, Any]] = []
    for filename, item in zip(INPUT_FILENAMES_V4_3, artifact_bindings, strict=True):
        row = _exact(
            item,
            {"filename", "byte_sha256", "size_bytes", "mode", "uid", "nlink"},
            f"private I/O binding {filename}",
        )
        if row["filename"] != filename:
            raise _error("private I/O artifact binding order mismatch")
        rows.append(
            {
                **copy.deepcopy(row),
                "semantic_sha256": _artifact_semantic_sha256(filename, artifacts[filename]),
            }
        )
    return _validate_report(
        _seal(
            {
                "schema_version": READBACK_REPORT_SCHEMA_VERSION_V4_3,
                "protocol_version": "v4",
                "filename": READBACK_REPORT_FILENAME_V4_3,
                "cycle_id": CYCLE_ID_V4_3,
                "publication_phase": "PRECOMMIT_INTENT_ONLY",
                "intended_destination": {
                    "root_suffix": list(ROOT_SUFFIX_V4_3),
                    "directory_name": CYCLE_ID_V4_3,
                },
                "exclusive_commit_primitive": "renameatx_np(RENAME_EXCL)",
                "exclusive_rename_completed": False,
                "durability_commit_verified": False,
                "publication_authority": False,
                "state_contract": {
                    "precommitted_persisted": True,
                    "precommitted_role": "INTRA_BUNDLE_LINEAGE_ONLY",
                    "discovery_persisted": True,
                    "sole_final_current_state": DISCOVERY,
                    "external_pointer_mutation": False,
                },
                "artifact_bindings": rows,
                "side_effects": copy.deepcopy(prereg.SIDE_EFFECT_FLAGS),
            }
        )
    )


_GENERIC_SCHEMA_BY_FILENAME_V4_3 = {
    CANDIDATE_SELECTION_SPEC_FILENAME_V4_3: prereg.SELECTION_SPEC_SCHEMA_VERSION,
    FUTURE_SOURCE_ENVELOPE_FILENAME_V4_3: prereg.SOURCE_ENVELOPE_SCHEMA_VERSION,
    DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_3: (
        prereg.DEFINITION_IDENTITY_COLLISION_AUDIT_SCHEMA_VERSION
    ),
    DISCOVERY_SOURCE_NODE_FILENAME_V4_3: prereg.DISCOVERY_SOURCE_NODE_SCHEMA_VERSION,
    PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_3: prereg.ORCHESTRATION_SCHEMA_VERSION,
}


def _validate_generic_pure_artifact(filename: str, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    if (
        payload.get("schema_version") != _GENERIC_SCHEMA_BY_FILENAME_V4_3[filename]
        or payload.get("protocol_version") != "v4"
    ):
        raise _error(f"{filename} schema/protocol mismatch")
    _validate_self(payload, filename)
    return payload


def _validate_artifact(filename: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if filename == AQUANT_IDEA_SOURCE_SET_RECEIPT_FILENAME_V4_3:
        return prereg.validate_aquant_source_set_receipt_v4_3(value)
    if filename == OPERATOR_SEMANTICS_FILENAME_V4_3:
        return prereg.validate_operator_semantics_v4_3(value)
    if filename == COMPARISON_CATALOG_RECEIPT_FILENAME_V4_3:
        return prereg.validate_comparison_catalog_receipt_v4_3(value)
    if filename == STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_3:
        return validate_strict_full_a_source_binding_v4_3(value)
    if filename == CODE_BINDING_SET_FILENAME_V4_3:
        return validate_code_binding_set_v4_3(value)
    if filename == CYCLE_ROOT_FILENAME_V4_3:
        return validate_cycle_root_v4_3(value)
    if filename in (PRECOMMITTED_STATE_FILENAME_V4_3, DISCOVERY_STATE_FILENAME_V4_3):
        return validate_cycle_state_v4_1(value)
    if filename == READBACK_REPORT_FILENAME_V4_3:
        return _validate_report(value)
    if filename in _GENERIC_SCHEMA_BY_FILENAME_V4_3:
        return _validate_generic_pure_artifact(filename, value)
    raise _error(f"unknown v4.3 bundle artifact: {filename}")


def _canonical_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _canonical_file(left) == _canonical_file(right)


def candidate_preregistration_bundle_contract_v4_3() -> PrivateBundleContract:
    return PrivateBundleContract(
        root_suffix=ROOT_SUFFIX_V4_3,
        input_filenames=INPUT_FILENAMES_V4_3,
        readback_report_filename=READBACK_REPORT_FILENAME_V4_3,
        canonicalize=_canonical_file,
        validate_artifact=_validate_artifact,
        validate_complete=validate_candidate_preregistration_bundle_artifacts_v4_3,
        build_readback_report=_build_readback_report,
    )


def _preflight_private_root(value: str | os.PathLike[str]) -> Path:
    root = _absolute_path(os.fspath(value), "private_root")
    if tuple(root.parts[-len(ROOT_SUFFIX_V4_3) :]) != ROOT_SUFFIX_V4_3:
        raise _error("private_root fixed suffix mismatch")
    current = Path("/")
    for part in root.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise _error("private_root must pre-exist; bundle publication never creates it") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise _error(f"private_root directory chain is unsafe: {current}")
    leaf = os.lstat(root)
    if int(leaf.st_uid) != os.getuid() or stat.S_IMODE(leaf.st_mode) != 0o700:
        raise _error("private_root must be exact owner mode 0700")
    return root


def _repository_root(value: str | os.PathLike[str]) -> Path:
    root = _absolute_path(os.fspath(value), "repository_root")
    _assert_owned_nofollow_chain(root, boundary=root, include_target=True, label="repository root")
    return root


def _observe_code_bindings(
    *, repository_root: Path, code_bindings: Sequence[Mapping[str, Any]]
) -> tuple[_StableFile, ...]:
    if not isinstance(code_bindings, Sequence) or isinstance(
        code_bindings, (str, bytes, bytearray)
    ) or len(code_bindings) != len(CODE_BINDING_PATHS_V4_3):
        raise _error("code binding descriptor inventory mismatch")
    observations: list[_StableFile] = []
    for item, relative in zip(code_bindings, CODE_BINDING_PATHS_V4_3, strict=True):
        row = _exact(item, {"relative_path", *_DESCRIPTOR_FIELDS}, f"code input {relative}")
        if row["relative_path"] != relative:
            raise _error("code input relative path/order mismatch")
        observations.append(
            _observe_descriptor(
                {key: row[key] for key in _DESCRIPTOR_FIELDS},
                expected_path=repository_root / relative,
                boundary=repository_root,
                label=f"code binding {relative}",
            )
        )
    return tuple(observations)


def _freeze_git_objects(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise _error("aquant_git_objects must be an exact path-keyed mapping")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if type(item) is bytes:
            frozen[key] = bytes(item)
        elif isinstance(item, Mapping) and set(item) == {"content"} and type(item["content"]) is bytes:
            frozen[key] = {"content": bytes(item["content"])}
        else:
            raise _error("aquant_git_objects values must be bytes or exact content rows")
    return frozen


def publish_candidate_preregistration_bundle_v4_3(
    *,
    private_root: Path,
    repository_root: Path,
    preregistered_at: str,
    aquant_git_objects: Mapping[str, bytes],
    strict_source_binding: Mapping[str, Any],
    code_bindings: Sequence[Mapping[str, Any]],
    protected_bindings: Sequence[Mapping[str, Any]],
    revalidate_inputs: Callable[[], None] | None = None,
    _test_fault_hook: Any = None,
    _test_race_hook: Any = None,
) -> dict[str, Any]:
    """Build and publish the fixed v4.3 cycle exactly once.

    The optional callback is an additive publication-lock check for callers
    that can reopen the pinned Git objects.  All file-backed source, code,
    comparison, and protected descriptors are revalidated here regardless.
    """

    root = _preflight_private_root(private_root)
    repo = _repository_root(repository_root)
    if revalidate_inputs is not None and not callable(revalidate_inputs):
        raise _error("revalidate_inputs must be callable or None")
    if _test_race_hook is not None and not callable(_test_race_hook):
        raise _error("_test_race_hook must be callable or None")
    frozen_git_objects = _freeze_git_objects(aquant_git_objects)
    frozen_code_bindings = copy.deepcopy(list(code_bindings))
    frozen_protected_bindings = copy.deepcopy(list(protected_bindings))
    strict_source = validate_strict_full_a_source_binding_v4_3(strict_source_binding)
    pointer = Path(strict_source["backend_binding"]["latest_pointer"]["absolute_path"])
    if pointer.parents[3] != repo:
        raise _error("strict source binding does not belong to repository_root")

    source_before = revalidate_strict_full_a_source_binding_v4_3(
        strict_source,
        repository_root=repo,
    )["snapshot"]
    code_before = _observe_code_bindings(
        repository_root=repo,
        code_bindings=frozen_code_bindings,
    )
    protected_before = _observe_protected_bindings(
        repository_root=repo,
        protected_bindings=frozen_protected_bindings,
    )
    comparison_before = _fixed_comparison_snapshot()
    v4_2_contract_lock_before = _observe_v4_2_contract_lock_snapshot(
        repository_root=repo
    )

    source_receipt = prereg.build_aquant_source_set_receipt_v4_3(
        aquant_git_objects=frozen_git_objects
    )
    operator = prereg.build_operator_semantics_v4_3()
    comparison = prereg.build_comparison_catalog_receipt_v4_3(
        descriptor=comparison_before.descriptor
    )
    selection = prereg.build_selection_spec_v4_3(
        aquant_source_set_receipt=source_receipt,
        operator_semantics=operator,
        preregistered_at=preregistered_at,
    )
    code = build_code_binding_set_v4_3(
        repository_root=repo,
        code_bindings=frozen_code_bindings,
    )
    artifacts = build_candidate_preregistration_bundle_artifacts_v4_3(
        aquant_source_set_receipt=source_receipt,
        operator_semantics=operator,
        comparison_catalog_receipt=comparison,
        candidate_selection_spec=selection,
        strict_full_a_source_binding=strict_source,
        code_binding_set=code,
        v4_2_contract_lock=v4_2_contract_lock_before.artifact,
    )

    def locked_revalidation() -> None:
        locked_source = revalidate_strict_full_a_source_binding_v4_3(
            strict_source,
            repository_root=repo,
        )["snapshot"]
        if locked_source != source_before:
            raise _error("strict source changed before exclusive commit")
        locked_code = _observe_code_bindings(
            repository_root=repo,
            code_bindings=frozen_code_bindings,
        )
        if locked_code != code_before:
            raise _error("code inputs changed before exclusive commit")
        revalidate_code_binding_set_v4_3(
            repository_root=repo,
            code_bindings=frozen_code_bindings,
            value=code,
        )
        locked_protected = _observe_protected_bindings(
            repository_root=repo,
            protected_bindings=frozen_protected_bindings,
        )
        if locked_protected != protected_before:
            raise _error("protected controls changed before exclusive commit")
        locked_comparison = _fixed_comparison_snapshot()
        if locked_comparison != comparison_before:
            raise _error("comparison sources changed before exclusive commit")
        locked_v4_2_contract_lock = _observe_v4_2_contract_lock_snapshot(
            repository_root=repo
        )
        if locked_v4_2_contract_lock != v4_2_contract_lock_before:
            raise _error("v4.2 contract lock changed before exclusive commit")
        embedded_contract_lock = artifacts[CYCLE_ROOT_FILENAME_V4_3][
            "v4_2_contract_lock"
        ]
        if not _canonical_equal(
            locked_v4_2_contract_lock.artifact,
            embedded_contract_lock,
        ):
            raise _error("v4.2 contract lock changed before exclusive commit")
        rebuilt_comparison = prereg.build_comparison_catalog_receipt_v4_3(
            descriptor=locked_comparison.descriptor
        )
        if not _canonical_equal(rebuilt_comparison, comparison):
            raise _error("comparison receipt changed before exclusive commit")
        rebuilt_source = prereg.build_aquant_source_set_receipt_v4_3(
            aquant_git_objects=frozen_git_objects
        )
        if not _canonical_equal(rebuilt_source, source_receipt):
            raise _error("A_quant Git-object receipt changed before exclusive commit")
        if revalidate_inputs is not None:
            revalidate_inputs()

    def final_race_revalidation() -> None:
        if _test_race_hook is not None:
            _test_race_hook()
        locked_revalidation()

    published = publish_private_bundle(
        private_root=root,
        run_id=CYCLE_ID_V4_3,
        artifacts=artifacts,
        contract=candidate_preregistration_bundle_contract_v4_3(),
        revalidate_inputs=locked_revalidation,
        _test_fault_hook=_test_fault_hook,
        _test_race_hook=final_race_revalidation,
    )
    report_descriptor = published["artifact_descriptors"][READBACK_REPORT_FILENAME_V4_3]
    report = published["readback_report"]
    return {
        **published,
        "protocol_version": "v4",
        "evidence_contract_version": "v4.3",
        "cycle_id": CYCLE_ID_V4_3,
        "publication_phase": "COMMITTED",
        "exclusive_rename_completed": True,
        "durability_commit_verified": True,
        "publication_authority": True,
        "readback_report_path": report_descriptor["absolute_path"],
        "readback_report_byte_sha256": report_descriptor["byte_sha256"],
        "readback_report_semantic_sha256": report["artifact_semantic_sha256"],
        "side_effects": copy.deepcopy(prereg.SIDE_EFFECT_FLAGS),
    }


def readback_candidate_preregistration_bundle_files_v4_3(
    *,
    bundle_path: Path,
    expected_readback_report_byte_sha256: str,
    expected_readback_report_semantic_sha256: str,
) -> dict[str, Any]:
    path = _absolute_path(os.fspath(bundle_path), "bundle_path")
    if path.name != CYCLE_ID_V4_3:
        raise _error("bundle_path must name the deterministic v4.3 cycle")
    expected_byte = _sha256(
        expected_readback_report_byte_sha256,
        "expected readback report byte SHA",
    )
    expected_semantic = _sha256(
        expected_readback_report_semantic_sha256,
        "expected readback report semantic SHA",
    )
    result = readback_private_bundle(
        path,
        contract=candidate_preregistration_bundle_contract_v4_3(),
    )
    descriptor = result["artifact_descriptors"][READBACK_REPORT_FILENAME_V4_3]
    report = result["readback_report"]
    report_path = path / READBACK_REPORT_FILENAME_V4_3
    if descriptor["absolute_path"] != str(report_path):
        raise _error("readback report absolute path mismatch")
    if descriptor["byte_sha256"] != expected_byte:
        raise _error("expected readback report byte SHA mismatch")
    if report["artifact_semantic_sha256"] != expected_semantic:
        raise _error("expected readback report semantic SHA mismatch")
    return {
        **result,
        "readback_report_path": str(report_path),
        "readback_report_byte_sha256": expected_byte,
        "readback_report_semantic_sha256": expected_semantic,
        "expected_hashes_verified": True,
    }


def readback_candidate_preregistration_bundle_v4_3(
    *,
    bundle_path: Path,
    expected_readback_report_byte_sha256: str,
    expected_readback_report_semantic_sha256: str,
) -> dict[str, Any]:
    """Reopen one explicit absolute bundle under exact report byte/semantic CAS."""

    return readback_candidate_preregistration_bundle_files_v4_3(
        bundle_path=bundle_path,
        expected_readback_report_byte_sha256=expected_readback_report_byte_sha256,
        expected_readback_report_semantic_sha256=expected_readback_report_semantic_sha256,
    )
