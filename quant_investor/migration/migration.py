"""Pure pre-CAS migration receipts and permanent cutover markers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Final

from ..contracts import (
    artifact_byte_sha256,
    canonical_json_bytes,
    get_contract,
    seal_artifact,
    validate_artifact,
)
from .canonical import (
    SHA256_RE,
    canonical_relative_path,
    parse_json_bytes,
    read_stable_regular_file,
    sha256_bytes,
    write_idempotent_bytes,
)
from .custody import (
    ARCHIVE_PLAN_CONTRACT_SHA256,
    ARCHIVE_PLAN_KIND,
    artifact_exact_ref,
    validate_authority_archive_plan,
)
from .errors import (
    ARCHIVE_PLAN_INVALID,
    PERMANENT_MARKER_PRESENT,
    RECEIPT_INVALID,
    RULES_NON_CANONICAL,
    RULES_SCHEMA_INVALID,
    SOURCE_TARGET_COLLISION,
    SYMLINK_REFUSED,
    UNIFIED_ACTIVE_PRESENT,
    UnifiedCutoverError,
)
from .resolver import INVENTORY_CONTRACT_SHA256, INVENTORY_KIND
from .rules import (
    ACTIVE_AUTHORITY,
    ACTIVE_CALLER,
    BASELINE_CUSTODY_FACTS_SHA256,
    CUSTODY_ONLY,
    INDEPENDENT_SOURCE,
    LEGACY_INACTIVE,
    NON_AUTHORITY_SHADOW,
    RULES_RELATIVE_PATH,
    SOURCE_TO_TARGET_CONTRACT_SHA256,
    SOURCE_TO_TARGET_KIND,
    SOURCE_TO_TARGET_RELATIVE_PATH,
    CutoverRules,
    load_rules,
    path_matches_glob,
    pointer_filename_matches,
)
from quant_investor.system_authority import (
    ACTIVE_POINTER_PATH,
    EMPTY_POINTER_SHA256,
    MIGRATION_MARKER_PATH,
)

MIGRATION_RECEIPT_KIND: Final = "system.migration.receipt"
MIGRATION_RECEIPT_CONTRACT_SHA256: Final = get_contract(MIGRATION_RECEIPT_KIND).contract_sha256
PERMANENT_MARKER_KIND: Final = "system.migration.complete"
PERMANENT_MARKER_CONTRACT_SHA256: Final = get_contract(PERMANENT_MARKER_KIND).contract_sha256

MIGRATION_RECEIPT_PAYLOAD_FIELDS: Final = frozenset(
    {
        "archive_plan_ref",
        "blocker_codes",
        "cas_performed",
        "cutover_id",
        "expected_active_pointer_sha256",
        "inventory_ref",
        "migration_receipt_id",
        "permanent_marker_path",
        "rules_ref",
        "source_to_target",
        "source_to_target_rules_ref",
        "status",
        "summary",
        "target_active_pointer_path",
        "target_active_pointer_ref",
        "target_generation_id",
        "target_generation_manifest_path",
        "target_generation_manifest_ref",
        "target_release_manifest_ref",
        "write_performed",
    }
)
PERMANENT_MARKER_PAYLOAD_FIELDS: Final = frozenset(
    {
        "active_pointer_ref",
        "archive_plan_ref",
        "blocker_codes",
        "cutover_id",
        "generation_id",
        "generation_manifest_ref",
        "inventory_ref",
        "legacy_replay_refused",
        "marker_id",
        "migration_receipt_ref",
        "migration_replay_refused",
        "permanent_marker_path",
        "status",
    }
)
_CUTOVER_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_GENERATION_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_TARGET_KIND_RE: Final = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ACTIVE_POINTER_FIELDS: Final = frozenset(
    {
        "activated_at",
        "generation_id",
        "manifest_sha256",
        "os_actor",
        "previous_pointer_sha256",
    }
)


@dataclass(frozen=True)
class SourceTargetRule:
    classification: str
    source_glob: str
    action: str
    target_kind: str | None
    target_extension: str | None


@dataclass(frozen=True)
class LoadedSourceTargetRules:
    relative_path: str
    raw: bytes
    mappings: tuple[SourceTargetRule, ...]


def _identity(kind: str, body: Mapping[str, Any], *, prefix: str) -> str:
    preimage = {"domain": "myquant-migration-identity", "kind": kind, "payload": dict(body)}
    return prefix + sha256_bytes(canonical_json_bytes(preimage))


def _validate_cutover_id(value: Any) -> str:
    if type(value) is not str or _CUTOVER_ID_RE.fullmatch(value) is None:
        raise UnifiedCutoverError(RECEIPT_INVALID, "cutover_id is not canonical")
    return value


def _load_source_target_rules(root: Path, relative_path: str) -> LoadedSourceTargetRules:
    relative = canonical_relative_path(relative_path, label="source-to-target rules path")
    try:
        raw = read_stable_regular_file(
            root / relative,
            label="source-to-target rules",
            max_bytes=1024 * 1024,
        )
        document = parse_json_bytes(raw, label="source-to-target rules", require_canonical=True)
    except UnifiedCutoverError as exc:
        raise UnifiedCutoverError(
            RULES_NON_CANONICAL, "source-to-target rules are not canonical"
        ) from exc
    if type(document) is not dict or set(document) != {
        "kind",
        "contract_sha256",
        "mappings",
    }:
        raise UnifiedCutoverError(RULES_SCHEMA_INVALID, "source-to-target fields are not exact")
    if (
        document["kind"] != SOURCE_TO_TARGET_KIND
        or document["contract_sha256"] != SOURCE_TO_TARGET_CONTRACT_SHA256
        or type(document["mappings"]) is not list
    ):
        raise UnifiedCutoverError(
            RULES_SCHEMA_INVALID, "source-to-target kind or contract SHA-256 is invalid"
        )
    expected_actions = {
        ACTIVE_AUTHORITY: "MIGRATE_AND_ARCHIVE",
        ACTIVE_CALLER: "SOURCE_BINDING_ONLY",
        NON_AUTHORITY_SHADOW: "RECORD_ONLY",
        INDEPENDENT_SOURCE: "PRESERVE_INDEPENDENT",
        LEGACY_INACTIVE: "RECORD_ONLY",
        CUSTODY_ONLY: "CUSTODY_RECORD_ONLY",
    }
    mappings: list[SourceTargetRule] = []
    keys: list[tuple[str, str]] = []
    for index, raw_row in enumerate(document["mappings"]):
        if type(raw_row) is not dict or set(raw_row) != {
            "action",
            "classification",
            "source_glob",
            "target_extension",
            "target_kind",
        }:
            raise UnifiedCutoverError(
                RULES_SCHEMA_INVALID, f"source-to-target mapping {index} fields are not exact"
            )
        classification = raw_row["classification"]
        source_glob = raw_row["source_glob"]
        action = raw_row["action"]
        target_kind = raw_row["target_kind"]
        target_extension = raw_row["target_extension"]
        if (
            classification not in expected_actions
            or action != expected_actions[classification]
            or type(source_glob) is not str
            or not source_glob
        ):
            raise UnifiedCutoverError(
                RULES_SCHEMA_INVALID, f"source-to-target mapping {index} is invalid"
            )
        if classification == ACTIVE_AUTHORITY:
            if (
                type(target_kind) is not str
                or _TARGET_KIND_RE.fullmatch(target_kind) is None
                or target_extension != "SOURCE"
            ):
                raise UnifiedCutoverError(
                    RULES_SCHEMA_INVALID, "authority mapping target is invalid"
                )
        elif target_kind is not None or target_extension is not None:
            raise UnifiedCutoverError(
                RULES_SCHEMA_INVALID, "non-authority mapping must have null target fields"
            )
        keys.append((classification, source_glob))
        mappings.append(
            SourceTargetRule(
                classification,
                source_glob,
                action,
                target_kind,
                target_extension,
            )
        )
    if keys != sorted(set(keys)):
        raise UnifiedCutoverError(
            RULES_SCHEMA_INVALID, "source-to-target mappings must be sorted and unique"
        )
    covered = {mapping.classification for mapping in mappings}
    if covered != set(expected_actions):
        raise UnifiedCutoverError(
            RULES_SCHEMA_INVALID, "source-to-target rules omit a custody classification"
        )
    return LoadedSourceTargetRules(relative, raw, tuple(mappings))


def load_source_target_rules(
    workspace_root: str | os.PathLike[str],
    relative_path: str = SOURCE_TO_TARGET_RELATIVE_PATH,
) -> LoadedSourceTargetRules:
    return _load_source_target_rules(Path(workspace_root).resolve(strict=True), relative_path)


def _assert_missing(root: Path, relative_path: str, *, code: str, label: str) -> None:
    path = root.joinpath(*PurePosixPath(relative_path).parts)
    current = root
    for part in PurePosixPath(relative_path).parts[:-1]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise UnifiedCutoverError(SYMLINK_REFUSED, f"{label} parent is a symlink")
        if not stat.S_ISDIR(metadata.st_mode):
            raise UnifiedCutoverError(code, f"{label} parent is not a directory")
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise UnifiedCutoverError(SYMLINK_REFUSED, f"{label} path is a symlink")
    raise UnifiedCutoverError(code, f"{label} already exists; cutover replay refused")


def assert_pre_cas_cutover_state(
    workspace_root: str | os.PathLike[str],
    *,
    rules_path: str | os.PathLike[str] = RULES_RELATIVE_PATH,
) -> None:
    """Refuse every migration replay once either unified sentinel exists."""

    root = Path(workspace_root).resolve(strict=True)
    rules = load_rules(root, rules_path).rules
    _assert_missing(
        root,
        rules.unified_layout.active_pointer,
        code=UNIFIED_ACTIVE_PRESENT,
        label="unified active pointer",
    )
    _assert_missing(
        root,
        rules.unified_layout.permanent_marker,
        code=PERMANENT_MARKER_PRESENT,
        label="permanent migration marker",
    )


def _validated_artifact(
    value: Mapping[str, Any] | bytes,
    *,
    expected_kind: str,
    label: str,
) -> dict[str, Any]:
    try:
        return validate_artifact(value, expected_kind=expected_kind)
    except (TypeError, ValueError) as exc:
        raise UnifiedCutoverError(RECEIPT_INVALID, f"{label} artifact is invalid") from exc


def _validated_active_pointer(value: Mapping[str, Any] | bytes) -> dict[str, Any]:
    try:
        if type(value) is bytes:
            document = parse_json_bytes(
                value, label="target active pointer", require_canonical=True
            )
        elif type(value) is dict:
            canonical_json_bytes(value)
            document = dict(value)
        else:
            raise TypeError("pointer must be object or bytes")
    except (TypeError, ValueError, UnifiedCutoverError) as exc:
        raise UnifiedCutoverError(RECEIPT_INVALID, "target active pointer is invalid") from exc
    if type(document) is not dict or set(document) != _ACTIVE_POINTER_FIELDS:
        raise UnifiedCutoverError(RECEIPT_INVALID, "active pointer fields are not exact")
    for key in ("generation_id", "manifest_sha256"):
        value_at_key = document[key]
        if type(value_at_key) is not str or SHA256_RE.fullmatch(value_at_key) is None:
            raise UnifiedCutoverError(RECEIPT_INVALID, f"active pointer {key} is invalid")
    if document["previous_pointer_sha256"] != EMPTY_POINTER_SHA256:
        raise UnifiedCutoverError(
            RECEIPT_INVALID, "first unified pointer must bind the empty prevalue"
        )
    if (
        type(document["activated_at"]) is not str
        or type(document["os_actor"]) is not str
        or not document["os_actor"].strip()
    ):
        raise UnifiedCutoverError(RECEIPT_INVALID, "active pointer metadata is invalid")
    return document


def build_initial_active_pointer(
    target_generation_manifest: Mapping[str, Any] | bytes,
    *,
    activated_at: str,
    os_actor: str,
) -> dict[str, Any]:
    """Build exact first-pointer content without consulting a clock or OS actor."""

    manifest = _validated_artifact(
        target_generation_manifest,
        expected_kind="system.generation_manifest",
        label="target generation manifest",
    )
    payload = manifest["payload"]
    if (
        payload.get("generation_state") != "OPERATIONAL"
        or payload.get("migration_receipt_ref") is not None
        or payload.get("migration_marker_ref") is not None
    ):
        raise UnifiedCutoverError(
            RECEIPT_INVALID, "initial target must be an acyclic operational generation"
        )
    pointer = {
        "generation_id": manifest["semantic_sha256"],
        "manifest_sha256": artifact_byte_sha256(manifest),
        "previous_pointer_sha256": EMPTY_POINTER_SHA256,
        "activated_at": activated_at,
        "os_actor": os_actor,
    }
    return _validated_active_pointer(pointer)


def _active_pointer_ref(pointer: Mapping[str, Any]) -> dict[str, str]:
    raw = canonical_json_bytes(dict(pointer))
    return {
        "generation_id": pointer["generation_id"],
        "manifest_sha256": pointer["manifest_sha256"],
        "byte_sha256": sha256_bytes(raw),
    }


def _inventory(value: Mapping[str, Any] | bytes) -> dict[str, Any]:
    try:
        document = validate_artifact(
            value,
            expected_kind=INVENTORY_KIND,
            expected_contract_sha256=INVENTORY_CONTRACT_SHA256,
        )
    except (TypeError, ValueError) as exc:
        raise UnifiedCutoverError(RECEIPT_INVALID, "inventory artifact is invalid") from exc
    if (
        document["payload"].get("status") != "COMPLETE"
        or document["payload"].get("summary", {}).get("blocker_codes") != []
    ):
        raise UnifiedCutoverError(RECEIPT_INVALID, "inventory is not blocker-free COMPLETE")
    return document


def _exact_rules_ref(relative_path: str, raw: bytes) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "byte_sha256": sha256_bytes(raw),
        "bytes": len(raw),
    }


def _extension(relative_path: str) -> str:
    suffix = PurePosixPath(relative_path).suffix.lower().lstrip(".")
    if not suffix:
        return "bin"
    if not suffix.isalnum() or len(suffix) > 16:
        raise UnifiedCutoverError(
            SOURCE_TARGET_COLLISION, f"source extension is not canonical: {relative_path}"
        )
    return suffix


def _matching_mapping(
    row: Mapping[str, Any], mappings: Sequence[SourceTargetRule]
) -> SourceTargetRule:
    matches = [
        mapping
        for mapping in mappings
        if mapping.classification == row.get("classification")
        and path_matches_glob(str(row.get("relative_path")), mapping.source_glob)
    ]
    if len(matches) != 1:
        raise UnifiedCutoverError(
            SOURCE_TARGET_COLLISION,
            f"source has {len(matches)} mapping rules: {row.get('relative_path')}",
        )
    return matches[0]


def _source_to_target_rows(
    inventory: Mapping[str, Any],
    archive_plan: Mapping[str, Any],
    *,
    rules: CutoverRules,
    mappings: Sequence[SourceTargetRule],
) -> list[dict[str, Any]]:
    archive_by_source = {
        row["source_relative_path"]: row for row in archive_plan["payload"]["entries"]
    }
    result: list[dict[str, Any]] = []
    archive_targets: set[str] = set()
    object_targets: dict[str, str] = {}
    pointer_targets: dict[str, str] = {}
    for source in inventory["payload"]["files"]:
        if type(source) is not dict:
            raise UnifiedCutoverError(RECEIPT_INVALID, "inventory source row is invalid")
        relative = canonical_relative_path(source.get("relative_path"), label="source path")
        mapping = _matching_mapping(source, mappings)
        archive_relative: str | None = None
        object_relative: str | None = None
        pointer_history: str | None = None
        if mapping.classification == ACTIVE_AUTHORITY:
            archive = archive_by_source.get(relative)
            if archive is None or archive.get("source_byte_sha256") != source.get("byte_sha256"):
                raise UnifiedCutoverError(
                    ARCHIVE_PLAN_INVALID, f"archive plan omits authority source {relative}"
                )
            archive_relative = archive["archive_relative_path"]
            extension = _extension(relative)
            object_relative = (
                f"{rules.unified_layout.object_root}/{mapping.target_kind}/"
                f"{source['byte_sha256']}.{extension}"
            )
            if extension == "json" and pointer_filename_matches(
                PurePosixPath(relative).name,
                (
                    *rules.pointer_filename_rules["active"],
                    *rules.pointer_filename_rules["reachable"],
                ),
            ):
                pointer_history = (
                    f"{rules.unified_layout.pointer_history_root}/" f"{source['byte_sha256']}.json"
                )
        elif relative in archive_by_source:
            raise UnifiedCutoverError(
                ARCHIVE_PLAN_INVALID, f"archive plan copies non-authority source {relative}"
            )

        if archive_relative is not None:
            if archive_relative in archive_targets:
                raise UnifiedCutoverError(
                    SOURCE_TARGET_COLLISION, f"archive target collision: {archive_relative}"
                )
            archive_targets.add(archive_relative)
        if object_relative is not None:
            previous = object_targets.setdefault(object_relative, source["byte_sha256"])
            if previous != source["byte_sha256"]:
                raise UnifiedCutoverError(
                    SOURCE_TARGET_COLLISION, f"object target collision: {object_relative}"
                )
        if pointer_history is not None:
            previous = pointer_targets.setdefault(pointer_history, source["byte_sha256"])
            if previous != source["byte_sha256"]:
                raise UnifiedCutoverError(
                    SOURCE_TARGET_COLLISION, f"pointer history collision: {pointer_history}"
                )
        result.append(
            {
                "action": mapping.action,
                "archive_relative_path": archive_relative,
                "classification": mapping.classification,
                "object_relative_path": object_relative,
                "pointer_history_relative_path": pointer_history,
                "source_byte_sha256": source["byte_sha256"],
                "source_bytes": source["bytes"],
                "source_relative_path": relative,
            }
        )
    result.sort(key=lambda row: row["source_relative_path"])
    if set(archive_by_source) != {
        row["source_relative_path"] for row in result if row["classification"] == ACTIVE_AUTHORITY
    }:
        raise UnifiedCutoverError(ARCHIVE_PLAN_INVALID, "archive plan authority set mismatch")
    return result


def build_pre_cas_migration_receipt(
    workspace_root: str | os.PathLike[str],
    inventory: Mapping[str, Any] | bytes,
    archive_plan: Mapping[str, Any] | bytes,
    target_active_pointer: Mapping[str, Any] | bytes,
    target_generation_manifest: Mapping[str, Any] | bytes,
    *,
    cutover_id: str,
    created_at: str,
    rules_path: str | os.PathLike[str] = RULES_RELATIVE_PATH,
) -> dict[str, Any]:
    """Build one deterministic, write-free receipt before the pointer CAS."""

    root = Path(workspace_root).resolve(strict=True)
    assert_pre_cas_cutover_state(root, rules_path=rules_path)
    loaded_rules = load_rules(root, rules_path)
    rules = loaded_rules.rules
    normalized_cutover = _validate_cutover_id(cutover_id)
    normalized_inventory = _inventory(inventory)
    normalized_plan = validate_authority_archive_plan(archive_plan)
    if normalized_plan["payload"]["cutover_id"] != normalized_cutover or normalized_plan["payload"][
        "inventory_ref"
    ] != artifact_exact_ref(normalized_inventory):
        raise UnifiedCutoverError(RECEIPT_INVALID, "archive plan scope mismatch")
    pointer = _validated_active_pointer(target_active_pointer)
    manifest = _validated_artifact(
        target_generation_manifest,
        expected_kind="system.generation_manifest",
        label="target generation manifest",
    )
    manifest_payload = manifest["payload"]
    if (
        manifest_payload.get("generation_state") != "OPERATIONAL"
        or manifest_payload.get("migration_receipt_ref") is not None
        or manifest_payload.get("migration_marker_ref") is not None
    ):
        raise UnifiedCutoverError(
            RECEIPT_INVALID, "target must be an acyclic operational generation"
        )
    release_ref = manifest_payload.get("release_manifest_ref")
    if type(release_ref) is not dict:
        raise UnifiedCutoverError(RECEIPT_INVALID, "target release binding is absent")
    generation_id = pointer.get("generation_id")
    if type(generation_id) is not str or _GENERATION_ID_RE.fullmatch(generation_id) is None:
        raise UnifiedCutoverError(RECEIPT_INVALID, "target generation_id is invalid")
    if generation_id != manifest.get("semantic_sha256"):
        raise UnifiedCutoverError(
            RECEIPT_INVALID, "active pointer generation does not bind manifest semantics"
        )
    if pointer.get("manifest_sha256") != artifact_byte_sha256(manifest):
        raise UnifiedCutoverError(
            RECEIPT_INVALID, "active pointer does not bind target manifest bytes"
        )
    generation_path = f"{rules.unified_layout.generation_root}/{generation_id}/manifest.json"
    source_rules = _load_source_target_rules(root, rules.source_to_target_table)
    source_to_target = _source_to_target_rows(
        normalized_inventory,
        normalized_plan,
        rules=rules,
        mappings=source_rules.mappings,
    )
    classification_counts = {
        classification: sum(
            1 for row in source_to_target if row["classification"] == classification
        )
        for classification in (
            ACTIVE_AUTHORITY,
            ACTIVE_CALLER,
            CUSTODY_ONLY,
            INDEPENDENT_SOURCE,
            LEGACY_INACTIVE,
            NON_AUTHORITY_SHADOW,
        )
    }
    body = {
        "archive_plan_ref": artifact_exact_ref(normalized_plan),
        "blocker_codes": [],
        "cas_performed": False,
        "cutover_id": normalized_cutover,
        "expected_active_pointer_sha256": EMPTY_POINTER_SHA256,
        "inventory_ref": artifact_exact_ref(normalized_inventory),
        "permanent_marker_path": rules.unified_layout.permanent_marker,
        "rules_ref": _exact_rules_ref(
            loaded_rules.path.relative_to(root).as_posix(), loaded_rules.raw
        ),
        "source_to_target": source_to_target,
        "source_to_target_rules_ref": _exact_rules_ref(
            source_rules.relative_path, source_rules.raw
        ),
        "status": "READY_FOR_CAS",
        "summary": {
            "baseline_custody_facts": rules.baseline_custody_facts,
            "classification_counts": classification_counts,
            "source_count": len(source_to_target),
            "archive_copy_count": classification_counts[ACTIVE_AUTHORITY],
            "unified_object_count": sum(
                1 for row in source_to_target if row["object_relative_path"] is not None
            ),
            "shadow_copy_count": 0,
            "independent_source_copy_count": 0,
        },
        "target_active_pointer_path": rules.unified_layout.active_pointer,
        "target_active_pointer_ref": _active_pointer_ref(pointer),
        "target_generation_id": generation_id,
        "target_generation_manifest_path": generation_path,
        "target_generation_manifest_ref": artifact_exact_ref(manifest),
        "target_release_manifest_ref": dict(release_ref),
        "write_performed": False,
    }
    payload = {
        **body,
        "migration_receipt_id": _identity(
            MIGRATION_RECEIPT_KIND, body, prefix="migration-receipt-"
        ),
    }
    if set(payload) != MIGRATION_RECEIPT_PAYLOAD_FIELDS:
        raise AssertionError("migration receipt payload fields drifted")
    return seal_artifact(
        MIGRATION_RECEIPT_KIND,
        payload,
        created_at=created_at,
        contract_sha256=MIGRATION_RECEIPT_CONTRACT_SHA256,
    )


def build_unified_migration_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return build_pre_cas_migration_receipt(*args, **kwargs)


def validate_pre_cas_migration_receipt(
    document_or_bytes: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    try:
        document = validate_artifact(
            document_or_bytes,
            expected_kind=MIGRATION_RECEIPT_KIND,
            expected_contract_sha256=MIGRATION_RECEIPT_CONTRACT_SHA256,
        )
    except (TypeError, ValueError) as exc:
        raise UnifiedCutoverError(RECEIPT_INVALID, "migration receipt artifact is invalid") from exc
    payload = document["payload"]
    if set(payload) != MIGRATION_RECEIPT_PAYLOAD_FIELDS:
        raise UnifiedCutoverError(RECEIPT_INVALID, "migration receipt fields are not exact")
    identity = payload["migration_receipt_id"]
    body = dict(payload)
    body.pop("migration_receipt_id")
    if identity != _identity(MIGRATION_RECEIPT_KIND, body, prefix="migration-receipt-"):
        raise UnifiedCutoverError(RECEIPT_INVALID, "migration receipt identity mismatch")
    if (
        payload["status"] != "READY_FOR_CAS"
        or payload["blocker_codes"] != []
        or payload["expected_active_pointer_sha256"] != EMPTY_POINTER_SHA256
        or payload["write_performed"] is not False
        or payload["cas_performed"] is not False
        or payload["summary"].get("shadow_copy_count") != 0
        or payload["summary"].get("independent_source_copy_count") != 0
    ):
        raise UnifiedCutoverError(RECEIPT_INVALID, "migration receipt is not pre-CAS ready")
    facts = payload["summary"].get("baseline_custody_facts")
    if (
        type(facts) is not dict
        or sha256_bytes(canonical_json_bytes(facts)) != BASELINE_CUSTODY_FACTS_SHA256
    ):
        raise UnifiedCutoverError(RECEIPT_INVALID, "baseline custody facts are inconsistent")
    rows = payload["source_to_target"]
    if type(rows) is not list or rows != sorted(
        rows, key=lambda row: row.get("source_relative_path", "") if type(row) is dict else ""
    ):
        raise UnifiedCutoverError(RECEIPT_INVALID, "source-to-target rows are not sorted")
    for row in rows:
        if type(row) is not dict or set(row) != {
            "action",
            "archive_relative_path",
            "classification",
            "object_relative_path",
            "pointer_history_relative_path",
            "source_byte_sha256",
            "source_bytes",
            "source_relative_path",
        }:
            raise UnifiedCutoverError(RECEIPT_INVALID, "source-to-target row is invalid")
        if row["classification"] == NON_AUTHORITY_SHADOW and (
            row["action"] != "RECORD_ONLY"
            or row["archive_relative_path"] is not None
            or row["object_relative_path"] is not None
        ):
            raise UnifiedCutoverError(RECEIPT_INVALID, "shadow source is not record-only")
        if row["classification"] == INDEPENDENT_SOURCE and (
            row["action"] != "PRESERVE_INDEPENDENT"
            or row["archive_relative_path"] is not None
            or row["object_relative_path"] is not None
        ):
            raise UnifiedCutoverError(
                RECEIPT_INVALID, "strategy source crossed the independent boundary"
            )
    return document


def validate_pre_cas_activation_target(
    migration_receipt: Mapping[str, Any] | bytes,
    target_active_pointer: Mapping[str, Any] | bytes,
    target_generation_manifest: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Deep-bind a detached receipt to one exact operational first pointer."""

    receipt = validate_pre_cas_migration_receipt(migration_receipt)
    pointer = _validated_active_pointer(target_active_pointer)
    manifest = _validated_artifact(
        target_generation_manifest,
        expected_kind="system.generation_manifest",
        label="target generation manifest",
    )
    manifest_payload = manifest["payload"]
    payload = receipt["payload"]
    generation_id = manifest["semantic_sha256"]
    generation_path = f"results/system/generations/{generation_id}/manifest.json"
    if (
        manifest_payload.get("generation_state") != "OPERATIONAL"
        or manifest_payload.get("migration_receipt_ref") is not None
        or manifest_payload.get("migration_marker_ref") is not None
        or pointer["generation_id"] != generation_id
        or pointer["manifest_sha256"] != artifact_byte_sha256(manifest)
        or payload["target_generation_id"] != generation_id
        or payload["target_generation_manifest_ref"] != artifact_exact_ref(manifest)
        or payload["target_generation_manifest_path"] != generation_path
        or payload["target_release_manifest_ref"]
        != manifest_payload.get("release_manifest_ref")
        or payload["target_active_pointer_ref"] != _active_pointer_ref(pointer)
        or payload["target_active_pointer_path"] != str(ACTIVE_POINTER_PATH)
        or payload["permanent_marker_path"] != str(MIGRATION_MARKER_PATH)
        or payload["expected_active_pointer_sha256"] != EMPTY_POINTER_SHA256
    ):
        raise UnifiedCutoverError(RECEIPT_INVALID, "activation target binding mismatch")
    return receipt


def write_pre_cas_migration_receipt(
    path: str | os.PathLike[str],
    receipt: Mapping[str, Any] | bytes,
) -> bool:
    document = validate_pre_cas_migration_receipt(receipt)
    return write_idempotent_bytes(Path(path), canonical_json_bytes(document))


def build_permanent_marker_payload(
    migration_receipt: Mapping[str, Any] | bytes,
    active_pointer: Mapping[str, Any] | bytes,
    generation_manifest: Mapping[str, Any] | bytes,
    *,
    completed_at: str,
) -> dict[str, Any]:
    """Build the permanent post-CAS marker without performing a write."""

    receipt = validate_pre_cas_activation_target(
        migration_receipt, active_pointer, generation_manifest
    )
    pointer = _validated_active_pointer(active_pointer)
    manifest = _validated_artifact(
        generation_manifest,
        expected_kind="system.generation_manifest",
        label="generation manifest",
    )
    payload_receipt = receipt["payload"]
    if (
        _active_pointer_ref(pointer) != payload_receipt["target_active_pointer_ref"]
        or artifact_exact_ref(manifest) != payload_receipt["target_generation_manifest_ref"]
        or pointer.get("generation_id") != payload_receipt["target_generation_id"]
    ):
        raise UnifiedCutoverError(RECEIPT_INVALID, "post-CAS marker target mismatch")
    body = {
        "active_pointer_ref": _active_pointer_ref(pointer),
        "archive_plan_ref": payload_receipt["archive_plan_ref"],
        "blocker_codes": [],
        "cutover_id": payload_receipt["cutover_id"],
        "generation_id": payload_receipt["target_generation_id"],
        "generation_manifest_ref": artifact_exact_ref(manifest),
        "inventory_ref": payload_receipt["inventory_ref"],
        "legacy_replay_refused": True,
        "migration_receipt_ref": artifact_exact_ref(receipt),
        "migration_replay_refused": True,
        "permanent_marker_path": payload_receipt["permanent_marker_path"],
        "status": "COMPLETE",
    }
    payload = {
        **body,
        "marker_id": _identity(PERMANENT_MARKER_KIND, body, prefix="migration-marker-"),
    }
    if set(payload) != PERMANENT_MARKER_PAYLOAD_FIELDS:
        raise AssertionError("permanent marker payload fields drifted")
    return seal_artifact(
        PERMANENT_MARKER_KIND,
        payload,
        created_at=completed_at,
        contract_sha256=PERMANENT_MARKER_CONTRACT_SHA256,
    )


def validate_permanent_marker(
    document_or_bytes: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    try:
        document = validate_artifact(
            document_or_bytes,
            expected_kind=PERMANENT_MARKER_KIND,
            expected_contract_sha256=PERMANENT_MARKER_CONTRACT_SHA256,
        )
    except (TypeError, ValueError) as exc:
        raise UnifiedCutoverError(RECEIPT_INVALID, "permanent marker artifact is invalid") from exc
    payload = document["payload"]
    if set(payload) != PERMANENT_MARKER_PAYLOAD_FIELDS:
        raise UnifiedCutoverError(RECEIPT_INVALID, "permanent marker fields are not exact")
    identity = payload["marker_id"]
    body = dict(payload)
    body.pop("marker_id")
    if identity != _identity(PERMANENT_MARKER_KIND, body, prefix="migration-marker-"):
        raise UnifiedCutoverError(RECEIPT_INVALID, "permanent marker identity mismatch")
    if (
        payload["status"] != "COMPLETE"
        or payload["blocker_codes"] != []
        or payload["migration_replay_refused"] is not True
        or payload["legacy_replay_refused"] is not True
    ):
        raise UnifiedCutoverError(RECEIPT_INVALID, "permanent marker is not closed")
    return document


__all__ = [
    "EMPTY_POINTER_SHA256",
    "MIGRATION_RECEIPT_CONTRACT_SHA256",
    "MIGRATION_RECEIPT_KIND",
    "PERMANENT_MARKER_CONTRACT_SHA256",
    "PERMANENT_MARKER_KIND",
    "LoadedSourceTargetRules",
    "SourceTargetRule",
    "assert_pre_cas_cutover_state",
    "build_permanent_marker_payload",
    "build_initial_active_pointer",
    "build_pre_cas_migration_receipt",
    "build_unified_migration_receipt",
    "load_source_target_rules",
    "validate_permanent_marker",
    "validate_pre_cas_activation_target",
    "validate_pre_cas_migration_receipt",
    "write_pre_cas_migration_receipt",
]
