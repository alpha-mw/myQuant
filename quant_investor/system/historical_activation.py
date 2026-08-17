"""Frozen, non-executing schema dispatch for the initial unified activation.

The permanent marker must remain readable after descendant releases change the
compiled contract catalog.  These validators authenticate the exact first-
activation schemas and deterministic identities without executing old code or
consulting the descendant registry.
"""

from __future__ import annotations

from collections.abc import Mapping
import base64
from datetime import datetime
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes

from .errors import SystemContractError, SystemPreconditionError

_ENVELOPE_FIELDS: Final = frozenset(
    {"kind", "contract_sha256", "artifact_id", "created_at", "payload", "semantic_sha256"}
)
_REF_FIELDS: Final = frozenset(
    {"kind", "contract_sha256", "artifact_id", "semantic_sha256", "byte_sha256"}
)
_SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_KIND_RE: Final = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")

INITIAL_PRODUCTION_RECEIPT_CONTRACT_SHA256: Final = (
    "8242ab01dbe9bd3b939d388e198c77edeee4d3f0b74eba7d392e70d7343a48a0"
)
INITIAL_FINAL_AUTHORIZATION_CONTRACT_SHA256: Final = (
    "621c315342a2906134a6a4e36185aa8fbac0669f2d6eb3533b1c2dd04baf7363"
)
INITIAL_ACTIVATION_AUTHORIZATION_CONTRACT_SHA256: Final = (
    "ad0949a37faaa4d6abadb465fe6b5d146fb3d4347df0919b731ac9fe870ab0e5"
)
INITIAL_ACTIVATION_PREPARED_CONTRACT_SHA256: Final = (
    "48d4ab79ab58a76444aa7df431943f8ac56e62d6d573bda27c760940105277d8"
)
INITIAL_PERMANENT_MARKER_CONTRACT_SHA256: Final = (
    "7ddfeaf9b7e675a25a2c1486f9607e5b68a13519a34ec52f60c9948695fc2b40"
)
INITIAL_MIGRATION_RECEIPT_CONTRACT_SHA256: Final = (
    "867c9113f66780d777b7474c8ed013d2fedbdd6e324fe62c2fd44e050470838e"
)
INITIAL_MAIN_CHECKOUT_ADOPTION_CONTRACT_SHA256: Final = (
    "4b8f948d6351cf1fbcd8ef641e74d54cb6923c7fb9387e062409058f19ab654f"
)
INITIAL_LEGACY_DISPOSITION_CONTRACT_SHA256: Final = (
    "5608974f21ea5b31f3f7cf453bb09b2e60f7d6b372d452513bf54c5011d21468"
)
INITIAL_CUTOVER_GATE_EVIDENCE_CONTRACT_SHA256: Final = (
    "9782280464ebebc6bc9ec2453826eb3b3a636a9e86a530893820d0e4b5add938"
)
INITIAL_RELEASE_INSTALL_EVIDENCE_CONTRACT_SHA256: Final = (
    "90a88a66b80883aa857815db2afb05296fd79bb0a8380b434754d37bf8ab5d3d"
)

INITIAL_GATE_RUNNER_ID: Final = "myquant.system.final-cutover-gate-runner"
INITIAL_FINAL_PREFLIGHT_GATES: Final = frozenset(
    {
        "clean_detached_clone",
        "contract_catalog",
        "flake8",
        "full_pytest",
        "legacy_zero_call",
        "mypy",
        "projection",
        "release_install_origin",
        "replacement_selectors",
    }
)
INITIAL_GATE_SPECS: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    "clean_detached_clone": (
        (
            "FROZEN_PYTHON",
            "-m",
            "quant_investor.system.release_install",
            "verify-detached-checkout",
        ),
    ),
    "contract_catalog": (("uv", "run", "pytest", "tests/unit/test_unified_contracts.py", "-q"),),
    "flake8": (
        (
            "uv",
            "run",
            "flake8",
            "quant_investor",
            "--count",
            "--select=E9,F63,F7,F82",
            "--show-source",
            "--statistics",
        ),
        (
            "uv",
            "run",
            "flake8",
            "quant_investor/contracts",
            "quant_investor/system",
            "quant_investor/factors/governance",
            "quant_investor/intelligence",
            "quant_investor/mainline",
            "quant_investor/cli",
            "--max-complexity=10",
            "--max-line-length=100",
        ),
    ),
    "full_pytest": (("uv", "run", "pytest", "tests/unit", "-q", "-ra"),),
    "legacy_zero_call": (
        (
            "uv",
            "run",
            "pytest",
            "tests/unit/test_unified_migration_resolver.py",
            "tests/unit/test_unified_cli_commands.py",
            "tests/unit/test_unified_cli_input.py",
            "tests/unit/test_unified_cli_output.py",
            "-q",
        ),
    ),
    "mypy": (
        (
            "uv",
            "run",
            "mypy",
            "quant_investor/contracts",
            "quant_investor/system",
            "quant_investor/factors/governance",
            "quant_investor/intelligence",
            "quant_investor/mainline",
            "quant_investor/cli",
            "--ignore-missing-imports",
        ),
    ),
    "projection": (("uv", "run", "python", "operations/codex/verify_projection.py"),),
    "release_install_origin": (("FROZEN_PYTHON", "-m", "quant_investor.system.release_install"),),
    "replacement_selectors": (
        ("uv", "run", "python", "scripts/run_unified_replacement_selectors.py"),
    ),
}

INITIAL_PRODUCTION_RECEIPT_FIELDS: Final = frozenset(
    {
        "production_bootstrap_receipt_id",
        "state",
        "bootstrap_operator_request_ref",
        "source_root_id",
        "input_source_rows",
        "deployed_release_ref",
        "calendar_authority_policy_ref",
        "calendar_compilation_ref",
        "calendar_capability_ref",
        "calendar_capture_execution_ref",
        "calendar_authorization_basis",
        "calendar_source_limitations",
        "release_code_manifest_sha256",
        "generation_created_at",
        "expected_assembly_id",
        "generation_intent_sha256",
        "mainline_ref",
        "source_refs",
        "factor_source_object_refs",
        "factor_policy_ref",
        "factor_evidence_refs",
        "factor_active_set_ref",
        "factor_validation_attestation_ref",
        "readiness_matrix_ref",
        "emergency_controller_sha256",
        "skill_tree_sha256",
        "automation_semantic_sha256",
        "source_blockers",
        "fundamental_machine_states",
        "signal_statistics",
        "signal_statistics_sha256",
        "assembler_module_path",
        "assembler_code_sha256",
    }
)
INITIAL_FINAL_AUTHORIZATION_FIELDS: Final = frozenset(
    {
        "final_authorization_id",
        "state",
        "accepted_baseline_commit",
        "historical_integration_commit",
        "historical_dirty_evidence_ref",
        "concurrent_task_handoff_ref",
        "main_checkout_adoption_ref",
        "legacy_disposition_ref",
        "deployed_release_ref",
        "production_generation_manifest_ref",
        "production_bootstrap_receipt_ref",
        "calendar_authority_policy_ref",
        "calendar_compilation_ref",
        "calendar_capability_ref",
        "calendar_capture_execution_ref",
        "calendar_authorization_basis",
        "calendar_source_limitations",
        "calendar_policy_authorized",
        "release_commit",
        "release_tree",
        "final_integration_commit",
        "final_integration_tree",
        "ancestry_rows",
        "excluded_commit_rows",
        "final_worktree_inventory_sha256",
        "clean_checkout_readback_rows",
        "user_authorization_basis",
        "preflight_rows",
        "final_build_authorized",
        "cas_authorized",
    }
)
INITIAL_ACTIVATION_AUTHORIZATION_FIELDS: Final = frozenset(
    {
        "authorization_id",
        "state",
        "final_cutover_authorization_ref",
        "migration_receipt_ref",
        "target_generation_id",
        "target_generation_manifest_ref",
        "deployed_release_ref",
        "calendar_authority_policy_ref",
        "calendar_compilation_ref",
        "calendar_capability_ref",
        "calendar_capture_execution_ref",
        "calendar_authorization_basis",
        "calendar_source_limitations",
        "target_active_pointer",
        "target_active_pointer_ref",
        "target_active_pointer_path",
        "permanent_marker_ref",
        "permanent_marker_path",
        "expected_active_pointer_sha256",
        "prepared_at",
        "activated_at",
        "actor_uid",
        "os_actor",
    }
)
INITIAL_ACTIVATION_PREPARED_FIELDS: Final = frozenset(
    {
        "transaction_id",
        "state",
        "activation_authorization_ref",
        "final_cutover_authorization_ref",
        "migration_receipt_ref",
        "target_active_pointer",
        "target_active_pointer_ref",
        "permanent_marker_ref",
        "expected_active_pointer_sha256",
        "prepared_at",
        "actor_uid",
    }
)
INITIAL_PERMANENT_MARKER_FIELDS: Final = frozenset(
    {
        "marker_id",
        "status",
        "cutover_id",
        "migration_receipt_ref",
        "inventory_ref",
        "archive_plan_ref",
        "active_pointer_ref",
        "generation_manifest_ref",
        "generation_id",
        "permanent_marker_path",
        "migration_replay_refused",
        "legacy_replay_refused",
        "blocker_codes",
    }
)
INITIAL_MIGRATION_RECEIPT_FIELDS: Final = frozenset(
    {
        "migration_receipt_id",
        "status",
        "cutover_id",
        "inventory_ref",
        "archive_plan_ref",
        "rules_ref",
        "source_to_target_rules_ref",
        "source_to_target",
        "target_generation_id",
        "target_generation_manifest_path",
        "target_generation_manifest_ref",
        "target_release_manifest_ref",
        "target_active_pointer_path",
        "target_active_pointer_ref",
        "expected_active_pointer_sha256",
        "permanent_marker_path",
        "write_performed",
        "cas_performed",
        "blocker_codes",
        "summary",
    }
)
INITIAL_MAIN_CHECKOUT_ADOPTION_FIELDS: Final = frozenset(
    {
        "adoption_id",
        "state",
        "task_name",
        "thread_id",
        "source_task_outcome",
        "handoff_type",
        "accepted_baseline_commit",
        "accepted_baseline_tree",
        "adoption_commit",
        "adoption_tree",
        "adoption_parent",
        "path_rows",
        "task_origin_paths",
        "orphan_paths",
        "disposition_rows",
        "focused_test_rows",
        "full_gate_refs",
        "source_task_completion",
        "writer_ended",
        "main_clean",
        "readback_rows",
        "user_authorization_basis",
        "task_authorship_claimed",
        "human_signature_claimed",
        "history_rewritten",
    }
)
INITIAL_LEGACY_DISPOSITION_FIELDS: Final = frozenset(
    {"disposition_id", "state", "source_commit", "rows", "blocked_unresolved_count"}
)
INITIAL_CUTOVER_GATE_EVIDENCE_FIELDS: Final = frozenset(
    {
        "evidence_id",
        "state",
        "gate_id",
        "runner_id",
        "runner_spec_sha256",
        "runner_code_sha256",
        "final_commit",
        "final_tree",
        "environment_sha256",
        "batch_results",
        "subject_ref",
        "started_at",
        "finished_at",
    }
)
INITIAL_RELEASE_INSTALL_EVIDENCE_FIELDS: Final = frozenset(
    {
        "release_install_id",
        "state",
        "final_commit",
        "final_tree",
        "code_tree_sha256",
        "git_code_manifest_sha256",
        "release_ref",
        "source_archive",
        "wheel",
        "install_root",
        "python_executable",
        "python_executable_sha256",
        "import_origin",
        "installed_code_manifest_sha256",
        "contract_catalog_sha256",
        "lockfile_sha256",
        "dependency_install_mode",
        "editable_install",
        "source_tree_import",
    }
)

_GIT_OID_RE: Final = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_THREAD_RE: Final = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_PATH_ROW_FIELDS: Final = frozenset(
    {"path", "status", "mode", "size", "git_blob_oid", "byte_sha256"}
)
_ADOPTION_DISPOSITION_ROW_FIELDS: Final = frozenset(
    {
        "path",
        "partition",
        "decision",
        "target_path",
        "target_blob_oid",
        "behavior_test_selector",
        "reason",
    }
)
_TEST_ROW_FIELDS: Final = frozenset({"command", "exit_code", "stdout_sha256", "status"})
_READBACK_ROW_FIELDS: Final = frozenset(
    {
        "commit",
        "tree",
        "status_porcelain_sha256",
        "path_inventory_sha256",
        "observed_at",
    }
)
_SOURCE_TASK_COMPLETION_FIELDS: Final = frozenset(
    {"status", "latest_turn_id", "completed_at", "final_message_sha256"}
)
_LEGACY_DISPOSITION_ROW_FIELDS: Final = frozenset(
    {
        "source_path",
        "source_blob_oid",
        "classification",
        "stable_target_path",
        "stable_target_blob_oid",
        "behavior_test_selector",
        "reason",
    }
)
_PREFLIGHT_ROW_FIELDS: Final = frozenset({"gate_id", "evidence_ref"})
_GATE_BATCH_FIELDS: Final = frozenset(
    {
        "argv",
        "exit_code",
        "stdout_base64",
        "stdout_sha256",
        "stderr_base64",
        "stderr_sha256",
        "executable_path",
        "executable_sha256",
        "stdin_sha256",
    }
)
_ARCHIVE_ROW_FIELDS: Final = frozenset({"path", "byte_sha256", "size"})
_GATE_OUTPUT_MAX_BYTES: Final = 64 * 1024 * 1024


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not lowercase SHA-256")
    return value


def _timestamp(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise SystemContractError(f"{label} is not canonical UTC") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise SystemContractError(f"{label} is not canonical UTC")
    return value


def _git_oid(value: Any, *, label: str) -> str:
    if type(value) is not str or _GIT_OID_RE.fullmatch(value) is None:
        raise SystemContractError(f"{label} is not a canonical Git object id")
    return value


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise SystemContractError(f"{label} is not canonical text")
    return value


def _path(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    text = _text(value, label=label)
    parsed = PurePosixPath(text)
    if (
        parsed.is_absolute()
        or str(parsed) != text
        or "\\" in text
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise SystemContractError(f"{label} is not a canonical relative path")
    return text


def _absolute_path(value: Any, *, label: str) -> str:
    if type(value) is not str or not Path(value).is_absolute():
        raise SystemContractError(f"{label} is not absolute")
    return value


def _canonical_base64(value: Any, *, label: str) -> bytes:
    if type(value) is not str:
        raise SystemContractError(f"{label} is not base64 text")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise SystemContractError(f"{label} is not canonical base64") from exc
    if len(raw) > _GATE_OUTPUT_MAX_BYTES or base64.b64encode(raw).decode("ascii") != value:
        raise SystemContractError(f"{label} is not bounded canonical base64")
    return raw


def _initial_gate_spec_sha256(gate_id: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "runner_id": INITIAL_GATE_RUNNER_ID,
                "gate_id": gate_id,
                "batches": [list(argv) for argv in INITIAL_GATE_SPECS[gate_id]],
            }
        )
    ).hexdigest()


def _initial_ref(value: Any, *, label: str) -> dict[str, str]:
    return validate_frozen_object_ref(value, label=label)


def validate_initial_preflight_rows(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise SystemPreconditionError("historical final preflight rows are absent")
    rows: list[dict[str, Any]] = []
    for index, value_row in enumerate(value):
        if type(value_row) is not dict or set(value_row) != _PREFLIGHT_ROW_FIELDS:
            raise SystemContractError("historical final preflight row fields are not exact")
        rows.append(
            {
                "gate_id": _text(
                    value_row["gate_id"], label=f"historical preflight[{index}].gate_id"
                ),
                "evidence_ref": _initial_ref(
                    value_row["evidence_ref"],
                    label=f"historical preflight[{index}].evidence_ref",
                ),
            }
        )
    if [row["gate_id"] for row in rows] != sorted(INITIAL_FINAL_PREFLIGHT_GATES):
        raise SystemPreconditionError("historical final preflight gate set is not exact")
    return rows


def _validate_initial_test_rows(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise SystemContractError("historical focused test evidence is absent")
    rows: list[dict[str, Any]] = []
    for index, value_row in enumerate(value):
        if type(value_row) is not dict or set(value_row) != _TEST_ROW_FIELDS:
            raise SystemContractError("historical focused test row fields are not exact")
        if value_row["exit_code"] != 0 or value_row["status"] != "PASS":
            raise SystemPreconditionError("historical focused task test did not pass")
        rows.append(
            {
                "command": _text(value_row["command"], label=f"historical tests[{index}].command"),
                "exit_code": 0,
                "stdout_sha256": _sha(
                    value_row["stdout_sha256"],
                    label=f"historical tests[{index}].stdout_sha256",
                ),
                "status": "PASS",
            }
        )
    if [row["command"] for row in rows] != sorted({row["command"] for row in rows}):
        raise SystemContractError("historical focused test commands are not sorted unique")
    return rows


def _validate_initial_readbacks(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != 2:
        raise SystemContractError("historical adoption requires exactly two readbacks")
    rows: list[dict[str, Any]] = []
    for index, value_row in enumerate(value):
        if type(value_row) is not dict or set(value_row) != _READBACK_ROW_FIELDS:
            raise SystemContractError("historical adoption readback fields are not exact")
        rows.append(
            {
                "commit": _git_oid(
                    value_row["commit"], label=f"historical readback[{index}].commit"
                ),
                "tree": _git_oid(value_row["tree"], label=f"historical readback[{index}].tree"),
                "status_porcelain_sha256": _sha(
                    value_row["status_porcelain_sha256"],
                    label=f"historical readback[{index}].status",
                ),
                "path_inventory_sha256": _sha(
                    value_row["path_inventory_sha256"],
                    label=f"historical readback[{index}].inventory",
                ),
                "observed_at": _timestamp(
                    value_row["observed_at"],
                    label=f"historical readback[{index}].observed_at",
                ),
            }
        )
    first = {key: item for key, item in rows[0].items() if key != "observed_at"}
    second = {key: item for key, item in rows[1].items() if key != "observed_at"}
    if first != second or rows[0]["observed_at"] == rows[1]["observed_at"]:
        raise SystemPreconditionError("historical adoption readback is not stable")
    return rows


def validate_frozen_object_ref(
    value: Any,
    *,
    label: str,
    dispatch: Mapping[tuple[str, str], str] | None = None,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != _REF_FIELDS:
        raise SystemContractError(f"{label} fields are not exact")
    kind = value.get("kind")
    contract_sha = value.get("contract_sha256")
    artifact_id = value.get("artifact_id")
    if (
        type(kind) is not str
        or _KIND_RE.fullmatch(kind) is None
        or type(artifact_id) is not str
        or not artifact_id
    ):
        raise SystemContractError(f"{label} identity is invalid")
    normalized = {
        "kind": kind,
        "contract_sha256": _sha(contract_sha, label=f"{label}.contract_sha256"),
        "artifact_id": artifact_id,
        "semantic_sha256": _sha(value.get("semantic_sha256"), label=f"{label}.semantic"),
        "byte_sha256": _sha(value.get("byte_sha256"), label=f"{label}.bytes"),
    }
    if dispatch is not None and dispatch.get((kind, normalized["contract_sha256"])) is None:
        raise SystemContractError(f"{label} contract pair is not initial-catalog anchored")
    return normalized


def frozen_object_ref(artifact: Mapping[str, Any]) -> dict[str, str]:
    return {
        "kind": artifact["kind"],
        "contract_sha256": artifact["contract_sha256"],
        "artifact_id": artifact["artifact_id"],
        "semantic_sha256": artifact["semantic_sha256"],
        "byte_sha256": hashlib.sha256(canonical_json_bytes(dict(artifact))).hexdigest(),
    }


def _artifact(
    document: Mapping[str, Any] | bytes,
    *,
    kind: str,
    contract_sha256: str,
    identity_field: str,
    payload_fields: frozenset[str],
) -> dict[str, Any]:
    if type(document) is bytes:
        value = parse_canonical_json_bytes(document, label=f"historical {kind}")
    elif type(document) is dict:
        canonical_json_bytes(document)
        value = dict(document)
    else:
        raise SystemContractError(f"historical {kind} is not an artifact")
    if (
        type(value) is not dict
        or set(value) != _ENVELOPE_FIELDS
        or value.get("kind") != kind
        or value.get("contract_sha256") != contract_sha256
        or type(value.get("payload")) is not dict
        or set(value["payload"]) != payload_fields
    ):
        raise SystemContractError(f"historical {kind} schema differs")
    payload = value["payload"]
    identity = payload.get(identity_field)
    if type(identity) is not str or not identity or value.get("artifact_id") != identity:
        raise SystemContractError(f"historical {kind} identity differs")
    created_at = _timestamp(value.get("created_at"), label=f"historical {kind}.created_at")
    semantic = _sha(value.get("semantic_sha256"), label=f"historical {kind}.semantic")
    preimage = {
        "domain": "myquant-artifact",
        "kind": kind,
        "contract_sha256": contract_sha256,
        "identity_field": identity_field,
        "artifact_id": identity,
        "created_at": created_at,
        "payload": payload,
    }
    if semantic != hashlib.sha256(canonical_json_bytes(preimage)).hexdigest():
        raise SystemContractError(f"historical {kind} semantic SHA differs")
    return value


def validate_initial_main_checkout_adoption(  # noqa: C901
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.main_checkout_adoption",
        contract_sha256=INITIAL_MAIN_CHECKOUT_ADOPTION_CONTRACT_SHA256,
        identity_field="adoption_id",
        payload_fields=INITIAL_MAIN_CHECKOUT_ADOPTION_FIELDS,
    )
    payload = artifact["payload"]
    if (
        payload["state"] != "IMMUTABLE"
        or payload["source_task_outcome"] != "COMPLETED_WITHOUT_COMMIT"
        or payload["handoff_type"] != "PROSPECTIVE_ADOPTION"
        or payload["writer_ended"] is not True
        or payload["main_clean"] is not True
        or payload["task_authorship_claimed"] is not False
        or payload["human_signature_claimed"] is not False
        or payload["history_rewritten"] is not False
    ):
        raise SystemPreconditionError("historical prospective adoption authority differs")
    _text(payload["adoption_id"], label="historical adoption_id")
    _text(payload["task_name"], label="historical task_name")
    if type(payload["thread_id"]) is not str or _THREAD_RE.fullmatch(payload["thread_id"]) is None:
        raise SystemContractError("historical adoption thread id is invalid")
    for field in (
        "accepted_baseline_commit",
        "accepted_baseline_tree",
        "adoption_commit",
        "adoption_tree",
        "adoption_parent",
    ):
        _git_oid(payload[field], label=f"historical {field}")

    raw_paths = payload["path_rows"]
    if type(raw_paths) is not list or len(raw_paths) != 22:
        raise SystemPreconditionError("historical adoption path cardinality is not 22")
    path_rows: list[dict[str, Any]] = []
    for index, value_row in enumerate(raw_paths):
        if type(value_row) is not dict or set(value_row) != _PATH_ROW_FIELDS:
            raise SystemContractError("historical adoption path fields are not exact")
        if (
            value_row["status"] not in {"ADDED", "MODIFIED"}
            or value_row["mode"] not in {"100600", "100644", "100755"}
            or type(value_row["size"]) is not int
            or value_row["size"] < 0
        ):
            raise SystemContractError("historical adoption path identity is invalid")
        path_rows.append(
            {
                **value_row,
                "path": _path(value_row["path"], label=f"historical paths[{index}].path"),
                "git_blob_oid": _git_oid(
                    value_row["git_blob_oid"], label=f"historical paths[{index}].blob"
                ),
                "byte_sha256": _sha(
                    value_row["byte_sha256"], label=f"historical paths[{index}].bytes"
                ),
            }
        )
    paths = [row["path"] for row in path_rows]
    if paths != sorted(set(paths)):
        raise SystemContractError("historical adoption paths are not sorted unique")

    partitions: list[list[str]] = []
    for field, expected_count in (("task_origin_paths", 17), ("orphan_paths", 5)):
        values = payload[field]
        if type(values) is not list or len(values) != expected_count:
            raise SystemPreconditionError(f"historical {field} cardinality differs")
        normalized = [
            _path(item, label=f"historical {field}[{index}]") for index, item in enumerate(values)
        ]
        if normalized != sorted(set(normalized)):
            raise SystemContractError(f"historical {field} is not sorted unique")
        partitions.append(normalized)
    task_paths, orphan_paths = partitions
    if set(task_paths) & set(orphan_paths) or sorted([*task_paths, *orphan_paths]) != paths:
        raise SystemPreconditionError("historical adoption path partition is not exact")

    dispositions = payload["disposition_rows"]
    if type(dispositions) is not list or len(dispositions) != len(paths):
        raise SystemPreconditionError("historical adoption disposition closure is incomplete")
    normalized_disposition_paths: list[str] = []
    for index, value_row in enumerate(dispositions):
        if type(value_row) is not dict or set(value_row) != _ADOPTION_DISPOSITION_ROW_FIELDS:
            raise SystemContractError("historical adoption disposition fields are not exact")
        source_path = _path(value_row["path"], label=f"historical dispositions[{index}].path")
        partition = "TASK_ORIGIN" if source_path in task_paths else "ORPHAN"
        if value_row["partition"] != partition:
            raise SystemContractError("historical adoption disposition partition differs")
        decision = value_row["decision"]
        if decision not in {
            "EXACT_PRESERVED",
            "REPAIRED_IN_FINAL_INTEGRATION",
            "PORTED_TO_STABLE",
            "LEGACY_CUSTODY_ONLY",
        }:
            raise SystemContractError("historical adoption disposition decision is invalid")
        target_path = _path(
            value_row["target_path"],
            label=f"historical dispositions[{index}].target_path",
            allow_empty=True,
        )
        if decision == "LEGACY_CUSTODY_ONLY":
            if target_path or value_row["target_blob_oid"] != "":
                raise SystemContractError("historical legacy-custody disposition has a target")
        else:
            if not target_path:
                raise SystemContractError("historical adoption target is absent")
            _git_oid(
                value_row["target_blob_oid"],
                label=f"historical dispositions[{index}].target_blob_oid",
            )
        if decision == "EXACT_PRESERVED" and target_path != source_path:
            raise SystemContractError("historical exact-preserved disposition changes path")
        _text(
            value_row["behavior_test_selector"],
            label=f"historical dispositions[{index}].selector",
        )
        _text(value_row["reason"], label=f"historical dispositions[{index}].reason")
        normalized_disposition_paths.append(source_path)
    if normalized_disposition_paths != paths:
        raise SystemPreconditionError("historical adoption dispositions are not path complete")

    _validate_initial_test_rows(payload["focused_test_rows"])
    validate_initial_preflight_rows(payload["full_gate_refs"])
    completion = payload["source_task_completion"]
    if type(completion) is not dict or set(completion) != _SOURCE_TASK_COMPLETION_FIELDS:
        raise SystemContractError("historical source task completion fields are not exact")
    if completion["status"] != "COMPLETED_WITHOUT_COMMIT":
        raise SystemPreconditionError("historical source task completion status differs")
    if (
        type(completion["latest_turn_id"]) is not str
        or _THREAD_RE.fullmatch(completion["latest_turn_id"]) is None
    ):
        raise SystemContractError("historical source task completion turn id is invalid")
    _timestamp(completion["completed_at"], label="historical completion time")
    _sha(completion["final_message_sha256"], label="historical completion message")
    readbacks = _validate_initial_readbacks(payload["readback_rows"])
    if any(
        row["commit"] != payload["adoption_commit"] or row["tree"] != payload["adoption_tree"]
        for row in readbacks
    ):
        raise SystemPreconditionError("historical adoption readback identity differs")
    _text(payload["user_authorization_basis"], label="historical authorization basis")
    return artifact


def validate_initial_legacy_source_disposition(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.legacy_source_disposition",
        contract_sha256=INITIAL_LEGACY_DISPOSITION_CONTRACT_SHA256,
        identity_field="disposition_id",
        payload_fields=INITIAL_LEGACY_DISPOSITION_FIELDS,
    )
    payload = artifact["payload"]
    if payload["state"] != "IMMUTABLE" or payload["blocked_unresolved_count"] != 0:
        raise SystemPreconditionError("historical legacy disposition is unresolved")
    _text(payload["disposition_id"], label="historical disposition_id")
    _git_oid(payload["source_commit"], label="historical disposition source commit")
    rows = payload["rows"]
    if type(rows) is not list or not rows:
        raise SystemContractError("historical legacy disposition rows are absent")
    paths: list[str] = []
    for index, value_row in enumerate(rows):
        if type(value_row) is not dict or set(value_row) != _LEGACY_DISPOSITION_ROW_FIELDS:
            raise SystemContractError("historical legacy disposition row fields are not exact")
        source_path = _path(
            value_row["source_path"], label=f"historical legacy[{index}].source_path"
        )
        _git_oid(value_row["source_blob_oid"], label=f"historical legacy[{index}].source_blob")
        classification = value_row["classification"]
        if classification not in {
            "PORTED_TO_STABLE",
            "PACKAGING_ONLY_NOT_REQUIRED",
            "LEGACY_CUSTODY_ONLY",
        }:
            raise SystemPreconditionError("historical legacy disposition classification blocks")
        target_path = _path(
            value_row["stable_target_path"],
            label=f"historical legacy[{index}].target_path",
            allow_empty=True,
        )
        target_oid = value_row["stable_target_blob_oid"]
        if target_path:
            _git_oid(target_oid, label=f"historical legacy[{index}].target_blob")
        elif target_oid != "":
            raise SystemContractError("historical legacy disposition target blob is orphaned")
        if classification == "PORTED_TO_STABLE" and not target_path:
            raise SystemContractError("historical stable port target is absent")
        _text(value_row["behavior_test_selector"], label=f"historical legacy[{index}].selector")
        _text(value_row["reason"], label=f"historical legacy[{index}].reason")
        paths.append(source_path)
    if paths != sorted(set(paths)):
        raise SystemContractError("historical legacy disposition paths are not sorted unique")
    return artifact


def validate_initial_cutover_gate_evidence(  # noqa: C901
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.cutover_gate_evidence",
        contract_sha256=INITIAL_CUTOVER_GATE_EVIDENCE_CONTRACT_SHA256,
        identity_field="evidence_id",
        payload_fields=INITIAL_CUTOVER_GATE_EVIDENCE_FIELDS,
    )
    payload = artifact["payload"]
    gate = payload["gate_id"]
    if gate not in INITIAL_FINAL_PREFLIGHT_GATES or payload["state"] != "PASS":
        raise SystemPreconditionError("historical cutover gate did not pass")
    if payload["runner_id"] != INITIAL_GATE_RUNNER_ID or payload[
        "runner_spec_sha256"
    ] != _initial_gate_spec_sha256(gate):
        raise SystemPreconditionError("historical cutover gate runner identity differs")
    for field in ("runner_code_sha256", "environment_sha256"):
        _sha(payload[field], label=f"historical gate {field}")
    for field in ("final_commit", "final_tree"):
        _git_oid(payload[field], label=f"historical gate {field}")
    rows = payload["batch_results"]
    specs = INITIAL_GATE_SPECS[gate]
    if type(rows) is not list or len(rows) != len(specs):
        raise SystemContractError("historical cutover gate batch count differs")
    detached_input = hashlib.sha256(
        canonical_json_bytes(
            {"final_commit": payload["final_commit"], "final_tree": payload["final_tree"]}
        )
    ).hexdigest()
    for index, (value_row, argv) in enumerate(zip(rows, specs)):
        if type(value_row) is not dict or set(value_row) != _GATE_BATCH_FIELDS:
            raise SystemContractError("historical cutover gate batch fields are not exact")
        if value_row["argv"] != list(argv) or value_row["exit_code"] != 0:
            raise SystemPreconditionError("historical cutover gate command did not pass")
        stdout = _canonical_base64(
            value_row["stdout_base64"], label=f"historical batch[{index}].stdout"
        )
        stderr = _canonical_base64(
            value_row["stderr_base64"], label=f"historical batch[{index}].stderr"
        )
        if (
            value_row["stdout_sha256"] != hashlib.sha256(stdout).hexdigest()
            or value_row["stderr_sha256"] != hashlib.sha256(stderr).hexdigest()
        ):
            raise SystemContractError("historical cutover gate output hash differs")
        _absolute_path(value_row["executable_path"], label="historical gate executable")
        _sha(value_row["executable_sha256"], label="historical executable SHA")
        stdin_sha = _sha(value_row["stdin_sha256"], label="historical stdin SHA")
        if gate == "clean_detached_clone" and stdin_sha != detached_input:
            raise SystemContractError("historical detached gate input differs")
        if gate not in {"clean_detached_clone", "release_install_origin"} and stdin_sha != (
            hashlib.sha256(b"").hexdigest()
        ):
            raise SystemContractError("historical cutover gate input is not empty")
    _initial_ref(payload["subject_ref"], label="historical gate subject")
    started = _timestamp(payload["started_at"], label="historical gate started_at")
    finished = _timestamp(payload["finished_at"], label="historical gate finished_at")
    if finished < started or artifact["created_at"] != finished:
        raise SystemContractError("historical cutover gate time binding differs")
    body = {key: payload[key] for key in sorted(payload) if key != "evidence_id"}
    expected = "cutover-gate-" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if payload["evidence_id"] != expected:
        raise SystemContractError("historical cutover gate identity differs")
    return artifact


def validate_initial_release_install_evidence(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.release_install_evidence",
        contract_sha256=INITIAL_RELEASE_INSTALL_EVIDENCE_CONTRACT_SHA256,
        identity_field="release_install_id",
        payload_fields=INITIAL_RELEASE_INSTALL_EVIDENCE_FIELDS,
    )
    payload = artifact["payload"]
    if (
        payload["state"] != "VALIDATED"
        or payload["dependency_install_mode"] != "UV_LOCKED_NON_EDITABLE_WHEEL"
        or payload["editable_install"] is not False
        or payload["source_tree_import"] is not False
    ):
        raise SystemPreconditionError("historical release install evidence is not fail-closed")
    for field in ("final_commit", "final_tree"):
        _git_oid(payload[field], label=f"historical release {field}")
    for field in (
        "code_tree_sha256",
        "git_code_manifest_sha256",
        "python_executable_sha256",
        "installed_code_manifest_sha256",
        "contract_catalog_sha256",
        "lockfile_sha256",
    ):
        _sha(payload[field], label=f"historical release {field}")
    release_ref = _initial_ref(payload["release_ref"], label="historical release ref")
    if release_ref["kind"] != "system.release":
        raise SystemContractError("historical installed release kind is invalid")
    for field in ("source_archive", "wheel"):
        value_row = payload[field]
        if type(value_row) is not dict or set(value_row) != _ARCHIVE_ROW_FIELDS:
            raise SystemContractError("historical release archive fields are not exact")
        _absolute_path(value_row["path"], label=f"historical {field}.path")
        _sha(value_row["byte_sha256"], label=f"historical {field}.bytes")
        if type(value_row["size"]) is not int or value_row["size"] <= 0:
            raise SystemContractError("historical release archive size is invalid")
    for field in ("install_root", "python_executable", "import_origin"):
        _absolute_path(payload[field], label=f"historical release {field}")
    body = {key: payload[key] for key in sorted(payload) if key != "release_install_id"}
    expected = "release-install-" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if payload["release_install_id"] != expected:
        raise SystemContractError("historical release install identity differs")
    return artifact


def validate_initial_release_install_gate_binding(
    *,
    gate_evidence: Mapping[str, Any],
    install_evidence: Mapping[str, Any],
    deployed_release: Mapping[str, Any],
    deployed_release_ref: Mapping[str, Any],
) -> None:
    """Replay the exact first release-gate input/output without descendant schemas."""

    evidence = validate_initial_cutover_gate_evidence(gate_evidence)
    install = validate_initial_release_install_evidence(install_evidence)
    release_ref = _initial_ref(deployed_release_ref, label="historical deployed release")
    if (
        evidence["payload"]["gate_id"] != "release_install_origin"
        or install["payload"]["release_ref"] != release_ref
        or frozen_object_ref(deployed_release) != release_ref
    ):
        raise SystemPreconditionError("historical release gate identity differs")
    release_payload = deployed_release.get("payload")
    if type(release_payload) is not dict:
        raise SystemContractError("historical deployed release payload is absent")
    install_payload = install["payload"]
    if (
        install_payload["code_tree_sha256"] != release_payload.get("code_sha256")
        or install_payload["wheel"]["byte_sha256"] != release_payload.get("wheel_sha256")
        or install_payload["installed_code_manifest_sha256"]
        != release_payload.get("code_manifest_sha256")
    ):
        raise SystemPreconditionError("historical installed release identity differs")
    exact_input = canonical_json_bytes(
        {"release_install_evidence": install, "deployed_release": dict(deployed_release)}
    )
    batches = evidence["payload"]["batch_results"]
    if len(batches) != 1 or batches[0]["stdin_sha256"] != hashlib.sha256(exact_input).hexdigest():
        raise SystemPreconditionError("historical release gate input differs")
    output = parse_canonical_json_bytes(
        _canonical_base64(batches[0]["stdout_base64"], label="historical release output"),
        label="historical release output",
    )
    expected_output = {
        "state": "PASS",
        "release_ref": release_ref,
        "source_archive_sha256": install_payload["source_archive"]["byte_sha256"],
        "wheel_sha256": install_payload["wheel"]["byte_sha256"],
        "code_tree_sha256": install_payload["code_tree_sha256"],
        "installed_code_manifest_sha256": install_payload["installed_code_manifest_sha256"],
        "contract_catalog_sha256": install_payload["contract_catalog_sha256"],
        "import_origin": install_payload["import_origin"],
    }
    if output != expected_output:
        raise SystemPreconditionError("historical release gate output differs")


def validate_initial_production_receipt(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.production_bootstrap_receipt",
        contract_sha256=INITIAL_PRODUCTION_RECEIPT_CONTRACT_SHA256,
        identity_field="production_bootstrap_receipt_id",
        payload_fields=INITIAL_PRODUCTION_RECEIPT_FIELDS,
    )
    payload = artifact["payload"]
    body = {
        key: payload[key] for key in sorted(payload) if key != "production_bootstrap_receipt_id"
    }
    expected = "production-bootstrap-" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if payload["state"] != "VERIFIED" or payload["production_bootstrap_receipt_id"] != expected:
        raise SystemPreconditionError("historical production receipt is not VERIFIED")
    return artifact


def validate_initial_final_authorization(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.final_cutover_authorization",
        contract_sha256=INITIAL_FINAL_AUTHORIZATION_CONTRACT_SHA256,
        identity_field="final_authorization_id",
        payload_fields=INITIAL_FINAL_AUTHORIZATION_FIELDS,
    )
    payload = artifact["payload"]
    if (
        payload["state"] != "AUTHORIZED"
        or payload["calendar_policy_authorized"] is not True
        or payload["final_build_authorized"] is not True
        or payload["cas_authorized"] is not True
    ):
        raise SystemPreconditionError("historical final authorization is not authorized")
    return artifact


def _activation_identity(body: Mapping[str, Any]) -> str:
    return (
        "activation-authorization-"
        + hashlib.sha256(
            canonical_json_bytes(
                {"domain": "myquant-system-activation-authorization", "payload": dict(body)}
            )
        ).hexdigest()
    )


def _prepared_identity(body: Mapping[str, Any]) -> str:
    return (
        "activation-transaction-"
        + hashlib.sha256(
            canonical_json_bytes(
                {"domain": "myquant-system-activation-prepared", "payload": dict(body)}
            )
        ).hexdigest()
    )


def _migration_identity(kind: str, body: Mapping[str, Any], *, prefix: str) -> str:
    return (
        prefix
        + hashlib.sha256(
            canonical_json_bytes(
                {"domain": "myquant-migration-identity", "kind": kind, "payload": dict(body)}
            )
        ).hexdigest()
    )


def validate_initial_activation_authorization(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.activation_authorization",
        contract_sha256=INITIAL_ACTIVATION_AUTHORIZATION_CONTRACT_SHA256,
        identity_field="authorization_id",
        payload_fields=INITIAL_ACTIVATION_AUTHORIZATION_FIELDS,
    )
    body = dict(artifact["payload"])
    identity = body.pop("authorization_id")
    if body["state"] != "AUTHORIZED" or identity != _activation_identity(body):
        raise SystemPreconditionError("historical activation authorization is invalid")
    return artifact


def validate_initial_activation_prepared(
    document: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.activation_prepared",
        contract_sha256=INITIAL_ACTIVATION_PREPARED_CONTRACT_SHA256,
        identity_field="transaction_id",
        payload_fields=INITIAL_ACTIVATION_PREPARED_FIELDS,
    )
    body = dict(artifact["payload"])
    identity = body.pop("transaction_id")
    if body["state"] != "PREPARED" or identity != _prepared_identity(body):
        raise SystemPreconditionError("historical prepared transaction is invalid")
    return artifact


def validate_initial_permanent_marker(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.migration.complete",
        contract_sha256=INITIAL_PERMANENT_MARKER_CONTRACT_SHA256,
        identity_field="marker_id",
        payload_fields=INITIAL_PERMANENT_MARKER_FIELDS,
    )
    body = dict(artifact["payload"])
    identity = body.pop("marker_id")
    if (
        body["status"] != "COMPLETE"
        or body["blocker_codes"] != []
        or body["migration_replay_refused"] is not True
        or body["legacy_replay_refused"] is not True
        or identity
        != _migration_identity("system.migration.complete", body, prefix="migration-marker-")
    ):
        raise SystemPreconditionError("historical permanent marker is invalid")
    return artifact


def validate_initial_migration_receipt(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    artifact = _artifact(
        document,
        kind="system.migration.receipt",
        contract_sha256=INITIAL_MIGRATION_RECEIPT_CONTRACT_SHA256,
        identity_field="migration_receipt_id",
        payload_fields=INITIAL_MIGRATION_RECEIPT_FIELDS,
    )
    body = dict(artifact["payload"])
    identity = body.pop("migration_receipt_id")
    if (
        body["status"] != "READY_FOR_CAS"
        or body["expected_active_pointer_sha256"] != "EMPTY"
        or body["write_performed"] is not False
        or body["cas_performed"] is not False
        or body["blocker_codes"] != []
        or identity
        != _migration_identity("system.migration.receipt", body, prefix="migration-receipt-")
    ):
        raise SystemPreconditionError("historical migration receipt is invalid")
    return artifact


def validate_initial_activation_bundle(
    *,
    final_authorization: Mapping[str, Any] | bytes,
    activation_authorization: Mapping[str, Any] | bytes,
    prepared_transaction: Mapping[str, Any] | bytes,
    migration_receipt: Mapping[str, Any] | bytes,
    permanent_marker: Mapping[str, Any] | bytes,
    active_pointer: Mapping[str, Any],
    generation_manifest: Mapping[str, Any],
    deployed_release_ref: Mapping[str, Any],
    current_uid: int,
) -> dict[str, dict[str, Any]]:
    """Cross-bind every immutable first-activation artifact without current schemas."""

    final = validate_initial_final_authorization(final_authorization)
    authorization = validate_initial_activation_authorization(activation_authorization)
    prepared = validate_initial_activation_prepared(prepared_transaction)
    receipt = validate_initial_migration_receipt(migration_receipt)
    marker = validate_initial_permanent_marker(permanent_marker)
    pointer = dict(active_pointer)
    pointer_fields = {
        "generation_id",
        "manifest_sha256",
        "previous_pointer_sha256",
        "activated_at",
        "os_actor",
    }
    if type(active_pointer) is not dict or set(pointer) != pointer_fields:
        raise SystemContractError("historical active pointer fields differ")
    pointer_raw = canonical_json_bytes(pointer)
    pointer_ref = {
        "generation_id": _sha(pointer["generation_id"], label="historical generation id"),
        "manifest_sha256": _sha(pointer["manifest_sha256"], label="historical manifest byte SHA"),
        "byte_sha256": hashlib.sha256(pointer_raw).hexdigest(),
    }
    manifest_ref = frozen_object_ref(generation_manifest)
    release_ref = validate_frozen_object_ref(
        deployed_release_ref,
        label="historical deployed release ref",
    )
    receipt_ref = frozen_object_ref(receipt)
    final_ref = frozen_object_ref(final)
    authorization_ref = frozen_object_ref(authorization)
    marker_ref = frozen_object_ref(marker)
    receipt_payload = receipt["payload"]
    if (
        receipt_payload["target_generation_id"] != pointer["generation_id"]
        or receipt_payload["target_generation_manifest_ref"] != manifest_ref
        or receipt_payload["target_release_manifest_ref"] != release_ref
        or receipt_payload["target_active_pointer_ref"] != pointer_ref
        or receipt_payload["target_active_pointer_path"] != "results/system/_active.json"
        or receipt_payload["permanent_marker_path"] != "results/system/_migration_complete.json"
    ):
        raise SystemPreconditionError("historical migration target binding differs")
    final_payload = final["payload"]
    auth_payload = authorization["payload"]
    expected_auth = {
        "final_cutover_authorization_ref": final_ref,
        "migration_receipt_ref": receipt_ref,
        "target_generation_id": pointer["generation_id"],
        "target_generation_manifest_ref": manifest_ref,
        "deployed_release_ref": release_ref,
        "calendar_authority_policy_ref": final_payload["calendar_authority_policy_ref"],
        "calendar_compilation_ref": final_payload["calendar_compilation_ref"],
        "calendar_capability_ref": final_payload["calendar_capability_ref"],
        "calendar_capture_execution_ref": final_payload["calendar_capture_execution_ref"],
        "calendar_authorization_basis": final_payload["calendar_authorization_basis"],
        "calendar_source_limitations": final_payload["calendar_source_limitations"],
        "target_active_pointer": pointer,
        "target_active_pointer_ref": pointer_ref,
        "target_active_pointer_path": "results/system/_active.json",
        "permanent_marker_ref": marker_ref,
        "permanent_marker_path": "results/system/_migration_complete.json",
        "expected_active_pointer_sha256": "EMPTY",
        "activated_at": pointer["activated_at"],
        "actor_uid": current_uid,
        "os_actor": f"uid:{current_uid}",
    }
    if any(auth_payload[field] != value for field, value in expected_auth.items()):
        raise SystemPreconditionError("historical activation authorization binding differs")
    prepared_at = _timestamp(auth_payload["prepared_at"], label="historical prepared_at")
    activated_at = _timestamp(auth_payload["activated_at"], label="historical activated_at")
    if prepared_at > activated_at or pointer["os_actor"] != f"uid:{current_uid}":
        raise SystemPreconditionError("historical activation actor/time binding differs")

    marker_payload = marker["payload"]
    expected_marker = {
        "migration_receipt_ref": receipt_ref,
        "active_pointer_ref": pointer_ref,
        "generation_manifest_ref": manifest_ref,
        "generation_id": pointer["generation_id"],
        "inventory_ref": receipt_payload["inventory_ref"],
        "archive_plan_ref": receipt_payload["archive_plan_ref"],
        "cutover_id": receipt_payload["cutover_id"],
        "permanent_marker_path": receipt_payload["permanent_marker_path"],
    }
    if any(marker_payload[field] != value for field, value in expected_marker.items()):
        raise SystemPreconditionError("historical permanent marker binding differs")

    prepared_payload = prepared["payload"]
    expected_prepared = {
        "state": "PREPARED",
        "activation_authorization_ref": authorization_ref,
        "final_cutover_authorization_ref": final_ref,
        "migration_receipt_ref": receipt_ref,
        "target_active_pointer": pointer,
        "target_active_pointer_ref": pointer_ref,
        "permanent_marker_ref": marker_ref,
        "expected_active_pointer_sha256": "EMPTY",
        "prepared_at": auth_payload["prepared_at"],
        "actor_uid": current_uid,
    }
    prepared_body = dict(prepared_payload)
    prepared_body.pop("transaction_id")
    if prepared_body != expected_prepared:
        raise SystemPreconditionError("historical prepared transaction binding differs")
    return {
        "final_authorization": final,
        "activation_authorization": authorization,
        "prepared_transaction": prepared,
        "migration_receipt": receipt,
        "permanent_marker": marker,
    }


__all__ = [
    "INITIAL_ACTIVATION_AUTHORIZATION_CONTRACT_SHA256",
    "INITIAL_ACTIVATION_PREPARED_CONTRACT_SHA256",
    "INITIAL_CUTOVER_GATE_EVIDENCE_CONTRACT_SHA256",
    "INITIAL_FINAL_PREFLIGHT_GATES",
    "INITIAL_FINAL_AUTHORIZATION_CONTRACT_SHA256",
    "INITIAL_GATE_RUNNER_ID",
    "INITIAL_LEGACY_DISPOSITION_CONTRACT_SHA256",
    "INITIAL_MAIN_CHECKOUT_ADOPTION_CONTRACT_SHA256",
    "INITIAL_PERMANENT_MARKER_CONTRACT_SHA256",
    "INITIAL_PRODUCTION_RECEIPT_CONTRACT_SHA256",
    "INITIAL_RELEASE_INSTALL_EVIDENCE_CONTRACT_SHA256",
    "frozen_object_ref",
    "validate_frozen_object_ref",
    "validate_initial_activation_authorization",
    "validate_initial_activation_bundle",
    "validate_initial_activation_prepared",
    "validate_initial_cutover_gate_evidence",
    "validate_initial_final_authorization",
    "validate_initial_legacy_source_disposition",
    "validate_initial_main_checkout_adoption",
    "validate_initial_migration_receipt",
    "validate_initial_permanent_marker",
    "validate_initial_preflight_rows",
    "validate_initial_production_receipt",
    "validate_initial_release_install_evidence",
    "validate_initial_release_install_gate_binding",
]
