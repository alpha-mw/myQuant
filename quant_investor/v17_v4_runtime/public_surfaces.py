"""Read-only public projections and canary-only schedule publication for V17 v4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Final, Mapping

from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_resource_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import load_canonical_resource
from quant_investor.v17_v4_contract.identities import (
    require_opaque_id,
    require_sha256,
    require_utc_timestamp,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field,
)
from quant_investor.v17_v4_runtime.formal_activation import (
    COMPLETION_VERSION,
    FORMAL_OUTPUT_VERSION,
    POINTER_VERSION,
    FormalActivationError,
    FormalActivationService,
    FormalState,
    artifact_ref,
)
from quant_investor.v17_v4_runtime.source_storage import (
    CANARY_ROOT,
    GovernedStore,
    SourceStorageSecurityError,
    canonical_governed_path,
)

PUBLIC_RUN_DTO_VERSION: Final = "myquant.v17.v4.public-run-dto.v1"
CANARY_SNAPSHOT_VERSION: Final = "myquant.v17.v4.canary-public-snapshot.v1"
PUBLIC_SURFACE_RECEIPT_VERSION: Final = (
    "myquant.v17.v4.public-surface-compatibility-receipt.v1"
)
PUBLIC_SURFACES: Final = frozenset({"CLI", "DASHBOARD", "SCHEDULE", "WEB"})
PUBLICATION_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": True,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
NO_SIDE_EFFECTS: Final = {
    "broker_calls": False,
    "execution_calls": False,
    "llm_control_calls": False,
    "order_calls": False,
    "provider_calls": False,
    "selector_writes": False,
    "trade_calls": False,
}
_DASHBOARD_SCHEMA_PATH: Final = (
    Path(__file__).resolve().parents[2]
    / "portfolio_dashboard"
    / "schema"
    / "dashboard_contract.v4.schema.json"
)
_PUBLIC_SURFACE_FILES: Final = {
    "CLI": (
        "quant_investor/cli/main.py",
        "quant_investor/v17_v4_runtime/cli.py",
        "quant_investor/v17_v4_runtime/public_surfaces.py",
    ),
    "DASHBOARD": (
        "portfolio_dashboard/schema/dashboard_contract.v4.schema.json",
        "quant_investor/v17_v4_runtime/public_surfaces.py",
    ),
    "SCHEDULE": (
        (
            "quant_investor/v17_v4_contract/resources/"
            "canary_schedule_policy.v1.json"
        ),
        "quant_investor/v17_v4_runtime/cli.py",
        "quant_investor/v17_v4_runtime/public_surfaces.py",
    ),
    "WEB": (
        "web/models/v17_v4_models.py",
        "web/routers/v17_v4_research.py",
        "web/workspace_app.py",
    ),
}
_V15_COMPATIBILITY_FILES: Final = {
    "CLI": ("quant_investor/cli/main.py",),
    "DASHBOARD": (
        "portfolio_dashboard/schema/dashboard_contract.v3.schema.json",
    ),
    "SCHEDULE": ("quant_investor/automation/daily_runner.py",),
    "WEB": (
        "web/models/research_models.py",
        "web/routers/research.py",
    ),
}


class PublicSurfaceError(RuntimeError):
    """Raised when a public consumer cannot prove an exact formal v4 run."""


def _blocked(reason: str) -> None:
    raise PublicSurfaceError(f"V17_V4_PUBLIC_SURFACE_BLOCKED:{reason}")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _repo_file_ref(
    repo_root: Path,
    relative_path: str,
) -> dict[str, str]:
    candidate = repo_root / relative_path
    try:
        if candidate.is_symlink():
            _blocked("SURFACE_FILE_SYMLINK")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repo_root)
        before = resolved.stat()
        if not stat.S_ISREG(before.st_mode):
            _blocked("SURFACE_FILE_TYPE")
        raw = resolved.read_bytes()
        after = resolved.stat()
    except (OSError, ValueError) as exc:
        raise PublicSurfaceError(
            "V17_V4_PUBLIC_SURFACE_BLOCKED:SURFACE_FILE_READ"
        ) from exc
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity != after_identity or len(raw) != after.st_size:
        _blocked("SURFACE_FILE_CHANGED")
    return {
        "byte_sha256": _sha(raw),
        "relative_path": relative_path,
    }


def _file_refs(
    repo_root: Path,
    relative_paths: tuple[str, ...],
) -> list[dict[str, str]]:
    return [
        _repo_file_ref(repo_root, relative_path)
        for relative_path in sorted(relative_paths)
    ]


def _verify_v15_public_contracts(repo_root: Path) -> None:
    cli = (
        repo_root / "quant_investor/cli/main.py"
    ).read_text(encoding="utf-8")
    dashboard = (
        repo_root
        / "portfolio_dashboard/schema/dashboard_contract.v3.schema.json"
    ).read_text(encoding="utf-8")
    web_model = (
        repo_root / "web/models/research_models.py"
    ).read_text(encoding="utf-8")
    web_route = (
        repo_root / "web/routers/research.py"
    ).read_text(encoding="utf-8")
    if (
        'default="v15"' not in cli
        or '"dashboard_contract.v3"' not in dashboard
        or "class ResearchRunRequest" not in web_model
        or 'prefix="/api/research"' not in web_route
    ):
        _blocked("V15_PUBLIC_CONTRACT_DRIFT")


def build_public_surface_compatibility_receipts(
    repo_root: str | Path,
    workspace_root: str | Path,
    *,
    strategy_id: str,
    created_at: str,
) -> tuple[dict[str, Any], ...]:
    """Hash-bind all four opt-in surfaces and their unchanged V15 counterparts."""

    root = Path(repo_root).resolve()
    if not root.is_absolute() or not root.is_dir():
        _blocked("REPO_ROOT")
    strategy = require_opaque_id(strategy_id, label="strategy_id")
    recorded_at = require_utc_timestamp(created_at, label="created_at")
    public_run = resolve_public_run(
        workspace_root,
        strategy_id=strategy,
        surface="CLI",
    )
    cutoff = str(public_run["cutoff"])
    if recorded_at < cutoff:
        _blocked("RECEIPT_PREDATES_CUTOFF")
    pointer_ref = dict(public_run["formal_active_pointer_ref"])
    _verify_v15_public_contracts(root)
    receipts: list[dict[str, Any]] = []
    for surface in sorted(PUBLIC_SURFACES):
        receipt = seal_semantic(
            {
                "authority": {
                    **PUBLICATION_AUTHORITY,
                    "formal_research_publication": False,
                },
                "canary_output_only": True,
                "created_at": recorded_at,
                "cutoff": cutoff,
                "explicit_opt_in": True,
                "formal_active_pointer_ref": pointer_ref,
                "protocol_version": PROTOCOL_VERSION,
                "read_only_view": True,
                "receipt_id": (
                    "public-surface-"
                    f"{surface.lower()}-"
                    f"{pointer_ref['byte_sha256'][:16]}"
                ),
                "status": "ACCEPTED",
                "strategy_id": strategy,
                "surface": surface,
                "surface_file_refs": _file_refs(
                    root,
                    _PUBLIC_SURFACE_FILES[surface],
                ),
                "v15_compatibility_refs": _file_refs(
                    root,
                    _V15_COMPATIBILITY_FILES[surface],
                ),
                "v15_default_unchanged": True,
                "version": PUBLIC_SURFACE_RECEIPT_VERSION,
                "view_label": "CANARY",
            }
        )
        validate_artifact(receipt)
        receipts.append(receipt)
    return tuple(receipts)


def _load_referenced_artifact(
    service: FormalActivationService,
    reference: Mapping[str, Any],
    *,
    expected_version: str,
) -> dict[str, Any]:
    try:
        raw = service.artifact_loader(reference)
        document = load_canonical_resource(raw, label=expected_version)
        if type(document) is not dict:
            _blocked("ARTIFACT_ROOT")
        validated = validate_artifact(
            document,
            artifact_loader=service.artifact_loader,
        )
        identity_field = artifact_identity_field(expected_version)
    except (OSError, TypeError, ValueError) as exc:
        raise PublicSurfaceError(
            "V17_V4_PUBLIC_SURFACE_BLOCKED:ARTIFACT_VALIDATION"
        ) from exc
    if (
        validated.version != expected_version
        or document.get("version") != expected_version
        or document.get(identity_field) != reference.get("artifact_id")
        or document.get("semantic_sha256")
        != reference.get("semantic_sha256")
        or _sha(raw) != reference.get("byte_sha256")
    ):
        _blocked("ARTIFACT_REFERENCE_MISMATCH")
    return dict(document)


def _formal_refs(
    state: FormalState,
) -> tuple[dict[str, str], dict[str, str]]:
    if state.intent is None or state.pointer is None or state.completion is None:
        _blocked("FORMAL_CHAIN_INCOMPLETE")
    strategy_id = str(state.intent["strategy_id"])
    intent_id = str(state.intent["intent_id"])
    root = (
        "results/v17_v4_formal_research/strategies/"
        f"{strategy_id}"
    )
    pointer_ref = artifact_ref(
        state.pointer,
        relative_path=f"{root}/_active.json",
    )
    completion_ref = artifact_ref(
        state.completion,
        relative_path=(
            f"{root}/completion_receipts/{intent_id}.json"
        ),
    )
    return pointer_ref, completion_ref


def _resolve_documents(
    workspace_root: str | Path,
    strategy_id: str,
) -> tuple[
    FormalState,
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, str],
]:
    service = FormalActivationService(workspace_root)
    try:
        state = service.resolve(strategy_id)
    except FormalActivationError as exc:
        raise PublicSurfaceError(
            "V17_V4_PUBLIC_SURFACE_BLOCKED:FORMAL_REVALIDATION"
        ) from exc
    if (
        state.status != "FORMAL_ACTIVE"
        or state.intent is None
        or state.pointer is None
        or state.completion is None
    ):
        _blocked("FORMAL_ACTIVE_REQUIRED")
    formal = _load_referenced_artifact(
        service,
        state.intent["formal_output_ref"],
        expected_version=FORMAL_OUTPUT_VERSION,
    )
    portfolio = _load_referenced_artifact(
        service,
        state.intent["portfolio_output_ref"],
        expected_version="myquant.v17.v4.portfolio-output.v1",
    )
    pointer_ref, completion_ref = _formal_refs(state)
    if (
        state.pointer["version"] != POINTER_VERSION
        or state.completion["version"] != COMPLETION_VERSION
        or state.completion["pointer_ref"] != pointer_ref
        or state.intent["portfolio_output_ref"]
        not in formal["evidence_refs"]
        or portfolio["strategy_id"] != strategy_id
        or portfolio["cutoff"] != formal["cutoff"]
    ):
        _blocked("FORMAL_PUBLIC_BINDING")
    return state, formal, portfolio, pointer_ref, completion_ref


def resolve_public_run(
    workspace_root: str | Path,
    *,
    strategy_id: str,
    surface: str,
) -> dict[str, Any]:
    """Resolve one exact FORMAL_ACTIVE run without writing or choosing defaults."""

    strategy = require_opaque_id(strategy_id, label="strategy_id")
    normalized_surface = str(surface).strip().upper()
    if normalized_surface not in PUBLIC_SURFACES:
        _blocked("SURFACE_UNSUPPORTED")
    state, formal, portfolio, pointer_ref, completion_ref = _resolve_documents(
        workspace_root,
        strategy,
    )
    assert state.intent is not None
    targets = [
        {
            "current_target": row["current_target"],
            "final_target": row["final_target"],
            "lane": row["lane"],
            "symbol": row["symbol"],
        }
        for row in portfolio["targets"]
    ]
    document = seal_semantic(
        {
            "authority": dict(PUBLICATION_AUTHORITY),
            "cash_weight": portfolio["cash_weight"],
            "cutoff": formal["cutoff"],
            "formal_activation_receipt_ref": completion_ref,
            "formal_active_pointer_ref": pointer_ref,
            "formal_output_ref": dict(state.intent["formal_output_ref"]),
            "gross_weight": portfolio["gross_weight"],
            "is_default": False,
            "portfolio_output_ref": dict(
                state.intent["portfolio_output_ref"]
            ),
            "protocol_version": PROTOCOL_VERSION,
            "read_only": True,
            "run_id": portfolio["run_id"],
            "side_effects": dict(NO_SIDE_EFFECTS),
            "state": "FORMAL_ACTIVE",
            "strategy_id": strategy,
            "surface": normalized_surface,
            "targets": targets,
            "version": PUBLIC_RUN_DTO_VERSION,
            "view_label": "CANARY",
        }
    )
    validate_artifact(document)
    return document


def build_dashboard_contract_v4(
    workspace_root: str | Path,
    *,
    strategy_id: str,
) -> dict[str, Any]:
    """Project the same exact public DTO into the separate Dashboard v4 contract."""

    public_run = resolve_public_run(
        workspace_root,
        strategy_id=strategy_id,
        surface="DASHBOARD",
    )
    try:
        schema_raw = _DASHBOARD_SCHEMA_PATH.read_bytes()
        schema = json.loads(schema_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicSurfaceError(
            "V17_V4_PUBLIC_SURFACE_BLOCKED:DASHBOARD_SCHEMA"
        ) from exc
    payload = {
        "authority": dict(public_run["authority"]),
        "cash_weight": public_run["cash_weight"],
        "cutoff": public_run["cutoff"],
        "formal_activation_receipt_ref": dict(
            public_run["formal_activation_receipt_ref"]
        ),
        "formal_active_pointer_ref": dict(
            public_run["formal_active_pointer_ref"]
        ),
        "gross_weight": public_run["gross_weight"],
        "is_default": False,
        "positions": [dict(row) for row in public_run["targets"]],
        "protocol_version": PROTOCOL_VERSION,
        "read_only": True,
        "run_id": public_run["run_id"],
        "schema_sha256": _sha(schema_raw),
        "schema_version": "dashboard_contract.v4",
        "status": "canary",
        "strategy_id": public_run["strategy_id"],
        "v15_run_readiness": None,
        "v17_v4_run_readiness": {
            "formal_activation_receipt_ref": dict(
                public_run["formal_activation_receipt_ref"]
            ),
            "formal_active_pointer_ref": dict(
                public_run["formal_active_pointer_ref"]
            ),
            "state": "FORMAL_ACTIVE",
        },
        "view_label": "CANARY",
    }
    if (
        type(schema) is not dict
        or schema.get("$id")
        != "https://myquant.local/schema/dashboard_contract.v4.schema.json"
        or schema.get("additionalProperties") is not False
        or type(schema.get("properties")) is not dict
        or schema["properties"].get("schema_version")
        != {"const": "dashboard_contract.v4"}
        or schema["properties"].get("protocol_version")
        != {"const": PROTOCOL_VERSION}
        or set(schema.get("required", ())) != set(payload)
    ):
        _blocked("DASHBOARD_SCHEMA_IDENTITY")
    return payload


class _CanaryPublicWriter(GovernedStore):
    """Restrict schedule publication to immutable canary snapshots."""

    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        path = canonical_governed_path(value)
        parts = path.parts
        prefix = (
            "results",
            "v17_v4_canary",
            "strategies",
        )
        if (
            parts[:3] != prefix
            or len(parts) != 7
            or parts[4:6] != ("public_snapshots", "sessions")
            or not parts[6].endswith(".json")
        ):
            raise SourceStorageSecurityError(
                "path is outside the v4 canary public snapshot whitelist"
            )
        try:
            require_opaque_id(parts[3], label="strategy_id")
            require_opaque_id(parts[6][:-5], label="session_id")
        except ValueError as exc:
            raise SourceStorageSecurityError(
                "canary public snapshot path identity is invalid"
            ) from exc
        return path


def publish_canary_snapshot(
    workspace_root: str | Path,
    *,
    strategy_id: str,
    session_id: str,
    created_at: str,
    expected_formal_pointer_sha256: str,
) -> dict[str, Any]:
    """Write exactly one schedule artifact under the V17 v4 canary root."""

    strategy = require_opaque_id(strategy_id, label="strategy_id")
    session = require_opaque_id(session_id, label="session_id")
    created = require_utc_timestamp(created_at, label="created_at")
    expected = require_sha256(
        expected_formal_pointer_sha256,
        label="expected_formal_pointer_sha256",
    )
    public_run = resolve_public_run(
        workspace_root,
        strategy_id=strategy,
        surface="SCHEDULE",
    )
    pointer_ref = public_run["formal_active_pointer_ref"]
    if pointer_ref["byte_sha256"] != expected:
        _blocked("FORMAL_POINTER_PREVALUE")
    snapshot = seal_semantic(
        {
            "authority": dict(PUBLICATION_AUTHORITY),
            "created_at": created,
            "cutoff": public_run["cutoff"],
            "formal_activation_receipt_ref": dict(
                public_run["formal_activation_receipt_ref"]
            ),
            "formal_active_pointer_ref": dict(pointer_ref),
            "is_default": False,
            "protocol_version": PROTOCOL_VERSION,
            "public_run": public_run,
            "read_only": True,
            "session_id": session,
            "snapshot_id": f"canary-public-{session}",
            "strategy_id": strategy,
            "version": CANARY_SNAPSHOT_VERSION,
            "view_label": "CANARY",
        }
    )
    validate_artifact(snapshot)
    relative_path = (
        CANARY_ROOT
        / "strategies"
        / strategy
        / "public_snapshots"
        / "sessions"
        / f"{session}.json"
    )
    write = _CanaryPublicWriter(workspace_root).write_exact_once(
        relative_path,
        canonical_resource_bytes(snapshot),
    )
    return {
        "byte_sha256": write.byte_sha256,
        "created": write.created,
        "relative_path": write.relative_path,
        "snapshot": snapshot,
    }


__all__ = [
    "CANARY_SNAPSHOT_VERSION",
    "NO_SIDE_EFFECTS",
    "PUBLIC_RUN_DTO_VERSION",
    "PUBLIC_SURFACE_RECEIPT_VERSION",
    "PUBLIC_SURFACES",
    "PublicSurfaceError",
    "build_dashboard_contract_v4",
    "build_public_surface_compatibility_receipts",
    "publish_canary_snapshot",
    "resolve_public_run",
]
