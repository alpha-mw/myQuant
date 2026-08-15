from __future__ import annotations

import os
from typing import Any

from quant_investor.contracts import canonical_json_bytes
from quant_investor.migration.custody import build_authority_archive_plan
from quant_investor.migration.migration import (
    build_initial_active_pointer,
    build_pre_cas_migration_receipt,
)
from quant_investor.system.activation import build_activation_authorization
from quant_investor.system.store import SystemStore

from test_unified_migration_custody import _inventory, _workspace


def prepare_migration_context(
    store: SystemStore,
    *,
    cutover_id: str = "test-initial-activation",
    created_at: str = "2026-08-14T00:00:00Z",
) -> dict[str, Any]:
    root = store.workspace_root
    rules_path = _workspace(root)
    inventory = _inventory(root)
    archive_plan = build_authority_archive_plan(
        root,
        inventory,
        cutover_id=cutover_id,
        created_at=created_at,
        rules_path=rules_path,
    )
    return {
        "archive_plan": archive_plan,
        "cutover_id": cutover_id,
        "inventory": inventory,
        "rules_path": rules_path,
    }


def prepare_initial_activation(
    store: SystemStore,
    generation: dict[str, Any],
    release_ref: dict[str, str],
    *,
    cutover_id: str = "test-initial-activation",
    prepared_at: str = "2026-08-14T00:00:00Z",
    activated_at: str = "2026-08-14T00:00:01Z",
    migration_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = store.workspace_root
    context = migration_context or prepare_migration_context(
        store,
        cutover_id=cutover_id,
        created_at=prepared_at,
    )
    if context["cutover_id"] != cutover_id:
        raise AssertionError("test migration context cutover mismatch")
    rules_path = context["rules_path"]
    inventory = context["inventory"]
    archive_plan = context["archive_plan"]
    pointer = build_initial_active_pointer(
        generation["manifest"],
        activated_at=activated_at,
        os_actor=f"uid:{os.geteuid()}",
    )
    receipt = build_pre_cas_migration_receipt(
        root,
        inventory,
        archive_plan,
        pointer,
        generation["manifest"],
        cutover_id=cutover_id,
        created_at=prepared_at,
        rules_path=rules_path,
    )
    authorization = build_activation_authorization(
        migration_receipt=receipt,
        target_active_pointer=pointer,
        target_generation_manifest=generation["manifest"],
        deployed_release_ref=release_ref,
        prepared_at=prepared_at,
        actor_uid=os.geteuid(),
    )
    return {
        "activation_authorization_raw": canonical_json_bytes(authorization),
        "deployed_release_ref": release_ref,
        "migration_receipt_raw": canonical_json_bytes(receipt),
        "target_active_pointer_raw": canonical_json_bytes(pointer),
    }


def activate_initial(
    store: SystemStore,
    generation: dict[str, Any],
    release_ref: dict[str, str],
    **kwargs: Any,
) -> dict[str, Any]:
    prepared = prepare_initial_activation(
        store, generation, release_ref, **kwargs
    )
    return store.activate_initial_generation(**prepared)
