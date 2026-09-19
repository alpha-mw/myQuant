"""Deep archived install/claim proof, reachable only through a completed snapshot."""

from pathlib import Path, PurePosixPath
import hashlib
from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.migration.canonical import read_stable_regular_file
from quant_investor.system.release_install import verify_archived_release_install_input
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.market.maintenance_journal import logical_task_claim
from .completed_handoff_snapshot import CompletedHandoffSnapshot, _require_snapshot
from .daily_contract import ContractError, validate_ref
from .maintenance_handoff_contract import handoff_task_date, validate_handoff_shape


def verify_archived_handoff_context(snapshot: CompletedHandoffSnapshot) -> dict:
    _require_snapshot(snapshot)
    snapshot.recheck()
    context = snapshot.document("loop_context")
    handoff = snapshot.document("handoff")
    validate_handoff_shape(handoff)
    bound = None
    if handoff["schema_version"] == "cn-daily-maintenance-handoff.v3":
        from .catchup_binding import read_catchup_binding

        bound = read_catchup_binding(
            workspace=snapshot.workspace, binding_ref=handoff["catchup_binding_ref"]
        )
        if bound["binding"]["execution_request_ref"]["sha256"] != handoff["request_ref"][
            "sha256"
        ] or bound["recipe"] != snapshot.document("recipe"):
            raise ContractError("ARCHIVED_CATCHUP_BINDING_MISMATCH")
    origin = None
    if handoff["schema_version"] == "cn-daily-maintenance-handoff.v4":
        from .automatic_origin import (
            read_automatic_origin,
            recheck_origin,
            require_origin_execution,
        )

        if snapshot.reference("automatic_origin") != handoff["automatic_origin_ref"]:
            raise ContractError("ARCHIVED_AUTOMATIC_ORIGIN_REF_MISMATCH")
        origin = read_automatic_origin(
            workspace=snapshot.workspace,
            reference=handoff["automatic_origin_ref"],
            synthetic=snapshot.document("completion")["synthetic"],
        )
        if origin["origin"]["execution_request_ref"]["sha256"] != handoff["request_ref"][
            "sha256"
        ] or origin["bound"]["recipe"] != snapshot.document("recipe"):
            raise ContractError("ARCHIVED_AUTOMATIC_ORIGIN_MISMATCH")
        require_origin_execution(
            origin,
            request_ref=handoff["request_ref"],
            request=origin["bound"]["request"],
            recipe=snapshot.document("recipe"),
            retained=True,
            sealed_at=handoff["sealed_at"],
        )
    from quant_investor.market.future_calendar_context import validate_loop_context

    validate_loop_context(context)
    if context.get("release_install_input_ref") != snapshot.reference("release_install_input"):
        raise ContractError("ARCHIVED_LOOP_CONTEXT_INVALID")
    release_raw = next(
        raw for role, _, _, raw in snapshot.documents if role == "release_install_input"
    )
    verified = verify_archived_release_install_input(
        release_raw, repository_root=context["release_repository_root"]
    )
    if (
        verified.get("state") != "PASS"
        or verified["release_ref"]["byte_sha256"] != handoff["release_ref"]["sha256"]
    ):
        raise ContractError("ARCHIVED_HANDOFF_RELEASE_MISMATCH")
    installation = parse_canonical_json_bytes(release_raw)["release_install_evidence"]["payload"]
    module = Path(verified["import_origin"]).parent / "market/maintenance_journal.py"
    module_raw = read_stable_regular_file(module, label="archived native maintenance module")
    module_path = str(module.resolve(strict=True))
    reader = SecureSystemStorage(snapshot.workspace)
    started_ref = validate_ref(handoff["maintenance_started_ref"])
    started_raw = reader.read_workspace_file_bytes(started_ref["path"], maximum_bytes=1024 * 1024)
    if started_raw.byte_sha256 != started_ref["sha256"]:
        raise ContractError("ARCHIVED_MAINTENANCE_START_SHA_MISMATCH")
    started = parse_canonical_json_bytes(started_raw.data)
    if started.get("state") != "STARTED" or started.get("mode") != "execute":
        raise ContractError("ARCHIVED_MAINTENANCE_START_INVALID")
    run_date = handoff_task_date(handoff, started)
    key = run_date + "-2020-execute"
    claim_ref = snapshot.reference("logical_claim")
    claim_path = PurePosixPath(claim_ref["path"])
    if len(claim_path.parts) < 4 or claim_path.parts[-3:] != ("logical_tasks", key, "claim.json"):
        raise ContractError("ARCHIVED_SLOT_CLAIM_PATH_INVALID")
    expected = logical_task_claim(logical_key=key, slot="2020", mode="execute")
    expected["installation"] = {
        "python": installation["python_executable"],
        "module": module_path,
        "implementation_sha256": hashlib.sha256(module_raw).hexdigest(),
    }
    claim_raw = next(raw for role, _, _, raw in snapshot.documents if role == "logical_claim")
    if claim_raw != canonical_json_bytes(expected):
        raise ContractError("ARCHIVED_SLOT_CLAIM_IDENTITY_MISMATCH")
    if read_stable_regular_file(module, label="archived native module recheck") != module_raw:
        raise ContractError("ARCHIVED_MODULE_CHANGED")
    if (
        reader.read_workspace_file_bytes(started_ref["path"], maximum_bytes=1024 * 1024)
        != started_raw
    ):
        raise ContractError("ARCHIVED_MAINTENANCE_START_CHANGED")
    snapshot.recheck()
    if bound is not None:
        bound["sources"].recheck()
    if origin is not None:
        recheck_origin(origin)
    return {
        "context": context,
        "installation": verified,
        "claim": {
            "claim_ref": claim_ref,
            "logical_key": key,
            "run_date": run_date,
            "slot": "2020",
            "mode": "execute",
            "maintenance_run_root": str(claim_path.parent.parent.parent),
            "attempt_budget": 2,
            "close_request_budget": 2,
            "new_budget_granted": False,
            "execution_authorized": False,
        },
    }
