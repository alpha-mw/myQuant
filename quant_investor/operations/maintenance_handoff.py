"""Publish the early, non-authorizing recovery anchor from native core evidence."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib

from quant_investor.contracts import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
    validate_artifact,
)
from quant_investor.system.store import object_ref_for_artifact
from quant_investor.market.daily_factor_loop import read_factor_loop_context
from quant_investor.factors.production_rollover import validate_daily_maintenance_receipt
from quant_investor.factors.production_authority import FactorProductionStore
from quant_investor.system.storage import SecureSystemStorage
from .core_handoff import inspect_core_handoff
from .daily_contract import ContractError, GRAPH_SHA256, utc_stamp, validate_ref
from .daily_journal import DailyJournal, FALSE_AUTHORITY
from .execution_recipe import validate_execution_recipe
from .production_request import validate_production_request
from .slot_claim import inspect_handoff_slot_claim
from .prediction_policy import validate_prediction_policy
from .bootstrap import validate_bootstrap_declaration

from .maintenance_handoff_contract import (
    LEGACY_SCHEMA,
    SCHEMA,
    HISTORICAL_SCHEMA,
    AUTOMATIC_SCHEMA,
    HANDOFF_FIELDS,
    HANDOFF_V2_FIELDS as _HANDOFF_V2_FIELDS,
    HANDOFF_V3_FIELDS as _HANDOFF_V3_FIELDS,
    validate_handoff_shape,
    handoff_task_date,
)

HANDOFF_V2_FIELDS = _HANDOFF_V2_FIELDS
HANDOFF_V3_FIELDS = _HANDOFF_V3_FIELDS


def _historical_core_evidence(*, root, core_ref, checkpoint, recipe, read_file):
    from quant_investor.market.historical_session import (
        CORE_SCHEMA,
        CORE_FIELDS,
        read_historical_core,
        validate_historical_core_timing,
    )
    from quant_investor.market.close_session_authority import CloseSessionAuthorityError
    from quant_investor.market.tushare_transport import TushareHttpsError

    if checkpoint.get("schema_version") != CORE_SCHEMA or set(checkpoint) != CORE_FIELDS:
        raise ContractError("MAINTENANCE_HANDOFF_CORE_CONTRACT_INVALID")
    try:
        evidence = read_historical_core(
            attempt_root=(root / core_ref["path"]).parent,
            proof_ref=checkpoint["historical_session_ref"],
            close_ref=checkpoint["close_session_receipt_ref"],
            started_ref=checkpoint["started_ref"],
            target=checkpoint["target_date"],
            read=read_file,
        )
        proof = evidence["historical_session"]
        validate_historical_core_timing(
            observed_at=proof["observed_at"],
            started_at=evidence["started_at"],
            sealed_at=checkpoint["sealed_at"],
        )
    except (OSError, ValueError, CloseSessionAuthorityError, TushareHttpsError) as exc:
        raise ContractError("HISTORICAL_HANDOFF_CORE_INVALID") from exc
    previous = recipe["previous_completion_ref"]
    if (
        previous is None
        or recipe["bootstrap_ref"] is not None
        or Path(previous["path"]).parent.name != proof["previous_trade_date"]
    ):
        raise ContractError("HISTORICAL_HANDOFF_PREDECESSOR_INVALID")
    return proof


def publish_maintenance_handoff(
    *,
    workspace: str,
    request_ref: dict,
    state: dict,
    _catchup_binding_ref=None,
    _automatic_origin_ref=None,
) -> dict:
    """Invoked by the fixed core hook after its day lock exits, before auxiliaries.

    The coordinator validates recipe policies/anchor before invoking maintenance.
    This publisher proves the resulting native core bindings and grants no writes
    or investment authority to consumers by itself.
    """
    root = Path(workspace).resolve(strict=True)
    storage = SecureSystemStorage(workspace)
    observed = {}

    def local(ref):
        path = Path(ref["path"])
        if path.is_absolute():
            path = path.relative_to(root)
        return validate_ref({"path": str(path), "sha256": ref["sha256"]})

    def raw(ref):
        ref = local(ref)
        value = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=32 * 1024 * 1024)
        if value.byte_sha256 != ref["sha256"]:
            raise ContractError("MAINTENANCE_HANDOFF_SOURCE_SHA_MISMATCH")
        observed[ref["path"]] = value.data
        return value.data

    def read(ref):
        return parse_canonical_json_bytes(raw(ref))

    def read_file(path, *, label):
        relative = str(Path(path).relative_to(root))
        content = storage.read_workspace_file_bytes(relative, maximum_bytes=32 * 1024 * 1024).data
        observed[relative] = content
        return content

    request_ref = local(request_ref)
    request = read(request_ref)
    request = validate_production_request(
        request, release_install_ref=request["release_install_ref"]
    )
    if request["action"] != "EXECUTE":
        raise ContractError("MAINTENANCE_HANDOFF_EXECUTE_REQUIRED")
    from .production_request import HISTORICAL_SCHEMA as REQUEST_HISTORICAL_SCHEMA
    from .catchup_binding import read_catchup_binding, verify_bound_historical_core

    if (request["schema_version"] == REQUEST_HISTORICAL_SCHEMA) != (
        _catchup_binding_ref is not None
    ):
        raise ContractError("MAINTENANCE_HANDOFF_CATCHUP_BINDING_REQUIRED")
    bound = None
    if _catchup_binding_ref is not None:
        bound = read_catchup_binding(workspace=workspace, binding_ref=_catchup_binding_ref)
        if bound["binding"]["execution_request_ref"] != request_ref or bound["request"] != request:
            raise ContractError("MAINTENANCE_HANDOFF_CATCHUP_REQUEST_MISMATCH")
    recipe_ref = request["recipe_ref"]
    recipe = validate_execution_recipe(read(recipe_ref), request=request)
    origin = None
    if _automatic_origin_ref is not None:
        from .automatic_origin import (
            read_automatic_origin,
            require_origin_execution,
            require_live_origin,
        )

        if bound is not None or recipe["schema_version"] != "cn-daily-execute-recipe.v6":
            raise ContractError("MAINTENANCE_HANDOFF_AUTOMATIC_PROFILE_INVALID")
        raw(_automatic_origin_ref)
        origin = read_automatic_origin(workspace=workspace, reference=_automatic_origin_ref)
        require_origin_execution(origin, request_ref=request_ref, request=request, recipe=recipe)
        require_live_origin(origin, workspace=workspace)
    if recipe["bootstrap_ref"] is not None:
        validate_bootstrap_declaration(read(recipe["bootstrap_ref"]), recipe=recipe)
    policy_ref = recipe["policy_refs"]["prospective"]
    if policy_ref is not None:
        validate_prediction_policy(read(policy_ref), trade_date=request["target_trade_date"])
    context_ref = recipe["factor_loop_context_ref"]
    raw(context_ref)
    context, installation = read_factor_loop_context(
        workspace_root=workspace,
        context_path=str(root / context_ref["path"]),
        context_sha256=context_ref["sha256"],
    )
    if context["release_install_input_ref"] != recipe["release_install_ref"]:
        raise ContractError("MAINTENANCE_HANDOFF_INSTALL_BINDING_MISMATCH")
    day = request["target_trade_date"]
    core_ref = local(state["core_checkpoint_ref"])
    checkpoint = read(core_ref)
    from quant_investor.market.historical_session import CORE_SCHEMA, CORE_FIELDS

    historical = type(checkpoint) is dict and checkpoint.get("schema_version") == CORE_SCHEMA
    if (
        type(checkpoint) is not dict
        or (historical and set(checkpoint) != CORE_FIELDS)
        or (
            not historical
            and (
                checkpoint.get("schema_version") != "cn-daily-maintenance-core.v1"
                or "historical_session_ref" in checkpoint
            )
        )
    ):
        raise ContractError("MAINTENANCE_HANDOFF_CORE_CONTRACT_INVALID")
    if historical != (bound is not None):
        raise ContractError("MAINTENANCE_HANDOFF_CATCHUP_CORE_REQUIRED")
    native = validate_daily_maintenance_receipt(
        workspace_root=root,
        receipt_path=root / core_ref["path"],
        expected_receipt_sha256=core_ref["sha256"],
    )
    if type(checkpoint) is not dict or not {
        "logical_claim_ref",
        "started_ref",
        "close_session_receipt_ref",
        "stage_results",
    } <= set(checkpoint):
        raise ContractError("MAINTENANCE_HANDOFF_NATIVE_REFS_MISSING")
    if native["target_date"] != day:
        raise ContractError("MAINTENANCE_HANDOFF_TARGET_MISMATCH")
    historical_ref = None
    if historical:
        proof = _historical_core_evidence(
            root=root, core_ref=core_ref, checkpoint=checkpoint, recipe=recipe, read_file=read_file
        )
        historical_ref = local(checkpoint["historical_session_ref"])
        raw(historical_ref)
    claim_ref = local(checkpoint["logical_claim_ref"])
    started_ref = local(checkpoint["started_ref"])
    started = read(started_ref)
    raw(claim_ref)
    handoff_ref = local(state["core_handoff_ref"])
    handoff = inspect_core_handoff(
        workspace=workspace,
        trade_date=day,
        handoff_ref=handoff_ref,
        release_ref=recipe["release_ref"],
    )
    raw(handoff_ref)
    factor_ref = handoff["factor_pointer_ref"]
    snapshot = FactorProductionStore(workspace).inspect_recorded_research_inputs(
        pointer_raw=raw(factor_ref),
        expected_pointer_sha256=factor_ref["sha256"],
        expected_trade_date=day,
    )
    calendar_ref = local(checkpoint["close_session_receipt_ref"])
    calendar = read(calendar_ref)
    calendar_raw_ref = local(
        {"path": calendar["raw_response_path"], "sha256": calendar["raw_response_sha256"]}
    )
    raw(calendar_raw_ref)
    if bound is not None:
        verify_bound_historical_core(
            derived=bound, proof=proof, calendar=calendar, raw=raw(calendar_raw_ref)
        )
    if not historical and recipe["previous_completion_ref"] is not None:
        _current_recipe_predecessor(recipe, day, calendar, raw(calendar_raw_ref))
    market = next(
        row["evidence"] for row in checkpoint["stage_results"] if row["stage"] == "MARKET"
    )
    if type(market) is not dict or not {
        "pointer_path",
        "pointer_sha256",
        "snapshot_manifest_path",
        "snapshot_manifest_sha256",
    } <= set(market):
        raise ContractError("MAINTENANCE_HANDOFF_MARKET_REFS_MISSING")
    market_ref = local({"path": market["pointer_path"], "sha256": market["pointer_sha256"]})
    snapshot_ref = local(
        {"path": market["snapshot_manifest_path"], "sha256": market["snapshot_manifest_sha256"]}
    )
    market_raw = raw(market_ref)
    raw(snapshot_ref)
    if (
        market_ref["sha256"] != snapshot["market_pointer_sha256"]
        or snapshot_ref["sha256"] != snapshot["market_manifest_sha256"]
    ):
        raise ContractError("MAINTENANCE_HANDOFF_FACTOR_MARKET_MISMATCH")
    release = validate_artifact(read(recipe["release_ref"]))
    release_object_ref = object_ref_for_artifact(release)
    if (
        release_object_ref != installation["release_ref"]
        or release_object_ref != snapshot["factor_generation"]["payload"]["deployed_release_ref"]
    ):
        raise ContractError("MAINTENANCE_HANDOFF_RELEASE_MISMATCH")
    raw(recipe["release_install_ref"])
    lower_bound = max(
        [utc_stamp(started["started_at"])]
        + [utc_stamp(read(ref)["finished_at"]) for ref in handoff["node_terminal_refs"].values()]
    )
    if historical:
        lower_bound = max(lower_bound, utc_stamp(checkpoint["sealed_at"]))
    if origin is not None:
        lower_bound = max(lower_bound, utc_stamp(origin["resolution"]["resolution"]["resolved_at"]))
    if lower_bound > datetime.now(timezone.utc):
        raise ContractError("MAINTENANCE_HANDOFF_FUTURE_CORE")
    journal = DailyJournal(workspace, day)
    execution = journal.root / "executions" / request_ref["sha256"]
    path = str(execution / "maintenance-handoff.v1.json")

    def retained_ref(name, content):
        digest = hashlib.sha256(content).hexdigest()
        return {"path": str(execution / "inputs" / f"{name}-{digest}.json"), "sha256": digest}

    base = {
        "schema_version": HISTORICAL_SCHEMA if historical else SCHEMA,
        "trade_date": day,
        "graph_sha256": GRAPH_SHA256,
        "request_ref": retained_ref("request", observed[request_ref["path"]]),
        "recipe_ref": retained_ref("recipe", observed[recipe_ref["path"]]),
        "release_ref": recipe["release_ref"],
        "release_install_ref": recipe["release_install_ref"],
        "logical_claim_ref": claim_ref,
        "maintenance_started_ref": started_ref,
        "maintenance_core_ref": core_ref,
        "calendar_ref": calendar_ref,
        "raw_calendar_ref": calendar_raw_ref,
        "core_handoff_ref": handoff_ref,
        "factor_pointer_ref": factor_ref,
        "market_pointer_ref": retained_ref("market-pointer", market_raw),
        "market_snapshot_ref": snapshot_ref,
        "prospective_policy_ref": (
            retained_ref("prospective-policy", observed[policy_ref["path"]])
            if policy_ref is not None
            else None
        ),
        "authority": FALSE_AUTHORITY,
    }
    if historical:
        base["historical_session_ref"] = historical_ref
        base["catchup_binding_ref"] = _catchup_binding_ref
    elif origin is not None:
        base["schema_version"] = AUTOMATIC_SCHEMA
        base["automatic_origin_ref"] = _automatic_origin_ref
    inspect_handoff_slot_claim(
        workspace=workspace,
        claim_ref=claim_ref,
        handoff={**base, "sealed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
        started=started,
    )
    with journal.locked():
        if bound is not None:
            bound["sources"].recheck()
        if origin is not None:
            require_live_origin(origin, workspace=workspace)
        occupied = journal.storage.read(path)
        if (
            occupied is not None
            and parse_canonical_json_bytes(occupied.data).get("schema_version") == LEGACY_SCHEMA
        ):
            raise ContractError("MAINTENANCE_HANDOFF_LEGACY_PATH_OCCUPIED")
        if (
            journal.storage.read(str(journal.root / "completion.v1.json")) is not None
            and journal.storage.read(path) is None
        ):
            raise ContractError("MAINTENANCE_HANDOFF_EOD_ALREADY_COMPLETED")
        # Recheck all observed producer/input bytes under the publication lock.
        for source, content in observed.items():
            if (
                storage.read_workspace_file_bytes(source, maximum_bytes=32 * 1024 * 1024).data
                != content
            ):
                raise ContractError("MAINTENANCE_HANDOFF_SOURCE_CHANGED")

        retained_sources = [
            ("request", observed[request_ref["path"]]),
            ("recipe", observed[recipe_ref["path"]]),
            ("market-pointer", market_raw),
        ]
        if policy_ref is not None:
            retained_sources.append(("prospective-policy", observed[policy_ref["path"]]))
        for name, content in retained_sources:
            destination = retained_ref(name, content)
            saved = journal.storage.write(destination["path"], content)
            if saved.byte_sha256 != destination["sha256"]:
                raise ContractError("MAINTENANCE_HANDOFF_RETAINED_SHA_MISMATCH")

        old = journal.storage.read(path)
        if old is not None:
            value = parse_canonical_json_bytes(old.data)
            if set(value) != set(base) | {"sealed_at"} or any(
                value[k] != v for k, v in base.items()
            ):
                raise ContractError("MAINTENANCE_HANDOFF_IMMUTABLE_CONFLICT")
            if not lower_bound <= utc_stamp(value["sealed_at"]) <= datetime.now(timezone.utc):
                raise ContractError("MAINTENANCE_HANDOFF_FUTURE_TIME")
            return {"path": path, "sha256": old.byte_sha256}
        value = {**base, "sealed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        saved = journal.storage.write(path, canonical_json_bytes(value))
        return {"path": saved.relative_path, "sha256": saved.byte_sha256}


def maintenance_handoff_hook(*, workspace: str, request_ref: dict):
    """Code-owned closure for DailyFactorLoop; never deserialized from a request."""

    def publish(state):
        ref = publish_maintenance_handoff(workspace=workspace, request_ref=request_ref, state=state)
        return {"status": "SEALED", "maintenance_handoff_ref": ref}

    return publish


def read_maintenance_handoff(*, workspace: str, handoff_ref: dict) -> dict:
    """Unfinished execution reader always requires the original running installation."""
    return _read_maintenance_handoff(workspace=workspace, handoff_ref=handoff_ref)


def read_recorded_maintenance_handoff(completed_snapshot) -> dict:
    """Completed-history entry accepts only the validator's immutable capability."""
    from .completed_handoff_snapshot import _require_snapshot
    from .archived_handoff_context import verify_archived_handoff_context

    _require_snapshot(completed_snapshot)
    completed_snapshot.recheck()
    archived = verify_archived_handoff_context(completed_snapshot)
    result = _read_maintenance_handoff(
        workspace=completed_snapshot.workspace,
        handoff_ref=completed_snapshot.reference("handoff"),
        completed_snapshot=completed_snapshot,
        archived=archived,
    )
    completed_snapshot.recheck()
    return result


def _read_maintenance_handoff(
    *, workspace: str, handoff_ref: dict, completed_snapshot=None, archived=None
) -> dict:
    """Replay retained evidence without reading the mutable Market/PIT heads.

    Native downstream adapters still validate their own business inputs. Auxiliary
    stage records are intentionally left to the separate maintenance-lock guard.
    """
    from pathlib import PurePosixPath
    from .daily_journal import _validate_day
    from quant_investor.market.close_session_authority import replay_close_session_authority

    root = Path(workspace).resolve(strict=True)
    storage = SecureSystemStorage(workspace)
    observed = {}
    captured = {}
    if completed_snapshot is not None:
        from .completed_handoff_snapshot import _require_snapshot

        _require_snapshot(completed_snapshot)
        if (
            completed_snapshot.workspace != str(root)
            or completed_snapshot.reference("handoff") != handoff_ref
        ):
            raise ContractError("RECORDED_HANDOFF_CAPABILITY_MISMATCH")
        captured = {path: (sha, raw) for _, path, sha, raw in completed_snapshot.documents}

    def local(ref):
        path = Path(ref["path"])
        if path.is_absolute():
            path = path.relative_to(root)
        return validate_ref({"path": str(path), "sha256": ref["sha256"]})

    def raw(ref):
        checked = validate_ref(ref)
        if checked["path"] in captured:
            sha, content = captured[checked["path"]]
            if sha != checked["sha256"]:
                raise ContractError("RECORDED_HANDOFF_CAPTURED_SHA_MISMATCH")
            observed[checked["path"]] = content
            return content
        value = storage.read_workspace_file_bytes(checked["path"], maximum_bytes=32 * 1024 * 1024)
        if value.byte_sha256 != checked["sha256"]:
            raise ContractError("MAINTENANCE_HANDOFF_READBACK_SHA_MISMATCH")
        observed[checked["path"]] = value.data
        return value.data

    def read(ref):
        return parse_canonical_json_bytes(raw(ref))

    value = read(handoff_ref)
    validate_handoff_shape(value)
    historical = value["schema_version"] == HISTORICAL_SCHEMA

    def read_file(path, *, label):
        relative = str(Path(path).relative_to(root))
        content = storage.read_workspace_file_bytes(relative, maximum_bytes=32 * 1024 * 1024).data
        observed[relative] = content
        return content

    if historical:
        validate_ref(value["historical_session_ref"])
        raw(value["historical_session_ref"])
    day = value["trade_date"]
    _validate_day(day)
    sealed = utc_stamp(value["sealed_at"])
    if sealed > datetime.now(timezone.utc):
        raise ContractError("MAINTENANCE_HANDOFF_FUTURE_TIME")
    for name in HANDOFF_FIELDS:
        if name.endswith("_ref"):
            validate_ref(value[name])
    execution = DailyJournal(workspace, day).root / "executions" / value["request_ref"]["sha256"]
    if handoff_ref["path"] != str(execution / "maintenance-handoff.v1.json"):
        raise ContractError("MAINTENANCE_HANDOFF_READBACK_PATH_INVALID")
    for field, prefix in (
        ("request_ref", "request"),
        ("recipe_ref", "recipe"),
        ("market_pointer_ref", "market-pointer"),
    ):
        if value[field]["path"] != str(
            execution / "inputs" / f"{prefix}-{value[field]['sha256']}.json"
        ):
            raise ContractError("MAINTENANCE_HANDOFF_RETAINED_PATH_INVALID")
    for name in HANDOFF_FIELDS:
        if name.endswith("_ref"):
            raw(value[name])
    request = validate_production_request(
        read(value["request_ref"]), release_install_ref=value["release_install_ref"]
    )
    if (
        request["action"] != "EXECUTE"
        or request["target_trade_date"] != day
        or request["recipe_ref"]["sha256"] != value["recipe_ref"]["sha256"]
    ):
        raise ContractError("MAINTENANCE_HANDOFF_REQUEST_MISMATCH")
    recipe = validate_execution_recipe(read(value["recipe_ref"]), request=request)
    from .production_request import HISTORICAL_SCHEMA as REQUEST_HISTORICAL_SCHEMA
    from .catchup_binding import read_catchup_binding, verify_bound_historical_core

    if (request["schema_version"] == REQUEST_HISTORICAL_SCHEMA) != historical:
        raise ContractError("MAINTENANCE_HANDOFF_CATCHUP_VERSION_MISMATCH")
    bound = None
    if historical:
        bound = read_catchup_binding(workspace=workspace, binding_ref=value["catchup_binding_ref"])
        if bound["request"] != request or bound["recipe"] != recipe:
            raise ContractError("MAINTENANCE_HANDOFF_CATCHUP_REQUEST_MISMATCH")
    origin = None
    if value["schema_version"] == AUTOMATIC_SCHEMA:
        from .automatic_origin import (
            read_automatic_origin,
            require_origin_execution,
            recheck_origin,
        )

        raw(value["automatic_origin_ref"])
        origin = read_automatic_origin(workspace=workspace, reference=value["automatic_origin_ref"])
        require_origin_execution(
            origin,
            request_ref=value["request_ref"],
            request=request,
            recipe=recipe,
            retained=True,
            sealed_at=value["sealed_at"],
        )
    if recipe["bootstrap_ref"] is not None:
        validate_bootstrap_declaration(read(recipe["bootstrap_ref"]), recipe=recipe)
    if value["schema_version"] in {SCHEMA, HISTORICAL_SCHEMA, AUTOMATIC_SCHEMA}:
        selected_policy = recipe["policy_refs"]["prospective"]
        retained_policy = value["prospective_policy_ref"]
        if (selected_policy is None) != (retained_policy is None):
            raise ContractError("MAINTENANCE_HANDOFF_POLICY_BINDING_INVALID")
        if retained_policy is not None:
            validate_ref(retained_policy)
            if retained_policy["sha256"] != selected_policy["sha256"] or retained_policy[
                "path"
            ] != str(execution / "inputs" / f"prospective-policy-{retained_policy['sha256']}.json"):
                raise ContractError("MAINTENANCE_HANDOFF_POLICY_BINDING_INVALID")
            validate_prediction_policy(read(retained_policy), trade_date=day)
    if recipe["release_ref"] != value["release_ref"]:
        raise ContractError("MAINTENANCE_HANDOFF_RELEASE_MISMATCH")
    context_ref = recipe["factor_loop_context_ref"]
    raw(context_ref)
    if completed_snapshot is None:
        context, installation = read_factor_loop_context(
            workspace_root=workspace,
            context_path=str(root / context_ref["path"]),
            context_sha256=context_ref["sha256"],
        )
    else:
        context, installation = archived["context"], archived["installation"]
    if context["release_install_input_ref"] != value["release_install_ref"]:
        raise ContractError("MAINTENANCE_HANDOFF_INSTALL_BINDING_MISMATCH")
    started = read(value["maintenance_started_ref"])
    if (
        type(started) is not dict
        or started.get("state") != "STARTED"
        or started.get("mode") != "execute"
    ):
        raise ContractError("MAINTENANCE_HANDOFF_STARTED_INVALID")
    run_date = handoff_task_date(value, started)
    if completed_snapshot is None:
        claim = inspect_handoff_slot_claim(
            workspace=workspace,
            claim_ref=value["logical_claim_ref"],
            handoff=value,
            started=started,
        )
    else:
        claim = archived["claim"]
        if claim["run_date"] != run_date or claim["claim_ref"] != value["logical_claim_ref"]:
            raise ContractError("RECORDED_HANDOFF_CLAIM_MISMATCH")
    checkpoint = read(value["maintenance_core_ref"])
    from quant_investor.market.historical_session import CORE_SCHEMA

    if (
        type(checkpoint) is not dict
        or checkpoint.get("schema_version")
        != (CORE_SCHEMA if historical else "cn-daily-maintenance-core.v1")
        or (not historical and "historical_session_ref" in checkpoint)
        or checkpoint.get("producer") != "quant_investor.market.daily_maintenance"
        or checkpoint.get("scope") != "FACTOR_INPUTS_ONLY"
        or checkpoint.get("other_authority") != "NONE"
        or checkpoint.get("target_date") != day
        or checkpoint.get("mode") != "execute"
        or checkpoint.get("maintenance_status") != "IN_PROGRESS"
        or checkpoint.get("status") != "CORE_COMPLETE"
        or checkpoint.get("blockers") != []
    ):
        raise ContractError("MAINTENANCE_HANDOFF_CORE_CONTRACT_INVALID")
    for native_key, field in (
        ("logical_claim_ref", "logical_claim_ref"),
        ("started_ref", "maintenance_started_ref"),
        ("close_session_receipt_ref", "calendar_ref"),
    ):
        if local(checkpoint[native_key]) != value[field]:
            raise ContractError("MAINTENANCE_HANDOFF_CORE_REF_MISMATCH")
    attempt = PurePosixPath(value["maintenance_core_ref"]["path"]).parent
    if (
        attempt.parent != PurePosixPath(claim["maintenance_run_root"]) / "attempts"
        or value["maintenance_core_ref"]["path"] != str(attempt / "core-completion.json")
        or value["maintenance_started_ref"]["path"] != str(attempt / "started.json")
    ):
        raise ContractError("MAINTENANCE_HANDOFF_ATTEMPT_PATH_INVALID")
    rows, refs = checkpoint.get("stage_results"), checkpoint.get("stage_refs")
    if (
        type(rows) is not list
        or any(type(r) is not dict for r in rows)
        or [r.get("stage") for r in rows] != ["PIT", "MARKET", "HISTORY"]
        or type(refs) is not list
        or len(refs) != 3
    ):
        raise ContractError("MAINTENANCE_HANDOFF_CORE_STAGES_INVALID")
    for row, ref in zip(rows, refs):
        if row.get("status") not in {"READY", "NO_ACTION"} or row.get("blockers") != []:
            raise ContractError("MAINTENANCE_HANDOFF_CORE_STAGE_NOT_READY")
        ref = local(ref)
        if ref["path"] != str(attempt / f"stage-{row['stage']}.json"):
            raise ContractError("MAINTENANCE_HANDOFF_STAGE_PATH_INVALID")
        stage = read(ref)
        if stage.get("state") != "STAGE_COMPLETED" or canonical_json_bytes(
            stage.get("result")
        ) != canonical_json_bytes(row):
            raise ContractError("MAINTENANCE_HANDOFF_STAGE_BINDING_INVALID")
    market = rows[1]["evidence"]
    if (
        market["pointer_sha256"] != value["market_pointer_ref"]["sha256"]
        or local(
            {"path": market["snapshot_manifest_path"], "sha256": market["snapshot_manifest_sha256"]}
        )
        != value["market_snapshot_ref"]
    ):
        raise ContractError("MAINTENANCE_HANDOFF_MARKET_BINDING_INVALID")
    calendar = read(value["calendar_ref"])
    if (
        local({"path": calendar["raw_response_path"], "sha256": calendar["raw_response_sha256"]})
        != value["raw_calendar_ref"]
    ):
        raise ContractError("MAINTENANCE_HANDOFF_CALENDAR_BINDING_INVALID")
    calendar_result = replay_close_session_authority(calendar, raw(value["raw_calendar_ref"]))
    if historical:
        if local(checkpoint["historical_session_ref"]) != value["historical_session_ref"]:
            raise ContractError("HISTORICAL_HANDOFF_PROOF_REF_MISMATCH")
        proof = _historical_core_evidence(
            root=root,
            core_ref=value["maintenance_core_ref"],
            checkpoint=checkpoint,
            recipe=recipe,
            read_file=read_file,
        )
        verify_bound_historical_core(
            derived=bound, proof=proof, calendar=calendar, raw=raw(value["raw_calendar_ref"])
        )
    elif calendar_result.receipt["target_trade_date"] != day:
        raise ContractError("MAINTENANCE_HANDOFF_CALENDAR_DATE_INVALID")
    if not historical and recipe["previous_completion_ref"] is not None:
        _current_recipe_predecessor(recipe, day, calendar, raw(value["raw_calendar_ref"]))
    core = inspect_core_handoff(
        workspace=workspace,
        trade_date=day,
        handoff_ref=value["core_handoff_ref"],
        release_ref=value["release_ref"],
    )
    if core["factor_pointer_ref"] != value["factor_pointer_ref"]:
        raise ContractError("MAINTENANCE_HANDOFF_FACTOR_BINDING_INVALID")
    snapshot = FactorProductionStore(workspace).inspect_recorded_research_inputs(
        pointer_raw=raw(value["factor_pointer_ref"]),
        expected_pointer_sha256=value["factor_pointer_ref"]["sha256"],
        expected_trade_date=day,
    )
    release = object_ref_for_artifact(validate_artifact(read(value["release_ref"])))
    if (
        release != installation["release_ref"]
        or release != snapshot["factor_generation"]["payload"]["deployed_release_ref"]
        or snapshot["market_pointer_sha256"] != value["market_pointer_ref"]["sha256"]
        or snapshot["market_manifest_sha256"] != value["market_snapshot_ref"]["sha256"]
    ):
        raise ContractError("MAINTENANCE_HANDOFF_REPLAY_BINDING_INVALID")
    times = [utc_stamp(started["started_at"])] + [
        utc_stamp(read(ref)["finished_at"]) for ref in core["node_terminal_refs"].values()
    ]
    if historical:
        times.append(utc_stamp(checkpoint["sealed_at"]))
    if max(times) > sealed:
        raise ContractError("MAINTENANCE_HANDOFF_PREMATURE_SEAL")
    for path, content in observed.items():
        if storage.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024).data != content:
            raise ContractError("MAINTENANCE_HANDOFF_CHANGED_DURING_REPLAY")
    if bound is not None:
        bound["sources"].recheck()
    if origin is not None:
        recheck_origin(origin)
    return {
        "handoff_ref": dict(handoff_ref),
        "handoff": value,
        "request": request,
        "recipe": recipe,
        "factor_snapshot": snapshot,
        "native_replay_required": True,
        "execution_authorized": False,
        "validation_scope": "RETAINED_HANDOFF_FACTOR_CALENDAR_REPLAY",
    }


def _current_recipe_predecessor(recipe, day, calendar, raw):
    from quant_investor.market.requested_session import classify_current_session_edge
    from quant_investor.market.close_session_authority import CloseSessionAuthorityError
    from quant_investor.market.tushare_transport import TushareHttpsError

    try:
        classify_current_session_edge(
            requested_trade_date=day,
            previous_trade_date=Path(recipe["previous_completion_ref"]["path"]).parent.name,
            receipt=calendar,
            raw=raw,
        )
    except (CloseSessionAuthorityError, TushareHttpsError) as exc:
        raise ContractError("MAINTENANCE_HANDOFF_CURRENT_PREDECESSOR_INVALID") from exc
