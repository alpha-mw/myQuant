"""Calendar-ordered downstream catch-up through existing native run/seal entries."""

from pathlib import PurePosixPath

from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.operations.catchup import plan_catchup
from quant_investor.operations.completion_readback import inspect_recorded_completion
from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS, validate_ref
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY
from quant_investor.operations.production_result import SCHEMA, validate_production_result
from quant_investor.system.storage import SecureSystemStorage
from scripts.daily_completion import run_materialized_native_input
from scripts.daily_completion_replay import replay_native_completion


def _serving_gate(workspace, row, *, read_only=False):
    if row["completion_ref"] is None or row["business_state"] != "COMPLETE":
        return True
    from scripts.daily_dashboard_publication import complete_serving_result, observed_serving_status

    if read_only:
        state = observed_serving_status(workspace, row["trade_date"])
        if (
            state is None
            or state.get("sealed_evidence_ref") != row["completion_ref"]
            or state.get("validation_scope") != "RECORDED_SERVING_BYTES_ONLY"
        ):
            raise ContractError("AUTO_SERVING_SCOPE_UNCONFIRMED")
        status = state["publication_state"]
        if status in {"RECORDED_EOD_PUBLICATION_EXPIRED", "EVIDENCE_SEALED_PUBLICATION_EXPIRED"}:
            raise ContractError("AUTO_PUBLICATION_EXPIRED")
        if status not in {"RECORDED_EOD_PUBLICATION", "EVIDENCE_SEALED_PUBLICATION_PENDING"}:
            raise ContractError("AUTO_SERVING_STATE_INVALID")
        complete = status == "RECORDED_EOD_PUBLICATION"
        if complete:
            ref = validate_ref(state.get("publication_ref"))
            if ref["path"] != (
                f"results/operations/daily_production/CN/{row['trade_date']}/"
                "dashboard/serving-publication.v2.json"
            ):
                raise ContractError("AUTO_SERVING_RECEIPT_REF_INVALID")
    else:
        result = complete_serving_result(workspace=workspace, completion_ref=row["completion_ref"])
        complete = result["status"] == "COMPLETE"
    if not complete:
        row.update(execution_state="PARTIAL", business_state="INCOMPLETE", completion_ref=None)
        return False
    return True


def _replay(workspace, day, ref):
    result = replay_native_completion(workspace=workspace, trade_date=day, completion_ref=ref)
    if (
        result.get("native_replay_validated") is not True
        or result.get("completion_ref") != ref
        or result.get("trade_date") != day
        or result.get("validated_nodes") != sorted(EOD_NODE_IDS)
    ):
        raise ContractError("CATCHUP_NATIVE_ANCHOR_INVALID")


def _completed_day(workspace, day, input_ref):
    journal = DailyJournal(workspace, day)
    path = str(journal.root / "completion.v1.json")
    stored = journal.storage.read(path)
    if stored is None:
        return None
    ref = {"path": path, "sha256": stored.byte_sha256}
    recorded = inspect_recorded_completion(workspace=workspace, trade_date=day, completion_ref=ref)
    if recorded["recorded_completion"]["native_inputs_ref"] != input_ref:
        raise ContractError("CATCHUP_COMPLETED_INPUT_CONFLICT")
    _replay(workspace, day, ref)
    return ref


def run_native_catchup(
    *,
    workspace: str,
    calendar_ref: dict,
    raw_calendar_ref: dict,
    previous_completion_ref: dict,
    target_trade_date: str,
    day_input_refs: dict,
    synthetic: bool,
) -> dict:
    """Internal caller owns provenance/permission; this entry never starts maintenance.

    Every missing date must have exact existing native inputs. Earlier completion
    is fully replayed; completed dates use no-write replay and are never rerun.
    A failed date stops later dates. Native writes remain under one lock per day.
    """
    if type(synthetic) is not bool:
        raise ContractError("CATCHUP_PROVENANCE_INVALID")
    validate_ref(previous_completion_ref)
    previous = PurePosixPath(previous_completion_ref["path"]).parent.name
    plan = plan_catchup(
        workspace=workspace,
        calendar_ref=calendar_ref,
        raw_calendar_ref=raw_calendar_ref,
        previous_trade_date=previous,
        target_trade_date=target_trade_date,
        day_input_refs=day_input_refs,
    )
    _replay(workspace, previous, previous_completion_ref)
    dates = plan["ordered_trade_dates"]
    reader = SecureSystemStorage(workspace)
    predecessor = previous
    # Check all available date/previous-day bindings before starting any writer.
    for day in dates:
        if day in day_input_refs:
            ref = day_input_refs[day]
            stored = reader.read_workspace_file_bytes(ref["path"], maximum_bytes=1024 * 1024)
            if stored.byte_sha256 != ref["sha256"]:
                raise ContractError("CATCHUP_INPUT_CHANGED")
            value = parse_canonical_json_bytes(stored.data)
            if value["previous_trade_date"] != predecessor:
                raise ContractError("CATCHUP_PREDECESSOR_BINDING_INVALID")
        predecessor = day
    stopped = bool(plan["missing_input_dates"])
    rows = []
    for day in dates:
        row = {
            "trade_date": day,
            "execution_state": "BLOCKED",
            "business_state": "INCOMPLETE",
            "completion_ref": None,
        }
        if not stopped:
            ref = _completed_day(workspace, day, day_input_refs[day])
            if ref is not None:
                row.update(
                    execution_state="NO_ACTION", business_state="COMPLETE", completion_ref=ref
                )
            else:
                result = run_materialized_native_input(
                    workspace=workspace,
                    input_ref=day_input_refs[day],
                    resume=True,
                    synthetic=synthetic,
                )
                if result["status"] == "COMPLETE" and result["completion_ref"] is not None:
                    row.update(
                        execution_state="SUCCEEDED",
                        business_state="COMPLETE",
                        completion_ref=result["completion_ref"],
                    )
                else:
                    row["execution_state"] = result["status"]
                    stopped = True
        if not _serving_gate(workspace, row):
            stopped = True
        rows.append(row)
    if not dates and target_trade_date == previous:
        dates = [previous]
        rows = [
            {
                "trade_date": previous,
                "execution_state": "NO_ACTION",
                "business_state": "COMPLETE",
                "completion_ref": previous_completion_ref,
            }
        ]
        _serving_gate(workspace, rows[0])
    states = {row["execution_state"] for row in rows}
    if not rows:
        state, business = "NO_ACTION", "NON_TRADING_DAY"
    elif states <= {"SUCCEEDED", "NO_ACTION"}:
        state = "NO_ACTION" if states == {"NO_ACTION"} else "SUCCEEDED"
        business = "COMPLETE"
    else:
        state = (
            "FAILED"
            if "FAILED" in states
            else "PARTIAL" if states & {"SUCCEEDED", "NO_ACTION", "PARTIAL"} else "BLOCKED"
        )
        business = "INCOMPLETE"
    return validate_production_result(
        {
            "schema_version": SCHEMA,
            "action": "CATCH_UP",
            "target_trade_date": target_trade_date,
            "execution_state": state,
            "business_state": business,
            "days": rows,
            "authority": FALSE_AUTHORITY,
        },
        action="CATCH_UP",
        target_trade_date=target_trade_date,
        expected_dates=dates,
    )


def run_upstream_catchup(
    *,
    workspace: str,
    request_ref: dict,
    synthetic: bool,
    no_producers: bool = False,
    read_only: bool = False,
    committed_recovery_only: bool = False,
) -> dict:
    """Public batch route: exact templates -> native maintenance -> completed EOD."""
    from quant_investor.operations.catchup_binding import (
        binding_path,
        read_catchup_collection,
        read_catchup_binding,
        derive_catchup_binding,
        persist_catchup_binding,
        check_template_sources,
        collection_version,
        validate_routed_native_input,
        verify_completed_edge,
        requires_current_serving,
    )
    from scripts.daily_materialization import execute_daily_recipe

    if any(
        type(value) is not bool
        for value in (synthetic, no_producers, read_only, committed_recovery_only)
    ):
        raise ContractError("CATCHUP_PROVENANCE_INVALID")
    if committed_recovery_only and (no_producers or read_only):
        raise ContractError("AUTO_COMMITTED_RECOVERY_MODE_INVALID")
    if read_only and not no_producers:
        raise ContractError("AUTO_READ_ONLY_REQUIRES_NO_PRODUCERS")
    context = read_catchup_collection(workspace=workspace, request_ref=request_ref)
    request, plan = context["request"], context["plan"]
    version = collection_version(context["collection"])
    if (no_producers or read_only) and version not in {"v2", "v3"}:
        raise ContractError("AUTO_REPLAY_REQUIRES_COLLECTION_V2")
    dates = plan["ordered_trade_dates"]
    previous_ref = request["previous_completion_ref"]
    recipes, inputs = context["collection"]["recipes"], request["day_input_refs"]
    recovery = {}
    if committed_recovery_only:
        from quant_investor.operations.automatic_catchup_storage import current_automatic_origin
        from quant_investor.operations.automatic_catchup_resolution import read_automatic_resolution
        from quant_investor.operations.automatic_registered_recovery import inspect_registered_batch

        current = current_automatic_origin(
            workspace=workspace, derived_request_ref=request_ref, synthetic=synthetic
        )
        if current is None or version != "v3":
            raise ContractError("AUTO_COMMITTED_RECOVERY_CAPABILITY_REQUIRED")
        automatic = read_automatic_resolution(
            workspace=workspace,
            resolution_ref=current["resolution_ref"],
            release_install_ref=request["release_install_ref"],
            synthetic=synthetic,
        )
        recovery = inspect_registered_batch(
            automatic, resolution_ref=current["resolution_ref"], synthetic=synthetic, required=True
        )

    def replay(day, ref):
        result = replay_native_completion(workspace=workspace, trade_date=day, completion_ref=ref)
        if (
            result.get("native_replay_validated") is not True
            or result.get("completion_ref") != ref
            or result.get("trade_date") != day
            or result.get("validated_nodes") != sorted(EOD_NODE_IDS)
            or result.get("synthetic") is not synthetic
        ):
            raise ContractError("CATCHUP_NATIVE_COMPLETION_INVALID")

    def completed(day, previous=None, predecessor_ref=None):
        journal = DailyJournal(workspace, day)
        bound = journal.storage.read(binding_path(day, request_ref, version))
        derived = None
        if bound is not None:
            derived = read_catchup_binding(
                workspace=workspace,
                binding_ref={"path": bound.relative_path, "sha256": bound.byte_sha256},
            )
        stored = journal.storage.read(str(journal.root / "completion.v1.json"))
        if stored is None:
            return None
        ref = {"path": stored.relative_path, "sha256": stored.byte_sha256}
        inspected = None
        if derived is not None:
            recorded = inspect_recorded_completion(
                workspace=workspace, trade_date=day, completion_ref=ref
            )
            inspected = recorded
            snapshot = recorded.get("completed_handoff_snapshot")
            if (
                snapshot is None
                or snapshot.document("handoff")["request_ref"]["sha256"]
                != derived["binding"]["execution_request_ref"]["sha256"]
            ):
                raise ContractError("CATCHUP_COMPLETED_REQUEST_CONFLICT")
        replay(day, ref)
        if version in {"v2", "v3"}:
            if previous is None or predecessor_ref is None:
                raise ContractError("CATCHUP_COMPLETED_PREFIX_GAP")
            native = verify_completed_edge(
                workspace=workspace,
                day=day,
                completion_ref=ref,
                previous=previous,
                previous_ref=predecessor_ref,
                inspected=inspected,
            )
            if requires_current_serving(context, day):
                validate_routed_native_input(
                    native,
                    day=day,
                    previous=previous,
                    routing=context["routing"][day],
                    version=(
                        "v3"
                        if native["schema_version"]
                        in {"cn-daily-native-inputs.v6", "cn-daily-native-inputs.v7"}
                        else "v2"
                    ),
                )
            if (
                day in inputs
                and inputs[day]
                != (
                    inspected
                    or inspect_recorded_completion(
                        workspace=workspace, trade_date=day, completion_ref=ref
                    )
                )["recorded_completion"]["native_inputs_ref"]
            ):
                raise ContractError("CATCHUP_COMPLETED_INPUT_CONFLICT")
        return ref

    replay(plan["previous_trade_date"], previous_ref)
    missing = False
    completed_prefix = {}
    predecessor = plan["previous_trade_date"]
    prefix_ref = previous_ref
    # Prove all date ownership and declared fresh sources before the first writer.
    for day in dates:
        existing = completed(day, predecessor, prefix_ref)
        if existing is not None:
            completed_prefix[day] = existing
            prefix_ref = existing
        else:
            if version in {"v2", "v3"}:
                prefix_ref = None
            if committed_recovery_only:
                if day not in recovery:
                    raise ContractError("AUTO_COMMITTED_RECOVERY_REQUIRED")
            elif day in recipes:
                check_template_sources(context["sources"], recipes[day], profile="FRESH")
            elif day in inputs:
                value = context["sources"].document(inputs[day])
                if value["previous_trade_date"] != predecessor:
                    raise ContractError("CATCHUP_PREDECESSOR_BINDING_INVALID")
                if version in {"v2", "v3"}:
                    validate_routed_native_input(
                        value,
                        day=day,
                        previous=predecessor,
                        routing=context["routing"][day],
                        version=version,
                    )
            else:
                missing = True
        predecessor = day
    context["sources"].recheck()
    if no_producers and len(completed_prefix) != len(dates):
        raise ContractError("AUTO_PRODUCERS_REQUIRED")
    stopped, rows = missing, []
    for day in dates:
        row = {
            "trade_date": day,
            "execution_state": "BLOCKED",
            "business_state": "INCOMPLETE",
            "completion_ref": None,
        }
        if (
            missing
            and day in completed_prefix
            and all(previous_row["business_state"] == "COMPLETE" for previous_row in rows)
        ):
            row.update(
                execution_state="NO_ACTION",
                business_state="COMPLETE",
                completion_ref=completed_prefix[day],
            )
        if not stopped:
            previous_day = PurePosixPath(previous_ref["path"]).parent.name
            ref = completed(day, previous_day, previous_ref)
            if ref is not None:
                row.update(
                    execution_state="NO_ACTION", business_state="COMPLETE", completion_ref=ref
                )
            else:
                if no_producers:
                    raise ContractError("AUTO_PRODUCERS_REQUIRED")
                replay(PurePosixPath(previous_ref["path"]).parent.name, previous_ref)
                if committed_recovery_only:
                    selected = recovery[day]
                    selected["bound"]["sources"].recheck()
                    result = execute_daily_recipe(
                        workspace=workspace,
                        request_ref=selected["bound"]["binding"]["execution_request_ref"],
                        synthetic=synthetic,
                        committed_recovery_only=True,
                        _automatic_origin_ref=selected["origin_ref"],
                    )
                elif day in recipes:
                    # Reconstruct from immutable evidence after fresh-source preflight.
                    derived = derive_catchup_binding(
                        workspace=workspace,
                        request_ref=request_ref,
                        day=day,
                        previous_completion_ref=previous_ref,
                    )
                    journal = DailyJournal(workspace, day)
                    with journal.locked():
                        concurrent = journal.storage.read(str(journal.root / "completion.v1.json"))
                        binding_ref = (
                            None
                            if concurrent is not None
                            else persist_catchup_binding(journal=journal, derived=derived)
                        )
                    if concurrent is not None:
                        ref = completed(day, previous_day, previous_ref)
                        result = {"status": "COMPLETE", "completion_ref": ref}
                    else:
                        checked = read_catchup_binding(workspace=workspace, binding_ref=binding_ref)
                        checked["sources"].recheck()
                        origin_kwargs = {}
                        if (
                            checked["recipe"]["schema_version"] == "cn-daily-execute-recipe.v6"
                            and checked["binding"]["maintenance_mode"] == "CURRENT"
                        ):
                            from quant_investor.operations.automatic_catchup_storage import (
                                current_automatic_origin,
                            )
                            from quant_investor.operations.automatic_origin import (
                                origin_document,
                                publish_automatic_origin,
                            )

                            current = current_automatic_origin(
                                workspace=workspace,
                                derived_request_ref=request_ref,
                                synthetic=synthetic,
                            )
                            if current is not None:
                                origin_kwargs["_automatic_origin_ref"] = publish_automatic_origin(
                                    workspace=workspace,
                                    synthetic=synthetic,
                                    origin=origin_document(
                                        current=current,
                                        binding_ref=binding_ref,
                                        execution_request_ref=checked["binding"][
                                            "execution_request_ref"
                                        ],
                                        day=day,
                                    ),
                                )
                        result = execute_daily_recipe(
                            workspace=workspace,
                            request_ref=checked["binding"]["execution_request_ref"],
                            synthetic=synthetic,
                            _catchup_binding_ref=(
                                binding_ref
                                if checked["request"]["schema_version"]
                                == "cn-daily-production-request.v2"
                                else None
                            ),
                            **origin_kwargs,
                        )
                else:
                    result = run_materialized_native_input(
                        workspace=workspace, input_ref=inputs[day], resume=True, synthetic=synthetic
                    )
                status, ref = result.get("status"), result.get("completion_ref")
                replayed_ref = None
                if (
                    status == "PARTIAL"
                    and ref is None
                    and recipes.get(day, {}).get("schema_version") == "cn-daily-execute-recipe.v6"
                ):
                    sealed = completed(day, previous_day, previous_ref)
                    if sealed is not None:
                        status, ref = "COMPLETE", sealed
                        replayed_ref = sealed
                if status == "COMPLETE" and ref is not None:
                    verified_ref = (
                        replayed_ref
                        if replayed_ref is not None
                        else completed(day, previous_day, previous_ref)
                    )
                    if ref != verified_ref:
                        raise ContractError("CATCHUP_COMPLETED_RESULT_MISMATCH")
                    row.update(
                        execution_state="SUCCEEDED", business_state="COMPLETE", completion_ref=ref
                    )
                elif (
                    status in {"PARTIAL", "BLOCKED", "FAILED", "PENDING", "RUNNING"} and ref is None
                ):
                    row["execution_state"] = (
                        status if status in {"PARTIAL", "BLOCKED", "FAILED"} else "PARTIAL"
                    )
                    stopped = True
                else:
                    raise ContractError("CATCHUP_NATIVE_RESULT_INVALID")
            if row["completion_ref"] is not None:
                previous_ref = row["completion_ref"]
        if requires_current_serving(context, day):
            if recipes.get(day, {}).get("schema_version") == "cn-daily-execute-recipe.v6":
                # Refuse exact expiry only after native EOD completion. The copy
                # keeps pending publication from erasing the internal EOD ref.
                _serving_gate(workspace, dict(row), read_only=True)
            if not _serving_gate(workspace, row, read_only=read_only):
                stopped = True
        rows.append(row)
    if not dates and request["target_trade_date"] == plan["previous_trade_date"]:
        dates = [plan["previous_trade_date"]]
        rows = [
            {
                "trade_date": dates[0],
                "execution_state": "NO_ACTION",
                "business_state": "COMPLETE",
                "completion_ref": previous_ref,
            }
        ]
        if version in {"v2", "v3"}:
            source = context["sources"]
            recorded = source.document(previous_ref)
            native = source.document(recorded["native_inputs_ref"])
            verify_completed_edge(
                workspace=workspace,
                day=dates[0],
                completion_ref=previous_ref,
                previous=native["previous_trade_date"],
            )
            from quant_investor.market.requested_session import require_calendar_predecessor

            require_calendar_predecessor(
                requested_trade_date=dates[0],
                previous_trade_date=native["previous_trade_date"],
                receipt=context["calendar"],
                raw=source.raw(request["raw_calendar_ref"]),
            )
            if requires_current_serving(context, dates[0]):
                validate_routed_native_input(
                    native,
                    day=dates[0],
                    previous=native["previous_trade_date"],
                    routing={"dashboard_mode": "CURRENT_LATEST_EOD"},
                    version=(
                        "v3"
                        if native["schema_version"]
                        in {"cn-daily-native-inputs.v6", "cn-daily-native-inputs.v7"}
                        else "v2"
                    ),
                )
        if requires_current_serving(context, dates[0]):
            _serving_gate(workspace, rows[0], read_only=read_only)
    states = {row["execution_state"] for row in rows}
    if not rows:
        state, business = "NO_ACTION", "NON_TRADING_DAY"
    elif states <= {"SUCCEEDED", "NO_ACTION"}:
        state, business = ("NO_ACTION" if states == {"NO_ACTION"} else "SUCCEEDED"), "COMPLETE"
    else:
        state = (
            "FAILED"
            if "FAILED" in states
            else "PARTIAL" if states & {"SUCCEEDED", "NO_ACTION", "PARTIAL"} else "BLOCKED"
        )
        business = "INCOMPLETE"
    return validate_production_result(
        {
            "schema_version": SCHEMA,
            "action": "CATCH_UP",
            "target_trade_date": request["target_trade_date"],
            "execution_state": state,
            "business_state": business,
            "days": rows,
            "authority": FALSE_AUTHORITY,
        },
        action="CATCH_UP",
        target_trade_date=request["target_trade_date"],
        expected_dates=dates,
    )
