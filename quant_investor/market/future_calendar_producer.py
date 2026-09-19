"""Opt-in synthetic future Calendar production with retained, zero-call recovery."""

from pathlib import Path

from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.system.errors import SystemPreconditionError
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.journal_revisions import selected_binding
from quant_investor.operations.future_calendar_binding import future_calendar_outputs
from .future_calendar_context import (
    STATE_SCHEMA,
    PRODUCTION_STATE_SCHEMA,
    PRODUCTION_CONTEXT_SCHEMA,
    CONTEXT_SCHEMA,
    FUTURE_STATE_SCHEMAS,
    future_mode,
    validate_future_state,
)
from ._calendar_fixture_capability import (
    require_fixture_capability,
    publish_fixture_evidence,
    retained_fixture_evidence,
    publish_bound_fixture_proof,
)
from .next_session_acquisition import capture_next_session_calendar
from .next_session_failure import publish_next_session_failure


def new_future_state(*, day, context_sha, state, context_schema=CONTEXT_SCHEMA):
    production = context_schema == PRODUCTION_CONTEXT_SCHEMA
    return {
        **state,
        "schema_version": PRODUCTION_STATE_SCHEMA if production else STATE_SCHEMA,
        "phase": "CORE_READY",
        "trade_date": day,
        "context_sha256": context_sha,
        "future_capture_refs": None,
        ("transport_evidence_ref" if production else "fixture_transport_evidence_ref"): None,
        "next_session_calendar_proof_ref": None,
        "next_session_calendar_failure_ref": None,
        "core_handoff_ref": None,
    }


def _retained_capture(journal, install_sha, refs=None):
    name = "future-" + journal.trade_date + "-" + install_sha[:16]
    result = {}
    for role, leaf in (
        ("execution", "capture-execution.json"),
        ("success", "capture-success.json"),
    ):
        relative = name + "/" + leaf
        stored = journal.storage.read(str(journal.root / "calendar-future/captures" / relative))
        if stored is None:
            return None
        ref = {"relative_path": relative, "byte_sha256": stored.byte_sha256}
        if refs is not None and refs[role + "_ref"] != ref:
            raise ContractError("NEXT_SESSION_CAPTURE_FILE_REF_MISMATCH")
        result["capture_" + role] = parse_canonical_json_bytes(stored.data)
        result["capture_" + role + "_file_ref"] = ref
    return result


def _retained_native_failure(journal, install_sha):
    from .tushare_calendar_authority import _failure_root_name

    name = "future-" + journal.trade_date + "-" + install_sha[:16]
    relative = _failure_root_name(name) + "/capture-failure.json"
    stored = journal.storage.read(str(journal.root / "calendar-future/captures" / relative))
    if stored is None:
        return None
    # The typed failure publisher replays the native root and exact reference.
    return {"relative_path": relative, "byte_sha256": stored.byte_sha256}


def _projection_failure_code(exc):
    return {
        "NEXT_SESSION_HORIZON_INCOMPLETE": "HORIZON_INCOMPLETE",
        "NEXT_SESSION_HORIZON_MISMATCH": "HORIZON_INCOMPLETE",
        "NEXT_SESSION_EXCHANGE_DISAGREEMENT": "EXCHANGE_DISAGREEMENT",
        "NEXT_SESSION_EOD_NOT_OPEN": "EOD_NOT_OPEN",
        "NEXT_SESSION_NO_LATER_OPEN": "NO_LATER_OPEN",
        "PROVENANCE_UNAVAILABLE": "PROVENANCE_UNAVAILABLE",
    }.get(str(exc))


def bind_future_calendar(*, workspace, context, context_sha, state, save_state, recovery=False):
    if context.get("schema_version") == PRODUCTION_CONTEXT_SCHEMA:
        from .production_future_calendar import (
            bind_fresh_production_calendar,
            recover_production_calendar,
        )

        binder = recover_production_calendar if recovery else bind_fresh_production_calendar
        return binder(
            workspace=workspace,
            context=context,
            context_sha=context_sha,
            state=state,
            save_state=save_state,
        )
    mode = future_mode(context)
    day = state["trade_date"]
    validate_future_state(
        state,
        context_sha256=context_sha,
        trade_date=day,
        mode=mode,
        context_schema=context["schema_version"],
    )
    journal = DailyJournal(str(workspace), day)
    install_sha = context["release_install_input_ref"]["sha256"]
    if mode != "DISABLED":
        require_fixture_capability(workspace=workspace, trade_date=day, install_sha=install_sha)
    if state["phase"] in {"FUTURE_BOUND", "CORE_PUBLISHED"}:
        future_calendar_outputs(
            workspace=str(workspace),
            trade_date=day,
            proof_ref=state["next_session_calendar_proof_ref"],
            failure_ref=state["next_session_calendar_failure_ref"],
        )
        return state
    if (
        journal.storage.read(str(journal.root / "completion.v1.json")) is not None
        or selected_binding(journal.storage, str(journal.root / "nodes/calendar"))[0] is not None
    ):
        raise ContractError("NEXT_SESSION_CALENDAR_ALREADY_SELECTED")

    def persist(**changes):
        state.update(changes)
        validate_future_state(state, context_sha256=context_sha, trade_date=day, mode=mode)
        save_state(state)

    def failure(code, captured=None, native_failure_ref=None):
        refs = (
            {}
            if captured is None
            else {
                "execution_ref": captured["capture_execution_file_ref"],
                "success_ref": captured["capture_success_file_ref"],
            }
        )
        ref = publish_next_session_failure(
            workspace=str(workspace),
            eod_trade_date=day,
            phase="ACQUISITION" if captured is None else "POST_CAPTURE_VALIDATION",
            failure_code=code,
            native_failure_ref=native_failure_ref,
            **refs,
        )
        persist(phase="FUTURE_BOUND", next_session_calendar_failure_ref=ref)
        return state

    if mode == "DISABLED":
        persist(phase="FUTURE_BOUND")
        return state
    captured = None
    if recovery or state["phase"] == "CAPTURE_BOUND":
        captured = _retained_capture(journal, install_sha, state["future_capture_refs"])
        if captured is None:
            if state["phase"] == "CAPTURE_BOUND":
                raise ContractError("NEXT_SESSION_CAPTURE_FILE_REF_MISMATCH")
            return failure(
                "ACQUISITION_FAILED",
                native_failure_ref=_retained_native_failure(journal, install_sha),
            )
        try:
            evidence = retained_fixture_evidence(
                workspace=workspace, trade_date=day, install_sha=install_sha, captured=captured
            )
        except ContractError as exc:
            code = _projection_failure_code(exc)
            if code is None:
                raise
            return failure(code, captured)
        if (
            state["fixture_transport_evidence_ref"] is not None
            and state["fixture_transport_evidence_ref"] != evidence
        ):
            raise ContractError("PROVENANCE_UNAVAILABLE")
    else:
        # Saved before acquisition: a crash never authorizes recovery to recapture.
        save_state(state)
        from quant_investor.factors.production_rollover import _read_owner_file

        ref = context["release_install_input_ref"]
        raw, sha = _read_owner_file(
            Path(workspace) / ref["path"], root=Path(workspace), label="future Calendar release"
        )
        if sha != install_sha:
            raise ContractError("NEXT_SESSION_RELEASE_INPUT_SHA_MISMATCH")
        try:
            result = capture_next_session_calendar(
                workspace=str(workspace),
                eod_trade_date=day,
                release_install_input_raw=raw,
                expected_release_install_input_sha256=sha,
                release_repository_root=context["release_repository_root"],
            )
        except SystemPreconditionError as exc:
            native_ref = exc.public_fields.get("capture_failure_file_ref")
            if native_ref is None:
                raise
            if native_ref != _retained_native_failure(journal, install_sha):
                raise ContractError("NEXT_SESSION_FAILURE_SOURCE_SHA_MISMATCH") from exc
            return failure("ACQUISITION_FAILED", native_failure_ref=native_ref)
        except ContractError as exc:
            code = _projection_failure_code(exc)
            if code is None:
                raise
            captured = _retained_capture(journal, install_sha)
            if captured is None:
                raise
            return failure(code, captured)
        captured = result["capture"]
        try:
            evidence = publish_fixture_evidence(
                workspace=workspace, trade_date=day, install_sha=install_sha, captured=captured
            )
        except ContractError as exc:
            code = _projection_failure_code(exc)
            if code is None:
                raise
            return failure(code, captured)
    persist(
        phase="CAPTURE_BOUND",
        future_capture_refs={
            "execution_ref": captured["capture_execution_file_ref"],
            "success_ref": captured["capture_success_file_ref"],
        },
        fixture_transport_evidence_ref=evidence,
    )
    proof = publish_bound_fixture_proof(
        workspace=workspace,
        trade_date=day,
        install_sha=install_sha,
        captured=captured,
        evidence_ref=evidence,
    )
    persist(phase="FUTURE_BOUND", next_session_calendar_proof_ref=proof)
    return state


def core_future_arguments(state):
    if state.get("schema_version") not in FUTURE_STATE_SCHEMAS:
        return {}
    return {
        "next_session_calendar_proof_ref": state["next_session_calendar_proof_ref"],
        "next_session_calendar_failure_ref": state["next_session_calendar_failure_ref"],
    }
