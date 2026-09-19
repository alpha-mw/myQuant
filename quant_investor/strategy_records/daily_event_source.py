"""Read-only source semantics for standing-policy daily empty event closures."""

from datetime import date, datetime, time
import hashlib
import json
from pathlib import Path

from quant_investor.market.close_session_authority import (
    replay_close_session_authority,
    CloseSessionAuthorityError,
)
from quant_investor.market.tushare_transport import TushareHttpsError
from quant_investor.market.requested_session import classify_requested_session
from .event_contracts import SYMBOLIC_RECEIPT, instant, SHANGHAI
from .event_store import StrategyEventStoreError
from .store import StrategyRecordStoreError

RECORD_ROOT = "results/strategy_records/CN/aggressive_tech_manufacturing"
POLICY_SCHEMA = "myquant.cn_daily_official_close_policy.v1"
MAINTENANCE_FIELDS = {
    "schema_version",
    "status",
    "maintenance_status",
    "same_day_status",
    "factor_input_readiness",
    "factor_input_shadow_readiness",
    "factor_input_change",
    "core_blockers",
    "macro_status",
    "macro_blockers",
    "macro_used_by_factor",
    "fundamental_used_by_factor",
    "factor_rollover_eligible",
    "fundamental_integrity_status",
    "fundamental_refresh_status",
    "mode",
    "attempt_slot",
    "target_date",
    "canonical_unchanged",
    "canonical_write_count",
    "usable_for_investment_research",
    "close_session_receipt_ref",
    "stage_results",
    "blockers",
    "protected_surfaces",
    "state_ref",
}
PROVIDER_FIELDS = {"provider_calls", "provider_request_attempts", "request_count_scope"}
MAINTENANCE_OPTIONAL = PROVIDER_FIELDS | {
    "transport_retry",
    "core_completion_ref",
    "factor_loop",
    "write_veto_ref",
    "macro_write_veto_ref",
    "logical_claim_ref",
    "started_ref",
}
STATE_FIELDS = {
    "schema_version",
    "status",
    "maintenance_status",
    "same_day_status",
    "factor_input_readiness",
    "factor_input_shadow_readiness",
    "factor_input_change",
    "core_blockers",
    "macro_status",
    "macro_blockers",
    "macro_used_by_factor",
    "fundamental_used_by_factor",
    "factor_rollover_eligible",
    "fundamental_integrity_status",
    "fundamental_refresh_status",
    "mode",
    "attempt_slot",
    "target_date",
    "stage_states",
    "blockers",
}


def validate_official_close_policy(value):
    """The existing official-close scope check, shared without a script import."""
    required_allowed = {
        "DAILY_NO_ACTION_CONTINUITY_RECEIPT",
        "OFFICIAL_VALUATION_RECORD",
        "PERFORMANCE_APPEND",
        "IMMUTABLE_CATALOG_GENERATION",
        "STRATEGY_RECORD_POINTER_CAS",
    }
    forbidden = value.get("forbidden")
    if (
        value.get("schema_id") != POLICY_SCHEMA
        or value.get("policy_id") != "cn-daily-official-close-policy-v1"
        or value.get("strategy_label") != "aggressive_tech_manufacturing"
        or value.get("record_root") != RECORD_ROOT
        or value.get("revoked_at") is not None
        or not required_allowed.issubset(set(value.get("allowed_writes") or []))
        or not isinstance(forbidden, dict)
        or not all(
            forbidden.get(name) is True
            for name in (
                "broker_connection",
                "order_creation",
                "trade_execution",
                "unregistered_share_mutation",
                "unregistered_cash_mutation",
            )
        )
        or value.get("broker_order_trade_authority") is not False
        or value.get("actual_holdings_mutation_authority") is not False
    ):
        raise StrategyRecordStoreError("official-close policy scope/authority mismatch")
    return value


def validate_standing_policy(value, *, at):
    validate_official_close_policy(value)
    inbox = value.get("event_inbox")
    if (
        type(inbox) is not dict
        or inbox.get("sealed_empty_inventory_is_owner_authorized_closure") is not True
        or any(
            inbox.get(k) != v
            for k, v in {
                "pointer_path": RECORD_ROOT + "/_event_store/current.v1.json",
                "owner_append_cutoff_local": "15:30:00",
                "timezone": "Asia/Shanghai",
                "sealed_empty_inventory_is_owner_authorized_closure": True,
                "late_event_behavior": "OFFICIAL_CLOSE_RESTATEMENT_REQUIRED",
            }.items()
        )
    ):
        raise StrategyEventStoreError("DAILY_EVENT_STANDING_POLICY_INVALID")
    if instant(value.get("effective_from"), label="policy effective") > at:
        raise StrategyEventStoreError("DAILY_EVENT_POLICY_NOT_EFFECTIVE")
    return value


class DailyEventSources:
    def __init__(self, workspace):
        # Lazy import avoids source-consumer import cycles. No producer is invoked.
        from quant_investor.operations.daily_preparation import Sources

        self.root = Path(workspace).resolve(strict=True)
        self.files = Sources(str(self.root))

    def ref(self, value):
        from quant_investor.operations.daily_contract import validate_ref

        if (
            type(value) is not dict
            or set(value) != {"path", "sha256"}
            or any(type(value[k]) is not str for k in ("path", "sha256"))
        ):
            raise StrategyEventStoreError("DAILY_EVENT_SOURCE_REF_INVALID")
        path = Path(value["path"])
        if path.is_absolute():
            try:
                path = path.relative_to(self.root)
            except ValueError as exc:
                raise StrategyEventStoreError("DAILY_EVENT_SOURCE_OUTSIDE_WORKSPACE") from exc
        return validate_ref({"path": str(path), "sha256": value["sha256"]})

    def read(self, ref):
        return self.files.raw(self.ref(ref))

    def document(self, ref):
        value = json.loads(self.read(ref))
        if type(value) is not dict:
            raise StrategyEventStoreError("DAILY_EVENT_SOURCE_DOCUMENT_INVALID")
        return value

    def recheck(self):
        self.files.recheck()


def read_calendar_source(sources, *, calendar_ref, trade_date, raw_ref=None):
    calendar_ref = sources.ref(calendar_ref)
    receipt = sources.document(calendar_ref)
    if receipt.get("schema_version") != "cn-close-session-receipt.v1":
        raise StrategyEventStoreError("DAILY_EVENT_CALENDAR_SCHEMA_INVALID")
    linked = sources.ref(
        {"path": receipt.get("raw_response_path"), "sha256": receipt.get("raw_response_sha256")}
    )
    if raw_ref is not None and sources.ref(raw_ref) != linked:
        raise StrategyEventStoreError("DAILY_EVENT_CALENDAR_RAW_BINDING_INVALID")
    raw = sources.read(linked)
    day = trade_date.replace("-", "")
    try:
        verified = replay_close_session_authority(receipt, raw).receipt
        result = classify_requested_session(requested_trade_date=day, receipt=receipt, raw=raw)
    except (CloseSessionAuthorityError, TushareHttpsError) as exc:
        raise StrategyEventStoreError("DAILY_EVENT_CALENDAR_INVALID") from exc
    if result["classification"] != "MATCHED_OPEN" or verified["target_trade_date"] != day:
        raise StrategyEventStoreError("DAILY_EVENT_OPEN_SESSION_REQUIRED")
    return {
        "calendar_ref": calendar_ref,
        "raw_calendar_ref": linked,
        "calendar": verified,
        "observed_at": datetime.strptime(verified["observed_local_time"], "%Y-%m-%dT%H:%M:%S%z"),
    }


def _maintenance_state(sources, ref, value):
    state_ref = sources.ref(value["state_ref"])
    parent = Path(sources.ref(ref)["path"]).parent
    if Path(state_ref["path"]) != parent / "state.json":
        raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_STATE_PATH_INVALID")
    state = sources.document(state_ref)
    if set(state) != STATE_FIELDS or state["schema_version"] != "cn-daily-maintenance-state.v1":
        raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_STATE_INVALID")
    if any(state[k] != value[k] for k in STATE_FIELDS - {"schema_version", "stage_states"}):
        raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_STATE_MISMATCH")
    if state["stage_states"] != {row["stage"]: row["status"] for row in value["stage_results"]}:
        raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_STAGE_STATE_MISMATCH")
    modern = bool(set(value) & (PROVIDER_FIELDS | {"logical_claim_ref", "started_ref"}))
    completed_at = None
    if modern:
        ended_ref = sources.files.pin(str(parent / "ended.json"))
        ended = sources.document(ended_ref)
        if set(ended) != {"state", "workflow_status", "receipt_ref", "ended_at"} or (
            ended["state"] != "COMPLETED"
            or ended["workflow_status"] != value["status"]
            or sources.ref(ended["receipt_ref"]) != sources.ref(ref)
        ):
            raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_TERMINAL_INVALID")
        completed_at = instant(ended["ended_at"], label="maintenance ended")
    return modern, completed_at


def read_maintenance_source(sources, *, receipt_ref, trade_date):
    from quant_investor.market.daily_maintenance import (
        STAGES,
        ATTEMPT_SLOTS,
        DailyMaintenanceError,
        _validate_component_result,
    )

    value = sources.document(receipt_ref)
    if not MAINTENANCE_FIELDS <= set(value) <= MAINTENANCE_FIELDS | MAINTENANCE_OPTIONAL:
        raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_FIELDS_INVALID")
    if value["schema_version"] != "cn-daily-maintenance-attempt.v1" or (
        value["target_date"] != trade_date.replace("-", "")
        or value["mode"] != "execute"
        or value["attempt_slot"] not in ATTEMPT_SLOTS
        or value["status"]
        not in {
            "COMPLETE",
            "NO_ACTION",
            "PARTIAL",
            "BLOCKED",
            "RETRY_PENDING",
            "SAME_DAY_SLA_MISSED",
        }
        or value["maintenance_status"] != value["status"]
        or value["macro_used_by_factor"] is not False
        or value["fundamental_used_by_factor"] is not False
        or type(value["factor_rollover_eligible"]) is not bool
    ):
        raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_SCOPE_INVALID")
    rows = value["stage_results"]
    if type(rows) is not list or [r.get("stage") for r in rows if type(r) is dict] != list(STAGES):
        raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_STAGES_INVALID")
    for row in rows:
        if type(row) is not dict or set(row) != {
            "stage",
            "status",
            "write_performed",
            "blockers",
            "evidence",
        }:
            raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_STAGE_INVALID")
        try:
            normalized = _validate_component_result(row["stage"], row)
        except DailyMaintenanceError as exc:
            raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_STAGE_INVALID") from exc
        if normalized != row:
            raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_STAGE_INVALID")
    count = sum(row["write_performed"] for row in rows)
    if (
        type(value["canonical_write_count"]) is not int
        or value["canonical_write_count"] != count
        or type(value["canonical_unchanged"]) is not bool
        or value["canonical_unchanged"] != (count == 0)
    ):
        raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_WRITE_STATE_INVALID")
    present = set(value) & PROVIDER_FIELDS
    if present and (
        present != PROVIDER_FIELDS
        or value["request_count_scope"] != "OFFICIAL_TRANSPORT_CURRENT_OPERATION_CONTEXT"
        or type(value["provider_request_attempts"]) is not dict
    ):
        raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_PROVIDER_FIELDS_INVALID")
    # The veto refs record past diagnostics; the named veto files are mutable.
    # They are not Calendar/empty-event authority or evidence of Macro success.
    for key in ("write_veto_ref", "macro_write_veto_ref"):
        if key in value:
            sources.ref(value[key])
    for key in ("core_completion_ref", "logical_claim_ref", "started_ref"):
        if key in value:
            sources.read(value[key])
    modern, completed_at = _maintenance_state(sources, receipt_ref, value)
    result = read_calendar_source(
        sources, calendar_ref=value["close_session_receipt_ref"], trade_date=trade_date
    )
    if completed_at is not None and completed_at < result["observed_at"]:
        raise StrategyEventStoreError("DAILY_EVENT_MAINTENANCE_END_BEFORE_CALENDAR")
    return {
        **result,
        "source_profile": "NATIVE_COMPONENT_TERMINAL" if modern else "LEGACY_COMPONENT_TERMINAL",
        "maintenance_status": value["status"],
        "completed_at": completed_at,
    }


def validate_daily_closure_source(*, workspace, closure, sources=None):
    """Standing daily sources only; separate retrospective routes remain owning."""
    if closure["policy_ref"] != closure["owner_declaration_ref"]:
        return None
    ref = closure["source_receipt_ref"]
    if ref is None or SYMBOLIC_RECEIPT.fullmatch(ref["path"]):
        raise StrategyEventStoreError("DAILY_EVENT_PHYSICAL_SOURCE_REQUIRED")
    sources = sources or DailyEventSources(workspace)
    sealed = instant(closure["sealed_at"], label="event sealed")
    cutoff = datetime.combine(
        date.fromisoformat(closure["trade_date"]), time(15, 30), tzinfo=SHANGHAI
    )
    if instant(closure["cutoff_at"], label="event cutoff") != cutoff or sealed < cutoff:
        raise StrategyEventStoreError("DAILY_EVENT_OWNER_CUTOFF_INVALID")
    validate_standing_policy(sources.document(closure["policy_ref"]), at=cutoff)
    source = sources.document(ref)
    if source.get("schema_version") == "cn-close-session-receipt.v1":
        result = read_calendar_source(sources, calendar_ref=ref, trade_date=closure["trade_date"])
    elif source.get("schema_version") == "cn-daily-maintenance-attempt.v1":
        result = read_maintenance_source(sources, receipt_ref=ref, trade_date=closure["trade_date"])
    else:
        raise StrategyEventStoreError("DAILY_EVENT_SOURCE_SCHEMA_INVALID")
    if result["observed_at"] > sealed or (
        result.get("completed_at") is not None and result["completed_at"] > sealed
    ):
        raise StrategyEventStoreError("DAILY_EVENT_SOURCE_AFTER_SEAL")
    sources.recheck()
    return {
        **result,
        "execution_authorized": False,
        "eod_admission": False,
        "source_refs": [
            {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
            for path, raw in sorted(sources.files.observed.items())
        ],
    }
