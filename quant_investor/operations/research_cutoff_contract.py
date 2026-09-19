"""Exact cutoff commitment grammar; native source replay supplies its evidence."""

import re
from datetime import timezone

from quant_investor.strategy_records.corporate_contracts import instant
from .dependency_diagnostics import DependencyInputError
from .daily_contract import ContractError, validate_ref, utc_stamp
from .daily_journal import _false_authority, _validate_day
from .research_timing import CURRENT, HISTORICAL, acquisition_deadline
from quant_investor.intelligence.fundamental_time import SHANGHAI

SCHEMA = "cn-daily-research-cutoff.v1"
SCHEMA_V2 = "cn-daily-research-cutoff.v2"
REGISTERED_FIELDS = {
    "registered_event_declaration_ref",
    "registered_writer_pointer_ref",
    "registered_source_state",
    "registered_store_plan_ref",
}
REF_FIELDS = frozenset(
    {
        "request_ref",
        "maintenance_handoff_ref",
        "core_handoff_ref",
        "source_bundle_ref",
        "native_request_ref",
        "store_plan_ref",
        "portfolio_state_ref",
        "timing_policy_ref",
        "corporate_event_list_ref",
        "corporate_context_ref",
    }
)
FIELDS = REF_FIELDS | {
    "schema_version",
    "trade_date",
    "mode",
    "acquisition_deadline",
    "as_of",
    "portfolio_created_at",
    "physical_reads_completed_at",
    "sealed_at",
    "source_refs",
    "source_times",
    "projection_sha256s",
    "portfolio_timing_status",
    "prospective",
    "authority",
}
ROLE_SEMANTICS = {
    **dict.fromkeys(
        (
            "INDUSTRY_TAXONOMY",
            "INDUSTRY_MEMBERSHIP",
            "THEME_POOL_DC",
            "THEME_POOL_TDX",
            "THEME_FOCUS_DC",
            "THEME_FOCUS_TDX",
        ),
        "PROVIDER_CAPTURE",
    ),
    **dict.fromkeys(
        (
            "THEME_HANDOFF",
            "FUNDAMENTAL_NATIVE_DERIVATION",
            "MACRO_CLOSURE",
            "EVENT_GENERATION",
            "STORE_PLAN",
            "PORTFOLIO_SOURCE_SEAL",
        ),
        "LOCAL_CLOSURE",
    ),
    **dict.fromkeys(
        (
            "EXPOSURE_DECLARATION",
            "FUNDAMENTAL_DECLARATION",
            "CORPORATE_EVENT",
            "CORPORATE_REVIEW_DECLARATION",
        ),
        "SOURCE_DECLARED",
    ),
    "CORPORATE_POLICY": "OWNER_EFFECTIVE",
    "PORTFOLIO_POINTER_PUBLICATION": "LOCAL_PUBLICATION",
}
REGISTERED_ROLE_SEMANTICS = {
    "REGISTERED_OWNER_FACT": "SOURCE_DECLARED",
    "REGISTERED_DECLARATION": "LOCAL_REGISTRATION",
    "REGISTERED_STORE_PUBLICATION": "LOCAL_PUBLICATION",
}


def ref_key(ref):
    value = validate_ref(ref)
    return value["path"], value["sha256"]


def ordered_source_refs(refs):
    values = {}
    for ref in refs:
        path, sha = ref_key(ref)
        if path in values and values[path] != sha:
            raise ContractError("CUTOFF_SOURCE_REF_CONFLICT")
        values[path] = sha
    return [{"path": path, "sha256": sha} for path, sha in sorted(values.items())]


def ordered_source_times(rows, *, registered=False):
    semantics = {**ROLE_SEMANTICS, **(REGISTERED_ROLE_SEMANTICS if registered else {})}
    values = {}
    for row in rows:
        if type(row) is not dict or set(row) != {
            "role",
            "subject_id",
            "source_ref",
            "original_time",
            "time_semantics",
        }:
            raise DependencyInputError("CUTOFF_SOURCE_TIME_SHAPE_INVALID")
        role, subject = row["role"], row["subject_id"]
        if (
            type(role) is not str
            or semantics.get(role) != row["time_semantics"]
            or type(subject) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", subject) is None
        ):
            raise DependencyInputError("CUTOFF_SOURCE_TIME_ROLE_INVALID")
        instant(row["original_time"])
        key = role, subject, *ref_key(row["source_ref"])
        if key in values and values[key] != row:
            raise ContractError("CUTOFF_SOURCE_TIME_CONFLICT")
        values[key] = row
    return [values[key] for key in sorted(values)]


def validate_cutoff_contract(value):
    registered = type(value) is dict and value.get("schema_version") == SCHEMA_V2
    fields = FIELDS | (REGISTERED_FIELDS if registered else set())
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema_version"] not in {SCHEMA, SCHEMA_V2}
    ):
        raise DependencyInputError("CUTOFF_RECEIPT_FIELDS_INVALID")
    if registered:
        validate_ref(value["registered_event_declaration_ref"])
        validate_ref(value["registered_writer_pointer_ref"])
        if (
            value["registered_source_state"] != "REGISTERED_INTRADAY"
            or value["registered_store_plan_ref"] is not None
        ):
            raise ContractError("CUTOFF_REGISTERED_PROFILE_INVALID")
    _validate_day(value["trade_date"])
    for name in REF_FIELDS:
        if name != "corporate_event_list_ref" or value[name] is not None:
            validate_ref(value[name])
    if (
        type(value["mode"]) is not str
        or value["mode"] not in {CURRENT, HISTORICAL}
        or value["prospective"] is not False
        or not _false_authority(value["authority"])
    ):
        raise ContractError("CUTOFF_RECEIPT_POLICY_INVALID")
    cutoff, created, sealed = map(
        utc_stamp, (value["as_of"], value["portfolio_created_at"], value["sealed_at"])
    )
    physical = instant(value["physical_reads_completed_at"])
    canonical = (
        physical.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )
    if value["physical_reads_completed_at"] != canonical or not physical <= created <= sealed:
        raise ContractError("CUTOFF_CUSTODY_ORDER_INVALID")
    if (
        cutoff.strftime("%Y%m%d") != value["trade_date"]
        or cutoff.astimezone(SHANGHAI).strftime("%Y%m%d") != value["trade_date"]
    ):
        raise DependencyInputError("CUTOFF_RECEIPT_DAY_INVALID")
    if value["mode"] == CURRENT:
        deadline = acquisition_deadline(value["trade_date"])
        if (
            value["acquisition_deadline"] != deadline
            or cutoff != created
            or sealed > utc_stamp(deadline)
        ):
            raise ContractError("CUTOFF_CURRENT_BOUNDS_INVALID")
    elif value["acquisition_deadline"] is not None or cutoff > created:
        raise ContractError("CUTOFF_HISTORICAL_BOUNDS_INVALID")
    if type(value["portfolio_timing_status"]) is not str or value[
        "portfolio_timing_status"
    ] not in {"ON_TIME", "LATE_RECORDED"}:
        raise ContractError("CUTOFF_PORTFOLIO_TIMING_INVALID")
    refs, times = value["source_refs"], value["source_times"]
    if (
        type(refs) is not list
        or refs != ordered_source_refs(refs)
        or type(times) is not list
        or times != ordered_source_times(times, registered=registered)
    ):
        raise ContractError("CUTOFF_SOURCE_ORDER_INVALID")
    keys = {ref_key(ref) for ref in refs}
    if any(ref_key(row["source_ref"]) not in keys for row in times):
        raise ContractError("CUTOFF_SOURCE_TIME_REF_UNBOUND")
    if registered:
        for role in REGISTERED_ROLE_SEMANTICS:
            selected = [r for r in times if r["role"] == role]
            if len(selected) != 1:
                raise ContractError("CUTOFF_REGISTERED_TIME_REQUIRED")
            field = {
                "REGISTERED_DECLARATION": "registered_event_declaration_ref",
                "REGISTERED_STORE_PUBLICATION": "registered_writer_pointer_ref",
            }.get(role)
            if field is not None and selected[0]["source_ref"] != value[field]:
                raise ContractError("CUTOFF_REGISTERED_TIME_BINDING_INVALID")
    # Historical book/event seals may be late; native portfolio timing and the
    # historical mode preserve that fact. Native research/issuer validators still
    # enforce their own as-of rules, which this descriptive grammar cannot waive.
    if value["mode"] == CURRENT and any(instant(row["original_time"]) > cutoff for row in times):
        raise ContractError("CUTOFF_SOURCE_TIME_AFTER_CUTOFF")
    hashes = value["projection_sha256s"]
    if type(hashes) is not dict or set(hashes) != {
        "industry",
        "theme",
        "exposure",
        "fundamental",
        "macro",
    }:
        raise ContractError("CUTOFF_PROJECTION_SET_INVALID")
    for name, sha in hashes.items():
        validate_ref({"path": name + ".json", "sha256": sha})
    return value
