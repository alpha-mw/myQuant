"""Strict Theme handoff shape and binding; native capture replay is also required."""

from datetime import datetime, timezone
import hashlib
from quant_investor.contracts import canonical_json_bytes
from quant_investor.market.tushare.theme_capture import _company_keyset
from .daily_contract import ContractError, GRAPH_SHA256, validate_ref, utc_stamp
from .daily_journal import DailyJournal, _false_authority
from .theme_acquisition import (
    IDENTITY_FIELDS,
    IDENTITY_REFS,
    POLICY_SCHEMA,
    POLICY_SCHEMA_V2,
    POLICY_SCHEMA_V3,
)
from quant_investor.factors.production_pit import FOCUS_COMPANIES

SCHEMA = "cn-daily-theme-source-handoff.v1"
SCHEMA_V2 = "cn-daily-theme-source-handoff.v2"
SCHEMA_V3 = "cn-daily-theme-source-handoff.v3"
FIELDS = IDENTITY_FIELDS | {
    "schema_version",
    "trade_date",
    "graph_sha256",
    "claim_ref",
    "company_keyset",
    "dc_plan_ref",
    "dc_capture_ref",
    "dc_partition_refs",
    "tdx_plan_ref",
    "tdx_capture_ref",
    "tdx_partition_refs",
    "source_descriptor_ref",
    "sealed_at",
    "authority",
}
FOCUS_FIELDS = {
    "special_company_keyset",
    "special_company_set_sha256",
    "pit_selection_ref",
    "pit_generation_manifest_ref",
    "pit_membership_ref",
    "special_dc_plan_ref",
    "special_dc_capture_ref",
    "special_dc_partition_refs",
    "special_tdx_plan_ref",
    "special_tdx_capture_ref",
    "special_tdx_partition_refs",
}
FIELDS_V2 = FIELDS | FOCUS_FIELDS
PIT_FIELDS = ("pit_selection_ref", "pit_generation_manifest_ref", "pit_membership_ref")


def handoff_layout(binding):
    policy = binding.get("policy_schema", POLICY_SCHEMA)
    if policy == POLICY_SCHEMA_V3:
        return SCHEMA_V3, FIELDS_V2, "handoff.v3.json"
    if policy == POLICY_SCHEMA_V2:
        return SCHEMA_V2, FIELDS_V2, "handoff.v2.json"
    if policy == POLICY_SCHEMA:
        return SCHEMA, FIELDS, "handoff.v1.json"
    raise ContractError("THEME_HANDOFF_POLICY_VERSION_INVALID")


def validate_theme_handoff(
    value: dict,
    *,
    journal: DailyJournal,
    binding: dict,
    claim_ref: dict,
    fallback_company_keyset: list[str],
    special_fallback_company_keyset: list[str] | None = None,
) -> dict:
    """No authority from shape alone: callers must replay claim/core/native captures."""
    schema, fields, _ = handoff_layout(binding)
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema_version"] != schema
        or value["trade_date"] != journal.trade_date
        or binding["trade_date"] != journal.trade_date
        or value["graph_sha256"] != GRAPH_SHA256
        or not _false_authority(value["authority"])
        or type(value["company_keyset"]) is not list
        or type(fallback_company_keyset) is not list
    ):
        raise ContractError("THEME_HANDOFF_CONTRACT_INVALID")
    companies = _company_keyset(value["company_keyset"])
    for key in IDENTITY_REFS:
        validate_ref(value[key])
    if len(companies) > 100:
        raise ContractError("THEME_HANDOFF_COMPANY_LIMIT_EXCEEDED")
    if (
        companies != binding["company_keyset"]
        or hashlib.sha256(canonical_json_bytes(companies)).hexdigest()
        != value["company_set_sha256"]
    ):
        raise ContractError("THEME_HANDOFF_COMPANY_SET_MISMATCH")
    if {key: value[key] for key in IDENTITY_FIELDS} != binding["identity"]:
        raise ContractError("THEME_HANDOFF_IDENTITY_MISMATCH")
    if fallback_company_keyset:
        _company_keyset(fallback_company_keyset)
    if not set(fallback_company_keyset) <= set(companies):
        raise ContractError("THEME_HANDOFF_FALLBACK_SCOPE_INVALID")
    focus = []
    if schema in {SCHEMA_V2, SCHEMA_V3}:
        focus = list(FOCUS_COMPANIES)
        if (
            value["special_company_keyset"] != focus
            or binding["special_company_keyset"] != focus
            or value["special_company_set_sha256"]
            != hashlib.sha256(canonical_json_bytes(focus)).hexdigest()
            or value["special_company_set_sha256"] != binding["special_company_set_sha256"]
            or type(special_fallback_company_keyset) is not list
        ):
            raise ContractError("THEME_HANDOFF_FOCUS_SCOPE_INVALID")
        for name in PIT_FIELDS:
            if validate_ref(value[name]) != binding[name]:
                raise ContractError("THEME_HANDOFF_PIT_BINDING_MISMATCH")
        if special_fallback_company_keyset:
            _company_keyset(special_fallback_company_keyset)
        if not set(special_fallback_company_keyset) <= set(focus):
            raise ContractError("THEME_HANDOFF_FOCUS_FALLBACK_INVALID")
    elif special_fallback_company_keyset is not None:
        raise ContractError("THEME_HANDOFF_UNEXPECTED_FOCUS_FALLBACK")
    validate_ref(claim_ref)
    if value["claim_ref"] != claim_ref or claim_ref["path"] != str(
        journal.root / "theme-acquisition.v1.json"
    ):
        raise ContractError("THEME_HANDOFF_CLAIM_MISMATCH")
    prefix = (
        journal.root / "executions" / binding["identity"]["request_ref"]["sha256"] / "theme-source"
    )

    def exact(ref, path):
        if validate_ref(ref)["path"] != str(path):
            raise ContractError("THEME_HANDOFF_SOURCE_PATH_INVALID")

    exact(value["source_descriptor_ref"], prefix / "source.json")
    scopes = [("dc", "dc", companies), ("tdx", "tdx", fallback_company_keyset)]
    if focus:
        scopes.extend(
            [
                ("special_dc", "special-dc", focus),
                ("special_tdx", "special-tdx", special_fallback_company_keyset),
            ]
        )
    for name, path_name, keyset in scopes:
        plan, capture, parts = (
            value[name + "_plan_ref"],
            value[name + "_capture_ref"],
            value[name + "_partition_refs"],
        )
        if name in {"tdx", "special_tdx"} and not keyset:
            if plan is not None or capture is not None or parts != []:
                raise ContractError("THEME_HANDOFF_UNEXPECTED_TDX")
            continue
        exact(plan, prefix / (path_name + "-plan.json"))
        exact(capture, prefix / path_name / "capture.json")
        if type(parts) is not list or len(parts) != 1 + len(keyset):
            raise ContractError("THEME_HANDOFF_PARTITION_SET_INVALID")
        for index, ref in enumerate(parts):
            exact(ref, prefix / path_name / "partitions" / f"{index:05d}.json")
    if utc_stamp(value["sealed_at"]) > datetime.now(timezone.utc):
        raise ContractError("THEME_HANDOFF_FUTURE_SEAL")
    return value
