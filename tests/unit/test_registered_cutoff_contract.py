"""Registered cutoff grammar only; native source admission is tested separately."""

from copy import deepcopy

import pytest

from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.research_cutoff_contract import (
    REF_FIELDS,
    SCHEMA_V2,
    ordered_source_refs,
    ordered_source_times,
    validate_cutoff_contract,
)
from quant_investor.operations.research_timing import CURRENT, acquisition_deadline


def reference(name):
    return {"path": "synthetic-grammar/" + name + ".json", "sha256": "a" * 64}


def receipt():
    value = {k: reference(k) for k in REF_FIELDS}
    value.update(
        schema_version=SCHEMA_V2,
        trade_date="20260828",
        mode=CURRENT,
        acquisition_deadline=acquisition_deadline("20260828"),
        as_of="2026-08-28T13:30:00Z",
        portfolio_created_at="2026-08-28T13:30:00Z",
        physical_reads_completed_at="2026-08-28T13:29:59.250000Z",
        sealed_at="2026-08-28T13:30:01Z",
        portfolio_timing_status="ON_TIME",
        prospective=False,
        authority=dict(FALSE_AUTHORITY),
        registered_event_declaration_ref=reference("declaration"),
        registered_writer_pointer_ref=reference("writer"),
        registered_source_state="REGISTERED_INTRADAY",
        registered_store_plan_ref=None,
        projection_sha256s=dict.fromkeys(
            ("industry", "theme", "exposure", "fundamental", "macro"), "b" * 64
        ),
    )
    times = [
        {
            "role": role,
            "subject_id": "synthetic",
            "source_ref": ref,
            "original_time": stamp,
            "time_semantics": semantics,
        }
        for role, ref, stamp, semantics in (
            (
                "REGISTERED_OWNER_FACT",
                reference("owner"),
                "2026-08-28T02:02:00Z",
                "SOURCE_DECLARED",
            ),
            (
                "REGISTERED_DECLARATION",
                value["registered_event_declaration_ref"],
                "2026-08-28T12:20:00Z",
                "LOCAL_REGISTRATION",
            ),
            (
                "REGISTERED_STORE_PUBLICATION",
                value["registered_writer_pointer_ref"],
                "2026-08-28T02:01:00Z",
                "LOCAL_PUBLICATION",
            ),
        )
    ]
    value["source_times"] = ordered_source_times(times, registered=True)
    value["source_refs"] = ordered_source_refs([r["source_ref"] for r in times])
    return value


@pytest.mark.parametrize(
    "fault",
    [
        "legacy_schema",
        "declaration",
        "writer",
        "missing_role",
        "unbound_ref",
        "after_cutoff",
        "source_state",
        "finalized_plan",
        "prospective",
        "semantic",
    ],
)
def test_registered_cutoff_rejects_cross_version_binding_and_time_errors(fault):
    original = receipt()
    assert validate_cutoff_contract(original) == original
    value = deepcopy(original)
    if fault == "legacy_schema":
        value["schema_version"] = "cn-daily-research-cutoff.v1"
    elif fault in {"declaration", "writer"}:
        key = (
            "registered_event_declaration_ref"
            if fault == "declaration"
            else "registered_writer_pointer_ref"
        )
        value[key] = reference("wrong")
    elif fault == "missing_role":
        value["source_times"] = [
            r for r in value["source_times"] if r["role"] != "REGISTERED_OWNER_FACT"
        ]
    elif fault == "unbound_ref":
        value["source_refs"] = value["source_refs"][:-1]
    elif fault == "after_cutoff":
        value["source_times"][0]["original_time"] = "2026-08-28T13:30:01Z"
    elif fault == "source_state":
        value["registered_source_state"] = "FINALIZED_V2_RECOVERY"
    elif fault == "finalized_plan":
        value["registered_store_plan_ref"] = reference("new-plan")
    elif fault == "prospective":
        value["prospective"] = True
    else:
        value["source_times"][0]["time_semantics"] = "SOURCE_DECLARED"
    with pytest.raises(ContractError):
        validate_cutoff_contract(value)


def test_old_time_registry_does_not_admit_registered_roles():
    with pytest.raises(ContractError, match="ROLE_INVALID"):
        ordered_source_times(receipt()["source_times"])
