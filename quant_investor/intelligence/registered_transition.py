"""Registered BUY accounting evidence, without stop or trading authority."""

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal

from quant_investor.operations.daily_contract import ContractError, utc_stamp
from quant_investor.strategy_records import registered_event_contracts as contracts
from ._common import build_artifact, business_identity
from .pcb_ai_hardware import physical_refs

KIND = "registered_financial_transition_reconciliation"
REVALIDATION = "OWNER_POLICY_REVALIDATION_REQUIRED"
ADMITTED = "VALIDATED_REGISTERED_TRANSITION"


def position_rows(proof):
    before = contracts.positions(proof["baseline"]["record"])
    after = contracts.positions(proof["writer"]["record"])
    refs = physical_refs(
        ref for domain in proof["declaration"]["domains"].values() for ref in domain["fact_refs"]
    )
    rows = []
    for symbol in sorted(before.keys() | after.keys()):
        old = before.get(symbol, (Decimal(0), None, Decimal(0)))
        new = after.get(symbol, (Decimal(0), None, Decimal(0)))
        changed = old != new
        if changed and (symbol not in after or new[0] <= old[0]):
            raise ContractError("REGISTERED_TRANSITION_BUY_PROFILE_INVALID")
        row = {
            "symbol": symbol,
            "baseline_position_state": "PRESENT" if symbol in before else "ABSENT",
            "writer_position_state": "PRESENT" if symbol in after else "ABSENT",
            "change_kind": (
                "UNCHANGED"
                if not changed
                else "EXISTING_POSITION_ADD" if symbol in before else "NEW_POSITION"
            ),
            "fact_refs": deepcopy(refs) if changed else [],
            "policy_revalidation_required": changed,
            "risk_execution_state": "NON_EXECUTABLE" if changed else "NOT_EVALUATED",
            "blocker_codes": [REVALIDATION] if changed else [],
        }
        for name, index in (("shares", 0), ("avg_cost", 1), ("cost_basis", 2)):
            row[name + "_before"] = None if old[index] is None else format(old[index], "f")
            row[name + "_after"] = None if new[index] is None else format(new[index], "f")
            if name != "avg_cost":
                row[name + "_delta"] = format(new[index] - old[index], "f")
        rows.append(row)
    if not any(r["policy_revalidation_required"] for r in rows):
        raise ContractError("REGISTERED_TRANSITION_BUYS_MISSING")
    return rows


def build_registered_transition(*, proof, as_of, custody_at, source_refs):
    """Called only with an exact native declaration replay by the source owner."""
    declaration = contracts.validate_declaration(
        proof["declaration"], declaration_ref=proof["declaration_ref"]
    )
    custody, cutoff = utc_stamp(custody_at), utc_stamp(as_of)
    times = [
        declaration["registered_at"],
        declaration["owner_declared_at"],
        proof["writer"]["pointer"]["published_at"],
    ]
    if custody > datetime.now(timezone.utc) or any(contracts.stamp(t) > custody for t in times):
        raise ContractError("REGISTERED_TRANSITION_CUSTODY_INVALID")
    if proof["profile"]["profile"] != "OWNER_DECLARED_BUYS_V1":
        raise ContractError("REGISTERED_TRANSITION_PROFILE_UNSUPPORTED")
    before = contracts.number(proof["baseline"]["record"]["accounting"]["cash_after"])
    after = contracts.number(proof["writer"]["record"]["accounting"]["cash_after"])
    late = custody > cutoff
    fields = {
        "as_of": as_of,
        "trade_date": declaration["trade_date"].replace("-", ""),
        "strategy_id": declaration["strategy_id"],
        "source_profile": proof["profile"]["profile"],
        "registered_event_declaration_ref": proof["declaration_ref"],
        "decision_baseline_pointer_ref": declaration["baseline_store_pointer_ref"],
        "decision_baseline_catalog_ref": declaration["baseline_catalog_ref"],
        "decision_baseline_record_id": declaration["baseline_record_id"],
        "writer_pointer_ref": declaration["writer_store_pointer_ref"],
        "writer_catalog_ref": declaration["writer_catalog_ref"],
        "writer_record_id": declaration["writer_record_id"],
        "owner_fact_ref": declaration["owner_fact_ref"],
        "owner_declared_at": declaration["owner_declared_at"],
        "registered_at": declaration["registered_at"],
        "evidence_level": declaration["evidence_level"],
        "broker_statement_verified": declaration["broker_statement_verified"],
        "fee_evidence_level": proof["profile"]["fee_evidence_level"],
        "domain_states": deepcopy(declaration["domains"]),
        "position_rows": position_rows(proof),
        "cash_before_cny": format(before, "f"),
        "cash_after_cny": format(after, "f"),
        "cash_delta_cny": format(after - before, "f"),
        "financial_admission_state": ADMITTED,
        "risk_readiness_state": REVALIDATION,
        "blocker_codes": sorted(
            [REVALIDATION] + (["SOURCE_CUSTODY_AFTER_DECISION"] if late else [])
        ),
        "source_refs": physical_refs([*proof["source_refs"], *source_refs]),
        "custody_at": custody_at,
        "timing_status": "LATE_RECORDED" if late else "ON_TIME",
        "prospective": False,
    }
    return build_artifact(
        kind=KIND,
        identity_field="registered_transition_id",
        identity=business_identity(kind=KIND, identity_inputs=fields),
        created_at=custody_at,
        fields=fields,
    )
